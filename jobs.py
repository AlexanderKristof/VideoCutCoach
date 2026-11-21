"""
Háttérfeladatok (jobok) kezelése a szerveren belül.

A hosszú munkák (teljes meccs elemzése, félidő-felismerés, vágás, klipexport) nem
blokkolják a HTTP kérést: a kliens azonnal kap egy job azonosítót, és a
GET /jobs/{id} végponton (1–2 mp-enként) lekérdezi az állapotot és a haladást.

- submit(kind, func, *args, **kwargs) -> job_id. A func első paramétere egy
  ProgressReporter (progress), a többi a megadott args/kwargs.
- Állapotok: queued → running → done / error / cancelled. Szerver-újraindítás után
  a félbemaradt (queued / running) jobok állapota: "megszakadt".
- A megszakítás kooperatív: a /cancel csak jelzést állít, a munkafüggvény a
  progress.report() / progress.check_cancel() hívásnál kap JobCancelled kivételt.
- A metaadatok minden állapotváltozáskor (és haladáskor legfeljebb pár
  másodpercenként) a <jobs_dir>/<id>.json fájlba íródnak, a hibák teljes
  traceback-je a <jobs_dir>/<id>.log fájlba.

Csak a szabványos könyvtárat használja (torch / OpenCV nélkül is importálható, tesztelhető).
"""
import os
import json
import time
import uuid
import threading
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUS_INTERRUPTED = "megszakadt"   # a szerver újraindult futás közben

FINAL_STATUSES = (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED, STATUS_INTERRUPTED)

# A haladás legfeljebb ilyen gyakran íródik lemezre (az állapotváltozás mindig azonnal)
PERSIST_PROGRESS_EVERY_SEC = 5.0


class JobCancelled(Exception):
    """A felhasználó megszakította a jobot (kooperatív megszakítás)."""


def _now():
    return datetime.now().isoformat(timespec="seconds")


class ProgressReporter:
    """
    A munkafüggvénynek átadott haladásjelző.

    report(pct, phase): haladás 0–100 (nem csökken) + az aktuális fázis szövege;
        közben ellenőrzi a megszakítást (JobCancelled-et dobhat).
    check_cancel(): JobCancelled, ha megszakítást kértek.
    sub(start, end, label): rész-jelző, amely a 0–100-at a [start, end] sávba
        vetíti – a több lépéses folyamat egyes fázisaihoz.
    A példány hívható is: progress(pct, phase) == progress.report(pct, phase).
    """

    def __init__(self, on_update=None, cancel_event=None):
        self._on_update = on_update
        self._cancel = cancel_event or threading.Event()
        self.pct = 0.0
        self.phase = ""

    # -- megszakítás --
    @property
    def cancelled(self):
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()

    def check_cancel(self):
        if self._cancel.is_set():
            raise JobCancelled("A feladatot megszakították.")

    # -- haladás --
    def report(self, pct=None, phase=None):
        if pct is not None:
            try:
                p = float(pct)
            except (TypeError, ValueError):
                p = self.pct
            if p != p:  # NaN
                p = self.pct
            self.pct = max(self.pct, min(100.0, max(0.0, p)))
        if phase is not None:
            self.phase = str(phase)
        if self._on_update is not None:
            self._on_update(round(self.pct, 2), self.phase)
        self.check_cancel()

    __call__ = report

    def sub(self, start, end, label=None):
        return _SubReporter(self, float(start), float(end), label)


class _SubReporter:
    """A szülő jelző egy sávja: 0–100 → [start, end], a fázis elé a label kerül."""

    def __init__(self, parent, start, end, label):
        self.parent, self.start, self.end, self.label = parent, start, end, label

    @property
    def cancelled(self):
        return self.parent.cancelled

    def check_cancel(self):
        self.parent.check_cancel()

    def report(self, pct=None, phase=None):
        g = None
        if pct is not None:
            try:
                p = min(100.0, max(0.0, float(pct)))
                g = self.start + (self.end - self.start) * p / 100.0
            except (TypeError, ValueError):
                g = None
        if self.label and phase:
            text = f"{self.label}: {phase}"
        else:
            text = self.label or phase
        self.parent.report(g, text)

    __call__ = report

    def sub(self, start, end, label=None):
        span = self.end - self.start
        full = f"{self.label}: {label}" if (self.label and label) else (label or self.label)
        return _SubReporter(self.parent, self.start + span * start / 100.0,
                            self.start + span * end / 100.0, full)


class JobManager:
    """Folyamaton belüli jobkezelő ThreadPoolExecutorral és JSON-perzisztenciával."""

    def __init__(self, jobs_dir, max_workers=2):
        self.jobs_dir = os.path.abspath(jobs_dir)
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")
        self._lock = threading.RLock()
        self._jobs = {}        # id -> metaadat dict
        self._cancel = {}      # id -> threading.Event
        self._futures = {}     # id -> Future
        self._last_persist = {}
        self._t0 = {}          # id -> indulás (monotonic)
        self._load_existing()

    # ---------- perzisztencia ----------
    def _path(self, job_id):
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    def _persist(self, job_id):
        with self._lock:
            data = dict(self._jobs[job_id])
            path = self._path(job_id)
            tmp = path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                os.replace(tmp, path)
            except OSError as ex:
                print(f"⚠️ A job állapota nem menthető ({job_id}): {ex}")
            self._last_persist[job_id] = time.monotonic()

    def _load_existing(self):
        """Korábbi jobok betöltése; a futás közben félbemaradtak 'megszakadt' állapotot kapnak."""
        for name in sorted(os.listdir(self.jobs_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.jobs_dir, name), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            job_id = data.get("id") or name[:-5]
            data["id"] = job_id
            self._jobs[job_id] = data
            if data.get("status") in (STATUS_QUEUED, STATUS_RUNNING):
                data["status"] = STATUS_INTERRUPTED
                data["finished_at"] = data.get("finished_at") or _now()
                data["error"] = "A szerver újraindult, mielőtt a feladat befejeződött."
                data["phase"] = "Megszakadt (szerver-újraindítás)"
                self._persist(job_id)

    # ---------- nyilvános API ----------
    def submit(self, kind, func, *args, params=None, **kwargs):
        """
        Új job indítása. A func(progress, *args, **kwargs) visszatérési értéke (dict) a job eredménye.
        params: a kliensnek visszaadott, JSON-ba írható paraméter-összefoglaló (nem kötelező).
        """
        job_id = uuid.uuid4().hex[:12]
        meta = {
            "id": job_id,
            "kind": kind,
            "status": STATUS_QUEUED,
            "progress": 0.0,
            "phase": "Várakozik",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "elapsed_sec": None,
            "cancel_requested": False,
            "params": params or {},
            "result": None,
            "error": None,
            "log_path": None,
        }
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = meta
            self._cancel[job_id] = cancel_event
            self._persist(job_id)
            self._futures[job_id] = self._executor.submit(self._run, job_id, func, args, kwargs)
        return job_id

    def get(self, job_id):
        with self._lock:
            meta = self._jobs.get(job_id)
            return dict(meta) if meta is not None else None

    def list(self, limit=50, include_result=False):
        with self._lock:
            items = [dict(m) for m in self._jobs.values()]
        items.sort(key=lambda m: m.get("created_at") or "", reverse=True)
        if not include_result:
            for m in items:
                m.pop("result", None)
        return items[:limit] if limit else items

    def cancel(self, job_id):
        """Megszakítás kérése. Visszatér: a job friss állapota, vagy None, ha nincs ilyen job."""
        with self._lock:
            meta = self._jobs.get(job_id)
            if meta is None:
                return None
            if meta["status"] in FINAL_STATUSES:
                return dict(meta)
            meta["cancel_requested"] = True
            ev = self._cancel.get(job_id)
            if ev is not None:
                ev.set()
            fut = self._futures.get(job_id)
            if meta["status"] == STATUS_QUEUED and fut is not None and fut.cancel():
                # Még el sem indult – azonnal lezárható
                self._finish(job_id, STATUS_CANCELLED, phase="Megszakítva (nem indult el)")
            else:
                meta["phase"] = "Megszakítás kérve…"
                self._persist(job_id)
            return dict(meta)

    def wait(self, job_id, timeout=None):
        """Megvárja a job végét (főleg tesztekhez / parancssorhoz). Visszatér: a job állapota."""
        fut = self._futures.get(job_id)
        if fut is not None:
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass
        return self.get(job_id)

    def shutdown(self, wait=True):
        self._executor.shutdown(wait=wait)

    # ---------- belső ----------
    def _on_progress(self, job_id, pct, phase):
        with self._lock:
            meta = self._jobs.get(job_id)
            if meta is None or meta["status"] != STATUS_RUNNING:
                return
            meta["progress"] = pct
            if meta.get("cancel_requested"):
                meta["phase"] = "Megszakítás kérve…"
            else:
                meta["phase"] = phase
            if time.monotonic() - self._last_persist.get(job_id, 0.0) >= PERSIST_PROGRESS_EVERY_SEC:
                self._persist(job_id)

    def _finish(self, job_id, status, result=None, error=None, phase=None, log_path=None):
        with self._lock:
            meta = self._jobs[job_id]
            meta["status"] = status
            meta["finished_at"] = _now()
            t0 = self._t0.pop(job_id, None)
            if t0 is not None:
                meta["elapsed_sec"] = round(time.monotonic() - t0, 1)
            if result is not None:
                meta["result"] = result
            if error is not None:
                meta["error"] = error
            if log_path is not None:
                meta["log_path"] = log_path
            if status == STATUS_DONE:
                meta["progress"] = 100.0
            meta["phase"] = phase or {STATUS_DONE: "Kész", STATUS_ERROR: "Hiba",
                                      STATUS_CANCELLED: "Megszakítva"}.get(status, status)
            self._persist(job_id)

    def _run(self, job_id, func, args, kwargs):
        with self._lock:
            meta = self._jobs[job_id]
            if meta["status"] != STATUS_QUEUED:
                return
            cancel_event = self._cancel[job_id]
            if cancel_event.is_set():
                self._finish(job_id, STATUS_CANCELLED, phase="Megszakítva (nem indult el)")
                return
            meta["status"] = STATUS_RUNNING
            meta["started_at"] = _now()
            meta["phase"] = "Indul…"
            self._t0[job_id] = time.monotonic()
            self._persist(job_id)

        reporter = ProgressReporter(lambda pct, phase: self._on_progress(job_id, pct, phase), cancel_event)
        try:
            result = func(reporter, *args, **kwargs)
        except JobCancelled:
            self._finish(job_id, STATUS_CANCELLED)
        except BaseException as ex:  # noqa: B902 – a job hibája ne vigye el a szálat
            tb = traceback.format_exc()
            log_path = os.path.join(self.jobs_dir, f"{job_id}.log")
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"Job: {job_id} ({meta.get('kind')}), hiba ideje: {_now()}\n\n{tb}")
            except OSError:
                log_path = None
            print(f"❌ Job hiba ({job_id}, {meta.get('kind')}): {ex}")
            self._finish(job_id, STATUS_ERROR, error=f"{type(ex).__name__}: {ex}", log_path=log_path)
        else:
            if isinstance(result, dict) and result.get("status") == "error":
                # A régi végpontok hibát dict-ben adnak vissza (nem kivételként)
                self._finish(job_id, STATUS_ERROR, result=result,
                             error=result.get("message") or "Ismeretlen hiba")
            else:
                self._finish(job_id, STATUS_DONE, result=result)

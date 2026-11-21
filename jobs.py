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
- Szüneteltetés (pause / resume): szintén kooperatív – a progress.report() /
  progress.check_pause() hívásnál a munkafüggvény várakozik, amíg folytatást nem kérnek
  (állapot közben: "paused"). A már futó ffmpeg lépés a végéig fut.
- Háttér-mód (background=True): külön, alacsony prioritású szálon fut (Linux: a szál nice
  értéke, Windows: BELOW_NORMAL szálprioritás), az ffmpeg gyermekfolyamatok alacsony
  prioritással és `-threads N`-nel indulnak, a torch / OpenCV szálszáma is korlátozott
  (ez utóbbi az egész folyamatra hat, a háttérjob végén visszaáll). Lásd background.py.
- Részeredmény: progress.publish(dict) – a job "result" mezője már futás közben frissül
  (pl. a vágott meccsvideó elérhető, miközben az elemzés még fut).
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

import background

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUS_INTERRUPTED = "megszakadt"   # a szerver újraindult futás közben
STATUS_PAUSED = "paused"            # szüneteltetve (a munkafüggvény a következő jelzésnél várakozik)

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

    def __init__(self, on_update=None, cancel_event=None, pause_event=None, on_pause=None, on_publish=None):
        self._on_update = on_update
        self._cancel = cancel_event or threading.Event()
        self._pause = pause_event or threading.Event()   # beállítva = szünetelés kérve
        self._on_pause = on_pause                         # on_pause(True/False): tényleges várakozás
        self._on_publish = on_publish
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

    # -- szüneteltetés --
    @property
    def paused(self):
        return self._pause.is_set()

    def pause(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    def check_pause(self, poll_sec=0.2):
        """Ha szüneteltetést kértek: várakozik a folytatásig (közben a megszakítás is működik)."""
        if not self._pause.is_set():
            return
        if self._on_pause is not None:
            self._on_pause(True)
        try:
            while self._pause.is_set():
                self.check_cancel()
                self._cancel.wait(poll_sec)
            self.check_cancel()
        finally:
            if self._on_pause is not None:
                self._on_pause(False)

    # -- részeredmény --
    def publish(self, partial):
        """Részeredmény (JSON-ba írható dict) – a job result mezője már futás közben frissül."""
        if self._on_publish is not None:
            self._on_publish(partial)

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
        self.check_pause()

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

    def check_pause(self):
        self.parent.check_pause()

    def publish(self, partial):
        self.parent.publish(partial)

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

    def __init__(self, jobs_dir, max_workers=2, background_workers=1, priority=None):
        """
        priority: a prioritás-kezelő függvények (tesztekhez cserélhető):
            {"lower_thread": fn() -> bool, "limit_threads": fn(n) -> előző, "restore_threads": fn(előző),
             "default_threads": fn() -> int}
        """
        self.jobs_dir = os.path.abspath(jobs_dir)
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._priority = {"lower_thread": background.lower_current_thread_priority,
                          "limit_threads": background.limit_library_threads,
                          "restore_threads": background.restore_library_threads,
                          "default_threads": background.default_threads}
        self._priority.update(priority or {})
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")
        # A háttérjobok külön, alacsony prioritású szálakon futnak (a szál prioritását az
        # inicializáló egyszer csökkenti – vissza nem kell állítani, mert a szál csak ilyet futtat).
        self._bg_executor = ThreadPoolExecutor(max_workers=max(1, background_workers),
                                               thread_name_prefix="job-hatter", initializer=self._bg_thread_init)
        self._lock = threading.RLock()
        self._jobs = {}        # id -> metaadat dict
        self._cancel = {}      # id -> threading.Event
        self._pause = {}       # id -> threading.Event (beállítva = szünetelés kérve)
        self._futures = {}     # id -> Future
        self._last_persist = {}
        self._t0 = {}          # id -> indulás (monotonic)
        self._bg_running = 0
        self._bg_prev_threads = None
        self._load_existing()

    def _bg_thread_init(self):
        try:
            ok = self._priority["lower_thread"]()
        except Exception:
            ok = False
        if not ok:
            print("⚠️ A háttérszál prioritása nem csökkenthető ezen a rendszeren (csak az ffmpeg fut alacsonyan).")

    # ---------- perzisztencia ----------
    def _path(self, job_id):
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    def _persist(self, job_id):
        with self._lock:
            data = self._public(self._jobs[job_id])
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
            if data.get("status") in (STATUS_QUEUED, STATUS_RUNNING, STATUS_PAUSED):
                data["status"] = STATUS_INTERRUPTED
                data["finished_at"] = data.get("finished_at") or _now()
                data["error"] = "A szerver újraindult, mielőtt a feladat befejeződött."
                data["phase"] = "Megszakadt (szerver-újraindítás)"
                self._persist(job_id)

    # ---------- nyilvános API ----------
    def submit(self, kind, func, *args, params=None, background=False, threads=None, **kwargs):
        """
        Új job indítása. A func(progress, *args, **kwargs) visszatérési értéke (dict) a job eredménye.
        params: a kliensnek visszaadott, JSON-ba írható paraméter-összefoglaló (nem kötelező).
        background: háttér-mód (alacsony prioritás, kevés szál); threads: szálkorlát (None = a
        háttér-módban javasolt érték: a magok fele, akkumulátorról a negyede).
        """
        if background and not threads:
            try:
                threads = int(self._priority["default_threads"]())
            except Exception:
                threads = 2
        threads = int(threads) if threads else None
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
            "paused": False,
            "background": bool(background),
            "threads": threads,
            "params": params or {},
            "result": None,
            "error": None,
            "log_path": None,
        }
        cancel_event = threading.Event()
        executor = self._bg_executor if background else self._executor
        with self._lock:
            self._jobs[job_id] = meta
            self._cancel[job_id] = cancel_event
            self._pause[job_id] = threading.Event()
            self._persist(job_id)
            self._futures[job_id] = executor.submit(self._run, job_id, func, args, kwargs)
        return job_id

    def get(self, job_id):
        with self._lock:
            meta = self._jobs.get(job_id)
            return self._public(meta) if meta is not None else None

    def list(self, limit=50, include_result=False):
        with self._lock:
            items = [self._public(m) for m in self._jobs.values()]
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
                return self._public(meta)
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
            return self._public(meta)

    def pause(self, job_id):
        """
        Szüneteltetés kérése (kooperatív: a munkafüggvény a következő haladásjelzésnél áll meg).
        Visszatér: a job friss állapota, vagy None, ha nincs ilyen job.
        """
        with self._lock:
            meta = self._jobs.get(job_id)
            if meta is None:
                return None
            if meta["status"] in FINAL_STATUSES:
                return self._public(meta)
            ev = self._pause.get(job_id)
            if ev is not None:
                ev.set()
            meta["paused"] = True
            if meta["status"] == STATUS_RUNNING:
                meta["_phase_before_pause"] = meta.get("phase")
                meta["phase"] = "Szüneteltetés kérve…"
            self._persist(job_id)
            return self._public(meta)

    def resume(self, job_id):
        """Folytatás. Visszatér: a job friss állapota, vagy None, ha nincs ilyen job."""
        with self._lock:
            meta = self._jobs.get(job_id)
            if meta is None:
                return None
            if meta["status"] in FINAL_STATUSES:
                return self._public(meta)
            ev = self._pause.get(job_id)
            if ev is not None:
                ev.clear()
            meta["paused"] = False
            if meta["status"] in (STATUS_RUNNING, STATUS_PAUSED) and \
                    meta.get("phase") in ("Szüneteltetve", "Szüneteltetés kérve…"):
                meta["phase"] = meta.get("_phase_before_pause") or "Folytatás…"
            self._persist(job_id)
            return self._public(meta)

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
        # a szüneteltetett jobok ne tartsák vissza a leállást
        with self._lock:
            for ev in self._pause.values():
                ev.clear()
        self._executor.shutdown(wait=wait)
        self._bg_executor.shutdown(wait=wait)

    # ---------- szüneteltetés / részeredmény (a munkaszálból hívva) ----------
    @staticmethod
    def _public(meta):
        return {k: v for k, v in meta.items() if not k.startswith("_")}

    def _on_pause(self, job_id, waiting):
        with self._lock:
            meta = self._jobs.get(job_id)
            if meta is None or meta["status"] in FINAL_STATUSES:
                return
            if waiting:
                meta["status"] = STATUS_PAUSED
                meta["phase"] = "Szüneteltetve"
            else:
                meta["status"] = STATUS_RUNNING
                if meta.get("cancel_requested"):
                    meta["phase"] = "Megszakítás kérve…"
                elif meta.get("phase") in ("Szüneteltetve", "Szüneteltetés kérve…", "Folytatás…"):
                    meta["phase"] = meta.get("_phase_before_pause") or "Folytatás…"
            self._persist(job_id)

    def _on_publish(self, job_id, partial):
        with self._lock:
            meta = self._jobs.get(job_id)
            if meta is None or meta["status"] in FINAL_STATUSES:
                return
            meta["result"] = partial
            self._persist(job_id)

    def _bg_enter(self, threads):
        with self._lock:
            self._bg_running += 1
            if self._bg_running == 1 and threads:
                try:
                    self._bg_prev_threads = self._priority["limit_threads"](threads)
                except Exception:
                    self._bg_prev_threads = None

    def _bg_exit(self):
        with self._lock:
            self._bg_running = max(0, self._bg_running - 1)
            if self._bg_running == 0 and self._bg_prev_threads is not None:
                try:
                    self._priority["restore_threads"](self._bg_prev_threads)
                except Exception:
                    pass
                self._bg_prev_threads = None

    # ---------- belső ----------
    def _on_progress(self, job_id, pct, phase):
        with self._lock:
            meta = self._jobs.get(job_id)
            if meta is None or meta["status"] != STATUS_RUNNING:
                return
            meta["progress"] = pct
            if meta.get("cancel_requested"):
                meta["phase"] = "Megszakítás kérve…"
            elif meta.get("paused"):
                meta["_phase_before_pause"] = phase
                meta["phase"] = "Szüneteltetés kérve…"
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
            meta["paused"] = False
            meta.pop("_phase_before_pause", None)
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

        reporter = ProgressReporter(lambda pct, phase: self._on_progress(job_id, pct, phase), cancel_event,
                                    pause_event=self._pause.get(job_id),
                                    on_pause=lambda waiting: self._on_pause(job_id, waiting),
                                    on_publish=lambda partial: self._on_publish(job_id, partial))
        bg = bool(meta.get("background"))
        threads = meta.get("threads")
        if bg:
            self._bg_enter(threads)
        try:
            # háttér-mód / szálkorlát: az ebből a szálból indított ffmpeg-folyamatokra (fast_cut, ...)
            with background.context(enabled=bg, threads=threads):
                reporter.check_pause()  # a várakozás közben szüneteltetett job el sem indul
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
        finally:
            if bg:
                self._bg_exit()

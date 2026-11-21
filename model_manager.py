"""
Súlyfájlok (YOLO .pt) kezelése: megkeresés a models mappában, letöltés haladásjelzéssel.

Miért kell: az ultralytics a hiányzó hivatalos súlyokat (yolov8n.pt ...) magától letölti, de a
MUNKAKÖNYVTÁRBA – egy telepített alkalmazásnál ez a Program Files lenne (írásvédett), és a letöltés
az első elemzés közben, haladásjelzés nélkül történne. Itt:
  * find_weights(spec): a beállított súlyfájl megkeresése (megadott út, models mappa, adatmappa,
    csomagolt erőforrások);
  * download_weights(name): hivatalos súly letöltése a models mappába (.part fájlba, majd átnevezés),
    haladásjelzővel – a /models/download végpont háttérfeladatként futtatja (jobs.py);
  * resolve_for_load(spec): amit a YOLO(...) kap. Ha a fájl nincs meg és az automatikus letöltés
    engedélyezett (alkalmazás-módban alapból igen, fejlesztői módban nem – ott a régi viselkedés
    marad: az ultralytics tölti le), előbb letölti.

Környezeti változók:
  TACTICAL_AUTO_DOWNLOAD  – "1"/"0": automatikus letöltés első használatkor (alap: alkalmazás-módban 1)
  TACTICAL_MODEL_BASE_URL – a hivatalos kiadás helyett más letöltési alap-URL (pl. belső tükör)

Csak a szabványos könyvtárat használja (torch / ultralytics nélkül importálható, tesztelhető).
A tesztek soha nem töltenek le az internetről (helyi HTTP szervert adnak meg base_url-ként).
"""
import os
import time
import threading
import urllib.request

import app_paths

# Az ultralytics hivatalos súlyai (github.com/ultralytics/assets kiadás)
OFFICIAL_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
OFFICIAL_WEIGHTS = tuple(
    [f"yolov8{s}.pt" for s in "nsmlx"] + [f"yolo11{s}.pt" for s in "nsmlx"]
)
ENV_AUTO_DOWNLOAD = "TACTICAL_AUTO_DOWNLOAD"
ENV_BASE_URL = "TACTICAL_MODEL_BASE_URL"
USER_AGENT = "TacAnaly-Engine"

_download_locks = {}
_download_locks_guard = threading.Lock()


class ModelDownloadError(RuntimeError):
    """A súlyfájl nem tölthető le / nem érvényes."""


def _env(env):
    return os.environ if env is None else env


def auto_download_enabled(env=None):
    env = _env(env)
    v = env.get(ENV_AUTO_DOWNLOAD)
    if v is not None and str(v).strip() != "":
        return str(v).strip().lower() in ("1", "true", "igen", "yes", "on")
    return app_paths.app_mode(env)


def is_official(name):
    return os.path.basename(str(name or "")) in OFFICIAL_WEIGHTS


def official_url(name, env=None, base_url=None):
    """A hivatalos súly letöltési címe, vagy None (nem hivatalos / egyedi modell)."""
    base = os.path.basename(str(name or ""))
    if base not in OFFICIAL_WEIGHTS:
        return None
    root = base_url or _env(env).get(ENV_BASE_URL) or OFFICIAL_BASE_URL
    if not root.endswith("/"):
        root += "/"
    return root + base


def candidate_paths(spec, env=None):
    """A keresési sorrend (duplikátumok nélkül)."""
    spec = str(spec)
    base = os.path.basename(spec)
    mdir = app_paths.models_dir(env)
    cands = [spec]
    if not os.path.isabs(spec):
        cands += [
            os.path.join(mdir, spec),
            os.path.join(mdir, base),
            os.path.join(app_paths.data_dir(env), spec),
            app_paths.resource_path(spec),
            app_paths.resource_path("models", base),
            os.path.join(app_paths.install_dir(), "models", base),  # a telepítő mellékelheti
        ]
    out, seen = [], set()
    for c in cands:
        key = os.path.normcase(os.path.abspath(c))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def find_weights(spec, env=None):
    """A súlyfájl abszolút útvonala, ha valahol megvan; különben None."""
    if not spec:
        return None
    for c in candidate_paths(spec, env):
        if os.path.isfile(c) and os.path.getsize(c) > 0:
            return os.path.abspath(c)
    return None


def target_path(name, env=None):
    return os.path.join(app_paths.models_dir(env), os.path.basename(str(name)))


def _lock_for(path):
    key = os.path.normcase(os.path.abspath(path))
    with _download_locks_guard:
        return _download_locks.setdefault(key, threading.Lock())


def download_weights(name, env=None, progress=None, base_url=None, url=None, dest_dir=None,
                     timeout=60, chunk_size=1 << 16):
    """
    Súlyfájl letöltése a models mappába. progress(pct, phase) – a jobs.ProgressReporter is jó
    (a megszakítást kivétellel jelzi: ekkor a félkész .part fájl törlődik).
    Visszatér: {"name", "path", "size", "url", "downloaded": bool}.
    """
    base = os.path.basename(str(name))
    url = url or official_url(base, env, base_url=base_url)
    if not url:
        raise ModelDownloadError(f"Ismeretlen súlyfájl, nincs hivatalos letöltési címe: {base} "
                                 f"(egyedi modellt másolj a models mappába)")
    dest_dir = dest_dir or app_paths.models_dir(env)
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, base)

    def _report(pct, phase):
        if progress is not None:
            progress(pct, phase)

    with _lock_for(path):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            _report(100.0, f"Már megvan: {base}")
            return {"name": base, "path": os.path.abspath(path), "size": os.path.getsize(path),
                    "url": url, "downloaded": False}
        part = path + ".part"
        _report(0.0, f"Letöltés: {base}")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done, last = 0, 0.0
                with open(part, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last >= 0.25:
                            last = now
                            mb = done / 1e6
                            if total:
                                _report(min(99.0, 100.0 * done / total),
                                        f"Letöltés: {base} ({mb:.1f} / {total / 1e6:.1f} MB)")
                            else:
                                _report(None, f"Letöltés: {base} ({mb:.1f} MB)")
            if total and done != total:
                raise ModelDownloadError(f"Hiányos letöltés: {done} / {total} bájt")
            with open(part, "rb") as f:
                magic = f.read(2)
            if magic != b"PK":  # a torch .pt fájl zip-archívum
                raise ModelDownloadError("A letöltött fájl nem érvényes PyTorch súlyfájl "
                                         "(proxy / bejelentkező oldal?)")
            os.replace(part, path)
        except BaseException as ex:
            try:
                os.remove(part)
            except OSError:
                pass
            if isinstance(ex, OSError) and not isinstance(ex, ModelDownloadError):
                raise ModelDownloadError(f"A letöltés nem sikerült ({url}): {ex}") from ex
            raise
        _report(100.0, f"Kész: {base}")
        return {"name": base, "path": os.path.abspath(path), "size": os.path.getsize(path),
                "url": url, "downloaded": True}


def resolve_for_load(spec, env=None, progress=None, allow_download=None):
    """
    Amit a YOLO(...) kap: a megtalált fájl abszolút útja; hiányzó hivatalos súlynál – ha az
    automatikus letöltés engedélyezett – letöltés után a models mappabeli út; különben a
    változatlan spec (régi viselkedés: az ultralytics kezeli / tölti le).
    """
    found = find_weights(spec, env)
    if found:
        return found
    if allow_download is None:
        allow_download = auto_download_enabled(env)
    if allow_download and official_url(spec, env):
        return download_weights(spec, env, progress=progress)["path"]
    return spec


def _item(name, env=None, role=None):
    found = find_weights(name, env)
    return {
        "name": str(name),
        "role": role,
        "path": found,
        "present": found is not None,
        "size": os.path.getsize(found) if found else None,
        "downloadable": official_url(name, env) is not None,
        "target_path": os.path.abspath(target_path(name, env)),
    }


def status(cfg, env=None):
    """A /models/status válasz magja: a beállított súlyok és a models mappa tartalma."""
    mdir = app_paths.models_dir(env)
    items, seen = [], set()
    if cfg.get("weights"):
        roles = [("weights", cfg["weights"])]
    else:
        roles = [("cuda", cfg.get("weights_cuda") or "yolov8x.pt"),
                 ("cpu", cfg.get("weights_cpu") or "yolov8n.pt")]
    roles.append(("segmenter", "yolov8n.pt"))
    for role, name in roles:
        if name in seen:
            for it in items:
                if it["name"] == name:
                    it["role"] = f"{it['role']},{role}"
            continue
        seen.add(name)
        items.append(_item(name, env, role))
    local = []
    if os.path.isdir(mdir):
        for fn in sorted(os.listdir(mdir)):
            if fn.endswith((".pt", ".onnx")):
                p = os.path.join(mdir, fn)
                local.append({"name": fn, "path": os.path.abspath(p), "size": os.path.getsize(p)})
    return {
        "models_dir": os.path.abspath(mdir),
        "auto_download": auto_download_enabled(env),
        "configured": items,
        "local_files": local,
        "official": list(OFFICIAL_WEIGHTS),
    }

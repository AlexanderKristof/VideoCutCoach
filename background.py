"""
Háttér-mód: alacsony prioritás és kevés szál, hogy a hosszú munka (vágás, elemzés)
egy laptopon is futhasson úgy, hogy közben a gép használható marad.

- BackgroundConfig(enabled, threads): a beállítás. Szálhoz kötött (threading.local):
  a háttérfeladatot futtató szál a `context(...)` blokkban állítja be, és az ebből a
  szálból indított ffmpeg-folyamatok (fast_cut, match_segmenter) ezt látják.
  Parancssorból (elemzes.py --hatterben) a `set_default(...)` az egész folyamatra állítja.
- prepare(cmd): a gyermekfolyamat (ffmpeg) indítási paraméterei:
    Windows: creationflags=BELOW_NORMAL_PRIORITY_CLASS;
    POSIX: a parancs elé `nice -n 10` (ha a hívó szál még nem alacsony prioritású).
  Háttér-módban az ffmpeg parancsba `-threads N` is kerül (ffmpeg_thread_args).
- lower_current_thread_priority(): a jobot futtató szál prioritása (Linux: a szál
  nice értéke, Windows: THREAD_PRIORITY_BELOW_NORMAL). A szerver többi szála
  (HTTP kérések) normál prioritáson marad.
- lower_process_priority(): az egész folyamat prioritása (parancssori háttér-mód).
- limit_library_threads(n): torch / OpenCV / OpenMP szálszám korlátozása.
- on_battery(): akkumulátorról megy-e a gép (psutil-lal; ha nincs psutil: None).

Csak a szabványos könyvtárat használja; a psutil, torch, cv2 opcionális.
"""
import os
import sys
import shutil
import threading
import subprocess
from contextlib import contextmanager

# Windows prioritási osztályok / szálprioritás (a subprocess modulban csak Windows-on vannak meg)
BELOW_NORMAL_PRIORITY_CLASS = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
NORMAL_PRIORITY_CLASS = getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0x00000020)
THREAD_PRIORITY_BELOW_NORMAL = -1

# POSIX: ennyivel "kedvesebb" (alacsonyabb prioritású) a háttérmunka
NICE_LEVEL = 10

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS")


class BackgroundConfig:
    """Háttér-mód beállítása: enabled = alacsony prioritás, threads = szálkorlát (None = nincs)."""

    __slots__ = ("enabled", "threads")

    def __init__(self, enabled=True, threads=None):
        self.enabled = bool(enabled)
        self.threads = int(threads) if threads else None

    def __repr__(self):
        return f"BackgroundConfig(enabled={self.enabled}, threads={self.threads})"


_local = threading.local()
_default = None   # folyamatszintű alapérték (parancssor)


def set_default(cfg):
    """Folyamatszintű háttér-beállítás (pl. elemzes.py --hatterben). None = kikapcsolva."""
    global _default
    _default = cfg


def current():
    """Az aktuális szálra érvényes beállítás (vagy None, ha nincs háttér-mód és szálkorlát)."""
    cfg = getattr(_local, "cfg", None)
    return cfg if cfg is not None else _default


@contextmanager
def context(enabled=True, threads=None):
    """A blokkban (ebben a szálban) indított ffmpeg-folyamatok háttér-módban futnak."""
    prev = getattr(_local, "cfg", None)
    _local.cfg = BackgroundConfig(enabled, threads)
    try:
        yield _local.cfg
    finally:
        _local.cfg = prev


# ------------------------------------------------------------------
# Akkumulátor és alapértelmezett szálszám
# ------------------------------------------------------------------

def _import_psutil():
    try:
        import psutil  # opcionális
        return psutil
    except Exception:
        return None


def on_battery(_psutil=None):
    """True: akkumulátorról megy, False: hálózatról, None: nem állapítható meg (nincs psutil / akku)."""
    ps = _psutil if _psutil is not None else _import_psutil()
    if ps is None or not hasattr(ps, "sensors_battery"):
        return None
    try:
        b = ps.sensors_battery()
    except Exception:
        return None
    if b is None:
        return None
    return not bool(b.power_plugged)


def default_threads(cpu_count=None, battery=None):
    """
    Háttér-módban javasolt szálszám: a magok fele (legalább 1); akkumulátorról a negyede
    – így a gép többi része (és az akku) kíméletesebb terhelést kap.
    """
    n = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
    if battery is None:
        battery = on_battery()
    return max(1, n // 4) if battery else max(1, n // 2)


# ------------------------------------------------------------------
# Gyermekfolyamatok (ffmpeg)
# ------------------------------------------------------------------

def ffmpeg_thread_args(cfg=None):
    """['-threads', 'N'], ha szálkorlát van érvényben, különben []."""
    cfg = cfg if cfg is not None else current()
    if cfg is not None and cfg.threads:
        return ["-threads", str(cfg.threads)]
    return []


def _current_nice(_os=os):
    try:
        return _os.getpriority(_os.PRIO_PROCESS, 0)  # Linuxon a hívó szálé
    except (AttributeError, OSError):
        try:
            return _os.nice(0)
        except (AttributeError, OSError):
            return 0


def prepare(cmd, cfg=None, _os_name=None, _which=shutil.which, _nice_now=None):
    """
    A parancs és a Popen kulcsszavas paraméterei háttér-módhoz.
    Visszatér: (cmd, kwargs). Háttér-mód nélkül a parancs változatlan, kwargs üres.
    """
    cfg = cfg if cfg is not None else current()
    cmd = list(cmd)
    if cfg is None or not cfg.enabled:
        return cmd, {}
    os_name = _os_name or os.name
    if os_name == "nt":
        return cmd, {"creationflags": BELOW_NORMAL_PRIORITY_CLASS}
    nice_now = _nice_now if _nice_now is not None else _current_nice()
    if nice_now >= NICE_LEVEL:
        return cmd, {}  # a hívó szál már alacsony prioritású – a gyermek örökli
    nice_bin = _which("nice")
    if nice_bin:
        return [nice_bin, "-n", str(NICE_LEVEL - max(0, nice_now))] + cmd, {}
    return cmd, {}


def run(cmd, **kwargs):
    """subprocess.run háttér-mód szerinti prioritással."""
    cmd, extra = prepare(cmd)
    extra.update(kwargs)
    return subprocess.run(cmd, **extra)


def popen(cmd, **kwargs):
    """subprocess.Popen háttér-mód szerinti prioritással."""
    cmd, extra = prepare(cmd)
    extra.update(kwargs)
    return subprocess.Popen(cmd, **extra)


# ------------------------------------------------------------------
# A saját szál / folyamat prioritása
# ------------------------------------------------------------------

def lower_current_thread_priority(_os=os, _os_name=None, _kernel32=None):
    """
    A hívó szál prioritásának csökkentése. Visszatér: True, ha sikerült.
    Linux: a szál saját nice értéke (os.setpriority a szál natív azonosítójával);
    Windows: SetThreadPriority(BELOW_NORMAL). Más rendszeren (pl. macOS) nincs szálszintű
    nice – ott csak a gyermekfolyamatok kapnak alacsony prioritást.
    """
    os_name = _os_name or os.name
    if os_name == "nt":
        try:
            k32 = _kernel32
            if k32 is None:
                import ctypes
                k32 = ctypes.windll.kernel32
            return bool(k32.SetThreadPriority(k32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL))
        except Exception:
            return False
    if not sys.platform.startswith("linux") and _os is os:
        return False
    try:
        tid = threading.get_native_id()
        cur = _os.getpriority(_os.PRIO_PROCESS, tid)
        if cur < NICE_LEVEL:
            _os.setpriority(_os.PRIO_PROCESS, tid, NICE_LEVEL)
        return True
    except (AttributeError, OSError):
        return False


def lower_process_priority(_os=os, _os_name=None, _psutil=None, _kernel32=None):
    """
    Az egész folyamat prioritásának csökkentése (parancssori háttér-mód).
    Windows: psutil (ha van), különben SetPriorityClass(BELOW_NORMAL); POSIX: os.nice.
    A Windows-on alacsony prioritású folyamat gyermekei (ffmpeg) is alacsony prioritást örökölnek.
    Visszatér: True, ha sikerült.
    """
    os_name = _os_name or os.name
    if os_name == "nt":
        ps = _psutil if _psutil is not None else _import_psutil()
        if ps is not None and hasattr(ps, "BELOW_NORMAL_PRIORITY_CLASS"):
            try:
                ps.Process().nice(ps.BELOW_NORMAL_PRIORITY_CLASS)
                return True
            except Exception:
                pass
        try:
            k32 = _kernel32
            if k32 is None:
                import ctypes
                k32 = ctypes.windll.kernel32
            return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS))
        except Exception:
            return False
    try:
        cur = _os.nice(0)
        if cur < NICE_LEVEL:
            _os.nice(NICE_LEVEL - cur)
        return True
    except (AttributeError, OSError):
        return False


def limit_library_threads(n, set_env=True):
    """
    torch / OpenCV / OpenMP szálszám korlátozása n-re. A torch és a cv2 csak akkor kap
    beállítást, ha már be van töltve (itt nem töltjük be őket). Az OMP_* változók csak a
    később betöltött könyvtárakra hatnak (parancssorban a torch importja előtt kell hívni).
    Visszatér: a korábbi értékek (restore_library_threads-nek).
    """
    n = max(1, int(n))
    prev = {"env": {}, "torch": None, "cv2": None}
    if set_env:
        for var in THREAD_ENV_VARS:
            prev["env"][var] = os.environ.get(var)
            os.environ[var] = str(n)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            prev["torch"] = torch.get_num_threads()
            torch.set_num_threads(n)
        except Exception:
            prev["torch"] = None
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            prev["cv2"] = cv2.getNumThreads()
            cv2.setNumThreads(n)
        except Exception:
            prev["cv2"] = None
    return prev


def restore_library_threads(prev):
    if not prev:
        return
    for var, val in (prev.get("env") or {}).items():
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val
    torch = sys.modules.get("torch")
    if torch is not None and prev.get("torch"):
        try:
            torch.set_num_threads(prev["torch"])
        except Exception:
            pass
    cv2 = sys.modules.get("cv2")
    if cv2 is not None and prev.get("cv2") is not None:
        try:
            cv2.setNumThreads(prev["cv2"])
        except Exception:
            pass

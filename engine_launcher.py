"""
A motor indítója az asztali alkalmazáshoz (PyInstaller belépési pont: tacanaly_engine.exe).

    tacanaly_engine.exe --port 0 --parent-pid 1234
    python engine_launcher.py --port 0 --data-dir ./adat --console

Mit csinál:
  * alapból CSAK a 127.0.0.1-en figyel (a --lan kapcsolóval 0.0.0.0: a telefon is elérheti a
    helyi hálózaton – ekkor a hálózat bármely gépe hívhatja az API-t, lásd README_CSOMAGOLAS.md);
  * --port 0: az operációs rendszer választ szabad portot (versenyhelyzet nélkül: a launcher maga
    nyitja meg a socketet, és azt adja át a uvicornnak);
  * ha a szerver fogad kéréseket, a szabványos kimenetre EGYETLEN gépi sort ír:
        TACANALY_READY port=<port> pid=<pid>
    (hiba esetén: TACANALY_ERROR <üzenet>), és ugyanezt az <adatmappa>/engine.json fájlba is;
  * a többi kiírás (a motor print()-jei, uvicorn napló) a <logs>/engine.log forgó naplóba megy –
    így a gazdaalkalmazásnak nem kell folyamatosan olvasnia a stdout-ot (nem telik be a cső);
  * --parent-pid: ha a gazdafolyamat (MAUI) megszűnik, a motor kecsesen leáll (psutil, ha van,
    különben OS-szintű ellenőrzés);
  * a modellt NEM tölti be induláskor (a /health azonnal válaszol); a --preload device a torch
    importját és a GPU felismerését háttérben elvégzi, a --preload model a modellt is betölti.

Kilépési kódok: 0 rendben, 2 hibás paraméter, 3 a motor nem tölthető be, 4 a port nem nyitható meg.
"""
import os
import sys
import json
import time
import socket
import logging
import argparse
import threading
from logging.handlers import RotatingFileHandler

READY_PREFIX = "TACANALY_READY"
ERROR_PREFIX = "TACANALY_ERROR"
LOG_FILENAME = "engine.log"
STATE_FILENAME = "engine.json"

log = logging.getLogger("tacanaly.launcher")


def build_parser():
    p = argparse.ArgumentParser(prog="tacanaly_engine", description="TacAnaly elemző motor (helyi HTTP szerver)")
    p.add_argument("--host", default="127.0.0.1", help="figyelési cím (alap: 127.0.0.1 – csak a helyi gép)")
    p.add_argument("--port", type=int, default=0, help="port; 0 = szabad port választása (alap: 0)")
    p.add_argument("--lan", action="store_true",
                   help="figyelés minden hálózati címen (0.0.0.0) – a telefon is elérheti a helyi hálózaton. "
                        "BIZTONSÁG: ekkor a hálózat bármely gépe elérheti az API-t!")
    p.add_argument("--data-dir", default=None, help="adatmappa (alap: TACTICAL_DATA_DIR, csomagolva %%LOCALAPPDATA%%\\TacAnaly)")
    p.add_argument("--parent-pid", type=int, default=None, help="a gazdafolyamat PID-je: ha megszűnik, a motor leáll")
    p.add_argument("--control-token", default=None,
                   help="ha meg van adva, a /shutdown csak X-TacAnaly-Token fejléccel fogadható el "
                        "(alternatíva: TACTICAL_CONTROL_TOKEN környezeti változó)")
    p.add_argument("--preload", choices=("none", "device", "model"), default="device",
                   help="háttérben induláskor: none = semmi, device = torch + GPU felismerés (alap), "
                        "model = a modell betöltése is (szükség esetén letöltése)")
    p.add_argument("--console", action="store_true", help="a napló a képernyőre (stderr) is kerüljön")
    p.add_argument("--log-level", default="info", choices=("debug", "info", "warning", "error"))
    return p


# ---------------------------------------------------------------- READY sor
def ready_line(port, pid):
    return f"{READY_PREFIX} port={int(port)} pid={int(pid)}"


def parse_ready_line(line):
    """'TACANALY_READY port=1234 pid=56' -> {"port": 1234, "pid": 56}; más sor: None."""
    parts = (line or "").strip().split()
    if not parts or parts[0] != READY_PREFIX:
        return None
    out = {}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if sep and value.isdigit():
            out[key] = int(value)
    return out if "port" in out and "pid" in out else None


def open_listen_socket(host, port):
    """A figyelő socket megnyitása (port 0: az OS választ). Visszatér: (socket, valós port)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    if os.name != "nt":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    else:  # Windows: ne oszthassa meg más folyamat ugyanazt a portot
        excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if excl is not None:
            sock.setsockopt(socket.SOL_SOCKET, excl, 1)
    sock.bind((host, port))
    sock.listen(128)
    sock.set_inheritable(False)
    return sock, sock.getsockname()[1]


# ---------------------------------------------------------------- szülőfolyamat figyelése
def pid_alive(pid):
    """Él-e a folyamat (psutil, ha telepítve van; különben OS-szintű ellenőrzés)."""
    if pid is None or pid <= 0:
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return True
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: létezik
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def watch_parent(pid, on_dead, interval=1.0):
    """Háttérszál: ha a pid folyamat megszűnik, egyszer meghívja az on_dead()-et."""
    def _loop():
        while True:
            if not pid_alive(pid):
                on_dead()
                return
            time.sleep(interval)
    t = threading.Thread(target=_loop, name="parent-watch", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------- naplózás
class _LogStream:
    """print() -> napló: soronként a megadott loggerbe (opcionálisan egy másik streambe is)."""

    def __init__(self, logger, level, tee=None):
        self.logger, self.level, self.tee = logger, level, tee
        self._buf = ""
        self._lock = threading.Lock()
        self.encoding = "utf-8"

    def write(self, text):
        if not text:
            return 0
        if self.tee is not None:
            try:
                self.tee.write(text)
            except Exception:
                pass
        with self._lock:
            self._buf += text.replace("\r", "\n")
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self.logger.log(self.level, line)
        return len(text)

    def flush(self):
        if self.tee is not None:
            try:
                self.tee.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def setup_logging(logs_dir, console=False, level="info"):
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, LOG_FILENAME)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    fh = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    real_err = sys.__stderr__ or sys.stderr
    if console and real_err is not None:
        sh = logging.StreamHandler(real_err)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    return path


def _emit(stream, line):
    """Egy sor a gazdának (stdout); csomagolt, konzol nélküli futásnál a stream None lehet."""
    if stream is None:
        return
    try:
        stream.write(line + "\n")
        stream.flush()
    except Exception:
        pass


def _write_state(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as ex:
        log.warning("Az állapotfájl nem írható (%s): %s", path, ex)


def _hard_exit_later(delay, code=1):
    def _kill():
        time.sleep(delay)
        log.error("A kecses leállás nem fejeződött be %.0f mp alatt – kényszerített kilépés.", delay)
        logging.shutdown()
        os._exit(code)
    threading.Thread(target=_kill, name="hard-exit", daemon=True).start()


# ---------------------------------------------------------------- fő program
def main(argv=None):
    args = build_parser().parse_args(argv)
    real_stdout = sys.stdout

    if args.data_dir:
        os.environ["TACTICAL_DATA_DIR"] = os.path.abspath(os.path.expanduser(args.data_dir))
    if args.control_token:
        os.environ["TACTICAL_CONTROL_TOKEN"] = args.control_token
    host = "0.0.0.0" if args.lan else args.host
    if not (0 <= args.port <= 65535):
        _emit(real_stdout, f"{ERROR_PREFIX} érvénytelen port: {args.port}")
        return 2

    import app_paths
    try:
        app_paths.ensure_dirs()
    except OSError as ex:
        _emit(real_stdout, f"{ERROR_PREFIX} az adatmappa nem hozható létre: {ex}")
        return 3
    data_dir = os.path.abspath(app_paths.data_dir())
    logs_dir = os.path.abspath(app_paths.logs_dir())
    # A relatív utak (és az ultralytics esetleges kiírásai, pl. runs/) az adatmappába kerüljenek,
    # ne a telepítési mappába (Program Files). Fejlesztői módban a data_dir ".", nincs változás.
    os.chdir(data_dir)
    log_path = setup_logging(logs_dir, console=args.console, level=args.log_level)

    # A motor print()-jei a naplóba; a stdout-on csak a READY / ERROR sor marad
    sys.stdout = _LogStream(logging.getLogger("engine"), logging.INFO)
    sys.stderr = _LogStream(logging.getLogger("engine.stderr"), logging.WARNING)

    log.info("Indulás: pid=%s, host=%s, port=%s, adatmappa=%s, napló=%s",
             os.getpid(), host, args.port, data_dir, log_path)
    if args.lan:
        log.warning("LAN mód: a motor minden hálózati címen figyel – a helyi hálózat bármely gépe elérheti!")
    log.info("Útvonalak: %s", app_paths.summary())

    if app_paths.app_mode():
        # ultralytics: a beállításai az adatmappába kerüljenek (ne %APPDATA%\Ultralytics), és a
        # csomagolt programban ne próbáljon pip-pel csomagot telepíteni futás közben
        os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(data_dir, "ultralytics"))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    ffmpeg = app_paths.find_ffmpeg()
    if ffmpeg:
        os.environ.setdefault("FFMPEG_PATH", ffmpeg)
        # a main.py "ffmpeg.exe"-t hív – a PATH elejére tesszük a mappáját
        os.environ["PATH"] = os.path.dirname(ffmpeg) + os.pathsep + os.environ.get("PATH", "")
        log.info("ffmpeg: %s", ffmpeg)
    else:
        log.warning("Nem található ffmpeg (FFMPEG_PATH, a csomag ffmpeg mappája, PATH) – a vágás nem fog működni.")

    try:
        sock, port = open_listen_socket(host, args.port)
    except OSError as ex:
        log.error("A port nem nyitható meg (%s:%s): %s", host, args.port, ex)
        _emit(real_stdout, f"{ERROR_PREFIX} a port nem nyitható meg ({host}:{args.port}): {ex}")
        return 4

    t0 = time.monotonic()
    try:
        import main as engine  # torch / ultralytics NEM töltődik be (lusta modellbetöltés)
        import engine_control
    except BaseException as ex:
        log.exception("A motor nem tölthető be")
        _emit(real_stdout, f"{ERROR_PREFIX} a motor nem tölthető be: {type(ex).__name__}: {ex}")
        sock.close()
        return 3
    log.info("A motor betöltve %.2f mp alatt", time.monotonic() - t0)

    state_path = os.path.join(data_dir, STATE_FILENAME)

    def _on_started(server):
        info = {"port": port, "pid": os.getpid(), "host": host,
                "version": engine_control.ENGINE_VERSION,
                "url": f"http://127.0.0.1:{port}",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _write_state(state_path, info)
        log.info("KÉSZ: %s", ready_line(port, os.getpid()))
        _emit(real_stdout, ready_line(port, os.getpid()))

    if args.parent_pid:
        if not pid_alive(args.parent_pid):
            log.warning("A megadott szülőfolyamat (%s) már nem fut.", args.parent_pid)

        def _parent_dead():
            log.warning("A gazdafolyamat (%s) megszűnt – leállás.", args.parent_pid)
            engine_control.request_shutdown(engine.job_manager, grace_sec=5.0, reason="a gazdafolyamat megszűnt")
            _hard_exit_later(30.0)
        watch_parent(args.parent_pid, _parent_dead)

    if args.preload != "none":
        def _preload():
            try:
                if args.preload == "model":
                    engine.ensure_model()
                else:
                    engine.detect_device()
                log.info("Előtöltés kész: device=%s, modell=%s", engine.device, engine.model_weights)
            except BaseException as ex:
                log.warning("Előtöltés sikertelen (%s): %s", args.preload, ex)
        threading.Thread(target=_preload, name="preload", daemon=True).start()

    code = 0
    try:
        engine_control.serve(engine.app, host=host, port=port, sockets=[sock], on_started=_on_started,
                             log_config=False, log_level=args.log_level)
    except BaseException:
        log.exception("A szerver hibával állt le")
        code = 1
    finally:
        try:
            engine_control.mark_interrupted(engine.job_manager)
        except Exception:
            pass
        try:
            os.remove(state_path)
        except OSError:
            pass
        log.info("Leállt (kód: %s).", code)
        logging.shutdown()
    # A job-szálak (ThreadPoolExecutor) nem démonok: ne várjunk egy félbemaradt ffmpeg / elemzés végére
    os._exit(code)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # PyInstaller + Windows: a gyermekfolyamatok ne indítsák újra a szervert
    sys.exit(main())

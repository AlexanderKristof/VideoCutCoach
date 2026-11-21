"""
Az indító (engine_launcher.py) és a motor-vezérlés (engine_control.py) tesztjei.
Futtatás: python3 tests/test_launcher.py

A unit tesztek csak a szabványos könyvtárat használják. Az integrációs teszt a valódi main.py-t
indítja el egy alfolyamatban (torch / ultralytics csonkolva, internet nélkül), lekérdezi a /health,
/version, /models/* végpontokat, majd /shutdown-nal és a szülőfolyamat megszűnésével is leállítja.
Ha a szerver függőségei (fastapi, uvicorn, sqlalchemy, ...) nincsenek telepítve, az integrációs
teszt KIHAGYVA (a többi teszt lefut).
"""
import os
import sys
import json
import time
import queue
import shutil
import socket
import tempfile
import threading
import subprocess
import importlib.util
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import engine_launcher as L
import engine_control as C
from jobs import JobManager

SERVER_DEPS = ("fastapi", "uvicorn", "sqlalchemy", "jose", "passlib", "pydantic", "email_validator",
               "multipart", "cv2", "numpy")


class Skip(Exception):
    pass


# ---------------------------------------------------------------- unit
def test_ready_line_format_and_parse():
    line = L.ready_line(51234, 777)
    assert line == "TACANALY_READY port=51234 pid=777"
    assert L.parse_ready_line(line + "\r\n") == {"port": 51234, "pid": 777}
    assert L.parse_ready_line("INFO valami") is None
    assert L.parse_ready_line("TACANALY_READY port=x pid=1") is None
    assert L.parse_ready_line("") is None


def test_free_port_pick():
    s1, p1 = L.open_listen_socket("127.0.0.1", 0)
    s2, p2 = L.open_listen_socket("127.0.0.1", 0)
    try:
        assert 0 < p1 < 65536 and 0 < p2 < 65536 and p1 != p2
        c = socket.create_connection(("127.0.0.1", p1), timeout=2)  # már figyel (listen)
        c.close()
        try:
            L.open_listen_socket("127.0.0.1", p1)
            raise AssertionError("foglalt portot vártunk")
        except OSError:
            pass
    finally:
        s1.close()
        s2.close()


def test_pid_alive():
    assert L.pid_alive(os.getpid())
    assert not L.pid_alive(None) and not L.pid_alive(0)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not L.pid_alive(p.pid)


def test_watch_parent_calls_back():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    fired = threading.Event()
    L.watch_parent(p.pid, fired.set, interval=0.1)
    time.sleep(0.3)
    assert not fired.is_set()
    p.kill()
    p.wait()
    assert fired.wait(5)


def test_log_stream_splits_lines():
    import logging

    class H(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())

    lg = logging.getLogger("test.logstream")
    lg.propagate = False
    h = H()
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    st = L._LogStream(lg, logging.INFO)
    print("🔥 első sor\nmásodik", end="", file=st)
    print(" vége", file=st)
    st.write("\r[###---] 50%")
    st.write("\n")
    assert h.lines == ["🔥 első sor", "második vége", "[###---] 50%"], h.lines


def test_local_client_and_token():
    assert C.is_local_client("127.0.0.1") and C.is_local_client("::1") and C.is_local_client("127.0.1.1")
    assert not C.is_local_client("192.168.1.20") and not C.is_local_client(None)
    assert C.token_ok({}, env={})
    assert not C.token_ok({}, env={"TACTICAL_CONTROL_TOKEN": "abc"})
    assert C.token_ok({"x-tacanaly-token": "abc"}, env={"TACTICAL_CONTROL_TOKEN": "abc"})


def test_shutdown_cancels_and_marks_jobs():
    d = tempfile.mkdtemp(prefix="launcher_jobs_")
    try:
        jm = JobManager(d, max_workers=2)
        started = threading.Event()

        def coop(progress):
            started.set()
            while True:
                progress(10, "fut")
                time.sleep(0.02)

        stuck_release = threading.Event()
        stuck_started = threading.Event()

        def stuck(progress):  # nem kooperatív (pl. hosszú ffmpeg hívás)
            stuck_started.set()
            stuck_release.wait(10)
            return {"ok": True}

        j1 = jm.submit("coop", coop)
        j2 = jm.submit("stuck", stuck)
        assert started.wait(5)
        assert stuck_started.wait(5)  # különben a még sorban álló job "cancelled" lesz (időzítési verseny)
        stopped = threading.Event()

        class FakeServer:
            started = True

            @property
            def should_exit(self):
                return stopped.is_set()

            @should_exit.setter
            def should_exit(self, v):
                if v:
                    stopped.set()

        old = dict(C._state)
        try:
            C._state.update({"server": None, "shutting_down": False})
            C.register_server(FakeServer())
            ids = C.request_shutdown(jm, grace_sec=0.5, delay_sec=0.0)
            assert set(ids) == {j1, j2}
            assert C.request_shutdown(jm) is None  # második kérés: már folyamatban
            assert stopped.wait(5)
            assert jm.get(j1)["status"] == "cancelled"
            assert jm.get(j2)["status"] == "megszakadt"
        finally:
            stuck_release.set()
            C._state.clear()
            C._state.update(old)
            jm.shutdown(wait=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- integráció
STUB_TORCH = """
import os, types
open(os.environ["STUB_MARKER"], "a").write("torch\\n")
cuda = types.SimpleNamespace(is_available=lambda: False)
"""
STUB_ULTRALYTICS = """
import os
open(os.environ["STUB_MARKER"], "a").write("ultralytics\\n")
class YOLO:
    def __init__(self, weights, *a, **k):
        self.weights = weights
        self.names = {0: "person", 32: "sports ball"}
        self.callbacks = {}
    def add_callback(self, name, fn):
        self.callbacks.setdefault(name, []).append(fn)
"""


def _deps_missing():
    return [m for m in SERVER_DEPS if importlib.util.find_spec(m) is None]


def _http(method, url, data=None, timeout=10):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as ex:
        return ex.code, json.loads(ex.read() or b"null")


def _start_launcher(tmp, extra_args=(), env_extra=None):
    stubs = os.path.join(tmp, "stubs")
    for name, src in (("torch", STUB_TORCH), ("ultralytics", STUB_ULTRALYTICS)):
        os.makedirs(os.path.join(stubs, name), exist_ok=True)
        with open(os.path.join(stubs, name, "__init__.py"), "w", encoding="utf-8") as f:
            f.write(src)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": stubs + os.pathsep + env.get("PYTHONPATH", ""),
        "STUB_MARKER": os.path.join(tmp, "imported.txt"),
        "TACTICAL_AUTO_DOWNLOAD": "0",
        "TACTICAL_MODEL_BASE_URL": "http://127.0.0.1:9/",  # soha ne az internet
        "PYTHONIOENCODING": "utf-8",
    })
    env.pop("TACTICAL_MODEL_WEIGHTS", None)
    env.update(env_extra or {})
    data = os.path.join(tmp, "adat")
    cmd = [sys.executable, os.path.join(ROOT, "engine_launcher.py"), "--port", "0", "--data-dir", data,
           "--preload", "none", *extra_args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=tmp,
                            text=True, encoding="utf-8")
    lines = queue.Queue()

    def _reader():
        for line in proc.stdout:
            lines.put(line)
    threading.Thread(target=_reader, daemon=True).start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        info = L.parse_ready_line(line)
        if info:
            return proc, info, data, lines
        assert not line.startswith("TACANALY_ERROR"), line
    proc.kill()
    raise AssertionError("nincs READY sor; stderr: " + (proc.stderr.read() or "")[-2000:])


def _wait_job(base, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, r = _http("GET", f"{base}/jobs/{job_id}")
        job = (r or {}).get("job") or {}
        if code == 200 and job.get("status") in ("done", "error", "cancelled", "megszakadt"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"a job nem fejeződött be: {job_id}")


def test_integration_health_models_shutdown():
    missing = _deps_missing()
    if missing:
        raise Skip("hiányzó szerver-függőségek: " + ", ".join(missing))
    tmp = tempfile.mkdtemp(prefix="launcher_it_")
    proc = None
    try:
        t0 = time.monotonic()
        proc, info, data, lines = _start_launcher(tmp)
        assert info["pid"] == proc.pid
        base = f"http://127.0.0.1:{info['port']}"
        code, h = _http("GET", f"{base}/health")
        assert code == 200 and h["status"] == "ok", h
        assert h["model_loaded"] is False and h["device"] is None
        assert h["data_dir"] == os.path.abspath(data) and h["mode"] == "app"
        assert set(h) >= {"status", "version", "model_loaded", "device", "ffmpeg", "data_dir"}
        # indításkor sem a torch, sem az ultralytics nem töltődött be
        assert not os.path.exists(os.path.join(tmp, "imported.txt"))
        print(f"      (READY {time.monotonic() - t0:.1f} mp alatt, port {info['port']})")
        state = json.load(open(os.path.join(data, "engine.json"), encoding="utf-8"))
        assert state["port"] == info["port"] and state["pid"] == proc.pid
        code, v = _http("GET", f"{base}/version")
        assert code == 200 and v["version"] == C.ENGINE_VERSION and v["api_version"] == C.API_VERSION
        code, st = _http("GET", f"{base}/models/status")
        assert code == 200 and st["models_dir"] == os.path.join(os.path.abspath(data), "models")
        code, r = _http("POST", f"{base}/models/download", json.dumps({"name": "sajat.pt"}).encode())
        assert code == 400, (code, r)
        # a modell betöltése háttérfeladatként (a csonk YOLO-val)
        code, r = _http("POST", f"{base}/models/load")
        assert code == 200 and r["job_id"], r
        job = _wait_job(base, r["job_id"])
        assert job["status"] == "done", job
        code, h = _http("GET", f"{base}/health")
        assert h["model_loaded"] is True and h["device"] == "cpu", h
        # tábla és adatmappa
        for sub in ("clips", "jobs", "boards", "models", "logs"):
            assert os.path.isdir(os.path.join(data, sub)), sub
        assert os.path.isfile(os.path.join(data, "secret.key"))
        assert os.path.isfile(os.path.join(data, "tactical_app.db"))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{base}/tabla", timeout=10) as resp:
            assert resp.status == 200 and b"<html" in resp.read(200)
        # leállítás
        code, r = _http("POST", f"{base}/shutdown", json.dumps({"grace_sec": 2}).encode())
        assert code == 200 and r["status"] == "shutting_down", r
        assert proc.wait(20) == 0
        assert not os.path.exists(os.path.join(data, "engine.json"))
        log_text = open(os.path.join(data, "logs", "engine.log"), encoding="utf-8").read()
        assert "TACANALY_READY" in log_text and "Leállt" in log_text
        # a stdout-on a READY soron kívül semmi (a motor kiírásai a naplóba mennek)
        rest = []
        while not lines.empty():
            rest.append(lines.get())
        assert not [x for x in rest if x.strip()], rest
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def test_integration_exits_when_parent_dies():
    missing = _deps_missing()
    if missing:
        raise Skip("hiányzó szerver-függőségek: " + ", ".join(missing))
    tmp = tempfile.mkdtemp(prefix="launcher_parent_")
    proc = parent = None
    try:
        parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        proc, info, data, _ = _start_launcher(tmp, ["--parent-pid", str(parent.pid)])
        code, h = _http("GET", f"http://127.0.0.1:{info['port']}/health")
        assert code == 200
        parent.kill()
        parent.wait()
        assert proc.wait(20) == 0
    finally:
        for p in (proc, parent):
            if p is not None and p.poll() is None:
                p.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Skip as ex:
            print(f"KIHAGYVA {name}: {ex}")
        except Exception as ex:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"HIBA  {name}: {ex!r}")
    print(f"{len(tests) - failed}/{len(tests)} teszt sikeres (a kihagyottak sikeresnek számítanak)")
    sys.exit(1 if failed else 0)

"""
Az útvonalak (app_paths.py), a titkos kulcs és a súlykezelő (model_manager.py) tesztjei.
Futtatás: python3 tests/test_app_paths.py   (pytest, fastapi, torch nélkül fut; internetet nem használ –
a letöltést egy helyi HTTP szerverről teszteljük)
"""
import os
import io
import sys
import shutil
import zipfile
import tempfile
import threading
import contextlib
import http.server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app_paths
import model_manager
from jobs import ProgressReporter, JobCancelled


@contextlib.contextmanager
def frozen(meipass=None):
    """sys.frozen / sys._MEIPASS szimulálása (mint a PyInstaller csomagban)."""
    had_frozen, old_frozen = hasattr(sys, "frozen"), getattr(sys, "frozen", None)
    had_mei, old_mei = hasattr(sys, "_MEIPASS"), getattr(sys, "_MEIPASS", None)
    sys.frozen = True
    if meipass is not None:
        sys._MEIPASS = meipass
    try:
        yield
    finally:
        if had_frozen:
            sys.frozen = old_frozen
        else:
            del sys.frozen
        if had_mei:
            sys._MEIPASS = old_mei
        elif hasattr(sys, "_MEIPASS"):
            del sys._MEIPASS


def _tmp():
    return tempfile.mkdtemp(prefix="app_paths_test_")


# ---------------------------------------------------------------- útvonalak
def test_dev_mode_keeps_legacy_relative_paths():
    env = {}
    assert not app_paths.is_frozen()
    assert not app_paths.app_mode(env)
    assert app_paths.data_dir(env) == "."
    assert app_paths.clips_dir(env) == "clips"
    assert app_paths.jobs_dir(env) == os.path.join("clips", "jobs")
    assert app_paths.boards_dir(env) == os.path.join("clips", "boards")
    assert app_paths.models_dir(env) == "models"
    assert app_paths.database_url(env) == "sqlite:///./tactical_app.db"
    assert app_paths.resource_dir() == ROOT
    assert os.path.isfile(app_paths.resource_path("static", "taktikai_tabla.html"))
    assert app_paths.model_config_path(env) == os.path.join(ROOT, "model_config.json")
    assert app_paths.secret_key_path(env) == os.path.join(".", ".tacanaly_secret.key")


def test_env_data_dir_switches_to_app_mode():
    d = _tmp()
    try:
        env = {"TACTICAL_DATA_DIR": d}
        assert app_paths.app_mode(env)
        assert app_paths.data_dir(env) == os.path.abspath(d)
        assert app_paths.clips_dir(env) == os.path.join(d, "clips")
        assert app_paths.jobs_dir(env) == os.path.join(d, "jobs")
        assert app_paths.boards_dir(env) == os.path.join(d, "boards")
        assert app_paths.models_dir(env) == os.path.join(d, "models")
        assert app_paths.logs_dir(env) == os.path.join(d, "logs")
        url = app_paths.database_url(env)
        assert url.startswith("sqlite:///") and url.endswith("/tactical_app.db") and "\\" not in url
        created = app_paths.ensure_dirs(env)
        assert all(os.path.isdir(p) for p in created) and len(created) == 5
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_frozen_defaults_per_platform():
    env = {"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}
    assert app_paths.default_user_data_dir(env, platform="win32") == os.path.join(
        r"C:\Users\x\AppData\Local", "TacAnaly")
    home = os.path.expanduser("~")
    assert app_paths.default_user_data_dir({}, platform="linux") == os.path.join(home, ".local", "share", "TacAnaly")
    assert app_paths.default_user_data_dir({"XDG_DATA_HOME": "/xdg"}, platform="linux") == os.path.join("/xdg", "TacAnaly")
    mei = _tmp()
    try:
        with frozen(meipass=mei):
            assert app_paths.is_frozen()
            assert app_paths.app_mode({})
            assert app_paths.data_dir({}) == app_paths.default_user_data_dir({})
            assert app_paths.resource_dir() == mei
            assert app_paths.resource_path("static", "x.html") == os.path.join(mei, "static", "x.html")
            # a felhasználói model_config.json az adatmappában; ha nincs, a csomagolt
            d = _tmp()
            try:
                env = {"TACTICAL_DATA_DIR": d}
                assert app_paths.data_dir(env) == os.path.abspath(d)  # a környezeti változó erősebb
                assert app_paths.model_config_path(env) == os.path.join(d, "model_config.json")
                with open(os.path.join(mei, "model_config.json"), "w") as f:
                    f.write("{}")
                assert app_paths.model_config_path(env) == os.path.join(mei, "model_config.json")
                with open(os.path.join(d, "model_config.json"), "w") as f:
                    f.write("{}")
                assert app_paths.model_config_path(env) == os.path.join(d, "model_config.json")
            finally:
                shutil.rmtree(d, ignore_errors=True)
        assert not app_paths.is_frozen()
        assert app_paths.resource_dir() == ROOT
    finally:
        shutil.rmtree(mei, ignore_errors=True)


def test_find_ffmpeg_prefers_bundled_and_env():
    mei = _tmp()
    try:
        name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        os.makedirs(os.path.join(mei, "ffmpeg"))
        exe = os.path.join(mei, "ffmpeg", name)
        with open(exe, "w") as f:
            f.write("")
        with frozen(meipass=mei):
            assert app_paths.find_ffmpeg({}) == os.path.abspath(exe)
            other = os.path.join(mei, "egyedi_ffmpeg")
            with open(other, "w") as f:
                f.write("")
            assert app_paths.find_ffmpeg({"FFMPEG_PATH": other}) == os.path.abspath(other)
    finally:
        shutil.rmtree(mei, ignore_errors=True)


# ---------------------------------------------------------------- titkos kulcs
def test_secret_key_generated_once_and_persisted():
    d = _tmp()
    try:
        env = {"TACTICAL_DATA_DIR": d}
        k1 = app_paths.load_or_create_secret_key(env)
        assert len(k1) == 64 and all(c in "0123456789abcdef" for c in k1)
        path = os.path.join(d, "secret.key")
        assert os.path.isfile(path)
        if os.name != "nt":
            assert (os.stat(path).st_mode & 0o777) == 0o600
        k2 = app_paths.load_or_create_secret_key(env)
        assert k1 == k2  # újraindítás után is ugyanaz (a tokenek érvényesek maradnak)
        assert app_paths.load_or_create_secret_key({"TACTICAL_DATA_DIR": d, "TACTICAL_SECRET_KEY": "x" * 40}) == "x" * 40
        # túl rövid / sérült fájl: új kulcs
        with open(path, "w") as f:
            f.write("rovid")
        k3 = app_paths.load_or_create_secret_key(env)
        assert len(k3) == 64 and k3 != k1
        # másik telepítés: másik kulcs
        d2 = _tmp()
        try:
            assert app_paths.load_or_create_secret_key({"TACTICAL_DATA_DIR": d2}) != k3
        finally:
            shutil.rmtree(d2, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- súlykezelő
def _fake_pt_bytes(size_kb=200):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", os.urandom(size_kb * 1024))
    return buf.getvalue()


@contextlib.contextmanager
def http_server(files):
    """Helyi HTTP szerver: {"/név": bájtok}."""
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = files.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    old = {k: os.environ.get(k) for k in ("no_proxy", "NO_PROXY")}
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/"
    finally:
        srv.shutdown()
        srv.server_close()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_official_urls_and_auto_download_policy():
    assert model_manager.official_url("yolov8n.pt", {}) == model_manager.OFFICIAL_BASE_URL + "yolov8n.pt"
    assert model_manager.official_url("models/yolo11x.pt", {}).endswith("/yolo11x.pt")
    assert model_manager.official_url("football-players-detection.pt", {}) is None
    assert model_manager.official_url("yolov8n.pt", {"TACTICAL_MODEL_BASE_URL": "http://tukor/x"}) == "http://tukor/x/yolov8n.pt"
    assert not model_manager.auto_download_enabled({})                       # fejlesztői mód: régi viselkedés
    assert model_manager.auto_download_enabled({"TACTICAL_DATA_DIR": "/x"})  # alkalmazás-mód
    assert not model_manager.auto_download_enabled({"TACTICAL_DATA_DIR": "/x", "TACTICAL_AUTO_DOWNLOAD": "0"})
    assert model_manager.auto_download_enabled({"TACTICAL_AUTO_DOWNLOAD": "1"})


def test_find_weights_search_order():
    d = _tmp()
    try:
        env = {"TACTICAL_DATA_DIR": d}
        assert model_manager.find_weights("yolov8n.pt", env) is None
        os.makedirs(os.path.join(d, "models"))
        p = os.path.join(d, "models", "sajat.pt")
        with open(p, "wb") as f:
            f.write(b"PK..")
        assert model_manager.find_weights("sajat.pt", env) == os.path.abspath(p)
        assert model_manager.find_weights("models/sajat.pt", env) == os.path.abspath(p)
        assert model_manager.find_weights(p, env) == os.path.abspath(p)
        # fejlesztői módban hiányzó súly: változatlan név megy a YOLO-nak, letöltés NINCS
        assert model_manager.resolve_for_load("yolov8n.pt", {}) == "yolov8n.pt"
        # alkalmazás-módban, ha a letöltés tiltva: szintén a név
        assert model_manager.resolve_for_load("yolov8n.pt", dict(env, TACTICAL_AUTO_DOWNLOAD="0")) == "yolov8n.pt"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_download_with_progress_from_local_server():
    d = _tmp()
    data = _fake_pt_bytes()
    try:
        with http_server({"/yolov8n.pt": data, "/yolov8s.pt": b"<html>login</html>"}) as base:
            env = {"TACTICAL_DATA_DIR": d, "TACTICAL_MODEL_BASE_URL": base}
            seen = []
            res = model_manager.download_weights("yolov8n.pt", env, progress=lambda p, ph: seen.append((p, ph)))
            assert res["downloaded"] and res["size"] == len(data)
            assert res["path"] == os.path.join(os.path.abspath(d), "models", "yolov8n.pt")
            assert not os.path.exists(res["path"] + ".part")
            assert seen[0][0] == 0.0 and seen[-1][0] == 100.0
            # már megvan: nem tölt le újra
            again = model_manager.download_weights("yolov8n.pt", env)
            assert not again["downloaded"]
            # resolve_for_load alkalmazás-módban: a models mappabeli fájl
            assert model_manager.resolve_for_load("yolov8n.pt", env) == res["path"]
            # nem .pt (pl. proxy bejelentkező oldal): hiba, nincs félkész fájl
            try:
                model_manager.download_weights("yolov8s.pt", env)
                raise AssertionError("hibát vártunk")
            except model_manager.ModelDownloadError:
                pass
            assert not os.path.exists(os.path.join(d, "models", "yolov8s.pt"))
            assert not os.path.exists(os.path.join(d, "models", "yolov8s.pt.part"))
            # 404
            try:
                model_manager.download_weights("yolov8m.pt", env)
                raise AssertionError("hibát vártunk")
            except model_manager.ModelDownloadError:
                pass
            # ismeretlen név
            try:
                model_manager.download_weights("sajat.pt", env)
                raise AssertionError("hibát vártunk")
            except model_manager.ModelDownloadError:
                pass
            # automatikus letöltés első használatkor (alkalmazás-mód)
            os.remove(res["path"])
            assert model_manager.resolve_for_load("yolov8n.pt", env) == res["path"]
            assert os.path.getsize(res["path"]) == len(data)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_download_cancel_removes_partial_file():
    d = _tmp()
    try:
        with http_server({"/yolov8n.pt": _fake_pt_bytes(2048)}) as base:
            env = {"TACTICAL_DATA_DIR": d, "TACTICAL_MODEL_BASE_URL": base}
            reporter = ProgressReporter()
            reporter.cancel()
            try:
                model_manager.download_weights("yolov8n.pt", env, progress=reporter)
                raise AssertionError("JobCancelled-et vártunk")
            except JobCancelled:
                pass
            assert os.listdir(os.path.join(d, "models")) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_status_shape():
    d = _tmp()
    try:
        env = {"TACTICAL_DATA_DIR": d}
        st = model_manager.status({"weights": None, "weights_cuda": "yolov8x.pt", "weights_cpu": "yolov8n.pt"}, env)
        names = [(i["name"], i["role"]) for i in st["configured"]]
        assert names == [("yolov8x.pt", "cuda"), ("yolov8n.pt", "cpu,segmenter")]
        assert all(not i["present"] and i["downloadable"] for i in st["configured"])
        assert st["auto_download"] and st["local_files"] == []
        st2 = model_manager.status({"weights": "models/football.pt"}, env)
        assert st2["configured"][0]["role"] == "weights" and not st2["configured"][0]["downloadable"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Exception as ex:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"HIBA  {name}: {ex!r}")
    print(f"{len(tests) - failed}/{len(tests)} teszt sikeres")
    sys.exit(1 if failed else 0)

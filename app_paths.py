"""
Az alkalmazás adat- és erőforrás-útvonalai – EGY helyen.

Két üzemmód:
  * fejlesztői (python main.py / python elemzes.py, nem "fagyasztott"): a régi, a munkakönyvtárhoz
    (CWD) relatív útvonalak VÁLTOZATLANOK: clips/, clips/jobs/, clips/boards/, ./tactical_app.db,
    models/, logs/;
  * alkalmazás-mód: ha be van állítva a TACTICAL_DATA_DIR környezeti változó, VAGY a program
    PyInstallerrel csomagolt (sys.frozen). Ekkor minden írható adat az adatmappába kerül:
        Windows:  %LOCALAPPDATA%\\TacAnaly
        máshol:   ~/.local/share/TacAnaly  (vagy $XDG_DATA_HOME/TacAnaly)
    benne: clips/, jobs/, boards/, models/, logs/, tactical_app.db, secret.key, model_config.json

Erőforrások (a csomagba fordított, csak olvasható fájlok: static/, model_config.example.json):
  fejlesztői módban a forrásmappa (e fájl mappája), csomagolva a PyInstaller kicsomagolási mappája
  (sys._MEIPASS; one-folder buildnél az _internal mappa).

A függvények minden híváskor a környezeti változókat olvassák (tesztelhető); a main.py az
importkor egyszer hívja őket. Csak a szabványos könyvtárat használja.
"""
import os
import sys
import shutil
import secrets

APP_NAME = "TacAnaly"
ENV_DATA_DIR = "TACTICAL_DATA_DIR"
ENV_SECRET_KEY = "TACTICAL_SECRET_KEY"
SECRET_KEY_FILENAME = "secret.key"
# fejlesztői módban a CWD-be kerül – rejtett név, a .gitignore kizárja
DEV_SECRET_KEY_FILENAME = ".tacanaly_secret.key"


def _env(env):
    return os.environ if env is None else env


def is_frozen():
    """PyInstallerrel (vagy más fagyasztóval) csomagolt futás."""
    return bool(getattr(sys, "frozen", False))


def default_user_data_dir(env=None, platform=None):
    """Az operációs rendszer szerinti felhasználói adatmappa (létrehozás nélkül)."""
    env = _env(env)
    platform = platform or sys.platform
    if platform.startswith("win"):
        base = env.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, APP_NAME)
    base = env.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_NAME)


def app_mode(env=None):
    """Alkalmazás-mód: van TACTICAL_DATA_DIR, vagy csomagolt a program."""
    return bool(_env(env).get(ENV_DATA_DIR)) or is_frozen()


def data_dir(env=None):
    """
    Az írható adatok gyökere. Alkalmazás-módban abszolút útvonal (TACTICAL_DATA_DIR vagy az
    OS szerinti alap), fejlesztői módban "." (a CWD – a régi viselkedés).
    """
    env = _env(env)
    if env.get(ENV_DATA_DIR):
        return os.path.abspath(os.path.expanduser(env[ENV_DATA_DIR]))
    if is_frozen():
        return default_user_data_dir(env)
    return "."


def _sub(name, env=None, legacy=None):
    """Almappa: alkalmazás-módban <adatmappa>/<name>, fejlesztői módban a régi relatív út."""
    if app_mode(env):
        return os.path.join(data_dir(env), name)
    return legacy if legacy is not None else name


def clips_dir(env=None):
    """Kimeneti klipek, képek, eseményfájlok (a szerver /clips alatt szolgálja ki)."""
    return _sub("clips", env)


def jobs_dir(env=None):
    """Háttérfeladatok metaadatai. Fejlesztői módban a régi clips/jobs."""
    return _sub("jobs", env, legacy=os.path.join("clips", "jobs"))


def boards_dir(env=None):
    """Mentett taktikai táblák. Fejlesztői módban a régi clips/boards."""
    return _sub("boards", env, legacy=os.path.join("clips", "boards"))


def models_dir(env=None):
    """A letöltött / bemásolt súlyfájlok (.pt) helye."""
    return _sub("models", env)


def logs_dir(env=None):
    """Naplófájlok (engine_launcher.py)."""
    return _sub("logs", env)


def db_path(env=None):
    """Az SQLite adatbázis fájl."""
    if app_mode(env):
        return os.path.join(data_dir(env), "tactical_app.db")
    return "./tactical_app.db"


def database_url(env=None):
    """SQLAlchemy URL. Fejlesztői módban pontosan a régi: sqlite:///./tactical_app.db"""
    path = db_path(env)
    if app_mode(env):
        return "sqlite:///" + os.path.abspath(path).replace("\\", "/")
    return "sqlite:///" + path


def secret_key_path(env=None):
    if app_mode(env):
        return os.path.join(data_dir(env), SECRET_KEY_FILENAME)
    return os.path.join(".", DEV_SECRET_KEY_FILENAME)


def resource_dir():
    """A csak olvasható, csomagolt fájlok gyökere (static/, model_config.example.json)."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return meipass
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def install_dir():
    """A program mappája: csomagolva az exe mappája (one-folder: ahol a tacanaly_engine.exe van),
    fejlesztői módban a forrásmappa. Ide tehet a telepítő ffmpeg/ és models/ almappát."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    return os.path.join(resource_dir(), *parts)


def model_config_path(env=None):
    """
    A felhasználói model_config.json helye: alkalmazás-módban az adatmappában (a csomag
    írásvédett lehet), fejlesztői módban a forrásmappában (a main.py mellett – a régi hely).
    """
    if app_mode(env):
        user = os.path.join(data_dir(env), "model_config.json")
        bundled = resource_path("model_config.json")  # a telepítő mellékelheti
        if not os.path.isfile(user) and os.path.isfile(bundled):
            return bundled
        return user
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config.json")


def event_tuning_path(env=None):
    """
    A címkékből hangolt esemény-küszöbök (tools/hangolas.py kimenete): alkalmazás-módban az adatmappában,
    fejlesztői módban a forrásmappában (a model_config.json mellett).
    """
    if app_mode(env):
        return os.path.join(data_dir(env), "esemeny_hangolas.json")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "esemeny_hangolas.json")


def ensure_dirs(env=None):
    """Az írható mappák létrehozása (alkalmazás-módban az adatmappával együtt)."""
    paths = [clips_dir(env), jobs_dir(env), boards_dir(env)]
    if app_mode(env):
        paths += [models_dir(env), logs_dir(env)]
    for p in paths:
        os.makedirs(p, exist_ok=True)
    return paths


# ---------------------------------------------------------------- titkos kulcs
def _write_private(path, text):
    """Új fájl csak a tulajdonos számára olvasható jogosultsággal (POSIX 0600; Windows-on a
    %LOCALAPPDATA% eleve felhasználónkénti). Ha már létezik, FileExistsError."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_or_create_secret_key(env=None, path=None):
    """
    A JWT aláírókulcs: TACTICAL_SECRET_KEY környezeti változó, különben az adatmappában tárolt
    véletlen kulcs (első induláskor generálva, 256 bit). Így minden telepítésnek saját kulcsa van,
    és újraindítás után is érvényesek maradnak a kiadott tokenek.
    """
    env = _env(env)
    if env.get(ENV_SECRET_KEY):
        return env[ENV_SECRET_KEY]
    path = path or secret_key_path(env)
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if len(key) >= 32:
            return key
        print(f"⚠️ A titkos kulcs fájl túl rövid ({path}) – újat generálunk.")
        os.remove(path)
    except FileNotFoundError:
        pass
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    key = secrets.token_hex(32)
    try:
        _write_private(path, key + "\n")
    except FileExistsError:  # egy másik folyamat közben létrehozta
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return key


# ---------------------------------------------------------------- ffmpeg
def find_ffmpeg(env=None):
    """
    Az ffmpeg futtatható állomány: FFMPEG_PATH, a csomag mellé tett ffmpeg (ffmpeg/ffmpeg.exe,
    ffmpeg.exe a resource / exe mappában), végül a PATH. Nincs találat: None.
    """
    env = _env(env)
    explicit = env.get("FFMPEG_PATH")
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        found = shutil.which(explicit)
        if found:
            return found
    names = ("ffmpeg.exe", "ffmpeg") if os.name == "nt" else ("ffmpeg", "ffmpeg.exe")
    roots = [resource_dir()]
    if install_dir() not in roots:
        roots.append(install_dir())
    for root in roots:
        for sub in ("ffmpeg", os.path.join("ffmpeg", "bin"), ""):
            for n in names:
                cand = os.path.join(root, sub, n)
                if os.path.isfile(cand):
                    return os.path.abspath(cand)
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    return None


def summary(env=None):
    """Áttekintés naplózáshoz / a /health végponthoz."""
    return {
        "mode": "app" if app_mode(env) else "dev",
        "frozen": is_frozen(),
        "data_dir": os.path.abspath(data_dir(env)),
        "clips_dir": os.path.abspath(clips_dir(env)),
        "jobs_dir": os.path.abspath(jobs_dir(env)),
        "boards_dir": os.path.abspath(boards_dir(env)),
        "models_dir": os.path.abspath(models_dir(env)),
        "logs_dir": os.path.abspath(logs_dir(env)),
        "db_path": os.path.abspath(db_path(env)),
        "resource_dir": resource_dir(),
        "install_dir": install_dir(),
    }

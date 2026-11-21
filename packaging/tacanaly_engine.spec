# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller leíró a TacAnaly elemző motorhoz (one-folder build).

Futtatás a repó gyökeréből (a build_engine.ps1 ezt teszi):
    pyinstaller --noconfirm --clean --distpath packaging/dist --workpath packaging/build packaging/tacanaly_engine.spec

Kimenet: packaging/dist/tacanaly_engine/
    tacanaly_engine.exe        – a belépési pont (engine_launcher.py)
    _internal/                 – Python futtatókörnyezet, torch, OpenCV, static/, model_config.example.json
    (ide másolja a build_engine.ps1 az ffmpeg.exe-t is, ha megadod)

Miért one-folder: a torch (különösen a CUDA-s) több GB; one-file módban minden indításkor egy temp
mappába csomagolná ki (lassú indulás, telenő %TEMP%). A telepítő az egész mappát másolja.

Környezeti változók a buildhez:
    TACANALY_SPEC_NO_ML=1  – torch / ultralytics NÉLKÜLI próbabuild (a csomag szerkezetének és az
                             indító /health /shutdown működésének ellenőrzésére; elemezni nem tud)
"""
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
WITH_ML = os.environ.get("TACANALY_SPEC_NO_ML") != "1"

# A motor saját moduljai (a main.py statikusan importálja őket, de a függvényen belüli importok
# miatt biztosra megyünk)
APP_MODULES = [
    "main", "app_paths", "engine_control", "model_manager", "model_config", "database", "models", "auth",
    "jobs", "pipeline", "boards", "tactical_engine", "events", "pitch_filters", "pitch_tracker",
    "ball_tracker", "team_classifier", "match_segmenter", "heatmap", "clip_export", "review_export",
]

datas = [
    (os.path.join(ROOT, "static"), "static"),
    (os.path.join(ROOT, "model_config.example.json"), "."),
]
# Opcionális: a telepítő által mellékelt beállítás (ha van model_config.json a repó gyökerében)
if os.path.isfile(os.path.join(ROOT, "model_config.json")):
    datas.append((os.path.join(ROOT, "model_config.json"), "."))

binaries = []
# OpenCV H.264 kódoló Windows-on (a VideoWriter tölti be) – a repóban mellékelt DLL
_h264 = os.path.join(ROOT, "openh264-2.5.0-win64.dll")
if os.path.isfile(_h264):
    binaries.append((_h264, "."))

hiddenimports = list(APP_MODULES)
hiddenimports += collect_submodules("uvicorn")          # loops / protocols / lifespan dinamikus import
hiddenimports += [
    "passlib.handlers.bcrypt", "bcrypt",                 # a passlib a sémát név szerint tölti be
    "jose", "jose.jwt", "email_validator",
    "multipart", "python_multipart",                     # Form(...) mezők
    "sqlalchemy.dialects.sqlite",
    "psutil",                                            # --parent-pid figyelés (opcionális)
]

excludes = [
    "tests", "tools", "synth_match", "fake_detector", "eval_events", "analyzer",
    "tkinter", "pytest", "IPython", "jupyter", "notebook", "tensorboard",
]

if WITH_ML:
    # Az ultralytics a cfg/*.yaml fájlokat (default.yaml, trackers/bytetrack.yaml) futáskor olvassa
    datas += collect_data_files("ultralytics")
    hiddenimports += collect_submodules("ultralytics")
    hiddenimports += ["lap"]                             # ByteTrack hozzárendelés (különben pip-pel telepítené)
else:
    excludes += ["torch", "torchvision", "ultralytics", "lap"]

a = Analysis(
    [os.path.join(ROOT, "engine_launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # one-folder
    name="tacanaly_engine",
    debug=False,
    strip=False,
    upx=False,                  # a UPX tönkreteheti a torch / CUDA DLL-eket
    console=True,               # a READY sor a stdout-on jön; a MAUI CreateNoWindow=true-val indítja
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="tacanaly_engine",
)

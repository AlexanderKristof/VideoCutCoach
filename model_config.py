"""
Felismerő modell beállításai (súlyfájl, osztálynevek, gyors mód).

Forrás (erősorrendben):
  1. környezeti változók:
       TACTICAL_MODEL_WEIGHTS   – a .pt súlyfájl útvonala (mindkét eszközre)
       TACTICAL_DETECT_EVERY_N  – csak minden N. képkockán fut felismerés (gyors mód)
       TACTICAL_MODEL_CONFIG    – a JSON beállítófájl útvonala
       TACTICAL_DETECTOR_BACKEND – "ultralytics" (alap) | "onnx" | "ultralytics_predict"
       TACTICAL_ONNX_PATH       – az ONNX modellfájl útvonala (backend = "onnx")
       TACTICAL_BALL_CROP_PASS  – natív felbontású labda-kivágás: auto / 1 / 0
       TACTICAL_PITCH_REFINE_MAX_EVERY – pályakövetés: a vonal-finomítás legritkább üteme (képkocka; 3 = régi)
       TACTICAL_GRASS_MASK_EVERY – a fűmaszk újraszámolása ennyi képkockánként (1 = minden képkockán)
       TACTICAL_PARALLEL_PITCH  – pályakövetés külön szálon, a felismeréssel párhuzamosan: auto / 1 / 0
  2. model_config.json (opcionális, a main.py mellett; a telepített alkalmazásban az adatmappában,
     lásd app_paths.py)
  3. alapértelmezés: yolov8x.pt (CUDA) / yolov8n.pt (CPU), COCO osztályok, minden képkocka

Ez a modul szándékosan nem importál torch-ot / ultralytics-et, hogy önállóan tesztelhető legyen.
"""
import json
import os

ROLES = ("player", "goalkeeper", "referee", "ball")

# Szerepkör -> ismert osztálynevek (kisbetűs, normalizált alak).
# COCO: "person", "sports ball"; Roboflow focis modellek: "player", "goalkeeper", "referee", "ball".
# Magyar nevek is (saját tanítású modell: docs/MODELL_TANITAS.md – "jatekos", "kapus", "biro", "labda").
DEFAULT_ROLE_ALIASES = {
    "player": {"player", "person", "players", "játékos", "jatekos"},
    "goalkeeper": {"goalkeeper", "goalie", "keeper", "goalkeepers", "kapus"},
    "referee": {"referee", "referees", "ref", "bíró", "biro", "játékvezető", "jatekvezeto"},
    "ball": {"ball", "sports ball", "football", "soccer ball", "soccerball", "labda"},
}

DEFAULTS = {
    "weights": None,          # ha meg van adva, mindkét eszközre ezt használjuk
    "weights_cuda": "yolov8x.pt",
    "weights_cpu": "yolov8n.pt",
    "class_names": None,      # opcionális: {"player": ["..."], "ball": ["..."], ...}
    "detect_every_n": 1,
    "ball_conf": 0.35,
    # Külön labda-felismerő menet (második modellpéldány), ha a nyers labda-callback nem érhető el
    # (lásd main.py NYERS LABDA-DETEKTÁLÁSOK, docs/MODELL_BEALLITAS.md) – drága, alapból KI
    "ball_extra_pass": False,
    # --- SEBESSÉG / KIS LABDA (lásd main.py LABDA-KIVÁGÁS, docs/MODELL_BEALLITAS.md) ---
    "imgsz": 640,             # a fő (játékos) felismerés bemeneti mérete (ultralytics imgsz)
    "half": "auto",           # félpontosság (FP16): "auto" = CUDA-n be, CPU-n ki; true / false
    # Natív felbontású labda-kivágás: a becsült labdahely körüli (ball_crop_size px-es) kivágáson külön
    # labda-felismerés (a teljes kép 640-re kicsinyítve a 1080p Veo-képen 2–4 px-es labdát nem látja);
    # elveszett labdánál a pálya csempéin körbeforgó keresés (ball_tile_every képkockánként egy csempe).
    # "auto" = CUDA-n be, CPU-n ki
    "ball_crop_pass": "auto",
    "ball_crop_size": 640,
    "ball_tile_every": 2,
    "ball_crop_conf": 0.10,   # a kivágásos menet minimális pontszáma (a labdakövető szűr tovább)
    # --- CPU-SZAKASZOK GYORSÍTÁSA (lásd main.py, pitch_tracker.py, docs/MODELL_BEALLITAS.md) ---
    # Pályakövetés, adaptív vonal-finomítás: stabil követésnél (a vonalak rendre megerősítik a
    # kamera-előrejelzést) legfeljebb ennyi képkockánként fut a drága vonalkeresés + illesztés;
    # gyanús jelnél azonnal vissza a 3 képkockás ütemre. 3 = a régi, rögzített ütemezés.
    "pitch_refine_max_every": 5,
    # A (videónként tanult) fűmaszk újraszámolása ennyi képkockánként; közte az előző maszk a
    # kameramozgással eltolva (a fű nem mozog, a kamera lassan pásztáz). 1 = minden képkockán (régi).
    "grass_mask_every": 2,
    # Pályakövetés (+ kameramozgás) külön szálon, a felismeréssel párhuzamosan: a GPU-s felismerés
    # alatt a CPU a pályakövetést számolja. Az eredmény ugyanaz, csak az átfedés gyorsít. "auto" = CUDA-n be.
    "parallel_pitch": "auto",
    # --- DETEKTOR-HÁTTÉR (lásd detection/, docs/MODELL_BEALLITAS.md 8. pont) ---
    # "ultralytics": model.track() + beépített ByteTrack (AGPL-3.0 – csak fejlesztéshez!)
    # "onnx": ONNX Runtime detektor + saját követő (detection/tracker.py) – a kereskedelmi út
    # "ultralytics_predict": ultralytics predict + saját követő (fejlesztői összevetéshez)
    "backend": "ultralytics",
    "onnx_path": None,        # pl. "models/focis_detektor.onnx" (relatív útvonal: a main.py mappájához)
    "input_size": None,       # [szélesség, magasság] vagy egy szám; alap: a modellből
    "onnx_class_names": None, # a modell osztálynevei (lista / {id: név} / "coco80"), ha a modell nem tárolja
    "onnx": None,             # további OnnxDetector beállítások (preset, iou, normalize, ...)
    "providers": "auto",      # "auto" = CUDA > DirectML > CPU, vagy lista
    "role_conf": None,        # szerepkörönkénti min. pontszám, pl. {"ball": 0.05, "player": 0.1}
    "tracker": None,          # a saját követő paraméterei (detection/tracker.py ByteTracker)
}

BACKENDS = ("ultralytics", "onnx", "ultralytics_predict")

DEFAULT_CONFIG_FILENAME = "model_config.json"


def _normalize_name(name):
    """Osztálynév egységesítése: kisbetű, '_' és '-' helyett szóköz, felesleges szóközök nélkül."""
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


def _parse_every_n(value):
    """A gyors mód lépésköze: egész szám, legalább 1. Hibás értéknél 1 (kikapcsolva)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        print(f"⚠️ Érvénytelen detect_every_n érték: {value!r} – 1-et használunk (minden képkocka).")
        return 1
    if n < 1:
        print(f"⚠️ A detect_every_n legalább 1 lehet (kapott: {n}) – 1-et használunk.")
        return 1
    return n


def load_model_config(config_path=None, env=None):
    """
    Beállítások betöltése: alapértékek <- JSON fájl (ha létezik) <- környezeti változók.
    A hiányzó JSON fájl nem hiba (visszafelé kompatibilis alapműködés).
    """
    env = os.environ if env is None else env
    cfg = dict(DEFAULTS)

    if config_path is None:
        # fejlesztői módban a main.py mellett, alkalmazás-módban az adatmappában (app_paths.py)
        import app_paths
        config_path = env.get("TACTICAL_MODEL_CONFIG") or app_paths.model_config_path(env)

    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig: a PowerShell Set-Content BOM-mal ír
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("a fájl gyökere nem JSON objektum")
            for key in DEFAULTS:
                if key in data:
                    cfg[key] = data[key]
            unknown = [k for k in data if k not in DEFAULTS and not k.startswith("_")]
            if unknown:
                print(f"⚠️ Ismeretlen kulcsok a modell beállításban ({config_path}): {', '.join(unknown)}")
        except (OSError, ValueError) as ex:
            print(f"⚠️ A modell beállítófájl nem olvasható ({config_path}): {ex} – alapértékekkel futunk.")

    if env.get("TACTICAL_MODEL_WEIGHTS"):
        cfg["weights"] = env["TACTICAL_MODEL_WEIGHTS"]
    if env.get("TACTICAL_DETECT_EVERY_N"):
        cfg["detect_every_n"] = env["TACTICAL_DETECT_EVERY_N"]

    if env.get("TACTICAL_DETECTOR_BACKEND"):
        cfg["backend"] = env["TACTICAL_DETECTOR_BACKEND"]
    if env.get("TACTICAL_ONNX_PATH"):
        cfg["onnx_path"] = env["TACTICAL_ONNX_PATH"]
    backend = str(cfg.get("backend") or "ultralytics").strip().lower()
    if backend not in BACKENDS:
        print(f"⚠️ Ismeretlen detektor-háttér: {cfg['backend']!r} (lehet: {', '.join(BACKENDS)}) – ultralytics")
        backend = "ultralytics"
    cfg["backend"] = backend

    cfg["detect_every_n"] = _parse_every_n(cfg["detect_every_n"])
    try:
        cfg["ball_conf"] = float(cfg["ball_conf"])
    except (TypeError, ValueError):
        print(f"⚠️ Érvénytelen ball_conf érték: {cfg['ball_conf']!r} – 0.35-öt használunk.")
        cfg["ball_conf"] = DEFAULTS["ball_conf"]
    v = cfg["ball_extra_pass"]
    cfg["ball_extra_pass"] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "igen", "yes", "on")
    if env.get("TACTICAL_BALL_CROP_PASS"):
        cfg["ball_crop_pass"] = env["TACTICAL_BALL_CROP_PASS"]
    for key, var in (("pitch_refine_max_every", "TACTICAL_PITCH_REFINE_MAX_EVERY"),
                     ("grass_mask_every", "TACTICAL_GRASS_MASK_EVERY"), ("parallel_pitch", "TACTICAL_PARALLEL_PITCH")):
        if env.get(var):
            cfg[key] = env[var]
    for key, conv, lo in (("imgsz", int, 32), ("ball_crop_size", int, 64), ("ball_tile_every", int, 1),
                          ("ball_crop_conf", float, 0.0), ("pitch_refine_max_every", int, 1),
                          ("grass_mask_every", int, 1)):
        try:
            cfg[key] = max(lo, conv(cfg[key]))
        except (TypeError, ValueError):
            print(f"⚠️ Érvénytelen {key} érték: {cfg[key]!r} – alapérték: {DEFAULTS[key]}")
            cfg[key] = DEFAULTS[key]
    return cfg


def auto_flag(value, device):
    """"auto" / None -> CUDA-n igaz, máshol hamis; egyébként logikai érték (szövegből is)."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("auto", "")):
        return device == "cuda"
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "igen", "yes", "on")


def resolve_weights(cfg, device):
    """A betöltendő súlyfájl az eszköz (cuda / cpu) szerint."""
    if cfg.get("weights"):
        return cfg["weights"]
    if device == "cuda":
        return cfg.get("weights_cuda") or DEFAULTS["weights_cuda"]
    return cfg.get("weights_cpu") or DEFAULTS["weights_cpu"]


def build_role_class_map(names, class_name_overrides=None):
    """
    A modell osztályneveiből (model.names: {id: név} vagy lista) szerepkör -> {osztály id} térkép.
    class_name_overrides: {"player": ["Player-L", ...]} – az adott szerepkörnél csak ezeket a neveket
    fogadjuk el az alapértelmezett álnevek helyett.
    COCO esetén: player = person, ball = sports ball, kapus és bíró osztály nincs (üres halmaz).
    """
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)

    aliases = {role: set(DEFAULT_ROLE_ALIASES[role]) for role in ROLES}
    for role, role_names in (class_name_overrides or {}).items():
        if role not in aliases:
            print(f"⚠️ Ismeretlen szerepkör a class_names beállításban: {role!r} "
                  f"(lehetséges: {', '.join(ROLES)})")
            continue
        if isinstance(role_names, str):
            role_names = [role_names]
        aliases[role] = {_normalize_name(n) for n in role_names}

    role_map = {role: set() for role in ROLES}
    for cls_id, cls_name in items:
        norm = _normalize_name(cls_name)
        for role in ROLES:
            if norm in aliases[role]:
                role_map[role].add(int(cls_id))
    return role_map


def describe_role_map(role_map, names):
    """Olvasható összefoglaló a naplóhoz, pl. 'player=[0:person], ball=[32:sports ball]'."""
    lookup = names if isinstance(names, dict) else dict(enumerate(names))
    parts = []
    for role in ROLES:
        ids = sorted(role_map.get(role, ()))
        label = ", ".join(f"{i}:{lookup.get(i, '?')}" for i in ids) if ids else "nincs"
        parts.append(f"{role}=[{label}]")
    return ", ".join(parts)


def should_detect(frame_idx, every_n):
    """Gyors mód: csak minden N. képkockán fut felismerés (a 0. képkockán mindig)."""
    return every_n <= 1 or frame_idx % every_n == 0


def reset_tracker(model):
    """
    A követő (ByteTrack) állapotának törlése egy új elemzés előtt.
    Az ultralytics a model.predictor.trackers listában tartja; az első
    track() hívás előtt a predictor még None lehet – ekkor nincs mit törölni.
    Visszatér: a törölt követők száma.
    """
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None) if predictor is not None else None
    count = 0
    for tracker in trackers or []:
        if hasattr(tracker, "reset"):
            tracker.reset()
            # Régebbi ultralytics verzióknál a reset() az azonosító-számlálót nem nullázza
            if hasattr(tracker, "reset_id"):
                tracker.reset_id()
            count += 1
    return count

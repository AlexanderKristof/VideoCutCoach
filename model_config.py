"""
Felismerő modell beállításai (súlyfájl, osztálynevek, gyors mód).

Forrás (erősorrendben):
  1. környezeti változók:
       TACTICAL_MODEL_WEIGHTS   – a .pt súlyfájl útvonala (mindkét eszközre)
       TACTICAL_DETECT_EVERY_N  – csak minden N. képkockán fut felismerés (gyors mód)
       TACTICAL_MODEL_CONFIG    – a JSON beállítófájl útvonala
  2. model_config.json (opcionális, a main.py mellett)
  3. alapértelmezés: yolov8x.pt (CUDA) / yolov8n.pt (CPU), COCO osztályok, minden képkocka

Ez a modul szándékosan nem importál torch-ot / ultralytics-et, hogy önállóan tesztelhető legyen.
"""
import json
import os

ROLES = ("player", "goalkeeper", "referee", "ball")

# Szerepkör -> ismert osztálynevek (kisbetűs, normalizált alak).
# COCO: "person", "sports ball"; Roboflow focis modellek: "player", "goalkeeper", "referee", "ball".
DEFAULT_ROLE_ALIASES = {
    "player": {"player", "person", "players"},
    "goalkeeper": {"goalkeeper", "goalie", "keeper", "goalkeepers"},
    "referee": {"referee", "referees", "ref"},
    "ball": {"ball", "sports ball", "football", "soccer ball", "soccerball"},
}

DEFAULTS = {
    "weights": None,          # ha meg van adva, mindkét eszközre ezt használjuk
    "weights_cuda": "yolov8x.pt",
    "weights_cpu": "yolov8n.pt",
    "class_names": None,      # opcionális: {"player": ["..."], "ball": ["..."], ...}
    "detect_every_n": 1,
    "ball_conf": 0.35,
}

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
        config_path = env.get("TACTICAL_MODEL_CONFIG") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), DEFAULT_CONFIG_FILENAME)

    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
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

    cfg["detect_every_n"] = _parse_every_n(cfg["detect_every_n"])
    try:
        cfg["ball_conf"] = float(cfg["ball_conf"])
    except (TypeError, ValueError):
        print(f"⚠️ Érvénytelen ball_conf érték: {cfg['ball_conf']!r} – 0.35-öt használunk.")
        cfg["ball_conf"] = DEFAULTS["ball_conf"]
    return cfg


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

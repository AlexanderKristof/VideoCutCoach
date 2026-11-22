"""
Végponttól végpontig tartó kiértékelés szintetikus meccsen (valódi Veo felvétel és GPU nélkül).

Lépések:
  1. tools/synth_match.py: rövid szintetikus meccs (alap: 40 s, 1280x720) + ground truth JSON
  2. main.cut_and_analyze_sync futtatása a tools/fake_detector.py ál-YOLO-val
     (a fastapi / pydantic / sqlalchemy / torch / ultralytics / database / models / auth modulok
     csonkolva, 'ffmpeg.exe' a PATH-on egy szimbolikus linkkel). Az ál-YOLO az ultralytics
     callback-mechanizmusát is utánozza (a követő az első track()-nél regisztrált callback),
     így a main.py nyers labda-callbackje a valódi kódúton fut.
  3. pontozás: játékos-pozíció hiba és lefedés, csapat-pontosság, labda-találati arány, a motor
     labdájának pálya-hibája, esemény-detektálás pontosság / felidézés idő-tűréssel.

Módok (E2E_MODES, alap "dyn_manual,dyn_auto,oracle"):
  dyn_manual – képkockánkénti homográfia (pitch_tracker) valószerű kézi kalibrációval: az első
               képkockán LÁTHATÓ pályapontok (felezővonal vége, kezdőkör metszéspontjai, 16-os
               sarkai ...) 1,5 px kattintási zajjal, calib_pitch_points-szal
  dyn_auto   – képkockánkénti homográfia, kalibráció nélkül (automatikus kalibráció a vonalakból)
  oracle     – a motor a valós képkockánkénti H-t kapja (referencia)
  oracle_full – mint az oracle, de a main.py szűrői (pályahatár, követés-épség, labda-priorok) is a
               valós H-t kapják a pályakövető helyett (szűrő-hangolás a H-sodródástól függetlenül)
  oracle_owntracker – mint az oracle, de az ultralytics-szerű model.track() helyett a SAJÁT háttér fut:
               Detector-interfészű ál-detektor (FakeDetector, ugyanazok a zajos detektálások) + a
               detection/tracker.py ByteTrack-szerű követő; a labda nyers jelöltjei közvetlenül
  dyn_owntracker – mint a dyn_manual, a saját háttérrel (a kamera-kompenzáció a pályakövető becsült
               képtér-mozgását kapja)
  onnx_owntracker – mint az oracle_owntracker, de VALÓDI ONNX detektorral (E2E_ONNX_PATH, pl. a YOLOX
               COCO modell; E2E_ONNX_CFG: további OnnxDetector beállítások JSON-ban) – mérés, nincs épség-kapu
  régi módok: static_off / static_on (statikus H a négy – akár képen kívüli – pályasarokból),
              oracle_off / oracle_on (labdakövető nélkül / vele)

Futtatás:  python3 tests/test_e2e_synth.py
Környezeti változók:
  E2E_DURATION (mp, alap 40), E2E_WIDTH / E2E_HEIGHT (alap 1280x720), E2E_SEED (alap 6),
  E2E_MODES (lásd fent), E2E_WORKDIR (megtartott munkakönyvtár),
  E2E_REUSE=1 (a meglévő videót újrahasznosítja a munkakönyvtárban), E2E_TOL (esemény-tűrés, mp, alap 2.5),
  E2E_DROP_UNTRACKED=0 (az ál-ByteTrack ne dobja el a meg nem erősített dobozokat – alapból eldobja,
  mint az ultralytics model.track()), E2E_MIN_H_PCT (min. játékosmagasság, alap a /cut-raw alapértéke),
  E2E_BALL_EXTRA_PASS=1 (külön labda-menet), E2E_FRAMES_DIR (annotált kimeneti képkockák JPG-be,
  dinamikus módokban), E2E_COMPARE (egy korábbi e2e_summary.json: előtte -> utána összesítés),
  E2E_DEGRADE (képromlások a szintetikus videón, lásd tools/synth_match.py: pl. "rossz", "arnyek",
  "nap", "esti", "tomorites=42", "arnyek=0.5,nap=0.8,zaj=6,elmosodas=1" – a ground truth nem változik),
  E2E_AUTO_TEAMS=1 (nincs kézi csapat-HSV: az automatikus csapatba sorolás fut, mint a /cut-raw alapból)
  E2E_BALL_RES=640 (az ál-detektor felbontás-függő labda-kihagyása: a teljes képet ekkora bemenetre
  kicsinyíti, a kis labdát nem látja – mint a valódi COCO modell 1080p-n), E2E_BALL_CROP=1 / 0 (natív
  felbontású labda-kivágás be / ki; alap: a model_config szerint, CPU-n KI)

A teszt csak laza épség-küszöböket ellenőriz (mérőeszköz), a teljes táblázatot kiírja.
Ha valamelyik előfeltétel hiányzik (cv2, numpy, ffmpeg), a teszt kihagyva (0-s kilépési kóddal) ér véget.
Az eredmények nem 'szépítettek': a cél a mérés, nem a zöld teszt.
"""
import os
import sys
import json
import math
import time
import types
import shutil
import tempfile
import importlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _skip(msg):
    print(f"KIHAGYVA: {msg}")
    sys.exit(0)


try:
    import numpy as np
    import cv2
except ImportError as ex:  # pragma: no cover
    _skip(f"hiányzó csomag: {ex}")

FFMPEG = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
if not FFMPEG:
    _skip("nincs ffmpeg a PATH-on")

import synth_match  # noqa: E402
from fake_detector import FakeYOLO, FakeDetector  # noqa: E402

EVENT_TYPES = ("shot", "goal", "gk_save", "gk_conceded", "gk_goal_kick", "gk_long_ball", "gk_short_pass")
EVENT_LABELS = {"shot": "lövés", "goal": "gól", "gk_save": "védés", "gk_conceded": "kapott gól",
                "gk_goal_kick": "kirúgás", "gk_long_ball": "kapus ívelés", "gk_short_pass": "kapus rövid passz"}
# Mód -> (felirat, kalibráció, orákulum H, labdakövető, dinamikus homográfia)
#   kalibráció: "landmarks" = az első képkockán LÁTHATÓ pályapontok (felezővonal vége, kezdőkör
#   metszéspontjai, 16-os sarkai ...) kattintási zajjal + calib_pitch_points; "corners" = a régi
#   négy pályasarok (a képen kívül is lehetnek – a valóságban nem kattintható); "none" = automatikus
MODES = {
    "dyn_manual": ("dinamikus H, kézi kalibráció (látható pontok)", "landmarks", False, True, True),
    "dyn_auto": ("dinamikus H, automatikus kalibráció", "none", False, True, True),
    "oracle": ("valós H (orákulum) – referencia", "landmarks", True, True, True),
    "static_off": ("statikus H (sarkok), követő nélkül", "corners", False, False, False),
    "static_on": ("statikus H (sarkok) + labdakövető", "corners", False, True, False),
    "oracle_off": ("valós H (orákulum), követő nélkül", "landmarks", True, False, True),
    "oracle_on": ("valós H (orákulum) + labdakövető", "landmarks", True, True, True),
}
MODES["oracle_full"] = ("valós H a motornak ÉS a main.py szűrőinek", "landmarks", True, True, True)
MODES["oracle_owntracker"] = ("valós H + saját detektor-háttér és követő", "landmarks", True, True, True)
MODES["dyn_owntracker"] = ("dinamikus H, kézi kalibr. + saját követő", "landmarks", False, True, True)
MODES["onnx_owntracker"] = ("valós H + VALÓDI ONNX detektor + saját követő", "landmarks", True, True, True)
# a valódi detektoros mód mérőeszköz: az épség-ellenőrzés kihagyja (pl. a COCO modell a kis labdát alig látja)
MEASURE_ONLY_MODES = {"onnx_owntracker"}
# ezekben a módokban a main.py az OwnTrackerBackend-et (FakeDetector + saját ByteTracker) használja
OWN_TRACKER_MODES = {"oracle_owntracker", "dyn_owntracker"}
# ezekben a módokban a pályakövető helyett is a valós H fut (a szűrők hangolása a H-sodródástól függetlenül)
ORACLE_TRACKER_MODES = {"oracle_full"}
MODE_LABELS = {k: v[0] for k, v in MODES.items()}


class OraclePitchTracker:
    """A pitch_tracker.PitchTracker helyettesítője: a valós képkockánkénti H (és képtér-mozgás)."""

    def __init__(self, gt):
        self.H = [np.array(f["H"], np.float64).reshape(3, 3) for f in gt["frames"]]
        self.i = -1

    def update(self, frame, boxes=None):
        self.i += 1
        return self.H[min(self.i, len(self.H) - 1)], {"status": "ok", "conf": 1.0, "lost": 0, "inliers": 0,
                                                      "source": "orákulum", "refined": False}

    def frame_motion(self):
        if self.i <= 0 or self.i >= len(self.H):
            return None
        F = np.linalg.inv(self.H[self.i]) @ self.H[self.i - 1]
        return F / F[2, 2]
DEFAULT_MODES = "dyn_manual,dyn_auto,oracle"

# Kattintható pályapontok (m): (név, x, y) – a 105x68-as pályán
PITCH_LANDMARKS = [
    ("felezővonal felső vége", 52.5, 0.0), ("felezővonal alsó vége", 52.5, 68.0),
    ("kezdőkör felső metszéspont", 52.5, 34.0 - 9.15), ("kezdőkör alsó metszéspont", 52.5, 34.0 + 9.15),
    ("kezdőkör bal széle", 52.5 - 9.15, 34.0), ("kezdőkör jobb széle", 52.5 + 9.15, 34.0),
    ("bal 16-os felső sarka", 16.5, 13.84), ("bal 16-os alsó sarka", 16.5, 54.16),
    ("bal 16-os felső alapvonal-pont", 0.0, 13.84), ("bal 16-os alsó alapvonal-pont", 0.0, 54.16),
    ("jobb 16-os felső sarka", 88.5, 13.84), ("jobb 16-os alsó sarka", 88.5, 54.16),
    ("jobb 16-os felső alapvonal-pont", 105.0, 13.84), ("jobb 16-os alsó alapvonal-pont", 105.0, 54.16),
    ("bal felső szöglet", 0.0, 0.0), ("jobb felső szöglet", 105.0, 0.0),
    ("bal alsó szöglet", 0.0, 68.0), ("jobb alsó szöglet", 105.0, 68.0),
]


def landmark_calibration(gt, click_sigma_px=1.5, margin_px=12, seed=0):
    """
    Valószerű kézi kalibráció: az első képkockán látható pályapontok (legalább 4) kattintási
    zajjal (szórás 1,5 px 1280 széles képen). Visszatér: (calib_points, calib_pitch_points, nevek).
    """
    w, h = gt["width"], gt["height"]
    H0 = np.array(gt["frames"][0]["H"], np.float64).reshape(3, 3)
    G0 = np.linalg.inv(H0)
    q = H0 @ np.array([w / 2.0, h * 0.9, 1.0])
    ref = G0 @ (q / q[2])          # egy biztosan a kamera előtt lévő pálya-pont mélység-előjele
    rng = np.random.default_rng(seed)
    sig = click_sigma_px * w / 1280.0
    img, pit, names = [], [], []
    for name, x, y in PITCH_LANDMARKS:
        p = G0 @ np.array([x, y, 1.0])
        if np.sign(p[2]) != np.sign(ref[2]):
            continue
        u, v = p[0] / p[2], p[1] / p[2]
        if not (margin_px <= u <= w - margin_px and margin_px <= v <= h - margin_px):
            continue
        img.append((u + rng.normal(0, sig), v + rng.normal(0, sig)))
        pit.append((x, y))
        names.append(name)
    if len(img) < 4:
        return None, None, names
    calib = ";".join(f"{u / w:.6f},{v / h:.6f}" for u, v in img)
    pitch = ";".join(f"{x:g},{y:g}" for x, y in pit)
    return calib, pitch, names


# ====================================================================== csonkok
def install_stubs(fake_model):
    """A main.py importjához szükséges, itt nem telepített modulok csonkolása."""

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _App:
        def __init__(self, *a, **k):
            pass

        def mount(self, *a, **k):
            pass

        def _deco(self, *a, **k):
            return lambda f: f

        post = get = put = delete = on_event = middleware = _deco

    class _HTTPException(Exception):
        def __init__(self, status_code=500, detail=None, **k):
            super().__init__(detail)
            self.status_code, self.detail = status_code, detail

    status = types.SimpleNamespace(HTTP_400_BAD_REQUEST=400, HTTP_401_UNAUTHORIZED=401, HTTP_404_NOT_FOUND=404,
                                   HTTP_403_FORBIDDEN=403, HTTP_500_INTERNAL_SERVER_ERROR=500)
    mod("fastapi", FastAPI=_App, Form=lambda default=None, *a, **k: default, Depends=lambda *a, **k: None, Request=object,
        HTTPException=_HTTPException, status=status, File=lambda *a, **k: None, UploadFile=object)
    mod("fastapi.staticfiles", StaticFiles=lambda *a, **k: None)
    mod("fastapi.security", OAuth2PasswordRequestForm=object, OAuth2PasswordBearer=lambda *a, **k: None)
    mod("fastapi.responses", HTMLResponse=lambda *a, **k: None, FileResponse=lambda *a, **k: None,
        JSONResponse=lambda *a, **k: None)

    class _BaseModel:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    mod("pydantic", BaseModel=_BaseModel, EmailStr=str)
    mod("sqlalchemy")
    mod("sqlalchemy.orm", Session=object)
    mod("database", get_db=lambda: None, engine=None)

    class _Meta:
        @staticmethod
        def create_all(bind=None):
            pass

    mod("models", Base=types.SimpleNamespace(metadata=_Meta()), User=object, Match=object)
    mod("auth", hash_password=lambda p: p, verify_password=lambda a, b: a == b,
        create_access_token=lambda d: "token", get_current_user=lambda: None)
    mod("torch", cuda=types.SimpleNamespace(is_available=lambda: False))
    mod("ultralytics", YOLO=lambda *a, **k: fake_model)


def import_main(workdir, fake_model):
    """A main.py importja a munkakönyvtárban (clips/ ide kerül), 'ffmpeg.exe' symlinkkel."""
    bindir = os.path.join(workdir, "bin")
    os.makedirs(bindir, exist_ok=True)
    link = os.path.join(bindir, "ffmpeg.exe")
    if not os.path.exists(link):
        os.symlink(FFMPEG, link)
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.environ.pop("TACTICAL_DETECT_EVERY_N", None)
    install_stubs(fake_model)
    os.chdir(workdir)
    if "main" in sys.modules:
        del sys.modules["main"]
    return importlib.import_module("main")


# ====================================================================== rögzítő motor
def make_recording_engine(main_mod, gt, oracle):
    base = main_mod.TacticalEngine
    gt_H = [np.array(f["H"], np.float64).reshape(3, 3) for f in gt["frames"]]

    class RecordingEngine(base):
        records = []
        static_H = None

        def update_tracks(self, frame_idx, timestamp, detections, ball_pos, H):
            if RecordingEngine.static_H is None and H is not None:
                RecordingEngine.static_H = np.array(H)
            H_use = gt_H[frame_idx] if (oracle and frame_idx < len(gt_H)) else H
            players, real_ball = super().update_tracks(frame_idx, timestamp, detections, ball_pos, H_use)
            RecordingEngine.records.append({
                "frame": frame_idx, "ball_px": ball_pos, "real_ball": real_ball,
                "players": [{"bbox": p["bbox"], "pitch": p.get("pitch_coords"), "team": p["team_id"],
                             "keeper": p["is_keeper"], "tid": p["track_id"]} for p in players],
            })
            return players, real_ball

    RecordingEngine.records = []
    RecordingEngine.static_H = None
    return RecordingEngine


# ====================================================================== pontozás
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _to_pitch(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def _stats(vals):
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    a = np.asarray(vals, np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}


def score_ball(records, gt):
    frames = gt["frames"]
    tol_px = 15.0 * gt["width"] / 1280.0
    by_frame = {r["frame"]: r for r in records}
    vis = hit = out_n = wrong = 0
    px_err, pitch_err_true_h, engine_err = [], [], []
    for i, f in enumerate(frames):
        b = f["ball"]
        r = by_frame.get(i)
        out = r["ball_px"] if r else None
        if b.get("vis"):
            vis += 1
        if out is not None:
            out_n += 1
            if b.get("vis"):
                d = math.hypot(out[0] - b["uv"][0], out[1] - b["uv"][1])
                if d <= tol_px:
                    hit += 1
                    px_err.append(d)
                else:
                    wrong += 1
            else:
                wrong += 1
            if b.get("vis") and b["xyz"][2] < 0.5:
                H = np.array(f["H"]).reshape(3, 3)
                x, y = _to_pitch(H, out[0], out[1])
                pitch_err_true_h.append(math.hypot(x - b["xyz"][0], y - b["xyz"][1]))
        if r and r["real_ball"] is not None and b["xyz"][2] < 0.5:
            engine_err.append(math.hypot(r["real_ball"][0] - b["xyz"][0], r["real_ball"][1] - b["xyz"][1]))
    return {
        "gt_visible_frames": vis, "output_frames": out_n,
        "detection_rate": hit / max(1, vis), "wrong_rate": wrong / max(1, out_n),
        "px_err": _stats(px_err), "pitch_err_with_true_H": _stats(pitch_err_true_h),
        "engine_ball_err": _stats(engine_err), "tol_px": tol_px,
    }


def score_players(records, gt, min_h_px=50.0):
    frames = gt["frames"]
    roles = {p["id"]: p["role"] for p in gt["people"]}
    teams = {p["id"]: p["team"] for p in gt["people"]}
    errs, tooth_small, gt_vis, matched = [], 0, 0, 0
    team_ok = team_n = specs_out = 0
    seg_errs = [[] for _ in range(10)]   # a klip tizedeiként (hol romlik el a pályakövetés)
    for r in records:
        f = frames[r["frame"]]
        objs = f["objs"]
        cand = [o for o in objs if roles.get(o[0]) in ("player", "goalkeeper")]
        for o in cand:
            gt_vis += 1
            if o[8] - o[6] < min_h_px:
                tooth_small += 1
        used = set()
        for p in r["players"]:
            best, bi = 0.0, None
            for o in objs:
                if o[0] in used:
                    continue
                v = _iou(p["bbox"], o[5:9])
                if v > best:
                    best, bi = v, o
            if bi is None or best < 0.3:
                continue
            used.add(bi[0])
            role = roles.get(bi[0])
            if role not in ("player", "goalkeeper"):
                specs_out += 1
                continue
            matched += 1
            if p["pitch"] is not None:
                errs.append(math.hypot(p["pitch"][0] - bi[1], p["pitch"][1] - bi[2]))
                seg_errs[min(9, int(10 * r["frame"] / max(1, len(frames))))].append(errs[-1])
            team_n += 1
            team_ok += int(p["team"] == teams.get(bi[0]))
    return {"pitch_err": _stats(errs), "gt_visible_player_boxes": gt_vis,
            "pitch_err_by_tenth": [_stats(e)["median"] for e in seg_errs],
            "gt_below_min_h_ratio": tooth_small / max(1, gt_vis), "min_h_px": min_h_px,
            "coverage": matched / max(1, gt_vis), "team_accuracy": team_ok / max(1, team_n),
            "non_player_outputs": specs_out}


def score_identity(records, gt, iou_thr=0.5):
    """
    Azonosító-stabilitás (MOT-szerű): képkockánként a kiadott játékos-dobozok párosítása a valós
    játékos / kapus dobozokkal (IoU >= iou_thr, mohó), és valós objektumonként az azonosító-sorozat.
      id_switches – hányszor változik egy valós játékos kiadott azonosítója (a kihagyásokat átugorva)
      ids_per_player – átlagosan hány különböző azonosítót kap egy valós játékos (1.0 = tökéletes)
      purity – a kiadott azonosítók hány %-a tartozik végig egyetlen valós játékoshoz
    """
    frames = gt["frames"]
    roles = {p["id"]: p["role"] for p in gt["people"]}
    seq, owner = {}, {}
    for r in records:
        objs = [o for o in frames[r["frame"]]["objs"] if roles.get(o[0]) in ("player", "goalkeeper")]
        pairs = []
        for pi, p in enumerate(r["players"]):
            if p["tid"] is None:
                continue
            for o in objs:
                v = _iou(p["bbox"], o[5:9])
                if v >= iou_thr:
                    pairs.append((v, pi, o[0]))
        pairs.sort(reverse=True)
        used_p, used_o = set(), set()
        for v, pi, oid in pairs:
            if pi in used_p or oid in used_o:
                continue
            used_p.add(pi)
            used_o.add(oid)
            tid = int(r["players"][pi]["tid"])
            seq.setdefault(oid, []).append(tid)
            owner.setdefault(tid, set()).add(oid)
    switches = 0
    for ids in seq.values():
        switches += sum(1 for a, b in zip(ids, ids[1:]) if a != b)
    n_ids = [len(set(ids)) for ids in seq.values()]
    pure = sum(1 for o in owner.values() if len(o) == 1)
    return {"id_switches": switches, "gt_players_seen": len(seq),
            "ids_per_player": float(np.mean(n_ids)) if n_ids else None,
            "purity": pure / max(1, len(owner)), "output_ids": len(owner),
            "matched_boxes": int(sum(len(v) for v in seq.values()))}


def ideal_events(gt):
    """Felső korlát: az events.py detektor a VALÓS pálya-pozíciókkal (labda + 22 játékos) etetve."""
    from events import EventDetector, EventConfig
    period = (gt.get("periods") or [{}])[0]
    cfg = EventConfig()
    cfg.team1_attacks_right = bool(period.get("team1_attacks_right", True))
    det = EventDetector(cfg)
    people = gt["people"][:22]
    for f in gt["frames"]:
        players = [{"track_id": p["id"], "team_id": p["team"], "is_keeper": p["role"] == "goalkeeper",
                    "pitch_coords": tuple(f["xy"][p["id"]])} for p in people]
        b = f["ball"]
        ball = tuple(b["xyz"][:2]) if b.get("on", True) else None
        det.update(f["t"], ball, players)
    det.finalize(gt["frames"][-1]["t"])
    return det.events


def score_events(detected, gt_events, tol):
    res = {}
    for typ in EVENT_TYPES:
        g = sorted(e["time"] for e in gt_events if e["type"] == typ)
        d = sorted(e["time"] for e in detected if e["type"] == typ)
        used = set()
        tp = 0
        for t in g:
            best, bj = None, None
            for j, td in enumerate(d):
                if j in used or abs(td - t) > tol:
                    continue
                if best is None or abs(td - t) < best:
                    best, bj = abs(td - t), j
            if bj is not None:
                used.add(bj)
                tp += 1
        res[typ] = {"gt": len(g), "det": len(d), "tp": tp}
    tot_gt = sum(v["gt"] for v in res.values())
    tot_det = sum(v["det"] for v in res.values())
    tot_tp = sum(v["tp"] for v in res.values())
    other = [e["type"] for e in detected if e["type"] not in EVENT_TYPES and e["type"] != "gk_positioning_error"]
    res["_total"] = {"gt": tot_gt, "det": tot_det, "tp": tot_tp,
                     "precision": tot_tp / tot_det if tot_det else None,
                     "recall": tot_tp / tot_gt if tot_gt else None,
                     "other": len(other), "other_types": sorted(set(other)),
                     "positioning_errors": sum(e["type"] == "gk_positioning_error" for e in detected),
                     "team_ok": _team_agreement(detected, gt_events, tol)}
    return res


def _team_agreement(detected, gt_events, tol):
    ok = n = 0
    for g in gt_events:
        if g["type"] not in EVENT_TYPES:
            continue
        cands = [d for d in detected if d["type"] == g["type"] and abs(d["time"] - g["time"]) <= tol]
        if cands:
            n += 1
            ok += int(min(cands, key=lambda d: abs(d["time"] - g["time"])).get("team") == g.get("team"))
    return f"{ok}/{n}"


# ====================================================================== futtatás
def click_range(bgr):
    """A kliens színkattintásának (/analyze-click-ai) megfelelő HSV tartomány: H ±12, S/V ±50."""
    h, s_, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0].tolist()
    lo = [max(0, h - 12), max(20, s_ - 50), max(20, v - 50)]
    up = [min(179, h + 12), min(255, s_ + 50), min(255, v + 50)]
    return lo, up


def _default_min_h_pct():
    """A /cut-raw alapértéke (pipeline.ANALYSIS_DEFAULTS) – E2E_MIN_H_PCT felülírja."""
    if os.environ.get("E2E_MIN_H_PCT"):
        return float(os.environ["E2E_MIN_H_PCT"])
    try:
        import pipeline
        return float(pipeline.ANALYSIS_DEFAULTS["min_player_height_pct"])
    except Exception:  # pragma: no cover
        return 0.022


def parse_degrade(spec):
    """E2E_DEGRADE -> synth_match rontás-szótár. "rossz" = az előre beállított csomag; "név" vagy "név=érték"."""
    if not spec:
        return None
    kw = {"rossz": False}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.partition("=")
        name = name.strip().lower()
        if name == "rossz":
            kw["rossz"] = True
        elif name == "esti":
            kw["esti"] = True
        elif name in ("arnyek", "nap", "tomorites", "zaj", "elmosodas"):
            default = {"arnyek": 0.5, "nap": 1.0, "tomorites": 42, "zaj": 6.0, "elmosodas": 1.0}[name]
            kw[name] = (int(val) if name == "tomorites" else float(val)) if val else default
        else:
            raise SystemExit(f"Ismeretlen rontás: {name}")
    return synth_match.degrade_options(**kw)


def base_config(gt, tracker_on, dynamic=True):
    period = (gt.get("periods") or [{}])[0]
    kits = synth_match.KITS
    t1, t2 = click_range(kits["team1"]["shirt"]), click_range(kits["team2"]["shirt"])
    k1, k2 = click_range(kits["keeper1"]["shirt"]), click_range(kits["keeper2"]["shirt"])
    cfg = {
        "team1_attacks_right": bool(period.get("team1_attacks_right", True)),
        "review_export": False,
        "roi_top_pct": 0.28, "roi_bottom_pct": 0.10, "min_player_height_pct": _default_min_h_pct(),
        "use_static_roi": False, "grass_min_ratio": 0.3,
        "stationary_window_sec": 4.0, "stationary_max_move": 0.5,
        "grass_lower": [30, 40, 40], "grass_upper": [85, 255, 255],
        "ref_lower": [0, 0, 0], "ref_upper": [179, 255, 255],
        # a kliensben kattintással megadott mezszínek megfelelői (piros / kék / sárga / lila)
        "team1_lower": t1[0], "team1_upper": t1[1],
        "team2_lower": t2[0], "team2_upper": t2[1],
        "keeper1_lower": k1[0], "keeper1_upper": k1[1],
        "keeper2_lower": k2[0], "keeper2_upper": k2[1],
        "ball_tracker": tracker_on,
        "dynamic_homography": dynamic, "draw_pitch_overlay": True,
        "ball_extra_pass": os.environ.get("E2E_BALL_EXTRA_PASS") == "1",
    }
    if os.environ.get("E2E_BALL_CROP") in ("0", "1"):
        cfg["ball_crop_pass"] = os.environ["E2E_BALL_CROP"] == "1"
    if os.environ.get("E2E_AUTO_TEAMS") == "1":
        # nincs kézi csapat-/kapus-HSV: automatikus csapatba sorolás (a /cut-raw alapértéke)
        for k in ("team1", "team2", "keeper1", "keeper2"):
            cfg[f"{k}_lower"], cfg[f"{k}_upper"] = [0, 0, 0], [179, 255, 255]
        cfg["auto_teams"] = True
    return cfg


def run_mode(main_mod, fake, gt, video, mode):
    _label, calib_kind, oracle, tracker_on, dynamic = MODES[mode]
    Engine = make_recording_engine(main_mod, gt, oracle)
    original = main_mod.TacticalEngine
    main_mod.TacticalEngine = Engine
    fake.load_ground_truth(gt)
    config = base_config(gt, tracker_on, dynamic)
    calib, calib_names = "", []
    if calib_kind == "corners":
        calib = gt["calibration_first_frame"]["calib_points"]
    elif calib_kind == "landmarks":
        calib, pitch_pts, calib_names = landmark_calibration(gt, seed=gt["seed"])
        if calib is None:
            print(f"⚠️ Kevesebb mint 4 látható pályapont az első képkockán ({calib_names}) – automatikus kalibráció")
            calib = ""
        else:
            config["calib_pitch_points"] = pitch_pts
            print(f"Kézi kalibráció {len(calib_names)} látható ponttal: {', '.join(calib_names)}")
    original_cpt = main_mod.create_pitch_tracker
    if hasattr(main_mod, "ensure_model"):
        main_mod.ensure_model()   # lusta betöltés: az ál-YOLO és a szerepkörök előbb töltődjenek be
    original_backend = getattr(main_mod, "OWN_BACKEND", None)
    original_roles = main_mod.role_classes
    if mode == "onnx_owntracker":
        from detection.backend import create_backend
        import model_config
        onnx_path = os.environ.get("E2E_ONNX_PATH")
        if not onnx_path or not os.path.exists(onnx_path):
            raise SystemExit("onnx_owntracker: az E2E_ONNX_PATH környezeti változóban add meg az .onnx fájlt")
        mcfg = dict(model_config.DEFAULTS)
        mcfg.update({"backend": "onnx", "onnx_path": onnx_path,
                     "onnx": json.loads(os.environ.get("E2E_ONNX_CFG", "{}"))})
        main_mod.OWN_BACKEND, main_mod.role_classes = create_backend(mcfg, "cpu")
        print(f"Valódi detektor: {main_mod.OWN_BACKEND.describe()}, osztályok: "
              f"{model_config.describe_role_map(main_mod.role_classes, main_mod.OWN_BACKEND.names)}")
    elif mode in OWN_TRACKER_MODES:
        from detection.backend import OwnTrackerBackend
        main_mod.OWN_BACKEND = OwnTrackerBackend(FakeDetector(fake), role_classes=main_mod.role_classes)
    if mode in ORACLE_TRACKER_MODES:
        # a main.py szűrői (pályahatár, követés-épség, labda-priorok) is a valós H-t kapják
        main_mod.create_pitch_tracker = lambda *a, **k: (OraclePitchTracker(gt), "orákulum (valós H)")
    t0 = time.time()
    try:
        _url, out_path, events_path = main_mod.cut_and_analyze_sync(
            video, 0.0, gt["duration"], True, "team", calib, config)
    finally:
        main_mod.TacticalEngine = original
        main_mod.create_pitch_tracker = original_cpt
        own_stats = dict(main_mod.OWN_BACKEND.stats) if main_mod.OWN_BACKEND is not None else None
        main_mod.OWN_BACKEND = original_backend
        main_mod.role_classes = original_roles
    elapsed = time.time() - t0
    with open(events_path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("timing"):
        from stage_timer import format_timing
        print("Időmérés: " + format_timing(payload["timing"]))
    if payload.get("ball_crop"):
        print(f"Labda-kivágás: {payload['ball_crop']['stats']}")
    return {"records": Engine.records, "static_H": Engine.static_H, "events": payload["events"],
            "timing": payload.get("timing"), "ball_crop": payload.get("ball_crop"),
            "video_quality": payload.get("video_quality"), "grass_model": payload.get("grass_model"),
            "elapsed": elapsed, "fake_stats": dict(fake.stats), "out_video": out_path,
            "pitch_tracking": payload.get("pitch_tracking"), "calib_names": calib_names,
            "min_h_pct": config["min_player_height_pct"], "own_backend_stats": own_stats}


def save_annotated_frames(video_path, out_dir, mode, frame_ids):
    """A kimeneti (annotált, pályavonal-réteges) videó néhány képkockája JPG-be."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    saved = []
    idx = 0
    want = set(frame_ids)
    while want:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            path = os.path.join(out_dir, f"e2e_{mode}_kepkocka_{idx:05d}.jpg")
            cv2.imwrite(path, fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(path)
            want.discard(idx)
        idx += 1
    cap.release()
    return saved


def static_h_error(static_H, gt):
    """A statikus (első képkockás) homográfia hibája a kép közepén és alsó harmadában, képkockánként."""
    w, h = gt["width"], gt["height"]
    pts = [(w * 0.5, h * 0.5), (w * 0.25, h * 0.75), (w * 0.75, h * 0.75)]
    errs = []
    for f in gt["frames"]:
        Ht = np.array(f["H"]).reshape(3, 3)
        for u, v in pts:
            a = _to_pitch(Ht, u, v)
            b = _to_pitch(static_H, u, v)
            errs.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    first = []
    Ht0 = np.array(gt["frames"][0]["H"]).reshape(3, 3)
    for u, v in pts:
        a = _to_pitch(Ht0, u, v)
        b = _to_pitch(static_H, u, v)
        first.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    return _stats(errs), max(first)


def _f(v, fmt="{:.2f}", none="–"):
    return none if v is None else fmt.format(v)


def print_report(results, gt, tol, h_err, ideal=None):
    print("\n" + "=" * 96)
    print(f"SZINTETIKUS E2E KIÉRTÉKELÉS – {gt['duration']:.0f} s, {gt['width']}x{gt['height']} @ {gt['fps']} fps, "
          f"seed {gt['seed']}")
    print("=" * 96)
    if h_err is not None:
        st, first = h_err
        print(f"Statikus homográfia (első képkocka kalibráció) – 0. képkockán max. hiba: {first:.3f} m; "
              f"a teljes klipen a képpontok vetítési hibája: medián {st['median']:.1f} m, átlag {st['mean']:.1f} m, "
              f"p90 {st['p90']:.1f} m  (a forgó kamera miatt)")
    print("\nLABDA (a TacticalEngine-nek átadott labdapont képkockánként)")
    print(f"{'mód':44s} {'találat%':>9s} {'téves%':>7s} {'px hiba (med)':>14s} {'pálya hiba valós H (med)':>25s} "
          f"{'motor labda hiba (med)':>23s}")
    for mode, r in results.items():
        b = r["ball"]
        print(f"{MODE_LABELS[mode]:44s} {100 * b['detection_rate']:9.1f} {100 * b['wrong_rate']:7.1f} "
              f"{_f(b['px_err']['median'], '{:.1f}'):>14s} {_f(b['pitch_err_with_true_H']['median']) + ' m':>25s} "
              f"{_f(b['engine_ball_err']['median']) + ' m':>23s}")
    print("  (találat: a valós labdától legfeljebb "
          f"{next(iter(results.values()))['ball']['tol_px']:.0f} px; a látható labdás képkockák arányában)")

    print("\nJÁTÉKOSOK (a motor által elfogadott játékosok pálya-pozíciója vs. valós pozíció)")
    print(f"{'mód':44s} {'medián':>8s} {'átlag':>8s} {'p90':>8s} {'lefedés%':>9s} {'csapat ok%':>11s} {'nem játékos':>12s}")
    for mode, r in results.items():
        p = r["players"]
        e = p["pitch_err"]
        print(f"{MODE_LABELS[mode]:44s} {_f(e['median']) + ' m':>8s} {_f(e['mean']) + ' m':>8s} {_f(e['p90']) + ' m':>8s} "
              f"{100 * p['coverage']:9.1f} {100 * p['team_accuracy']:11.1f} {p['non_player_outputs']:12d}")
    for mode, r in results.items():
        segs = r["players"].get("pitch_err_by_tenth") or []
        print(f"  {mode:12s} medián hiba a klip tizedeiben: " + " ".join(_f(v, "{:.1f}") for v in segs))
    p0 = next(iter(results.values()))["players"]
    print(f"  (a látható valós játékos-dobozok {100 * p0['gt_below_min_h_ratio']:.0f}%-a alacsonyabb a main.py "
          f"magasságszűrőjénél ({p0['min_h_px']:.1f} px) – ezeket eldobja)")

    print("\nAZONOSÍTÓK (a kiadott játékos-azonosítók stabilitása, IoU >= 0.5 a valós dobozzal)")
    print(f"{'mód':44s} {'ID-váltás':>10s} {'azon./játékos':>14s} {'tisztaság%':>11s} {'kiadott azon.':>14s} {'párosított doboz':>17s}")
    for mode, r in results.items():
        i = r["identity"]
        print(f"{MODE_LABELS[mode]:44s} {i['id_switches']:10d} {_f(i['ids_per_player']):>14s} {100 * i['purity']:11.1f} "
              f"{i['output_ids']:14d} {i['matched_boxes']:17d}")
    for mode, r in results.items():
        if r.get("own_backend_stats"):
            print(f"  Saját háttér [{mode}]: {r['own_backend_stats']}")

    print(f"\nESEMÉNYEK (±{tol:.1f} s tűrés)")
    header = f"{'esemény':18s} {'valós':>6s} " + " ".join(f"{m:>16s}" for m in results)
    print(header)
    for typ in EVENT_TYPES:
        row = f"{EVENT_LABELS[typ]:18s} {next(iter(results.values()))['events'][typ]['gt']:6d} "
        row += " ".join(f"{(str(r['events'][typ]['tp']) + '/' + str(r['events'][typ]['det']) + ' tal/det'):>16s}"
                        for r in results.values())
        print(row)
    row = f"{'pontosság':18s} {'':6s} " + " ".join(f"{_f(r['events']['_total']['precision']):>16s}" for r in results.values())
    print(row)
    row = f"{'felidézés':18s} {'':6s} " + " ".join(f"{_f(r['events']['_total']['recall']):>16s}" for r in results.values())
    print(row)
    row = f"{'csapat egyezik':18s} {'':6s} " + " ".join(f"{r['events']['_total']['team_ok']:>16s}" for r in results.values())
    print(row)
    row = f"{'egyéb (nincs GT)':18s} {'':6s} " + " ".join(f"{r['events']['_total']['other']:>16d}" for r in results.values())
    print(row)
    row = f"{'helyezkedési hiba':18s} {'':6s} " + " ".join(f"{r['events']['_total']['positioning_errors']:>16d}"
                                                       for r in results.values())
    print(row)
    # gól: megerősített (középkezdés / a vonal mögött maradt) vs. 'Valószínű GÓL' (megerősítetlen)
    gc = []
    for r in results.values():
        goals = [e for e in r.get("detected_events", []) if e["type"] == "goal"]
        conf = sum(1 for e in goals if e.get("confirmed", True))
        gc.append(f"{conf} megerősített / {len(goals) - conf} valószínű")
    print(f"{'gól megerősítés':18s} {'':6s} " + " ".join(f"{c:>16s}" for c in gc))
    if ideal is not None:
        tot = ideal["_total"]
        per = ", ".join(f"{EVENT_LABELS[t]} {ideal[t]['tp']}/{ideal[t]['gt']}" for t in EVENT_TYPES if ideal[t]["gt"])
        print(f"Felső korlát – events.py a valós pozíciókkal etetve: pontosság {_f(tot['precision'])}, "
              f"felidézés {_f(tot['recall'])} ({per}; egyéb {tot['other']} {tot['other_types']})")
    print("\nFutásidő: " + ", ".join(f"{m}: {r['elapsed']:.0f} s" for m, r in results.items()))
    fs = next(iter(results.values()))["fake_stats"]
    print(f"Ál-detektor: labda látható {fs['ball_gt_visible']} képkockán, detektálva {fs['ball_detected']}, "
          f"téves labda-jelölt {fs['ball_fp']}, a követő által eldobott labda-doboz {fs['ball_dropped_by_tracker']}, "
          f"néző 'player'-ként {fs['spectator_fp']}")
    for m, r in results.items():
        if r.get("pitch_tracking"):
            print(f"Pályakövetés [{m}]: {r['pitch_tracking'].get('setup')} – {r['pitch_tracking'].get('frames')}")
    print("=" * 96)


SUMMARY_COLS = [("játékos hiba med (m)", "player_err_med", "{:.2f}"), ("lefedés %", "coverage", "{:.1f}"),
                ("labda találat %", "ball_hit", "{:.1f}"), ("labda pálya hiba med (m)", "ball_pitch_err_med", "{:.2f}"),
                ("csapat ok %", "team_acc", "{:.1f}"), ("ID-váltás", "id_switches", "{:d}"),
                ("esemény pontosság", "ev_precision", "{:.2f}"),
                ("esemény felidézés", "ev_recall", "{:.2f}")]


def summary_row(r):
    """Módonként egy sor a fő mutatókkal (a labda pálya hibája: a motor labdapontja vs. valós, földön)."""
    return {"player_err_med": r["players"]["pitch_err"]["median"], "coverage": 100 * r["players"]["coverage"],
            "ball_hit": 100 * r["ball"]["detection_rate"], "ball_pitch_err_med": r["ball"]["engine_ball_err"]["median"],
            "team_acc": 100 * r["players"]["team_accuracy"], "id_switches": r["identity"]["id_switches"],
            "ev_precision": r["events"]["_total"]["precision"], "ev_recall": r["events"]["_total"]["recall"]}


def print_summary(results, before=None):
    """Összesítő táblázat; before: egy korábbi e2e_summary.json 'summary' része (E2E_COMPARE) – előtte -> utána."""
    print("\nÖSSZESÍTÉS" + (" (előtte -> utána)" if before else ""))
    print(f"{'mód':12s} " + " ".join(f"{c[0]:>24s}" for c in SUMMARY_COLS))
    for m, r in results.items():
        row = summary_row(r)
        cells = []
        for _name, key, fmt in SUMMARY_COLS:
            now = _f(row[key], fmt)
            if before and m in before:
                now = f"{_f(before[m].get(key), fmt)} -> {now}"
            cells.append(f"{now:>24s}")
        print(f"{m:12s} " + " ".join(cells))


def oracle_mode(mode):
    return MODES[mode][2]


def sanity_check(results):
    """
    Csak laza épség-ellenőrzés (ez mérőeszköz, nem minőségi kapu): a futás eredményt ad,
    a játékosok egy része átjut, a labda néha megvan, a csapat nem véletlen.
    """
    for m, r in results.items():
        if m in MEASURE_ONLY_MODES:
            continue
        p, b = r["players"], r["ball"]
        assert p["coverage"] > 0.2, f"{m}: túl kicsi játékos-lefedés ({p['coverage']:.2f})"
        assert b["detection_rate"] > 0.15, f"{m}: túl kicsi labda-találati arány ({b['detection_rate']:.2f})"
        assert p["team_accuracy"] > 0.6, f"{m}: csapatbesorolás közel véletlen ({p['team_accuracy']:.2f})"
        if oracle_mode(m) or MODES[m][1] == "landmarks":
            med = p["pitch_err"]["median"]
            assert med is not None and med < 5.0, f"{m}: túl nagy játékos-pozíció hiba ({med})"
    print("Épség-ellenőrzés: OK")


def main():
    duration = float(os.environ.get("E2E_DURATION", "40"))
    width = int(os.environ.get("E2E_WIDTH", "1280"))
    height = int(os.environ.get("E2E_HEIGHT", "720"))
    seed = int(os.environ.get("E2E_SEED", "6"))
    tol = float(os.environ.get("E2E_TOL", "2.5"))
    modes = [m.strip() for m in os.environ.get("E2E_MODES", DEFAULT_MODES).split(",") if m.strip()]
    for m in modes:
        if m not in MODE_LABELS:
            raise SystemExit(f"Ismeretlen mód: {m} (lehet: {', '.join(MODE_LABELS)})")
    keep = os.environ.get("E2E_WORKDIR")
    workdir = os.path.abspath(keep) if keep else tempfile.mkdtemp(prefix="e2e_synth_")
    os.makedirs(workdir, exist_ok=True)
    degrade = parse_degrade(os.environ.get("E2E_DEGRADE", ""))
    tag = "" if not degrade else "_" + "_".join(f"{k}{v if v is not True else ''}" for k, v in sorted(degrade.items()))
    video = os.path.join(workdir, f"synth_{seed}_{int(duration)}s_{width}x{height}{tag}.mp4")
    gt_path = os.path.splitext(video)[0] + "_gt.json"
    cwd = os.getcwd()
    try:
        if not (os.environ.get("E2E_REUSE") and os.path.exists(video) and os.path.exists(gt_path)):
            print(f"Szintetikus meccs generálása: {video}")
            synth_match.generate(video, gt_path, duration=duration, width=width, height=height, fps=25, seed=seed,
                                 degrade=degrade)
        if degrade:
            print(f"Képromlások: {degrade}")
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        if not gt["calibration_first_frame"].get("all_in_front_of_camera", True):
            print("⚠️ Az első képkockán nem minden pályasarok van a kamera előtt – a kalibráció torz lehet")

        frames_dir = os.environ.get("E2E_FRAMES_DIR")
        prof = {"infer_imgsz": int(os.environ["E2E_BALL_RES"])} if os.environ.get("E2E_BALL_RES") else None
        fake = FakeYOLO(seed=seed + 100, drop_untracked=os.environ.get("E2E_DROP_UNTRACKED", "1") != "0", profile=prof)
        main_mod = import_main(workdir, fake)
        results = {}
        h_err = None
        for mode in modes:
            print(f"\n--- Mód: {MODE_LABELS[mode]} ---")
            r = run_mode(main_mod, fake, gt, video, mode)
            if r["static_H"] is not None and h_err is None:
                h_err = static_h_error(r["static_H"], gt)
            results[mode] = {
                "ball": score_ball(r["records"], gt),
                "players": score_players(r["records"], gt, r["min_h_pct"] * gt["height"]),
                "events": score_events(r["events"], gt["events"], tol),
                "identity": score_identity(r["records"], gt), "own_backend_stats": r["own_backend_stats"],
                "elapsed": r["elapsed"], "fake_stats": r["fake_stats"],
                "timing": r.get("timing"), "ball_crop": r.get("ball_crop"),
                "detected_events": [{k: e.get(k) for k in ("type", "time", "team", "confirmed", "confirmation")} for e in r["events"]],
                "pitch_tracking": r["pitch_tracking"], "calib_names": r["calib_names"],
                "video_quality": r.get("video_quality"), "grass_model": r.get("grass_model"),
                "event_reliability": [e.get("reliability") for e in r["events"]],
            }
            if r.get("video_quality"):
                print(r["video_quality"].get("summary_text"))
            if r.get("grass_model"):
                print(r["grass_model"])
            if frames_dir and r["out_video"] and not oracle_mode(mode) and MODES[mode][4]:
                n = gt["n_frames"]
                for path in save_annotated_frames(r["out_video"], frames_dir, mode, [int(n * 0.35), int(n * 0.7)]):
                    print(f"Annotált képkocka: {path}")
        ideal = score_events(ideal_events(gt), gt["events"], tol)
        print_report(results, gt, tol, h_err, ideal)
        before = None
        if os.environ.get("E2E_COMPARE") and os.path.exists(os.environ["E2E_COMPARE"]):
            with open(os.environ["E2E_COMPARE"], encoding="utf-8") as f:
                before = json.load(f).get("summary")
        print_summary(results, before)
        summary_path = os.path.join(workdir, "e2e_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"gt_events": gt["events"], "ideal_events": ideal, "results": results,
                       "summary": {m: summary_row(r) for m, r in results.items()}}, f,
                      ensure_ascii=False, indent=1, default=str)
        print(f"Részletes eredmény: {summary_path}")
        sanity_check(results)
    finally:
        os.chdir(cwd)
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

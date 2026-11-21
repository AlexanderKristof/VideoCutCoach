"""
Végponttól végpontig tartó kiértékelés szintetikus meccsen (valódi Veo felvétel és GPU nélkül).

Lépések:
  1. tools/synth_match.py: rövid szintetikus meccs (alap: 75 s, 1280x720) + ground truth JSON
  2. main.cut_and_analyze_sync futtatása a tools/fake_detector.py ál-YOLO-val
     (a fastapi / pydantic / sqlalchemy / torch / ultralytics / database / models / auth modulok
     csonkolva, 'ffmpeg.exe' a PATH-on egy szimbolikus linkkel), az első képkocka valós
     kalibrációs pontjaival
  3. pontozás: labda-találati arány és hiba labdakövetővel / nélküle, játékos-pozíció hiba
     (a main.py statikus homográfiájával, összevetve a valós képkockánkénti homográfiával =
     'orákulum'), esemény-detektálás pontosság / felidézés idő-tűréssel.

Futtatás:  python3 tests/test_e2e_synth.py
Környezeti változók:
  E2E_DURATION (mp, alap 75), E2E_WIDTH / E2E_HEIGHT (alap 1280x720), E2E_SEED (alap 6),
  E2E_MODES (alap "static_off,static_on,oracle_off,oracle_on"), E2E_WORKDIR (megtartott munkakönyvtár),
  E2E_REUSE=1 (a meglévő videót újrahasznosítja a munkakönyvtárban), E2E_TOL (esemény-tűrés, mp, alap 2.5),
  E2E_DROP_UNTRACKED=0 (az ál-ByteTrack ne dobja el a meg nem erősített dobozokat – alapból eldobja,
  mint az ultralytics model.track())

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
from fake_detector import FakeYOLO  # noqa: E402

EVENT_TYPES = ("shot", "goal", "gk_save", "gk_conceded", "gk_goal_kick", "gk_long_ball", "gk_short_pass")
EVENT_LABELS = {"shot": "lövés", "goal": "gól", "gk_save": "védés", "gk_conceded": "kapott gól",
                "gk_goal_kick": "kirúgás", "gk_long_ball": "kapus ívelés", "gk_short_pass": "kapus rövid passz"}
MODE_LABELS = {
    "static_off": "statikus H, követő nélkül (mai állapot)",
    "static_on": "statikus H + labdakövető",
    "oracle_off": "valós H (orákulum), követő nélkül",
    "oracle_on": "valós H (orákulum) + labdakövető",
}


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
    mod("fastapi", FastAPI=_App, Form=lambda default=None, *a, **k: default, Depends=lambda *a, **k: None,
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


def score_players(records, gt):
    frames = gt["frames"]
    roles = {p["id"]: p["role"] for p in gt["people"]}
    teams = {p["id"]: p["team"] for p in gt["people"]}
    errs, tooth_small, gt_vis, matched = [], 0, 0, 0
    team_ok = team_n = specs_out = 0
    for r in records:
        f = frames[r["frame"]]
        objs = f["objs"]
        cand = [o for o in objs if roles.get(o[0]) in ("player", "goalkeeper")]
        for o in cand:
            gt_vis += 1
            if o[8] - o[6] < 50:
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
            team_n += 1
            team_ok += int(p["team"] == teams.get(bi[0]))
    return {"pitch_err": _stats(errs), "gt_visible_player_boxes": gt_vis,
            "gt_below_50px_ratio": tooth_small / max(1, gt_vis),
            "coverage": matched / max(1, gt_vis), "team_accuracy": team_ok / max(1, team_n),
            "non_player_outputs": specs_out}


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


def base_config(gt, tracker_on):
    period = (gt.get("periods") or [{}])[0]
    kits = synth_match.KITS
    t1, t2 = click_range(kits["team1"]["shirt"]), click_range(kits["team2"]["shirt"])
    k1, k2 = click_range(kits["keeper1"]["shirt"]), click_range(kits["keeper2"]["shirt"])
    return {
        "team1_attacks_right": bool(period.get("team1_attacks_right", True)),
        "review_export": False,
        "roi_top_pct": 0.28, "roi_bottom_pct": 0.10, "min_player_height_pct": 0.048,
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
    }


def run_mode(main_mod, fake, gt, video, mode):
    oracle = mode.startswith("oracle")
    tracker_on = mode.endswith("_on")
    Engine = make_recording_engine(main_mod, gt, oracle)
    original = main_mod.TacticalEngine
    main_mod.TacticalEngine = Engine
    fake.load_ground_truth(gt)
    calib = gt["calibration_first_frame"]["calib_points"]
    t0 = time.time()
    try:
        _url, out_path, events_path = main_mod.cut_and_analyze_sync(
            video, 0.0, gt["duration"], True, "team", calib, base_config(gt, tracker_on))
    finally:
        main_mod.TacticalEngine = original
    elapsed = time.time() - t0
    with open(events_path, encoding="utf-8") as f:
        payload = json.load(f)
    return {"records": Engine.records, "static_H": Engine.static_H, "events": payload["events"],
            "elapsed": elapsed, "fake_stats": dict(fake.stats), "out_video": out_path}


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
    p0 = next(iter(results.values()))["players"]
    print(f"  (a látható valós játékos-dobozok {100 * p0['gt_below_50px_ratio']:.0f}%-a 50 px-nél alacsonyabb – "
          f"a main.py min. 50 px-es magasságszűrője ezeket eldobja)")

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
    print("=" * 96)


def main():
    duration = float(os.environ.get("E2E_DURATION", "75"))
    width = int(os.environ.get("E2E_WIDTH", "1280"))
    height = int(os.environ.get("E2E_HEIGHT", "720"))
    seed = int(os.environ.get("E2E_SEED", "6"))
    tol = float(os.environ.get("E2E_TOL", "2.5"))
    modes = [m.strip() for m in os.environ.get("E2E_MODES", "static_off,static_on,oracle_off,oracle_on").split(",")
             if m.strip()]
    for m in modes:
        if m not in MODE_LABELS:
            raise SystemExit(f"Ismeretlen mód: {m} (lehet: {', '.join(MODE_LABELS)})")
    keep = os.environ.get("E2E_WORKDIR")
    workdir = os.path.abspath(keep) if keep else tempfile.mkdtemp(prefix="e2e_synth_")
    os.makedirs(workdir, exist_ok=True)
    video = os.path.join(workdir, f"synth_{seed}_{int(duration)}s_{width}x{height}.mp4")
    gt_path = os.path.splitext(video)[0] + "_gt.json"
    cwd = os.getcwd()
    try:
        if not (os.environ.get("E2E_REUSE") and os.path.exists(video) and os.path.exists(gt_path)):
            print(f"Szintetikus meccs generálása: {video}")
            synth_match.generate(video, gt_path, duration=duration, width=width, height=height, fps=25, seed=seed)
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        if not gt["calibration_first_frame"].get("all_in_front_of_camera", True):
            print("⚠️ Az első képkockán nem minden pályasarok van a kamera előtt – a kalibráció torz lehet")

        fake = FakeYOLO(seed=seed + 100, drop_untracked=os.environ.get("E2E_DROP_UNTRACKED", "1") != "0")
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
                "players": score_players(r["records"], gt),
                "events": score_events(r["events"], gt["events"], tol),
                "elapsed": r["elapsed"], "fake_stats": r["fake_stats"],
                "detected_events": [{k: e.get(k) for k in ("type", "time", "team")} for e in r["events"]],
            }
        ideal = score_events(ideal_events(gt), gt["events"], tol)
        print_report(results, gt, tol, h_err, ideal)
        summary_path = os.path.join(workdir, "e2e_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"gt_events": gt["events"], "ideal_events": ideal, "results": results}, f,
                      ensure_ascii=False, indent=1, default=str)
        print(f"Részletes eredmény: {summary_path}")
    finally:
        os.chdir(cwd)
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

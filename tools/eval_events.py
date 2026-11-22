#!/usr/bin/env python3
"""
Az eseménydetektor (events.py) offline kiértékelése szintetikus ground truth alapján.

A tools/synth_match.py által írt ground-truth JSON képkockánkénti VALÓS labda- (x, y, magasság)
és játékos-pozícióit szándékosan "elrontjuk", hogy egy valódi detektor + homográfia kimenetét
utánozzuk, majd ezzel etetjük az EventDetector-t, és eseménytípusonként pontosságot /
felidézést számolunk idő-tűréssel.

Zajmodell (mind konfigurálható, lásd NOISE_PRESETS):
  * labda-kiesés: kétállapotú Markov-lánc (látható / hiányzó) -> csomós, többkockás kiesések,
    gyors labdánál (elmosódás) plusz kiesési esély;
  * labdamagasság-vetítés: egyetlen kamera (synth kamera-pozíció) -> a levegőben lévő labda
    talajra vetített pontja a kamerától távolodva "elcsúszik" (pl. a kapus kezében 1 m magasan
    tartott labda a gólvonal mögé vetül);
  * pozíciózaj: fehér Gauss-zaj + lassan vándorló (homográfia-) torzítás;
  * téves labda-felvillanások (1-3 kocka) játékosok közelében vagy véletlen helyen;
  * játékosok: kiesés, pozíció-remegés (a kapusoké külön), track-ID cserék közeli játékosok
    között és új ID kiesés után (fragmentáció);
  * a pályán kívüli (±2 m-nél távolabbi) pontokat eldobjuk, mint a TacticalEngine.transform_to_pitch.

Futtatás:
  python3 tools/eval_events.py --seeds 3,6,7 --levels gt,enyhe,kozepes,eros
  python3 tools/eval_events.py --gt meccs_gt.json --levels kozepes --noise-seeds 1,2,3
  python3 tools/eval_events.py --detector /út/régi_events.py ...   (összehasonlítás egy másik verzióval)

Csak a Python standard könyvtár kell (a GT generálásához numpy is: tools/synth_match.py --gt-only).
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EVENT_TYPES = ("shot", "goal", "gk_save", "gk_conceded", "gk_goal_kick", "gk_long_ball", "gk_short_pass")
# Labdás mezőny-események: külön értékelve, az "ÖSSZES" sor (és a régi küszöbök) csak az EVENT_TYPES-ra
# vonatkozik. A szintetikus meccsben (synth_match) nincs bedobás / szöglet / szabadrúgás forgatókönyv.
PLAY_TYPES = ("pass", "pass_failed", "interception", "kickoff")
ALL_TYPES = EVENT_TYPES + PLAY_TYPES
EVENT_LABELS = {"shot": "lövés", "goal": "gól", "gk_save": "védés", "gk_conceded": "kapott gól",
                "gk_goal_kick": "kirúgás", "gk_long_ball": "kapus ívelés", "gk_short_pass": "kapus rövid passz",
                "pass": "passz", "pass_failed": "sikertelen passz", "interception": "labdaszerzés",
                "kickoff": "kezdőrúgás"}

DEFAULT_CAMERA = (52.5, 74.0, 7.0)

# Zajszintek. ball_detect_rate: a játékban lévő labda észlelési aránya; ball_gap_frames: átlagos
# kiesés-hossz (kocka); ball_sigma_m: fehér zaj szórása; ball_bias_m: lassan vándorló torzítás.
NOISE_PRESETS = {
    "gt": dict(ball_detect_rate=1.0, ball_gap_frames=1.0, fast_miss_extra=0.0, project_height=False,
               ball_sigma_m=0.0, ball_bias_m=0.0, blip_rate=0.0,
               player_miss_rate=0.0, player_sigma_m=0.0, keeper_sigma_m=0.0, player_bias_m=0.0,
               id_swap_per_min=0.0, id_fragment_prob=0.0, keeper_flag_miss=0.0),
    "vetites": dict(ball_detect_rate=1.0, ball_gap_frames=1.0, fast_miss_extra=0.0, project_height=True,
                    ball_sigma_m=0.0, ball_bias_m=0.0, blip_rate=0.0,
                    player_miss_rate=0.0, player_sigma_m=0.0, keeper_sigma_m=0.0, player_bias_m=0.0,
                    id_swap_per_min=0.0, id_fragment_prob=0.0),
    "enyhe": dict(ball_detect_rate=0.8, ball_gap_frames=3.0, fast_miss_extra=0.1, project_height=True,
                  ball_sigma_m=0.3, ball_bias_m=0.2, blip_rate=0.002,
                  player_miss_rate=0.03, player_sigma_m=0.25, keeper_sigma_m=0.3, player_bias_m=0.2,
                  id_swap_per_min=0.5, id_fragment_prob=0.1, keeper_flag_miss=0.05),
    "kozepes": dict(ball_detect_rate=0.65, ball_gap_frames=5.0, fast_miss_extra=0.2, project_height=True,
                    ball_sigma_m=0.6, ball_bias_m=0.4, blip_rate=0.005,
                    player_miss_rate=0.06, player_sigma_m=0.4, keeper_sigma_m=0.6, player_bias_m=0.3,
                    id_swap_per_min=1.0, id_fragment_prob=0.25, keeper_flag_miss=0.1),
    "eros": dict(ball_detect_rate=0.45, ball_gap_frames=8.0, fast_miss_extra=0.3, project_height=True,
                 ball_sigma_m=0.9, ball_bias_m=0.6, blip_rate=0.01,
                 player_miss_rate=0.1, player_sigma_m=0.6, keeper_sigma_m=1.0, player_bias_m=0.5,
                 id_swap_per_min=2.0, id_fragment_prob=0.4, keeper_flag_miss=0.15),
}


# ------------------------------------------------------------------ betöltés
def load_detector_module(path=None):
    """Az events.py (vagy egy megadott másik verziója) betöltése."""
    if not path:
        import events
        return events
    spec = importlib.util.spec_from_file_location("events_under_test_%d" % abs(hash(path)), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_gt(seed, duration, out_dir):
    """Renderelés nélküli ground truth a tools/synth_match.py-val (numpy kell)."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import synth_match
    path = os.path.join(out_dir, f"synth_{seed}_{int(duration)}s_gt.json")
    if not os.path.exists(path):
        synth_match.generate(os.path.join(out_dir, f"synth_{seed}.mp4"), path, duration=duration,
                             fps=25, seed=seed, progress=False, gt_only=True)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------- zajosítás
class _AR1:
    """Lassan vándorló 2D torzítás (első rendű autoregresszív folyamat)."""

    def __init__(self, rng, amp, tau_s, dt):
        self.rng, self.amp = rng, amp
        self.a = math.exp(-dt / tau_s) if tau_s > 0 else 0.0
        self.s = amp * math.sqrt(max(0.0, 1.0 - self.a * self.a))
        self.v = [rng.gauss(0, amp), rng.gauss(0, amp)] if amp > 0 else [0.0, 0.0]

    def step(self):
        if self.amp <= 0:
            return 0.0, 0.0
        self.v = [self.a * self.v[0] + self.rng.gauss(0, self.s), self.a * self.v[1] + self.rng.gauss(0, self.s)]
        return self.v[0], self.v[1]


def project_to_ground(x, y, z, cam):
    """Egykamerás vetítés: a z magasságú labda képe a talajon ott van, ahol a kamera-labda egyenes metszi."""
    cx, cy, cz = cam
    if z <= 0.0 or z >= cz - 0.2:
        return x, y
    k = cz / (cz - z)
    return cx + (x - cx) * k, cy + (y - cy) * k


def degrade(gt, noise, seed=0):
    """A GT képkockákból zajos detektor-bemenet: [(t, ball_xy | None, players)]."""
    rng = random.Random(seed)
    fps = float(gt.get("fps", 25))
    dt = 1.0 / fps
    L = gt["pitch"]["length"]
    W = gt["pitch"]["width"]
    cam = tuple((gt.get("camera") or {}).get("position") or DEFAULT_CAMERA)
    n = dict(NOISE_PRESETS["gt"])
    n.update(noise)

    miss = 1.0 - n["ball_detect_rate"]
    p_mv = 1.0 / max(1.0, n["ball_gap_frames"])           # hiányzó -> látható
    p_vm = min(1.0, miss * p_mv / max(1e-6, 1.0 - miss))  # látható -> hiányzó
    ball_state_vis = True
    ball_bias = _AR1(rng, n["ball_bias_m"], 3.0, dt)
    blip_left, blip_pos = 0, None

    people = [p for p in gt["people"] if p["role"] in ("player", "goalkeeper")]
    biases = {p["id"]: _AR1(rng, n["player_bias_m"], 3.0, dt) for p in people}
    tid_of = {p["id"]: p["id"] for p in people}
    next_tid = 1000
    missing_since = {}
    swap_p = n["id_swap_per_min"] / 60.0 * dt

    out = []
    prev_xy = None
    for f in gt["frames"]:
        t = f["t"]
        xyz = f["ball"]["xyz"]
        on = f["ball"].get("on", True)
        bx, by, bz = float(xyz[0]), float(xyz[1]), float(xyz[2])
        speed = 0.0
        if prev_xy is not None:
            speed = math.hypot(bx - prev_xy[0], by - prev_xy[1]) / dt
        prev_xy = (bx, by)

        # ----- labda
        if ball_state_vis:
            if rng.random() < p_vm:
                ball_state_vis = False
        elif rng.random() < p_mv:
            ball_state_vis = True
        bias = ball_bias.step()
        ball = None
        if on and ball_state_vis and not (speed > 15.0 and rng.random() < n["fast_miss_extra"]):
            px, py = project_to_ground(bx, by, bz, cam) if n["project_height"] else (bx, by)
            ball = (px + bias[0] + rng.gauss(0, n["ball_sigma_m"]) if n["ball_sigma_m"] else px + bias[0],
                    py + bias[1] + rng.gauss(0, n["ball_sigma_m"]) if n["ball_sigma_m"] else py + bias[1])
        if blip_left <= 0 and n["blip_rate"] > 0 and rng.random() < n["blip_rate"]:
            blip_left = rng.randint(1, 3)
            if rng.random() < 0.6:
                p = people[rng.randrange(len(people))]
                x0, y0 = f["xy"][p["id"]]
                blip_pos = (x0 + rng.uniform(-3, 3), y0 + rng.uniform(-3, 3))
            else:
                blip_pos = (rng.uniform(0, L), rng.uniform(0, W))
        if blip_left > 0:
            blip_left -= 1
            ball = (blip_pos[0] + rng.gauss(0, 0.3), blip_pos[1] + rng.gauss(0, 0.3))
        if ball is not None and not (-2.0 <= ball[0] <= L + 2.0 and -2.0 <= ball[1] <= W + 2.0):
            ball = None   # transform_to_pitch: a pályán kívüli pontot eldobja

        # ----- játékosok
        if swap_p > 0 and rng.random() < swap_p:
            a = people[rng.randrange(len(people))]
            ax, ay = f["xy"][a["id"]]
            near = [p for p in people if p["id"] != a["id"]
                    and math.hypot(f["xy"][p["id"]][0] - ax, f["xy"][p["id"]][1] - ay) < 2.5]
            if near:
                b = near[rng.randrange(len(near))]
                tid_of[a["id"]], tid_of[b["id"]] = tid_of[b["id"]], tid_of[a["id"]]
        players = []
        for p in people:
            pid = p["id"]
            keeper = p["role"] == "goalkeeper"
            if n["player_miss_rate"] > 0 and rng.random() < n["player_miss_rate"]:
                missing_since.setdefault(pid, t)
                continue
            if pid in missing_since:
                if rng.random() < n["id_fragment_prob"]:
                    tid_of[pid] = next_tid
                    next_tid += 1
                del missing_since[pid]
            x, y = f["xy"][pid]
            sb = biases[pid].step()
            sig = n["keeper_sigma_m"] if p["role"] == "goalkeeper" else n["player_sigma_m"]
            x = x + sb[0] + (rng.gauss(0, sig) if sig else 0.0)
            y = y + sb[1] + (rng.gauss(0, sig) if sig else 0.0)
            if keeper and n["keeper_flag_miss"] > 0 and rng.random() < n["keeper_flag_miss"]:
                keeper = False   # a kapust mezőnyjátékosnak látja az osztályozó
            players.append({"track_id": tid_of[pid], "team_id": p["team"], "is_keeper": keeper,
                            "pitch_coords": (x, y)})
        out.append((t, ball, players))
    return out


# ----------------------------------------------------------------- futtatás
def run_detector(mod, gt, frames):
    period = (gt.get("periods") or [{}])[0]
    cfg = mod.EventConfig()
    cfg.team1_attacks_right = bool(period.get("team1_attacks_right", True))
    det = mod.EventDetector(cfg, gt["pitch"]["length"], gt["pitch"]["width"])
    for t, ball, players in frames:
        det.update(t, ball, players)
    if frames:
        det.finalize(frames[-1][0])
    return det.events


def score_events(detected, gt_events, tol, types=ALL_TYPES):
    """Eseménytípusonként mohó párosítás (legközelebbi időpont, max. tol mp) -> tp / det / gt / csapat."""
    res = {}
    for typ in types:
        g = sorted((e for e in gt_events if e["type"] == typ), key=lambda e: e["time"])
        d = sorted((e for e in detected if e["type"] == typ), key=lambda e: e["time"])
        used = set()
        tp = team_ok = 0
        for ge in g:
            best, bj = None, None
            for j, de in enumerate(d):
                if j in used or abs(de["time"] - ge["time"]) > tol:
                    continue
                if best is None or abs(de["time"] - ge["time"]) < best:
                    best, bj = abs(de["time"] - ge["time"]), j
            if bj is not None:
                used.add(bj)
                tp += 1
                team_ok += int(d[bj].get("team") == ge.get("team"))
        res[typ] = {"gt": len(g), "det": len(d), "tp": tp, "team_ok": team_ok}
    return res


def add_scores(acc, res):
    for typ, v in res.items():
        a = acc.setdefault(typ, {"gt": 0, "det": 0, "tp": 0, "team_ok": 0})
        for k in a:
            a[k] += v[k]
    return acc


def totals(acc, types=EVENT_TYPES):
    tot = {"gt": 0, "det": 0, "tp": 0, "team_ok": 0}
    for typ, v in acc.items():
        if types is not None and typ not in types:
            continue
        for k in tot:
            tot[k] += v[k]
    return tot


def pr(v):
    p = v["tp"] / v["det"] if v["det"] else None
    r = v["tp"] / v["gt"] if v["gt"] else None
    return p, r


def evaluate(gts, levels, noise_seeds, tol=2.0, detector=None):
    """{szint: {típus: {gt, det, tp, team_ok}}} az összes GT és zaj-mag összegeként."""
    mod = detector if detector is not None and not isinstance(detector, str) else load_detector_module(detector)
    out = {}
    for lvl in levels:
        noise = NOISE_PRESETS[lvl] if isinstance(lvl, str) else lvl
        acc = {}
        for gt in gts:
            for ns in (noise_seeds if lvl != "gt" else [0]):
                frames = degrade(gt, noise, seed=ns * 7919 + int(gt.get("seed", 0)))
                add_scores(acc, score_events(run_detector(mod, gt, frames), gt["events"], tol))
        out[lvl if isinstance(lvl, str) else "egyedi"] = acc
    return out


def _f(x):
    return "  – " if x is None else f"{x:.2f}"


def format_table(results, title=None):
    """Szöveges táblázat: soronként egy eseménytípus, oszloponként egy zajszint (P / R / tp/det/gt)."""
    lines = []
    if title:
        lines.append(title)
    levels = list(results.keys())
    head = f"{'esemény':18s}" + "".join(f" | {lvl:^22s}" for lvl in levels)
    lines.append(head)
    lines.append(f"{'':18s}" + "".join(f" | {'P':>5s} {'R':>5s} {'tp/det/gt':>10s}" for _ in levels))
    lines.append("-" * len(head))
    for typ in list(EVENT_TYPES) + ["_összes"] + list(PLAY_TYPES):
        row = f"{(EVENT_LABELS.get(typ, typ) if typ != '_összes' else 'ÖSSZES'):18s}"
        for lvl in levels:
            v = totals(results[lvl]) if typ == "_összes" else results[lvl].get(typ, {"gt": 0, "det": 0, "tp": 0})
            p, r = pr(v)
            cnt = f"{v['tp']}/{v['det']}/{v['gt']}"
            row += f" | {_f(p):>5s} {_f(r):>5s} {cnt:>10s}"
        lines.append(row)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="events.py offline kiértékelése zajosított szintetikus ground truth-tal.")
    ap.add_argument("--gt", action="append", default=[], help="ground-truth JSON (többször is megadható)")
    ap.add_argument("--seeds", default="3,6,7", help="ha nincs --gt: synth_match magok (alap: 3,6,7)")
    ap.add_argument("--duration", type=float, default=120.0, help="generált GT hossza mp-ben (alap: 120)")
    ap.add_argument("--gt-dir", default=None, help="a generált GT JSON-ok könyvtára (alap: ideiglenes)")
    ap.add_argument("--levels", default="gt,vetites,enyhe,kozepes,eros",
                    help=f"zajszintek vesszővel ({', '.join(NOISE_PRESETS)})")
    ap.add_argument("--noise-seeds", default="1,2,3", help="zaj-magok (alap: 1,2,3)")
    ap.add_argument("--tol", type=float, default=2.0, help="idő-tűrés mp-ben (alap: 2.0)")
    ap.add_argument("--detector", action="append", default=[],
                    help="kiértékelendő events.py útvonal(ak); alap: a repó events.py-ja")
    ap.add_argument("--json", default=None, help="eredmények mentése JSON-ba")
    args = ap.parse_args(argv)

    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    for lvl in levels:
        if lvl not in NOISE_PRESETS:
            raise SystemExit(f"Ismeretlen zajszint: {lvl}")
    noise_seeds = [int(s) for s in args.noise_seeds.split(",") if s.strip()]
    gts = []
    for p in args.gt:
        with open(p, encoding="utf-8") as f:
            gts.append(json.load(f))
    if not gts:
        d = args.gt_dir or tempfile.mkdtemp(prefix="eval_events_")
        os.makedirs(d, exist_ok=True)
        gts = [generate_gt(int(s), args.duration, d) for s in args.seeds.split(",") if s.strip()]
    n_ev = sum(len([e for e in g["events"] if e["type"] in EVENT_TYPES]) for g in gts)
    print(f"{len(gts)} ground truth, {n_ev} értékelt esemény, zaj-magok: {noise_seeds}, tűrés ±{args.tol} s")
    all_res = {}
    for det_path in (args.detector or [None]):
        name = det_path or "events.py"
        res = evaluate(gts, levels, noise_seeds, args.tol, det_path)
        all_res[name] = res
        print()
        print(format_table(res, title=f"Detektor: {name}"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

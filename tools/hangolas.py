#!/usr/bin/env python3
"""
Esemény-küszöbök hangolása az edző címkéiből (videó és GPU nélkül).

Bemenet: egy vagy több elemzett meccs (match_events_*.json / events_*.json) és a hozzájuk tartozó
visszajelzés (feedback_<név>.json: ✓ / ✗ ítéletek, kézzel felvett hiányzó események, a címkéző csomag
végignézett ablakai). A detektor bemenete a detektor-naplóból (detector_log.py – pontos), ha nincs, a mentett
(ritkított) pályákból jön (tools/replay_events.py).

Módszer (koordinátánkénti keresés):
  * csak a címkék környékét játssza vissza (minden címke előtt --elotte mp, utána --utana mp) – gyors;
  * paraméterenként végigpróbálja a jelölt értékeket (SEARCH), és csak akkor tartja meg az újat, ha a
    pontszám (F1 a megtalált / kimaradt / téves eseményekből) nő ÉS legalább egy eseménnyel jobb
    (több találat vagy kevesebb téves) – a zajra való túlhangolás ellen;
  * csak azokat a paramétereket hangolja, amelyek a címkézett eseménytípusokat érintik;
  * szintetikus őr: minden elfogadott változás után a szintetikus meccsen (tools/synth_match.py, közepes zaj)
    is lefut a detektor; ha ott az összesített F1 --szint-tures-nél többet romlik, a változást elveti.

Kimenet: esemeny_hangolas.json ({"overrides": {param: érték}, ...}) – a motor a következő elemzéstől ezt
használja (events.load_tuning, app_paths.event_tuning_path). A régi állapot "enabled": false-szal vagy a fájl
törlésével visszaállítható.

Példák:
    python tools/hangolas.py clips/match_events_ab12.json
    python tools/hangolas.py clips/match_events_ab12.json clips/match_events_cd34.json --ment
    python tools/hangolas.py m.json:feedback_m.json --ki proba_hangolas.json --menetek 2
"""
import argparse
import datetime
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import detector_log  # noqa: E402
import events as events_mod  # noqa: E402
import feedback as fb  # noqa: E402

# Hangolható paraméterek: jelölt értékek + az általuk érintett eseménytípusok ("*" = mindegyik)
SEARCH = {
    # labda-szűrés (minden eseményt érint)
    "max_ball_speed_mps": ([40.0, 55.0, 70.0], "*"),
    "ball_gate_m": ([1.5, 2.0, 3.0, 4.0], "*"),
    "ball_switch_samples": ([3, 4, 6], "*"),
    # birtoklás -> passz / sikertelen passz / szerelés / labdavezetés / beadás
    "ctl_radius_m": ([1.5, 2.0, 2.5, 3.0, 3.5], "pass pass_failed interception tackle dribble cross"),
    "ctl_min_samples": ([2, 3, 4], "pass pass_failed interception tackle dribble cross"),
    "ctl_min_span_s": ([0.1, 0.2, 0.4], "pass pass_failed interception tackle dribble cross"),
    "ctl_gap_s": ([0.6, 1.0, 1.5, 2.0], "pass pass_failed interception tackle dribble cross"),
    "ctl_max_ball_speed_mps": ([6.0, 8.0, 11.0], "pass pass_failed interception tackle dribble cross"),
    "pass_min_m": ([4.0, 5.0, 6.0, 8.0], "pass"),
    "pass_max_gap_s": ([3.0, 4.0, 6.0], "pass pass_failed interception cross"),
    "pass_max_unseen_s": ([1.0, 1.5, 2.5, 3.5], "pass pass_failed interception cross"),
    "pass_receive_s": ([0.8, 1.2, 2.0, 3.0], "pass pass_failed interception cross"),
    "pass_dribble_m": ([1.5, 2.5, 3.5], "pass dribble"),
    "pass_failed_min_m": ([6.0, 8.0, 12.0], "pass_failed interception"),
    "tackle_max_m": ([3.0, 4.0, 6.0], "tackle"),
    "tackle_contact_m": ([2.0, 2.5, 3.5], "tackle"),
    "dribble_min_m": ([6.0, 8.0, 10.0, 14.0], "dribble"),
    "dribble_min_s": ([1.0, 1.5, 2.0, 3.0], "dribble"),
    # beadás
    "cross_final_third_m": ([30.0, 35.0, 40.0], "cross"),
    "cross_min_m": ([6.0, 8.0, 12.0], "cross"),
    "cross_window_flight_s": ([3.0, 4.0, 6.0], "cross"),
    "cross_box_tol_m": ([1.0, 2.0, 3.5], "cross"),
    # rögzített helyzetek
    "restart_launch_dist_m": ([2.0, 3.0, 4.5], "throw_in corner free_kick kickoff"),
    "restart_min_speed_mps": ([3.0, 5.0, 7.0], "throw_in corner free_kick kickoff"),
    "restart_kicker_m": ([2.0, 2.5, 3.5, 5.0], "throw_in corner free_kick kickoff"),
    "touch_zone_m": ([2.0, 2.5, 3.5, 5.0], "throw_in"),
    "touch_out_margin_m": ([0.3, 0.5, 1.0], "throw_in"),
    "throw_in_still_s": ([0.5, 0.8, 1.2], "throw_in"),
    "throw_in_rest_alone_s": ([1.0, 1.5, 2.5], "throw_in"),
    "throw_in_min_dead_s": ([2.0, 3.0, 4.0], "throw_in"),
    "throw_in_max_dx_m": ([15.0, 25.0, 35.0], "throw_in"),
    "corner_radius_m": ([2.5, 3.5, 5.0], "corner"),
    "corner_still_s": ([0.6, 1.0, 1.5], "corner"),
    "free_kick_still_s": ([1.0, 1.5, 2.5], "free_kick"),
    "free_kick_dead_s": ([2.0, 3.0, 4.5], "free_kick"),
    "free_kick_min_players": ([1, 2, 3], "free_kick"),
    "kickoff_still_s": ([0.6, 1.0, 1.5], "kickoff"),
    # lövés / gól / védés
    "shot_speed_kmh": ([40.0, 45.0, 55.0], "shot goal gk_save"),
    "shot_max_distance_m": ([30.0, 35.0, 40.0], "shot gk_save"),
    "shot_min_x_ratio": ([0.4, 0.5, 0.65], "shot gk_save"),
    "save_contact_radius_m": ([1.5, 2.0, 3.0], "gk_save"),
    "save_loose_radius_m": ([2.5, 3.0, 4.0], "gk_save"),
    # kapus-kihozatal
    "long_ball_m": ([30.0, 35.0, 40.0], "gk_long_ball gk_short_pass"),
    "min_distribution_m": ([3.0, 4.0, 6.0], "gk_long_ball gk_short_pass gk_goal_kick"),
    "distribution_lost_s": ([2.5, 3.5, 5.0], "gk_long_ball gk_short_pass"),
    "distribution_return_m": ([10.0, 15.0, 20.0], "gk_long_ball gk_short_pass"),
    "gk_keep_radius_m": ([2.0, 2.5, 3.5], "gk_long_ball gk_short_pass gk_claim_cross gk_save"),
    "gk_release_radius_m": ([2.5, 3.0, 4.0], "gk_long_ball gk_short_pass"),
    "goal_kick_still_s": ([0.6, 1.0, 1.5], "gk_goal_kick"),
    "goal_kick_min_speed_mps": ([6.0, 8.0, 10.0], "gk_goal_kick"),
    # kapus-kifutás
    "sweeper_speed_mps": ([3.0, 4.0, 5.0], "gk_sweeper"),
    "sweeper_min_line_dist_m": ([10.0, 12.0, 16.0], "gk_sweeper"),
    "sweeper_max_ball_dist_m": ([20.0, 30.0, 40.0], "gk_sweeper"),
}


# "Engedékeny" küszöbök: ezekkel a detektor a kétes helyzeteket is jelzi – ami csak így jön elő, az a
# "majdnem esemény" (a címkéző csomag "kétes" jelenetei: ezekből tanul a hangolás a legtöbbet).
PERMISSIVE = {
    "ctl_radius_m": 3.0, "ctl_min_samples": 2, "ctl_min_span_s": 0.1, "ctl_gap_s": 2.0, "ctl_max_ball_speed_mps": 11.0,
    "pass_min_m": 4.0, "pass_max_gap_s": 6.0, "pass_max_unseen_s": 3.5, "pass_receive_s": 3.0,
    "pass_failed_min_m": 6.0, "dribble_min_m": 6.0, "dribble_min_s": 1.0,
    "cross_final_third_m": 40.0, "cross_min_m": 6.0, "cross_window_flight_s": 6.0, "cross_box_tol_m": 3.5,
    "restart_launch_dist_m": 2.0, "restart_min_speed_mps": 3.0, "restart_kicker_m": 5.0,
    "touch_zone_m": 5.0, "throw_in_still_s": 0.5, "throw_in_rest_alone_s": 1.0, "throw_in_min_dead_s": 2.0,
    "throw_in_max_dx_m": 35.0, "corner_radius_m": 5.0, "corner_still_s": 0.6,
    "free_kick_still_s": 1.0, "free_kick_dead_s": 2.0, "free_kick_min_players": 1, "kickoff_still_s": 0.6,
    "shot_speed_kmh": 38.0, "shot_max_distance_m": 40.0, "shot_min_x_ratio": 0.4,
    "min_distribution_m": 3.0, "distribution_lost_s": 5.0, "goal_kick_still_s": 0.6,
    "sweeper_speed_mps": 3.0, "sweeper_min_line_dist_m": 10.0,
}


def near_misses(segments, base_overrides=None, n=20, avoid_times=(), min_gap_s=6.0, avoid_gap_s=3.0, tol_s=4.0,
                seed=0):
    """
    "Majdnem események": amit az engedékeny küszöbökkel futó detektor talál, de a normál (base_overrides)
    nem (azonos típus ±tol_s). Legfeljebb n darab, típusonként váltogatva, egymástól min_gap_s-re, a már
    meglévő események (avoid_times) idejétől legalább avoid_gap_s-re. Visszatér: eseménylista ("kétes": True mezővel).
    """
    import random
    pieces = [{"rows": seg["rows"], "t1a": seg["t1a"],
               "end": seg.get("end") if seg.get("end") is not None else
               max((r[1] for r in seg["rows"] if r[0] == "frame"), default=0.0)} for seg in segments]
    normal = run_pieces(pieces, dict(base_overrides or {}))
    loose = run_pieces(pieces, dict(base_overrides or {}, **PERMISSIVE))
    by_type = {}
    for e in loose:
        if e.get("type") == "gk_positioning_error":
            continue
        t = float(e["time"])
        if any(x["type"] == e["type"] and abs(float(x["time"]) - t) <= tol_s for x in normal):
            continue
        by_type.setdefault(e["type"], []).append(e)
    rng = random.Random(seed)
    for v in by_type.values():
        rng.shuffle(v)
    avoid = [float(t) for t in avoid_times]
    taken = []
    out = []
    while len(out) < n and any(by_type.values()):
        for typ in sorted(by_type):
            lst = by_type[typ]
            while lst:
                e = lst.pop()
                t = float(e["time"])
                if all(abs(t - x) >= min_gap_s for x in taken) and all(abs(t - x) >= avoid_gap_s for x in avoid):
                    ev = dict(e)
                    ev["ketes"] = True
                    out.append(ev)
                    taken.append(t)
                    break
            if len(out) >= n:
                break
    out.sort(key=lambda e: float(e["time"]))
    return out


# ------------------------------------------------------------------ bemenet
def _load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_match(spec):
    """'meccs.json' vagy 'meccs.json:feedback.json' -> {"name", "data", "feedback", "segments"}."""
    if ":" in spec and not os.path.exists(spec) and os.path.exists(spec.rsplit(":", 1)[0]):
        mpath, fpath = spec.rsplit(":", 1)
    else:
        mpath, fpath = spec, None
    data = _load_json(mpath)
    if fpath is None:
        stem = os.path.basename(mpath)[:-5]
        fpath = os.path.join(os.path.dirname(os.path.abspath(mpath)), f"feedback_{stem}.json")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Nincs visszajelzés-fájl: {fpath}")
    feedback = _load_json(fpath)
    for k in ("items", "missing", "reviewed"):
        feedback.setdefault(k, [])
    return {"name": os.path.basename(mpath), "data": data, "feedback": feedback,
            "segments": load_segments(data, os.path.dirname(os.path.abspath(mpath)))}


def load_segments(data, base_dir):
    """A detektor-bemenet szakaszai: [{"name", "rows": [("frame", t, ball, players) | ("dir", t, irány)],
    "t1a": bool, "end": t}] – detektor-naplóból, vagy a ritkított pályákból."""
    segs = []
    logs = detector_log.find_logs(data, base_dir)
    if logs:
        for name, path in logs:
            header, rows = detector_log.read(path)
            rows = [(r[0], r[1], r[2], r[5]) if r[0] == "frame" else r for r in rows]
            segs.append({"name": name or os.path.basename(path), "rows": rows, "source": "napló",
                         "t1a": bool(header.get("team1_attacks_right", True)), "end": header.get("end_sec")})
        return segs
    import replay_events
    for name, t0, t1, half in replay_events.load_segments(data):
        frames = replay_events.build_frames(data, t0, t1, half)
        segs.append({"name": name, "rows": [("frame", t, b, ps) for t, b, ps in frames], "source": "pályák",
                     "t1a": True, "end": t1})
    return segs


def label_times(feedback):
    ts = [float(it["time"]) for it in feedback["items"]]
    ts += [float(it["javitott_ido"]) for it in feedback["items"] if it.get("javitott_ido") is not None]
    ts += [float(m["time"]) for m in feedback["missing"]]
    ts += [float(w["start"]) for w in feedback["reviewed"]] + [float(w["end"]) for w in feedback["reviewed"]]
    return sorted(ts)


def count_labels(feedback):
    """A megítélt események száma: valódi (✓, hiányzó, javított típus) + téves (✗) + "semmi" ablakok."""
    n = len(feedback["items"]) + len(feedback["missing"])
    n += sum(1 for it in feedback["items"] if it.get("javitott_tipus") not in (None, "", "semmi"))
    n += sum(1 for w in feedback["reviewed"] if w.get("label") == "semmi")
    return n


def labelled_types(feedbacks):
    out = set()
    for f in feedbacks:
        out.update(it["type"] for it in f["items"])
        out.update(it["javitott_tipus"] for it in f["items"] if it.get("javitott_tipus") not in (None, "", "semmi"))
        out.update(m["type"] for m in f["missing"])
        out.update(w["label"] for w in f["reviewed"] if w.get("label") not in (None, "", "semmi"))
    return out


def crop_windows(match, before_s, after_s):
    """A visszajátszandó darabok: a címkék körüli [t - before, t + after] ablakok összevonva, szakaszonként."""
    ts = label_times(match["feedback"])
    pieces = []
    for seg in match["segments"]:
        frames_t = [r[1] for r in seg["rows"] if r[0] == "frame"]
        if not frames_t:
            continue
        lo, hi = frames_t[0], frames_t[-1]
        iv = []
        for t in ts:
            if lo - after_s <= t <= hi + before_s:
                a, b = max(lo, t - before_s), min(hi, t + after_s)
                if iv and a <= iv[-1][1]:
                    iv[-1][1] = max(iv[-1][1], b)
                else:
                    iv.append([a, b])
        for a, b in iv:
            rows = [r for r in seg["rows"] if a <= r[1] <= b]
            if rows:
                pieces.append({"rows": rows, "t1a": seg["t1a"], "end": b})
    return pieces


# ------------------------------------------------------------------ futtatás / pontozás
def run_pieces(pieces, overrides):
    evs = []
    for pc in pieces:
        cfg = events_mod.EventConfig(**overrides)
        cfg.team1_attacks_right = pc["t1a"]
        det = events_mod.EventDetector(cfg)
        for r in pc["rows"]:
            if r[0] == "dir":
                det.set_attack_direction(r[2])
            else:
                det.update(r[1], r[2], r[3])
        det.finalize(pc["end"])
        evs.extend(det.events)
    return evs


def score(matches, overrides, tol_s):
    """Összesített eredmény az összes meccsre: {"tp", "fp", "fn", "f1", "meccsenkent": [...]}."""
    tp = fp = fn = 0
    per = []
    for m in matches:
        evs = run_pieces(m["pieces"], overrides)
        r = fb.evaluate(m["feedback"], evs, tol_s=tol_s)
        tp += r["talalt"]
        fn += r["kimaradt"]
        fp += r["teves_maradt"] + r["teves_csendes"]
        per.append(r)
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4), "meccsenkent": per}


# ------------------------------------------------------------------ szintetikus őr
class SynthGuard:
    """A detektor a szintetikus meccsen (közepes zaj): az összesített F1 nem romolhat --szint-tures-nél többet."""

    def __init__(self, seeds=(6,), duration=240.0, cache_dir=None, tol=2.0):
        import tempfile
        import eval_events as ev
        self.ev = ev
        cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "tacanaly_hangolas_szint")
        os.makedirs(cache_dir, exist_ok=True)
        self.sets = []
        for s in seeds:
            gt = ev.generate_gt(s, duration, cache_dir)
            frames = ev.degrade(gt, dict(ev.NOISE_PRESETS["kozepes"]), seed=s * 7919)
            t1a = bool((gt.get("periods") or [{}])[0].get("team1_attacks_right", True))
            self.sets.append((gt, frames, t1a))
        self.tol = tol

    def f1(self, overrides):
        acc = {}
        for gt, frames, t1a in self.sets:
            cfg = events_mod.EventConfig(**overrides)
            cfg.team1_attacks_right = t1a
            det = events_mod.EventDetector(cfg, gt["pitch"]["length"], gt["pitch"]["width"])
            for t, b, ps in frames:
                det.update(t, b, ps)
            det.finalize(frames[-1][0])
            self.ev.add_scores(acc, self.ev.score_events(det.events, gt["events"], self.tol))
        tot = self.ev.totals(acc, types=None)
        denom = tot["det"] + tot["gt"]
        return round(2 * tot["tp"] / denom, 4) if denom else 0.0


# ------------------------------------------------------------------ keresés
def _eval_candidate(args):
    matches, overrides, tol_s = args
    return score(matches, overrides, tol_s)


def tune(matches, start=None, passes=2, tol_s=3.0, guard=None, guard_tol=0.02, log=print, workers=None,
         params=None):
    """Koordinátánkénti keresés. Visszatér: (overrides, kezdő eredmény, végső eredmény, változások listája)."""
    start = dict(start or {})
    types = labelled_types([m["feedback"] for m in matches])
    names = params or [k for k, (_vals, ts) in SEARCH.items() if ts == "*" or types & set(ts.split())]
    base = score(matches, start, tol_s)
    best, cur = base, dict(start)
    guard_base = guard.f1(cur) if guard is not None else None
    log(f"Kiinduló állapot: {best['tp']} talált, {best['fn']} kimaradt, {best['fp']} téves  (F1 {best['f1']})"
        + (f"; szintetikus F1 {guard_base}" if guard is not None else ""))
    log(f"Címkézett típusok: {', '.join(sorted(types)) or '-'}; hangolt paraméterek: {len(names)}")
    changes = []
    pool = None
    if workers and workers > 1:
        import multiprocessing as mp
        try:
            pool = mp.get_context("fork").Pool(workers)
        except (ValueError, OSError):
            pool = None
    try:
        ref = events_mod.EventConfig()
        for p in range(passes):
            improved = False
            for name in names:
                vals = SEARCH[name][0]
                now = cur.get(name, getattr(ref, name))
                cands = [v for v in vals if v != now]
                if not cands:
                    continue
                jobs = [(matches, dict(cur, **{name: v}), tol_s) for v in cands]
                results = pool.map(_eval_candidate, jobs) if pool is not None else [_eval_candidate(j) for j in jobs]
                order = sorted(zip(cands, results), key=lambda cr: (-cr[1]["f1"], abs(cr[0] - now)))
                # a minden eseményt érintő (labda-szűrő) paraméterhez erősebb bizonyíték kell
                need = 2 if SEARCH[name][1] == "*" else 1
                for v, r in order:
                    net = (r["tp"] - best["tp"]) - (r["fp"] - best["fp"])
                    if not (r["f1"] > best["f1"] + 1e-9 and net >= need):
                        break
                    trial = dict(cur, **{name: v})
                    if guard is not None:
                        g = guard.f1(trial)
                        if g < guard_base - guard_tol:
                            log(f"  {name}: {now} -> {v} elvetve (szintetikus F1 {guard_base} -> {g})")
                            continue
                    changes.append({"param": name, "regi": now, "uj": v,
                                    "elotte": {k: best[k] for k in ("tp", "fp", "fn", "f1")},
                                    "utana": {k: r[k] for k in ("tp", "fp", "fn", "f1")}})
                    log(f"  {name}: {now} -> {v}   talált {best['tp']}->{r['tp']}, téves {best['fp']}->{r['fp']}, "
                        f"F1 {best['f1']}->{r['f1']}")
                    cur, best, improved = trial, r, True
                    break
            if not improved:
                break
    finally:
        if pool is not None:
            pool.close()
    # csak az alapértelmezéstől eltérő értékek
    ref = events_mod.EventConfig()
    final = {k: v for k, v in cur.items() if getattr(ref, k, None) != v}
    return final, base, best, changes


def main(argv=None):
    ap = argparse.ArgumentParser(description="Esemény-küszöbök hangolása az edző címkéiből (videó nélkül).")
    ap.add_argument("meccsek", nargs="+", help="match_events_*.json (vagy meccs.json:feedback.json)")
    ap.add_argument("--ki", default=None, help="a kimeneti hangolás-fájl (alap: csak kiírás, mentés --ment-tel)")
    ap.add_argument("--ment", action="store_true", help="mentés a motor helyére (esemeny_hangolas.json)")
    ap.add_argument("--nullarol", action="store_true", help="ne a meglévő hangolásból induljon, hanem az alapból")
    ap.add_argument("--menetek", type=int, default=2)
    ap.add_argument("--tures", type=float, default=3.0, help="idő-tűrés a párosításhoz (mp)")
    ap.add_argument("--elotte", type=float, default=90.0, help="a címkék előtt ennyi mp-t játszik vissza")
    ap.add_argument("--utana", type=float, default=20.0)
    ap.add_argument("--nincs-szint", action="store_true", help="a szintetikus őr kikapcsolása")
    ap.add_argument("--szint-tures", type=float, default=0.02)
    ap.add_argument("--szalak", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--min-cimke", type=int, default=10,
                    help="legalább ennyi megítélt esemény (valódi + téves) kell a hangoláshoz (alap 10) – kevés "
                         "címkéből a hangolás a véletlenre tanulna rá")
    args = ap.parse_args(argv)

    import app_paths
    out_path = app_paths.event_tuning_path() if args.ment else args.ki
    start = {} if args.nullarol else events_mod.load_tuning(app_paths.event_tuning_path())
    t0 = time.time()
    matches = []
    for spec in args.meccsek:
        m = load_match(spec)
        m["pieces"] = crop_windows(m, args.elotte, args.utana)
        src = ", ".join(sorted({s["source"] for s in m["segments"]})) or "-"
        nf = sum(len(pc["rows"]) for pc in m["pieces"])
        fbk = m["feedback"]
        print(f"{m['name']}: {len(fbk['items'])} ítélet, {len(fbk['missing'])} hiányzó, {len(fbk['reviewed'])} "
              f"végignézett ablak; forrás: {src}; visszajátszás: {nf} képkocka")
        if not m["pieces"]:
            print("  (nincs visszajátszható szakasz a címkék körül – kimarad)")
            continue
        matches.append(m)
    if not matches:
        print("Nincs hangolható meccs.")
        return 1
    n_labels = sum(count_labels(m["feedback"]) for m in matches)
    if n_labels < args.min_cimke:
        print(f"Még kevés a címke a hangoláshoz: {n_labels} megítélt esemény (legalább {args.min_cimke} kell). "
              f"Címkézz még néhány jelenetet, és futtasd újra – addig az eddigi küszöbök maradnak.")
        return 0
    guard = None
    if not args.nincs_szint:
        try:
            guard = SynthGuard()
        except Exception as ex:   # pl. nincs numpy: őr nélkül fut
            print(f"(szintetikus őr nélkül: {ex})")
    final, base, best, changes = tune(matches, start, args.menetek, args.tures, guard, args.szint_tures,
                                      workers=args.szalak)
    print(f"\nEredmény: talált {base['tp']} -> {best['tp']}, kimaradt {base['fn']} -> {best['fn']}, "
          f"téves {base['fp']} -> {best['fp']}, F1 {base['f1']} -> {best['f1']}  ({time.time() - t0:.0f} mp)")
    if final:
        print("Eltérés az alapértelmezéstől: " + ", ".join(f"{k}={v}" for k, v in sorted(final.items())))
    else:
        print("Az alapértelmezett küszöbök a legjobbak ezekre a címkékre.")
    if out_path:
        payload = {"version": 1, "enabled": True,
                   "created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "meccsek": [m["name"] for m in matches], "overrides": final, "valtozasok": changes,
                   "eredmeny": {"elotte": {k: base[k] for k in ("tp", "fp", "fn", "f1")},
                                "utana": {k: best[k] for k in ("tp", "fp", "fn", "f1")}}}
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)
        print(f"Mentve: {out_path}" + ("  – a motor a következő elemzéstől ezt használja." if args.ment else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

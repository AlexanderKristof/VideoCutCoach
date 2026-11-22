#!/usr/bin/env python3
"""
Valódi (mentett) pályák visszajátszása az eseménydetektoron (events.py) – a detektor ellenőrzéséhez
újra-elemzés (videó, GPU) nélkül.

Bemenet: egy elemzés events_*.json-ja vagy egy meccs match_events_*.json-ja, amelynek
"trajectories" mezője a heatmap.compact_trajectories formátumú:
    {"hz": 5.0, "tracks": {"12": {"team": 1, "keeper": false, "points": [[x, y, t], ...],
                                   "half": 1 (csak meccs-összesítőben)}, ...},
     "ball_hz": 10.0, "ball": [[x, y, t], ...]}
A játékosok 5 Hz-en, a labda 10 Hz-en van mentve. A visszajátszás a labda ütemén (alap 10 Hz)
halad: minden ütemben a labda (ha van ott minta) és a játékosok (a két szomszédos mentett pont
közti lineáris interpolációval, ha a rés rövid) kerülnek a detektorba.

Korlátok (ezért a visszajátszás NEM azonos az eredeti futással):
  * a csapat- és kapus-jelölés követésenként összesített (a képkockánkénti besorolás nincs mentve);
    a "keeper" jelző azt jelenti, hogy a követés VALAMIKOR kapusnak látszott – ez a pálya közepén
    futó játékost is kapussá tenné, ezért alapból csak a kapu közelében (a pontok többsége a
    gólvonaltól --keeper-max-line-m méteren belül) maradó követés kapus (--keeper-mode track: mind);
  * a labda 10 Hz-re ritkítva (az eredeti ~7–25 kép/mp helyett), a játékosok 5 Hz-en.

A meccs-összesítő félidői (halves) külön detektor-futások (mint az eredeti elemzésben).

Példa:
    python3 tools/replay_events.py match_events_d9a1828f.json
    python3 tools/replay_events.py events_*.json --types pass,throw_in --json kimenet.json
"""
import argparse
import bisect
import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import events as events_mod  # noqa: E402


def load_segments(data):
    """A visszajátszandó szakaszok: [(név, t_kezdet, t_vég, félidő-szám vagy None)]."""
    tr = data.get("trajectories") or {}
    halves = data.get("halves") or {}
    segs = []
    for idx, key in enumerate(("first_half", "second_half"), 1):
        h = halves.get(key)
        if h:
            segs.append((key, float(h["start_sec"]), float(h["end_sec"]), idx))
    if not segs:
        src = data.get("source") or {}
        ts = [p[2] for p in tr.get("ball") or []]
        for t in tr.get("tracks", {}).values():
            ts.extend(p[2] for p in t.get("points") or [])
        if not ts:
            return []
        lo = float(src.get("start_sec", min(ts))) if src.get("start_sec") is not None else min(ts)
        hi = float(src.get("end_sec", max(ts))) if src.get("end_sec") is not None else max(ts)
        segs.append(("klip", min(lo, min(ts)), max(hi, max(ts)), None))
    return segs


def _keeper_ok(points, mode, max_line_m, L):
    if mode == "track":
        return True
    if mode == "none":
        return False
    near = sum(1 for p in points if min(p[0], L - p[0]) <= max_line_m)
    return near >= 0.7 * len(points)


def build_frames(data, t0, t1, half=None, hz=None, keeper_mode="near_goal", keeper_max_line_m=25.0,
                 interp_max_gap_s=0.45, L=105.0):
    """A detektor bemenete a [t0, t1] szakaszra: [(t, labda | None, játékosok)]."""
    tr = data.get("trajectories") or {}
    ball_hz = float(tr.get("ball_hz") or 10.0)
    hz = float(hz or ball_hz)
    step = 1.0 / hz
    ball = sorted((float(p[2]), float(p[0]), float(p[1])) for p in tr.get("ball") or [] if len(p) >= 3)
    ball = [b for b in ball if t0 - 1e-6 <= b[0] <= t1 + 1e-6]
    bts = [b[0] for b in ball]
    tracks = []
    for tid, trk in (tr.get("tracks") or {}).items():
        if half is not None and trk.get("half") not in (None, half):
            continue
        pts = sorted((float(p[2]), float(p[0]), float(p[1])) for p in trk.get("points") or [] if len(p) >= 3)
        pts = [p for p in pts if t0 - 1.0 <= p[0] <= t1 + 1.0]
        if not pts:
            continue
        team = trk.get("team")
        keeper = bool(trk.get("keeper")) and _keeper_ok([(p[1], p[2]) for p in pts], keeper_mode,
                                                        keeper_max_line_m, L)
        tracks.append({"tid": int(trk.get("original_track_id", tid)), "team": team, "keeper": keeper,
                       "ts": [p[0] for p in pts], "pts": pts})
    frames = []
    n = int((t1 - t0) / step) + 1
    for i in range(n):
        t = t0 + i * step
        b = None
        j = bisect.bisect_left(bts, t - 0.5 * step)
        if j < len(ball) and abs(ball[j][0] - t) <= 0.5 * step + 1e-6:
            b = (ball[j][1], ball[j][2])
        players = []
        for trk in tracks:
            ts = trk["ts"]
            k = bisect.bisect_left(ts, t)
            pos = None
            if k < len(ts) and abs(ts[k] - t) <= 0.06:
                pos = trk["pts"][k][1:]
            elif 0 < k < len(ts) and ts[k] - ts[k - 1] <= interp_max_gap_s:
                a, c = trk["pts"][k - 1], trk["pts"][k]
                f = (t - a[0]) / (c[0] - a[0])
                pos = (a[1] + (c[1] - a[1]) * f, a[2] + (c[2] - a[2]) * f)
            elif k > 0 and t - ts[k - 1] <= 0.11:
                pos = trk["pts"][k - 1][1:]
            if pos is None:
                continue
            players.append({"track_id": trk["tid"], "team_id": trk["team"] if trk["team"] in (1, 2) else 1,
                            "team_known": trk["team"] in (1, 2), "is_keeper": trk["keeper"],
                            "pitch_coords": (float(pos[0]), float(pos[1]))})
        frames.append((round(t, 3), b, players))
    return frames


def replay(data, hz=None, keeper_mode="near_goal", keeper_max_line_m=25.0, team1_attacks_right=True,
           detector_module=None, timing=False):
    """Visszajátszás szakaszonként. Visszatér: (események, statisztikák szakaszonként, ms/képkocka)."""
    mod = detector_module or events_mod
    out, stats = [], {}
    import time
    spent, nframes = 0.0, 0
    for name, t0, t1, half in load_segments(data):
        frames = build_frames(data, t0, t1, half, hz, keeper_mode, keeper_max_line_m)
        det = mod.EventDetector(mod.EventConfig(team1_attacks_right=team1_attacks_right))
        c0 = time.perf_counter()
        for t, b, players in frames:
            det.update(t, b, players)
        det.finalize(t1)
        spent += time.perf_counter() - c0
        nframes += len(frames)
        for ev in det.events:
            e = dict(ev)
            e["segment"] = name
            out.append(e)
        stats[name] = det.stats
    out.sort(key=lambda e: e["time"])
    return out, stats, (1000.0 * spent / max(1, nframes))


def _fmt_t(t):
    t = float(t)
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mentett pályák visszajátszása az events.py detektoron.")
    ap.add_argument("files", nargs="+", help="events_*.json / match_events_*.json")
    ap.add_argument("--hz", type=float, default=None, help="visszajátszási ütem (alap: a labda mentési üteme)")
    ap.add_argument("--keeper-mode", default="near_goal", choices=("near_goal", "track", "none"))
    ap.add_argument("--keeper-max-line-m", type=float, default=25.0)
    ap.add_argument("--team1-left", action="store_true", help="tartalék irány: az 1. csapat a bal kaput védi")
    ap.add_argument("--types", default="", help="csak ezek az eseménytípusok (vesszővel)")
    ap.add_argument("--json", default=None, help="az események mentése JSON-ba (fájlonként, statisztikával)")
    ap.add_argument("--events-json", default=None,
                    help='az (összes fájl) eseményeinek mentése {"events": [...]} alakban – pl. tools/eval_feedback.py-hoz')
    args = ap.parse_args(argv)
    types = {s.strip() for s in args.types.split(",") if s.strip()}
    all_out = {}
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        evs, stats, ms = replay(data, args.hz, args.keeper_mode, args.keeper_max_line_m,
                                team1_attacks_right=not args.team1_left, timing=True)
        print(f"\n=== {os.path.basename(path)}: {len(evs)} esemény, detektor: {ms:.3f} ms/képkocka")
        cnt = Counter((e["type"], e.get("team")) for e in evs)
        by_type = Counter(e["type"] for e in evs)
        for typ in sorted(by_type):
            per = ", ".join(f"{tm or '?'}: {cnt[(typ, tm)]}" for tm in ("team1", "team2", None) if cnt[(typ, tm)])
            print(f"  {typ:14s} {by_type[typ]:3d}   ({per})  [{events_mod.event_category(typ)}]")
        for e in evs:
            if types and e["type"] not in types:
                continue
            extra = []
            for k in ("distance_m", "zone", "progressive", "is_long", "into_box", "completed", "dangerous",
                      "team_source", "won_by"):
                if k in e:
                    extra.append(f"{k}={e[k]}")
            print(f"  {_fmt_t(e['time'])} ({e['time']:7.2f}) {e['type']:13s} {str(e.get('team')):6s} "
                  f"{e.get('desc', '')}" + (f"  [{', '.join(extra)}]" if extra else ""))
        all_out[path] = {"events": evs, "stats": stats}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_out, f, ensure_ascii=False, indent=1)
    if args.events_json:
        evs = sorted((e for v in all_out.values() for e in v["events"]), key=lambda e: e["time"])
        with open(args.events_json, "w", encoding="utf-8") as f:
            json.dump({"events": evs}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

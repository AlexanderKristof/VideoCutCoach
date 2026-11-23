"""
Kapus Modul: egy csapat kapusát érintő jelenetek egy helyen, egy meccs-eredményből
(match_events_*.json / events_*.json), újraelemzés nélkül.

Mit gyűjt (a választott csapat = a kapus csapata):
  * a kapus saját eseményeit (gk_*: védés, beadás leszedése, kirúgás, ívelés, rövid passz, kifutás, kapott gól);
  * az ellenfél befejezéseit a kapura (lövés, gól), beadásait, szögleteit, veszélyes szabadrúgásait,
    a 16-osba érkező passzait;
  * "támadás a saját zónában": a labda (a mentett labdapálya szerint) legalább min_attack_s ideig a kapus
    kapujától attack_zone_m méteren belül van – ezt a pályákból számolja, nem az eseménydetektor adja;
  * a kapus mozgásait: ha a kapus legalább min_move_m métert elmozdul (a mentett 5 Hz-es pályából),
    egy "gk_move" bejegyzés (honnan, hová, mennyit, milyen gyorsan).

Minden bejegyzés: {"type", "time", "start_sec", "end_sec", "team", "group", "desc", ...} – a group a
Kapus Modul szűrője: "kapus" (saját esemény), "lovesek", "beadasok", "rogzitett", "zona", "mozgas".
"""
import math

PITCH_LEN = 105.0
PITCH_WID = 68.0
BOX_DEPTH = 16.5
BOX_HALF_W = 20.16

GROUP_LABELS = {
    "kapus": "A kapus eseményei",
    "lovesek": "Lövések a kapura",
    "beadasok": "Beadások, 16-osba érkező labdák",
    "rogzitett": "Szögletek, veszélyes szabadrúgások",
    "zona": "Támadás a saját zónában",
    "mozgas": "Kapus-mozgások",
}

TEAM_LABELS = {"team1": "1. csapat", "team2": "2. csapat"}


def _other(team):
    return "team2" if team == "team1" else "team1"


def _team_num(team):
    return 1 if team == "team1" else 2


def _halves(data):
    """[(félidő-szám vagy None, kezdet, vég)] – a meccs-összesítő félidői, különben egy szakasz."""
    halves = data.get("halves") or {}
    out = []
    for idx, key in enumerate(("first_half", "second_half"), 1):
        h = halves.get(key)
        if h and h.get("start_sec") is not None and h.get("end_sec") is not None:
            out.append((idx, float(h["start_sec"]), float(h["end_sec"])))
    if not out:
        out.append((None, -math.inf, math.inf))
    return out


def _half_of(t, halves):
    for idx, a, b in halves:
        if a - 1.0 <= t <= b + 1.0:
            return idx
    return None


def defended_goal_x(data, team, half=None):
    """
    A csapat által védett kapu x-e (0 vagy 105) a félidőben: a csapat mezőnyjátékosainak átlagos
    x-e alapján (a saját kapu felé állnak hátrébb); ha nincs elég pont, a csapat kapus-eseményeiből.
    """
    tracks = ((data.get("trajectories") or {}).get("tracks")) or {}
    num = _team_num(team)
    xs = {1: [], 2: []}
    for tr in tracks.values():
        if half is not None and tr.get("half") not in (None, half):
            continue
        if tr.get("keeper") or tr.get("team") not in (1, 2):
            continue
        xs[tr["team"]].extend(p[0] for p in tr.get("points") or [])
    if len(xs[1]) >= 20 and len(xs[2]) >= 20:
        m_own = sum(xs[num]) / len(xs[num])
        m_opp = sum(xs[3 - num]) / len(xs[3 - num])
        if abs(m_own - m_opp) >= 2.0:
            return 0.0 if m_own < m_opp else PITCH_LEN
    for ev in data.get("events") or []:
        if half is not None and ev.get("half") not in (None, half):
            continue
        if str(ev.get("type", "")).startswith("gk_") and ev.get("team") == team and ev.get("goal_x") is not None:
            return 0.0 if float(ev["goal_x"]) < PITCH_LEN / 2 else PITCH_LEN
    return None


def _in_box(xy, goal_x):
    if xy is None:
        return False
    x, y = float(xy[0]), float(xy[1])
    return abs(x - goal_x) <= BOX_DEPTH and abs(y - PITCH_WID / 2) <= BOX_HALF_W


def _toward(ev, goal_x):
    """Az esemény a goal_x kapura irányul-e (ha az esemény tudja); None = nem tudni."""
    for k in ("goal_x", "attack_goal_x"):
        if ev.get(k) is not None:
            return abs(float(ev[k]) - goal_x) < 1.0
    return None


def _event_group(ev, team, goal_x):
    """A Kapus Modul csoportja egy meccs-eseménynek, vagy None, ha nem érinti a kapust."""
    typ = ev.get("type", "")
    ev_team = ev.get("team")
    if typ.startswith("gk_"):
        if typ == "gk_positioning_error":
            return None
        return ("rogzitett" if typ == "gk_goal_kick" else "kapus") if ev_team == team else None
    if ev_team == team:
        return None
    toward = _toward(ev, goal_x) if goal_x is not None else None
    if toward is False:
        return None
    if typ in ("shot", "goal"):
        return "lovesek"
    if typ == "cross":
        return "beadasok"
    if typ == "corner":
        return "rogzitett"
    if typ == "free_kick":
        if ev.get("dangerous") or ev.get("is_penalty"):
            return "rogzitett"
        if goal_x is not None and ev.get("distance_to_goal_m") is None:
            pos = ev.get("pos") or ev.get("from_xy")
            if pos is not None and abs(float(pos[0]) - goal_x) <= 35.0:
                return "rogzitett"
        return None
    if typ in ("pass", "pass_failed") and goal_x is not None and _in_box(ev.get("to_xy"), goal_x):
        return "beadasok"
    return None


def attacks_in_zone(data, goal_x, halves, attack_zone_m=30.0, min_attack_s=3.0, max_gap_s=1.5):
    """A labda a kapu előtti zónában (gólvonaltól attack_zone_m-en belül) legalább min_attack_s-ig."""
    ball = ((data.get("trajectories") or {}).get("ball")) or []
    out = []
    cur = None
    for p in sorted(ball, key=lambda q: q[2]):
        x, y, t = float(p[0]), float(p[1]), float(p[2])
        inside = abs(x - goal_x) <= attack_zone_m and -2.0 <= y <= PITCH_WID + 2.0
        if inside and cur is not None and t - cur["t1"] <= max_gap_s:
            cur["t1"] = t
            cur["deep"] = min(cur["deep"], abs(x - goal_x))
            cur["box"] = cur["box"] or _in_box((x, y), goal_x)
            continue
        if cur is not None and cur["t1"] - cur["t0"] >= min_attack_s:
            out.append(cur)
        cur = {"t0": t, "t1": t, "deep": abs(x - goal_x), "box": _in_box((x, y), goal_x)} if inside else None
    if cur is not None and cur["t1"] - cur["t0"] >= min_attack_s:
        out.append(cur)
    res = []
    for a in out:
        dur = a["t1"] - a["t0"]
        res.append({
            "type": "attack_in_zone", "time": round(a["t0"], 2),
            "start_sec": round(a["t0"] - 2.0, 2), "end_sec": round(a["t1"] + 2.0, 2),
            "duration_s": round(dur, 1), "closest_m": round(a["deep"], 1), "reached_box": a["box"],
            "half": _half_of(a["t0"], halves),
            "desc": f"Támadás a saját zónában – {dur:.0f} mp, a labda {a['deep']:.0f} m-re a gólvonaltól"
                    + (", a 16-oson belül is" if a["box"] else ""),
        })
    return res


def from_detector_logs(data, base_dir, bin_s=0.2):
    """
    Pontosabb kapus- és labdapályák a detektor-naplóból (detector_log.py), ha van: a kapusok képkockánkénti
    jelölése alapján (nem "valaha kapus volt"), bin_s-es átlagolással (a képkockánkénti remegés ne tűnjön
    mozgásnak). Visszatér: {"tracks": {"<félidő>:<id>": {"team", "keeper": True, "half", "points"}},
    "ball": [[x, y, t], ...]} vagy None, ha nincs napló.
    """
    try:
        import detector_log
    except ImportError:
        return None
    logs = detector_log.find_logs(data, base_dir) if base_dir else []
    if not logs:
        return None
    tracks, ball = {}, []
    for idx, (key, path) in enumerate(logs, 1):
        half = {"first_half": 1, "second_half": 2}.get(key, idx if len(logs) > 1 else None)
        _h, rows = detector_log.read(path)
        bins = {}
        for r in rows:
            if r[0] != "frame":
                continue
            t, b, players = r[1], r[2], r[5]
            if b is not None:
                ball.append([b[0], b[1], t])
            for p in players:
                if not p.get("is_keeper"):
                    continue
                k = f"{half or 0}:{p['track_id']}"
                slot = int(t / bin_s)
                acc = bins.setdefault(k, {}).setdefault(slot, [0.0, 0.0, 0.0, 0, p.get("team_id")])
                acc[0] += p["pitch_coords"][0]
                acc[1] += p["pitch_coords"][1]
                acc[2] += t
                acc[3] += 1
        for k, slots in bins.items():
            pts = [[a[0] / a[3], a[1] / a[3], a[2] / a[3]] for _, a in sorted(slots.items())]
            teams = [a[4] for a in slots.values() if a[4] in (1, 2)]
            tracks[k] = {"team": max(set(teams), key=teams.count) if teams else None, "keeper": True,
                         "half": half, "points": pts}
    return {"tracks": tracks, "ball": sorted(ball, key=lambda q: q[2])}


def keeper_tracks(data, team, goal_x, half=None, max_line_m=25.0):
    """A csapat kapus-követései: "keeper" jelzésű, és a pontok többsége a saját gólvonal közelében."""
    tracks = ((data.get("trajectories") or {}).get("tracks")) or {}
    num = _team_num(team)
    out = []
    for tid, tr in tracks.items():
        if not tr.get("keeper"):
            continue
        if half is not None and tr.get("half") not in (None, half):
            continue
        pts = tr.get("points") or []
        if len(pts) < 3:
            continue
        if goal_x is not None:
            near = sum(1 for p in pts if abs(float(p[0]) - goal_x) <= max_line_m)
            if near < 0.6 * len(pts):
                continue
        elif tr.get("team") != num:
            continue
        out.append((tid, sorted(pts, key=lambda q: q[2])))
    return out


def keeper_moves(points, min_move_m=2.0, max_gap_s=1.5, settle_s=0.8, settle_m=0.6, max_speed_ms=10.0):
    """
    Kapus-mozgások egy követés pontjaiból: egy mozgás egy nyugvópontból indul, és addig tart, amíg a kapus
    settle_s ideig settle_m-en belül nem marad; ha a teljes elmozdulás >= min_move_m, bejegyzés lesz belőle.
    """
    moves = []
    if len(points) < 2:
        return moves
    anchor = points[0]
    seg = None  # mozgásban: {"start": pont, "last": pont, "path": hossz, "vmax": m/s}
    still_since = None
    prev = points[0]
    for p in points[1:]:
        dt = float(p[2]) - float(prev[2])
        if dt <= 0:
            continue
        if dt > max_gap_s:
            if seg is not None:
                moves.append(seg)
            seg, anchor, still_since, prev = None, p, None, p
            continue
        step = math.hypot(float(p[0]) - float(prev[0]), float(p[1]) - float(prev[1]))
        if seg is None:
            if math.hypot(float(p[0]) - float(anchor[0]), float(p[1]) - float(anchor[1])) >= settle_m:
                # a mozgás az utolsó, még a nyugvóponton mért időpontból indul
                start = [anchor[0], anchor[1], prev[2]]
                seg = {"start": start, "last": p, "path": math.hypot(float(p[0]) - float(anchor[0]),
                                                                       float(p[1]) - float(anchor[1])),
                       "vmax": step / dt}
                still_since = None
            else:
                anchor = anchor if float(p[2]) - float(anchor[2]) < 3.0 else p
        else:
            seg["path"] += step
            seg["vmax"] = max(seg["vmax"], step / dt)
            if math.hypot(float(p[0]) - float(seg["last"][0]), float(p[1]) - float(seg["last"][1])) < settle_m:
                still_since = still_since or float(seg["last"][2])
                if float(p[2]) - still_since >= settle_s:
                    moves.append(seg)
                    seg, anchor, still_since = None, p, None
            else:
                seg["last"] = p
                still_since = None
        prev = p
    if seg is not None:
        moves.append(seg)
    out = []
    for m in moves:
        a, b = m["start"], m["last"]
        dist = math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        if dist < min_move_m or m["vmax"] > max_speed_ms:  # gyorsabb ugrás = követési hiba / ID-csere
            continue
        out.append({"t0": float(a[2]), "t1": float(b[2]), "from_xy": [round(float(a[0]), 1), round(float(a[1]), 1)],
                    "to_xy": [round(float(b[0]), 1), round(float(b[1]), 1)], "distance_m": round(dist, 1),
                    "path_m": round(m["path"], 1), "max_speed_ms": round(m["vmax"], 1)})
    return out


def _move_desc(m, goal_x):
    if goal_x is None:
        return f"Kapus mozgás – {m['distance_m']:.1f} m, max. {m['max_speed_ms']:.1f} m/s"
    d0 = abs(m["from_xy"][0] - goal_x)
    d1 = abs(m["to_xy"][0] - goal_x)
    if d1 - d0 >= 1.0:
        way = "kijjebb lép"
    elif d0 - d1 >= 1.0:
        way = "visszalép a kapuhoz"
    else:
        way = "oldalra mozog"
    return (f"Kapus mozgás – {way}, {m['distance_m']:.1f} m "
            f"({m['t1'] - m['t0']:.1f} mp, max. {m['max_speed_ms']:.1f} m/s), "
            f"{d1:.0f} m-re a gólvonaltól")


def build(data, team, min_move_m=2.0, attack_zone_m=30.0, min_attack_s=3.0, include_moves=True, base_dir=None):
    """
    A Kapus Modul listája egy meccs-eredményből a `team` kapusára. base_dir: az eredményfájl mappája – ha ott
    van detektor-napló, a kapus-mozgások és a zóna-támadások abból (teljes ütem, képkockánkénti kapus-jelölés).
    """
    if team not in ("team1", "team2"):
        raise ValueError("A csapat 'team1' vagy 'team2' lehet.")
    halves = _halves(data)
    goal_by_half = {idx: defended_goal_x(data, team, idx) for idx, _, _ in halves}
    precise = from_detector_logs(data, base_dir) if base_dir else None
    source = "detektor-napló" if precise else "mentett pályák"
    if precise:
        # a védett kapu továbbra is a mezőnyjátékosok átlagából (a mentett pályák), a kapus és a labda a naplóból
        data = dict(data, trajectories={"tracks": precise["tracks"], "ball": precise["ball"]})
    items = []
    for ev in data.get("events") or []:
        t = float(ev.get("time", 0.0))
        half = ev.get("half") if ev.get("half") is not None else _half_of(t, halves)
        group = _event_group(ev, team, goal_by_half.get(half))
        if group is None:
            continue
        it = dict(ev)
        it["group"] = group
        it.setdefault("half", half)
        items.append(it)
    for idx, a, b in halves:
        gx = goal_by_half.get(idx)
        if gx is None:
            continue
        sub = data if idx is None else {"trajectories": {
            "ball": [p for p in ((data.get("trajectories") or {}).get("ball") or []) if a <= p[2] <= b]}}
        for z in attacks_in_zone(sub, gx, halves, attack_zone_m, min_attack_s):
            z.update({"team": _other(team), "group": "zona", "goal_x": gx})
            items.append(z)
        if not include_moves:
            continue
        for tid, pts in keeper_tracks(data, team, gx, idx):
            pts = [p for p in pts if a <= p[2] <= b]
            for m in keeper_moves(pts, min_move_m=min_move_m):
                items.append({
                    "type": "gk_move", "time": round(m["t0"], 2),
                    "start_sec": round(m["t0"] - 2.0, 2), "end_sec": round(m["t1"] + 2.0, 2),
                    "team": team, "group": "mozgas", "keeper_track_id": tid, "half": idx, "goal_x": gx,
                    "from_xy": m["from_xy"], "to_xy": m["to_xy"], "distance_m": m["distance_m"],
                    "path_m": m["path_m"], "max_speed_ms": m["max_speed_ms"],
                    "desc": _move_desc(m, gx),
                })
    items.sort(key=lambda e: float(e.get("time", 0.0)))
    counts = {}
    for it in items:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
    return {
        "team": team, "team_label": TEAM_LABELS[team], "source": source,
        "defended_goal_x": {str(k): v for k, v in goal_by_half.items()},
        "groups": [{"key": k, "label": v, "count": counts.get(k, 0)} for k, v in GROUP_LABELS.items()],
        "items": items,
    }

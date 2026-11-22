"""
Edzői visszajelzés az automatikus eseményekre (a MAUI "Meccs elemzés" fülének ✓ / ✗ / "+ Hiányzó esemény"
gombjai). Egy meccs-JSON-hoz (match_events_*.json / events_*.json) egy feedback_<név>.json tartozik a
clips mappában:

    {"events": "/clips/match_events_ab12.json",
     "items":   [{"key": "shot@634.20", "type": "shot", "time": 634.2, "team": "team1", "verdict": "ok"|"hamis"}],
     "missing": [{"id": "k1a2b3", "type": "gk_long_ball", "time": 642.0, "team": "team2", "note": ""}],
     "updated": "2026-11-22T21:10:00"}

Az ítélet az esemény típusa + ideje (key) szerint tapad – így egy újraelemzés után is összevethető
(evaluate(): mennyi a valódi / téves / kimaradt esemény egy új eseménylistában, ±tol_s idő-tűréssel).
"""
import datetime
import json
import os
import re
import uuid

VERDICTS = ("ok", "hamis")


def event_key(ev):
    return f"{ev.get('type', '')}@{float(ev.get('time', 0.0)):.2f}"


def feedback_path(clips_dir, events_url):
    """A visszajelzés-fájl helye egy események-URL-hez (csak a clips mappán belüli fájlnév számít)."""
    name = os.path.basename(str(events_url or "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name):
        raise ValueError("Érvénytelen eseményfájl-név.")
    stem = name[:-5]
    return os.path.join(clips_dir, f"feedback_{stem}.json")


def load(clips_dir, events_url):
    path = feedback_path(clips_dir, events_url)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data.setdefault("events", events_url)
    data.setdefault("items", [])
    data.setdefault("missing", [])
    return data


def _save(clips_dir, events_url, data):
    path = feedback_path(clips_dir, events_url)
    data["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return data


def set_verdict(clips_dir, events_url, ev, verdict):
    """Egy automatikus esemény ítélete: "ok" (valódi), "hamis" (téves), vagy None / "" (törlés)."""
    if verdict not in VERDICTS + (None, ""):
        raise ValueError(f"Ismeretlen ítélet: {verdict!r} (lehet: ok, hamis)")
    data = load(clips_dir, events_url)
    key = event_key(ev)
    data["items"] = [it for it in data["items"] if it.get("key") != key]
    if verdict:
        data["items"].append({"key": key, "type": ev.get("type"), "time": round(float(ev.get("time", 0.0)), 2),
                              "team": ev.get("team"), "verdict": verdict})
    return _save(clips_dir, events_url, data)


def add_missing(clips_dir, events_url, typ, time_sec, team=None, note=""):
    """Az automatikából kimaradt esemény felvétele (a videó adott pillanatára)."""
    if not typ:
        raise ValueError("Hiányzik az esemény típusa.")
    data = load(clips_dir, events_url)
    item = {"id": uuid.uuid4().hex[:8], "type": str(typ), "time": round(float(time_sec), 2),
            "team": team or None, "note": str(note or "")[:200]}
    data["missing"].append(item)
    _save(clips_dir, events_url, data)
    return item


def remove_missing(clips_dir, events_url, item_id):
    data = load(clips_dir, events_url)
    data["missing"] = [m for m in data["missing"] if m.get("id") != item_id]
    return _save(clips_dir, events_url, data)


def evaluate(feedback, events, tol_s=3.0):
    """
    Egy (akár új elemzésből származó) eseménylista összevetése a visszajelzéssel.
    Igazság: az "ok" ítéletű események + a kézzel felvett hiányzók; téves: a "hamis" ítéletűek.
    Visszatér: {"talalt", "kimaradt", "teves_maradt", "teves_eltunt", "pontossag", "felidezes", "tipusonkent"}.
      talalt        – valódi eseményből ennyit megtalált (azonos típus, ±tol_s)
      kimaradt      – valódi eseményből ennyit nem talált meg
      teves_maradt  – a téves jelöltek közül ennyit most is kiad
      pontossag     – talalt / (talalt + teves_maradt)  (csak a megítélt eseményekre!)
      felidezes     – talalt / (talalt + kimaradt)
    """
    truth = [{"type": it["type"], "time": float(it["time"])} for it in feedback.get("items", [])
             if it.get("verdict") == "ok"]
    truth += [{"type": m["type"], "time": float(m["time"])} for m in feedback.get("missing", [])]
    false = [{"type": it["type"], "time": float(it["time"])} for it in feedback.get("items", [])
             if it.get("verdict") == "hamis"]
    evs = [{"type": e.get("type"), "time": float(e.get("time", 0.0))} for e in (events or [])]

    def matched(target, pool, used):
        best, bd = None, None
        for i, e in enumerate(pool):
            if i in used or e["type"] != target["type"]:
                continue
            d = abs(e["time"] - target["time"])
            if d <= tol_s and (bd is None or d < bd):
                best, bd = i, d
        if best is not None:
            used.add(best)
        return best is not None

    used = set()
    per = {}
    found = 0
    for t in truth:
        ok = matched(t, evs, used)
        found += ok
        p = per.setdefault(t["type"], {"talalt": 0, "kimaradt": 0, "teves_maradt": 0})
        p["talalt" if ok else "kimaradt"] += 1
    still_false = 0
    for t in false:
        if matched(t, evs, used):
            still_false += 1
            per.setdefault(t["type"], {"talalt": 0, "kimaradt": 0, "teves_maradt": 0})["teves_maradt"] += 1
    missed = len(truth) - found
    return {
        "talalt": found, "kimaradt": missed, "teves_maradt": still_false, "teves_eltunt": len(false) - still_false,
        "pontossag": round(found / (found + still_false), 3) if (found + still_false) else None,
        "felidezes": round(found / len(truth), 3) if truth else None,
        "tipusonkent": per,
    }

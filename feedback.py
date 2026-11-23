"""
Edzői visszajelzés az automatikus eseményekre (a MAUI "Meccs elemzés" fülének ✓ / ✗ / "+ Hiányzó esemény"
gombjai). Egy meccs-JSON-hoz (match_events_*.json / events_*.json) egy feedback_<név>.json tartozik a
clips mappában:

    {"events": "/clips/match_events_ab12.json",
     "items":   [{"key": "shot@634.20", "type": "shot", "time": 634.2, "team": "team1", "verdict": "ok"|"hamis"}],
     "missing": [{"id": "k1a2b3", "type": "gk_long_ball", "time": 642.0, "team": "team2", "note": ""}],
     "reviewed": [{"start": 692.1, "end": 701.1, "label": "semmi", "source": "cimkezo"}],
     "updated": "2026-11-22T21:10:00"}

A "fontos" jelző (ha van) az edző ítélete arról, hogy a jelenetet látni akarja-e a válogatásban – ebből tanulja
később a program, mit tart az edző lényegesnek. A "reviewed" a végignézett jelenetek ablaka (a címkéző csomag klipjei): a "semmi" címkéjű ablak erős negatív
minta – ami ott esemény, az téves –, ezért a hangolás (tools/hangolas.py) ezekből is tanul.

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
    data.setdefault("reviewed", [])
    return data


def _save(clips_dir, events_url, data):
    path = feedback_path(clips_dir, events_url)
    data["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return data


def set_verdict(clips_dir, events_url, ev, verdict, corrected_type=None, corrected_team=None, note=None,
                corrected_time=None, fontos=None):
    """
    Egy automatikus esemény ítélete: "ok" (valódi), "hamis" (téves), vagy None / "" (törlés).
    Téves eseménynél megadható, mi volt valójában: corrected_type (pl. "cross"; "semmi" = nem történt
    semmi), corrected_team, corrected_time (mp a videóban) – ezek valódi eseményként számítanak a mérésben.
    """
    if verdict not in VERDICTS + (None, ""):
        raise ValueError(f"Ismeretlen ítélet: {verdict!r} (lehet: ok, hamis)")
    data = load(clips_dir, events_url)
    key = event_key(ev)
    prev = next((it for it in data["items"] if it.get("key") == key), None)
    if fontos is None and prev is not None and "fontos" in prev:
        fontos = prev["fontos"]          # az újbóli ítélet nem törli a "fontos" jelölést
    data["items"] = [it for it in data["items"] if it.get("key") != key]
    if not verdict and fontos:
        # ítélet nélkül, csak "fontos" jelöléssel marad meg
        data["items"].append({"key": key, "type": ev.get("type"), "time": round(float(ev.get("time", 0.0)), 2),
                              "team": ev.get("team"), "verdict": None, "fontos": True})
    if verdict:
        item = {"key": key, "type": ev.get("type"), "time": round(float(ev.get("time", 0.0)), 2),
                "team": ev.get("team"), "verdict": verdict}
        if corrected_type:
            item["javitott_tipus"] = str(corrected_type)
        if corrected_team:
            item["javitott_csapat"] = str(corrected_team)
        if corrected_time is not None:
            item["javitott_ido"] = round(float(corrected_time), 2)
        if note:
            item["note"] = str(note)[:300]
        if fontos is not None:
            item["fontos"] = bool(fontos)
        data["items"].append(item)
    return _save(clips_dir, events_url, data)


def set_important(clips_dir, events_url, ev, fontos):
    """Egy esemény "fontos" jelölése (⭐) – az ítélettől függetlenül; a hiányzó (kézzel felvett) eseményre is."""
    data = load(clips_dir, events_url)
    if ev.get("id"):
        for m in data["missing"]:
            if m.get("id") == ev["id"]:
                m["fontos"] = bool(fontos)
                return _save(clips_dir, events_url, data)
        raise ValueError("Nincs ilyen kézzel felvett esemény.")
    key = event_key(ev)
    for it in data["items"]:
        if it.get("key") == key:
            it["fontos"] = bool(fontos)
            if it.get("verdict") is None and not fontos:
                data["items"] = [x for x in data["items"] if x is not it]
            return _save(clips_dir, events_url, data)
    if fontos:
        data["items"].append({"key": key, "type": ev.get("type"), "time": round(float(ev.get("time", 0.0)), 2),
                              "team": ev.get("team"), "verdict": None, "fontos": True})
    return _save(clips_dir, events_url, data)


def add_missing(clips_dir, events_url, typ, time_sec, team=None, note="", fontos=None):
    """Az automatikából kimaradt esemény felvétele (a videó adott pillanatára)."""
    if not typ:
        raise ValueError("Hiányzik az esemény típusa.")
    data = load(clips_dir, events_url)
    t = round(float(time_sec), 2)
    for m in data["missing"]:   # ugyanaz kétszer (pl. a címkék újbóli beolvasása): nem duplikáljuk
        if m.get("type") == str(typ) and abs(float(m.get("time", 0.0)) - t) < 0.05:
            m["team"] = team or m.get("team")
            if note:
                m["note"] = str(note)[:200]
            if fontos is not None:
                m["fontos"] = bool(fontos)
            _save(clips_dir, events_url, data)
            return m
    item = {"id": uuid.uuid4().hex[:8], "type": str(typ), "time": t,
            "team": team or None, "note": str(note or "")[:200]}
    if fontos is not None:
        item["fontos"] = bool(fontos)
    data["missing"].append(item)
    _save(clips_dir, events_url, data)
    return item


def add_reviewed(clips_dir, events_url, start, end, label=None, source="cimkezo", fontos=None):
    """Egy végignézett jelenet-ablak (mp a videóban) és a címkéje ("semmi" = nem történt semmi lényeges)."""
    data = load(clips_dir, events_url)
    a, b = round(float(start), 2), round(float(end), 2)
    if b <= a:
        raise ValueError("Az ablak vége a kezdete előtt van.")
    data["reviewed"] = [w for w in data["reviewed"]
                        if not (abs(float(w["start"]) - a) < 0.05 and abs(float(w["end"]) - b) < 0.05)]
    w = {"start": a, "end": b, "label": label, "source": source}
    if fontos is not None:
        w["fontos"] = bool(fontos)
    data["reviewed"].append(w)
    return _save(clips_dir, events_url, data)


def remove_missing(clips_dir, events_url, item_id):
    data = load(clips_dir, events_url)
    data["missing"] = [m for m in data["missing"] if m.get("id") != item_id]
    return _save(clips_dir, events_url, data)


def evaluate(feedback, events, tol_s=3.0):
    """
    Egy (akár új elemzésből származó) eseménylista összevetése a visszajelzéssel.
    Igazság: az "ok" ítéletű események + a kézzel felvett hiányzók; téves: a "hamis" ítéletűek.
    Visszatér: {"talalt", "kimaradt", "teves_maradt", "teves_eltunt", "teves_csendes", "pontossag", "felidezes",
                "pontszam", "tipusonkent"}.
      talalt        – valódi eseményből ennyit megtalált (azonos típus, ±tol_s)
      kimaradt      – valódi eseményből ennyit nem talált meg
      teves_maradt  – a téves jelöltek közül ennyit most is kiad
      teves_csendes – ennyi eseményt ad egy "semmi" címkéjű (végignézett, eseménytelen) ablakban
      pontossag     – talalt / (talalt + teves_maradt + teves_csendes)  (csak a megítélt eseményekre!)
      felidezes     – talalt / (talalt + kimaradt)
    """
    truth = [{"type": it["type"], "time": float(it["time"])} for it in feedback.get("items", [])
             if it.get("verdict") == "ok"]
    truth += [{"type": m["type"], "time": float(m["time"])} for m in feedback.get("missing", [])]
    # téves eseménynél megadott valódi típus ("ez valójában beadás volt") – valódi eseményként számít
    truth += [{"type": it["javitott_tipus"], "time": float(it.get("javitott_ido", it["time"]))}
              for it in feedback.get("items", [])
              if it.get("verdict") == "hamis" and it.get("javitott_tipus") not in (None, "", "semmi")]
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
    # "semmi" ablakok: ami itt (még ki nem párosított) esemény, az téves
    quiet = [(float(w["start"]), float(w["end"])) for w in feedback.get("reviewed", []) if w.get("label") == "semmi"]
    in_quiet = 0
    for i, e in enumerate(evs):
        if i in used:
            continue
        if any(a <= e["time"] <= b for a, b in quiet):
            in_quiet += 1
            used.add(i)
            per.setdefault(e["type"], {"talalt": 0, "kimaradt": 0, "teves_maradt": 0})["teves_maradt"] += 1
    missed = len(truth) - found
    fp = still_false + in_quiet
    return {
        "talalt": found, "kimaradt": missed, "teves_maradt": still_false, "teves_eltunt": len(false) - still_false,
        "teves_csendes": in_quiet,
        "pontossag": round(found / (found + fp), 3) if (found + fp) else None,
        "felidezes": round(found / len(truth), 3) if truth else None,
        # összesített pontszám (F1): a hangolás ezt maximalizálja
        "pontszam": round(2 * found / (2 * found + fp + missed), 4) if (found + fp + missed) else None,
        "tipusonkent": per,
    }

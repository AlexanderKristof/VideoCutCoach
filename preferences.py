"""
Az edző ízlése: mit jelöl "fontosnak" (⭐) – ebből a program megtanulja, mely jeleneteket ajánlja előre.

Tanítóadat: az összes visszajelzés-fájl (feedback_*.json a clips mappában) és a hozzá tartozó eseménylista.
  * pozitív: az edző ⭐-gal jelölte (fontos: true);
  * negatív: az edző megnézte (✓ ítélet, vagy a címkéző csomagban végignézte), de nem jelölte fontosnak
    (fontos: false, vagy ✓ ítélet ⭐ nélkül).

Modell: naiv Bayes-szerű log-esély összeg az esemény jellemzőire (típus, kalap, térfél, hosszú / előre ívelt
passz, 16-osba érkező, veszélyes, kapura tartó …), Laplace-simítással – kevés jelölésből is értelmes, és
átlátható: minden jellemzőnél látszik, mennyit nyom a latban (explain()). A pontszám 0..1 ("ajánlottság").

Használat:
    model = preferences.train(clips_dir)            # vagy train_from_pairs([(events, feedback), ...])
    preferences.score_events(model, events)         # -> [0..1] eseményenként
"""
import json
import math
import os

FEATURE_FIELDS = ("zone", "is_long", "progressive", "into_box", "dangerous", "on_target", "completed", "is_penalty",
                  "side", "by_keeper")
MIN_POSITIVES = 3   # ennél kevesebb ⭐ esetén nincs modell (nincs mit ajánlani)
MIN_NEGATIVES = 3   # ... és ennyi "megnézte, de nem ⭐" példa is kell (különben minden "fontos" lenne)


def _cat(ev):
    try:
        import events as events_mod
        return events_mod.event_category(ev.get("type", ""))
    except Exception:
        return ev.get("category") or "egyeb"


def features(ev):
    """Az esemény diszkrét jellemzői: {"type=pass", "kalap=passz", "zone=ellenfel_terfel", "progressive=1", ...}."""
    out = {f"type={ev.get('type', '')}", f"kalap={ev.get('category') or _cat(ev)}"}
    for k in FEATURE_FIELDS:
        v = ev.get(k)
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out.add(f"{k}={v}")
    d = ev.get("distance_m")
    if isinstance(d, (int, float)):
        out.add("tav=" + ("rovid" if d < 15 else "kozepes" if d < 30 else "hosszu"))
    return out


def _key(ev):
    return f"{ev.get('type', '')}@{float(ev.get('time', 0.0)):.2f}"


def labelled_examples(events, feedback, tol_s=1.0):
    """[(esemény, címke 1/0)] egy meccsre: a ⭐ pozitív, a megnézett de nem ⭐ negatív."""
    evs = list(events or [])
    by_key = {_key(e): e for e in evs}
    out = []
    seen = set()

    def find(typ, t):
        e = by_key.get(f"{typ}@{float(t):.2f}")
        if e is not None:
            return e
        best = None
        for x in evs:
            if x.get("type") == typ and abs(float(x.get("time", 0.0)) - float(t)) <= tol_s:
                if best is None or abs(float(x["time"]) - float(t)) < abs(float(best["time"]) - float(t)):
                    best = x
        return best

    for it in feedback.get("items", []):
        ev = find(it.get("type"), it.get("time", 0.0)) or {"type": it.get("type"), "time": it.get("time")}
        k = _key(ev)
        if it.get("fontos") is True:
            out.append((ev, 1))
            seen.add(k)
        elif it.get("verdict") == "ok" and it.get("fontos") is not True:
            out.append((ev, 0))
            seen.add(k)
    for m in feedback.get("missing", []):
        if m.get("fontos") is True:
            out.append(({"type": m.get("type"), "time": m.get("time")}, 1))
    # a címkéző csomag végignézett ablakai: az ottani (nem ⭐) események negatívak, a ⭐ ablakbeliek pozitívak
    for w in feedback.get("reviewed", []):
        if w.get("fontos") is None:
            continue
        a, b = float(w["start"]), float(w["end"])
        for e in evs:
            k = _key(e)
            if k in seen or not (a <= float(e.get("time", 0.0)) <= b):
                continue
            if w.get("label") not in (None, "semmi") and e.get("type") != w.get("label"):
                continue
            out.append((e, 1 if w["fontos"] else 0))
            seen.add(k)
    return out


def train_from_pairs(pairs, alpha=1.0):
    """pairs: [(események, visszajelzés)] -> modell dict (vagy None, ha kevés a ⭐)."""
    pos, neg = {}, {}
    n_pos = n_neg = 0
    for events, feedback in pairs:
        for ev, y in labelled_examples(events, feedback):
            fs = features(ev)
            if y:
                n_pos += 1
                for f in fs:
                    pos[f] = pos.get(f, 0) + 1
            else:
                n_neg += 1
                for f in fs:
                    neg[f] = neg.get(f, 0) + 1
    if n_pos < MIN_POSITIVES or n_neg < MIN_NEGATIVES:
        return None
    vocab = set(pos) | set(neg)
    weights = {}
    for f in vocab:
        p = (pos.get(f, 0) + alpha) / (n_pos + 2 * alpha)
        q = (neg.get(f, 0) + alpha) / (n_neg + 2 * alpha)
        weights[f] = round(math.log(p / q), 4)
    prior = math.log((n_pos + alpha) / (n_neg + alpha))
    return {"version": 1, "prior": round(prior, 4), "weights": weights, "n_pos": n_pos, "n_neg": n_neg}


def score_event(model, ev):
    if not model:
        return None
    z = model["prior"] + sum(model["weights"].get(f, 0.0) for f in features(ev))
    z = max(-30.0, min(30.0, z))
    return round(1.0 / (1.0 + math.exp(-z)), 4)


def score_events(model, events):
    return [score_event(model, e) for e in events or []]


def explain(model, ev, top=4):
    """A pontszámot leginkább befolyásoló jellemzők: [(jellemző, súly)] – a kliens "miért ajánlja" szövegéhez."""
    if not model:
        return []
    ws = [(f, model["weights"][f]) for f in features(ev) if f in model["weights"]]
    ws.sort(key=lambda x: -abs(x[1]))
    return ws[:top]


def train(clips_dir):
    """A clips mappa összes visszajelzéséből (feedback_<meccs>.json + <meccs>.json)."""
    pairs = []
    for name in sorted(os.listdir(clips_dir)):
        if not (name.startswith("feedback_") and name.endswith(".json")):
            continue
        mpath = os.path.join(clips_dir, name[len("feedback_"):])
        if not os.path.isfile(mpath):
            continue
        try:
            with open(os.path.join(clips_dir, name), encoding="utf-8") as f:
                fbk = json.load(f)
            with open(mpath, encoding="utf-8") as f:
                evs = (json.load(f) or {}).get("events") or []
        except (OSError, ValueError):
            continue
        fbk.setdefault("items", [])
        fbk.setdefault("missing", [])
        fbk.setdefault("reviewed", [])
        pairs.append((evs, fbk))
    return train_from_pairs(pairs)

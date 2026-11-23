"""
Az edző ízlése (preferences.py): a ⭐ jelölésekből tanult ajánlottság.
Futtatás: python3 tests/test_preferences.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import preferences as pf  # noqa: E402


def _match(offset=0.0):
    evs = []
    t = offset
    for i in range(12):   # előre ívelt passzok az ellenfél térfelén ...
        evs.append({"type": "pass", "time": t, "zone": "ellenfel_terfel", "progressive": True, "distance_m": 25})
        t += 10
    for i in range(12):   # ... és oldalpasszok a saját térfélen
        evs.append({"type": "pass", "time": t, "zone": "sajat_terfel", "progressive": False, "distance_m": 8})
        t += 10
    evs.append({"type": "shot", "time": t, "on_target": True})
    return evs


def _feedback(evs):
    """Az edző: a progresszív passzokat és a lövést ⭐-ozza, az oldalpasszokat csak ✓-olja."""
    items = []
    for e in evs:
        star = e["type"] == "shot" or e.get("progressive")
        items.append({"key": f"{e['type']}@{e['time']:.2f}", "type": e["type"], "time": e["time"],
                      "verdict": "ok", "fontos": bool(star)})
    return {"items": items, "missing": [], "reviewed": []}


def test_learns_preference_and_ranks_new_match():
    evs = _match()
    model = pf.train_from_pairs([(evs, _feedback(evs))])
    assert model and model["n_pos"] == 13 and model["n_neg"] == 12
    new = [{"type": "pass", "time": 1, "zone": "ellenfel_terfel", "progressive": True, "distance_m": 22},
           {"type": "pass", "time": 2, "zone": "sajat_terfel", "progressive": False, "distance_m": 6},
           {"type": "shot", "time": 3, "on_target": True}]
    s = pf.score_events(model, new)
    assert s[0] > 0.8 and s[1] < 0.2 and s[2] > 0.5, s
    why = dict(pf.explain(model, new[0]))
    assert why.get("progressive=1", 0) > 0 and why.get("zone=ellenfel_terfel", 0) > 0


def test_too_few_stars_gives_no_model():
    evs = _match()
    fbk = _feedback(evs)
    for it in fbk["items"][2:]:
        it["fontos"] = False
    assert pf.train_from_pairs([(evs, fbk)]) is None
    assert pf.score_event(None, evs[0]) is None
    # csak ⭐, "nem fontos" példa nélkül: nincs modell (különben minden ajánlott lenne)
    fbk = _feedback(evs)
    for it in fbk["items"]:
        if not it["fontos"]:
            it["verdict"] = None
    assert pf.train_from_pairs([(evs, fbk)]) is None


def test_reviewed_windows_and_clips_dir():
    d = tempfile.mkdtemp()
    try:
        evs = _match()
        fbk = {"items": [], "missing": [],
               "reviewed": [{"start": e["time"] - 1, "end": e["time"] + 1, "label": e["type"],
                             "fontos": bool(e.get("progressive") or e["type"] == "shot")} for e in evs]}
        with open(os.path.join(d, "m.json"), "w", encoding="utf-8") as f:
            json.dump({"events": evs}, f)
        with open(os.path.join(d, "feedback_m.json"), "w", encoding="utf-8") as f:
            json.dump(fbk, f)
        model = pf.train(d)
        assert model and model["n_pos"] == 13 and model["n_neg"] == 12
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)

"""
Jelenet-címkéző csomag (tools/cimkezo_csomag.py): időablakok, véletlen jelenetek, a címkék beolvasása.
Futtatás: python3 tests/test_cimkezo.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import cimkezo_csomag as cc  # noqa: E402
import feedback as fb  # noqa: E402


def test_clip_window():
    assert cc._clip_window({"time": 100.0}, 5, 4) == (95.0, 104.0)
    s, e = cc._clip_window({"time": 100.0, "start_sec": 60.0, "end_sec": 130.0}, 5, 4)
    assert e - s <= 15.0 + 1e-6 and s < 100.0 < e
    assert cc._clip_window({"time": 1.0}, 5, 4)[0] == 0.0


def test_random_times_avoid_events():
    halves = {"first_half": {"start_sec": 0.0, "end_sec": 600.0}}
    ts = cc.random_times(halves, [{"time": 300.0}], 10, 5, 4, seed=1)
    assert len(ts) == 10 and all(abs(t - 300.0) > 9 for t in ts) and all(5 <= t <= 596 for t in ts)


def test_import_labels():
    d = tempfile.mkdtemp()
    try:
        lab = {"meccs": "match_events_x", "events_url": "/clips/match_events_x.json", "klipek": [
            {"id": 1, "kind": "esemeny", "type": "shot", "time": 50.0, "start": 45.0, "team": "team1",
             "cimke": {"itelet": "ok"}},
            {"id": 2, "kind": "esemeny", "type": "gk_sweeper", "time": 80.0, "start": 75.0, "team": "team2",
             "cimke": {"itelet": "hamis", "tipus": "gk_long_ball", "csapat": "team1", "pillanat": 3.5}},
            {"id": 3, "kind": "minta", "type": "?", "time": 120.0, "start": 115.0,
             "cimke": {"itelet": "minta", "tipus": "cross", "csapat": "team2", "pillanat": 2.0, "megjegyzes": "jobb szél"}},
            {"id": 4, "kind": "minta", "type": "?", "time": 200.0, "start": 195.0, "cimke": {"itelet": "minta", "tipus": "semmi"}},
            {"id": 5, "kind": "esemeny", "type": "pass", "time": 250.0, "start": 245.0, "cimke": None},
        ]}
        p = os.path.join(d, "cimkek.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(lab, f)
        assert cc.import_labels(p, d, log=lambda *_: None) == (1, 1, 1)
        data = fb.load(d, lab["events_url"])
        bad = [i for i in data["items"] if i["verdict"] == "hamis"][0]
        assert bad["javitott_tipus"] == "gk_long_ball" and bad["javitott_ido"] == 78.5 and bad["javitott_csapat"] == "team1"
        assert data["missing"][0]["type"] == "cross" and data["missing"][0]["time"] == 117.0
        # a mérés a javított típust valódi eseményként kezeli
        r = fb.evaluate(data, [{"type": "gk_long_ball", "time": 79.0}, {"type": "shot", "time": 50.5}])
        assert r["talalt"] == 2 and r["kimaradt"] == 1, r
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        try:
            t()
            ok += 1
            print(f"OK    {t.__name__}")
        except Exception as ex:
            print(f"HIBA  {t.__name__}: {ex!r}")
    print(f"{ok}/{len(tests)} teszt sikeres")
    sys.exit(0 if ok == len(tests) else 1)

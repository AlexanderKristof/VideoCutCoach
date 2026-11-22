"""
Edzői visszajelzés (feedback.py): ítéletek mentése, hiányzó események, összevetés egy új eseménylistával.
Futtatás: python3 tests/test_feedback.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import feedback as fb  # noqa: E402

URL = "/clips/match_events_ab12.json"


def test_verdicts_and_missing_roundtrip():
    d = tempfile.mkdtemp()
    try:
        shot = {"type": "shot", "time": 634.2, "team": "team1"}
        sweep = {"type": "gk_sweeper", "time": 642.0, "team": "team2"}
        fb.set_verdict(d, URL, shot, "ok")
        fb.set_verdict(d, URL, sweep, "hamis")
        fb.set_verdict(d, URL, sweep, "ok")          # felülírás: egy eseménynek egy ítélete van
        item = fb.add_missing(d, URL, "gk_long_ball", 642.3, "team2", "kirúgás kézből")
        data = fb.load(d, URL)
        assert len(data["items"]) == 2 and {i["verdict"] for i in data["items"]} == {"ok"}
        assert data["missing"][0]["type"] == "gk_long_ball" and data["missing"][0]["note"] == "kirúgás kézből"
        fb.remove_missing(d, URL, item["id"])
        fb.set_verdict(d, URL, shot, None)           # ítélet törlése
        data = fb.load(d, URL)
        assert not data["missing"] and len(data["items"]) == 1
        assert os.path.exists(os.path.join(d, "feedback_match_events_ab12.json"))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_rejects_path_tricks():
    for bad in ("../../etc/passwd", "/clips/x.txt", "a b.json"):
        try:
            fb.feedback_path("/tmp", bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_evaluate_new_event_list():
    feedback = {
        "items": [{"type": "shot", "time": 100.0, "verdict": "ok"},
                  {"type": "goal", "time": 101.0, "verdict": "hamis"},
                  {"type": "gk_sweeper", "time": 300.0, "verdict": "hamis"}],
        "missing": [{"type": "gk_long_ball", "time": 200.0}],
    }
    new_events = [{"type": "shot", "time": 101.5},        # a valódi lövés (±3 mp)
                  {"type": "goal", "time": 101.0},        # a téves gól még mindig ott van
                  {"type": "gk_long_ball", "time": 199.0}]  # a korábban kimaradt kirúgás most megvan
    r = fb.evaluate(feedback, new_events)
    assert r["talalt"] == 2 and r["kimaradt"] == 0 and r["teves_maradt"] == 1 and r["teves_eltunt"] == 1, r
    assert r["felidezes"] == 1.0 and abs(r["pontossag"] - 2 / 3) < 0.01, r
    assert r["tipusonkent"]["goal"]["teves_maradt"] == 1


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

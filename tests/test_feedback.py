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


def test_quiet_windows_count_as_false_and_dedupe():
    d = tempfile.mkdtemp()
    try:
        fb.add_missing(d, URL, "throw_in", 744.36)
        fb.add_missing(d, URL, "throw_in", 744.37, "team1")      # újra-beolvasás: nem duplikál
        fb.add_reviewed(d, URL, 692.0, 701.0, "semmi")
        fb.add_reviewed(d, URL, 692.0, 701.0, "semmi")           # ugyanaz az ablak egyszer
        fb.add_reviewed(d, URL, 740.0, 749.0, "throw_in")
        data = fb.load(d, URL)
        assert len(data["missing"]) == 1 and data["missing"][0]["team"] == "team1"
        assert len(data["reviewed"]) == 2
        evs = [{"type": "pass", "time": 695.0},       # "semmi" ablakban: téves
               {"type": "throw_in", "time": 745.0},   # valódi
               {"type": "pass", "time": 746.0},       # nem "semmi" ablak: nem számít
               {"type": "pass", "time": 800.0}]       # nem nézett rész: nem számít
        r = fb.evaluate(data, evs)
        assert r["talalt"] == 1 and r["teves_csendes"] == 1 and r["kimaradt"] == 0, r
        assert abs(r["pontossag"] - 0.5) < 1e-6 and abs(r["pontszam"] - 2 / 3) < 1e-3, r
        try:
            fb.add_reviewed(d, URL, 10.0, 5.0)
            raise AssertionError("fordított ablakra hibát várunk")
        except ValueError:
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_important_flag_survives_verdicts():
    d = tempfile.mkdtemp()
    try:
        ev = {"type": "corner", "time": 300.0, "team": "team1"}
        fb.set_important(d, URL, ev, True)                      # ítélet nélkül is megmarad
        data = fb.load(d, URL)
        assert data["items"][0]["fontos"] is True and data["items"][0]["verdict"] is None
        assert fb.evaluate(data, [])["talalt"] == 0              # a mérést nem befolyásolja
        fb.set_verdict(d, URL, ev, "ok")                         # az ítélet nem törli a jelölést
        assert fb.load(d, URL)["items"][0]["fontos"] is True
        fb.set_verdict(d, URL, ev, None)                         # ítélet visszavonása: a jelölés marad
        it = fb.load(d, URL)["items"]
        assert len(it) == 1 and it[0]["verdict"] is None and it[0]["fontos"] is True
        fb.set_important(d, URL, ev, False)                      # se ítélet, se jelölés: törlődik
        assert fb.load(d, URL)["items"] == []
        m = fb.add_missing(d, URL, "cross", 400.0)
        fb.set_important(d, URL, {"id": m["id"]}, True)
        assert fb.load(d, URL)["missing"][0]["fontos"] is True
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

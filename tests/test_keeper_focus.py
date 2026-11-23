"""
Kapus Modul (keeper_focus.py): a kapust érintő jelenetek gyűjtése egy meccs-eredményből.
Futtatás: python3 tests/test_keeper_focus.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import keeper_focus as kf  # noqa: E402


def _match():
    """1. félidő 0–100 mp: team1 a bal (x=0) kaput védi; team2 kapusa a jobb kapuban."""
    tracks = {
        # mezőnyjátékosok: team1 hátrébb (bal oldalon), team2 jobbra
        "1": {"team": 1, "keeper": False, "half": 1, "points": [[30.0, 30.0, t] for t in range(0, 100, 2)]},
        "2": {"team": 2, "keeper": False, "half": 1, "points": [[70.0, 30.0, t] for t in range(0, 100, 2)]},
        # team1 kapusa: áll, majd 20–23 mp között 4 m-t kilép, utána áll; 60 mp-nél egy 15 m-es "ugrás" (hiba)
        "6": {"team": 1, "keeper": True, "half": 1, "points":
              [[3.0, 34.0, t / 5] for t in range(0, 100)]
              + [[3.0 + (t - 100) * 0.27, 34.0, t / 5] for t in range(100, 115)]
              + [[7.0, 34.0, t / 5] for t in range(115, 300)]
              + [[22.0, 34.0, 60.2], [22.0, 34.0, 60.4], [22.0, 34.0, 60.6], [22.0, 34.0, 61.6]]},
        # team2 kapusa a jobb kapuban
        "9": {"team": 2, "keeper": True, "half": 1, "points": [[102.0, 34.0, t / 5] for t in range(0, 300)]},
    }
    # labda: 40–48 mp között a bal kapu előtt (team1 zónája)
    ball = [[50.0, 34.0, t / 10] for t in range(0, 400)] + [[12.0, 30.0, 40 + t / 10] for t in range(0, 80)] \
        + [[50.0, 34.0, 48 + t / 10] for t in range(0, 300)]
    events = [
        {"type": "shot", "time": 45.0, "team": "team2", "goal_x": 0.0, "half": 1},
        {"type": "shot", "time": 70.0, "team": "team1", "goal_x": 105.0, "half": 1},
        {"type": "gk_save", "time": 45.5, "team": "team1", "goal_x": 0.0, "half": 1},
        {"type": "corner", "time": 50.0, "team": "team2", "goal_x": 0.0, "half": 1},
        {"type": "cross", "time": 55.0, "team": "team2", "attack_goal_x": 0.0, "half": 1},
        {"type": "pass", "time": 56.0, "team": "team2", "to_xy": [8.0, 30.0], "half": 1},
        {"type": "pass", "time": 57.0, "team": "team2", "to_xy": [40.0, 30.0], "half": 1},
        {"type": "free_kick", "time": 58.0, "team": "team2", "dangerous": True, "half": 1},
        {"type": "free_kick", "time": 59.0, "team": "team2", "dangerous": False, "distance_to_goal_m": 60, "half": 1},
        {"type": "gk_positioning_error", "time": 61.0, "team": "team1", "half": 1},
        {"type": "gk_long_ball", "time": 80.0, "team": "team2", "goal_x": 105.0, "half": 1},
    ]
    return {"events": events, "halves": {"first_half": {"start_sec": 0, "end_sec": 100}},
            "trajectories": {"hz": 5.0, "tracks": tracks, "ball_hz": 10.0, "ball": ball}}


def test_defended_goal_from_player_positions():
    m = _match()
    assert kf.defended_goal_x(m, "team1", 1) == 0.0
    assert kf.defended_goal_x(m, "team2", 1) == 105.0


def test_keeper_relevant_events_for_team1():
    r = kf.build(_match(), "team1")
    by = [(i["type"], i["group"], i["time"]) for i in r["items"] if i["type"] != "gk_move"]
    types = [b[0] for b in by]
    assert ("shot", "lovesek", 45.0) in by          # az ellenfél lövése a mi kapunkra
    assert ("shot", "lovesek", 70.0) not in by      # a saját lövésünk nem a kapusunkat érinti
    assert ("gk_save", "kapus", 45.5) in by
    assert ("corner", "rogzitett", 50.0) in by
    assert ("cross", "beadasok", 55.0) in by
    assert ("pass", "beadasok", 56.0) in by         # 16-osba érkező passz
    assert ("pass", "beadasok", 57.0) not in by
    assert ("free_kick", "rogzitett", 58.0) in by and ("free_kick", "rogzitett", 59.0) not in by
    assert "gk_positioning_error" not in types      # véleményt nem mutatunk
    assert "gk_long_ball" not in types              # a másik kapus eseménye
    zones = [i for i in r["items"] if i["group"] == "zona"]
    assert len(zones) == 1 and 39.0 <= zones[0]["time"] <= 41.0 and zones[0]["reached_box"]
    counts = {g["key"]: g["count"] for g in r["groups"]}
    assert counts["lovesek"] == 1 and counts["zona"] == 1


def test_keeper_moves_over_2m_and_jumps_skipped():
    r = kf.build(_match(), "team1")
    moves = [i for i in r["items"] if i["type"] == "gk_move"]
    assert len(moves) == 1, moves
    mv = moves[0]
    assert 19.5 <= mv["time"] <= 20.5 and 3.5 <= mv["distance_m"] <= 4.5 and mv["keeper_track_id"] == "6"
    assert "kijjebb" in mv["desc"]
    r2 = kf.build(_match(), "team1", min_move_m=5.0)
    assert not [i for i in r2["items"] if i["type"] == "gk_move"]
    r3 = kf.build(_match(), "team1", include_moves=False)
    assert not [i for i in r3["items"] if i["type"] == "gk_move"]


def test_team2_view_and_bad_team():
    r = kf.build(_match(), "team2")
    types = [(i["type"], i["time"]) for i in r["items"]]
    assert ("shot", 70.0) in types and ("gk_long_ball", 80.0) in types and ("shot", 45.0) not in types
    try:
        kf.build(_match(), "team3")
        raise AssertionError("hibás csapatra hibát várunk")
    except ValueError:
        pass


def test_moves_from_detector_log_full_rate():
    """25 Hz-es napló, zajos kapus-pozíció: a remegés nem mozgás, a valódi 5 m-es kilépés igen."""
    import random
    import shutil
    import tempfile
    import detector_log
    tmp = tempfile.mkdtemp(prefix="test_kf_")
    try:
        rng = random.Random(1)
        path = os.path.join(tmp, "events_k" + detector_log.SUFFIX)
        w = detector_log.DetectorLogWriter(path, header={"start_sec": 0.0, "end_sec": 60.0})
        for i in range(60 * 25):
            t = i / 25
            kx = 3.0 + (min(max(t - 30.0, 0.0), 2.0) * 2.5)        # 30-32 mp: 5 m kilépés, utána áll
            ps = [{"track_id": 6, "team_id": 1, "is_keeper": True,
                   "pitch_coords": (kx + rng.gauss(0, 0.15), 34.0 + rng.gauss(0, 0.15))},
                  {"track_id": 1, "team_id": 1, "is_keeper": False, "pitch_coords": (30.0, 30.0)},
                  {"track_id": 2, "team_id": 2, "is_keeper": False, "pitch_coords": (70.0, 30.0)}]
            w.frame(t, (50.0, 34.0), ps)
        w.close()
        m = _match()
        m["halves"] = {"first_half": {"start_sec": 0, "end_sec": 60}}
        m["detector_logs"] = {"first_half": {"path": path}}
        m["events"] = []
        r = kf.build(m, "team1", base_dir=tmp)
        assert r["source"] == "detektor-napló"
        moves = [i for i in r["items"] if i["type"] == "gk_move"]
        assert len(moves) == 1, [(x["time"], x["distance_m"], x["max_speed_ms"]) for x in moves]
        assert 29.0 <= moves[0]["time"] <= 30.5 and 4.0 <= moves[0]["distance_m"] <= 6.0, moves[0]
        assert moves[0]["keeper_track_id"] == "1:6"
        # napló nélkül a mentett pályákból
        assert kf.build(_match(), "team1", base_dir=tmp)["source"] == "mentett pályák"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)

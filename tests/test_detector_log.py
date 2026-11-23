"""
Detektor-napló (detector_log.py): a rögzített detektor-bemenet visszajátszása ugyanazt az eseménylistát adja,
mint az eredeti futás; a csonka (megszakított) napló is olvasható.
Futtatás: python3 tests/test_detector_log.py
"""
import gzip
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import detector_log  # noqa: E402
import events  # noqa: E402
import eval_events as ev  # noqa: E402


def _record(frames, path, t1a=True):
    w = detector_log.DetectorLogWriter(path, header={"start_sec": frames[0][0], "end_sec": frames[-1][0],
                                                     "team1_attacks_right": t1a})
    for t, ball, players in frames:
        w.frame(t, ball, players, ball_status="d" if ball is not None else None, ball_conf=0.9)
    w.close()
    return w


def _key(evs):
    return [(e["type"], round(e["time"], 1), e.get("team")) for e in evs]


def test_replay_matches_direct_run():
    tmp = tempfile.mkdtemp(prefix="test_detlog_")
    try:
        gt = ev.generate_gt(6, 120, tmp)
        frames = ev.degrade(gt, dict(ev.NOISE_PRESETS["kozepes"]), seed=1)
        # a napló 1 cm-re (0,01 m) kerekít – az összevetés is a kerekített bemenettel fut
        frames = [(round(t, 3), None if b is None else (round(b[0], 2), round(b[1], 2)),
                   [dict(p, pitch_coords=(round(p["pitch_coords"][0], 2), round(p["pitch_coords"][1], 2)))
                    for p in ps]) for t, b, ps in frames]
        t1a = bool((gt.get("periods") or [{}])[0].get("team1_attacks_right", True))
        direct = ev.run_detector(events, gt, frames)
        path = os.path.join(tmp, "events_x" + detector_log.SUFFIX)
        w = _record(frames, path, t1a)
        assert w.frames == len(frames) and w.error is None
        replayed, stats, header = detector_log.replay(path)
        assert header["team1_attacks_right"] == t1a
        assert len(direct) > 5, len(direct)
        assert _key(sorted(direct, key=lambda e: e["time"])) == _key(replayed)
        size = os.path.getsize(path)
        assert size / len(frames) < 400, size / len(frames)   # tömör: kb. 100-200 bájt / képkocka
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_truncated_log_is_readable_and_paths():
    tmp = tempfile.mkdtemp(prefix="test_detlog_")
    try:
        path = os.path.join(tmp, "events_y" + detector_log.SUFFIX)
        frames = [(i / 10, (50.0 + i * 0.1, 34.0), [{"track_id": 3, "team_id": 1, "is_keeper": False,
                                                     "pitch_coords": (50.0, 34.0)}]) for i in range(5000)]
        _record(frames, path)
        raw = open(path, "rb").read()
        with open(path, "wb") as f:
            f.write(raw[: len(raw) * 2 // 3])          # megszakadt elemzés: csonka gzip
        header, rows = detector_log.read(path)
        assert header.get("kind") == "detlog"
        assert 0 < len(rows) < 5000
        # nem létező / nem gzip fájl: üres eredmény, nem kivétel
        bad = os.path.join(tmp, "rossz" + detector_log.SUFFIX)
        with open(bad, "w") as f:
            f.write("nem gzip")
        assert detector_log.read(bad) == ({}, [])
        assert detector_log.log_path_for("/a/events_ab.json") == "/a/events_ab" + detector_log.SUFFIX
        data = {"detector_logs": {"first_half": {"path": path, "url": "/clips/x"},
                                  "second_half": {"path": "/nincs/ilyen.gz"}}}
        assert detector_log.find_logs(data) == [("first_half", path)]
        assert detector_log.find_logs({"detector_log": "/clips/" + os.path.basename(path)}, tmp) == [(None, path)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_writer_survives_unwritable_path():
    w = detector_log.DetectorLogWriter("/nincs/ilyen/mappa/x.gz")
    assert not w.active and w.error
    w.frame(1.0, (1, 2), [])
    w.close()


def test_engine_hook_records_frames():
    tmp = tempfile.mkdtemp(prefix="test_detlog_")
    try:
        import numpy as np
        from tactical_engine import TacticalEngine
        eng = TacticalEngine()
        path = os.path.join(tmp, "e" + detector_log.SUFFIX)
        eng.recorder = detector_log.DetectorLogWriter(path, header={"end_sec": 2.0})
        H = np.eye(3)
        for i in range(20):
            dets = [{"track_id": 1, "team_id": 2, "team_known": True, "is_keeper": False,
                     "screen_foot": (30.0 + i * 0.1, 20.0)}]
            eng.update_tracks(i, i / 10, dets, (40.0, 30.0), H, ball_status="d", ball_conf=0.8)
        eng.set_attack_direction(False, timestamp=2.0)
        eng.recorder.close()
        header, rows = detector_log.read(path)
        frames = [r for r in rows if r[0] == "frame"]
        assert len(frames) == 20 and frames[0][2] == (40.0, 30.0) and frames[0][3] == "d"
        assert frames[0][5][0]["team_id"] == 2 and rows[-1] == ("dir", 2.0, False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)

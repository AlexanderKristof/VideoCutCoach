"""
Hangolás a címkékből (tools/hangolas.py): elrontott kezdő küszöbökből a címkék alapján visszatalál egy
jobb beállításhoz; a hangolt fájlt a motor (events.load_tuning) beolvassa.
Futtatás: python3 tests/test_hangolas.py
"""
import json
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
import hangolas  # noqa: E402


def _synthetic_match(tmp, seed=6, duration=180):
    """Szintetikus meccs detektor-naplóval + "edzői" visszajelzés a valódi (GT) passzokból / lövésekből."""
    gt = ev.generate_gt(seed, duration, tmp)
    frames = ev.degrade(gt, dict(ev.NOISE_PRESETS["kozepes"]), seed=3)
    t1a = bool((gt.get("periods") or [{}])[0].get("team1_attacks_right", True))
    log = os.path.join(tmp, "events_s" + detector_log.SUFFIX)
    w = detector_log.DetectorLogWriter(log, header={"start_sec": frames[0][0], "end_sec": frames[-1][0],
                                                    "team1_attacks_right": t1a})
    for t, b, ps in frames:
        w.frame(t, b, ps)
    w.close()
    mpath = os.path.join(tmp, "events_s.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({"events": [], "detector_log": "/clips/" + os.path.basename(log)}, f)
    truth = [e for e in gt["events"] if e["type"] in ("pass", "shot")]
    fbk = {"items": [], "missing": [{"type": e["type"], "time": e["time"], "team": e.get("team")} for e in truth],
           "reviewed": []}
    with open(os.path.join(tmp, "feedback_events_s.json"), "w", encoding="utf-8") as f:
        json.dump(fbk, f)
    return mpath, len(truth)


def test_tuner_recovers_from_bad_start():
    tmp = tempfile.mkdtemp(prefix="test_hangolas_")
    try:
        mpath, n_truth = _synthetic_match(tmp)
        assert n_truth >= 10, n_truth
        m = hangolas.load_match(mpath)
        assert m["segments"] and m["segments"][0]["source"] == "napló"
        m["pieces"] = hangolas.crop_windows(m, 90.0, 20.0)
        bad = {"pass_min_m": 14.0, "ctl_radius_m": 1.0}     # szándékosan rossz: a passzok nagy része elveszik
        hangolas.SEARCH["pass_min_m"] = ([4.0, 6.0, 8.0, 14.0], "pass")
        hangolas.SEARCH["ctl_radius_m"] = ([1.0, 1.5, 2.0, 2.5], "pass")
        logs = []
        final, base, best, changes = hangolas.tune([m], start=bad, passes=2, tol_s=2.0, guard=None,
                                                   log=logs.append, params=["pass_min_m", "ctl_radius_m"])
        assert best["f1"] > base["f1"] + 0.1, (base, best)
        assert best["tp"] > base["tp"] and changes
        assert final.get("pass_min_m", events.EventConfig().pass_min_m) <= 8.0, final
        # a default-tól eltérő értékek csak
        for k, v in final.items():
            assert getattr(events.EventConfig(), k) != v
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_change_without_net_gain():
    tmp = tempfile.mkdtemp(prefix="test_hangolas_")
    try:
        mpath, _ = _synthetic_match(tmp, duration=120)
        m = hangolas.load_match(mpath)
        # üres igazság: semmit nem lehet javítani -> marad az alap
        m["feedback"] = {"items": [], "missing": [], "reviewed": []}
        m["pieces"] = hangolas.crop_windows(m, 90.0, 20.0)
        final, base, best, changes = hangolas.tune([m], start={}, passes=1, tol_s=2.0, guard=None,
                                                   log=lambda *_: None, params=["pass_min_m"])
        assert final == {} and changes == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_tuning_file():
    tmp = tempfile.mkdtemp(prefix="test_hangolas_")
    try:
        p = os.path.join(tmp, "esemeny_hangolas.json")
        with open(p, "w", encoding="utf-8-sig") as f:   # BOM-mal is (PowerShell)
            json.dump({"overrides": {"pass_min_m": 5, "ctl_min_samples": 2.0, "nincs_ilyen": 3,
                                     "team1_attacks_right": False, "camera_pos": "x"}}, f)
        t = events.load_tuning(p)
        assert t == {"pass_min_m": 5.0, "ctl_min_samples": 2}, t
        assert isinstance(t["ctl_min_samples"], int)
        cfg = events.EventConfig(**t)
        assert cfg.pass_min_m == 5.0
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"enabled": False, "overrides": {"pass_min_m": 5}}, f)
        assert events.load_tuning(p) == {}
        assert events.load_tuning(os.path.join(tmp, "nincs.json")) == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)

"""
Hőtérkép-modul tesztjei. Futtatás: python3 tests/test_heatmap.py
(pytest nem kell; torch/ultralytics nélkül fut, a main.py-t nem importálja)
"""
import os
import sys
import json
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

import heatmap as hm


def _line_track(x0, y0, vx, vy, t0, t1, fps=25.0):
    n = int(round((t1 - t0) * fps))
    return [(x0 + vx * i / fps, y0 + vy * i / fps, t0 + i / fps) for i in range(n)]


def test_sample_weights_caps_gaps():
    times = [0.0, 0.04, 0.08, 5.0, 5.04]
    w = hm.sample_weights(times, max_gap_sec=1.0)
    # A 4.92 mp-es kimaradás nem számít teljes hosszal, csak tipikus lépésközzel
    assert abs(w.sum() - 0.2) < 1e-6, w
    assert abs(w[2] - 0.04) < 1e-9


def test_grid_and_normalize():
    pts = [(10.2, 20.7, t * 0.04) for t in range(100)]
    arr = hm._prepare(pts)
    grid = hm.occupancy_grid(arr, hm.sample_weights(arr[:, 2]))
    assert grid.shape == (68, 105)
    assert abs(grid.sum() - 4.0) < 1e-6
    assert grid[20, 10] == grid.max()
    heat = hm.smooth_normalize(grid)
    assert abs(heat.max() - 1.0) < 1e-6 and heat.min() >= 0
    iy, ix = np.unravel_index(np.argmax(heat), heat.shape)
    assert (iy, ix) == (20, 10)


def test_flip_second_half():
    first = [(20.0, 30.0, t * 0.04) for t in range(250)]
    second = [(20.0, 30.0, 100 + t * 0.04) for t in range(250)]
    heat, total = hm.build_heatmap([first + second], flip_at_sec=50.0)
    assert abs(total - 20.0) < 0.1
    # A második félidő x-tükrözve: 105 - 20 = 85 m
    assert heat[30, 20] > 0.9 and heat[30, 85] > 0.9


def test_player_stats_distance_and_jitter():
    # 4 m/s 30 mp-ig = 120 m
    pts = _line_track(10, 34, 4.0, 0.0, 0.0, 30.0)
    st = hm.player_stats(pts)
    assert abs(st["distance_m"] - 120.0) < 1.5, st
    assert abs(st["time_sec"] - 30.0) < 0.1, st
    assert abs(st["avg_x"] - 70.0) < 1.0 and abs(st["avg_y"] - 34.0) < 0.1
    # Egyetlen 40 m-es ugrás (azonosító-csere / téves vetítés) nem számít bele
    jumped = list(pts)
    t_mid = jumped[400][2]
    jumped[400] = (jumped[400][0] + 40.0, 34.0, t_mid)
    st2 = hm.player_stats(jumped)
    assert abs(st2["distance_m"] - 120.0) < 3.0, st2


def test_compact_and_meta():
    col = hm.TrackMetaCollector()
    for _ in range(5):
        col.add([{"track_id": 7, "team_id": 2, "is_keeper": False}])
    col.add([{"track_id": 7, "team_id": 1, "is_keeper": True}])
    meta = col.summary()
    assert meta[7] == {"team": 2, "keeper": True}

    traj = {7: _line_track(1.234, 2.345, 1.0, 0.0, 0.0, 2.0)}
    comp = hm.compact_trajectories(traj, meta, hz=5)
    json.dumps(comp)
    pts = comp["tracks"]["7"]["points"]
    assert 9 <= len(pts) <= 11, len(pts)
    assert pts[0][:2] == [1.2, 2.3]
    t2, m2 = hm.load_trajectories({"trajectories": comp})
    assert list(t2) == [7] and m2[7]["team"] == 2 and m2[7]["keeper"] is True


def _synthetic_events():
    rng = np.random.default_rng(1)
    traj, meta = {}, {}
    # Csapat 1: bal térfél, csapat 2: jobb térfél, 5-5 játékos, 60 mp, 25 fps
    for tid in range(1, 11):
        team = 1 if tid <= 5 else 2
        cx = 25.0 if team == 1 else 80.0
        cy = 10.0 + (tid % 5) * 12.0
        t = np.arange(0, 60, 0.04)
        x = cx + 8 * np.sin(t / 7 + tid) + rng.normal(0, 0.2, len(t))
        y = cy + 4 * np.cos(t / 5 + tid) + rng.normal(0, 0.2, len(t))
        traj[tid] = list(zip(x.tolist(), y.tolist(), t.tolist()))
        meta[tid] = {"team": team, "keeper": tid in (1, 6)}
    traj[99] = [(50.0, 34.0, 0.0), (50.5, 34.0, 0.04)]  # túl rövid
    meta[99] = {"team": 1, "keeper": False}
    return {"events": [], "match_stats": {}, "trajectories": hm.compact_trajectories(traj, meta)}


def test_generate_heatmaps_end_to_end():
    out = tempfile.mkdtemp(prefix="heatmap_test_")
    path = os.path.join(out, "events_test.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_synthetic_events(), f)
    res = hm.generate_heatmaps_from_file(path, out)
    assert res["status"] == "success", res
    assert set(res["team_heatmaps"]) == {"team1", "team2"}
    assert len(res["player_heatmaps"]) == 10  # a #99 túl rövid
    assert len(res["player_stats"]) == 11
    json.dumps(res)
    for url in list(res["team_heatmaps"].values()) + [p["url"] for p in res["player_heatmaps"]]:
        img = cv2.imread(os.path.join(out, url.split("/")[-1]))
        assert img is not None and img.shape[1] == 840
    # Csapat 1 hőtérképe a bal térfélen meleg, a jobb szélen a fű látszik
    img = cv2.imread(os.path.join(out, res["team_heatmaps"]["team1"].split("/")[-1]))
    body = img[34:]
    h, w = body.shape[:2]
    assert np.abs(body[h // 2 - 5, int(w * 0.9)].astype(int) - [60, 130, 60]).sum() < 5
    assert np.abs(body[h // 2 - 5, int(w * 0.24)].astype(int) - [60, 130, 60]).sum() > 40

    one = hm.generate_heatmaps_from_file(path, out, team="2", player_id=6, flip_at_sec=30.0)
    assert one["status"] == "success" and len(one["player_heatmaps"]) == 1
    assert hm.generate_heatmaps_from_file(path, out, player_id=12345)["status"] == "error"
    assert hm.generate_heatmaps_from_file(path, out, team="3")["status"] == "error"
    assert hm.generate_heatmaps_from_file(os.path.join(out, "nincs.json"), out)["status"] == "error"
    print(f"   (minta PNG-k: {out})")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Exception as ex:
            failed += 1
            print(f"HIBA  {name}: {ex!r}")
    print(f"{len(tests) - failed}/{len(tests)} teszt sikeres")
    sys.exit(1 if failed else 0)

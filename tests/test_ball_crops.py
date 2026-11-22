"""
Labda-kivágás (ball_crops.py), időmérő (stage_timer.py), annotált videó-író (video_writer.py) és a
kalibráció nélküli néző-szűrő (pitch_filters.BoundaryCrowdFilter) tesztjei.
Futtatás: python3 tests/test_ball_crops.py   (torch / ultralytics nélkül)
"""
import os
import sys
import shutil
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np

from ball_crops import BallCropSearch
from stage_timer import StageTimer, format_timing


class _State:
    def __init__(self, x, y, vx=0.0, vy=0.0, found=True):
        self.x, self.y, self.vx, self.vy, self.found = x, y, vx, vy, found


def test_tiles_cover_lower_frame():
    s = BallCropSearch((1920, 1080), crop=640, tile_every=2)
    assert all(t[2] - t[0] == 640 and t[3] - t[1] == 640 for t in s.tiles)
    assert all(0 <= t[0] and t[2] <= 1920 and 0 <= t[1] and t[3] <= 1080 for t in s.tiles)
    # a kép alsó 70%-ának minden pontját lefedi legalább egy csempe
    for x in range(0, 1920, 37):
        for y in range(int(1080 * 0.3), 1080, 29):
            assert any(t[0] <= x < t[2] and t[1] <= y < t[3] for t in s.tiles), (x, y)


def test_plan_tracks_predicted_ball_and_clamps():
    s = BallCropSearch((1920, 1080), crop=640, track_every=2)
    w = s.plan(10, _State(100.0, 50.0, vx=5.0))
    assert w == [(0, 0, 640, 640)]                       # a kép szélére tolva
    w = s.plan(11, _State(1000.0, 600.0, vx=10.0, vy=-4.0))
    assert w == [(690, 276, 1330, 916)]                  # a következő képkockára becsült hely köré
    # ha a teljes képes menet megtalálta, csak minden 2. képkockán
    assert s.plan(13, _State(1000.0, 600.0), full_frame_hit=True) == []
    assert len(s.plan(14, _State(1000.0, 600.0), full_frame_hit=True)) == 1


def test_lost_ball_round_robin_near_last_position():
    s = BallCropSearch((1920, 1080), crop=640, tile_every=3)
    s.plan(0, _State(1700.0, 900.0))
    seen = []
    for f in range(1, 1 + 3 * len(s.tiles)):
        w = s.plan(f, None)
        if w:
            seen.append(w[0])
    # az első csempe a legutóbbi ismert hely közelében
    t0 = seen[0]
    assert t0[0] <= 1700 < t0[2] and t0[1] <= 900 < t0[3], t0
    # tile_every képkockánként egy csempe, és mind sorra kerül
    assert len(seen) == len(s.tiles)
    assert set(seen) == set(s.tiles)


def test_run_maps_back_and_merges():
    s = BallCropSearch((1920, 1080), crop=640)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    windows = [(600, 300, 1240, 940)]

    def detect(crops):
        assert crops[0].shape[:2] == (640, 640)
        return [[((100.0, 200.0, 108.0, 208.0), 0.4)]]
    boxes = s.run(frame, windows, detect)
    assert boxes == [((700.0, 500.0, 708.0, 508.0), 0.4)]
    full = [((701.0, 501.0, 709.0, 509.0), 0.2), ((10.0, 10.0, 14.0, 14.0), 0.5)]
    merged = s.merge(full, boxes, count_conf=0.35)
    assert len(merged) == 2 and ((700.0, 500.0, 708.0, 508.0), 0.4) in merged   # duplikátum: a jobb marad
    assert s.stats["kivágás-találat"] == 1 and s.stats["teljes-kép-találat"] == 1


def test_fake_detector_resolution_dependent_ball():
    from fake_detector import FakeYOLO
    gt = {"width": 1920, "height": 1080, "people": [],
          "frames": [{"objs": [], "ball": {"vis": True, "bbox": [995.0, 595.0, 1005.0, 605.0], "r": 4.0, "uv": [1000, 600]}}
                     for _ in range(200)]}
    prof = {"infer_imgsz": 640, "ball_miss": 0.05, "ball_fp_rate": 0.0}
    fy = FakeYOLO(gt, seed=1, profile=prof)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    full_hits = crop_hits = 0
    for _ in range(200):
        res = fy.track(frame)
        full_hits += sum(1 for b in res[0].boxes if int(b.cls[0]) == 0)
        crop = frame[300:940, 680:1320]
        cr = fy.predict([crop], classes=[0], conf=0.0)[0]
        for b in cr.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            assert abs((x1 + x2) / 2 + 680 - 1000) < 5 and abs((y1 + y2) / 2 + 300 - 600) < 5
            crop_hits += 1
    print(f"  ál-detektor, 8 px-es labda 1080p-n: teljes kép (640) {full_hits}/200, natív kivágás {crop_hits}/200")
    assert full_hits < 40 and crop_hits > 150, (full_hits, crop_hits)
    # felbontás-modell nélkül a régi viselkedés (a teljes képen is látszik)
    fy2 = FakeYOLO(gt, seed=1, profile={"ball_miss": 0.05, "ball_fp_rate": 0.0})
    hits2 = sum(1 for _ in range(100) for b in fy2.track(frame)[0].boxes if int(b.cls[0]) == 0)
    assert hits2 > 70, hits2


def test_stage_timer():
    t = StageTimer()
    import time
    with t("detect"):
        time.sleep(0.02)
    with t("encode"):
        time.sleep(0.01)
    t.frames = 2
    t.stop()
    s = t.summary()
    assert s["frames"] == 2 and "detect" in s["stages"] and "encode" in s["stages"]
    assert abs(sum(v["share_pct"] for v in s["stages"].values()) - 100.0) < 1.0
    assert "kép/mp" in t.one_line() and "kódolás" in t.table_text()
    assert "ms/képkocka" in format_timing(s)


def test_video_writer_scale():
    ff = shutil.which("ffmpeg")
    if not ff:
        print("  (nincs ffmpeg – kihagyva)")
        return
    import video_writer
    d = tempfile.mkdtemp(prefix="vw_")
    try:
        out = os.path.join(d, "o.mp4")
        w = video_writer.AnnotatedVideoWriter(out, 640, 360, 25, output_scale=0.5, ffmpeg_bin=ff)
        for i in range(30):
            w.write(np.full((360, 640, 3), i * 5, np.uint8))
        assert w.close() is None
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames",
                                "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
        print(f"  író: {w.describe()} -> {probe}")
        assert probe.startswith("320,180"), probe
        assert video_writer.output_size(1920, 1080, 0.5) == (960, 540)
        assert video_writer.output_size(1919, 1079, 1.0) == (1920, 1080)
        assert video_writer.encoder_args("h264_nvenc", 1920, 1080)[:2] == ["-c:v", "h264_nvenc"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_boundary_crowd_filter():
    from pitch_filters import BoundaryCrowdFilter, grass_top_profile
    H, W = 1080, 1920
    mask = np.zeros((H // 2, W // 2), np.uint8)
    mask[200:] = 255                       # a fű felső széle: 400 px (teljes képen)
    prof = grass_top_profile(mask)
    assert abs(float(np.nanmedian(prof)) - 200) < 1
    f = BoundaryCrowdFilter(25, window_sec=2.0)
    spectator = player = mover = 0
    for i in range(100):
        f.set_frame(i, mask, 0.5, H)
        # álló néző a kerítés előtt (a talppont 10 px-re a fű szélétől), a kamera lassan fordul
        cam = 0.8 * i
        spectator += f.update(i, 500 - cam, 410, 500.0, 410.0, 50)
        # a pálya közepén álló játékos: nem a sávban
        player += f.update(i, 900 - cam, 700, 900.0, 700.0, 80)
        # partjelző / bedobó a sávban, de mozog (2 px / képkocka)
        mover += f.update(i, 1200 + 2 * i - cam, 415, 1200.0 + 2 * i, 415.0, 50)
    print(f"  néző-szűrő: álló néző kiszűrve {spectator}/100, játékos {player}/100, mozgó partjelző {mover}/100")
    assert spectator > 40 and player == 0 and mover == 0


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK   {name}")
        except Exception as ex:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"HIBA {name}: {ex!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} teszt sikeres")
    sys.exit(1 if failed else 0)

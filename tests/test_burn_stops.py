"""
A rajzos videóexport (video_annotate.py) és a /videos lista tesztjei.
Futtatás: python3 tests/test_burn_stops.py   (pytest és fastapi nélkül fut)
Szintetikus (lavfi testsrc2 + szinusz hang) videón ellenőrzi a kimerevítést, a rajzolást és a hosszakat.
Ha nincs ffmpeg a gépen, az ffmpeg-et igénylő tesztek kimaradnak.
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Linuxon nincs ffmpeg.exe – a tesztekhez a PATH-on lévő ffmpeg-et használjuk
if "FFMPEG_PATH" not in os.environ and shutil.which("ffmpeg"):
    os.environ["FFMPEG_PATH"] = shutil.which("ffmpeg")

import video_annotate as va
import boards

FFMPEG_OK = bool(os.environ.get("FFMPEG_PATH")) and shutil.which(os.environ["FFMPEG_PATH"]) is not None
FPS = 25
W, H = 320, 240
ARROW = {"type": "arrow", "pts": [[0.1, 0.5], [0.9, 0.5]], "color": "#4da3ff"}


class Skip(Exception):
    pass


def _need_ffmpeg():
    if not FFMPEG_OK:
        raise Skip("nincs ffmpeg")


def _make_video(path, seconds=4, audio=True):
    cmd = [os.environ["FFMPEG_PATH"], "-y", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc2=size={W}x{H}:rate={FPS}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)


def _frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def _decoded_duration(path):
    """A videó hossza a ténylegesen dekódolt képkockákból (mp)."""
    return len(_frames(path)) / FPS


def test_burn_two_stops_duration_and_pixels():
    _need_ffmpeg()
    d = tempfile.mkdtemp(prefix="burn_test_")
    try:
        src = os.path.join(d, "be.mp4")
        _make_video(src)
        stops = [
            {"t": 1.0, "hold_sec": 2.0, "note": "Árvíztűrő tükörfúrógép",
             "items": [ARROW,
                       {"type": "circle", "pts": [[0.5, 0.3], [0.6, 0.3]], "color": "#ff4d4d"},
                       {"type": "zone", "pts": [[0.6, 0.6], [0.9, 0.9]], "color": "#4da3ff"},
                       {"type": "text", "pts": [[0.5, 0.85]], "text": "Ő ű", "color": "#ffffff"}]},
            {"t": 2.5, "hold_sec": 1.0,
             "items": [{"type": "run", "pts": [[0.1, 0.9], [0.5, 0.7]], "color": "#ff4d4d"},
                       {"type": "line", "pts": [[0.2, 0.2], [0.8, 0.2]], "color": "#111111"},
                       {"type": "free", "pts": [[0.1, 0.1], [0.2, 0.15], [0.3, 0.1]], "color": "#ffffff"},
                       {"type": "spot", "pts": [[0.3, 0.6], [0.37, 0.6]], "color": "#ffffff"}]},
        ]
        out = os.path.join(d, "ki.mp4")
        calls = []
        res = va.burn_stops(src, stops, out, progress_cb=lambda a, b, c: calls.append((a, b)))
        assert res["status"] == "success" and os.path.exists(out)
        assert res["stops"] == 2 and abs(res["hold_total_sec"] - 3.0) < 1e-6, res
        assert calls and calls[-1][0] == calls[-1][1], calls
        frames_out, frames_in = _frames(out), _frames(src)
        expected = len(frames_in) + 3 * FPS
        assert abs(len(frames_out) - expected) <= 1, (len(frames_out), expected)
        # a kimerevített szakasz: az 1. megálló (25. kocka) után 2 mp-ig ugyanaz a rajzos kép
        k = 25
        y, x = int(0.5 * H), int(0.35 * W)            # a nyíl közepe
        orig = frames_in[k][y - 1:y + 2, x - 1:x + 2].astype(int)
        for j in (k, k + 10, k + 2 * FPS - 1):
            got = frames_out[j][y - 1:y + 2, x - 1:x + 2].astype(int)
            assert np.abs(got - orig).mean() > 40, (j, got.mean(axis=(0, 1)), orig.mean(axis=(0, 1)))
            # kék nyíl (#4da3ff) a sárga háttéren: a kék csatorna jóval erősebb (a vékony vonal a
            # színalmintavételezés miatt elmosódik, ezért csak az irányt nézzük)
            b, g, r = frames_out[j][y, x].astype(int)
            assert b > int(frames_in[k][y, x][0]) + 100, (j, b, g, r)
        # a kimerevítés után a videó a megálló képkockájától folytatódik (rajz nélkül)
        after = frames_out[k + 2 * FPS + 2][y, x].astype(int)
        assert np.abs(after - frames_in[k + 2][y, x].astype(int)).mean() < 40
        # a kiemelésnél (2. megálló) a körön kívül sötétebb a kép
        k2 = int(2.5 * FPS) + 2 * FPS                  # kimeneti sorszám: 62 + 50 (1. kitartás)
        dark = frames_out[k2 + 5].astype(int).mean()
        assert dark < frames_in[62].astype(int).mean() - 15, (dark, frames_in[62].mean())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_burn_range_only():
    _need_ffmpeg()
    d = tempfile.mkdtemp(prefix="burn_range_")
    try:
        src = os.path.join(d, "be.mp4")
        _make_video(src, seconds=5, audio=False)
        stops = [{"t": 1.0, "items": [ARROW]}, {"t": 4.5, "items": [ARROW]}]   # a 2. a részleten kívül
        out = os.path.join(d, "ki.mp4")
        res = va.burn_stops(src, stops, out, start_sec=0.6, end_sec=2.2)
        assert res["stops"] == 1, res
        n = len(_frames(out))
        expected = (int(2.2 * FPS) - int(0.6 * FPS)) + 3 * FPS                  # 40 kocka + 3 mp kimerevítés
        assert abs(n - expected) <= 1, (n, expected)
        # megálló nélküli részlet is működik (sima képkocka-pontos vágás)
        out2 = os.path.join(d, "ki2.mp4")
        res2 = va.burn_stops(src, [], out2, start_sec=3.0, end_sec=4.0)
        assert res2["stops"] == 0 and abs(len(_frames(out2)) - FPS) <= 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_normalize_stops():
    plan = va.normalize_stops([
        {"t": 2.0, "items": [ARROW], "hold_sec": 100},
        {"t": 0.5, "items": [{"type": "ismeretlen", "pts": [[0, 0], [1, 1]]}], "note": "csak cím"},
        {"t": 2.01, "items": [{"type": "text", "pts": [[0.5, 0.5]], "text": "x", "color": "rossz"}], "hold_sec": 2},
        {"t": 3.0, "items": []},                                             # üres: kimarad
    ], fps=25, frames=100)
    assert [p["frame"] for p in plan] == [12, 50], plan
    assert plan[0]["items"] == [] and plan[0]["note"] == "csak cím"
    merged = plan[1]
    assert len(merged["items"]) == 2 and merged["hold_sec"] == va.MAX_HOLD_SEC
    assert merged["items"][1]["color"] == "#ffd400"                         # hibás szín → alapszín
    # az utolsó képkocka után: a végére igazítva
    assert va.normalize_stops([{"t": 99, "items": [ARROW]}], 25, 100)[0]["frame"] == 99
    for bad in ([], None, [{"t": -1, "items": [ARROW]}], [{"t": "x", "items": [ARROW]}], [{"t": 1, "items": []}], ["x"]):
        try:
            va.normalize_stops(bad, 25)
        except ValueError as ex:
            assert str(ex)
        else:
            raise AssertionError(f"ValueError várt: {bad!r}")


def test_render_overlay_both_backends():
    img = np.zeros((180, 320, 3), np.uint8) + 60
    items = [ARROW, {"type": "zone", "pts": [[0.6, 0.6], [0.9, 0.9]], "color": "#4da3ff"},
             {"type": "text", "pts": [[0.3, 0.2]], "text": "Gól!", "color": "#ffffff"}]
    a = va.render_overlay(img, items, note="Megjegyzés")
    b = va._render_cv(img, [va._clean_item(i) for i in items], "Megjegyzés")
    for out in (a, b):
        assert out.shape == img.shape and out is not img
        y, x = 90, 110
        assert out[y, x, 0] > 200 and out[y, x, 2] < 130                  # kék nyíl
        assert out[135, 240, 0] > img[135, 240, 0] + 20                   # kékes zóna
        assert out[150, 50].tolist() == img[150, 50].tolist()             # rajzon kívül változatlan
    assert (img == 60).all()                                              # az eredeti kép érintetlen
    assert va.strip_accents("Árvíztűrő tükörfúrógép") == "Arvizturo tukorfurogep"


def test_list_videos_and_resolve():
    d = tempfile.mkdtemp(prefix="videos_test_")
    try:
        for name in ("raw_ab12.mp4", "match_xy.mp4", "temp_cut_zz.mp4", "jegyzet.txt", "ures.mp4"):
            with open(os.path.join(d, name), "wb") as f:
                f.write(b"" if name == "ures.mp4" else b"\x00" * 64)
        for name in ("events_ab12.json", "match_events_run1.json", "egyeb.json"):
            with open(os.path.join(d, name), "w") as f:
                json.dump({}, f)
        os.makedirs(os.path.join(d, "boards"))
        os.utime(os.path.join(d, "match_xy.mp4"), (1, 1))                 # régebbi → a lista végén
        res = va.list_videos(d, probe=False)
        names = [v["name"] for v in res["videos"]]
        assert names == ["raw_ab12.mp4", "match_xy.mp4"], names
        raw = res["videos"][0]
        assert raw["url"] == "/clips/raw_ab12.mp4" and raw["kind"] == "raw" and raw["size"] == 64
        assert raw["events_url"] == "/clips/events_ab12.json" and "events_url" not in res["videos"][1]
        assert sorted(e["name"] for e in res["events"]) == ["events_ab12.json", "match_events_run1.json"]
        assert va.list_videos(os.path.join(d, "nincs"))["videos"] == []
        # URL → útvonal: csak a clips mappán belüli .mp4
        assert va.resolve_clip_url("/clips/raw_ab12.mp4?x=1", d) == os.path.join(os.path.abspath(d), "raw_ab12.mp4")
        for bad in ("/clips/../main.py", "/clips/%2E%2E%2Fx.mp4", "/clips/sub/a.mp4", "/clips/jegyzet.txt",
                    "/clips/nincs.mp4", "/egyeb/raw_ab12.mp4", "", None, "/clips/.rejtett.mp4"):
            try:
                va.resolve_clip_url(bad, d)
            except ValueError as ex:
                assert str(ex)
            else:
                raise AssertionError(f"ValueError várt: {bad!r}")
        if FFMPEG_OK:
            _make_video(os.path.join(d, "valodi.mp4"), seconds=2, audio=False)
            v = [x for x in va.list_videos(d)["videos"] if x["name"] == "valodi.mp4"][0]
            assert abs(v["fps"] - FPS) < 0.01 and abs(v["duration_sec"] - 2.0) < 0.05 and v["width"] == W, v
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_board_store_video_kind():
    d = tempfile.mkdtemp(prefix="boards_video_")
    try:
        st = boards.BoardStore(os.path.join(d, "boards"))
        data = {"kind": "video", "video_url": "/clips/raw_ab12.mp4", "video_name": "raw_ab12.mp4",
                "stops": [{"t": 1.04, "items": [ARROW], "hold_sec": 3, "note": "x"}]}
        meta = st.save("Videó elemzés", data)
        assert meta["kind"] == "video"                                     # a data.kind alapján
        lst = st.list(kind="video")
        assert len(lst) == 1 and lst[0]["video_name"] == "raw_ab12.mp4" and lst[0]["video_url"] == "/clips/raw_ab12.mp4"
        assert st.get(meta["id"])["data"]["stops"][0]["note"] == "x"
        try:
            st.save("rossz", {"kind": "video"}, kind="video")
        except boards.BoardError as ex:
            assert ex.status == 400
        else:
            raise AssertionError("BoardError várt")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Skip as ex:
            skipped += 1
            print(f"KIMARAD {name}: {ex}")
        except Exception as ex:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"HIBA  {name}: {ex!r}")
    print(f"{len(tests) - failed - skipped}/{len(tests)} teszt sikeres, {skipped} kimaradt")
    sys.exit(1 if failed else 0)

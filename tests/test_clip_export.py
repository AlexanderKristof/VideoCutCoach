"""
Klipexport tesztjei. Futtatás: python3 tests/test_clip_export.py
Szintetikus (lavfi testsrc + szinusz hang) videón ellenőrzi a vágást és az összefűzést.
Ha nincs ffmpeg a gépen, az ffmpeg-et igénylő tesztek kimaradnak.
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Linuxon nincs ffmpeg.exe – a tesztekhez a PATH-on lévő ffmpeg-et használjuk
if "FFMPEG_PATH" not in os.environ:
    found = shutil.which("ffmpeg")
    if not found:
        try:
            import imageio_ffmpeg  # csak teszteléshez, nem projektfüggőség
            found = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            found = None
    if found:
        os.environ["FFMPEG_PATH"] = found

import clip_export as ce

FFMPEG_OK = bool(os.environ.get("FFMPEG_PATH")) and shutil.which(os.environ["FFMPEG_PATH"]) is not None
FFPROBE = shutil.which("ffprobe")
FPS = 25


class Skip(Exception):
    pass


def _need_ffmpeg():
    if not FFMPEG_OK:
        raise Skip("nincs ffmpeg")


def _duration(path):
    """Klip hossza mp-ben (ffprobe, vagy OpenCV képkockaszám)."""
    if FFPROBE:
        out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                              "-of", "json", path], capture_output=True, text=True).stdout
        return float(json.loads(out)["format"]["duration"])
    import cv2
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / FPS


def _has_audio(path):
    if not FFPROBE:
        return True
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return bool(out.strip())


_VIDEO = None


def _source_video():
    """60 mp-es szintetikus videó hanggal, 2 mp-es kulcskocka-távolsággal."""
    global _VIDEO
    if _VIDEO:
        return _VIDEO
    d = tempfile.mkdtemp(prefix="clip_export_src_")
    path = os.path.join(d, "forras meccs.mp4")  # szóköz az útvonalban szándékosan
    subprocess.run([os.environ["FFMPEG_PATH"], "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc=size=320x240:rate={FPS}:duration=60",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", str(FPS * 2), "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", path], check=True, capture_output=True)
    _VIDEO = path
    return path


# ---- tiszta logika (ffmpeg nélkül) ------------------------------------------

def test_parse_and_merge_windows():
    items = [{"time": 10}, {"start_sec": 12, "end_sec": 16}, {"time": 30}, {"start_sec": 39.5, "end_sec": 45},
             {"time": 2}]
    w = ce.parse_windows(items, 5, 4, video_duration=60)
    assert w == [(5, 14), (12, 16), (25, 34), (39.5, 45), (0.0, 6)], w
    merged = ce.merge_windows(w)
    # 5-14 és 12-16 átfed, 0-6 és 5-16 átfed → 0-16; 34 és 39.5 között 5.5 mp rés → külön
    assert merged == [(0.0, 16), (25, 34), (39.5, 45)], merged
    # 0.5 mp-es rés (< 1 mp) → összevonva; 1.5 mp → külön
    assert ce.merge_windows([(0, 5), (5.5, 8)]) == [(0, 8)]
    assert ce.merge_windows([(0, 5), (6.5, 8)]) == [(0, 5), (6.5, 8)]
    assert ce.merge_windows([]) == []
    # A videó végén túli rész levágva
    assert ce.parse_windows([{"time": 58}], 5, 4, video_duration=60) == [(53, 60)]


def test_parse_errors():
    for bad in ([{"foo": 1}], [{"start_sec": 5}], [{"start_sec": 10, "end_sec": 5}], [{"time": "abc"}], ["x"],
                {"time": 3}):
        try:
            ce.parse_windows(bad, 5, 4)
        except ValueError as ex:
            assert str(ex)
        else:
            raise AssertionError(f"Nem dobott hibát: {bad}")


def test_export_validation_without_ffmpeg():
    out = tempfile.mkdtemp(prefix="clip_export_out_")
    assert ce.export_clips("/nincs/ilyen.mp4", [{"time": 3}], out)["status"] == "error"
    fake = os.path.join(out, "x.mp4")
    open(fake, "wb").close()
    r = ce.export_clips(fake, [], out)
    assert r["status"] == "error" and "üres" in r["message"], r
    assert ce.export_clips(fake, [{"start_sec": 3, "end_sec": 1}], out)["status"] == "error"
    assert ce.export_clips(fake, [{"time": 3}], out, pre_sec=-1)["status"] == "error"


# ---- ffmpeg-et igénylő tesztek ----------------------------------------------

def test_accurate_cut_and_merge():
    _need_ffmpeg()
    src = _source_video()
    out = tempfile.mkdtemp(prefix="clip_export_out_")
    # Az első kettő átfed (5-14 és 11-17 → 5-17), a harmadik külön (30.3-33.7)
    items = [{"time": 10}, {"start_sec": 11, "end_sec": 17}, {"start_sec": 30.3, "end_sec": 33.7}]
    r = ce.export_clips(src, items, out, title="Kapus védések – 1. félidő")
    assert r["status"] == "success", r
    assert len(r["clips"]) == 2, r["clips"]
    expected = [12.0, 3.4]
    for c, exp in zip(r["clips"], expected):
        d = _duration(c["local_path"])
        assert abs(d - exp) < 0.15, (c, d, exp)
        assert _has_audio(c["local_path"])
        assert c["url"].startswith("/clips/Kapus_vedesek")
    md = _duration(r["merged_local_path"])
    assert abs(md - sum(expected)) < 0.3, md
    assert _has_audio(r["merged_local_path"])
    assert r["merged_url"].endswith("_osszefuzott.mp4")
    json.dumps(r)


def test_single_clip_no_merge_and_stream_copy():
    _need_ffmpeg()
    src = _source_video()
    out = tempfile.mkdtemp(prefix="clip_export_out_")
    r = ce.export_clips(src, [{"time": 20}], out, pre_sec=3, post_sec=2, merge=False)
    assert r["status"] == "success" and len(r["clips"]) == 1 and r["merged_url"] is None, r
    assert abs(_duration(r["clips"][0]["local_path"]) - 5.0) < 0.15

    # Stream copy: csak kulcskockánál vág, a hossz pontatlanabb lehet, de a fájl létrejön
    r2 = ce.export_clips(src, [{"start_sec": 3.3, "end_sec": 8.1}, {"start_sec": 20, "end_sec": 24}], out,
                         accurate=False)
    assert r2["status"] == "success", r2
    assert len(r2["clips"]) == 2 and os.path.getsize(r2["merged_local_path"]) > 0
    for c in r2["clips"]:
        assert _duration(c["local_path"]) > 3.0


def test_missing_ffmpeg_binary():
    _need_ffmpeg()
    src = _source_video()
    old = ce.FFMPEG
    ce.FFMPEG = "/nincs/ilyen/ffmpeg"
    try:
        r = ce.export_clips(src, [{"time": 10}], tempfile.mkdtemp(prefix="clip_export_out_"))
    finally:
        ce.FFMPEG = old
    assert r["status"] == "error" and "ffmpeg" in r["message"], r


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Skip as ex:
            skipped += 1
            print(f"KIHAGYVA {name}: {ex}")
        except Exception as ex:
            failed += 1
            print(f"HIBA  {name}: {ex!r}")
    print(f"{len(tests) - failed - skipped}/{len(tests)} teszt sikeres, {skipped} kihagyva")
    sys.exit(1 if failed else 0)

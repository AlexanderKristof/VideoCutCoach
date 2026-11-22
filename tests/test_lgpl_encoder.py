"""
LGPL ffmpeg (libx264 nélkül) esetén a szoftveres H.264 tartalék az OpenH264 legyen –
a fast_cut (vágás) és a video_writer (elemzett videó) is ezt válassza.
Futtatás: python3 tests/test_lgpl_encoder.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fast_cut  # noqa: E402
import video_writer  # noqa: E402


def _patch(encoders, working=()):
    fast_cut._enc_cache.clear()
    video_writer._choice_cache.clear()
    fast_cut.list_encoders = lambda ffmpeg_bin=None: set(encoders)
    fast_cut.test_encoder = lambda name, ffmpeg_bin=None: name in working


def test_lgpl_build_uses_openh264():
    _patch({"libopenh264", "h264_nvenc"})
    assert fast_cut.encoder_candidates("h264") == ["libopenh264"]
    assert fast_cut.encoder_candidates("h264", override="cpu") == ["libopenh264"]
    assert video_writer.software_h264_encoder() == "libopenh264"
    assert video_writer.choose_h264_encoder() == "libopenh264"
    args = fast_cut._full_encoder_args("libopenh264", [])
    assert args[:2] == ["-c:v", "libopenh264"]


def test_lgpl_build_prefers_hardware():
    _patch({"libopenh264", "h264_nvenc"}, working={"h264_nvenc"})
    assert fast_cut.encoder_candidates("h264") == ["h264_nvenc", "libopenh264"]


def test_gpl_build_keeps_libx264():
    _patch({"libx264", "libopenh264"})
    assert fast_cut.encoder_candidates("h264") == ["libx264"]
    assert video_writer.software_h264_encoder() == "libx264"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in tests:
        try:
            fn()
            ok += 1
            print(f"OK    {fn.__name__}")
        except Exception as ex:
            print(f"HIBA  {fn.__name__}: {ex!r}")
    print(f"{ok}/{len(tests)} teszt sikeres")
    sys.exit(0 if ok == len(tests) else 1)

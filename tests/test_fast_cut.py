"""
A gyors, képkocka-pontos vágás (fast_cut.py) tesztjei. Futtatás: python3 tests/test_fast_cut.py

Szintetikus forrásvideó (720p, 25 fps, 90 mp, libx264, kulcskocka 2 mp-enként, B-kockákkal):
minden képkocka bal felső sarkában 16 bites bináris minta kódolja a képkocka sorszámát
(fekete / fehér négyzetek), a hangban minden egész másodpercnél 50 ms-os 1 kHz-es sípolás.
Így a vágás után kiolvasható, hogy pontosan melyik képkockával kezdődik / végződik a klip,
és hogy a hang a képpel szinkronban van-e.

Ha nincs ffmpeg / ffprobe / numpy, az ezeket igénylő tesztek kimaradnak.
"""
import os
import sys
import json
import math
import time
import shutil
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if "FFMPEG_PATH" not in os.environ and shutil.which("ffmpeg"):
    os.environ["FFMPEG_PATH"] = shutil.which("ffmpeg")

import background
import fast_cut as fc

FFMPEG = os.environ.get("FFMPEG_PATH")
FFPROBE = shutil.which("ffprobe")
FFMPEG_OK = bool(FFMPEG and shutil.which(FFMPEG) and FFPROBE)
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

W, H, FPS, DUR = 1280, 720, 25, 90
GOP = 50
SR = 48000
TIMINGS = {}


class Skip(Exception):
    pass


def _need():
    if not FFMPEG_OK:
        raise Skip("nincs ffmpeg / ffprobe")
    if np is None:
        raise Skip("nincs numpy")


_TMP = None
_SRC = {}


def _tmp():
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="fast_cut_test_")
    return _TMP


def _pattern_source(name="forras.mp4", dur=DUR, extra=None, pix_fmt="yuv420p", w=W, h=H, vf=None):
    """Mintás forrásvideó (lásd a modul leírását); extra: további ffmpeg kimeneti kapcsolók."""
    key = (name, dur, tuple(extra or ()), pix_fmt, w, h, vf)
    if key in _SRC:
        return _SRC[key]
    path = os.path.join(_tmp(), name)
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx * 255 // w + yy * 128 // h) % 256).astype(np.uint8)
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-f", "lavfi",
           "-i", f"aevalsrc=if(lt(mod(t\\,1)\\,0.05)\\,0.8*sin(2*PI*1000*t)\\,0):s={SR}:d={dur}"]
    if vf:
        cmd += ["-vf", vf]
    cmd += extra or ["-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-g", str(GOP), "-keyint_min", str(GOP), "-sc_threshold", "0",
            "-pix_fmt", pix_fmt, "-c:a", "aac", "-b:a", "128k", "-shortest", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(FPS * dur):
        f = np.roll(base, i * 7, axis=1)
        for b in range(16):
            f[0:40, b * 40:(b + 1) * 40] = 235 if (i >> b) & 1 else 16
        p.stdin.write(f.tobytes())
    p.stdin.close()
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError(err.decode("utf-8", "replace"))
    _SRC[key] = path
    return path


def frame_indices(path, max_frames=None):
    """A képkockákba kódolt sorszámok (az összes képkockáé, dekódolási sorrend helyett megjelenítésiben)."""
    cmd = [FFMPEG, "-v", "error", "-i", path, "-map", "0:v:0", "-vf", "crop=640:40:0:0"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    a = np.frombuffer(out, np.uint8).reshape(-1, 40, 640)
    res = []
    for f in a:
        v = 0
        for b in range(16):
            if f[10:30, b * 40 + 10:b * 40 + 30].mean() > 128:
                v |= 1 << b
        res.append(v)
    return res


def _stream_start(path, sel):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", sel, "-show_entries", "stream=start_time",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    try:
        return float(out.split("\n")[0])
    except ValueError:
        return None


def beep_onsets(path):
    """A sípolások kezdete a kimenet idővonalán (mp, a videó kezdetéhez képest)."""
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True).stdout
    a = np.abs(np.frombuffer(raw, np.int16).astype(np.float32)) / 32768.0
    env = np.convolve(a, np.ones(48) / 48, mode="same") > 0.2
    on = np.flatnonzero(env[1:] & ~env[:-1]) / float(SR)
    return on + (_stream_start(path, "a:0") or 0.0) - (_stream_start(path, "v:0") or 0.0)


def _probe(path):
    return json.loads(subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                                     capture_output=True, text=True).stdout)


def check_cut(path, start_sec, end_sec, label="", audio=True, eof_frames=None):
    """Képkocka-pontosság + A/V szinkron ellenőrzése. Visszatér: (első index, képkockaszám, A/V hiba mp)."""
    s_idx = int(round(start_sec * FPS))
    e_idx = int(round(end_sec * FPS))
    if eof_frames is not None:
        e_idx = min(e_idx, eof_frames)
    idx = frame_indices(path)
    assert idx, f"{label}: nincs képkocka"
    assert idx[0] == s_idx, f"{label}: az első képkocka {idx[0]}, várt {s_idx}"
    assert idx == list(range(s_idx, s_idx + len(idx))), f"{label}: nem folytonos képkocka-sorozat"
    assert abs(len(idx) - (e_idx - s_idx)) <= 1, f"{label}: {len(idx)} képkocka, várt {e_idx - s_idx}"
    av_err = None
    if audio:
        info = _probe(path)
        assert any(s["codec_type"] == "audio" for s in info["streams"]), f"{label}: nincs hangsáv"
        on = beep_onsets(path)
        s_time = s_idx / FPS
        # a teljes sípolás-kezdetek (a klip elején félbevágott sípolás kimarad)
        e_time = s_time + len(idx) / FPS
        expected = [k - s_time for k in range(math.ceil(s_time + 0.001), int(e_time) + 1) if k < e_time - 0.06]
        assert expected, label
        errs = []
        for e in expected:
            near = on[np.abs(on - e) < 0.2] if len(on) else []
            assert len(near), f"{label}: nincs sípolás {e:.3f} mp körül (talált: {on[:5]})"
            errs.append(float(near[0] - e))
        av_err = max(errs, key=abs)
        assert abs(av_err) < 0.5 / FPS, f"{label}: A/V eltérés {av_err * 1000:.1f} ms"
    return idx[0], len(idx), av_err


# ---------------------------------------------------------------------------
# Tiszta logika (ffmpeg nélkül)
# ---------------------------------------------------------------------------

def test_encoder_candidates_priority_and_override():
    old_list, old_test = fc.list_encoders, fc.test_encoder
    working = {"h264_qsv", "libx264", "hevc_nvenc"}
    fc.list_encoders = lambda ffmpeg_bin=None: {"h264_nvenc", "h264_qsv", "h264_amf", "libx264", "hevc_nvenc",
                                                "libx265"}
    fc.test_encoder = lambda name, ffmpeg_bin=None: name in working
    try:
        assert fc.encoder_candidates("h264", override="auto") == ["h264_qsv", "libx264"]
        assert fc.encoder_candidates("hevc", override="") == ["hevc_nvenc", "libx265"]
        assert fc.encoder_candidates("h264", override="cpu") == ["libx264"]
        assert fc.encoder_candidates("h264", override="libx264") == ["libx264"]
        working.add("h264_amf")
        # kért kódoló elsőként, utána a többi működő, a szoftveres mindig a végén
        assert fc.encoder_candidates("h264", override="h264_amf") == ["h264_amf", "h264_qsv", "libx264"]
        # nem működő kért kódoló → automatikus sorrend
        assert fc.encoder_candidates("h264", override="h264_nvenc") == ["h264_qsv", "h264_amf", "libx264"]
        # másik kodek kódolója nem keveredik ide
        assert fc.encoder_candidates("hevc", override="h264_qsv") == ["hevc_nvenc", "libx265"]
        os.environ["FAST_CUT_ENCODER"] = "software"
        try:
            assert fc.encoder_candidates("h264") == ["libx264"]
        finally:
            del os.environ["FAST_CUT_ENCODER"]
        assert fc.encoder_candidates("mpeg2video") == []
    finally:
        fc.list_encoders, fc.test_encoder = old_list, old_test


def test_level_profile_helpers():
    assert fc._level_str("h264", 31) == "3.1" and fc._level_str("h264", 40) == "4.0"
    assert fc._level_str("hevc", 93) == "3.1" and fc._level_str("hevc", 120) == "4.0"
    assert fc._level_str("h264", None) is None
    assert fc._profile_arg("h264", "Constrained Baseline") == "baseline"
    assert fc._profile_arg("h264", "High") == "high"
    assert fc._profile_arg("h264", "High 4:4:4 Predictive") is None
    assert fc._profile_arg("hevc", "Main 10") == "main10"
    assert fc._same_profile("Constrained Baseline", "Baseline")
    assert fc.resolve_ffprobe("/opt/ff/bin/ffmpeg") == "/opt/ff/bin/ffprobe"
    assert fc.resolve_ffprobe("C:/ff/ffmpeg.exe") == "C:/ff/ffprobe.exe"


def test_background_prepare_flags():
    cfg = background.BackgroundConfig(enabled=True, threads=3)
    cmd, kw = background.prepare(["ffmpeg", "-i", "x"], cfg, _os_name="nt")
    assert cmd == ["ffmpeg", "-i", "x"] and kw == {"creationflags": background.BELOW_NORMAL_PRIORITY_CLASS}
    cmd, kw = background.prepare(["ffmpeg"], cfg, _os_name="posix", _which=lambda n: "/usr/bin/nice", _nice_now=0)
    assert cmd == ["/usr/bin/nice", "-n", str(background.NICE_LEVEL), "ffmpeg"] and kw == {}
    # a már alacsony prioritású szál gyermeke örököl – nincs dupla nice
    cmd, kw = background.prepare(["ffmpeg"], cfg, _os_name="posix", _which=lambda n: "/usr/bin/nice", _nice_now=10)
    assert cmd == ["ffmpeg"] and kw == {}
    # háttér-mód nélkül semmi
    assert background.prepare(["ffmpeg"], background.BackgroundConfig(False, 2), _os_name="nt") == (["ffmpeg"], {})
    assert background.ffmpeg_thread_args(cfg) == ["-threads", "3"]
    assert background.ffmpeg_thread_args(background.BackgroundConfig(True, None)) == []
    # szálhoz kötött beállítás
    assert background.current() is None
    with background.context(True, 2):
        assert background.current().threads == 2 and background.current().enabled
    assert background.current() is None


# ---------------------------------------------------------------------------
# ffmpeg-et igénylő tesztek
# ---------------------------------------------------------------------------

def test_plan_finds_keyframes():
    _need()
    src = _pattern_source()
    info = fc.probe(src)
    assert info["video"]["codec"] == "h264" and info["video"]["nb_frames"] == FPS * DUR
    plan = fc.plan_cut(info, 33.32, 41.08, src=src)
    assert plan["s_idx"] == 833 and plan["frames"] == 194
    assert float(plan["k1"][0]) == 34.0 and float(plan["kend"][0]) == 40.0, (plan["k1"], plan["kend"])
    assert plan["head"] is not None and plan["tail"] is not None and plan["delay"] >= 1
    p2 = fc.plan_cut(info, 34.0, 40.0, src=src)
    assert p2["head"] is None and p2["tail"] is None          # mindkét vége kulcskocka → tiszta másolás
    p3 = fc.plan_cut(info, 80.03, 200, src=src)
    assert p3["eof"] and p3["tail"] is None and p3["frames"] == FPS * DUR - 2001


def test_smart_cut_frame_accurate_and_in_sync():
    _need()
    src = _pattern_source()
    out = os.path.join(_tmp(), "smart.mp4")
    cases = [
        (33.32, 41.08, "smart"),    # se az eleje, se a vége nem kulcskocka
        (14.0, 21.08, "smart"),     # kulcskockán kezdődik, a vége nem
        (13.32, 20.0, "smart"),     # a vége kulcskocka
        (34.0, 40.0, "copy"),       # mindkettő kulcskocka → újrakódolás nélkül
        (0.0, 3.3, "smart"),        # a videó elejétől
        (1.01, 9.97, "smart"),
        (80.03, 200.0, "smart"),    # a videó végéig (a vége túllóg)
        (36.5, 37.3, "full"),       # rövidebb egy GOP-nál, nincs benne kulcskocka → teljes újrakódolás
    ]
    for s, e, want in cases:
        info = fc.smart_cut(src, s, e, out)
        assert info["mode"] == want, (s, e, info)
        first, n, av = check_cut(out, s, e, f"{s}-{e}", eof_frames=FPS * DUR)
        print(f"      {s:6.2f}–{e:6.2f} mp: mód={info['mode']:5s} első kocka={first} ({n} kocka, "
              f"fej {info['head_frames']} / farok {info['tail_frames']} újrakódolva), "
              f"A/V eltérés {av * 1000:+.1f} ms, {info['elapsed_sec']:.2f} mp")


def test_timings_copy_smart_full():
    _need()
    src = _pattern_source()
    s, e = 5.3, 85.7
    d = _tmp()
    t = time.monotonic()
    subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{s}", "-i", src, "-t", f"{e - s}", "-c", "copy",
                    os.path.join(d, "t_copy.mp4")], check=True)
    t_copy = time.monotonic() - t
    fc.detected_encoders()  # a kódoló-felderítés ne számítson bele
    t = time.monotonic()
    smart = fc.smart_cut(src, s, e, os.path.join(d, "t_smart.mp4"))
    t_smart = time.monotonic() - t
    t = time.monotonic()
    full = fc.smart_cut(src, s, e, os.path.join(d, "t_full.mp4"), mode="full")
    t_full = time.monotonic() - t
    assert smart["mode"] == "smart" and full["mode"] == "full"
    check_cut(os.path.join(d, "t_smart.mp4"), s, e, "időmérés smart")
    check_cut(os.path.join(d, "t_full.mp4"), s, e, "időmérés full")
    copy_first = frame_indices(os.path.join(d, "t_copy.mp4"), 1)[0]
    TIMINGS.update({"copy": t_copy, "smart": t_smart, "full": t_full, "copy_first": copy_first,
                    "want_first": int(round(s * FPS)), "len": e - s})
    print(f"      {e - s:.1f} mp-es 720p szakasz: stream copy {t_copy:.2f} mp (első kocka {copy_first}, "
          f"várt {int(round(s * FPS))} – pontatlan), smart {t_smart:.2f} mp, teljes újrakódolás {t_full:.2f} mp "
          f"(kódoló: {full['encoder']})")
    assert t_smart < t_full / 2, (t_smart, t_full)


def test_fallback_vfr_source():
    _need()
    # minden 7. képkocka kimarad → változó képkocka-távolság
    src = _pattern_source("vfr.mp4", dur=30, extra=["-map", "0:v", "-map", "1:a", "-fps_mode", "vfr"],
                          vf="select='not(eq(mod(n\\,7)\\,3))'")
    out = os.path.join(_tmp(), "vfr_cut.mp4")
    info = fc.smart_cut(src, 5.3, 12.7, out)
    assert info["mode"] == "full" and info["reason"], info
    idx = frame_indices(out)
    # a 5.3 mp utáni első megmaradt eredeti képkocka (a 7-es mintából kimaradók nélkül)
    first = next(i for i in range(int(round(5.3 * FPS)), 10000) if i % 7 != 3)
    assert idx[0] == first, (idx[:3], first)
    assert _probe(out)["format"]["duration"]
    print(f"      VFR forrás → teljes újrakódolás ({info['reason']})")


def test_fallback_pix_fmt_and_mpegts():
    _need()
    src444 = _pattern_source("p444.mp4", dur=30, pix_fmt="yuv444p")
    out = os.path.join(_tmp(), "p444_cut.mp4")
    info = fc.smart_cut(src444, 13.32, 21.08, out)
    assert info["mode"] == "full" and "profil" in info["reason"], info
    check_cut(out, 13.32, 21.08, "yuv444p")
    print(f"      yuv444p forrás → teljes újrakódolás ({info['reason']})")

    # MPEG-TS: nem indexelt, a kezdete nem 0 (és a hang / kép kezdete eltér)
    ts = os.path.join(_tmp(), "forras.ts")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", _pattern_source(), "-t", "40", "-c", "copy", ts], check=True)
    assert fc.probe(ts)["start"] > 1.0
    out = os.path.join(_tmp(), "ts_cut.mp4")
    info = fc.smart_cut(ts, 13.32, 21.08, out)
    assert info["mode"] == "full", info
    check_cut(out, 13.32, 21.08, "mpegts")
    print(f"      MPEG-TS forrás → teljes újrakódolás ({info['reason']})")


def test_video_not_first_stream():
    _need()
    src = os.path.join(_tmp(), "hang_elol.mp4")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", _pattern_source(), "-t", "40", "-map", "0:a", "-map", "0:v",
                    "-c", "copy", src], check=True)
    out = os.path.join(_tmp(), "hang_elol_cut.mp4")
    for s, e in ((13.32, 21.08), (14.0, 21.08)):
        info = fc.smart_cut(src, s, e, out)
        assert info["mode"] == "smart", info
        check_cut(out, s, e, f"hang az 1. sáv {s}")


def test_no_audio_source():
    _need()
    src = os.path.join(_tmp(), "nema.mp4")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", _pattern_source(), "-t", "40", "-an", "-c", "copy", src],
                   check=True)
    out = os.path.join(_tmp(), "nema_cut.mp4")
    info = fc.smart_cut(src, 13.32, 21.08, out)
    assert info["mode"] == "smart", info
    check_cut(out, 13.32, 21.08, "hang nélkül", audio=False)
    assert not any(s["codec_type"] == "audio" for s in _probe(out)["streams"])


def test_hw_encoder_failure_falls_back_to_software():
    _need()
    src = _pattern_source()
    old = fc.encoder_candidates
    # egy "felderített", de valójában nem működő hardveres kódoló (pl. eltűnt a meghajtó)
    fc.encoder_candidates = lambda codec, ffmpeg_bin=None, override=None: ["h264_nem_letezik", fc.SW_ENCODERS[codec]]
    try:
        out = os.path.join(_tmp(), "hw_fallback.mp4")
        info = fc.smart_cut(src, 33.32, 41.08, out)
        assert info["mode"] == "smart" and info["encoder"] == "libx264", info
        check_cut(out, 33.32, 41.08, "hw tartalék")
        info = fc.smart_cut(src, 33.32, 41.08, out, mode="full")
        assert info["mode"] == "full" and info["encoder"] == "libx264", info
    finally:
        fc.encoder_candidates = old


def test_verify_failure_never_leaves_broken_file():
    _need()
    src = _pattern_source()
    out = os.path.join(_tmp(), "never_broken.mp4")
    old_verify = fc._verify
    fc._verify = lambda *a, **k: "szimulált hiba"
    try:
        info = fc.smart_cut(src, 33.32, 41.08, out)
    finally:
        fc._verify = old_verify
    assert info["mode"] == "full" and "szimulált" in info["reason"], info
    check_cut(out, 33.32, 41.08, "ellenőrzési hiba után")
    # ha az ffmpeg sem fut: hiba, és nem marad (fél)kész kimenet
    bad = os.path.join(_tmp(), "nincs_kimenet.mp4")
    try:
        fc.smart_cut(src, 33.32, 41.08, bad, ffmpeg_bin="/nincs/ilyen/ffmpeg", ffprobe_bin="/nincs/ilyen/ffprobe")
    except RuntimeError as ex:
        assert "ffmpeg" in str(ex)
    else:
        raise AssertionError("nem dobott hibát")
    assert not os.path.exists(bad)
    assert not [n for n in os.listdir(_tmp()) if n.startswith("fastcut_")], "ideiglenes mappa maradt"


def test_missing_ffprobe_uses_full_reencode():
    _need()
    out = os.path.join(_tmp(), "probe_nelkul.mp4")
    info = fc.smart_cut(_pattern_source(), 33.32, 41.08, out, ffprobe_bin="/nincs/ilyen/ffprobe")
    assert info["mode"] == "full", info
    check_cut(out, 33.32, 41.08, "ffprobe nélkül")


def test_background_mode_commands():
    _need()
    src = _pattern_source()
    seen = []
    old_run = background.run

    def spy(cmd, **kw):
        c, extra = background.prepare(cmd)
        seen.append((c, extra))
        return old_run(cmd, **kw)

    background.run = spy
    try:
        with background.context(enabled=True, threads=2):
            info = fc.smart_cut(src, 33.32, 41.08, os.path.join(_tmp(), "hatter.mp4"))
    finally:
        background.run = old_run
    assert info["mode"] == "smart"
    encodes = [c for c, _ in seen if "-c:v" in c and "libx264" in c]
    assert encodes and all("-threads" in c and c[c.index("-threads") + 1] == "2" for c in encodes), encodes
    # minden ffmpeg / ffprobe alacsony prioritással indulna (POSIX: nice, Windows: creationflags)
    for c, extra in seen:
        assert extra.get("creationflags") == background.BELOW_NORMAL_PRIORITY_CLASS or "nice" in os.path.basename(c[0]) \
            or background._current_nice() >= background.NICE_LEVEL, (c[:3], extra)


def test_compilation_concat_copy_and_reencode():
    _need()
    src = _pattern_source()
    d = os.path.join(_tmp(), "osszeallitas")
    os.makedirs(d, exist_ok=True)
    windows = [(3.3, 9.1), (20.0, 26.52), (47.18, 52.0), (70.44, 77.0)]
    parts, expected = [], []
    t = time.monotonic()
    for i, (s, e) in enumerate(windows):
        p = os.path.join(d, f"klip{i}.mp4")
        info = fc.smart_cut(src, s, e, p)
        assert info["mode"] in ("smart", "copy"), info
        parts.append(p)
        si, ei = int(round(s * FPS)), int(round(e * FPS))
        expected += list(range(si, ei))
    out = os.path.join(d, "osszefuzott.mp4")
    res = fc.concat_files(parts, out)
    t_total = time.monotonic() - t
    assert res["mode"] == "copy", res
    idx = frame_indices(out)
    assert idx == expected, (len(idx), len(expected), idx[:3])
    assert any(s["codec_type"] == "audio" for s in _probe(out)["streams"])
    dur = float(_probe(out)["format"]["duration"])
    assert abs(dur - len(expected) / FPS) < 0.15, dur
    TIMINGS["compilation"] = t_total
    print(f"      4 klip smart cut + összefűzés (copy): {t_total:.2f} mp, {len(idx)} kocka, sorrend pontos")

    # eltérő felbontású rész → újrakódolt összefűzés
    small = os.path.join(d, "kicsi.mp4")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", parts[0], "-vf", "scale=640:360", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", small], check=True)
    out2 = os.path.join(d, "vegyes.mp4")
    res2 = fc.concat_files([parts[1], small], out2)
    assert res2["mode"] == "reencode", res2
    v = [s for s in _probe(out2)["streams"] if s["codec_type"] == "video"][0]
    assert (v["width"], v["height"]) == (W, H)
    n = len(frame_indices(out2))
    assert abs(n - (len(range(500, 663)) + len(range(82, 228)))) <= 2, n


def test_clip_export_and_match_halves_use_smart_cut():
    _need()
    import clip_export as ce
    import match_segmenter as ms
    src = _pattern_source()
    d = os.path.join(_tmp(), "export")
    old = ce.FFMPEG
    ce.FFMPEG = FFMPEG
    try:
        r = ce.export_clips(src, [{"start_sec": 10.36, "end_sec": 15.0}, {"time": 40.2}], d, pre_sec=2, post_sec=3)
    finally:
        ce.FFMPEG = old
    assert r["status"] == "success", r
    assert [c["cut_mode"] for c in r["clips"]] == ["smart", "smart"], r["clips"]
    assert r["merge_mode"] == "copy"
    check_cut(r["clips"][1]["local_path"], 38.2, 43.2, "export (time)")
    idx = frame_indices(r["merged_local_path"])
    assert idx == list(range(259, 375)) + list(range(955, 1080)), (idx[:2], len(idx))

    halves = {"first_half": {"start_sec": 10.3, "end_sec": 40.1}, "second_half": {"start_sec": 50.22, "end_sec": 80.0}}
    seen = []
    out = ms.cut_match_halves(src, halves, d, padding_sec=1.0, duration_sec=DUR, ffmpeg_bin=FFMPEG,
                              on_part=lambda k, p: seen.append(k))
    assert seen == ["first_half", "second_half", "full_match"], seen
    assert set(out["cut_modes"].values()) <= {"smart", "copy"}, out
    check_cut(out["first_half"], 9.3, 41.1, "1. félidő")
    check_cut(out["second_half"], 49.22, 81.0, "2. félidő")
    idx = frame_indices(out["full_match"])
    assert idx == list(range(233, 1028)) + list(range(1230, 2025)), (idx[:2], len(idx))


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    if sys.argv[1:]:  # csak a megadott (rész)nevű tesztek
        tests = [(n, f) for n, f in tests if any(a in n for a in sys.argv[1:])]
    failed = skipped = 0
    try:
        for name, fn in tests:
            t0 = time.monotonic()
            try:
                fn()
                print(f"OK    {name} ({time.monotonic() - t0:.1f} mp)")
            except Skip as ex:
                skipped += 1
                print(f"KIHAGYVA {name}: {ex}")
            except Exception as ex:
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"HIBA  {name}: {ex!r}")
        if TIMINGS.get("smart"):
            print("\nIdőmérés ({:.1f} mp-es 720p szakasz, {} mag):".format(TIMINGS["len"], os.cpu_count()))
            print(f"  stream copy (pontatlan, kulcskockánál kezd): {TIMINGS['copy']:6.2f} mp")
            print(f"  smart cut (képkocka-pontos):                 {TIMINGS['smart']:6.2f} mp")
            print(f"  teljes újrakódolás (képkocka-pontos):        {TIMINGS['full']:6.2f} mp")
        if FFMPEG_OK:
            print(f"Felderített kódolók: {fc.detected_encoders()}")
    finally:
        if _TMP:
            shutil.rmtree(_TMP, ignore_errors=True)
    print(f"{len(tests) - failed - skipped}/{len(tests)} teszt sikeres, {skipped} kihagyva")
    sys.exit(1 if failed else 0)

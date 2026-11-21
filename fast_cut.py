"""
Gyors, képkocka-pontos vágás ("smart cut") ffmpeg-gel.

Egy [start, end] szakasz vágása:
  1. ffprobe-bal a kezdő- és a végpont környékén (csak egy rövid ablakban, `-read_intervals`)
     kiolvassuk a kulcskockák időbélyegeit, és a forrás paramétereit (kodek, profil, szint,
     pix_fmt, felbontás, fps / időalap, B-kocka késleltetés, hangsáv).
  2. Ha a kezdőpont kulcskocka (fél képkockán belül), nincs mit újrakódolni az elején.
     Különben csak a [start, következő kulcskocka) "fej" részt kódoljuk újra a forrással
     egyező paraméterekkel (ugyanaz a kodek, profil, pix_fmt, felbontás, időalap, zárt GOP,
     azonos B-kocka késleltetés), a többit újrakódolás nélkül (stream copy) másoljuk.
  3. A vége ugyanígy: ha a végpont nem kulcskockára esik, az utolsó kulcskockától a
     végpontig tartó "farok" részt kódoljuk újra – így a vége is képkocka-pontos.
  4. Az összefűzés a concat demuxerrel történik (`-c copy`); a középső rész közvetlenül
     a forrásfájlból jön (inpoint / outpoint), nincs köztes másolat. A hang a teljes
     szakaszra a forrásból jön, `-ss`-sel igazítva (AAC/MP3/AC3: másolva, egyébként AAC-ra
     kódolva) – az ffmpeg a hangot mintapontosan igazítja (edit list).
  5. Ellenőrzés: képkockaszám, A/V kezdet, időbélyegek monotonitása és dekódolási próba a
     csatlakozási pontok körül. Ha bármi nem stimmel (vagy a forrás nem alkalmas: VFR,
     ismeretlen profil, váltott soros kép, ...), teljes újrakódolásra váltunk – hibás fájl
     soha nem kerül a kimenetre (az ideiglenes fájl csak a sikeres ellenőrzés után kap nevet).

Hardveres kódolók (a rövid újrakódolt részekhez és a teljes újrakódoláshoz): egyszer, a
folyamat elején felderítjük (`ffmpeg -encoders` + egy apró próbakódolás), sorrend:
h264_nvenc / hevc_nvenc (NVIDIA), h264_qsv / hevc_qsv (Intel), h264_amf / hevc_amf (AMD),
végül libx264 / libx265 (-preset veryfast). Felülírás: FAST_CUT_ENCODER környezeti változó
("auto", "cpu" = szoftveres, vagy egy kódolónév, pl. "h264_qsv").
FAST_CUT_MODE=full: mindig teljes újrakódolás (hibakereséshez / ha egy lejátszó gondot okoz).

Háttér-mód (background.py): az ffmpeg alacsony prioritással és `-threads N`-nel indul.
"""
import os
import math
import json
import time
import shutil
import logging
import tempfile
import threading
import subprocess
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor

import background

log = logging.getLogger("fast_cut")

DEFAULT_FFMPEG = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

# Kulcskocka-keresési ablak (mp) és a legnagyobb ablak, ameddig bővítünk
KEY_WINDOW_SEC = 6.0
KEY_WINDOW_MAX_SEC = 64.0

# A hang konténerváltás nélkül (mp4-be) másolható kodekjei; a többit AAC-ra kódoljuk
AUDIO_COPY_CODECS = ("aac", "mp3", "ac3", "eac3", "alac")

SW_ENCODERS = {"h264": "libx264", "hevc": "libx265"}
HW_ENCODERS = {"h264": ("h264_nvenc", "h264_qsv", "h264_amf"),
               "hevc": ("hevc_nvenc", "hevc_qsv", "hevc_amf")}

# ffprobe profil → kódoló profil (csak ezeknél tudjuk a fej/farok részt egyezően kódolni)
H264_PROFILES = {"constrained baseline": "baseline", "baseline": "baseline", "main": "main", "high": "high"}
HEVC_PROFILES = {"main": "main", "main 10": "main10"}
PART_PIX_FMTS = {"h264": ("yuv420p", "yuvj420p"), "hevc": ("yuv420p", "yuv420p10le")}


class FastCutError(RuntimeError):
    """A gyors vágás nem végezhető el (a hívó teljes újrakódolásra vált)."""


# ------------------------------------------------------------------
# Segédek
# ------------------------------------------------------------------

def resolve_ffmpeg(ffmpeg_bin=None):
    return ffmpeg_bin or os.environ.get("FFMPEG_PATH") or DEFAULT_FFMPEG


def resolve_ffprobe(ffmpeg_bin=None, ffprobe_bin=None):
    """Az ffprobe útvonala: FFPROBE_PATH, különben az ffmpeg mellől (ffmpeg → ffprobe a névben)."""
    if ffprobe_bin:
        return ffprobe_bin
    if os.environ.get("FFPROBE_PATH"):
        return os.environ["FFPROBE_PATH"]
    ff = resolve_ffmpeg(ffmpeg_bin)
    d, name = os.path.split(ff)
    probe_name = name.replace("ffmpeg", "ffprobe") if "ffmpeg" in name else ("ffprobe.exe" if os.name == "nt" else "ffprobe")
    return os.path.join(d, probe_name) if d else probe_name


def _run(cmd, timeout=None):
    """ffmpeg / ffprobe futtatása háttér-mód szerinti prioritással. Visszatér: CompletedProcess."""
    try:
        return background.run(cmd, capture_output=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError(f"Az ffmpeg/ffprobe nem található ({cmd[0]}). Állítsd be az FFMPEG_PATH környezeti változót.")


def _err_tail(res, n=3):
    lines = res.stderr.decode("utf-8", "replace").strip().splitlines()
    return " | ".join(lines[-n:])


def _frac(text, default=None):
    try:
        f = Fraction(str(text))
        return f if f != 0 else default
    except (ValueError, ZeroDivisionError, TypeError):
        return default


def _float(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _ceil_us(t):
    """Másodperc (Fraction) → szöveg, mikroszekundumra felfelé kerekítve."""
    return f"{math.ceil(t * 1000000) / 1000000:.6f}"


def _floor_us(t):
    return f"{math.floor(t * 1000000) / 1000000:.6f}"


def _fmt(t):
    return f"{float(t):.6f}"


def _safe_path(p):
    return os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")


# ------------------------------------------------------------------
# ffprobe
# ------------------------------------------------------------------

_probe_cache = {}
_probe_lock = threading.Lock()


def probe(path, ffprobe_bin=None, use_cache=True):
    """
    A fájl paraméterei. Visszatér: {"format": .., "start": mp, "duration": mp, "video": {..} | None,
    "audio": {..} | None}. Hibánál FastCutError.
    """
    ffprobe_bin = resolve_ffprobe(None, ffprobe_bin)
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_size, st.st_mtime, ffprobe_bin)
    except OSError:
        raise FastCutError(f"A fájl nem található: {path}")
    if use_cache:
        with _probe_lock:
            if key in _probe_cache:
                return _probe_cache[key]
    try:
        res = _run([ffprobe_bin, "-v", "error", "-show_format", "-show_streams", "-of", "json", path], timeout=120)
    except RuntimeError as ex:
        raise FastCutError(str(ex))
    if res.returncode != 0:
        raise FastCutError(f"ffprobe hiba: {_err_tail(res)}")
    data = json.loads(res.stdout.decode("utf-8", "replace") or "{}")
    fmt = data.get("format") or {}
    info = {"format": fmt.get("format_name", ""), "start": _float(fmt.get("start_time"), 0.0) or 0.0,
            "duration": _float(fmt.get("duration")), "video": None, "audio": None}
    info["stream_types"] = [st.get("codec_type") for st in sorted(data.get("streams") or [],
                                                                  key=lambda st: int(st.get("index", 0)))]
    for s in data.get("streams") or []:
        kind = s.get("codec_type")
        if kind == "video" and info["video"] is None and not (s.get("disposition") or {}).get("attached_pic"):
            tb = _frac(s.get("time_base"), Fraction(1, 90000))
            start_pts = s.get("start_pts")
            info["video"] = {
                "index": int(s.get("index", 0)), "codec": s.get("codec_name"), "profile": s.get("profile"),
                "level": s.get("level"), "pix_fmt": s.get("pix_fmt"),
                "width": s.get("width"), "height": s.get("height"),
                "fps": _frac(s.get("r_frame_rate")), "avg_fps": _frac(s.get("avg_frame_rate")),
                "tb": tb, "start_pts": int(start_pts) if start_pts is not None else None,
                "start": _float(s.get("start_time"), info["start"]),
                "duration": _float(s.get("duration")),
                "nb_frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
                "field_order": s.get("field_order") or "unknown",
                "sar": s.get("sample_aspect_ratio"), "tag": s.get("codec_tag_string"),
                "color_range": s.get("color_range"), "color_space": s.get("color_space"),
                "color_transfer": s.get("color_transfer"), "color_primaries": s.get("color_primaries"),
            }
        elif kind == "audio" and info["audio"] is None:
            info["audio"] = {"index": int(s.get("index", 0)), "codec": s.get("codec_name"),
                             "sample_rate": _float(s.get("sample_rate")), "channels": s.get("channels"),
                             "start": _float(s.get("start_time"), info["start"]),
                             "duration": _float(s.get("duration"))}
    if use_cache:
        with _probe_lock:
            if len(_probe_cache) > 64:
                _probe_cache.clear()
            _probe_cache[key] = info
    return info


def probe_packets(path, stream_index, start_sec, end_sec, ffprobe_bin=None):
    """
    A videósáv csomagjai egy időablakban (abszolút időbélyegek, mp): [(pts, dts, kulcskocka), ...]
    fájlsorrendben. Csak a csomagfejléceket olvassa (nem dekódol) – gyors.
    """
    ffprobe_bin = resolve_ffprobe(None, ffprobe_bin)
    a = max(0.0, float(start_sec))
    iv = f"{a:.6f}%{float(end_sec):.6f}"
    res = _run([ffprobe_bin, "-v", "error", "-select_streams", str(stream_index), "-read_intervals", iv,
                "-show_entries", "packet=pts,dts,flags", "-of", "csv=p=0", path], timeout=120)
    if res.returncode != 0:
        raise FastCutError(f"ffprobe csomaglista hiba: {_err_tail(res)}")
    out = []
    for line in res.stdout.decode("utf-8", "replace").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
            continue
        pts = int(parts[0])
        dts = int(parts[1]) if parts[1].lstrip("-").isdigit() else pts
        out.append((pts, dts, "K" in parts[2]))
    return out


def count_video_packets(path, ffprobe_bin=None):
    res = _run([resolve_ffprobe(None, ffprobe_bin), "-v", "error", "-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path], timeout=600)
    txt = res.stdout.decode("utf-8", "replace").strip().split(",")[0] if res.returncode == 0 else ""
    return int(txt) if txt.isdigit() else None


# ------------------------------------------------------------------
# Kódolók felderítése
# ------------------------------------------------------------------

_enc_cache = {}
_enc_lock = threading.Lock()


def _encoder_family(name):
    if name in ("libx264",) or name.startswith("h264_"):
        return "h264"
    if name in ("libx265",) or name.startswith("hevc_"):
        return "hevc"
    return None


def list_encoders(ffmpeg_bin=None):
    """Az ffmpeg-be fordított kódolók nevei (ffmpeg -encoders)."""
    ffmpeg_bin = resolve_ffmpeg(ffmpeg_bin)
    key = ("list", ffmpeg_bin)
    with _enc_lock:
        if key in _enc_cache:
            return _enc_cache[key]
    names = set()
    try:
        res = _run([ffmpeg_bin, "-hide_banner", "-encoders"], timeout=30)
        for line in res.stdout.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
                names.add(parts[1])
    except (RuntimeError, subprocess.TimeoutExpired):
        pass
    with _enc_lock:
        _enc_cache[key] = names
    return names


def test_encoder(name, ffmpeg_bin=None):
    """Apró próbakódolás (pár fekete képkocka) – a hardveres kódolónak tényleg van-e eszköze."""
    ffmpeg_bin = resolve_ffmpeg(ffmpeg_bin)
    key = ("test", ffmpeg_bin, name)
    with _enc_lock:
        if key in _enc_cache:
            return _enc_cache[key]
    pix = "nv12" if name.endswith(("_qsv", "_amf")) else "yuv420p"
    cmd = [ffmpeg_bin, "-hide_banner", "-v", "error", "-nostdin", "-f", "lavfi",
           "-i", "color=c=black:s=320x240:r=25", "-frames:v", "5", "-pix_fmt", pix, "-c:v", name, "-f", "null", "-"]
    try:
        ok = _run(cmd, timeout=20).returncode == 0
    except (RuntimeError, subprocess.TimeoutExpired):
        ok = False
    with _enc_lock:
        _enc_cache[key] = ok
    log.info("Kódoló próba: %s → %s", name, "működik" if ok else "nem elérhető")
    return ok


def encoder_candidates(codec, ffmpeg_bin=None, override=None):
    """
    A használható kódolók prioritási sorrendben (a szoftveres mindig az utolsó tartalék).
    override (vagy FAST_CUT_ENCODER): "auto" / "" = felderítés, "cpu" / "software" = csak szoftveres,
    más érték = az adott kódoló elsőként (ha a kodekje egyezik és működik).
    """
    sw = SW_ENCODERS.get(codec)
    if sw is None:
        return []
    choice = (override if override is not None else os.environ.get("FAST_CUT_ENCODER", "auto")).strip().lower()
    if choice in ("cpu", "software", "szoftver", "sw") or choice == sw:
        return [sw]
    available = list_encoders(ffmpeg_bin)
    out = []
    if choice not in ("", "auto") and _encoder_family(choice) == codec:
        if choice in available and test_encoder(choice, ffmpeg_bin):
            out.append(choice)
        else:
            log.warning("A FAST_CUT_ENCODER=%s kódoló nem érhető el – automatikus választás.", choice)
    for name in HW_ENCODERS.get(codec, ()):
        if name not in out and name in available and test_encoder(name, ffmpeg_bin):
            out.append(name)
    out.append(sw)
    return out


def detected_encoders(ffmpeg_bin=None):
    """Összefoglaló a felderített kódolókról (naplózáshoz / API-hoz)."""
    return {codec: encoder_candidates(codec, ffmpeg_bin) for codec in ("h264", "hevc")}


# ------------------------------------------------------------------
# Kódolási paraméterek
# ------------------------------------------------------------------

def _level_str(codec, level):
    try:
        lv = int(level)
    except (TypeError, ValueError):
        return None
    if lv <= 0:
        return None
    if codec == "hevc":
        lv = lv / 30.0
        return f"{lv:.1f}"
    return f"{lv // 10}.{lv % 10}"


def _profile_arg(codec, profile):
    table = H264_PROFILES if codec == "h264" else HEVC_PROFILES
    return table.get(str(profile or "").strip().lower())


def _same_profile(a, b):
    norm = lambda p: str(p or "").strip().lower().replace("constrained baseline", "baseline")
    return norm(a) == norm(b)


def _color_args(v):
    args = []
    rng = v.get("color_range")
    if v.get("pix_fmt") == "yuvj420p":
        rng = "pc"
    if rng in ("tv", "pc"):
        args += ["-color_range", rng]
    for key, opt in (("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc"),
                     ("color_space", "-colorspace")):
        val = v.get(key)
        if val and val not in ("unknown", "reserved"):
            args += [opt, val]
    return args


def _bframe_variants(encoder, delay):
    """
    B-kocka beállítások, amelyekkel a kódoló a forráséval azonos dekódolási késleltetést
    (pts - dts az első csomagnál, képkockában) ad. A tényleges értéket mérjük, ezért
    több jelöltet is visszaadunk.
    """
    if encoder == "libx264":
        if delay <= 0:
            return [["-bf", "0"]]
        if delay == 1:
            return [["-bf", "1"], ["-bf", "2", "-x264-params", "b-pyramid=none"]]
        return [["-bf", "2", "-x264-params", "b-pyramid=normal"], ["-bf", "3", "-x264-params", "b-pyramid=normal"]]
    if encoder == "libx265":
        if delay <= 0:
            return [["bframes=0"]]
        if delay == 1:
            return [["bframes=2:b-pyramid=0"], ["bframes=1"]]
        return [["bframes=3:b-pyramid=1"], ["bframes=4"]]
    # hardveres kódolók: -bf a késleltetés szerint (a mérés dönt)
    if delay <= 0:
        return [["-bf", "0"]]
    if encoder.endswith("_nvenc") and delay >= 2:
        return [["-bf", str(delay), "-b_ref_mode", "middle"], ["-bf", str(delay)]]
    return [["-bf", str(delay)]]


def _part_pix_fmt(encoder, pix_fmt):
    """A kódolónak átadott pix_fmt (a qsv/amf nv12/p010 memóriaformátumot vár, a bitfolyam ugyanaz)."""
    if encoder.endswith(("_qsv", "_amf")):
        return {"yuv420p": "nv12", "yuv420p10le": "p010le"}.get(pix_fmt)
    if encoder.endswith("_nvenc"):
        return {"yuv420p": "yuv420p", "yuv420p10le": "p010le"}.get(pix_fmt)
    return pix_fmt


def _part_encoder_args(encoder, v, bvariant, threads_args):
    """A fej/farok rész kódolási paraméterei (a forrással egyező profil, szint, pix_fmt)."""
    codec = v["codec"]
    prof = _profile_arg(codec, v.get("profile"))
    level = _level_str(codec, v.get("level"))
    pix = _part_pix_fmt(encoder, v.get("pix_fmt"))
    if prof is None or pix is None:
        return None
    args = ["-c:v", encoder]
    if encoder == "libx264":
        args += ["-preset", "veryfast", "-crf", "16", "-profile:v", prof] + list(bvariant)
        if level:
            args += ["-level:v", level]
        args += threads_args
    elif encoder == "libx265":
        params = "log-level=error:open-gop=0:" + bvariant[0]
        if level:
            params += f":level-idc={level}"
        args += ["-preset", "veryfast", "-crf", "18", "-profile:v", prof, "-x265-params", params] + threads_args
    elif encoder.endswith("_nvenc"):
        args += ["-preset", "p4", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-profile:v", prof,
                 "-forced-idr", "1"] + list(bvariant)
    elif encoder.endswith("_qsv"):
        args += ["-preset", "veryfast", "-global_quality", "18", "-profile:v", prof] + list(bvariant)
    elif encoder.endswith("_amf"):
        args += ["-quality", "speed", "-rc", "cqp", "-qp_i", "18", "-qp_p", "18", "-qp_b", "20",
                 "-profile:v", prof] + list(bvariant)
    else:
        return None
    args += ["-pix_fmt", pix] + _color_args(v)
    if codec == "hevc":
        args += ["-tag:v", "hev1"]  # a paraméterkészletek a sávban is lehetnek (fej ↔ forrás váltás)
    return args


def _full_encoder_args(encoder, threads_args):
    """Teljes újrakódolás paraméterei (H.264, yuv420p – mint a korábbi pontos vágásnál)."""
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"] + threads_args
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "0", "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-c:v", encoder, "-preset", "veryfast", "-global_quality", "22", "-pix_fmt", "nv12"]
    if encoder == "h264_amf":
        return ["-c:v", encoder, "-quality", "speed", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22", "-qp_b", "24",
                "-pix_fmt", "nv12"]
    return None


# ------------------------------------------------------------------
# Teljes újrakódolás (tartalék út)
# ------------------------------------------------------------------

def full_reencode(src, start_sec, end_sec, out_path, ffmpeg_bin=None, frame_dur=None, has_audio=True,
                  encoder_override=None, seek_margin=0.0):
    """
    Képkocka-pontos vágás teljes újrakódolással (H.264 + AAC). A hardveres kódolót próbálja
    először, hibánál a libx264-et. start/end: mp a fájl elejétől (ffmpeg -ss értelmezés).
    seek_margin: nem indexelt konténernél (pl. MPEG-TS) az ffmpeg a kért pont UTÁNI
    kulcskockára is ugorhat – ennyivel korábbra ugrunk, és a pontos kezdetet a trim szűrő vágja.
    Visszatér: a használt kódoló neve.
    """
    ffmpeg_bin = resolve_ffmpeg(ffmpeg_bin)
    start_sec = float(start_sec)
    duration = max(0.04, float(end_sec) - start_sec)
    thr = background.ffmpeg_thread_args()
    # Negyed képkockával korábbról kérjük a bemenetet (lebegőpontos kerekítés ellen); a trim /
    # atrim szűrő vágja pontosan, a setpts / asetpts a kimenetet 0-ra igazítja.
    q = float(frame_dur) / 4.0 if frame_dur else 0.0
    coarse = max(0.0, start_sec - float(seek_margin) - q)
    vtrim = max(0.0, start_sec - coarse - q)
    atrim = max(0.0, start_sec - coarse)
    last_err = None
    for enc in encoder_candidates("h264", ffmpeg_bin, encoder_override):
        eargs = _full_encoder_args(enc, thr)
        if eargs is None:
            continue
        cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"] + thr + \
              ["-ss", _fmt(coarse), "-i", src,
               "-map", "0:v:0", "-vf", f"trim=start={vtrim:.6f},setpts=PTS-STARTPTS"]
        if has_audio:
            cmd += ["-map", "0:a:0?", "-af", f"atrim=start={atrim:.6f},asetpts=PTS-STARTPTS",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
        cmd += eargs + ["-t", _fmt(duration), "-movflags", "+faststart", out_path]
        res = _run(cmd)
        if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return enc
        last_err = _err_tail(res)
        log.warning("Teljes újrakódolás hiba (%s): %s", enc, last_err)
    raise RuntimeError("ffmpeg hiba: " + (last_err or "nincs használható kódoló"))


def _indexed(info):
    """Indexelt konténer (pontos keresés): mp4 / mov / matroska / webm."""
    fmt = (info or {}).get("format") or ""
    return any(x in fmt for x in ("mp4", "mov", "matroska", "webm"))


# ------------------------------------------------------------------
# Smart cut
# ------------------------------------------------------------------

def _keyframes(path, v, a_sec, b_sec, ffprobe_bin):
    """Kulcskockák (pts, dts) Fraction mp-ben + az összes csomag pts-e, egy ablakban."""
    tb = v["tb"]
    pk = probe_packets(path, v["index"], a_sec, b_sec, ffprobe_bin)
    keys = sorted({(Fraction(p) * tb, Fraction(d) * tb) for p, d, k in pk if k})
    pts = sorted({Fraction(p) * tb for p, _, _ in pk})
    return keys, pts


def _check_cfr(pts, fr):
    """Az ablak csomagjai egyenletes (CFR) képkocka-távolságúak-e."""
    if len(pts) < 3:
        return True
    tol = fr / 10
    return all(abs((b - a) - fr) <= tol for a, b in zip(pts, pts[1:]))


def plan_cut(info, start_sec, end_sec, ffprobe_bin=None, src=None):
    """
    A vágás terve. Visszatér: dict a fej / törzs / farok részekkel (abszolút Fraction mp), vagy
    FastCutError, ha a forrás nem alkalmas a gyors vágásra (a hívó teljes újrakódolásra vált).
    """
    v = info.get("video")
    if not v:
        raise FastCutError("nincs videósáv")
    if not _indexed(info):
        raise FastCutError(f"nem indexelt konténer ({info.get('format')}) – a pontos kereséshez újrakódolás kell")
    if v["codec"] not in SW_ENCODERS:
        raise FastCutError(f"nem támogatott kodek a gyors vágáshoz: {v['codec']}")
    if _profile_arg(v["codec"], v.get("profile")) is None:
        raise FastCutError(f"nem támogatott profil: {v.get('profile')}")
    if v.get("pix_fmt") not in PART_PIX_FMTS[v["codec"]]:
        raise FastCutError(f"nem támogatott pix_fmt: {v.get('pix_fmt')}")
    if v.get("field_order") not in ("progressive", "unknown"):
        raise FastCutError(f"váltott soros videó ({v.get('field_order')})")
    fps, avg = v.get("fps"), v.get("avg_fps")
    if not fps or fps <= 0 or fps > 240:
        raise FastCutError("ismeretlen képkockasebesség")
    if avg and abs(avg - fps) / fps > Fraction(1, 200):
        raise FastCutError(f"változó képkockasebesség (VFR: r={float(fps):.3f}, átlag={float(avg):.3f})")
    if v.get("start_pts") is None:
        raise FastCutError("ismeretlen kezdő időbélyeg")
    fr = 1 / fps
    half = fr / 2
    v0 = Fraction(v["start_pts"]) * v["tb"]
    fmt0 = Fraction(info.get("start") or 0.0).limit_denominator(1000000)

    total = v.get("nb_frames")
    if total is None and v.get("duration"):
        total = int(round(v["duration"] * float(fps)))
    s_idx = max(0, int(round(float(start_sec) * float(fps))))
    e_idx = int(round(float(end_sec) * float(fps)))
    eof = False
    if total is not None and e_idx >= total:
        e_idx, eof = total, True
    if e_idx - s_idx < 1:
        raise ValueError(f"Üres vágási szakasz ({start_sec} – {end_sec} mp).")
    S = v0 + s_idx * fr
    E = v0 + e_idx * fr

    # --- kezdőpont: az első kulcskocka >= S - fél képkocka ---
    w = KEY_WINDOW_SEC
    while True:
        keys, pts = _keyframes(src, v, float(S) - 1.0, float(min(S + w, E + 1)), ffprobe_bin)
        if not _check_cfr([p for p in pts if p >= S - 1], fr):
            raise FastCutError("egyenetlen képkocka-időbélyegek a kezdőpontnál (VFR / kimaradt kocka)")
        k1 = next((k for k in keys if k[0] >= S - half), None)
        if k1 is not None or S + w >= E or w >= KEY_WINDOW_MAX_SEC:
            break
        w *= 2
    if k1 is None or k1[0] >= E - half:
        raise FastCutError("a szakaszban nincs kulcskocka (rövid vágás) – teljes újrakódolás")
    delay = int(round((k1[0] - k1[1]) / fr))
    if delay < 0 or delay > 4:
        raise FastCutError(f"szokatlan dekódolási késleltetés: {delay}")
    head = None if abs(k1[0] - S) <= half else (S, k1[0])

    # --- végpont: az utolsó kulcskocka <= E + fél képkocka (és >= K1) ---
    tail = None
    kend = None
    if not eof:
        w = KEY_WINDOW_SEC
        while True:
            a = max(k1[0], E - w)
            keys_e, pts_e = _keyframes(src, v, float(a) - 0.5, float(E) + 0.5, ffprobe_bin)
            if not _check_cfr([p for p in pts_e if a - 1 <= p <= E + half], fr):
                raise FastCutError("egyenetlen képkocka-időbélyegek a végpontnál (VFR / kimaradt kocka)")
            cand = [k for k in keys_e if k1[0] - half <= k[0] <= E + half]
            if cand or a <= k1[0] or w >= KEY_WINDOW_MAX_SEC:
                break
            w *= 2
        if not cand:
            raise FastCutError("a végpont előtt nem található kulcskocka")
        kend = max(cand)
        if int(round((kend[0] - kend[1]) / fr)) != delay:
            raise FastCutError("a kulcskockák dekódolási késleltetése eltér")
        if abs(kend[0] - E) > half:
            tail = (kend[0], E)
    return {"fr": fr, "fps": fps, "v0": v0, "fmt0": fmt0, "S": S, "E": E, "s_idx": s_idx, "e_idx": e_idx,
            "frames": e_idx - s_idx, "eof": eof, "k1": k1, "kend": kend, "head": head, "tail": tail,
            "delay": delay}


def _probe_part(path, ffprobe_bin):
    res = _run([resolve_ffprobe(None, ffprobe_bin), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,time_base:packet=pts,dts",
                "-of", "json", path], timeout=60)
    if res.returncode != 0:
        return None
    return json.loads(res.stdout.decode("utf-8", "replace") or "{}")


def _encode_part(src, info, plan, seg, out_path, role, encoders, ffmpeg_bin, ffprobe_bin, timescale):
    """
    A fej vagy farok rész újrakódolása a forrással egyező paraméterekkel, és ellenőrzése
    (képkockaszám, kodek, profil, pix_fmt, felbontás, késleltetés). Visszatér: a kódoló neve.
    """
    v = info["video"]
    fr, fmt0 = plan["fr"], plan["fmt0"]
    n = int(round((seg[1] - seg[0]) / fr))
    want_delay = plan["delay"] if role == "head" else 0
    thr = background.ffmpeg_thread_args()
    ss = float(seg[0] - fmt0 - fr / 4)
    vf = "setpts=PTS-STARTPTS"
    sar = _frac(str(v.get("sar") or "").replace(":", "/"))
    if sar and sar != 1:
        vf += f",setsar={sar.numerator}/{sar.denominator}"
    # A concat demuxer sorszám szerint párosítja a sávokat: ha a forrásban a videó előtt más
    # sávok vannak (pl. a hang a 0.), a részbe is ugyanennyi sáv kerül elé (másolva).
    pre_maps = []
    types = info.get("stream_types") or []
    for i in range(int(v["index"])):
        if i >= len(types) or types[i] != "audio":
            raise FastCutError("a videósáv előtt nem hang típusú sáv van – a concat nem párosítható")
        pre_maps += ["-map", "1:a:0"]   # rövid csend (a végső kimenetbe nem kerül be)
    problems = []
    for enc in encoders:
        if enc.endswith(("_qsv", "_amf", "_nvenc")) and v.get("pix_fmt") == "yuvj420p":
            continue  # teljes tartományú (JPEG) színekhez csak a szoftveres kódolót használjuk
        for bvar in _bframe_variants(enc, want_delay):
            eargs = _part_encoder_args(enc, v, bvar, thr)
            if eargs is None:
                continue
            cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"] + thr + \
                  ["-ss", f"{max(0.0, ss):.6f}", "-i", src] + \
                  (["-f", "lavfi", "-t", "0.1", "-i", "anullsrc=r=48000:cl=stereo"] if pre_maps else []) + \
                  pre_maps + ["-map", f"0:{v['index']}"] + \
                  (["-an", "-sn", "-dn"] if not pre_maps else ["-c:a", "alac", "-sn", "-dn"]) + \
                  ["-filter:v", vf, "-frames:v", str(n)] + eargs + \
                  ["-video_track_timescale", str(timescale), "-map_metadata", "-1", out_path]
            res = _run(cmd)
            if res.returncode != 0 or not os.path.exists(out_path):
                problems.append(f"{enc}: {_err_tail(res, 1)}")
                break  # ez a kódoló nem működik – a következő
            p = _probe_part(out_path, ffprobe_bin) or {}
            st = (p.get("streams") or [{}])[0]
            pk = p.get("packets") or []
            tb = _frac(st.get("time_base"), Fraction(1, 90000))
            got_delay = None
            if pk and pk[0].get("pts") is not None and pk[0].get("dts") is not None:
                got_delay = int(round((Fraction(int(pk[0]["pts"])) - int(pk[0]["dts"])) * tb / fr))
            checks = {
                "képkockaszám": len(pk) == n,
                "kodek": st.get("codec_name") == v["codec"],
                "profil": _same_profile(st.get("profile"), v.get("profile")),
                "pix_fmt": st.get("pix_fmt") == v.get("pix_fmt"),
                "felbontás": (st.get("width"), st.get("height")) == (v.get("width"), v.get("height")),
                # fej: a késleltetés legalább a forrásé (a dts így monoton marad a csatlakozásnál),
                # farok: legfeljebb a forrásé (B-kocka nélkül 0)
                "késleltetés": got_delay is not None and (got_delay >= want_delay if role == "head"
                                                          else got_delay <= plan["delay"]),
            }
            bad = [k for k, ok in checks.items() if not ok]
            if not bad:
                return enc, n
            problems.append(f"{enc} {' '.join(bvar)}: eltér ({', '.join(bad)}; "
                            f"{len(pk)}/{n} kocka, késleltetés {got_delay}/{want_delay})")
            if "késleltetés" not in bad:
                break  # más B-kocka beállítás nem segít
    raise FastCutError(f"a(z) {role} rész nem kódolható egyezően: " + "; ".join(problems[-3:]))


def _junction_ok(path, times, fr, ffmpeg_bin, ffprobe_bin, max_gap_ratio=None):
    """
    A csatlakozási pontok körül: monoton dts, egyedi és egyenletes pts, hibátlan dekódolás.
    max_gap_ratio: ha meg van adva, a csatlakozásnál a képkocka-lépés legfeljebb ennyiszerese
    lehet a szokásosnak (több fájl összefűzésénél a hang pár ms-mal hosszabb lehet a képnél,
    ezért a következő rész egy hajszállal később kezdődik – ez nem hiba).
    """
    for t in times:
        t = float(t)
        res = _run([resolve_ffprobe(None, ffprobe_bin), "-v", "error", "-select_streams", "v:0",
                    "-read_intervals", f"{max(0.0, t - 1.0):.6f}%{t + 1.0:.6f}",
                    "-show_entries", "packet=pts,dts", "-of", "csv=p=0", path], timeout=60)
        rows = []
        for line in res.stdout.decode("utf-8", "replace").splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
                rows.append((int(parts[0]), int(parts[1])))
        if not rows:
            return f"nincs csomag a {t:.2f} mp-es csatlakozásnál"
        dts = [d for _, d in rows]
        if any(b <= a for a, b in zip(dts, dts[1:])):
            return f"nem monoton dts a {t:.2f} mp-es csatlakozásnál"
        pts = sorted(p for p, _ in rows)
        if len(set(pts)) != len(pts):
            return f"ismétlődő pts a {t:.2f} mp-es csatlakozásnál"
        steps = {b - a for a, b in zip(pts, pts[1:])}
        if max_gap_ratio and steps and min(steps) > 0 and max(steps) <= min(steps) * max_gap_ratio:
            steps = {min(steps)}
        if len(steps) > 1 and (max(steps) - min(steps)) > max(1, min(steps) // 10):
            return f"egyenetlen pts-lépés a {t:.2f} mp-es csatlakozásnál: {sorted(steps)[:4]}"
        dec = _run([ffmpeg_bin, "-hide_banner", "-v", "error", "-nostdin", "-ss", f"{max(0.0, t - 0.5):.6f}",
                    "-i", path, "-t", "1.5", "-map", "0:v:0", "-f", "null", "-"], timeout=120)
        if dec.returncode != 0 or dec.stderr.strip():
            return f"dekódolási hiba a {t:.2f} mp-es csatlakozásnál: {_err_tail(dec, 1)}"
    return None


def _first_packet_time(path, selector, ffprobe_bin):
    res = _run([resolve_ffprobe(None, ffprobe_bin), "-v", "error", "-select_streams", selector,
                "-read_intervals", "%+#1", "-show_entries", "packet=pts_time", "-of", "csv=p=0", path], timeout=60)
    txt = res.stdout.decode("utf-8", "replace").strip().split("\n")[0].strip().rstrip(",")
    return _float(txt)


def _verify(out_path, info, plan, junctions, ffmpeg_bin, ffprobe_bin):
    """A kész fájl ellenőrzése. Visszatér: None (rendben) vagy a hiba szövege."""
    try:
        o = probe(out_path, ffprobe_bin, use_cache=False)
    except FastCutError as ex:
        return str(ex)
    v, ov = info["video"], o.get("video")
    if not ov:
        return "nincs videósáv a kimenetben"
    if ov["codec"] != v["codec"] or (ov["width"], ov["height"]) != (v["width"], v["height"]):
        return "a kimenet kodekje / felbontása eltér"
    n = count_video_packets(out_path, ffprobe_bin)
    if n != plan["frames"]:
        return f"képkockaszám eltér: {n} != {plan['frames']}"
    fr = float(plan["fr"])
    vstart = ov.get("start") or 0.0
    if abs(vstart) > fr / 2:
        return f"a videó nem 0-nál kezdődik ({vstart:.3f} mp)"
    if info.get("audio"):
        oa = o.get("audio")
        if not oa:
            return "hiányzik a hangsáv"
        if abs((oa.get("start") or 0.0) - vstart) > fr:
            return f"A/V kezdet eltér ({oa.get('start')} / {vstart})"
        first = _first_packet_time(out_path, "a:0", ffprobe_bin)
        if first is not None and first < -fr:
            return f"rejtett hang-előtag a fájl elején ({first:.3f} mp)"
        vd = ov.get("duration") or plan["frames"] * fr
        if oa.get("duration") is not None and abs(oa["duration"] - vd) > max(0.15, 3 * fr):
            # a forrás végén a hang lehet rövidebb – csak a fájl végéig tartó vágásnál engedjük
            if not (plan["eof"] and oa["duration"] < vd):
                return f"A/V hossz eltér ({oa['duration']:.3f} / {vd:.3f} mp)"
    return _junction_ok(out_path, junctions, plan["fr"], ffmpeg_bin, ffprobe_bin)


def _timescale(info):
    v = info["video"]
    tb = v["tb"]
    if any(x in (info.get("format") or "") for x in ("mp4", "mov")) and tb.numerator == 1 and tb.denominator >= 600:
        return tb.denominator
    return 90000


def _extract_audio(src, info, plan, out_path, ffmpeg_bin):
    """
    A hang a teljes szakaszra. Másolásnál a bemeneti -ss a videó kulcskockájához ugrik, és a
    korábbi hangcsomagokat is átvinné (edit listtel elrejtve – az összefűzést elrontaná), ezért
    kicsit korábbról olvasunk, és a kimeneti -ss dobja el a felesleget (≤ 1 hangkeret pontosság).
    Nem másolható kodeknél AAC-ra kódolunk (pontos bemeneti kereséssel).
    """
    a = info["audio"]
    s_rel = float(plan["S"] - plan["fmt0"])
    total = float(plan["frames"] * plan["fr"])
    if a.get("codec") in AUDIO_COPY_CODECS:
        coarse = max(0.0, s_rel - 0.5)
        cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", _fmt(coarse), "-i", src,
               "-map", f"0:{a['index']}", "-vn", "-c:a", "copy", "-ss", _fmt(s_rel - coarse), "-t", _fmt(total),
               out_path]
    else:
        cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"] + background.ffmpeg_thread_args() + \
              ["-ss", _fmt(s_rel), "-i", src, "-map", f"0:{a['index']}", "-vn", "-c:a", "aac", "-b:a", "160k",
               "-t", _fmt(total), out_path]
    res = _run(cmd)
    if res.returncode != 0 or not os.path.exists(out_path):
        raise FastCutError(f"a hangsáv kivágása nem sikerült: {_err_tail(res)}")
    return out_path


def _run_parallel(tasks, split_threads):
    """
    Feladatok (szerep → függvény) párhuzamos futtatása a háttér-beállítás átvitelével.
    split_threads: ennyi CPU-igényes kódolás fut egyszerre – szálkorlát nélkül a magokat elosztjuk.
    """
    if not tasks:
        return {}
    cfg = background.current()
    enabled = bool(cfg and cfg.enabled)
    threads = cfg.threads if cfg else None
    if split_threads > 1 and not threads:
        threads = max(1, (os.cpu_count() or 2) // split_threads)

    def work(fn):
        with background.context(enabled, threads):  # a szálhoz kötött háttér-beállítás a munkaszálba
            return fn()

    if len(tasks) == 1:
        role, fn = next(iter(tasks.items()))
        return {role: work(fn)}
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futs = {role: ex.submit(work, fn) for role, fn in tasks.items()}
        return {role: f.result() for role, f in futs.items()}


def _smart(src, info, plan, out_tmp, work, ffmpeg_bin, ffprobe_bin, encoder_override):
    v = info["video"]
    fr, fmt0 = plan["fr"], plan["fmt0"]
    encoders = encoder_candidates(v["codec"], ffmpeg_bin, encoder_override)
    ts = _timescale(info)
    used = []
    entries = []      # (fájl, inpoint, outpoint, duration)
    junctions = []
    pos = Fraction(0)
    head_n = tail_n = 0
    hp, tp = os.path.join(work, "fej.mp4"), os.path.join(work, "farok.mp4")
    parts = {}
    if plan["head"]:
        parts["head"] = (plan["head"], hp)
    if plan["tail"]:
        parts["tail"] = (plan["tail"], tp)
    tasks = {}
    for role, (seg, path) in parts.items():
        tasks[role] = (lambda role=role, seg=seg, path=path:
                       _encode_part(src, info, plan, seg, path, role, encoders, ffmpeg_bin, ffprobe_bin, ts))
    ap = os.path.join(work, "hang.mp4")
    if info.get("audio"):
        tasks["audio"] = lambda: _extract_audio(src, info, plan, ap, ffmpeg_bin)
    # a fej, a farok és a hang egyszerre (a hang főleg I/O, a két kódolás CPU)
    done = _run_parallel(tasks, split_threads=len(parts))
    if plan["head"]:
        enc, head_n = done["head"]
        used.append(enc)
        entries.append((hp, None, None, head_n * fr))
        pos += head_n * fr
        junctions.append(pos)
    k1 = plan["k1"][0]
    kend = plan["kend"]
    body_end = None if plan["eof"] else kend[0]
    if body_end is None or body_end - k1 > fr / 2:
        body_dur = None if body_end is None else body_end - k1
        entries.append((src, _ceil_us(k1), None if body_end is None else _floor_us(kend[1]), body_dur))
        if body_dur is not None:
            pos += body_dur
    if plan["tail"]:
        if pos > 0 and (not junctions or junctions[-1] != pos):
            junctions.append(pos)
        enc, tail_n = done["tail"]
        used.append(enc)
        entries.append((tp, None, None, None))

    list_path = os.path.join(work, "lista.ffconcat")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for path, inp, outp, dur in entries:
            f.write(f"file '{_safe_path(path)}'\n")
            if inp is not None:
                f.write(f"inpoint {inp}\n")
            if outp is not None:
                f.write(f"outpoint {outp}\n")
            if dur is not None:
                f.write(f"duration {_fmt(dur)}\n")
    total = plan["frames"] * fr
    thr = background.ffmpeg_thread_args()
    # -copyts: az időbélyegek változatlanok maradnak (a concat 0-tól ad, a hang a pontos helyén
    # van) – enélkül az ffmpeg bemenetenként a saját kezdetére tolna, ami a B-kockás videónál és
    # a pár ms-mal később kezdődő hangnál elcsúsztatná az A/V szinkront.
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"] + thr + \
          ["-copyts", "-f", "concat", "-safe", "0", "-i", list_path]
    if info.get("audio"):
        cmd += ["-i", ap]
    cmd += ["-map", "0:v:0", "-c:v", "copy"]
    if info.get("audio"):
        cmd += ["-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-movflags", "+faststart", out_tmp]
    res = _run(cmd)
    if res.returncode != 0 or not os.path.exists(out_tmp):
        raise FastCutError(f"összefűzési hiba: {_err_tail(res)}")
    err = _verify(out_tmp, info, plan, junctions, ffmpeg_bin, ffprobe_bin)
    if err:
        raise FastCutError(f"ellenőrzés: {err}")
    return {"encoder": used[0] if used else None, "head_frames": head_n, "tail_frames": tail_n}


def smart_cut(src, start_sec, end_sec, out_path, ffmpeg_bin=None, ffprobe_bin=None, encoder=None,
              mode=None, work_dir=None):
    """
    Képkocka-pontos vágás a lehető legkevesebb újrakódolással.
    start_sec / end_sec: mp a videó első képkockájától (ahogy az elemzés / OpenCV számolja).
    mode: None / "auto" (alap; FAST_CUT_MODE), "full" = mindig teljes újrakódolás.
    Visszatér: {"mode": "copy" | "smart" | "full", "encoder", "head_frames", "tail_frames",
                "frames", "start_frame", "elapsed_sec", "reason"}.
    Hibánál RuntimeError (magyar üzenettel); a kimeneten soha nem marad hibás fájl.
    """
    t0 = time.monotonic()
    ffmpeg_bin = resolve_ffmpeg(ffmpeg_bin)
    if not os.path.exists(src):
        raise RuntimeError(f"A videófájl nem található: {src}")
    mode = (mode or os.environ.get("FAST_CUT_MODE") or "auto").strip().lower()
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="fastcut_", dir=work_dir or out_dir)
    out_tmp = os.path.join(work, "kimenet.mp4")
    result = {"mode": None, "encoder": None, "head_frames": 0, "tail_frames": 0, "frames": None,
              "start_frame": None, "reason": None}
    info = plan = None
    try:
        reason = None
        if mode == "full":
            reason = "FAST_CUT_MODE=full"
            try:
                info = probe(src, ffprobe_bin)  # csak az igazításhoz (első képkocka, hang)
            except FastCutError:
                info = None
        else:
            try:
                info = probe(src, ffprobe_bin)
                plan = plan_cut(info, start_sec, end_sec, ffprobe_bin, src=src)
                result["frames"], result["start_frame"] = plan["frames"], plan["s_idx"]
                r = _smart(src, info, plan, out_tmp, work, ffmpeg_bin, ffprobe_bin, encoder)
                result.update(r)
                result["mode"] = "smart" if (r["head_frames"] or r["tail_frames"]) else "copy"
            except FastCutError as ex:
                reason = str(ex)
        if result["mode"] is None:
            log.info("Teljes újrakódolás (%s): %s", os.path.basename(src), reason)
            result["reason"] = reason
            fr = plan["fr"] if plan else None
            if fr is None and info and info.get("video") and info["video"].get("fps"):
                fr = 1 / info["video"]["fps"]
            has_audio = True if info is None else bool(info.get("audio"))
            # A kért időket az első képkockához és a képkocka-rácshoz igazítjuk (ahogy a smart út is)
            off = 0.0
            s, e = float(start_sec), float(end_sec)
            if info and info.get("video"):
                v = info["video"]
                off = (v.get("start") or 0.0) - (info.get("start") or 0.0)
                if v.get("fps"):
                    f = float(v["fps"])
                    s, e = round(s * f) / f, round(e * f) / f
                    if v.get("duration"):
                        e = min(e, float(v["duration"]) + 1.0 / f)
            s, e = s + off, e + off
            if plan:
                s = float(plan["S"] - plan["fmt0"])
                e = float(plan["E"] - plan["fmt0"])
            if os.path.exists(out_tmp):
                os.remove(out_tmp)
            margin = 0.0 if (info is not None and _indexed(info)) else 10.0
            result["encoder"] = full_reencode(src, s, e, out_tmp, ffmpeg_bin, frame_dur=fr, has_audio=has_audio,
                                              encoder_override=encoder, seek_margin=margin)
            result["mode"] = "full"
        os.replace(out_tmp, out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    result["elapsed_sec"] = round(time.monotonic() - t0, 3)
    log.info("Vágás kész (%s, %s): %s", result["mode"], result["encoder"], os.path.basename(out_path))
    return result


# ------------------------------------------------------------------
# Több rész összefűzése
# ------------------------------------------------------------------

def _signature(info):
    v, a = info.get("video") or {}, info.get("audio") or {}
    return (v.get("codec"), v.get("width"), v.get("height"), v.get("pix_fmt"), v.get("fps"),
            a.get("codec"), a.get("sample_rate"), a.get("channels"), bool(info.get("audio")))


def concat_files(paths, out_path, ffmpeg_bin=None, ffprobe_bin=None, verify=True):
    """
    Több (azonos paraméterű) videó összefűzése újrakódolás nélkül (concat demuxer, -c copy),
    ellenőrzéssel; eltérő paramétereknél vagy hibánál újrakódolva (concat szűrő, H.264 + AAC).
    Visszatér: {"mode": "copy" | "reencode", "reason", "elapsed_sec"}.
    """
    t0 = time.monotonic()
    ffmpeg_bin = resolve_ffmpeg(ffmpeg_bin)
    out_path = os.path.abspath(out_path)
    work = tempfile.mkdtemp(prefix="fastcat_", dir=os.path.dirname(out_path))
    out_tmp = os.path.join(work, "osszefuzott.mp4")
    result = {"mode": None, "reason": None}
    try:
        infos = []
        try:
            infos = [probe(p, ffprobe_bin, use_cache=False) for p in paths]
        except FastCutError as ex:
            result["reason"] = str(ex)
        list_path = os.path.join(work, "lista.ffconcat")
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("ffconcat version 1.0\n")
            for p in paths:
                f.write(f"file '{_safe_path(p)}'\n")
        same = bool(infos) and len({_signature(i) for i in infos}) == 1 and all(i.get("video") for i in infos)
        if same:
            res = _run([ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-f", "concat", "-safe", "0", "-i", list_path, "-map", "0:v:0", "-map", "0:a:0?",
                        "-c", "copy", "-movflags", "+faststart", out_tmp])
            if res.returncode == 0 and os.path.exists(out_tmp):
                err = None
                if verify:
                    expected = [count_video_packets(p, ffprobe_bin) for p in paths]
                    got = count_video_packets(out_tmp, ffprobe_bin)
                    if None in expected or got != sum(expected):
                        err = f"képkockaszám eltér: {got} != {sum(e or 0 for e in expected)}"
                    else:
                        fr = 1 / infos[0]["video"]["fps"] if infos[0]["video"].get("fps") else Fraction(1, 25)
                        times, pos = [], Fraction(0)
                        for n in expected[:-1]:
                            pos += n * fr
                            times.append(pos)
                        o = probe(out_tmp, ffprobe_bin, use_cache=False)
                        if infos[0].get("audio") and not o.get("audio"):
                            err = "hiányzik a hangsáv"
                        else:
                            err = _junction_ok(out_tmp, times, fr, ffmpeg_bin, ffprobe_bin, max_gap_ratio=1.5)
                if not err:
                    result["mode"] = "copy"
                else:
                    result["reason"] = f"ellenőrzés: {err}"
            else:
                result["reason"] = f"összefűzési hiba: {_err_tail(res)}"
        elif not result["reason"]:
            result["reason"] = "eltérő paraméterű részek"
        if result["mode"] is None:
            log.info("Összefűzés újrakódolással: %s", result["reason"])
            _concat_reencode(paths, infos, out_tmp, ffmpeg_bin)
            result["mode"] = "reencode"
        os.replace(out_tmp, out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    result["elapsed_sec"] = round(time.monotonic() - t0, 3)
    return result


def _concat_reencode(paths, infos, out_tmp, ffmpeg_bin):
    """Összefűzés concat szűrővel (minden rész az első felbontására / fps-ére igazítva)."""
    first = (infos[0].get("video") if infos else None) or {}
    w, h = first.get("width") or 1280, first.get("height") or 720
    fps = first.get("fps") or Fraction(25)
    all_audio = bool(infos) and all(i.get("audio") for i in infos)
    thr = background.ffmpeg_thread_args()
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"] + thr
    for p in paths:
        cmd += ["-i", p]
    chains, labels = [], ""
    for i in range(len(paths)):
        chains.append(f"[{i}:v:0]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                      f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps.numerator}/{fps.denominator},"
                      f"format=yuv420p[v{i}]")
        labels += f"[v{i}]"
        if all_audio:
            chains.append(f"[{i}:a:0]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
            labels += f"[a{i}]"
    graph = ";".join(chains) + f";{labels}concat=n={len(paths)}:v=1:a={1 if all_audio else 0}[v]" + \
            ("[a]" if all_audio else "")
    last_err = None
    for enc in encoder_candidates("h264", ffmpeg_bin):
        eargs = _full_encoder_args(enc, thr)
        if eargs is None:
            continue
        c = cmd + ["-filter_complex", graph, "-map", "[v]"]
        if all_audio:
            c += ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
        c += eargs + ["-movflags", "+faststart", out_tmp]
        res = _run(c)
        if res.returncode == 0 and os.path.exists(out_tmp):
            return enc
        last_err = _err_tail(res)
    raise RuntimeError("ffmpeg összefűzési hiba: " + (last_err or "nincs használható kódoló"))

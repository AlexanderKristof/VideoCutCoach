"""
Klipek kivágása az EREDETI videóból ffmpeg-gel, opcionális összefűzéssel.

- Bemenet: időablakok listája ({start_sec, end_sec} vagy {time} – utóbbinál az
  ablak time - pre_sec .. time + post_sec).
- Az egymást átfedő vagy egymáshoz közeli (< 1 mp rés) ablakokat előbb összevonjuk,
  hogy ugyanaz a jelenet ne kerüljön kétszer a kész videóba.
- Pontos vágás (accurate=True, alapértelmezett): újrakódolás (libx264 veryfast +
  aac), így a vágás képkocka-pontos, és a darabok kódolási paraméterei egyeznek –
  a concat demuxer újrakódolás nélkül összefűzi őket.
- Gyors vágás (accurate=False): stream copy – nagyon gyors, de csak kulcskockánál
  tud vágni, ezért a klip eleje néhány másodperccel eltolódhat.
"""
import os
import re
import uuid
import subprocess

import cv2

FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg.exe")

MERGE_GAP_SEC = 1.0
DEFAULT_PRE_SEC = 5.0
DEFAULT_POST_SEC = 4.0


def _num(value, name):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Érvénytelen szám a(z) '{name}' mezőben: {value!r}")
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"Érvénytelen szám a(z) '{name}' mezőben: {value!r}")
    return v


def parse_windows(items, pre_sec=DEFAULT_PRE_SEC, post_sec=DEFAULT_POST_SEC, video_duration=None):
    """
    items: [{"start_sec": .., "end_sec": ..} | {"time": ..}, ...]
    Visszatér: [(start, end), ...] – 0-tól, illetve a videó hosszáig vágva.
    Hibás elemnél ValueError (magyar üzenettel).
    """
    if not isinstance(items, list):
        raise ValueError("Az events_json egy lista kell legyen.")
    windows = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"A(z) {i + 1}. elem nem objektum.")
        if "start_sec" in it or "end_sec" in it:
            if "start_sec" not in it or "end_sec" not in it:
                raise ValueError(f"A(z) {i + 1}. elemből hiányzik a start_sec vagy az end_sec.")
            s, e = _num(it["start_sec"], "start_sec"), _num(it["end_sec"], "end_sec")
        elif "time" in it:
            t = _num(it["time"], "time")
            s, e = t - pre_sec, t + post_sec
        else:
            raise ValueError(f"A(z) {i + 1}. elemben nincs start_sec/end_sec vagy time mező.")
        s = max(0.0, s)
        if video_duration is not None and video_duration > 0:
            e = min(e, video_duration)
        if e - s < 0.05:
            raise ValueError(f"A(z) {i + 1}. elem üres vagy fordított időablak ({s:.2f} – {e:.2f} mp).")
        windows.append((s, e))
    return windows


def merge_windows(windows, gap_sec=MERGE_GAP_SEC):
    """Időrendbe teszi és összevonja az átfedő / gap_sec-nél közelebbi ablakokat."""
    merged = []
    for s, e in sorted(windows):
        if merged and s - merged[-1][1] < gap_sec:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def video_duration_sec(video_path):
    """A videó hossza OpenCV-vel (None, ha nem állapítható meg)."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    finally:
        cap.release()
    if fps > 0 and frames > 0:
        return frames / fps
    return None


def _run(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(f"Az ffmpeg nem található ({FFMPEG}). Állítsd be az FFMPEG_PATH környezeti változót.")
    if res.returncode != 0:
        tail = res.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg hiba: " + " | ".join(tail))


def cut_clip(video_path, start_sec, end_sec, out_path, accurate=True):
    """Egy klip kivágása. accurate=True: újrakódolás (képkocka-pontos), False: stream copy."""
    duration = max(0.05, end_sec - start_sec)
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start_sec:.3f}", "-i", video_path, "-t", f"{duration:.3f}",
           "-map", "0:v:0", "-map", "0:a:0?"]
    if accurate:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    cmd += ["-movflags", "+faststart", out_path]
    _run(cmd)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("Az ffmpeg nem hozta létre a klipet.")


def concat_clips(paths, out_path):
    """Összefűzés a concat demuxerrel, újrakódolás nélkül."""
    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            safe = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    try:
        _run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
              "-i", list_path, "-c", "copy", "-movflags", "+faststart", out_path])
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def _slug(text, max_len=40):
    text = (text or "").translate(str.maketrans("áéíóöőúüűÁÉÍÓÖŐÚÜŰ", "aeiooouuuAEIOOOUUU"))
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:max_len]


def export_clips(video_path, items, out_dir, pre_sec=DEFAULT_PRE_SEC, post_sec=DEFAULT_POST_SEC,
                 merge=True, accurate=True, title=None, url_prefix="/clips", gap_sec=MERGE_GAP_SEC,
                 progress_cb=None):
    """
    Teljes folyamat: ellenőrzés → ablakok → összevonás → vágás → (összefűzés).
    Soha nem dob kivételt: hiba esetén {"status": "error", "message": ...}.
    progress_cb(kész, összes): minden kivágott klip után (háttérfeladatok haladásjelzéséhez);
    a horog kivétele (pl. megszakítás) továbbmegy a hívóhoz.
    """
    if not video_path or not os.path.exists(video_path):
        return {"status": "error", "message": "A videófájl nem található."}
    if pre_sec < 0 or post_sec < 0:
        return {"status": "error", "message": "A pre_sec és post_sec nem lehet negatív."}
    if not items:
        return {"status": "error", "message": "Nincs kivágandó esemény (üres lista)."}

    try:
        windows = merge_windows(parse_windows(items, pre_sec, post_sec, video_duration_sec(video_path)), gap_sec)
    except ValueError as ex:
        return {"status": "error", "message": str(ex)}
    if not windows:
        return {"status": "error", "message": "Nincs érvényes időablak."}

    os.makedirs(out_dir, exist_ok=True)
    job = uuid.uuid4().hex[:6]
    name = _slug(title)
    base = f"{name}_{job}" if name else f"export_{job}"

    clips = []
    try:
        for i, (s, e) in enumerate(windows, 1):
            fname = f"{base}_{i:02d}.mp4"
            out_path = os.path.abspath(os.path.join(out_dir, fname))
            cut_clip(video_path, s, e, out_path, accurate)
            clips.append({"start_sec": round(s, 3), "end_sec": round(e, 3),
                          "url": f"{url_prefix}/{fname}", "local_path": out_path})
            if progress_cb is not None:
                progress_cb(i, len(windows))

        result = {"status": "success", "clips": clips, "merged_url": None, "merged_local_path": None}
        if merge and clips:
            if len(clips) == 1:
                # Egyetlen klipnél nincs mit összefűzni – maga a klip a kész videó
                result["merged_url"], result["merged_local_path"] = clips[0]["url"], clips[0]["local_path"]
            else:
                mname = f"{base}_osszefuzott.mp4"
                mpath = os.path.abspath(os.path.join(out_dir, mname))
                concat_clips([c["local_path"] for c in clips], mpath)
                result["merged_url"], result["merged_local_path"] = f"{url_prefix}/{mname}", mpath
        return result
    except RuntimeError as ex:
        return {"status": "error", "message": str(ex), "clips": clips}

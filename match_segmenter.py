"""
A 90 perc automatikus kivágása: a játékrészek (1. és 2. félidő) felismerése
egy teljes Veo felvételből (bemelegítés, 1. félidő, szünet, 2. félidő, meccs utáni rész).

Jelek (olcsón, a teljes videón):
  a) Hang: sípszó keresése (tartós, keskeny sávú energia kb. 2,5–4,5 kHz között).
  b) Kép: másodpercenként 1 képkocka kicsinyítve – fű arány, mozgás a füvön
     (kameramozgás kompenzálva), kamera pásztázás. Opcionálisan létszám
     (person_counter(frame) -> int horog, pl. YOLO a main.py-ból).

A szegmentálás tiszta függvényekkel történik (tesztelhető), az ffmpeg csak a
dekódoláshoz és a vágáshoz kell. Új függőség nincs (numpy + OpenCV).
"""
import os
import uuid
import math
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from pitch_filters import build_grass_mask

# Az ffmpeg futtatható állomány neve – FFMPEG_PATH környezeti változóval felülírható.
# Windows-on (a kliens gépén) alapból "ffmpeg.exe", máshol (pl. teszt) "ffmpeg".
FFMPEG_BIN = os.environ.get("FFMPEG_PATH") or ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

AUDIO_SR = 16000
DEFAULT_GRASS_LOWER = (30, 40, 40)
DEFAULT_GRASS_UPPER = (85, 255, 255)


# ============================================================
# Segédfüggvények
# ============================================================

def format_hms(sec):
    """Másodperc -> 'ó:pp:mm' (vagy 'pp:mm' egy óra alatt) szöveg a kliensnek."""
    if sec is None:
        return ""
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_time_value(value):
    """
    Kézi időpont értelmezése: '754.5', '12:34', '1:02:03' vagy üres.
    Üres / hiányzó / hibás -> None (azaz: automatikus felismerés).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        if ":" in text:
            parts = [float(p) for p in text.split(":")]
            total = 0.0
            for p in parts:
                total = total * 60.0 + p
            return total if total >= 0 else None
        v = float(text)
        return v if v >= 0 else None
    except ValueError:
        return None


def _runs(mask):
    """Egybefüggő igaz szakaszok: [(kezdő index, záró index kizárólagosan), ...]."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _moving_average(x, win):
    """Középre igazított mozgóátlag, a széleken a ténylegesen átfedő mintákkal osztva."""
    x = np.asarray(x, dtype=np.float64)
    win = int(max(1, win))
    if win <= 1 or x.size == 0:
        return x.copy()
    k = np.ones(win)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones_like(x), k, mode="same")
    return num / np.maximum(den, 1e-9)


def _otsu_threshold(values, bins=64):
    """Otsu-küszöb egydimenziós (kétpúpú) eloszlásra."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) < 1e-9:
        return float(np.mean(v)) if v.size else 0.5
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mean0 = m0 / np.maximum(w0, 1)
    mean1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mean0 - mean1) ** 2
    between[(w0 == 0) | (w1 == 0)] = -1
    return float(edges[int(np.argmax(between)) + 1])


# ============================================================
# a) Hang – sípszó felismerés
# ============================================================

def audio_frame_features(samples, sr=AUDIO_SR, win_sec=0.05, hop_sec=0.025,
                         band=(2500.0, 4500.0), narrow_hz=150.0, start_index=0, chunk_frames=4096):
    """
    Rövid idejű spektrum jellemzők (kb. 50 ms ablak, 25 ms lépés).
    Visszatér: dict tömbökkel – time (ablak közepe, mp), band_ratio (sávenergia / teljes),
    tonality (a csúcs körüli keskeny sáv / sávenergia), peak_hz, level_db.
    start_index: az első ablak globális sorszáma (folyamatos, darabolt feldolgozáshoz).
    """
    x = np.asarray(samples, dtype=np.float32)
    n = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))
    empty = {k: np.zeros(0) for k in ("time", "band_ratio", "tonality", "peak_hz", "level_db")}
    if x.size < n:
        return empty
    n_frames = (x.size - n) // hop + 1
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    valid = freqs >= 150.0                       # szél, mélyzaj kihagyása
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    band_idx = np.flatnonzero(in_band)
    half_w = max(1, int(round(narrow_hz / (freqs[1] - freqs[0]))))
    window = np.hanning(n).astype(np.float32)
    frames_view = np.lib.stride_tricks.sliding_window_view(x, n)[::hop][:n_frames]

    out = {k: [] for k in ("band_ratio", "tonality", "peak_hz", "level_db")}
    for c0 in range(0, n_frames, chunk_frames):
        fr = frames_view[c0:c0 + chunk_frames] * window
        spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        total = spec[:, valid].sum(axis=1) + 1e-12
        band_spec = spec[:, band_idx]
        band_e = band_spec.sum(axis=1) + 1e-12
        peak_local = np.argmax(band_spec, axis=1)
        # keskeny sáv a csúcs körül (kumulált összeggel, ciklus nélkül)
        cs = np.concatenate([np.zeros((band_spec.shape[0], 1)), np.cumsum(band_spec, axis=1)], axis=1)
        lo = np.clip(peak_local - half_w, 0, band_spec.shape[1])
        hi = np.clip(peak_local + half_w + 1, 0, band_spec.shape[1])
        rows = np.arange(band_spec.shape[0])
        narrow = cs[rows, hi] - cs[rows, lo]
        out["band_ratio"].append(band_e / total)
        out["tonality"].append(narrow / band_e)
        out["peak_hz"].append(freqs[band_idx][peak_local])
        out["level_db"].append(10.0 * np.log10(total / n))

    res = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    idx = np.arange(start_index, start_index + n_frames)
    res["time"] = (idx * hop + n / 2.0) / sr
    return res


def detect_whistles_from_features(feat, hop_sec=0.025, min_duration_sec=0.3,
                                  min_band_ratio=0.35, min_tonality=0.45, min_level_above_floor_db=3.0,
                                  max_gap_sec=0.06, group_gap_sec=1.0):
    """
    Sípszók a spektrum jellemzőkből: tartósan (>= min_duration_sec) erős, keskeny sávú
    energia a sípszó sávban. Az egymást 1 mp-en belül követő sípolások (pl. a félidő végi
    2–3 hosszú sípszó) egy csoportba kerülnek ('blasts').
    Visszatér: [{"time_sec","end_sec","duration_sec","strength","freq_hz","blasts"}, ...]
    """
    t = np.asarray(feat.get("time", []))
    if t.size == 0:
        return []
    ratio = feat["band_ratio"]
    tonal = feat["tonality"]
    level = feat["level_db"]
    floor = float(np.median(level))
    cand = (ratio >= min_band_ratio) & (tonal >= min_tonality) & (level >= floor + min_level_above_floor_db)

    # rövid kiesések (pl. a síp golyójának rezgése) áthidalása
    max_gap = int(round(max_gap_sec / hop_sec))
    if max_gap > 0:
        for s, e in _runs(~cand):
            if s > 0 and e < cand.size and (e - s) <= max_gap:
                cand[s:e] = True

    blasts = []
    for s, e in _runs(cand):
        dur = (e - s) * hop_sec
        if dur < min_duration_sec:
            continue
        r = float(np.mean(ratio[s:e]))
        tn = float(np.mean(tonal[s:e]))
        loud = float(np.clip((np.mean(level[s:e]) - floor) / 20.0, 0.0, 1.0))
        strength = r * tn * (0.5 + 0.5 * min(dur, 1.5) / 1.5) * (0.6 + 0.4 * loud)
        blasts.append({
            "time_sec": float(t[s] - hop_sec),
            "end_sec": float(t[e - 1] + hop_sec),
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "freq_hz": float(np.median(feat["peak_hz"][s:e])),
        })

    groups = []
    for b in blasts:
        if groups and b["time_sec"] - groups[-1]["end_sec"] <= group_gap_sec:
            g = groups[-1]
            g["end_sec"] = b["end_sec"]
            g["blasts"] += 1
            g["_strengths"].append(b["strength"])
            g["_freqs"].append(b["freq_hz"])
        else:
            groups.append({"time_sec": b["time_sec"], "end_sec": b["end_sec"], "blasts": 1,
                           "_strengths": [b["strength"]], "_freqs": [b["freq_hz"]]})
    result = []
    for g in groups:
        st = max(g["_strengths"]) * (1.0 + 0.15 * (g["blasts"] - 1))
        result.append({
            "time_sec": round(g["time_sec"], 2),
            "end_sec": round(g["end_sec"], 2),
            "duration_sec": round(g["end_sec"] - g["time_sec"], 2),
            "strength": round(float(min(1.0, st)), 3),
            "freq_hz": round(float(np.median(g["_freqs"])), 0),
            "blasts": g["blasts"],
        })
    return result


def detect_whistles(samples, sr=AUDIO_SR, **kwargs):
    """Sípszó felismerés egy (mono, float vagy int16) hangmintatömbön."""
    x = np.asarray(samples)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    feat = audio_frame_features(x, sr=sr)
    return detect_whistles_from_features(feat, **kwargs)


def extract_audio_features(video_path, ffmpeg_bin=None, sr=AUDIO_SR, chunk_sec=30.0):
    """
    A hangsáv kinyerése ffmpeg-gel (mono, 16 kHz, s16le) csövön át, darabolva –
    egy kétórás felvétel sem kerül egyszerre a memóriába.
    Visszatér: (jellemzők dict, hiba szöveg vagy None).
    """
    ffmpeg_bin = ffmpeg_bin or FFMPEG_BIN
    cmd = [ffmpeg_bin, "-v", "error", "-nostdin", "-i", video_path, "-vn", "-sn",
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    n = int(round(0.05 * sr))
    hop = int(round(0.025 * sr))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, OSError) as ex:
        return None, f"ffmpeg nem indítható ({ffmpeg_bin}): {ex}"

    # stderr ürítése külön szálon, hogy ne akadjon el a cső
    err_buf = []
    t_err = threading.Thread(target=lambda: err_buf.append(proc.stderr.read()), daemon=True)
    t_err.start()

    parts = []
    buf = np.zeros(0, dtype=np.float32)
    frame_idx = 0
    chunk_bytes = int(chunk_sec * sr) * 2
    while True:
        raw = proc.stdout.read(chunk_bytes)
        if not raw:
            break
        if len(raw) % 2:
            raw = raw[:-1]
        buf = np.concatenate([buf, np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0])
        if buf.size < n:
            continue
        nf = (buf.size - n) // hop + 1
        feat = audio_frame_features(buf[:(nf - 1) * hop + n], sr=sr, start_index=frame_idx)
        parts.append(feat)
        frame_idx += nf
        buf = buf[nf * hop:]
    proc.wait()
    t_err.join(timeout=2)
    proc.stdout.close()
    proc.stderr.close()
    if not parts:
        msg = (err_buf[0].decode("utf-8", "ignore").strip() if err_buf and err_buf[0] else "")
        return None, "Nincs hangsáv vagy nem olvasható" + (f": {msg[:200]}" if msg else "")
    feat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return feat, None


# ============================================================
# b) Kép – fű arány, mozgás, pásztázás (+ opcionális létszám)
# ============================================================

class VisualSampler:
    """
    Egymást követő mintaképkockákból (pl. 1 mp-enként) számol:
      grass    – a képen a fű aránya (0..1)
      activity – mozgás a füvön: a kamera eltolását fáziskorrelációval kompenzálva
                 a megváltozott füves pixelek aránya (0..1)
      pan      – a kamera eltolása a képszélesség arányában
    """

    def __init__(self, grass_lower=DEFAULT_GRASS_LOWER, grass_upper=DEFAULT_GRASS_UPPER,
                 work_width=160, diff_threshold=22):
        self.grass_lower = np.array(grass_lower, dtype=np.uint8)
        self.grass_upper = np.array(grass_upper, dtype=np.uint8)
        self.work_width = work_width
        self.diff_threshold = diff_threshold
        self.prev = None
        self.window = None

    def update(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if w != self.work_width:
            nh = max(8, int(round(h * self.work_width / float(w))))
            frame_bgr = cv2.resize(frame_bgr, (self.work_width, nh), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask, _ = build_grass_mask(hsv, self.grass_lower, self.grass_upper, scale=1.0)
        if mask is None:
            mask = np.full(hsv.shape[:2], 255, dtype=np.uint8)
        grass = cv2.countNonZero(mask) / float(mask.size)

        gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        gray_f = np.float32(gray)
        activity, pan = 0.0, 0.0
        if self.prev is not None and self.prev.shape == gray_f.shape:
            if self.window is None or self.window.shape != gray_f.shape:
                self.window = cv2.createHanningWindow((gray_f.shape[1], gray_f.shape[0]), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(self.prev, gray_f, self.window)
            if response < 0.03:
                dx, dy = 0.0, 0.0
            pan = math.hypot(dx, dy) / float(gray_f.shape[1])
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            warped = cv2.warpAffine(self.prev, M, (gray_f.shape[1], gray_f.shape[0]))
            valid = cv2.warpAffine(np.ones_like(self.prev), M, (gray_f.shape[1], gray_f.shape[0])) > 0.99
            area = valid & (mask > 0)
            n_area = int(np.count_nonzero(area))
            if n_area > 0.01 * area.size:
                diff = np.abs(gray_f - warped)
                activity = float(np.count_nonzero((diff > self.diff_threshold) & area)) / n_area
        self.prev = gray_f
        return float(grass), float(activity), float(pan)


def _video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    dur = (n / fps) if fps > 0 and n > 0 else 0.0
    return {"fps": fps if fps > 0 else 25.0, "width": w, "height": h, "duration": dur}


def _iter_frames_ffmpeg(video_path, sample_fps, out_w, out_h, ffmpeg_bin):
    """Mintavételezett, kicsinyített képkockák ffmpeg csövön (gyors: csak dekódol, nem kódol)."""
    cmd = [ffmpeg_bin, "-v", "error", "-nostdin", "-i", video_path, "-an", "-sn",
           "-vf", f"fps={sample_fps},scale={out_w}:{out_h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = out_w * out_h * 3
    try:
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _iter_frames_opencv(video_path, sample_fps, out_w, out_h):
    """Tartalék út ffmpeg nélkül: grab() minden képkockára, retrieve() csak a mintáknál."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = fps / float(sample_fps)
    next_take = 0.0
    idx = 0
    try:
        while cap.grab():
            if idx >= next_take:
                ok, frame = cap.retrieve()
                if ok:
                    yield cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                next_take += step
            idx += 1
    finally:
        cap.release()


def sample_video_signals(video_path, sample_fps=1.0, person_counter=None, person_every_sec=5.0,
                         grass_lower=DEFAULT_GRASS_LOWER, grass_upper=DEFAULT_GRASS_UPPER,
                         ffmpeg_bin=None, progress_cb=None, meta=None):
    """
    A képi jelek mintavételezése az egész videón.
    Visszatér: dict – times, grass, activity, pan, persons (vagy None), source.
    """
    ffmpeg_bin = ffmpeg_bin or FFMPEG_BIN
    meta = meta or _video_meta(video_path) or {"width": 0, "height": 0, "duration": 0.0, "fps": 25.0}
    # létszámszámláláshoz nagyobb kép kell (a YOLO-nak 320 px kevés)
    out_w = 960 if person_counter else 320
    if meta["width"] > 0 and meta["height"] > 0:
        out_w = min(out_w, meta["width"] - meta["width"] % 2)
        out_h = max(2, int(round(meta["height"] * out_w / float(meta["width"]) / 2.0)) * 2)
    else:
        out_h = out_w * 9 // 16
    expected_n = int(meta["duration"] * sample_fps) if meta["duration"] else 0

    sampler = VisualSampler(grass_lower, grass_upper)
    grass, activity, pan, persons = [], [], [], []
    person_step = max(1, int(round(person_every_sec * sample_fps)))
    last_count = 0

    def _consume(frames):
        nonlocal last_count
        count = 0
        for frame in frames:
            g, a, p = sampler.update(frame)
            grass.append(g)
            activity.append(a)
            pan.append(p)
            if person_counter is not None:
                if count % person_step == 0:
                    try:
                        last_count = int(person_counter(frame))
                    except Exception as ex:  # a horog hibája ne állítsa le a felismerést
                        print(f"Létszámszámláló hiba: {ex}")
                persons.append(last_count)
            count += 1
            if progress_cb and expected_n and count % 60 == 0:
                progress_cb("kep", min(99.0, 100.0 * count / expected_n))
        return count

    source = "ffmpeg"
    try:
        n = _consume(_iter_frames_ffmpeg(video_path, sample_fps, out_w, out_h, ffmpeg_bin))
    except (FileNotFoundError, OSError):
        n = 0
    if n == 0:
        source = "opencv"
        sampler = VisualSampler(grass_lower, grass_upper)
        _consume(_iter_frames_opencv(video_path, sample_fps, out_w, out_h))

    times = np.arange(len(grass), dtype=np.float64) / sample_fps
    return {
        "times": times,
        "grass": np.array(grass, dtype=np.float64),
        "activity": np.array(activity, dtype=np.float64),
        "pan": np.array(pan, dtype=np.float64),
        "persons": np.array(persons, dtype=np.float64) if person_counter is not None else None,
        "source": source,
    }


# ============================================================
# Szegmentálás
# ============================================================

@dataclass
class SegmenterParams:
    """A szegmentálás paraméterei (másodpercben, ha nincs más jelölve)."""
    expected_half_min: float = 45.0
    min_half_min: Optional[float] = None      # alap: a várt hossz 44%-a (45 -> kb. 20 perc)
    max_half_min: Optional[float] = None      # alap: a várt hossz 125%-a (45 -> kb. 56 perc)
    min_break_sec: float = 180.0              # a két félidő közötti szünet minimuma
    snap_radius_sec: float = 90.0             # ennyin belüli sípszóhoz igazítunk
    smooth_sec: float = 20.0                  # a játék-pontszám simítása
    merge_gap_sec: float = 60.0               # ennél rövidebb megszakítás (sérülés, csere) nem szünet
    min_interval_sec: float = 60.0            # ennél rövidebb „játék” szakasz zaj
    stoppage_allow_frac: float = 0.15         # ráadás / pontatlanság tűrése a várt hossz körül
    min_snap_score: float = 0.12

    def half_bounds_sec(self):
        lo = self.min_half_min if self.min_half_min is not None else self.expected_half_min * 0.44
        hi = self.max_half_min if self.max_half_min is not None else self.expected_half_min * 1.25
        return lo * 60.0, hi * 60.0


def compute_play_score(grass, activity, pan, persons=None):
    """
    „Játék folyik” pontszám (kb. 0..1) mintánként. A jeleket a videó saját eloszlásához
    normáljuk (felvételenként más a fény, a pálya, a kamera), mert a meccs a felvétel
    nagyobb részét teszi ki.
    """
    g = np.asarray(grass, dtype=np.float64)
    a = np.asarray(activity, dtype=np.float64)
    p = np.asarray(pan, dtype=np.float64)
    if g.size == 0:
        return g

    def _norm(x, lo_val=None):
        hi = float(np.percentile(x, 85))
        lo = float(np.percentile(x, 5)) if lo_val is None else lo_val
        if hi - lo < 1e-6:
            return np.zeros_like(x) if hi < 1e-6 else np.ones_like(x) * 0.5
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    g_n = _norm(g, lo_val=min(0.15, float(np.percentile(g, 5))))
    a_n = _norm(a, lo_val=0.0)
    p_n = _norm(p, lo_val=0.0)
    # fű nélkül nincs játék (lelátó, parkoló, öltöző felé fordult kamera)
    score = (0.30 * g_n + 0.45 * a_n + 0.25 * p_n) * (0.4 + 0.6 * g_n)
    if persons is not None and len(persons) == g.size:
        n_n = _norm(np.asarray(persons, dtype=np.float64), lo_val=0.0)
        score = 0.7 * score + 0.3 * n_n
    return score


def _intervals_from_score(smooth, thr, band, merge_gap, min_len):
    """Hiszterézises küszöbölés + rövid rések összevonása + rövid szakaszok eldobása."""
    hi_t, lo_t = thr + band, thr - band
    state = False
    on = np.zeros(smooth.size, dtype=bool)
    for i, v in enumerate(smooth):
        if state and v < lo_t:
            state = False
        elif not state and v > hi_t:
            state = True
        on[i] = state
    runs = _runs(on)
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]


def _dedupe(values, tol):
    out = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _whistle_near(whistles, t, radius, kind, centered, fs):
    """A boundary-hez (kind: 'start' / 'end') legjobban illő sípszó és pontszáma."""
    best, best_score = None, 0.0
    ctx_win = int(round(30 * fs))
    n = centered.size
    for w in whistles:
        wt = w["time_sec"] if kind == "start" else w["end_sec"]
        d = abs(wt - t)
        if d > radius:
            continue
        i = int(round(wt * fs))
        before = centered[max(0, i - ctx_win):max(0, min(n, i))]
        after = centered[max(0, min(n, i)):min(n, i + ctx_win)]
        b = float(np.mean(before)) if before.size else 0.0
        a = float(np.mean(after)) if after.size else 0.0
        # valódi határ: előtte szünet, utána játék (kezdés) – vagy fordítva (lefújás);
        # a játék közbeni szabálytalansági sípszónak mindkét oldalán játék van
        if kind == "start":
            ctx = 1.0 if (b < 0 < a) else (0.5 if a > b else 0.2)
        else:
            ctx = 1.0 if (a < 0 < b) else (0.5 if b > a else 0.2)
        score = w["strength"] * (1.0 - 0.7 * d / radius) * ctx
        if score > best_score:
            best, best_score = w, score
    return best, best_score


def segment_halves(times, score, whistles, duration_sec, params=None):
    """
    A félidők kiválasztása a játék-pontszám idősorából és a sípszavakból.
    Két, hihető hosszúságú, legalább min_break_sec szünettel elválasztott szakaszt keres,
    amely a lehető legtöbb „játék” és legkevesebb „szünet” mintát fedi le, közel a várt
    hosszhoz; a határokat a közeli erős sípszavakhoz igazítja.
    Soha nem dob kivételt: bizonytalan esetben is ad legjobb tippet alacsony bizalommal.
    """
    params = params or SegmenterParams()
    whistles = whistles or []
    notes = []
    times = np.asarray(times, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    duration_sec = float(max(duration_sec or 0.0, times[-1] if times.size else 0.0))
    min_half, max_half = params.half_bounds_sec()
    exp_len = params.expected_half_min * 60.0

    def _whole(note, conf=0.1):
        notes.append(note)
        return {
            "first_half": _period(0.0, duration_sec),
            "second_half": None,
            "confidence": conf,
            "whistles": whistles,
            "notes": notes,
            "duration_sec": round(duration_sec, 2),
        }

    if duration_sec < 2 * min_half + params.min_break_sec or times.size < 10:
        return _whole("A videó túl rövid két félidőhöz – a teljes videót egy szakaszként adom vissza.")

    fs = 1.0 / float(np.median(np.diff(times))) if times.size > 1 else 1.0
    smooth = _moving_average(score, params.smooth_sec * fs)
    thr = _otsu_threshold(smooth)
    spread = float(np.percentile(smooth, 90) - np.percentile(smooth, 10))
    if spread < 0.05:
        notes.append("A képi jelek alig változnak – a felismerés főleg a sípszavakra és a várt hosszra épül.")
    centered = smooth - thr
    unit = float(np.mean(np.abs(centered))) + 1e-6
    csum = np.concatenate([[0.0], np.cumsum(centered)])

    def idx(t):
        return int(np.clip(round(t * fs), 0, smooth.size))

    # --- jelölt határok: játékszakaszok szélei több küszöbön + sípszavak ---
    starts, ends = [], []
    for k in (-0.25, 0.0, 0.25):
        ivs = _intervals_from_score(smooth, thr + k * spread, 0.1 * spread,
                                    params.merge_gap_sec * fs, params.min_interval_sec * fs)
        for s, e in ivs:
            starts.append(s / fs)
            ends.append(e / fs)
    # egy meccsen sok sípszó van (szabálytalanság, les) – csak a legerősebbek lehetnek jelöltek
    strong = sorted((w for w in whistles if w["strength"] >= 0.1), key=lambda w: -w["strength"])[:40]
    starts += [w["time_sec"] for w in strong]
    ends += [w["end_sec"] for w in strong]
    starts = _dedupe([max(0.0, s) for s in starts + [0.0]], 5.0)
    ends = _dedupe([min(duration_sec, e) for e in ends + [duration_sec]], 5.0)

    allow = params.stoppage_allow_frac * exp_len
    bonus_radius = min(params.snap_radius_sec, 20.0)
    bonus = {}
    for kind, cands in (("start", starts), ("end", ends)):
        for t_c in cands:
            w, ws = _whistle_near(strong, t_c, bonus_radius, kind, centered, fs)
            bonus[(kind, t_c)] = 30.0 * unit * fs * ws if w is not None else 0.0

    def half_value(s, e):
        L = e - s
        v = csum[idx(e)] - csum[idx(s)]
        v -= 0.5 * unit * fs * max(0.0, abs(L - exp_len) - allow)
        return v + bonus[("start", s)] + bonus[("end", e)]

    def best_pair(lo, hi, min_break):
        halves = []
        for s in starts:
            for e in ends:
                if lo <= e - s <= hi:
                    halves.append((s, e, half_value(s, e)))
        if len(halves) < 2:
            return None
        # minden 2. félidő jelöltre a legjobb, elég korán véget érő 1. félidő (prefix maximum)
        by_end = sorted(halves, key=lambda h: h[1])
        end_times = np.array([h[1] for h in by_end])
        pref_best, cur = [], None
        for h in by_end:
            if cur is None or h[2] > cur[2]:
                cur = h
            pref_best.append(cur)
        best = None
        for h2 in halves:
            j = int(np.searchsorted(end_times, h2[0] - min_break, side="right")) - 1
            if j < 0:
                continue
            h1 = pref_best[j]
            total = h1[2] + h2[2]
            if best is None or total > best[0]:
                best = (total, h1, h2)
        return best

    best = best_pair(min_half, max_half, params.min_break_sec)
    relaxed = False
    if best is None:
        relaxed = True
        best = best_pair(min_half * 0.6, max_half * 1.4, params.min_break_sec * 0.5)
    if best is None:
        return _whole("Nem találtam két hihető félidőt – a teljes videót egy szakaszként adom vissza, "
                      "add meg kézzel a határokat.")
    if relaxed:
        notes.append("Csak lazított feltételekkel találtam két félidőt (szokatlan hossz vagy rövid szünet).")

    _, h1, h2 = best
    bounds = [h1[0], h1[1], h2[0], h2[1]]
    snapped = [False] * 4
    labels = ["Az 1. félidő kezdete", "Az 1. félidő vége", "A 2. félidő kezdete", "A 2. félidő vége"]
    for i, kind in enumerate(("start", "end", "start", "end")):
        w, ws = _whistle_near(whistles, bounds[i], params.snap_radius_sec, kind, centered, fs)
        if w is None or ws < params.min_snap_score:
            continue
        new_t = w["time_sec"] if kind == "start" else w["end_sec"]
        trial = bounds.copy()
        trial[i] = new_t
        ok = (trial[0] < trial[1] < trial[2] < trial[3]
              and min_half * 0.9 <= trial[1] - trial[0] and min_half * 0.9 <= trial[3] - trial[2]
              and trial[2] - trial[1] >= params.min_break_sec * 0.5)
        if ok:
            bounds = trial
            snapped[i] = True
            notes.append(f"{labels[i]} sípszóhoz igazítva ({format_hms(new_t)}).")

    # --- bizalom ---
    inside = np.zeros(smooth.size, dtype=bool)
    inside[idx(bounds[0]):idx(bounds[1])] = True
    inside[idx(bounds[2]):idx(bounds[3])] = True
    c_in = float(np.mean(smooth[inside])) if inside.any() else 0.0
    c_out = float(np.mean(smooth[~inside])) if (~inside).any() else c_in
    contrast = float(np.clip((c_in - c_out) / max(spread, 0.05), 0.0, 1.0))
    len_scores = []
    for L in (bounds[1] - bounds[0], bounds[3] - bounds[2]):
        dev = max(0.0, abs(L - exp_len) - allow)
        len_scores.append(math.exp(-dev / (0.2 * exp_len)))
    conf = 0.5 * contrast + 0.25 * (sum(snapped) / 4.0) + 0.25 * float(np.mean(len_scores))
    if relaxed:
        conf *= 0.6
    if spread < 0.05:
        conf *= 0.5
    conf = round(float(np.clip(conf, 0.0, 1.0)), 2)
    if not whistles:
        notes.append("Nem találtam sípszót – a határok csak a képi jeleken alapulnak.")
    if conf < 0.5:
        notes.append("Bizonytalan felismerés – érdemes kézzel ellenőrizni a határokat.")

    return {
        "first_half": _period(bounds[0], bounds[1], snapped[0], snapped[1]),
        "second_half": _period(bounds[2], bounds[3], snapped[2], snapped[3]),
        "confidence": conf,
        "whistles": whistles,
        "notes": notes,
        "duration_sec": round(duration_sec, 2),
    }


def _period(start, end, start_snapped=False, end_snapped=False):
    return {
        "start_sec": round(float(start), 2),
        "end_sec": round(float(end), 2),
        "duration_sec": round(float(end - start), 2),
        "start_hms": format_hms(start),
        "end_hms": format_hms(end),
        "start_snapped_to_whistle": bool(start_snapped),
        "end_snapped_to_whistle": bool(end_snapped),
    }


# ============================================================
# Fő belépési pont
# ============================================================

def detect_match_halves(video_path, expected_half_min=45.0, person_counter: Optional[Callable] = None,
                        params: Optional[SegmenterParams] = None, ffmpeg_bin=None,
                        grass_lower=DEFAULT_GRASS_LOWER, grass_upper=DEFAULT_GRASS_UPPER,
                        sample_fps=1.0, progress_cb=None):
    """
    A félidők felismerése egy teljes felvételen.
    Visszatér: {"first_half": {...}, "second_half": {...} vagy None, "confidence": 0..1,
                "whistles": [...], "notes": [...], "duration_sec", "play_timeline": [[t, pont], ...]}
    """
    params = params or SegmenterParams(expected_half_min=expected_half_min)
    notes = []
    meta = _video_meta(video_path)
    if meta is None:
        notes.append("Az OpenCV nem tudta megnyitni a videót – csak ffmpeg-gel próbálkozom.")

    print(f"⏱️ Félidő-felismerés: hangsáv elemzése ({video_path})")
    feat, err = extract_audio_features(video_path, ffmpeg_bin=ffmpeg_bin)
    if feat is None:
        whistles = []
        notes.append(f"Hangsáv nem elemezhető – csak képi jelek alapján. ({err})")
    else:
        whistles = detect_whistles_from_features(feat)
        print(f"   Sípszó jelöltek: {len(whistles)}")
    if progress_cb:
        progress_cb("hang", 100.0)

    print("⏱️ Félidő-felismerés: képi jelek mintavételezése")
    sig = sample_video_signals(video_path, sample_fps=sample_fps, person_counter=person_counter,
                               grass_lower=grass_lower, grass_upper=grass_upper,
                               ffmpeg_bin=ffmpeg_bin, progress_cb=progress_cb, meta=meta)
    if sig["source"] == "opencv":
        notes.append("Az ffmpeg képdekódolás nem sikerült, OpenCV tartalék úton mintavételeztem (lassabb).")

    duration = max(meta["duration"] if meta else 0.0,
                   float(sig["times"][-1]) + 1.0 / sample_fps if sig["times"].size else 0.0,
                   float(feat["time"][-1]) if feat is not None and feat["time"].size else 0.0)

    if sig["times"].size == 0:
        result = segment_halves(np.zeros(0), np.zeros(0), whistles, duration, params)
        result["notes"] = notes + ["Nem sikerült képkockát olvasni a videóból."] + result["notes"]
        result["confidence"] = 0.0
        result["play_timeline"] = []
        return result

    score = compute_play_score(sig["grass"], sig["activity"], sig["pan"], sig["persons"])
    result = segment_halves(sig["times"], score, whistles, duration, params)
    result["notes"] = notes + result["notes"]

    # ritkított idősor a kliens grafikonjához (5 mp-enként)
    step = max(1, int(round(5 * sample_fps)))
    smooth = _moving_average(score, params.smooth_sec * sample_fps)
    result["play_timeline"] = [[round(float(sig["times"][i]), 1), round(float(smooth[i]), 3)]
                               for i in range(0, smooth.size, step)]
    return result


# ============================================================
# Vágás ffmpeg-gel (stream copy) + összefűzés szünet nélkül
# ============================================================

def cut_segment(video_path, start_sec, end_sec, out_path, ffmpeg_bin=None):
    """Szakasz kivágása újrakódolás nélkül (-ss az -i előtt: gyors, kulcskockához igazodik)."""
    ffmpeg_bin = ffmpeg_bin or FFMPEG_BIN
    cmd = [ffmpeg_bin, "-y", "-v", "error", "-nostdin", "-ss", f"{start_sec:.3f}", "-i", video_path,
           "-t", f"{max(0.1, end_sec - start_sec):.3f}", "-map", "0:v:0?", "-map", "0:a:0?",
           "-c", "copy", "-avoid_negative_ts", "make_zero", out_path]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg vágási hiba: {res.stderr.decode('utf-8', 'ignore')[:300]}")
    return out_path


def concat_segments(paths, out_path, work_dir, ffmpeg_bin=None):
    """Több szakasz összefűzése (concat demuxer, ideiglenes listafájllal, újrakódolás nélkül)."""
    ffmpeg_bin = ffmpeg_bin or FFMPEG_BIN
    list_path = os.path.join(work_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            safe = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    try:
        cmd = [ffmpeg_bin, "-y", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0",
               "-i", list_path, "-c", "copy", out_path]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"ffmpeg összefűzési hiba: {res.stderr.decode('utf-8', 'ignore')[:300]}")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
    return out_path


def cut_match_halves(video_path, halves, out_dir, padding_sec=5.0, duration_sec=None, ffmpeg_bin=None):
    """
    A félidők kivágása és a szünet nélküli teljes meccs elkészítése.
    halves: {"first_half": {"start_sec","end_sec"}, "second_half": {...} vagy None}
    Visszatér: {"first_half": útvonal, "second_half": útvonal vagy None, "full_match": útvonal}
    """
    clip_id = uuid.uuid4().hex[:8]
    out = {"first_half": None, "second_half": None, "full_match": None}
    parts = []
    for key, name in (("first_half", "half1"), ("second_half", "half2")):
        h = halves.get(key)
        if not h:
            continue
        s = max(0.0, float(h["start_sec"]) - padding_sec)
        e = float(h["end_sec"]) + padding_sec
        if duration_sec:
            e = min(e, float(duration_sec))
        if e <= s:
            continue
        path = os.path.abspath(os.path.join(out_dir, f"{name}_{clip_id}.mp4"))
        cut_segment(video_path, s, e, path, ffmpeg_bin)
        out[key] = path
        parts.append(path)
    if not parts:
        raise RuntimeError("Nincs kivágható szakasz.")
    full = os.path.abspath(os.path.join(out_dir, f"match_{clip_id}.mp4"))
    concat_segments(parts, full, out_dir, ffmpeg_bin)
    out["full_match"] = full
    return out

"""
A félidő-felismerő (match_segmenter) tesztjei.
Futtatás: python3 tests/test_match_segmenter.py
A videós (végponttól végpontig) teszt ffmpeg nélkül kimarad.
"""
import os
import sys
import shutil
import tempfile
import unittest
import wave
import subprocess

import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import match_segmenter as ms  # noqa: E402

SR = ms.AUDIO_SR
FFMPEG = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

# Tömörített idővonal (mp): bemelegítés, 1. félidő, szünet, 2. félidő, meccs utáni rész
T_H1 = (60.0, 300.0)
T_H2 = (390.0, 630.0)
T_TOTAL = 680.0
# sípszavak: (kezdet, hossz)
WHISTLES = [(60.0, 0.6),                                   # kezdőrúgás
            (150.0, 0.4),                                  # szabálytalanság
            (300.0, 0.6), (300.9, 0.6), (301.8, 1.0),      # félidő vége (3 sípszó)
            (390.0, 0.6),                                  # 2. félidő kezdete
            (480.0, 0.4),                                  # szabálytalanság
            (630.0, 0.6), (630.9, 0.6), (631.8, 1.0)]      # lefújás
TEST_PARAMS = dict(expected_half_min=4.0, min_break_sec=60.0, snap_radius_sec=30.0,
                   smooth_sec=6.0, merge_gap_sec=10.0, min_interval_sec=20.0)


def make_audio(total_sec, whistles, seed=1):
    """Zajos háttér (szél, tömeg) + kb. 3,2 kHz-es, enyhén trillázó sípszavak."""
    rng = np.random.default_rng(seed)
    n = int(total_sec * SR)
    noise = rng.normal(0, 0.05, n).astype(np.float32)
    # mélyebb „tömeg” zaj: aluláteresztett fehér zaj
    noise = np.convolve(noise, np.ones(8) / 8, mode="same").astype(np.float32) * 2.0
    t_all = np.arange(n) / SR
    noise += 0.03 * np.sin(2 * np.pi * 400 * t_all).astype(np.float32)   # zavaró hang
    for start, dur in whistles:
        i0, i1 = int(start * SR), int((start + dur) * SR)
        t = np.arange(i1 - i0) / SR
        freq = 3200 + 120 * np.sin(2 * np.pi * 35 * t)                     # a síp golyójának trillája
        phase = 2 * np.pi * np.cumsum(freq) / SR
        env = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
        noise[i0:i1] += (0.3 * env * np.sin(phase)).astype(np.float32)
    return np.clip(noise, -1, 1)


def phase_at(t):
    if T_H1[0] <= t < T_H1[1] or T_H2[0] <= t < T_H2[1]:
        return "play"
    if t < T_H1[0]:
        return "warmup"
    return "break"


def make_video(path, ffmpeg, fps=10, w=320, h=180):
    rng = np.random.default_rng(7)
    wav_path = path + ".wav"
    audio = make_audio(T_TOTAL, WHISTLES)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    # füves háttér csíkokkal (hogy a kamera pásztázása mérhető legyen)
    big_w = w * 3
    xs = np.arange(big_w)
    stripes = ((xs // 40) % 2).astype(np.float32)
    field_bgr = np.zeros((h, big_w, 3), np.uint8)
    field_bgr[:, :, 1] = (120 + 30 * stripes)[None, :].astype(np.uint8)
    field_bgr[:, :, 0] = 40
    field_bgr[:, :, 2] = 50
    field_bgr = cv2.add(field_bgr, rng.integers(0, 12, field_bgr.shape, dtype=np.uint8))
    grey = np.full((h, w, 3), 110, np.uint8)
    grey[: h // 3] = 170                                              # „égbolt / épület”
    n_players = 20
    pos = rng.uniform([0, 30], [big_w, h - 10], size=(n_players, 2))
    warm_pos = rng.uniform([w, 60], [2 * w, h - 20], size=(4, 2))

    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-i", wav_path, "-c:v", "libx264", "-preset", "ultrafast",
           "-g", str(fps * 2), "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_frames = int(T_TOTAL * fps)
    for i in range(n_frames):
        t = i / fps
        ph = phase_at(t)
        if ph == "play":
            cam_x = int(w + w * 0.8 * np.sin(2 * np.pi * t / 20.0))          # labdakövető pásztázás
            pos += rng.normal(0, 4, pos.shape)
            pos[:, 1] = np.clip(pos[:, 1], 30, h - 10)
            pos[:, 0] = np.clip(pos[:, 0], 0, big_w - 1)
            frame = field_bgr.copy()
            for x, y in pos:
                cv2.rectangle(frame, (int(x) - 3, int(y) - 12), (int(x) + 3, int(y)), (30, 30, 200), -1)
            frame = frame[:, cam_x:cam_x + w]
        elif ph == "warmup":
            frame = field_bgr[:, w:2 * w].copy()                              # álló kamera, lassú mozgás
            warm_pos[:, 0] += 0.2
            for x, y in warm_pos:
                cv2.rectangle(frame, (int(x - w) - 3, int(y) - 12), (int(x - w) + 3, int(y)), (200, 30, 30), -1)
        else:
            frame = grey.copy()
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    os.remove(wav_path)
    return path


class PureFunctionTests(unittest.TestCase):
    def test_whistle_detection_on_numpy_audio(self):
        audio = make_audio(40.0, [(5.0, 0.6), (20.0, 0.5), (20.8, 0.5), (21.6, 1.0), (30.0, 0.15)])
        found = ms.detect_whistles(audio, SR)
        times = [w["time_sec"] for w in found]
        self.assertEqual(len(found), 2, found)                         # a 0,15 mp-es nem sípszó
        self.assertAlmostEqual(times[0], 5.0, delta=0.15)
        self.assertAlmostEqual(times[1], 20.0, delta=0.15)
        self.assertEqual(found[1]["blasts"], 3)
        self.assertAlmostEqual(found[1]["end_sec"], 22.6, delta=0.15)
        self.assertTrue(2900 <= found[0]["freq_hz"] <= 3500)
        self.assertGreater(found[1]["strength"], found[0]["strength"])

    def test_no_whistle_in_noise(self):
        self.assertEqual(ms.detect_whistles(make_audio(30.0, []), SR), [])

    def test_segmentation_on_synthetic_signals(self):
        # valós léptékű idővonal: 15 p bemelegítés, 2x45 p (+ráadás), 15 p szünet, 10 p utána
        rng = np.random.default_rng(3)
        h1, h2 = (900.0, 3660.0), (4560.0, 7380.0)
        total = 8000
        t = np.arange(total, dtype=np.float64)
        play = ((t >= h1[0]) & (t < h1[1])) | ((t >= h2[0]) & (t < h2[1]))
        warm = t < h1[0]
        grass = np.where(play | warm, 0.7, 0.3) + rng.normal(0, 0.05, total)
        activity = np.where(play, 0.12, np.where(warm, 0.05, 0.01)) + np.abs(rng.normal(0, 0.03, total))
        pan = np.where(play, 0.05, 0.01) + np.abs(rng.normal(0, 0.02, total))
        # rövid megszakítás (sérülés) a félidőben – nem lehet szünet
        injury = (t >= 2000) & (t < 2040)
        activity[injury] = 0.01
        whistles = [
            {"time_sec": 905.0, "end_sec": 905.6, "duration_sec": 0.6, "strength": 0.5, "freq_hz": 3200, "blasts": 1},
            {"time_sec": 2500.0, "end_sec": 2500.4, "duration_sec": 0.4, "strength": 0.3, "freq_hz": 3200, "blasts": 1},
            {"time_sec": 3655.0, "end_sec": 3658.0, "duration_sec": 3.0, "strength": 0.7, "freq_hz": 3200, "blasts": 3},
            {"time_sec": 4570.0, "end_sec": 4570.5, "duration_sec": 0.5, "strength": 0.5, "freq_hz": 3200, "blasts": 1},
        ]
        score = ms.compute_play_score(grass, activity, pan)
        res = ms.segment_halves(t, score, whistles, float(total), ms.SegmenterParams(expected_half_min=45))
        fh, sh = res["first_half"], res["second_half"]
        self.assertAlmostEqual(fh["start_sec"], 905.0, delta=1.0)       # sípszóhoz igazítva
        self.assertAlmostEqual(fh["end_sec"], 3658.0, delta=1.0)
        self.assertAlmostEqual(sh["start_sec"], 4570.0, delta=1.0)
        self.assertAlmostEqual(sh["end_sec"], h2[1], delta=30.0)        # sípszó nélkül: képi jelből
        self.assertGreater(res["confidence"], 0.5)

    def test_segmentation_without_whistles(self):
        t = np.arange(6000, dtype=np.float64)
        play = ((t >= 300) & (t < 2700)) | ((t >= 3300) & (t < 5700))
        score = np.where(play, 0.8, 0.1)
        res = ms.segment_halves(t, score, [], 6000.0, ms.SegmenterParams(expected_half_min=40))
        self.assertAlmostEqual(res["first_half"]["start_sec"], 300, delta=20)
        self.assertAlmostEqual(res["first_half"]["end_sec"], 2700, delta=20)
        self.assertAlmostEqual(res["second_half"]["start_sec"], 3300, delta=20)
        self.assertAlmostEqual(res["second_half"]["end_sec"], 5700, delta=20)
        self.assertTrue(any("sípszó" in n for n in res["notes"]))

    def test_short_video_returns_whole(self):
        t = np.arange(600, dtype=np.float64)
        res = ms.segment_halves(t, np.ones(600), [], 600.0, ms.SegmenterParams())
        self.assertIsNone(res["second_half"])
        self.assertEqual(res["first_half"]["start_sec"], 0.0)
        self.assertEqual(res["first_half"]["end_sec"], 600.0)
        self.assertTrue(res["notes"])
        # üres bemenet sem dobhat kivételt
        res = ms.segment_halves(np.zeros(0), np.zeros(0), [], 0.0)
        self.assertIsNone(res["second_half"])

    def test_flat_signal_does_not_crash(self):
        t = np.arange(7000, dtype=np.float64)
        res = ms.segment_halves(t, np.full(7000, 0.5), [], 7000.0)
        self.assertIn("first_half", res)
        self.assertLessEqual(res["confidence"], 0.5)

    def test_parse_time_value(self):
        self.assertIsNone(ms.parse_time_value(""))
        self.assertIsNone(ms.parse_time_value(None))
        self.assertIsNone(ms.parse_time_value("abc"))
        self.assertEqual(ms.parse_time_value("754,5"), 754.5)
        self.assertEqual(ms.parse_time_value("12:34"), 754.0)
        self.assertEqual(ms.parse_time_value("1:02:03"), 3723.0)
        self.assertEqual(ms.format_hms(3723), "1:02:03")


@unittest.skipIf(FFMPEG is None, "ffmpeg nem található – a videós teszt kimarad")
class EndToEndVideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="segtest_")
        cls.video = make_video(os.path.join(cls.tmp, "synthetic_match.mp4"), FFMPEG)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_detect_and_cut(self):
        params = ms.SegmenterParams(**TEST_PARAMS)
        res = ms.detect_match_halves(self.video, params=params, ffmpeg_bin=FFMPEG)
        print("\nFelismerés:", {k: res[k] for k in ("first_half", "second_half", "confidence")})
        print("Megjegyzések:", res["notes"])
        whistle_times = [w["time_sec"] for w in res["whistles"]]
        for t0 in (60.0, 150.0, 300.0, 390.0, 480.0, 630.0):
            self.assertTrue(any(abs(t0 - wt) < 0.3 for wt in whistle_times), (t0, whistle_times))
        fh, sh = res["first_half"], res["second_half"]
        self.assertIsNotNone(sh)
        self.assertAlmostEqual(fh["start_sec"], T_H1[0], delta=3.0)
        self.assertAlmostEqual(fh["end_sec"], T_H1[1], delta=3.0)
        self.assertAlmostEqual(sh["start_sec"], T_H2[0], delta=3.0)
        self.assertAlmostEqual(sh["end_sec"], T_H2[1], delta=3.0)
        self.assertGreaterEqual(res["confidence"], 0.6)
        self.assertTrue(res["play_timeline"])

        out = ms.cut_match_halves(self.video, res, self.tmp, padding_sec=5.0,
                                  duration_sec=res["duration_sec"], ffmpeg_bin=FFMPEG)

        def dur(p):
            cap = cv2.VideoCapture(p)
            d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1)
            cap.release()
            return d

        d1, d2, dfull = dur(out["first_half"]), dur(out["second_half"]), dur(out["full_match"])
        print(f"Vágott hosszak: 1. félidő {d1:.1f} mp, 2. félidő {d2:.1f} mp, összefűzött {dfull:.1f} mp")
        exp1 = fh["end_sec"] - fh["start_sec"] + 10
        exp2 = sh["end_sec"] - sh["start_sec"] + 10
        self.assertAlmostEqual(d1, exp1, delta=3.0)
        self.assertAlmostEqual(d2, exp2, delta=3.0)
        self.assertAlmostEqual(dfull, d1 + d2, delta=3.0)

    def test_visual_only(self):
        """Sípszavak nélkül, csak a képi jelekből is a félidők közelébe kell találni."""
        params = ms.SegmenterParams(**TEST_PARAMS)
        sig = ms.sample_video_signals(self.video, ffmpeg_bin=FFMPEG)
        self.assertEqual(sig["source"], "ffmpeg")
        score = ms.compute_play_score(sig["grass"], sig["activity"], sig["pan"])
        res = ms.segment_halves(sig["times"], score, [], T_TOTAL, params)
        fh, sh = res["first_half"], res["second_half"]
        print(f"\nCsak kép: 1. {fh['start_sec']}–{fh['end_sec']}, 2. {sh['start_sec']}–{sh['end_sec']}, "
              f"bizalom {res['confidence']}")
        self.assertAlmostEqual(fh["start_sec"], T_H1[0], delta=8.0)
        self.assertAlmostEqual(fh["end_sec"], T_H1[1], delta=8.0)
        self.assertAlmostEqual(sh["start_sec"], T_H2[0], delta=8.0)
        self.assertAlmostEqual(sh["end_sec"], T_H2[1], delta=8.0)

    def test_person_counter_hook_and_opencv_fallback(self):
        """A létszám-horog meghívódik; hibás ffmpeg esetén az OpenCV tartalék út működik."""
        calls = []

        def counter(frame):
            calls.append(frame.shape)
            return 22

        sig = ms.sample_video_signals(self.video, person_counter=counter, person_every_sec=30.0,
                                      ffmpeg_bin="nincs_ilyen_ffmpeg")
        self.assertEqual(sig["source"], "opencv")
        self.assertTrue(calls)
        self.assertEqual(len(sig["persons"]), len(sig["grass"]))
        self.assertAlmostEqual(len(sig["grass"]), T_TOTAL, delta=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

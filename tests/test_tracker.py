"""
A saját ByteTrack-szerű követő (detection/tracker.py) egységtesztjei.
Futtatás: python3 tests/test_tracker.py
"""
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from detection.tracker import (ByteTracker, linear_assignment, _hungarian_min, warp_state,  # noqa: E402
                               KalmanXYAH, xyxy_to_xyah, xyah_to_xyxy)
from detection.detector_base import box_iou  # noqa: E402


def box(cx, cy, w=20.0, h=50.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def run(tracker, frames, motions=None):
    """frames: [[(gt_id, xyxy, conf), ...], ...] -> {gt_id: [kiadott követés-azonosító vagy None, ...]}."""
    seq = {}
    for k, dets in enumerate(frames):
        xyxy = np.array([d[1] for d in dets], np.float32).reshape(-1, 4)
        conf = np.array([d[2] for d in dets], np.float32)
        out = tracker.update(xyxy, conf, np.zeros(len(dets), np.int64),
                             motion=None if motions is None else motions[k])
        got = {}
        for i, di in enumerate(out["det_index"]):
            if di >= 0:
                got[dets[di][0]] = int(out["id"][i])
        for d in dets:
            seq.setdefault(d[0], []).append(got.get(d[0]))
        for gid in seq:
            if gid not in [d[0] for d in dets]:
                seq[gid].append(None)
    return seq


def id_switches(seq):
    """MOT-szerű azonosító-váltás: egy valós objektum kiadott azonosítója megváltozik (a kihagyásokat átugorva)."""
    n = 0
    for ids in seq.values():
        last = None
        for i in ids:
            if i is None:
                continue
            if last is not None and i != last:
                n += 1
            last = i
    return n


class AssignmentTests(unittest.TestCase):
    def test_hungarian_matches_bruteforce(self):
        rng = np.random.default_rng(1)
        for n, m in [(1, 1), (2, 3), (3, 3), (4, 2), (5, 6), (6, 6)]:
            for _ in range(20):
                c = rng.random((n, m))
                if n <= m:
                    r, k = _hungarian_min(c)
                else:
                    k, r = _hungarian_min(c.T)
                got = c[r, k].sum()
                best = min(sum(c[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n)) \
                    if n <= m else min(sum(c[p[j], j] for j in range(m)) for p in itertools.permutations(range(n), m))
                self.assertAlmostEqual(got, best, places=9)
                self.assertEqual(len(r), min(n, m))

    def test_threshold_respected(self):
        c = np.array([[0.1, 0.9], [0.95, 0.99]])
        for use_scipy in (True, False):
            pairs, ur, uc = linear_assignment(c, 0.8, use_scipy=use_scipy)
            self.assertEqual(pairs, [(0, 0)])
            self.assertEqual(ur, [1])
            self.assertEqual(uc, [1])
        pairs, ur, uc = linear_assignment(np.zeros((0, 3)), 0.8)
        self.assertEqual((pairs, ur, uc), ([], [], [0, 1, 2]))


class KalmanTests(unittest.TestCase):
    def test_xyah_roundtrip(self):
        b = np.array([10.0, 20.0, 40.0, 100.0])
        np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(b)), b, atol=1e-9)

    def test_velocity_learned(self):
        kf = KalmanXYAH()
        mean, cov = kf.initiate(xyxy_to_xyah(np.array(box(100, 100))))
        for k in range(1, 30):
            mean, cov = kf.predict(mean, cov)
            mean, cov = kf.update(mean, cov, xyxy_to_xyah(np.array(box(100 + 5 * k, 100))))
        self.assertAlmostEqual(mean[4], 5.0, delta=0.3)

    def test_warp_state_translation_and_scale(self):
        kf = KalmanXYAH()
        mean, cov = kf.initiate(xyxy_to_xyah(np.array(box(100, 200, 20, 50))))
        mean[4] = 3.0
        H = np.array([[2.0, 0, 10], [0, 2.0, -5], [0, 0, 1]])
        m2, c2 = warp_state(mean, cov, H)
        self.assertAlmostEqual(m2[0], 210.0, places=6)
        self.assertAlmostEqual(m2[1], 395.0, places=6)
        self.assertAlmostEqual(m2[3], 100.0, places=6)
        self.assertAlmostEqual(m2[2], mean[2], places=6)
        self.assertAlmostEqual(m2[4], 6.0, places=6)
        self.assertTrue(np.all(np.linalg.eigvalsh(c2) > 0))


class TrackerTests(unittest.TestCase):
    def test_first_frame_and_confirmation(self):
        tr = ByteTracker()
        out = tr.update(np.array([box(100, 100)]), np.array([0.9]))
        self.assertEqual(out["id"].tolist(), [1])            # az első képkockán azonnal aktív
        out = tr.update(np.array([box(101, 100), box(400, 100)]), np.array([0.9, 0.9]))
        self.assertEqual(out["id"].tolist(), [1])            # az új (2.) még megerősítetlen
        out = tr.update(np.array([box(102, 100), box(401, 100)]), np.array([0.9, 0.9]))
        self.assertEqual(sorted(out["id"].tolist()), [1, 2])
        tr.reset()
        out = tr.update(np.array([box(100, 100)]), np.array([0.9]))
        self.assertEqual(out["id"].tolist(), [1])            # reset után újra 1-től

    def test_single_spurious_detection_never_output(self):
        tr = ByteTracker()
        tr.update(np.array([box(100, 100)]), np.array([0.9]))
        for k in range(10):
            dets = [box(100 + k, 100)]
            if k == 4:
                dets.append(box(600, 300))                     # egyszeri téves pozitív
            out = tr.update(np.array(dets), np.full(len(dets), 0.9))
            self.assertEqual(out["id"].tolist(), [1])

    def test_id_survives_occlusion_gap(self):
        frames = []
        for k in range(60):
            x = 100 + 4 * k
            dets = [] if 20 <= k < 32 else [("a", box(x, 200), 0.9)]   # 12 képkocka takarás
            dets.append(("b", box(700, 400), 0.9))
            frames.append(dets)
        seq = run(ByteTracker(track_buffer=30, frame_rate=25), frames)
        ids_a = [i for i in seq["a"] if i is not None]
        self.assertEqual(len(set(ids_a)), 1, seq["a"])
        self.assertEqual(seq["a"][33], ids_a[0])             # a takarás után azonnal visszatér
        self.assertEqual(id_switches(seq), 0)

    def test_lost_track_removed_after_buffer(self):
        tr = ByteTracker(track_buffer=5, frame_rate=30)
        tr.update(np.array([box(100, 100)]), np.array([0.9]))
        for _ in range(7):
            tr.update(np.zeros((0, 4)), np.zeros(0))
        self.assertEqual(tr.lost, [])
        tr.update(np.array([box(100, 100)]), np.array([0.9]))
        out = tr.update(np.array([box(100, 100)]), np.array([0.9]))
        self.assertEqual(out["id"].tolist(), [2])            # új azonosító

    def test_crossing_paths_keep_ids(self):
        frames = []
        for k in range(50):
            frames.append([("l", box(100 + 8 * k, 300), 0.85), ("r", box(500 - 8 * k, 312), 0.85)])
        seq = run(ByteTracker(), frames)
        self.assertEqual(id_switches(seq), 0, seq)
        self.assertNotEqual(seq["l"][-1], seq["r"][-1])

    def test_crossing_with_occlusion_low_scores(self):
        # keresztezéskor a takart alak pontszáma lecsökken (0.15), egy képkockán el is tűnik
        frames = []
        for k in range(50):
            xl, xr = 100 + 8 * k, 500 - 8 * k
            close = abs(xl - xr) < 30
            dets = [("l", box(xl, 300), 0.85)]
            if not (close and k % 3 == 0):
                dets.append(("r", box(xr, 306), 0.15 if close else 0.85))
            frames.append(dets)
        seq = run(ByteTracker(), frames)
        self.assertEqual(id_switches(seq), 0, seq)

    def test_low_score_recovery(self):
        tr = ByteTracker()
        frames = []
        for k in range(40):
            conf = 0.15 if 10 <= k < 25 else 0.9            # a gyenge lépcső tartja a követést
            dets = [("a", box(200 + 3 * k, 200), conf)]
            if 12 <= k < 20:
                dets.append(("noise", box(900, 500), 0.15))    # gyenge, magányos doboz: nem indít követést
            frames.append(dets)
        seq = run(tr, frames)
        self.assertTrue(all(i == 1 for i in seq["a"]), seq["a"])   # végig kiadva, ugyanazzal az azonosítóval
        self.assertTrue(all(i is None for i in seq["noise"]))
        self.assertGreater(tr.stats["gyenge párosítás"], 10)

    def test_class_change_does_not_break_id(self):
        tr = ByteTracker()
        ids = []
        for k in range(20):
            out = tr.update(np.array([box(100 + 2 * k, 100)]), np.array([0.9]), np.array([k % 2]))
            ids.append(int(out["id"][0]))
            self.assertEqual(int(out["cls"][0]), k % 2)        # a kiadott osztály a friss detektálásé
        self.assertEqual(set(ids), {1})

    def test_camera_motion_compensation_reduces_id_switches(self):
        """
        Szintetikus kamerapásztázás: álló emberek sűrű sora (32 px-enként, 20 px széles dobozok), a kamera
        szinuszosan ide-oda pásztáz (csúcs 10 px / képkocka, 40 képkockás periódus), 8% véletlen kihagyás.
        A ByteTrack Kálmán-sebessége lassan követi a gyorsuló mozgást, így kompenzáció nélkül a
        fordulóknál a szomszéd dobozára ugrik; a H-val kompenzált követő nem vált azonosítót.
        Második eset: csak minden 3. képkockán van detektálás (gyors mód) – a mozgás összeszorozva.
        """
        import math
        results = {}
        for step in (1, 3):
            rng = np.random.default_rng(3)
            people = [(60 + 32 * i, 300 + (i % 2) * 6) for i in range(30)]
            frames, motions = [], []
            prev = 0.0
            for k in range(240):
                offset = 10.0 * 40 / (2 * math.pi) * math.sin(2 * math.pi * k / 40)
                if k % step:
                    continue
                dets = []
                for pid, (x, y) in enumerate(people):
                    if rng.random() < 0.08:
                        continue                                  # véletlen kihagyás
                    dets.append((pid, box(x - offset + rng.normal(0, 1.0), y + rng.normal(0, 1.0)), 0.85))
                frames.append(dets)
                d = -(offset - prev)                              # előző kép -> aktuális kép eltolás
                motions.append(np.array([[1, 0, d], [0, 1, 0], [0, 0, 1]], np.float64))
                prev = offset
            motions[0] = None
            sw_plain = id_switches(run(ByteTracker(frame_rate=25), frames))
            sw_cmc = id_switches(run(ByteTracker(frame_rate=25), frames, motions))
            results[step] = (sw_plain, sw_cmc)
            self.assertGreater(sw_plain, 20)
            self.assertLessEqual(sw_cmc, 1)
        print("\n  kamerapásztázás – azonosító-váltás (kompenzáció nélkül, kompenzációval): "
              + ", ".join(f"minden {s}. képkocka: {p} / {c}" for s, (p, c) in results.items()))

    def test_output_boxes_close_to_detections(self):
        tr = ByteTracker()
        rng = np.random.default_rng(0)
        errs = []
        for k in range(40):
            gt = np.array(box(300 + 2 * k, 300, 30, 80))
            det = gt + rng.normal(0, 2.0, 4)
            out = tr.update(det[None], np.array([0.9]))
            if k > 5:
                errs.append(float(box_iou(out["xyxy"], gt[None])[0, 0]))
        self.assertGreater(np.mean(errs), 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)

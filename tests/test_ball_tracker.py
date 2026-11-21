"""
A labdakövető (ball_tracker.py) egységtesztjei.
Futtatás: python3 tests/test_ball_tracker.py
"""
import os
import sys
import math
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ball_tracker import (BallTracker, interpolate_gaps, pitch_position,  # noqa: E402
                          STATUS_DETECTED, STATUS_PREDICTED, STATUS_LOST)


def cand(x, y, conf=0.6, size=8.0):
    return {"xyxy": (x - size / 2, y - size / 2, x + size / 2, y + size / 2), "conf": conf}


class BallTrackerTests(unittest.TestCase):
    def test_init_requires_strong_or_confirmed(self):
        tr = BallTracker(25, frame_size=(1280, 720))
        s = tr.update([cand(100, 100, conf=0.2)])
        self.assertEqual(s.status, STATUS_LOST)
        # a második, közeli gyenge jelölt megerősíti
        s = tr.update([cand(104, 101, conf=0.2)])
        self.assertEqual(s.status, STATUS_DETECTED)

    def test_prediction_through_short_gap(self):
        tr = BallTracker(25, frame_size=(1280, 720))
        for i in range(10):
            s = tr.update([cand(100 + 10 * i, 300)])
        self.assertEqual(s.status, STATUS_DETECTED)
        s = tr.update([])
        self.assertEqual(s.status, STATUS_PREDICTED)
        self.assertLess(abs(s.x - 200), 4.0)
        # hosszabb kimaradás után már nincs kiadott pozíció
        for _ in range(20):
            s = tr.update([])
        self.assertFalse(s.found)

    def test_picks_consistent_candidate_and_rejects_outlier(self):
        tr = BallTracker(25, frame_size=(1280, 720))
        for i in range(8):
            s = tr.update([cand(400 + 5 * i, 400), cand(1000, 100, conf=0.7)] if i > 2 else [cand(400 + 5 * i, 400)])
        self.assertLess(abs(s.x - 435), 3.0)
        self.assertGreater(tr.stats["rejected"], 0)

    def test_camera_shift_compensation(self):
        tr = BallTracker(25, frame_size=(1280, 720))
        for _ in range(5):
            tr.update([cand(640, 360)])
        # a kamera elfordul: a képtartalom 30 px-t mozdul balra, a labda mérése kimarad
        s = tr.update([], camera_shift=(-30.0, 0.0))
        self.assertEqual(s.status, STATUS_PREDICTED)
        self.assertLess(abs(s.x - 610), 2.0)

    def test_reinit_on_persistent_outside_candidate(self):
        tr = BallTracker(25, frame_size=(1280, 720))
        for _ in range(5):
            tr.update([cand(200, 200)])
        for i in range(4):
            s = tr.update([cand(900 + 3 * i, 500, conf=0.8)])
        self.assertEqual(s.status, STATUS_DETECTED)
        self.assertLess(abs(s.x - 909), 3.0)

    def test_interpolate_and_pitch(self):
        pts, flags = interpolate_gaps([(0, 0), None, None, (3, 3), None], max_gap=2)
        self.assertEqual(flags, ["measured", "interpolated", "interpolated", "measured", None])
        self.assertAlmostEqual(pts[1][0], 1.0)
        tr = BallTracker(25)
        s = tr.update([cand(10, 20, conf=0.9)])
        H = np.diag([0.5, 0.5, 1.0])
        x, y = pitch_position(s, H)
        self.assertTrue(math.isclose(x, 5.0, abs_tol=1e-6) and math.isclose(y, 10.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()

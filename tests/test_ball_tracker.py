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
from ball_tracker import (BallTracker, interpolate_gaps, pitch_position, _cand_center_size,  # noqa: E402
                          STATUS_DETECTED, STATUS_PREDICTED, STATUS_LOST)


def cand(x, y, conf=0.6, size=8.0):
    return {"xyxy": (x - size / 2, y - size / 2, x + size / 2, y + size / 2), "conf": conf}


class BallTrackerTests(unittest.TestCase):
    def test_init_requires_consecutive_detections(self):
        # gyenge jelöltből csak 3 egymást követő, következetes detektálás indít követést
        tr = BallTracker(25, frame_size=(1280, 720))
        s = tr.update([cand(100, 100, conf=0.2)])
        self.assertEqual(s.status, STATUS_LOST)
        s = tr.update([cand(104, 101, conf=0.2)])
        self.assertEqual(s.status, STATUS_LOST)
        s = tr.update([cand(108, 102, conf=0.2)])
        self.assertEqual(s.status, STATUS_DETECTED)
        self.assertAlmostEqual(s.vx, 4.0, delta=0.5)
        # erős jelöltből kettő elég; egyetlen erős (pl. téves) jelölt nem indít
        tr = BallTracker(25, frame_size=(1280, 720))
        self.assertEqual(tr.update([cand(500, 300, conf=0.8)]).status, STATUS_LOST)
        self.assertEqual(tr.update([cand(506, 302, conf=0.8)]).status, STATUS_DETECTED)
        # egymástól messzi, egyszeri téves jelöltek sosem indítanak
        tr = BallTracker(25, frame_size=(1280, 720))
        for i in range(6):
            s = tr.update([cand(100 + 200 * i, 150 + 90 * (i % 3), conf=0.8)])
            self.assertEqual(s.status, STATUS_LOST)

    def test_priors_head_size_pitch(self):
        # kép -> pálya H: 10 px / m, a kép (0,0) = pálya (0,0); a labda várható átmérője 2,2 px
        H = np.diag([0.1, 0.1, 1.0])
        tr = BallTracker(25, frame_size=(1280, 720))
        head_box = (300.0, 200.0, 320.0, 260.0)
        on_head = cand(310, 210, conf=0.8, size=3.0)
        tiny = cand(600, 400, conf=0.8, size=0.6)       # sokkal kisebb a várható labdánál
        off_pitch = cand(1200, 710, conf=0.8, size=3.0)  # pálya: (120, 71) m – messze kint
        good = cand(500, 300, conf=0.8, size=3.0)
        pr = tr._priors([_cand_center_size(c) for c in [on_head, tiny, off_pitch, good]],
                        tr._geometry(H, (1280, 720)), [head_box])
        self.assertTrue(pr[0][1], "fej-régió")
        self.assertLess(pr[1][0], 0.2, "túl kicsi")
        self.assertLess(pr[2][0], 0.6, "pályán kívül (enyhe: a szálló labda vetülete is kieshet)")
        self.assertAlmostEqual(pr[3][0], 1.0)
        self.assertFalse(pr[3][1])
        # fejen lévő jelöltekből nem indul követés, a jó jelöltekből igen
        for i in range(4):
            s = tr.update([cand(310 + i, 210, conf=0.8, size=3.0)], pitch_H=H, person_boxes=[head_box])
            self.assertEqual(s.status, STATUS_LOST)
        tr.update([good], pitch_H=H, person_boxes=[head_box])
        s = tr.update([cand(503, 300, conf=0.8, size=3.0)], pitch_H=H, person_boxes=[head_box])
        self.assertEqual(s.status, STATUS_DETECTED)

    def test_dt_frames_prediction(self):
        # gyors mód: minden 3. képkockán hívjuk – a becslés a valódi képkocka-különbséggel lép
        tr = BallTracker(25, frame_size=(1280, 720))
        for i in range(6):
            s = tr.update([cand(100 + 30 * i, 300)], frame_idx=3 * i, dt_frames=3)
        self.assertEqual(s.status, STATUS_DETECTED)
        self.assertAlmostEqual(s.vx, 10.0, delta=1.5)   # px / képkocka
        s = tr.update([], frame_idx=18, dt_frames=3)
        self.assertEqual(s.status, STATUS_PREDICTED)
        self.assertLess(abs(s.x - 280), 6.0)

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

    def test_relocated_ball_at_restart_spot(self):
        # LABDA-ÁTHELYEZÉS: kép -> pálya H 10 px / m; a középpont (52,5; 34) m = (525, 340) px
        H = np.diag([0.1, 0.1, 1.0])
        tr = BallTracker(25, frame_size=(1280, 720))
        for i in range(6):
            tr.update([cand(200 + 8 * i, 500, conf=0.7, size=3.0)], pitch_H=H)
        for _ in range(20):   # a labdát elviszik (gól után): a követés elvész
            s = tr.update([], pitch_H=H)
        self.assertFalse(s.found)
        # álló, gyenge (init_conf alatti) jelölt a középponton: két detektálás elég, köztük kimaradással is
        self.assertEqual(tr.update([cand(525, 340, conf=0.25, size=3.0)], pitch_H=H).status, STATUS_LOST)
        tr.update([], pitch_H=H)
        s = tr.update([cand(525.5, 340, conf=0.25, size=3.0)], pitch_H=H)
        self.assertEqual(s.status, STATUS_DETECTED)
        self.assertLess(abs(s.x - 525.5), 1.0)
        self.assertEqual(tr.stats["relocated"], 1)
        # máshol (nem újraindítási ponton) a gyenge álló jelölthöz továbbra is 3 detektálás kell
        tr = BallTracker(25, frame_size=(1280, 720))
        self.assertEqual(tr.update([cand(300, 200, conf=0.25, size=3.0)], pitch_H=H).status, STATUS_LOST)
        self.assertEqual(tr.update([cand(300, 200, conf=0.25, size=3.0)], pitch_H=H).status, STATUS_LOST)
        self.assertEqual(tr.update([cand(300, 200, conf=0.25, size=3.0)], pitch_H=H).status, STATUS_DETECTED)

    def test_static_ball_reacquired_while_coasting(self):
        # a mozgó labda hirtelen megáll a kapun kívül (alapvonalon túl, ill. hálóban): a becslés továbbsodródik,
        # az álló, következetes jelölt viszont nem kap álló-folt büntetést -> a követő átvált rá
        H = np.diag([0.1, 0.1, 1.0])
        tr = BallTracker(25, frame_size=(1280, 720))
        for i in range(8):
            tr.update([cand(800 + 25 * i, 345, conf=0.7, size=3.0)], pitch_H=H)
        net = (1060.0, 345.0)   # (106, 34,5) m: a jobb kapu hálójában
        found = None
        for k in range(8):
            s = tr.update([cand(*net, conf=0.5, size=3.0)] if k % 2 == 0 else [], pitch_H=H)
            if s.status == STATUS_DETECTED and abs(s.x - net[0]) < 2 and found is None:
                found = k
        self.assertIsNotNone(found, "a hálóban álló labdát vissza kell találni")
        self.assertLessEqual(found, 4)
        pr = tr._priors([_cand_center_size(cand(*net, conf=0.5, size=3.0))], tr._geometry(H, (1280, 720)), [])
        self.assertAlmostEqual(pr[0][0], 1.0)
        self.assertEqual(pr[0][3], "kapuháló")

    def test_engine_accepts_ball_in_net(self):
        # a motor a hálóban nyugvó labdát (a gólvonal mögött, a kapufák között) is átadja az events.py-nak
        try:
            from tactical_engine import TacticalEngine
        except ImportError as ex:   # pragma: no cover
            self.skipTest(f"hiányzó csomag: {ex}")
        eng = TacticalEngine()
        H = np.eye(3)
        self.assertIsNotNone(eng.transform_ball_to_pitch(-3.5, 34.0, H))     # bal háló (vetítési tűréssel)
        self.assertIsNotNone(eng.transform_ball_to_pitch(107.2, 37.0, H))    # jobb háló
        self.assertIsNone(eng.transform_ball_to_pitch(-3.5, 20.0, H))        # alapvonal mögött, a kapun kívül
        self.assertIsNone(eng.transform_ball_to_pitch(108.6, 28.0, H))       # pl. a kapus kezében tartott labda vetülete
        self.assertIsNotNone(eng.transform_ball_to_pitch(-1.5, 60.0, H))     # a régi 2 m-es tűrés megmaradt
        self.assertIsNone(eng.transform_ball_to_pitch(50.0, 71.0, H))

    def test_interpolate_and_pitch(self):
        pts, flags = interpolate_gaps([(0, 0), None, None, (3, 3), None], max_gap=2)
        self.assertEqual(flags, ["measured", "interpolated", "interpolated", "measured", None])
        self.assertAlmostEqual(pts[1][0], 1.0)
        tr = BallTracker(25)
        tr.update([cand(9, 20, conf=0.9)])
        s = tr.update([cand(10, 20, conf=0.9)])
        H = np.diag([0.5, 0.5, 1.0])
        x, y = pitch_position(s, H)
        self.assertTrue(math.isclose(x, 5.0, abs_tol=1e-6) and math.isclose(y, 10.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()

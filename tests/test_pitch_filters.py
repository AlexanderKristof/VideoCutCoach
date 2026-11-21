"""
A pálya-koordinátás szűrők (pitch_filters.py: PitchBoundary, PitchTrackSanity, KeeperGate,
pitch_h_trusted) egységtesztjei.
Futtatás: python3 tests/test_pitch_filters.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pitch_filters import (PitchBoundary, PitchTrackSanity, KeeperGate, PitchGeometry,  # noqa: E402
                           pitch_h_trusted, outside_distance_m, in_penalty_area)

# kép -> pálya: 10 px / m, a pálya (0, 0) a kép (100, 100) pontja; 1280x900-as kép
H = np.array([[0.1, 0.0, -10.0], [0.0, 0.1, -10.0], [0.0, 0.0, 1.0]])
SIZE = (1280, 900)


def img(x, y):
    """pálya (m) -> kép (px)"""
    return 100 + 10 * x, 100 + 10 * y


class PitchFilterTests(unittest.TestCase):
    def test_trust(self):
        self.assertFalse(pitch_h_trusted(None))
        self.assertTrue(pitch_h_trusted(H))   # statikus H (nincs pályakövető)
        self.assertTrue(pitch_h_trusted(H, {"status": "ok", "conf": 0.8, "lost": 0}))
        self.assertFalse(pitch_h_trusted(H, {"status": "ok", "conf": 0.2, "lost": 0}))
        self.assertFalse(pitch_h_trusted(H, {"status": "elveszett", "conf": 0.9, "lost": 20}))

    def test_geometry(self):
        g = PitchGeometry(H, SIZE)
        x, y = g.to_pitch(*img(30, 40))
        self.assertAlmostEqual(x, 30.0)
        self.assertAlmostEqual(y, 40.0)
        self.assertAlmostEqual(g.pixels_per_meter(30, 40), 10.0)
        self.assertEqual(outside_distance_m(50, 30), 0.0)
        self.assertAlmostEqual(outside_distance_m(50, -3), 3.0)
        self.assertTrue(in_penalty_area(10, 34))
        self.assertTrue(in_penalty_area(100, 20))
        self.assertFalse(in_penalty_area(50, 34))
        self.assertFalse(in_penalty_area(10, 2))

    def test_boundary(self):
        pb = PitchBoundary(SIZE, margin_m=2.5)
        # nem megbízható H: minden átmegy
        pb.set_frame(H, trusted=False)
        self.assertTrue(pb.check(*img(50, -8), 20)[0])
        pb.set_frame(H, trusted=True)
        # pályán, bedobásnál a vonal mögött 1 m-re, kapus az alapvonal mögött: marad
        for x, y in ((50, 30), (40, -1.0), (-1.5, 34), (106, 69)):
            ok, why, xy = pb.check(*img(x, y), 20)
            self.assertTrue(ok, (x, y, why))
        # a túlsó oldalvonal mögött 6 m-re (néző a fűsávon): képtérben kiesik
        ok, why, _ = pb.check(*img(50, -6), 20)
        self.assertFalse(ok)
        self.assertIn("képtér", why)
        # kapu mögött 6 m-re: pályatérben esik ki
        ok, why, _ = pb.check(*img(-6, 34), 20)
        self.assertFalse(ok)
        self.assertIn("pályatér", why)
        # a margó határán (2,8 m) a doboz-remegés tűrése (12% * 30 px = 3,6 px = 0,36 m) megtartja
        self.assertTrue(pb.check(*img(50, -2.8), 30)[0])

    def test_track_sanity_outside_and_recovery(self):
        ts = PitchTrackSanity(25)
        sup = [ts.update(1, f, (40.0 + 0.05 * f, -2.0))[0] for f in range(100)]   # edző a vonal mellett
        self.assertFalse(sup[10])
        self.assertTrue(sup[-1])
        # bejön a pályára: 1 mp után visszatér
        sup = [ts.update(1, 100 + f, (45.0, 5.0 + 0.1 * f))[0] for f in range(40)]
        self.assertFalse(sup[-1])
        # rövid bedobás (1 mp kint) nem nyomja el a játékost
        sup = [ts.update(2, f, (30.0 + 0.2 * f, 10.0 if f < 50 or f > 75 else -1.0))[0] for f in range(120)]
        self.assertFalse(any(sup))

    def test_track_sanity_id_switch_inherits(self):
        ts = PitchTrackSanity(25)
        for f in range(60):
            sup = ts.update(5, f, (50.0, -2.5))[0]
        self.assertTrue(sup)
        # azonosító-váltás ugyanott: az új követés azonnal elnyomva
        self.assertTrue(ts.update(6, 61, (50.4, -2.6))[0])
        # máshol (a pályán) született új követés nem
        self.assertFalse(ts.update(7, 61, (50.0, 10.0))[0])

    def test_track_sanity_never_moves(self):
        ts = PitchTrackSanity(25)
        sup = [ts.update(3, f, (60.0 + 0.002 * f, 20.0))[0] for f in range(120)]
        self.assertFalse(sup[50])
        self.assertTrue(sup[-1])
        # megerősített kapus kivétel
        sup = [ts.update(4, f, (3.0, 34.0), exempt_still=True)[0] for f in range(150)]
        self.assertFalse(any(sup))
        # ha elindul, magától visszatér
        sup = [ts.update(3, 120 + f, (60.0 + 0.3 * f, 20.0))[0] for f in range(10)]
        self.assertFalse(sup[-1])

    def test_keeper_gate(self):
        kg = KeeperGate(25)
        # színre kapus, de a 16-osokon kívül él (narancs mezes csere a vonal mellett): nem kapus
        for f in range(50):
            res = kg.update(10, f, False, True, (50.0, -2.0))
        self.assertEqual(res, (False, False, False))
        # színre kapus a 16-oson belül: megerősített
        for f in range(50):
            res = kg.update(11, f, False, True, (5.0, 30.0))
        self.assertEqual(res, (True, True, True))
        # pálya-koordináta nélkül: kapusnak jelölhető, de nem megerősített (nem mentesül)
        res = kg.update(12, 0, False, True, None)
        self.assertEqual(res, (True, False, True))
        # a modell kapus osztálya mindig megerősít
        res = kg.update(13, 0, True, False, (50.0, 30.0))
        self.assertEqual(res[:2], (True, True))


if __name__ == "__main__":
    unittest.main()

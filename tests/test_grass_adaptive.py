"""
Regressziós teszt: adaptív fűmodell (pitch_filters.GrassModel) és árnyéktűrő csapatba sorolás
rossz minőségű (árnyékos, napfényes, esti) szintetikus képeken.
Futtatás: python3 tests/test_grass_adaptive.py   (pytest nem kell; ~60 mp)

1. Fűmaszk: IoU a renderelő valós fűmaszkjával (Renderer.grass_gt_mask; az alakok dobozai nem
   számítanak), a régi fix HSV-dobozhoz (30,40,40–85,255,255) képest. Az adaptív modell nem lehet
   rosszabb, és a tiszta / árnyékos / esti képen ≥ 0,95.
2. Csapatba sorolás árnyékban: nehéz mezpár (fehér vs. világoskék), a pálya túlsó fele árnyékban.
   A régi leíróval a napos fehér mez külön ("egyéb") klaszterbe kerül; a fűmodell szerinti
   megvilágítás-korrekcióval (illumination_gain) a játékosok ≥ 95%-a jó csapatba kerül.
3. A fűmodell kimenetei: HSV-burok (a pályakövetőnek), módusok, a kézi tartomány elsőbbsége.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import synth_match as sm  # noqa: E402
import team_classifier as tc  # noqa: E402
from pitch_filters import (GrassModel, build_grass_mask, foot_on_grass, grass_mode_from_config,  # noqa: E402
                           extract_dominant_color)

HSV_LO, HSV_HI = np.array([30, 40, 40]), np.array([85, 255, 255])
_R = {}


def render(ids, degrade, duration, kits=None):
    old = {k: dict(v) for k, v in sm.KITS.items()}
    try:
        if kits:
            sm.KITS.update(kits)
        res, _R["r"] = sm.render_sample_frames(ids, duration=duration, seed=6, degrade=degrade, renderer=_R.get("r"))
    finally:
        sm.KITS.clear()
        sm.KITS.update(old)
    return res


def iou(pred, gt):
    g = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = g != 128
    p, t = pred > 0, g == 255
    return np.count_nonzero(p & t & valid) / max(1, np.count_nonzero((p | t) & valid))


class GrassMaskTests(unittest.TestCase):
    CASES = {"tiszta": None, "árnyék": {"arnyek": 0.5}, "erős árnyék": {"arnyek": 0.4},
             "nap": {"nap": 1.0}, "esti": {"esti": True}, "rossz": dict(sm.DEGRADE_PRESETS["rossz"])}
    MIN_IOU = {"tiszta": 0.95, "árnyék": 0.95, "erős árnyék": 0.95, "esti": 0.95, "nap": 0.80, "rossz": 0.80}

    def test_iou_vs_fixed_hsv(self):
        ids = [5, 60, 120, 180, 240, 300]
        rows = []
        for name, deg in self.CASES.items():
            res = render(ids, deg, 13.0)
            gm = GrassModel()
            for r in res[:3]:
                gm.observe(r["frame"])
            self.assertTrue(gm.ready, name)
            a = [iou(gm.mask(r["frame"])[0], r["grass_gt"]) for r in res]
            h = [iou(build_grass_mask(cv2.cvtColor(r["frame"], cv2.COLOR_BGR2HSV), HSV_LO, HSV_HI)[0], r["grass_gt"])
                 for r in res]
            rows.append((name, np.mean(h), np.mean(a), np.min(a), len(gm.modes)))
            self.assertGreaterEqual(np.mean(a), self.MIN_IOU[name], (name, a))
            self.assertGreaterEqual(np.mean(a), np.mean(h) - 0.01, (name, a, h))
            if name in ("árnyék", "erős árnyék"):
                self.assertEqual(len(gm.modes), 2, gm.describe())   # napos + árnyékos módus
        print("\nFűmaszk IoU (fix HSV -> adaptív, min):")
        for name, h, a, mn, nm in rows:
            print(f"  {name:12s} {h:.3f} -> {a:.3f} (min {mn:.3f}, {nm} módus)")

    def test_foot_on_grass_in_shadow(self):
        """Az árnyékban álló játékos lába is füvön van (a pálya túlsó, árnyékos felén)."""
        res = render([20, 100, 200], {"arnyek": 0.4}, 9.0)
        gm = GrassModel()
        for r in res:
            gm.observe(r["frame"])
        n = ok = 0
        for r in res:
            mask, sc = gm.mask(r["frame"])
            for pid, role, team, bb, xy in r["people"]:
                if role != "player" or bb[3] - bb[1] < 16 or xy[1] > float(sm.shadow_edge_y(xy[0])) - 1.0:
                    continue
                # takart láb (egy másik alak doboza a talp körüli sávba lóg) – ez nem a fűmaszk hibája
                pad = 0.35 * (bb[2] - bb[0])
                if any(o[0] != pid and o[3][0] < bb[2] and o[3][2] > bb[0] and o[3][1] < bb[3] + pad
                       and o[3][3] > bb[3] - pad for o in r["people"]):
                    continue
                on, _ = foot_on_grass(mask, sc, int(bb[0]), int(bb[2]), int(bb[3]), 0.3)
                n += 1
                ok += int(on)
        self.assertGreater(n, 10)
        self.assertGreaterEqual(ok / n, 0.95, (ok, n))


class TeamShadowTests(unittest.TestCase):
    def test_white_vs_sky_in_shadow(self):
        kits = {"team1": {"shirt": (235, 235, 235), "shorts": (30, 30, 30), "socks": (235, 235, 235)},
                "team2": {"shirt": (215, 175, 110), "shorts": (235, 235, 235), "socks": (215, 175, 110)}}
        res = render(list(range(0, 400, 4)), {"arnyek": 0.5}, 17.0, kits=kits)
        gm = GrassModel()
        for r in res[:3]:
            gm.observe(r["frame"])

        def run(use_gain):
            clf = tc.TeamClassifier(grass_lower=HSV_LO, grass_upper=HSV_HI, grass_model=gm)
            out = []
            for k, r in enumerate(res):
                fr, t = r["frame"], k * 0.16
                for pid, role, team, bb, xy in r["people"]:
                    if role != "player":
                        continue
                    x1, y1 = max(0, int(round(bb[0]))), max(0, int(round(bb[1])))
                    x2, y2 = min(fr.shape[1], int(round(bb[2]))), min(fr.shape[0], int(round(bb[3])))
                    if y2 - y1 < 16 or x2 <= x1:
                        continue
                    gain = gm.illumination_gain(fr, (x1, y1, x2, y2))[0] if use_gain else None
                    a = clf.observe(pid, fr[y1:y2, x1:x2], t, pitch_xy=xy, illum_gain=gain)
                    if t > 13.0:   # a végleges illesztés után
                        out.append((a["team_id"], team))
            acc = max(np.mean([o[0] == o[1] for o in out]), np.mean([o[0] == 3 - o[1] for o in out]))
            return acc, len(out), clf
        acc_old, n, _ = run(False)
        acc_new, _, clf = run(True)
        print(f"\nCsapat-pontosság árnyékban (fehér vs. világoskék, n={n}): korrekció nélkül {acc_old:.3f}, "
              f"megvilágítás-korrekcióval {acc_new:.3f}")
        self.assertGreater(n, 100)
        self.assertGreaterEqual(acc_new, 0.95, clf.describe())
        self.assertGreaterEqual(acc_new, acc_old)


class GrassModelApiTests(unittest.TestCase):
    def test_mode_and_outputs(self):
        self.assertEqual(grass_mode_from_config({"grass_lower": [30, 40, 40], "grass_upper": [85, 255, 255]}), "auto")
        self.assertEqual(grass_mode_from_config({"grass_lower": [0, 0, 0], "grass_upper": [179, 255, 255]}), "auto")
        self.assertEqual(grass_mode_from_config({"grass_lower": [35, 60, 60], "grass_upper": [80, 255, 255]}), "hsv")
        self.assertEqual(grass_mode_from_config({"grass_model": "hsv"}), "hsv")
        self.assertEqual(grass_mode_from_config({}), "auto")
        gm = GrassModel()
        lo, hi = gm.hsv_bounds()                       # tanulás előtt: az alap tartomány
        self.assertEqual(lo.tolist(), [30, 40, 40])
        img = np.zeros((360, 640, 3), np.uint8)
        img[:] = (50, 140, 60)
        img[:120] = (200, 180, 150)                    # "égbolt"
        m0, _ = gm.mask(img)                           # tanulás előtt: HSV tartalék
        self.assertIsNotNone(m0)
        for _ in range(3):
            gm.observe(img)
        self.assertTrue(gm.ready)
        m, sc = gm.mask(img)
        self.assertGreater(np.mean(m[100:] > 0), 0.95)
        self.assertLess(np.mean(m[:50] > 0), 0.05)
        lo, hi = gm.hsv_bounds()
        hsv = cv2.cvtColor(np.uint8([[[50, 140, 60]]]), cv2.COLOR_BGR2HSV)[0, 0]
        self.assertTrue(np.all(hsv >= lo) and np.all(hsv <= hi), (lo, hi, hsv))
        crop = np.zeros((40, 16, 3), np.uint8)
        crop[:] = (50, 140, 60)
        crop[8:24, 4:12] = (40, 40, 200)               # piros mez a füvön
        r, g, b = extract_dominant_color(crop, grass_model=gm)
        self.assertGreater(r, 150)
        self.assertLess(g, 80)


class TestGrassSpeed(unittest.TestCase):
    """SEBESSÉG: a gyorsított fűmaszk (cv2-műveletek, táblázat-index) pixelre azonos a régi (bool
    tömbös) számítással; a GrassMaskCache a köztes képkockákon az eltolt maszkot adja."""

    @staticmethod
    def _old_mask(gm, frame, scale=0.5):
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh = 8 - gm.LUT_BITS
        b, g, r = [small[..., i].astype(np.uint32) >> sh for i in range(3)]
        gm.labels(small[:2, :2].reshape(-1, 3))          # a táblázat felépítése
        lab = gm._lut[(b << (2 * gm.LUT_BITS)) | (g << gm.LUT_BITS) | r]
        raw = (lab > 0) & (lab < gm.WEAK)
        solid = (lab == 1) | (lab == 2)
        dense = solid & (cv2.blur(solid.astype(np.float32), (5, 5)) >= 0.7)
        near = cv2.dilate(dense.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        extra = (lab == gm.WEAK) & near
        h, w = raw.shape
        dens = cv2.blur(raw.astype(np.float32), (gm.DENS_WIN, gm.DENS_WIN))
        rawd = cv2.dilate(raw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        core = np.where((dens >= gm.DENS_MIN) & rawd, 255, 0).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        if n > 2:
            keep = np.zeros(n, bool)
            for i in range(1, n):
                y0, hh, area = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
                keep[i] = (y0 + hh > int(h * 0.55) and area >= 0.002 * h * w) or area >= 0.2 * h * w
            core = np.where(keep[cc], 255, 0).astype(np.uint8)
        zone = cv2.dilate(core, np.ones((5, 5), np.uint8)) > 0
        mask = np.where((raw | extra) & zone, 255, 0).astype(np.uint8)
        k = max(5, (int(h / 60) // 2) * 2 + 1)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), lab

    def test_fast_mask_and_labels_match_reference(self):
        for cond, deg in (("árnyék", {"arnyek": 0.4}), ("nap", {"nap": 1.0}), ("esti", {"esti": True})):
            res = render([30, 150], deg, 7.0)
            img = res[1]["frame"]
            gm = GrassModel()
            for _ in range(3):
                gm.observe(img)
            self.assertTrue(gm.ready)
            small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            ref_mask, ref_lab = self._old_mask(gm, img)
            self.assertTrue(np.array_equal(gm.labels(small), ref_lab), cond)
            # a nem kép alakú (N, 3) bemenet a régi úton
            px = small.reshape(-1, 3)[::97]
            self.assertTrue(np.array_equal(gm.labels(px), ref_lab.reshape(-1)[::97]), cond)
            m, sc = gm.mask(img)
            self.assertEqual(sc, 0.5)
            self.assertTrue(np.array_equal(m, ref_mask), (cond, int(np.count_nonzero(m != ref_mask))))

    def test_box_count_matches_blur(self):
        from pitch_filters import _box_count_ge
        rng = np.random.default_rng(0)
        m = np.where(rng.random((120, 160)) < 0.55, 255, 0).astype(np.uint8)
        for win, frac in ((5, 0.5), (5, 0.7), (3, 0.4)):
            ref = cv2.blur((m > 0).astype(np.float32), (win, win)) >= frac
            self.assertTrue(np.array_equal(_box_count_ge(m, win, frac) > 0, ref), (win, frac))

    def test_mask_cache_shifts_between_updates(self):
        from pitch_filters import GrassMaskCache
        calls = []
        base = np.zeros((100, 200), np.uint8)
        base[60:, :] = 255

        def compute(frame):
            calls.append(1)
            return base.copy(), 0.5

        cache = GrassMaskCache(compute, every=3)
        m0, _ = cache.get(None, np.array([0.0, 0.0]))
        # a kamera 20 px-t "lefelé" mozdult (teljes felbontás) -> a maszk 10 px-lel lejjebb kerül
        m1, s1 = cache.get(None, np.array([0.0, 20.0]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(s1, 0.5)
        self.assertEqual(int(np.argmax(m1[:, 0] > 0)), 70)
        cache.get(None, np.array([0.0, 20.0]))
        self.assertEqual(len(calls), 1)
        cache.get(None, np.array([0.0, 20.0]))         # every = 3: újraszámolás
        self.assertEqual(len(calls), 2)
        cache.get(None, np.array([0.0, 200.0]))        # nagy ugrás: mindig újraszámol
        self.assertEqual(len(calls), 3)
        one = GrassMaskCache(compute, every=1)
        for _ in range(3):
            one.get(None, np.zeros(2))
        self.assertEqual(len(calls), 6)                 # every = 1: a régi működés


if __name__ == "__main__":
    unittest.main(verbosity=2)

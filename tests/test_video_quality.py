"""
A videóminőség-figyelő (video_quality.py) egységtesztjei szintetikus képkockákon.
Futtatás: python3 tests/test_video_quality.py   (pytest nem kell; ~20 mp)

A képkockák a tools/synth_match.py renderelőjéből jönnek (videókódolás nélkül), a képhibák:
  * elmosódás: erős vízszintes mozgási elmosódás (forgó kamera),
  * napfény: --nap (csillanás, fátyolfény, beégett csúcsfények),
  * árnyék: --arnyek (éles lelátó-árnyék a pálya túlsó felén),
  * kockás tömörítés: erős JPEG-tömörítés (8x8-as blokkok),
  * sötét / zajos: --esti (sötét, meleg színezet, zaj) és erős Gauss-zaj.
Mindegyikre: a saját hibacímkéje megjelenik, a megbízhatóság kisebb, mint a tiszta képen; a tiszta
képen nincs hibacímke. Plusz: összesítő (osztályzat, idő-arányok, legrosszabb szakaszok) és az
események megbízhatóság-mezője.
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
import video_quality as vq  # noqa: E402
from pitch_filters import GrassModel  # noqa: E402

IDS = [5, 60, 120, 180, 240, 300]
_CACHE = {}


def frames(kind):
    """Renderelt képkockák egy rontással (gyorsítótárazva, a renderelő újrahasznosítva)."""
    if kind not in _CACHE:
        deg = {"clean": None, "arnyek": {"arnyek": 0.5}, "nap": {"nap": 1.0}, "esti": {"esti": True}}[kind]
        res, _CACHE["_r"] = sm.render_sample_frames(IDS, duration=13.0, seed=6, degrade=deg,
                                                    renderer=_CACHE.get("_r"))
        _CACHE[kind] = [r["frame"] for r in res]
    return _CACHE[kind]


def motion_blur(img, length=17):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    return cv2.filter2D(img, -1, k)


def jpeg(img, q=12):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def noisy(img, sigma=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def run_monitor(imgs, warm=None):
    """A figyelő végigfuttatása (0,5 mp-enként egy kép). warm: előtte tiszta képek (futó referencia)."""
    gm = GrassModel()
    for im in (warm or imgs)[:3]:
        gm.observe(im)
    qm = vq.VideoQualityMonitor(25, grass_model=gm)
    t = 0.0
    for im in (warm or []) + list(imgs):
        qm.update(im, t)
        t += 0.5
    return qm, qm.samples[len(warm or []):]


def flag_rate(samples, issue):
    return float(np.mean([issue in s["flags"] for s in samples]))


def mean_rel(samples):
    return float(np.mean([s["reliability"] for s in samples]))


class VideoQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.clean = run_monitor(frames("clean"))
        cls.clean_rel = mean_rel(cls.clean)

    def test_clean(self):
        for s in self.clean:
            self.assertEqual(s["flags"], [], s)
        self.assertGreaterEqual(self.clean_rel, 0.85)

    def _check(self, samples, issue, min_rate=0.8):
        rate = flag_rate(samples, issue)
        self.assertGreaterEqual(rate, min_rate, (issue, [s["flags"] for s in samples],
                                                 [s["severity"] for s in samples]))
        self.assertLess(mean_rel(samples), self.clean_rel - 0.08)

    def test_blur(self):
        # tiszta bemelegítés után elmosódott képek (a futó élesség-referencia a tiszta képekből)
        _, s = run_monitor([motion_blur(f) for f in frames("clean")], warm=frames("clean"))
        self._check(s, "blur")

    def test_glare(self):
        _, s = run_monitor(frames("nap"))
        self._check(s, "glare")

    def test_shadow(self):
        _, s = run_monitor(frames("arnyek"))
        self._check(s, "shadow")
        for x in s:
            self.assertGreater(x["metrics"]["shadow_frac"], 0.1)
            self.assertLess(x["metrics"]["shadow_ratio"], 0.7)

    def test_blocking(self):
        _, s = run_monitor([jpeg(f) for f in frames("clean")])
        self._check(s, "blocking")
        # a jó minőségű JPEG nem kockás
        _, s2 = run_monitor([jpeg(f, 95) for f in frames("clean")])
        self.assertEqual(flag_rate(s2, "blocking"), 0.0)

    def test_dark_noisy(self):
        _, s = run_monitor(frames("esti"))
        self._check(s, "dark_noise")
        _, s2 = run_monitor([noisy(f) for f in frames("clean")])
        self._check(s2, "dark_noise")

    def test_summary_and_events(self):
        clean = frames("clean")
        imgs = clean + [motion_blur(f) for f in clean[:4]] + clean
        qm, _ = run_monitor(imgs, warm=None)
        summ = qm.summary()
        self.assertIn(summ["grade"], ("A", "B", "C", "D"))
        self.assertIn("elmosódott", summ["issue_time_pct"])
        self.assertGreater(summ["issue_time_pct"]["elmosódott"], 10.0)
        worst = summ["worst_segments"][0]
        self.assertEqual(worst["issue"], "blur")
        # a homályos szakasz a 6. mintától (3,0 mp) a 9.-ig (4,5 mp) tart
        self.assertLessEqual(abs(worst["start_sec"] - 2.75), 0.6)
        self.assertLessEqual(abs(worst["end_sec"] - 4.75), 0.6)
        self.assertLess(qm.reliability_at(3.5), qm.reliability_at(0.5))
        evs = [{"type": "shot", "time": 3.6}, {"type": "goal", "time": 0.4}, {"type": "x"}]
        vq.annotate_event_reliability(evs, qm)
        self.assertLess(evs[0]["reliability"], evs[1]["reliability"])
        self.assertIn("elmosódott", evs[0]["quality_flags"])
        self.assertNotIn("quality_flags", evs[1])
        self.assertNotIn("reliability", evs[2])
        self.assertTrue(summ["summary_text"].startswith("Videóminőség:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
A tanítási segédeszközök tesztjei: tools/dataset_formats.py (YOLO <-> COCO, Label Studio, felosztás)
és tools/export_training_frames.py (homályos / majdnem azonos képek kiszűrése, változatos válogatás).
Futtatás: python3 tests/test_training_tools.py
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

import dataset_formats as dfm  # noqa: E402


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _items(tmp, n=6):
    os.makedirs(os.path.join(tmp, "src"), exist_ok=True)
    items = []
    for i in range(n):
        name = f"meccs{i % 2}_{i * 1000:07d}.jpg"
        if HAVE_CV2:
            cv2.imwrite(os.path.join(tmp, "src", name), np.full((90, 160, 3), 40 * i % 255, np.uint8))
        items.append({"file": name, "width": 160, "height": 90,
                      "boxes": [{"cls": 2, "xyxy": [10.0, 20.0, 30.0, 70.0], "score": 0.9, "track": 3},
                                {"cls": 0, "xyxy": [100.5, 50.5, 104.5, 54.5]},
                                {"cls": 3, "xyxy": [150.0, 10.0, 170.0, 60.0]}]})   # a képen túllógó -> vágva
    return items


class DatasetFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="adatkeszlet_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_coco_roundtrip(self):
        items = _items(self.tmp)
        out = os.path.join(self.tmp, "coco")
        dfm.write_coco(items, out, os.path.join(self.tmp, "src"))
        data = _load(os.path.join(out, "annotations", "train.json"))
        self.assertEqual([c["name"] for c in data["categories"]], dfm.CLASSES)
        self.assertEqual(data["categories"][0]["id"], 1)   # CVAT: a 0-s id "címke nélküli" lenne
        self.assertEqual(len(data["annotations"]), 18)
        a = data["annotations"][0]
        self.assertEqual(a["bbox"], [10.0, 20.0, 20.0, 50.0])
        self.assertEqual(a["attributes"]["track_id"], 3)
        back, classes = dfm.read_coco(os.path.join(out, "annotations", "train.json"))
        self.assertEqual(classes, dfm.CLASSES)
        self.assertEqual(back[0]["boxes"][2]["xyxy"], [150.0, 10.0, 160.0, 60.0])

    @unittest.skipUnless(HAVE_CV2, "nincs OpenCV")
    def test_yolo_roundtrip_and_split(self):
        items = _items(self.tmp)
        splits = dfm.split_items(items, val_frac=0.3, seed=1, group_key=dfm.video_block_key)
        self.assertEqual(sum(len(v) for v in splits.values()), len(items))
        self.assertTrue(splits["train"])
        out = os.path.join(self.tmp, "yolo")
        dfm.write_yolo(items, out, os.path.join(self.tmp, "src"), splits=splits)
        self.assertTrue(os.path.exists(os.path.join(out, "data.yaml")))
        back, classes = dfm.read_yolo(out)
        self.assertEqual(len(back), len(items))
        b = back[0]["boxes"]
        self.assertEqual([x["cls"] for x in b], [2, 0, 3])
        np.testing.assert_allclose(b[0]["xyxy"], [10, 20, 30, 70], atol=0.01)
        np.testing.assert_allclose(b[2]["xyxy"], [150, 10, 160, 60], atol=0.01)
        # COCO rfdetr elrendezés a YOLO-ból
        rf = os.path.join(self.tmp, "rf")
        dfm.write_coco(back, rf, out, classes, {"train": back[:4], "val": back[4:]}, layout="rfdetr")
        self.assertTrue(os.path.exists(os.path.join(rf, "valid", "_annotations.coco.json")))
        self.assertEqual(len([f for f in os.listdir(os.path.join(rf, "train")) if f.endswith(".jpg")]), 4)

    def test_video_block_key_groups_neighbours(self):
        k = dfm.video_block_key
        self.assertEqual(k({"file": "a_0000010.jpg"}), k({"file": "a_0000700.jpg"}))
        self.assertNotEqual(k({"file": "a_0000010.jpg"}), k({"file": "a_0002000.jpg"}))
        self.assertNotEqual(k({"file": "a_0000010.jpg"}), k({"file": "b_0000010.jpg"}))

    def test_labelstudio(self):
        items = _items(self.tmp, 1)
        p = dfm.write_labelstudio(items, os.path.join(self.tmp, "ls.json"))
        t = _load(p)[0]
        r = t["predictions"][0]["result"][0]
        self.assertEqual(r["value"]["rectanglelabels"], ["player"])
        self.assertAlmostEqual(r["value"]["x"], 100 * 10 / 160)
        self.assertAlmostEqual(r["value"]["height"], 100 * 50 / 90)
        self.assertAlmostEqual(r["score"], 0.9)


@unittest.skipUnless(HAVE_CV2, "nincs OpenCV")
class ExportFramesTests(unittest.TestCase):
    def test_dedupe_blur_and_diversity(self):
        import export_training_frames as etf
        tmp = tempfile.mkdtemp(prefix="kepkocka_export_")
        try:
            path = os.path.join(tmp, "proba.avi")
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
            rng = np.random.default_rng(0)
            base = (rng.random((180, 320, 3)) * 255).astype(np.uint8)
            for i in range(120):
                if i < 40:
                    f = base.copy()                                   # 4 mp álló, azonos kép
                elif i < 60:
                    f = np.full((180, 320, 3), 128, np.uint8)         # homályos (egyszínű) képek
                else:
                    f = np.roll(base, 7 * i, axis=1)                  # pásztázás
                    cv2.circle(f, (40 + i, 90), 20, (0, 0, 255), -1)
                    f = (f * (0.5 + 0.5 * (i % 3) / 2)).astype(np.uint8)   # változó fényerő
                vw.write(f)
            vw.release()
            out = os.path.join(tmp, "ki")
            rows = etf.export([path], out, per_video=5, every_sec=0.5, log=lambda *a: None)
            self.assertEqual(len(rows), 5)
            with open(os.path.join(out, "frames.csv"), encoding="utf-8") as f:
                rd = list(csv.DictReader(f))
            self.assertEqual(len(rd), 5)
            frames = [int(r["frame"]) for r in rd]
            self.assertLessEqual(sum(1 for x in frames if x < 40), 1)      # az álló részből legfeljebb egy
            self.assertFalse(any(40 <= x < 60 for x in frames))            # homályos kimarad
            for r in rd:
                self.assertTrue(os.path.exists(os.path.join(out, r["file"])))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_farthest_point_select(self):
        import export_training_frames as etf
        feats = [np.array([0.0]), np.array([0.01]), np.array([5.0]), np.array([10.0]), np.array([10.01])]
        sel = etf.farthest_point_select(feats, 3, seed=0)
        vals = sorted(float(feats[i][0]) for i in sel)
        self.assertLess(vals[0], 1)
        self.assertTrue(4 < vals[1] < 6)
        self.assertGreater(vals[2], 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)

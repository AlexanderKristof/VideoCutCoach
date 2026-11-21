"""
A detektor-réteg tesztjei: NMS, letterbox oda-vissza leképezés, ONNX detektor apró, a tesztben
épített ONNX modellekkel (YOLO / YOLOX / DETR kimeneti konvenció), a saját háttér (OwnTrackerBackend).
Az ONNX részek kimaradnak, ha nincs telepítve az onnx / onnxruntime csomag.
Futtatás: python3 tests/test_onnx_detector.py
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from detection.detector_base import Detections, Detector, nms, box_iou, letterbox  # noqa: E402
from detection.onnx_detector import select_providers, parse_class_names, COCO80  # noqa: E402
from detection.backend import OwnTrackerBackend, create_backend  # noqa: E402
from model_config import build_role_class_map  # noqa: E402

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import onnxruntime  # noqa: F401
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


# ---------------------------------------------------------------------- apró ONNX modell építése
def make_const_model(path, outputs, in_h=64, in_w=64, extra_int_input=None, metadata=None):
    """
    Olyan ONNX modell, amely a bemenettől (formailag) függve fix kimeneteket ad:
    out = const + 0 * mean(images). outputs: [(név, numpy tömb)], egész tömb esetén Cast.
    """
    nodes, inits, outs = [], [], []
    inputs = [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, in_h, in_w])]
    if extra_int_input:
        inputs.append(helper.make_tensor_value_info(extra_int_input, TensorProto.INT64, [1, 2]))
    nodes.append(helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0))
    inits.append(numpy_helper.from_array(np.array(0.0, np.float32), "zero"))
    nodes.append(helper.make_node("Mul", ["m", "zero"], ["z"]))
    for name, arr in outputs:
        cname = f"c_{name}"
        integer = np.issubdtype(arr.dtype, np.integer)
        inits.append(numpy_helper.from_array(arr.astype(np.float32), cname))
        if integer:
            nodes.append(helper.make_node("Add", [cname, "z"], [f"f_{name}"]))
            nodes.append(helper.make_node("Cast", [f"f_{name}"], [name], to=TensorProto.INT64))
            outs.append(helper.make_tensor_value_info(name, TensorProto.INT64, list(arr.shape)))
        else:
            nodes.append(helper.make_node("Add", [cname, "z"], [name]))
            outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape)))
    graph = helper.make_graph(nodes, "kicsi", inputs, outs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for k, v in (metadata or {}).items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def img(w=128, h=64):
    return np.full((h, w, 3), 90, np.uint8)


class NmsTests(unittest.TestCase):
    def brute(self, boxes, scores, thr, cls=None):
        order = list(np.argsort(-scores, kind="stable"))
        keep = []
        for i in order:
            ok = True
            for j in keep:
                if cls is not None and cls[i] != cls[j]:
                    continue
                if box_iou(boxes[i:i + 1], boxes[j:j + 1])[0, 0] > thr:
                    ok = False
                    break
            if ok:
                keep.append(i)
        return keep

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            n = int(rng.integers(1, 40))
            xy = rng.uniform(0, 100, (n, 2))
            wh = rng.uniform(5, 40, (n, 2))
            boxes = np.c_[xy, xy + wh].astype(np.float32)
            scores = rng.random(n).astype(np.float32)
            cls = rng.integers(0, 3, n)
            self.assertEqual(nms(boxes, scores, 0.5).tolist(), self.brute(boxes, scores, 0.5))
            self.assertEqual(nms(boxes, scores, 0.4, cls).tolist(), self.brute(boxes, scores, 0.4, cls))

    def test_class_aware_keeps_overlapping_other_class(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], np.float32)
        scores = np.array([0.9, 0.8, 0.7], np.float32)
        self.assertEqual(nms(boxes, scores, 0.5).tolist(), [0])
        self.assertEqual(sorted(nms(boxes, scores, 0.5, cls=[0, 0, 1]).tolist()), [0, 2])
        self.assertEqual(nms(np.zeros((0, 4)), np.zeros(0)).tolist(), [])
        self.assertEqual(len(nms(boxes, scores, 0.99, max_det=2)), 2)


@unittest.skipUnless(HAVE_CV2, "nincs OpenCV")
class LetterboxTests(unittest.TestCase):
    def test_roundtrip_all_modes(self):
        rng = np.random.default_rng(1)
        frame = np.zeros((720, 1280, 3), np.uint8)
        boxes = np.c_[rng.uniform(0, 1000, (20, 2)), rng.uniform(0, 1000, (20, 2)) + 50].astype(np.float32)
        boxes[:, [1, 3]] *= 0.6
        for mode in ("center", "topleft", "stretch"):
            out, info = letterbox(frame, (640, 640), mode=mode)
            self.assertEqual(out.shape, (640, 640, 3))
            back = info.to_original(info.to_input(boxes))
            np.testing.assert_allclose(back, np.clip(boxes, 0, [1280, 720, 1280, 720]), atol=1e-3)
        _, info = letterbox(frame, (640, 640), mode="center")
        self.assertAlmostEqual(info.pad_y, 140.0)
        self.assertAlmostEqual(info.sx, 0.5)
        _, info = letterbox(frame, (640, 640), mode="topleft")
        self.assertEqual((info.pad_x, info.pad_y), (0.0, 0.0))

    def test_pixel_content_maps(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        frame[40:60, 100:120] = 255
        out, info = letterbox(frame, (100, 100), mode="center")
        ys, xs = np.where(out[:, :, 0] > 200)
        back = info.to_original(np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]]))[0]
        np.testing.assert_allclose(back, [100, 40, 120, 60], atol=2.0)


class ProviderTests(unittest.TestCase):
    def test_order_and_fallback(self):
        avail = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(select_providers("auto", avail), ["DmlExecutionProvider", "CPUExecutionProvider"])
        avail = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(select_providers("auto", avail)[0], "CUDAExecutionProvider")
        self.assertEqual(select_providers(["TensorrtExecutionProvider"], ["CPUExecutionProvider"]),
                         ["CPUExecutionProvider"])

    def test_class_names(self):
        self.assertEqual(parse_class_names("coco80")[32], "sports ball")
        self.assertEqual(parse_class_names("{0: 'ball', 1: 'player'}"), {0: "ball", 1: "player"})
        self.assertEqual(parse_class_names(["a", "b"]), {0: "a", 1: "b"})
        self.assertEqual(parse_class_names('{"2": "x"}'), {2: "x"})
        self.assertEqual(parse_class_names("labda, játékos"), {0: "labda", 1: "játékos"})


@unittest.skipUnless(HAVE_ONNX and HAVE_CV2, "nincs onnx / onnxruntime / OpenCV")
class OnnxDetectorTests(unittest.TestCase):
    NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="onnx_teszt_")

    def _det(self, name, outputs, **kw):
        from detection.onnx_detector import OnnxDetector
        meta = kw.pop("metadata", None)
        extra = kw.pop("extra_int_input", None)
        path = make_const_model(os.path.join(self.tmp, name), outputs, metadata=meta, extra_int_input=extra)
        cfg = {"onnx_path": path, "providers": "auto"}
        cfg.update(kw)
        return OnnxDetector(cfg)

    # A 128x64-es kép a 64x64-es bemeneten: arány 0.5, középre igazítva pad_y = 16 (topleft: 0)
    def test_yolo_channels_first_with_nms(self):
        C, N = 4, 6
        out = np.zeros((1, 4 + C, N), np.float32)
        rows = [  # cx, cy, w, h (bemeneti képpont), osztály, pontszám
            (20, 30, 10, 20, 2, 0.9), (21, 30, 10, 20, 2, 0.6),   # átfedő duplikátum -> NMS
            (21, 30, 10, 20, 3, 0.7),                             # más osztály ugyanott -> marad
            (50, 40, 2, 2, 0, 0.08),                              # labda, alacsony pontszám
            (10, 10, 4, 4, 1, 0.02),                              # küszöb alatt
            (5, 5, 2, 2, 2, 0.0)]
        for i, (cx, cy, w, h, c, s) in enumerate(rows):
            out[0, :4, i] = (cx, cy, w, h)
            out[0, 4 + c, i] = s
        det = self._det("yolo.onnx", [("output0", out)], metadata={"names": str(self.NAMES)},
                        conf=0.25, class_conf={0: 0.05})
        self.assertEqual(det.output_format, "yolo")
        self.assertEqual(det.names, self.NAMES)
        d = det.detect(img())
        order = np.argsort(-d.conf)
        self.assertEqual(d.cls[order].tolist(), [2, 3, 0])
        # visszavetítés: x' = x / 0.5, y' = (y - 16) / 0.5
        np.testing.assert_allclose(d.xyxy[order[0]], [30, 8, 50, 48], atol=1e-4)
        np.testing.assert_allclose(d.xyxy[order[2]], [98, 46, 102, 50], atol=1e-4)
        self.assertAlmostEqual(float(d.conf[order[2]]), 0.08, places=5)
        # batch: két kép, ugyanaz az eredmény
        ds = det.detect_batch([img(), img()])
        self.assertEqual([len(x) for x in ds], [3, 3])

    def test_yolox_raw_grid_decoded(self):
        C = 4
        n = 8 * 8 + 4 * 4 + 2 * 2       # 64x64 bemenet, 8/16/32 lépésköz
        out = np.zeros((1, n, 5 + C), np.float32)
        # 8-as rács (3, 2) cellája: cx = (0.5 + 3) * 8 = 28, cy = (0.25 + 2) * 8 = 18, w = e^0 * 8, h = e^ln2 * 8
        k = 2 * 8 + 3
        out[0, k, :5] = (0.5, 0.25, 0.0, np.log(2.0), 0.9)
        out[0, k, 5 + 2] = 0.8
        # 32-es rács (1, 1): cx = cy = 32, w = h = 0.25 * 32 = 8 – labda (az alsó széle a képhatárra vágva)
        k2 = 64 + 16 + 1 * 2 + 1
        out[0, k2, :5] = (0.0, 0.0, np.log(0.25), np.log(0.25), 0.5)
        out[0, k2, 5 + 0] = 0.4
        det = self._det("yolox.onnx", [("output", out)], onnx_class_names=None, class_names=self.NAMES,
                        conf=0.3, class_conf={0: 0.1})
        self.assertEqual(det.output_format, "yolox")
        self.assertEqual(det.pre["letterbox"], "topleft")
        self.assertEqual(det.pre["color"], "bgr")
        d = det.detect(img())
        order = np.argsort(-d.conf)
        self.assertEqual(d.cls[order].tolist(), [2, 0])
        self.assertAlmostEqual(float(d.conf[order[0]]), 0.72, places=5)
        # topleft: x' = x / 0.5, y' = y / 0.5; doboz: 24..32 x 10..26 -> 48..64 x 20..52
        np.testing.assert_allclose(d.xyxy[order[0]], [48, 20, 64, 52], atol=1e-3)
        np.testing.assert_allclose(d.xyxy[order[1]], [56, 56, 72, 64], atol=1e-3)

    def test_detr_normalized_logits(self):
        Q, C = 5, 4
        boxes = np.zeros((1, Q, 4), np.float32)
        logits = np.full((1, Q, C), -8.0, np.float32)
        boxes[0, 0] = (0.5, 0.5, 0.25, 0.5)     # cx, cy, w, h (normalizált)
        logits[0, 0, 2] = 3.0                   # szigmoid ~0.95 player
        boxes[0, 1] = (0.5, 0.52, 0.25, 0.5)    # majdnem azonos doboz: DETR-nél nincs NMS -> marad
        logits[0, 1, 2] = 0.5
        boxes[0, 2] = (0.1, 0.2, 0.02, 0.02)
        logits[0, 2, 0] = -1.5                  # ~0.18 labda
        det = self._det("detr.onnx", [("dets", boxes), ("labels", logits)], class_names=self.NAMES,
                        conf=0.3, class_conf={0: 0.1})
        self.assertEqual(det.output_format, "detr")
        self.assertEqual(det.pre["letterbox"], "stretch")
        d = det.detect(img())
        order = np.argsort(-d.conf)
        self.assertEqual(d.cls[order].tolist(), [2, 2, 0])
        # stretch: x' = xn * 128, y' = yn * 64
        np.testing.assert_allclose(d.xyxy[order[0]], [48, 16, 80, 48], atol=1e-3)
        self.assertAlmostEqual(float(d.conf[order[2]]), 1 / (1 + np.exp(1.5)), places=4)

    def test_detr_postprocessed_with_target_sizes(self):
        Q = 3
        labels = np.array([[2, 0, 1]], np.int64)
        boxes = np.array([[[10, 20, 30, 60], [40, 40, 42, 42], [0, 0, 1, 1]]], np.float32)   # bemeneti képpont
        scores = np.array([[0.9, 0.2, 0.01]], np.float32)
        det = self._det("rtdetr.onnx", [("labels", labels), ("boxes", boxes), ("scores", scores)],
                        extra_int_input="orig_target_sizes", class_names=self.NAMES, conf=0.3,
                        class_conf={0: 0.1})
        self.assertEqual(det.output_format, "detr")
        self.assertEqual(len(det.extra_inputs), 1)
        d = det.detect(img())
        self.assertEqual(sorted(d.cls.tolist()), [0, 2])
        i = int(np.where(d.cls == 2)[0][0])
        np.testing.assert_allclose(d.xyxy[i], [20, 20, 60, 60], atol=1e-3)   # x*2, y*1 (stretch 128x64 <- 64x64)
        self.assertEqual(Q, 3)

    def test_backend_from_model_config(self):
        """create_backend: ONNX detektor + saját követő; a labda nyers jelöltként jön, a többit követi."""
        C, N = 4, 3
        out = np.zeros((1, 4 + C, N), np.float32)
        out[0, :4, 0] = (20, 30, 10, 20)
        out[0, 4 + 2, 0] = 0.9                 # player
        out[0, :4, 1] = (50, 40, 2, 2)
        out[0, 4 + 0, 1] = 0.07                # ball (alacsony, de a labda-küszöb fölött)
        out[0, :4, 2] = (40, 30, 10, 20)
        out[0, 4 + 1, 2] = 0.08                # kapus, a személy-küszöb (0.1) alatt -> kiesik
        path = make_const_model(os.path.join(self.tmp, "backend.onnx"), [("output0", out)],
                                metadata={"names": json.dumps({str(k): v for k, v in self.NAMES.items()})})
        cfg = {"backend": "onnx", "onnx_path": os.path.basename(path), "class_names": None,
               "role_conf": {"ball": 0.05}}
        backend, roles = create_backend(cfg, "cpu", build_role_class_map, base_dir=self.tmp)
        self.assertEqual(roles["ball"], {0})
        self.assertEqual(roles["player"], {2})
        backend.reset(frame_rate=25)
        res, raw_ball = backend.track(img())
        self.assertEqual(len(raw_ball), 1)
        self.assertAlmostEqual(raw_ball[0][1], 0.07, places=5)
        boxes = res[0].boxes
        self.assertEqual(len(boxes), 1)
        b = boxes[0]
        self.assertEqual(int(b.cls[0]), 2)
        self.assertEqual(int(b.id[0].item()), 1)
        self.assertEqual(tuple(map(int, b.xyxy[0])), (30, 8, 50, 48))
        self.assertIsNotNone(boxes.id)
        self.assertEqual(backend.count_role(img()), 1)


MAIN_IMPORT_SCRIPT = r"""
import os, sys, types
sys.path.insert(0, {root!r}); sys.path.insert(0, os.path.join({root!r}, "tests")); sys.path.insert(0, os.path.join({root!r}, "tools"))
import test_e2e_synth as e
e.install_stubs(None)
def _tilos(*a, **k):
    raise AssertionError("az ultralytics nem töltődhet be ONNX háttérnél")
m = types.ModuleType("ultralytics"); m.__getattr__ = lambda name: _tilos()
sys.modules["ultralytics"] = m
sys.modules["torch"] = None          # ImportError: az ONNX háttérhez a torch sem kell
os.chdir({tmp!r})
import main
import numpy as np
assert main.OWN_BACKEND is None, "a háttér csak az ensure_model()-ben jön létre (lusta betöltés)"
main.ensure_model()
assert main.OWN_BACKEND is not None and main.model is main.OWN_BACKEND, "nem a saját háttér töltődött be"
assert main.device == "cpu", main.device
assert main.ensure_model() is main.OWN_BACKEND
assert main.role_classes["player"] == {{2}} and main.role_classes["ball"] == {{0}}, main.role_classes
n = main._segmenter_person_counter(np.full((64, 128, 3), 90, np.uint8))
assert n == 1, n
print("RENDBEN", main.OWN_BACKEND.describe())
"""


@unittest.skipUnless(HAVE_ONNX and HAVE_CV2, "nincs onnx / onnxruntime / OpenCV")
class MainImportTests(unittest.TestCase):
    def test_main_onnx_backend_without_ultralytics_and_torch(self):
        """A main.py ONNX háttérrel betölthető úgy, hogy az ultralytics és a torch NEM importálható."""
        import shutil
        import subprocess
        if not (shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")):
            self.skipTest("nincs ffmpeg (a test_e2e_synth csonkjai kellenek)")
        tmp = tempfile.mkdtemp(prefix="main_onnx_")
        out = np.zeros((1, 8, 3), np.float32)
        out[0, :4, 0] = (20, 30, 10, 20)
        out[0, 4 + 2, 0] = 0.9
        path = make_const_model(os.path.join(tmp, "m.onnx"), [("output0", out)],
                                metadata={"names": str(OnnxDetectorTests.NAMES)})
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = dict(os.environ, TACTICAL_DETECTOR_BACKEND="onnx", TACTICAL_ONNX_PATH=path,
                   TACTICAL_MODEL_CONFIG=os.path.join(tmp, "nincs.json"))
        r = subprocess.run([sys.executable, "-c", MAIN_IMPORT_SCRIPT.format(root=root, tmp=tmp)],
                           env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-3000:])
        self.assertIn("RENDBEN", r.stdout)


class FakeListDetector(Detector):
    def __init__(self, frames, names):
        self.frames, self.names, self.i = frames, names, 0

    def detect(self, frame):
        d = self.frames[self.i]
        self.i += 1
        return d


class BackendAdapterTests(unittest.TestCase):
    def test_results_adapter_like_ultralytics(self):
        names = {0: "person", 32: "sports ball", 2: "car"}
        roles = build_role_class_map(names)
        frames = [Detections([[0, 0, 10, 30], [50, 50, 54, 54], [100, 100, 150, 130]], [0.9, 0.3, 0.9], [0, 32, 2], names)
                  for _ in range(3)]
        be = OwnTrackerBackend(FakeListDetector(frames, names), role_classes=roles)
        be.reset(frame_rate=25)
        for _ in range(3):
            be.note_motion(np.eye(3))
            res, ball = be.track(None)
        boxes = res[0].boxes
        self.assertEqual(len(boxes), 1)                    # az autó nem szerepkör, a labda nem követett
        self.assertEqual([int(b.cls[0]) for b in boxes], [0])
        self.assertEqual(ball, [((50.0, 50.0, 54.0, 54.0), 0.30000001192092896)])
        self.assertEqual(be.stats["mozgás-komp"], 3)
        be.note_motion(None)                               # megszakadt mozgás-lánc -> nincs kompenzáció
        frames.append(frames[0])
        be.track(None)
        self.assertEqual(be.stats["mozgás-komp"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
ONNX Runtime alapú detektor (onnxruntime: MIT licenc) – az ultralytics (AGPL-3.0) kiváltására.

Végrehajtók (providers) automatikus választása, ebben a sorrendben:
    CUDAExecutionProvider  (NVIDIA; az 'onnxruntime-gpu' csomag kell)
    DmlExecutionProvider   (DirectML: bármely Windows GPU, Intel / AMD integrált is; 'onnxruntime-directml')
    CPUExecutionProvider   (mindig elérhető; az alap 'onnxruntime' csomag)

Támogatott kimeneti konvenciók (a kimenetek alakjából / nevéből felismerve, felülírható):
  (a) YOLO-szerű, EGY kimenet, NMS-t mi végzünk (gyors numpy, osztály-tudatos):
        "yolo"  – [1, 4+C, N] vagy [1, N, 4+C]: cx, cy, w, h + osztálypontszámok
                  (előfeldolgozás: RGB, /255, középre igazított letterbox)
        "yolox" – [1, N, 5+C]: cx, cy, w, h, objektum-pontszám + osztálypontszámok; a hivatalos YOLOX
                  ONNX fájlok rács-dekódolatlan kimenetét (8/16/32 lépésköz) automatikusan dekódoljuk
                  (előfeldolgozás: BGR, 0..255, bal-felső letterbox – a YOLOX demo szerint)
  (b) DETR-szerű, TÖBB kimenet, NMS nélkül (RT-DETR, D-FINE, RF-DETR exportok):
        dobozok [1, Q, 4] (normalizált cx,cy,w,h vagy képpont x1,y1,x2,y2)
        + logitok [1, Q, C] (szigmoid) VAGY labels [1, Q] + scores [1, Q]
        Ha a modellnek van második bemenete (pl. RT-DETR / D-FINE 'orig_target_sizes'), azt a
        bemeneti mérettel töltjük ki, így a dobozok a bemeneti képen értendők.
        (előfeldolgozás: RGB, /255, torzító átméretezés; RF-DETR-nél "normalize": "imagenet")

Osztálynevek: a beállítás 'class_names' kulcsa (lista / {id: név} / "coco80"), különben a modell
metaadata ('names' vagy 'class_names' kulcs), végül sorszámok ("0", "1", ...).
"""
import ast
import json
import os

import numpy as np

from .detector_base import Detector, Detections, letterbox, nms

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
# A COCO eredeti (1..90, hézagos) kategória-azonosítói – a DETR-családnál gyakori (pl. RF-DETR, RT-DETR remap)
COCO91_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
              31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
              56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
              85, 86, 87, 88, 89, 90]

PROVIDER_ORDER = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")

PRESETS = {
    "yolo": {"color": "rgb", "scale": 1 / 255.0, "letterbox": "center", "normalize": None},
    "yolox": {"color": "bgr", "scale": 1.0, "letterbox": "topleft", "normalize": None},
    "detr": {"color": "rgb", "scale": 1 / 255.0, "letterbox": "stretch", "normalize": None},
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _preload_cuda_dlls(ort):
    """
    Az onnxruntime-gpu a CUDA / cuDNN DLL-eket a rendszerben keresi; ha nincsenek (külön CUDA-telepítés
    nélkül), csendben CPU-ra vált. Ha a környezetben van GPU-s PyTorch, annak a csomagolt DLL-jeit
    használjuk (a torch importja Windowson a DLL-keresési útra teszi őket), és az újabb onnxruntime
    preload_dlls() függvényét is meghívjuk. Hiba esetén nincs teendő (marad a régi viselkedés).
    """
    try:
        import torch  # noqa: F401  – csak a CUDA/cuDNN DLL-ek miatt
    except Exception:
        pass
    pre = getattr(ort, "preload_dlls", None)
    if pre is not None:
        try:
            pre()
        except Exception:
            pass


def select_providers(requested="auto", available=None):
    """A végrehajtók listája: 'auto' -> CUDA > DirectML > CPU közül az elérhetők; lista -> az elérhetők."""
    if available is None:
        import onnxruntime as ort
        available = ort.get_available_providers()
    if requested in (None, "auto"):
        wanted = list(PROVIDER_ORDER)
    elif isinstance(requested, str):
        wanted = [requested]
    else:
        wanted = list(requested)
    chosen = [p for p in wanted if p in available]
    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def parse_class_names(value):
    """class_names: lista / {id: név} / 'coco80' / JSON- vagy Python-dict szöveg -> {int: str}."""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v.lower() == "coco80":
            return dict(enumerate(COCO80))
        if v.lower() == "coco91":
            return {cid: n for cid, n in zip(COCO91_IDS, COCO80)}
        try:
            value = json.loads(v)
        except ValueError:
            try:
                value = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                value = [s.strip() for s in v.split(",") if s.strip()]
    if isinstance(value, dict):
        return {int(k): str(n) for k, n in value.items()}
    return {i: str(n) for i, n in enumerate(value)}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _cxcywh_to_xyxy(b):
    out = np.empty_like(b)
    out[..., 0] = b[..., 0] - b[..., 2] / 2
    out[..., 1] = b[..., 1] - b[..., 3] / 2
    out[..., 2] = b[..., 0] + b[..., 2] / 2
    out[..., 3] = b[..., 1] + b[..., 3] / 2
    return out


def yolox_grid_decode(pred, input_w, input_h, strides=(8, 16, 32)):
    """A YOLOX rács-dekódolatlan kimenete (x,y = rácseltolás, w,h = log-skála) -> bemeneti képpont cx,cy,w,h."""
    grids, exp_strides = [], []
    for s in strides:
        gw, gh = input_w // s, input_h // s
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        exp_strides.append(np.full((gw * gh, 1), s))
    grid = np.concatenate(grids, 0).astype(np.float32)
    st = np.concatenate(exp_strides, 0).astype(np.float32)
    if len(grid) != pred.shape[0]:
        return None
    out = pred.copy()
    out[:, :2] = (pred[:, :2] + grid) * st
    out[:, 2:4] = np.exp(np.clip(pred[:, 2:4], -20, 20)) * st
    return out


class OnnxDetector(Detector):
    """
    cfg kulcsok (mind elhagyható, kivéve onnx_path):
      onnx_path      – az .onnx fájl
      input_size     – [szélesség, magasság] vagy egy szám; alap: a modellből, különben 640
      preset         – "auto" | "yolo" | "yolox" | "detr" (előfeldolgozás + kimenet-értelmezés)
      output_format  – "auto" | "yolo" | "yolox" | "detr" (csak a kimenet értelmezése)
      class_names    – lásd parse_class_names
      conf           – általános minimális pontszám (alap 0.25)
      class_conf     – {osztály id: küszöb} (pl. a labdának alacsonyabb); a minimum a szűrő-küszöb
      iou            – NMS IoU küszöb (alap 0.5); agnostic_nms – osztályfüggetlen NMS (alap false)
      max_det        – legfeljebb ennyi doboz képkockánként (alap 300)
      normalize      – null | "imagenet" (DETR-családnál, pl. RF-DETR)
      color / scale / letterbox – az előfeldolgozás kézi felülírása
      box_format     – DETR: "auto" | "cxcywh_norm" | "xyxy_norm" | "xyxy" | "cxcywh"
      providers      – "auto" vagy lista; threads – CPU szálak száma (0 = automatikus)
    session: tesztekhez kész onnxruntime.InferenceSession is átadható.
    """

    def __init__(self, cfg, session=None):
        cfg = dict(cfg or {})
        self.cfg = cfg
        if session is None:
            import onnxruntime as ort
            path = cfg.get("onnx_path")
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"Az ONNX modellfájl nem található: {path!r} (lásd docs/MODELL_BEALLITAS.md)")
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if int(cfg.get("threads") or 0) > 0:
                so.intra_op_num_threads = int(cfg["threads"])
            providers = select_providers(cfg.get("providers", "auto"))
            if "CUDAExecutionProvider" in providers:
                _preload_cuda_dlls(ort)
            session = ort.InferenceSession(path, sess_options=so, providers=providers)
        self.session = session
        self.providers = list(session.get_providers()) if hasattr(session, "get_providers") else []
        inputs = session.get_inputs()
        self.input_name = inputs[0].name
        self.input_type = inputs[0].type
        self.extra_inputs = inputs[1:]
        ishape = list(inputs[0].shape)
        self.fixed_batch = isinstance(ishape[0], int) and ishape[0] > 0
        self.input_w, self.input_h = self._input_size(cfg.get("input_size"), ishape)
        self.outputs = session.get_outputs()
        self.output_names = [o.name for o in self.outputs]

        meta = {}
        try:
            meta = dict(session.get_modelmeta().custom_metadata_map or {})
        except Exception:
            pass
        names = parse_class_names(cfg.get("class_names"))
        if names is None:
            names = parse_class_names(meta.get("names") or meta.get("class_names"))
        self._names_from_model = names is not None

        fmt = (cfg.get("output_format") or cfg.get("preset") or "auto").lower()
        if fmt == "auto":
            fmt = self._guess_format(names, meta)
        self.output_format = fmt
        preset = (cfg.get("preset") or "auto").lower()
        pre = dict(PRESETS[fmt if preset == "auto" else preset])
        for k in ("color", "scale", "letterbox", "normalize"):
            if cfg.get(k) is not None:
                pre[k] = cfg[k]
        self.pre = pre
        if names is None:
            names = self._default_names()
        self.names = names
        self.conf = float(cfg.get("conf", 0.25))
        self.class_conf = {int(k): float(v) for k, v in (cfg.get("class_conf") or {}).items()}
        self.min_conf = min([self.conf] + list(self.class_conf.values()))
        self.iou = float(cfg.get("iou", 0.5))
        self.agnostic = bool(cfg.get("agnostic_nms", False))
        self.max_det = int(cfg.get("max_det", 300))
        self.box_format = (cfg.get("box_format") or "auto").lower()
        self._yolox_decode = cfg.get("yolox_decode", "auto")

    # ------------------------------------------------------------------ felismerés
    @staticmethod
    def _input_size(cfg_size, ishape):
        if cfg_size:
            if isinstance(cfg_size, (int, float)):
                return int(cfg_size), int(cfg_size)
            return int(cfg_size[0]), int(cfg_size[1])
        h = ishape[2] if len(ishape) > 2 and isinstance(ishape[2], int) and ishape[2] > 0 else 640
        w = ishape[3] if len(ishape) > 3 and isinstance(ishape[3], int) and ishape[3] > 0 else 640
        return int(w), int(h)

    def _guess_format(self, names, meta):
        float_outs = [o for o in self.outputs if "float" in o.type]
        if len(self.outputs) >= 2:
            return "detr"
        shp = list(float_outs[0].shape) if float_outs else []
        if len(shp) == 3 and all(isinstance(s, int) for s in shp[1:]):
            a, b = shp[1], shp[2]
            if names:
                nc = len(names)
                if a - 4 == nc or b - 4 == nc:
                    return "yolo"
                if a - 5 == nc or b - 5 == nc:
                    return "yolox"
            if "stride" in meta or "imgsz" in meta:
                return "yolo"
            return "yolo" if a < b else "yolox"
        return "yolo"

    def _default_names(self):
        """Ha sem a beállítás, sem a metaadat nem ad neveket: COCO-méretű kimenetnél COCO, különben sorszám."""
        n = None
        for o in self.outputs:
            shp = list(o.shape)
            if len(shp) == 3 and all(isinstance(s, int) for s in shp[1:]):
                if self.output_format == "yolo":
                    n = min(shp[1:]) - 4
                elif self.output_format == "yolox":
                    n = shp[2] - 5
                elif shp[2] != 4:
                    n = shp[2]
        if n == 80:
            return dict(enumerate(COCO80))
        return {i: str(i) for i in range(max(0, n or 0))}

    # ------------------------------------------------------------------ előfeldolgozás
    def preprocess(self, frame_bgr):
        img, info = letterbox(frame_bgr, (self.input_w, self.input_h), mode=self.pre["letterbox"])
        if self.pre["color"] == "rgb":
            img = img[:, :, ::-1]
        x = img.astype(np.float32) * float(self.pre["scale"])
        if self.pre.get("normalize") == "imagenet":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        return x, info

    def _feed(self, batch):
        if "float16" in self.input_type:
            batch = batch.astype(np.float16)
        feed = {self.input_name: batch}
        for inp in self.extra_inputs:   # pl. RT-DETR / D-FINE: orig_target_sizes [B, 2] = (w, h)
            dtype = np.int64 if "int64" in inp.type else (np.int32 if "int32" in inp.type else np.float32)
            feed[inp.name] = np.tile(np.array([[self.input_w, self.input_h]], dtype), (len(batch), 1))
        return feed

    # ------------------------------------------------------------------ futtatás
    def detect(self, frame_bgr):
        return self.detect_batch([frame_bgr])[0]

    def detect_batch(self, frames_bgr):
        if not frames_bgr:
            return []
        prepped = [self.preprocess(f) for f in frames_bgr]
        if self.fixed_batch and len(prepped) > 1:
            outs = [self.session.run(None, self._feed(p[0][None]))  for p in prepped]
            per_image = [[o[0] for o in out] for out in outs]
        else:
            out = self.session.run(None, self._feed(np.stack([p[0] for p in prepped])))
            per_image = [[o[i] for o in out] for i in range(len(prepped))]
        return [self.postprocess(per_image[i], prepped[i][1]) for i in range(len(prepped))]

    # ------------------------------------------------------------------ utófeldolgozás
    def _finish(self, xyxy, scores, cls, info, do_nms):
        keep = scores >= self.min_conf
        xyxy, scores, cls = xyxy[keep], scores[keep], cls[keep]
        if self.class_conf:
            thr = np.array([self.class_conf.get(int(c), self.conf) for c in cls], np.float32)
            keep = scores >= thr
        else:
            keep = scores >= self.conf
        xyxy, scores, cls = xyxy[keep], scores[keep], cls[keep]
        if do_nms and len(xyxy):
            idx = nms(xyxy, scores, self.iou, None if self.agnostic else cls, self.max_det)
            xyxy, scores, cls = xyxy[idx], scores[idx], cls[idx]
        elif len(scores) > self.max_det:
            idx = np.argsort(-scores)[:self.max_det]
            xyxy, scores, cls = xyxy[idx], scores[idx], cls[idx]
        xyxy = info.to_original(xyxy)
        ok = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        return Detections(xyxy[ok], scores[ok], cls[ok], self.names)

    def postprocess(self, outs, info):
        if self.output_format == "detr":
            return self._post_detr(outs, info)
        return self._post_yolo(outs, info)

    def _post_yolo(self, outs, info):
        pred = next(o for o in outs if np.issubdtype(np.asarray(o).dtype, np.floating))
        pred = np.asarray(pred, np.float32)
        if pred.ndim == 3:
            pred = pred[0]
        # [K, N] -> [N, K], ahol K = 4+C (yolo) vagy 5+C (yolox); ismert osztályszámnál az alapján
        k = (4 if self.output_format == "yolo" else 5) + len(self.names)
        if self._names_from_model or self.names:
            if pred.shape[0] == k and pred.shape[1] != k:
                pred = pred.T
        elif pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if self.output_format == "yolox":
            if self._yolox_decode in (True, "grid") or (self._yolox_decode == "auto" and self._looks_raw_grid(pred)):
                dec = yolox_grid_decode(pred, self.input_w, self.input_h)
                if dec is not None:
                    pred = dec
            cls_scores = pred[:, 5:] * pred[:, 4:5]
        else:
            cls_scores = pred[:, 4:]
        if cls_scores.shape[1] == 0:
            return Detections(names=self.names)
        cls = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(cls)), cls]
        m = scores >= self.min_conf
        boxes = _cxcywh_to_xyxy(pred[m, :4])
        return self._finish(boxes, scores[m], cls[m].astype(np.int64), info, do_nms=True)

    def _looks_raw_grid(self, pred):
        n = pred.shape[0]
        expect = sum((self.input_w // s) * (self.input_h // s) for s in (8, 16, 32))
        if n != expect:
            return False
        # dekódolt kimenetnél a cx a bemeneti szélesség nagyságrendjében van, nyersnél a rácseltolás kicsi
        return float(np.percentile(np.abs(pred[:, 0]), 99)) < 0.1 * self.input_w

    def _post_detr(self, outs, info):
        """outs: EGY kép kimenetei (a batch-dimenzió már levágva): dobozok (Q,4) + logitok (Q,C) vagy labels/scores (Q,)."""
        named = list(zip(self.output_names, [np.asarray(o) for o in outs]))
        boxes = scores = labels = logits = None
        box_name = None
        for name, o in named:                    # 1. név szerint
            lname = name.lower()
            if o.ndim == 2 and o.shape[-1] == 4 and boxes is None and any(t in lname for t in ("box", "det", "bbox")):
                boxes, box_name = o, name
        if boxes is None:                        # 2. név nélkül: az első (Q, 4) alakú lebegőpontos kimenet
            for name, o in named:
                if o.ndim == 2 and o.shape[-1] == 4 and np.issubdtype(o.dtype, np.floating):
                    boxes, box_name = o, name
                    break
        if boxes is None:
            raise ValueError(f"DETR-kimenet: nem található doboz-kimenet ({self.output_names})")
        for name, o in named:
            if name == box_name:
                continue
            lname = name.lower()
            if np.issubdtype(o.dtype, np.integer) and o.ndim == 1:
                labels = o
            elif o.ndim == 1 and np.issubdtype(o.dtype, np.floating):
                if "label" in lname or "class" in lname:
                    labels = o
                else:
                    scores = o
            elif o.ndim == 2 and np.issubdtype(o.dtype, np.floating) and logits is None:
                logits = o
        b = np.asarray(boxes, np.float32)
        if scores is not None and labels is not None:
            sc = np.asarray(scores, np.float32).reshape(-1)
            cl = np.asarray(labels).astype(np.int64).reshape(-1)
        elif logits is not None:
            lg = np.asarray(logits, np.float32)
            if lg.min() < 0 or lg.max() > 1:
                lg = _sigmoid(lg)
            cl = lg.argmax(1).astype(np.int64)
            sc = lg[np.arange(len(cl)), cl]
        else:
            raise ValueError(f"DETR-kimenet: nincs pontszám / logit kimenet ({self.output_names})")
        fmt = self.box_format
        if fmt == "auto":
            fmt = "cxcywh_norm" if float(np.abs(b).max(initial=0.0)) <= 1.5 else "xyxy"
        if fmt in ("cxcywh_norm", "cxcywh"):
            b = _cxcywh_to_xyxy(b)
        if fmt.endswith("_norm"):
            b = b * np.array([self.input_w, self.input_h, self.input_w, self.input_h], np.float32)
        return self._finish(b, sc, cl, info, do_nms=bool(self.cfg.get("detr_nms", False)))

    def describe(self):
        return (f"ONNX ({os.path.basename(str(self.cfg.get('onnx_path', '?')))}, {self.output_format}, "
                f"{self.input_w}x{self.input_h}, végrehajtó: {', '.join(self.providers) or '?'})")


def main(argv=None):
    """
    Modell-ellenőrzés:  python3 -m detection.onnx_detector modell.onnx [kép.jpg] [--cfg '{"preset": "detr"}']
    Kiírja a bemeneteket / kimeneteket / metaadatot, a felismert konvenciót és (képpel) a detektálásokat.
    """
    import argparse
    import time
    ap = argparse.ArgumentParser(description="ONNX detektor-modell ellenőrzése")
    ap.add_argument("onnx_path")
    ap.add_argument("image", nargs="?")
    ap.add_argument("--cfg", default="{}", help="további OnnxDetector beállítások JSON-ban")
    a = ap.parse_args(argv)
    cfg = json.loads(a.cfg)
    cfg["onnx_path"] = a.onnx_path
    det = OnnxDetector(cfg)
    print(f"Végrehajtók: {det.providers}")
    for i in det.session.get_inputs():
        print(f"  bemenet: {i.name} {i.shape} {i.type}")
    for o in det.outputs:
        print(f"  kimenet: {o.name} {o.shape} {o.type}")
    try:
        print(f"  metaadat: {dict(det.session.get_modelmeta().custom_metadata_map)}")
    except Exception:
        pass
    print(f"Konvenció: {det.output_format}, előfeldolgozás: {det.pre}, bemenet: {det.input_w}x{det.input_h}")
    print(f"Osztályok ({len(det.names)}): {dict(list(det.names.items())[:12])}{' ...' if len(det.names) > 12 else ''}")
    if a.image:
        import cv2
        img = cv2.imread(a.image)
        if img is None:
            raise SystemExit(f"A kép nem olvasható: {a.image}")
        det.detect(img)
        t0 = time.time()
        d = det.detect(img)
        print(f"{len(d)} detektálás, {1000 * (time.time() - t0):.1f} ms")
        for k in np.argsort(-d.conf)[:30]:
            print(f"  {det.names.get(int(d.cls[k]), d.cls[k])!s:14s} {d.conf[k]:.2f}  {[round(float(v), 1) for v in d.xyxy[k]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

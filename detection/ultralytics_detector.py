"""
FEJLESZTŐI háttér: ultralytics YOLO predict() (követés NÉLKÜL) a Detector interfész mögött.

FIGYELEM – licenc: az ultralytics AGPL-3.0. Zárt forrású, eladott termékbe NEM kerülhet bele
(lásd docs/LICENC_ES_ARUSITAS.md). Ezt a modult csak akkor importáljuk, ha a beállításban
kifejezetten ezt választják; a csomag többi része nem hivatkozik rá.

A saját követővel (detection/tracker.py) együtt is használható ("ultralytics_predict" háttér),
így a saját követő valódi YOLO-detektálásokon is összevethető a beépített ByteTrack-kel.
"""
import numpy as np

from .detector_base import Detector, Detections


def _np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


class UltralyticsDetector(Detector):
    def __init__(self, weights=None, device="cpu", conf=0.05, iou=0.5, model=None, imgsz=None):
        if model is None:
            from ultralytics import YOLO  # noqa: WPS433 – szándékosan lusta import (AGPL, csak fejlesztéshez)
            model = YOLO(weights)
        self.model = model
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = imgsz
        names = getattr(model, "names", {}) or {}
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))

    def detect_batch(self, frames_bgr):
        kw = {"conf": self.conf, "iou": self.iou, "device": self.device, "verbose": False}
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        results = self.model.predict(list(frames_bgr) if len(frames_bgr) > 1 else frames_bgr[0], **kw)
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(Detections(names=self.names))
                continue
            out.append(Detections(_np(b.xyxy)[:, :4], _np(b.conf), _np(b.cls), self.names))
        return out

    def describe(self):
        return f"ultralytics predict ({getattr(self.model, 'ckpt_path', None) or '?'}) – csak fejlesztéshez (AGPL-3.0)"

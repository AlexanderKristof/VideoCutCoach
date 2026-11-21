"""
Detektor + saját követő összekötése a main.py elemző ciklusához.

A main.py felismerő blokkja az ultralytics eredményformátumát olvassa (results[i].boxes, és dobozonként
.cls[0] / .conf[0] / .xyxy[0] / .id[0]). A TrackResult ugyanezt adja numpy tömbökből, így a blokk
logikája változatlan marad. A labda NEM megy a követőbe: minden nyers labda-jelölt külön listában
jön vissza (a BallTracker-nek) – ennél a háttérnél nincs szükség az ultralytics callback-trükkre.

    backend = OwnTrackerBackend(detector, ByteTracker(), role_classes)
    backend.reset(frame_rate=25)
    backend.note_motion(F)                   # képkockánként: előző kép -> aktuális kép (vagy None)
    results, raw_ball = backend.track(frame) # results: [TrackResult], raw_ball: [((x1,y1,x2,y2), conf)]
"""
import os

import numpy as np

from .detector_base import Detections
from .tracker import ByteTracker

TRACKED_ROLES = ("player", "goalkeeper", "referee")

# Szerepkörönkénti minimális pontszám a detektorban (a labda alacsonyabb: a BallTracker pontoz,
# a személyeknél a követő gyenge-párosítási lépcsője 0.1-től használja a dobozokat)
DEFAULT_ROLE_CONF = {"player": 0.1, "goalkeeper": 0.1, "referee": 0.1, "ball": 0.05}
DEFAULT_TRACKER = {"high_thresh": 0.25, "low_thresh": 0.1, "new_track_thresh": 0.25, "match_thresh": 0.8,
                   "track_buffer": 30, "fuse_score": True, "class_aware": False, "output": "kalman"}


class TrackBox:
    """Egy követett doboz ultralytics-szerű mezőkkel (1 elemű numpy tömbök)."""
    __slots__ = ("cls", "conf", "xyxy", "id")

    def __init__(self, xyxy, conf, cls, tid):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(1, 4)
        self.conf = np.asarray([conf], np.float32)
        self.cls = np.asarray([cls], np.float32)
        self.id = None if tid is None else np.asarray([tid], np.float32)


class TrackBoxes:
    def __init__(self, xyxy, conf, cls, ids):
        n = len(xyxy)
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.asarray(cls, np.float32).reshape(-1)
        self.id = None if ids is None else np.asarray(ids, np.float32).reshape(-1)
        self._boxes = [TrackBox(self.xyxy[i], self.conf[i], self.cls[i],
                                None if self.id is None else self.id[i]) for i in range(n)]

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, i):
        return self._boxes[i]


class TrackResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class OwnTrackerBackend:
    """Detector (bármelyik) + saját ByteTracker. A labdát nem követi, nyersen adja vissza."""

    kind = "own_tracker"

    def __init__(self, detector, tracker=None, role_classes=None, tracker_cfg=None):
        self.detector = detector
        self.tracker_cfg = dict(DEFAULT_TRACKER)
        self.tracker_cfg.update(tracker_cfg or {})
        self.tracker = tracker or ByteTracker(**self.tracker_cfg)
        self.names = dict(getattr(detector, "names", {}) or {})
        self.role_classes = role_classes or {r: set() for r in TRACKED_ROLES + ("ball",)}
        self._motion = None
        self._motion_ok = True
        self.last_detections = Detections(names=self.names)
        self.stats = {"képkocka": 0, "detektálás": 0, "labda-jelölt": 0, "követett": 0, "mozgás-komp": 0}

    def set_role_classes(self, role_classes):
        self.role_classes = role_classes

    def reset(self, frame_rate=None):
        """Új klip előtt: a követő állapota és az azonosító-számláló nullázódik."""
        if frame_rate:
            cfg = dict(self.tracker_cfg)
            cfg["frame_rate"] = float(frame_rate)
            self.tracker = ByteTracker(**cfg)
        else:
            self.tracker.reset()
        self._motion, self._motion_ok = None, True
        for k in self.stats:
            self.stats[k] = 0
        return 1

    def note_motion(self, F):
        """Képkockánként hívandó (a detektálás nélküli képkockákon is): a mozgások összeszorzódnak."""
        if F is None:
            self._motion_ok = False
            return
        F = np.asarray(F, np.float64).reshape(3, 3)
        self._motion = F if self._motion is None else F @ self._motion

    def _take_motion(self):
        m = self._motion if self._motion_ok else None
        self._motion, self._motion_ok = None, True
        return m

    def track(self, frame, motion=None):
        det = self.detector.detect(frame)
        self.last_detections = det
        m = motion if motion is not None else self._take_motion()
        if motion is not None:
            self._motion, self._motion_ok = None, True
        ball_ids = sorted(self.role_classes.get("ball") or ())
        tracked_ids = sorted(set().union(*[self.role_classes.get(r) or set() for r in TRACKED_ROLES]))
        is_ball = np.isin(det.cls, ball_ids) if ball_ids else np.zeros(len(det), bool)
        is_person = np.isin(det.cls, tracked_ids) if tracked_ids else np.zeros(len(det), bool)
        raw_ball = [(tuple(float(v) for v in det.xyxy[i]), float(det.conf[i])) for i in np.where(is_ball)[0]]
        pd = det.subset(is_person)
        out = self.tracker.update(pd.xyxy, pd.conf, pd.cls, motion=m)
        self.stats["képkocka"] += 1
        self.stats["detektálás"] += int(len(det))
        self.stats["labda-jelölt"] += len(raw_ball)
        self.stats["követett"] += int(len(out["id"]))
        self.stats["mozgás-komp"] += int(m is not None)
        boxes = TrackBoxes(out["xyxy"], out["conf"], out["cls"], out["id"])
        return [TrackResult(boxes, self.names)], raw_ball

    def count_role(self, frame, roles=("player", "goalkeeper"), conf=0.3):
        """Létszám egy képkockán (követés nélkül) – pl. a félidő-felismeréshez."""
        det = self.detector.detect(frame)
        ids = sorted(set().union(*[self.role_classes.get(r) or set() for r in roles]))
        return int(np.sum(np.isin(det.cls, ids) & (det.conf >= conf)))

    def describe(self):
        d = self.detector.describe() if hasattr(self.detector, "describe") else type(self.detector).__name__
        return f"{d} + saját ByteTrack-szerű követő"


def _role_class_conf(role_classes, role_conf):
    out = {}
    for role, ids in role_classes.items():
        thr = role_conf.get(role)
        if thr is None:
            continue
        for c in ids:
            out[int(c)] = float(thr)
    return out


def create_backend(model_cfg, device="cpu", build_role_map=None, base_dir=None):
    """
    A model_config szerinti saját háttér. backend:
      "onnx"                – OnnxDetector + saját követő (a kereskedelmi út: MIT/Apache licencű komponensek)
      "ultralytics_predict" – ultralytics predict (követés nélkül) + saját követő (CSAK fejlesztéshez, AGPL!)
    Visszatér: (OwnTrackerBackend, role_classes). Az "ultralytics" (model.track) hátteret a main.py kezeli.
    """
    if build_role_map is None:
        from model_config import build_role_class_map as build_role_map
    kind = str(model_cfg.get("backend") or "ultralytics").lower()
    role_conf = dict(DEFAULT_ROLE_CONF)
    role_conf.update(model_cfg.get("role_conf") or {})
    if kind == "onnx":
        from .onnx_detector import OnnxDetector
        path = model_cfg.get("onnx_path")
        if path and not os.path.isabs(path) and base_dir:
            cand = os.path.join(base_dir, path)
            if os.path.exists(cand) or not os.path.exists(path):
                path = cand
        cfg = dict(model_cfg.get("onnx") or {})
        cfg["onnx_path"] = path
        for k in ("input_size", "providers"):
            if model_cfg.get(k) is not None:
                cfg[k] = model_cfg[k]
        if model_cfg.get("onnx_class_names") is not None:
            cfg["class_names"] = model_cfg["onnx_class_names"]
        cfg.setdefault("conf", min(role_conf.values()))
        detector = OnnxDetector(cfg)
    elif kind == "ultralytics_predict":
        from .ultralytics_detector import UltralyticsDetector
        from model_config import resolve_weights
        detector = UltralyticsDetector(resolve_weights(model_cfg, device), device=device,
                                       conf=min(role_conf.values()))
    else:
        raise ValueError(f"Ismeretlen saját háttér: {kind!r} (lehet: onnx, ultralytics_predict)")
    role_classes = build_role_map(detector.names, model_cfg.get("class_names"))
    if hasattr(detector, "class_conf"):
        detector.class_conf = _role_class_conf(role_classes, role_conf)
        # a szerepkörhöz nem tartozó osztályokat (pl. COCO autó) a küszöb fölé emeljük -> kiesnek
        if detector.class_conf:
            detector.conf = 1.01
        detector.min_conf = min([detector.conf] + list(detector.class_conf.values()))
    backend = OwnTrackerBackend(detector, role_classes=role_classes, tracker_cfg=model_cfg.get("tracker"))
    return backend, role_classes

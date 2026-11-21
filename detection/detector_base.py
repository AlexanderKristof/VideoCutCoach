"""
A detektor-interfész és a közös segédfüggvények (csak numpy – torch / ultralytics nélkül).

    det = detector.detect(frame_bgr)          # -> Detections
    dets = detector.detect_batch([f1, f2])    # -> [Detections, ...]

Detections mezői (mind numpy):
    xyxy  float32 (N, 4) – képpont-koordináták az EREDETI képkockán
    conf  float32 (N,)
    cls   int64   (N,)
    names dict {osztály id: név} – a szerepkörökhöz a model_config.build_role_class_map() kell
"""
import numpy as np


class Detections:
    __slots__ = ("xyxy", "conf", "cls", "names")

    def __init__(self, xyxy=None, conf=None, cls=None, names=None):
        self.xyxy = np.zeros((0, 4), np.float32) if xyxy is None else np.asarray(xyxy, np.float32).reshape(-1, 4)
        n = len(self.xyxy)
        self.conf = np.zeros(n, np.float32) if conf is None else np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.zeros(n, np.int64) if cls is None else np.asarray(cls).astype(np.int64).reshape(-1)
        if len(self.conf) != n or len(self.cls) != n:
            raise ValueError(f"Detections: eltérő hosszak (xyxy={n}, conf={len(self.conf)}, cls={len(self.cls)})")
        self.names = dict(names or {})

    def __len__(self):
        return len(self.xyxy)

    def subset(self, mask_or_idx):
        """Részhalmaz (logikai maszk vagy indexlista) – a names megmarad."""
        return Detections(self.xyxy[mask_or_idx], self.conf[mask_or_idx], self.cls[mask_or_idx], self.names)

    def of_classes(self, class_ids):
        ids = np.fromiter((int(c) for c in class_ids), np.int64) if class_ids else np.zeros(0, np.int64)
        return self.subset(np.isin(self.cls, ids))

    def __repr__(self):
        return f"Detections(n={len(self)})"


class Detector:
    """
    Interfész. Az alosztály a detect()-et (vagy a detect_batch()-et) valósítja meg.
    names: {osztály id: név}. A szerepkör-szűrés (játékos / kapus / bíró / labda) NEM itt történik.
    """
    names = {}

    def detect(self, frame_bgr):
        return self.detect_batch([frame_bgr])[0]

    def detect_batch(self, frames_bgr):
        return [self.detect(f) for f in frames_bgr]

    def describe(self):
        return type(self).__name__


# ---------------------------------------------------------------------- doboz-segédek
def box_iou(a, b):
    """Páronkénti IoU: a (N,4), b (M,4) xyxy -> (N, M)."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def nms(xyxy, scores, iou_thr=0.5, cls=None, max_det=None):
    """
    Gyors numpy NMS. cls megadásával osztályonként (osztály-tudatos): a dobozokat osztályonként
    nagy eltolással szétválasztjuk, így egyetlen menetben fut. Visszatér: a megtartott indexek
    (pontszám szerint csökkenő sorrendben).
    """
    xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if len(xyxy) == 0:
        return np.zeros(0, np.int64)
    boxes = xyxy
    if cls is not None:
        span = float(max(boxes[:, 2].max(), boxes[:, 3].max()) - min(boxes[:, 0].min(), boxes[:, 1].min())) + 1.0
        boxes = boxes + (np.asarray(cls, np.float32).reshape(-1, 1) * (span + 1.0))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if max_det is not None and len(keep) >= max_det:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, np.int64)


# ---------------------------------------------------------------------- letterbox
class LetterboxInfo:
    """A modell-bemenet és az eredeti kép közti leképezés (arány + eltolás tengelyenként)."""
    __slots__ = ("sx", "sy", "pad_x", "pad_y", "orig_w", "orig_h")

    def __init__(self, sx, sy, pad_x, pad_y, orig_w, orig_h):
        self.sx, self.sy, self.pad_x, self.pad_y = float(sx), float(sy), float(pad_x), float(pad_y)
        self.orig_w, self.orig_h = int(orig_w), int(orig_h)

    def to_original(self, xyxy):
        """Modell-bemeneti koordináták -> eredeti kép (a képre vágva)."""
        b = np.asarray(xyxy, np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / self.sx
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / self.sy
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, self.orig_w)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, self.orig_h)
        return b

    def to_input(self, xyxy):
        """Eredeti kép -> modell-bemeneti koordináták (tesztekhez)."""
        b = np.asarray(xyxy, np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * self.sx + self.pad_x
        b[:, [1, 3]] = b[:, [1, 3]] * self.sy + self.pad_y
        return b


def letterbox(img, size, mode="center", pad_value=114):
    """
    Átméretezés a modell bemenetére. size = (szélesség, magasság).
    mode: "center" – arányos, középre igazított kitöltés (YOLO / ultralytics);
          "topleft" – arányos, bal-felső sarokba igazított kitöltés (YOLOX);
          "stretch" – torzító átméretezés kitöltés nélkül (a legtöbb DETR export).
    Visszatér: (kép uint8 HxWx3, LetterboxInfo).
    """
    import cv2  # csak itt kell; a modul többi része cv2 nélkül is tesztelhető
    tw, th = int(size[0]), int(size[1])
    h, w = img.shape[:2]
    if mode == "stretch":
        out = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        return out, LetterboxInfo(tw / w, th / h, 0, 0, w, h)
    r = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    out = np.full((th, tw, 3), pad_value, np.uint8)
    if mode == "topleft":
        px, py = 0, 0
    else:
        px, py = (tw - nw) // 2, (th - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, LetterboxInfo(nw / w, nh / h, px, py, w, h)

import cv2
import numpy as np
from collections import deque

FULL_HSV_RANGE = (np.array([0, 0, 0]), np.array([179, 255, 255]))


def is_full_range(lower, upper):
    """A teljes HSV tartomány = 'nincs beállítva' (így jelzi a kliens)."""
    return np.array_equal(lower, FULL_HSV_RANGE[0]) and np.array_equal(upper, FULL_HSV_RANGE[1])


def build_grass_mask(hsv_frame, lower, upper, scale=0.5):
    """
    Fűmaszk a képkockáról, csökkentett felbontáson (gyorsabb).
    A zárás (closing) kitölti a fehér pályavonalakat és az árnyékokat,
    hogy a vonalon álló játékos se essen ki.
    Visszatér: (maszk, scale) vagy (None, scale), ha nincs fűszín beállítva.
    """
    if is_full_range(lower, upper):
        return None, scale
    small = cv2.resize(hsv_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    mask = cv2.inRange(small, lower, upper)
    k = max(5, (int(small.shape[0] / 60) // 2) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, scale


def foot_on_grass(grass_mask, scale, x1, x2, y2, min_ratio):
    """
    Igaz, ha a láb körüli sávban elég sok a fű.
    A sáv a doboz szélességű, a talppont fölött és alatt kis magasságú csík:
    a lelátón / reklámtábla mögött álló nézőnél itt nincs fű.
    """
    if grass_mask is None or min_ratio <= 0:
        return True, 1.0
    mh, mw = grass_mask.shape[:2]
    box_w = max(1, x2 - x1)
    pad_y = max(2, int(box_w * 0.35))
    gx1 = max(0, int(x1 * scale))
    gx2 = min(mw, int(x2 * scale) + 1)
    gy1 = max(0, int((y2 - pad_y) * scale))
    gy2 = min(mh, int((y2 + pad_y) * scale) + 1)
    patch = grass_mask[gy1:gy2, gx1:gx2]
    if patch.size == 0:
        return True, 1.0
    ratio = cv2.countNonZero(patch) / float(patch.size)
    return ratio >= min_ratio, ratio


class CameraMotion:
    """
    A forgó (Veo) kamera elmozdulásának becslése fáziskorrelációval.
    A képernyő-koordinátából kivonva az összesített eltolást egy
    'kamera-független' koordinátát kapunk, amiben az álló néző tényleg áll.
    (Csak eltolást kezel, zoomot nem – néhány másodperces ablakra ez elég.)
    """

    def __init__(self, scale=0.25):
        self.scale = scale
        self.prev = None
        self.window = None
        self.offset = np.zeros(2, dtype=np.float64)

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        small = np.float32(small)
        if self.window is None or self.window.shape != small.shape:
            self.window = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
        if self.prev is not None:
            (dx, dy), response = cv2.phaseCorrelate(self.prev, small, self.window)
            if response > 0.05:
                self.offset += np.array([dx, dy]) / self.scale
        self.prev = small
        return self.offset

    def stabilize(self, x, y):
        return x - self.offset[0], y - self.offset[1]


class StationaryFilter:
    """
    Kiszűri azokat a követett alakokat, akik 'window_sec' ideig szinte
    nem mozdulnak (nézők, edző, kispad). A mozgást kamera-független
    koordinátában méri, a küszöb a dobozmagassághoz viszonyított.
    window_sec <= 0 esetén kikapcsolva.
    """

    def __init__(self, fps, window_sec=4.0, max_move_ratio=0.5):
        self.enabled = window_sec > 0
        self.window_frames = max(1, int(window_sec * fps))
        self.max_move_ratio = max_move_ratio
        self.history = {}

    def update(self, tid, frame_idx, sx, sy, box_h):
        """Visszatér: True, ha az alak állónak számít."""
        if not self.enabled or tid is None:
            return False
        h = self.history.get(tid)
        if h is None:
            h = deque()
            self.history[tid] = h
        h.append((frame_idx, sx, sy))
        while h and frame_idx - h[0][0] > self.window_frames:
            h.popleft()
        if frame_idx - h[0][0] < self.window_frames * 0.9:
            return False  # még nincs elég előzmény
        xs = [p[1] for p in h]
        ys = [p[2] for p in h]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        return span < self.max_move_ratio * box_h

    def prune(self, frame_idx):
        """Rég nem látott azonosítók törlése (memória)."""
        limit = frame_idx - self.window_frames * 2
        for tid in [t for t, h in self.history.items() if not h or h[-1][0] < limit]:
            del self.history[tid]


def extract_dominant_color(image_crop, grass_lower=None, grass_upper=None):
    """
    Mezszín a felsőtestből – gyors változat (KMeans helyett medián).
    A fűszínű pixeleket kihagyja, hogy a háttér ne torzítsa.
    Visszatér: [R, G, B]
    """
    if image_crop.size == 0:
        return [0, 0, 0]
    h, w = image_crop.shape[:2]
    torso = image_crop[int(h * 0.2):int(h * 0.6), int(w * 0.2):int(w * 0.8)]
    if torso.size == 0:
        return [0, 0, 0]

    pixels = torso.reshape(-1, 3)
    if grass_lower is not None and grass_upper is not None and not is_full_range(grass_lower, grass_upper):
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        not_grass = cv2.inRange(hsv, grass_lower, grass_upper).reshape(-1) == 0
        if np.count_nonzero(not_grass) >= 5:
            pixels = pixels[not_grass]
    if len(pixels) < 5:
        return [0, 0, 0]

    b, g, r = np.median(pixels, axis=0)
    return [int(r), int(g), int(b)]

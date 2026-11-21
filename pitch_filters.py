import math

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


# ======================================================================
# Pályahatár, követés-épség és kapus-megerősítés pálya-koordinátában
# (a képkockánkénti homográfiával; ha a H nem megbízható / None, mindegyik kikapcsol)
# ======================================================================
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PENALTY_DEPTH = 16.5          # a 16-os mélysége (m)
PENALTY_HALF_WIDTH = 20.16    # a 16-os félszélessége (40,32 m / 2)
PITCH_TRUST_CONF = 0.5        # a pályakövető bizalma, ami felett a H-t a szűrők használják


def pitch_h_trusted(H, pitch_q=None, min_conf=PITCH_TRUST_CONF):
    """
    Megbízható-e a kép -> pálya H a szűrőkhöz. None -> nem. Pályakövetővel (pitch_q): csak
    'ok' állapot, elég nagy bizalom és nincs elveszett lépés. Pályakövető nélkül (statikus H)
    igen – a motor (TacticalEngine) is ugyanezzel a H-val szűr.
    """
    if H is None:
        return False
    if pitch_q is None:
        return True
    if pitch_q.get("status") != "ok":
        return False
    conf = pitch_q.get("conf")
    if conf is not None and float(conf) < min_conf:
        return False
    return not pitch_q.get("lost")


def outside_distance_m(x, y, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """Mennyivel van a pont a pálya téglalapján kívül (m); a pályán belül 0."""
    dx = max(0.0, -x, x - length)
    dy = max(0.0, -y, y - width)
    return math.hypot(dx, dy)


def in_penalty_area(x, y, margin_m=0.0, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """Igaz, ha a pont valamelyik 16-oson belül (vagy legfeljebb margin_m-re tőle) van."""
    if abs(y - width / 2.0) > PENALTY_HALF_WIDTH + margin_m:
        return False
    return x <= PENALTY_DEPTH + margin_m or x >= length - PENALTY_DEPTH - margin_m


class PitchGeometry:
    """
    Egy képkocka kép <-> pálya vetítése: talppont -> pálya (m), pálya -> kép, és a horizont
    feletti (a kamera mögé vetülő) pontok kiszűrése. A referencia-előjel a kép alsó közepe
    (az biztosan a talajon van).
    """

    def __init__(self, H, frame_size):
        self.H = np.asarray(H, np.float64)
        self.W, self.Hh = frame_size
        try:
            self.G = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.G = None
        ref = self.H @ np.array([self.W / 2.0, self.Hh - 1.0, 1.0])
        self.sign = 1.0 if ref[2] >= 0 else -1.0
        self.gsign = 1.0
        if self.G is not None and abs(ref[2]) > 1e-12:
            q = ref / ref[2]
            g = self.G @ np.array([q[0], q[1], 1.0])
            self.gsign = 1.0 if g[2] >= 0 else -1.0

    def to_pitch(self, u, v):
        p = self.H @ np.array([float(u), float(v), 1.0])
        if p[2] * self.sign <= 1e-12:
            return None   # horizont felett
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_image(self, x, y):
        if self.G is None:
            return None
        p = self.G @ np.array([float(x), float(y), 1.0])
        if p[2] * self.gsign <= 1e-12:
            return None   # a kamera mögött
        return float(p[0] / p[2]), float(p[1] / p[2])

    def pixels_per_meter(self, x, y, d=1.0):
        """Képpont / méter a pálya (x, y) pontjában (a kevésbé rövidülő talaj-irány szerint)."""
        p0 = self.to_image(x, y)
        p1 = self.to_image(x + d, y)
        p2 = self.to_image(x, y + d)
        if p0 is None or p1 is None or p2 is None:
            return None
        return max(math.hypot(p1[0] - p0[0], p1[1] - p0[1]), math.hypot(p2[0] - p0[0], p2[1] - p0[1])) / d


class PitchBoundary:
    """
    Pályahatár-szűrő a képkocka homográfiájával (Veo: képkockánként más).
      1. képtér: a margin_m-rel kitolt oldalvonalak (y = -margin és y = 68 + margin) képi vetülete
         oszloponként; a vonalon dobozmagasság-arányos tűréssel túl álló talppont -> kizárva
         (a túlsó oldalvonal mögötti füves sávon álló néző, csere, melegítő);
      2. pályatér: a talppont (± a doboz-remegés tűrése) a pályán kívül több mint margin_m-re
         -> kizárva (a kapuk mögött is). A bedobás / szöglet / alapvonal mögötti kapus a
         margón belül marad.
    set_frame(H, trusted) képkockánként; nem megbízható H-nál minden átmegy.
    """

    def __init__(self, frame_size, margin_m=2.5, tol_ratio=0.12, min_tol_px=2.0, samples=43):
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.margin = float(margin_m)
        self.enabled = self.margin > 0
        self.tol_ratio = float(tol_ratio)
        self.min_tol_px = float(min_tol_px)
        self.samples = int(samples)
        self.geom = None
        self.lines = {}
        self.stats = {"képtér": 0, "pályatér": 0, "horizont": 0}

    def set_frame(self, H, trusted=True):
        """A képkocka H-ja; nem megbízható / None H-nál a szűrő átenged. Visszatér: PitchGeometry vagy None."""
        self.geom = None
        self.lines = {}
        if H is None or not trusted:
            return None
        self.geom = PitchGeometry(H, self.frame_size)
        if self.enabled:
            m = self.margin
            xs = np.linspace(-m, PITCH_LENGTH + m, self.samples)
            for name, yy in (("far", -m), ("near", PITCH_WIDTH + m)):
                pts = [self.geom.to_image(x, yy) for x in xs]
                pts = sorted(p for p in pts if p is not None)
                if len(pts) >= 2:
                    self.lines[name] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        return self.geom

    @property
    def active(self):
        return self.geom is not None

    def _line_y(self, name, u):
        ln = self.lines.get(name)
        if ln is None:
            return None
        us, vs = ln
        if u < us[0] or u > us[-1]:
            return None
        return float(np.interp(u, us, vs))

    def project(self, foot_x, foot_y):
        return None if self.geom is None else self.geom.to_pitch(foot_x, foot_y)

    def check(self, foot_x, foot_y, box_h):
        """Visszatér: (megtartható, ok-szöveg, pálya-pont vagy None)."""
        if self.geom is None:
            return True, "", None
        xy = self.geom.to_pitch(foot_x, foot_y)
        if not self.enabled:
            return True, "", xy
        tol = max(self.min_tol_px, self.tol_ratio * float(box_h))
        # 1. képtér: a kitolt oldalvonalak képi vetülete
        far_y = self._line_y("far", foot_x)
        if far_y is not None and foot_y + tol < far_y:
            self.stats["képtér"] += 1
            return False, (f"Túlsó oldalvonalon túl (képtér: talp Y={foot_y} < vonal-{self.margin:.1f}m "
                           f"Y={far_y:.1f}, tűrés {tol:.1f}px)"), xy
        near_y = self._line_y("near", foot_x)
        if near_y is not None and foot_y - tol > near_y:
            self.stats["képtér"] += 1
            return False, (f"Közeli oldalvonalon túl (képtér: talp Y={foot_y} > vonal+{self.margin:.1f}m "
                           f"Y={near_y:.1f}, tűrés {tol:.1f}px)"), xy
        # 2. pályatér: a talppont és a ± tűréssel eltolt változatai közül a legkedvezőbb
        ds = []
        for dv in (0.0, -tol, tol):
            p = xy if dv == 0.0 else self.geom.to_pitch(foot_x, foot_y + dv)
            if p is not None:
                ds.append(outside_distance_m(*p))
        if not ds:
            self.stats["horizont"] += 1
            return False, "Pályán kívül (a talppont a horizont fölé vetül)", None
        d = min(ds)
        if d > self.margin:
            self.stats["pályatér"] += 1
            where = f"({xy[0]:.1f}, {xy[1]:.1f}) m" if xy is not None else "(horizont felett)"
            return False, f"Pályán kívül (pályatér: {where}, {d:.1f} m > {self.margin:.1f} m)", xy
        return True, "", xy


class PitchTrackSanity:
    """
    Követés-szintű épség pálya-koordinátában (megbízható H mellett):
      * 'kint él': a követés megfigyeléseinek többsége (felejtő átlag, half_life_sec felezési idő)
        a pályán kívül (> outside_m) van, legalább min_life_sec óta -> a követés ELNYOMVA
        (edző, csere, melegítő a vonal mellett); visszatér, ha recover_sec-ig a pályán van;
      * 'sosem mozdul': a követés teljes (látott) élete alatt a pályán mért kiterjedése
        < still_max_m, legalább still_sec óta -> ELNYOMVA (pályatérben mérve, így a forgó kamera
        nem zavarja). Ha elmozdul, magától visszatér (a kiterjedés megnő).
    A megerősített kapus a 'sosem mozdul' alól kivétel (ezt a hívó adja meg).
    """

    def __init__(self, fps, outside_m=0.5, outside_frac=0.6, half_life_sec=3.0, min_life_sec=1.5,
                 recover_sec=1.0, still_sec=4.0, still_max_m=1.0, ttl_sec=10.0,
                 zone_radius_m=2.0, zone_ttl_sec=6.0):
        self.fps = float(fps or 25.0)
        self.outside_m = float(outside_m)
        self.outside_frac = float(outside_frac)
        self.half_life = float(half_life_sec) * self.fps
        self.min_life = float(min_life_sec) * self.fps
        self.recover = float(recover_sec) * self.fps
        self.still = float(still_sec) * self.fps
        self.still_max = float(still_max_m)
        self.ttl = float(ttl_sec) * self.fps
        self.zone_radius = float(zone_radius_m)
        self.zone_ttl = float(zone_ttl_sec) * self.fps
        self.zones = []          # a nemrég elnyomott 'kint élő' követések utolsó pozíciói (x, y, képkocka)
        self.tracks = {}
        self.stats = {"kint él": 0, "sosem mozdul": 0}

    def update(self, tid, frame_idx, pitch_xy, exempt_still=False):
        """Visszatér: (elnyomva, ok-szöveg). pitch_xy None -> nincs döntés (átmegy)."""
        if tid is None or pitch_xy is None:
            return False, ""
        x, y = float(pitch_xy[0]), float(pitch_xy[1])
        out = outside_distance_m(x, y) > self.outside_m
        tr = self.tracks.get(tid)
        if tr is None:
            tr = {"first": frame_idx, "last": frame_idx, "ema": 0.5 if out else 0.0, "in_since": None,
                  "bb": [x, y, x, y]}
            self.tracks[tid] = tr
            # azonosító-váltás (a kis, távoli alakoknál gyakori): a nemrég elnyomott 'kint élő'
            # követés helyén, szintén kint születő új követés örökli az elnyomást
            self.zones = [z for z in self.zones if frame_idx - z[2] <= self.zone_ttl]
            if out and any(math.hypot(x - zx, y - zy) <= self.zone_radius for zx, zy, _ in self.zones):
                tr["first"] = frame_idx - self.min_life
                tr["ema"] = 1.0
        else:
            a = 1.0 - 0.5 ** (max(1, frame_idx - tr["last"]) / self.half_life)
            tr["ema"] += a * ((1.0 if out else 0.0) - tr["ema"])
            tr["last"] = frame_idx
            bb = tr["bb"]
            bb[0], bb[1], bb[2], bb[3] = min(bb[0], x), min(bb[1], y), max(bb[2], x), max(bb[3], y)
        if out:
            tr["in_since"] = None
        elif tr["in_since"] is None:
            tr["in_since"] = frame_idx
        life = frame_idx - tr["first"]
        # visszatérés: a pályára jött és ott is maradt -> a 'kint' előzmény törlődik
        if tr["in_since"] is not None and frame_idx - tr["in_since"] >= self.recover:
            tr["ema"] = 0.0
        if life >= self.min_life and tr["ema"] >= self.outside_frac:
            self.stats["kint él"] += 1
            self.zones.append((x, y, frame_idx))
            self.zones = self.zones[-64:]
            return True, f"Pályán kívül él (felejtő arány {100 * tr['ema']:.0f}% > {self.outside_m:.1f} m-re kint)"
        bb = tr["bb"]
        span = max(bb[2] - bb[0], bb[3] - bb[1])
        if not exempt_still and self.still > 0 and life >= self.still and span < self.still_max:
            self.stats["sosem mozdul"] += 1
            return True, f"Sosem mozdul a pályán ({span:.2f} m < {self.still_max:.1f} m, {life / self.fps:.1f} s óta)"
        return False, ""

    def prune(self, frame_idx):
        for tid in [t for t, tr in self.tracks.items() if frame_idx - tr["last"] > self.ttl]:
            del self.tracks[tid]


class KeeperGate:
    """
    Kapus-megerősítés követésenként. Megerősített kapus (ez mentesül az álló alak szűrő alól):
      * a modell kapus osztálya (az aktuális detektálás, vagy a követés megfigyeléseinek >= 30%-a), vagy
      * mezszín-egyezés ÉS pálya-pozíció: a követés többnyire (felejtő átlag >= min_frac) valamelyik
        16-oson belül / annak közelében (area_margin_m) van.
    Csak színegyezés (pl. narancs mezes néző a sárga kapus tartományában): pálya-koordinátával a
    16-osokon kívül -> NEM kapus; pálya-koordináta nélkül kapusnak jelölhető, de nem mentesül.
    """

    def __init__(self, fps, area_margin_m=4.0, half_life_sec=5.0, min_frac=0.4, model_frac=0.3, ttl_sec=10.0):
        self.fps = float(fps or 25.0)
        self.area_margin = float(area_margin_m)
        self.half_life = float(half_life_sec) * self.fps
        self.min_frac = float(min_frac)
        self.model_frac = float(model_frac)
        self.ttl = float(ttl_sec) * self.fps
        self.tracks = {}

    def update(self, tid, frame_idx, model_keeper, colour_keeper, pitch_xy=None):
        """Visszatér: (kapus-e, megerősített-e, a színegyezés elfogadva-e)."""
        tr = self.tracks.get(tid) if tid is not None else None
        if tr is None:
            tr = {"n": 0, "nm": 0, "ema": None, "last": frame_idx}
            if tid is not None:
                self.tracks[tid] = tr
        tr["n"] += 1
        tr["nm"] += int(bool(model_keeper))
        if pitch_xy is not None:
            inside = 1.0 if in_penalty_area(pitch_xy[0], pitch_xy[1], self.area_margin) else 0.0
            if tr["ema"] is None:
                tr["ema"] = inside
            else:
                a = 1.0 - 0.5 ** (max(1, frame_idx - tr["last"]) / self.half_life)
                tr["ema"] += a * (inside - tr["ema"])
        tr["last"] = frame_idx
        model_ok = bool(model_keeper) or (tr["n"] >= 3 and tr["nm"] / tr["n"] >= self.model_frac)
        has_pos = tr["ema"] is not None
        pos_ok = has_pos and tr["ema"] >= self.min_frac
        colour_ok = bool(colour_keeper) and (pos_ok or not has_pos)
        confirmed = model_ok or (bool(colour_keeper) and pos_ok)
        return model_ok or colour_ok, confirmed, colour_ok

    def prune(self, frame_idx):
        for tid in [t for t, tr in self.tracks.items() if frame_idx - tr["last"] > self.ttl]:
            del self.tracks[tid]

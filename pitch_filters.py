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


def grass_top_profile(grass_mask, smooth_frac=0.03):
    """
    A fő (legnagyobb) összefüggő fűterület felső széle oszloponként, a maszk felbontásán: float tömb
    (oszloponként a legfelső fű-sor), NaN, ahol az oszlopban nincs fű. A tipikus Veo-képen ez a túlsó
    pályaszél menti kerítés / reklámtábla alja. Medián-simítás a szélesség smooth_frac részén.
    """
    if grass_mask is None:
        return None
    m = (grass_mask > 0).astype(np.uint8)
    h, w = m.shape[:2]
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 1:
        return None
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[main, cv2.CC_STAT_AREA] < 0.1 * h * w:
        return None
    mm = cc == main
    has = mm.any(axis=0)
    top = np.where(has, mm.argmax(axis=0), -1).astype(np.float32)
    k = max(3, (int(w * smooth_frac) // 2) * 2 + 1)
    valid = top >= 0
    if valid.sum() < k:
        return None
    filled = np.interp(np.arange(w), np.where(valid)[0], top[valid]).astype(np.float32)
    pad = np.pad(filled, (k // 2, k // 2), mode="edge")
    sm = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1).astype(np.float32)
    sm[~valid] = np.nan
    return sm


class BoundaryCrowdFilter:
    """
    NÉZŐ A TÚLSÓ PÁLYASZÉLEN – kalibráció nélküli heurisztika a tipikus Veo-elrendezéshez: a nézők a
    túlsó kerítés / reklámtábla előtti füves sávban állnak. Ha nincs megbízható pálya-homográfia (a
    pályahatár-szűrő nem működik), az ilyen alak a fő fűterület felső széléhez (a kerítés aljához)
    közel áll (band_ratio * dobozmagasság), ÉS ugyanazon a – kamera-mozgással kompenzált – helyen
    több másodpercig folyamatosan látszik valaki. A követő-azonosító cserélődése (valódi felvételen
    gyakori) itt nem számít: a HELY állandósága dönt, nem a követésé. A partjelző és a bedobó
    mozog (a hely nem állandó), így legfeljebb rövid ideig esik ki.
    """

    def __init__(self, fps, window_sec=3.0, band_ratio=0.45, min_band_frac=0.012, above_ratio=0.3,
                 radius_ratio=0.6, min_presence=0.7, profile_every=5):
        self.fps = float(fps or 25.0)
        self.window = max(1, int(round(window_sec * self.fps)))
        self.band_ratio, self.min_band_frac, self.above_ratio = band_ratio, min_band_frac, above_ratio
        self.radius_ratio, self.min_presence = radius_ratio, min_presence
        self.profile_every = max(1, int(profile_every))
        self.profile, self.scale, self.frame_h = None, 1.0, None
        self._prof_frame = None
        self.hist = deque()           # (képkocka, sx, sy, magasság) – a sávban látott alakok
        self.frames = deque()         # a vizsgált képkockák
        self.stats = {"sávban": 0, "néző (álló a sávban)": 0}

    def set_frame(self, frame_idx, grass_mask, grass_scale, frame_h):
        """Képkockánként (detektáláskor) hívandó: a fű felső szélének frissítése és az ablak léptetése."""
        self.frame_h = frame_h
        if grass_mask is not None and (self._prof_frame is None or frame_idx - self._prof_frame >= self.profile_every):
            self.profile, self.scale = grass_top_profile(grass_mask), float(grass_scale)
            self._prof_frame = frame_idx
        self.frames.append(frame_idx)
        lim = frame_idx - self.window
        while self.frames and self.frames[0] < lim:
            self.frames.popleft()
        while self.hist and self.hist[0][0] < lim:
            self.hist.popleft()

    def edge_y(self, x):
        """A fű felső széle (teljes képes px) az x oszlopban, vagy None."""
        if self.profile is None:
            return None
        i = int(min(max(0, x * self.scale), len(self.profile) - 1))
        v = self.profile[i]
        return None if not np.isfinite(v) else float(v) / self.scale

    def in_band(self, foot_x, foot_y, box_h):
        e = self.edge_y(foot_x)
        if e is None:
            return False
        band = max(self.band_ratio * box_h, self.min_band_frac * (self.frame_h or 1080))
        return -self.above_ratio * box_h <= foot_y - e <= band

    def update(self, frame_idx, foot_x, foot_y, sx, sy, box_h):
        """Visszatér: True, ha a (sávban álló) alak nézőnek számít. sx, sy: kamera-kompenzált talppont."""
        if not self.in_band(foot_x, foot_y, box_h):
            return False
        self.stats["sávban"] += 1
        r = max(4.0, self.radius_ratio * box_h)
        hit_frames = {f for (f, hx, hy, _h) in self.hist if f < frame_idx and abs(hx - sx) <= r and abs(hy - sy) <= r}
        self.hist.append((frame_idx, sx, sy, box_h))
        past = [f for f in self.frames if f < frame_idx]
        if not past or frame_idx - past[0] < 0.8 * self.window:
            return False
        if len(hit_frames) >= self.min_presence * len(past):
            self.stats["néző (álló a sávban)"] += 1
            return True
        return False


def extract_dominant_color(image_crop, grass_lower=None, grass_upper=None, grass_model=None):
    """
    Mezszín a felsőtestből – gyors változat (KMeans helyett medián).
    A fűszínű pixeleket kihagyja, hogy a háttér ne torzítsa (grass_model: adaptív fűmodell –
    ha tanult, ez dönti el, mi a fű, a HSV tartomány helyett).
    Visszatér: [R, G, B]
    """
    if image_crop.size == 0:
        return [0, 0, 0]
    h, w = image_crop.shape[:2]
    torso = image_crop[int(h * 0.2):int(h * 0.6), int(w * 0.2):int(w * 0.8)]
    if torso.size == 0:
        return [0, 0, 0]

    pixels = torso.reshape(-1, 3)
    if grass_model is not None and getattr(grass_model, "ready", False):
        not_grass = ~grass_model.is_grass_pixels(pixels)
        if np.count_nonzero(not_grass) >= 5:
            pixels = pixels[not_grass]
    elif grass_lower is not None and grass_upper is not None and not is_full_range(grass_lower, grass_upper):
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


# ======================================================================
# ADAPTÍV FŰMODELL: a fű színeloszlása videónként tanulva (árnyék-, csillanás- és színezet-tűrő)
# ======================================================================
DEFAULT_GRASS_HSV = (np.array([30, 40, 40]), np.array([85, 255, 255]))   # a /cut-raw alapértéke = "auto"
_LOG_K = 3.0       # eltolás a log-színarányban (a sötét pixelek zaja ellen)
_LOG_LUT = np.log(np.arange(256, dtype=np.float32) + _LOG_K).astype(np.float32)


def grass_mode_from_config(config):
    """
    Melyik fűmaszk fusson: "auto" (adaptív modell) vagy "hsv" (a kliens fix HSV tartománya).
    A kézzel beállított (az alapértéktől és a teljes tartománytól eltérő) HSV tartomány mindig nyer;
    grass_model = "hsv" a régi, fix tartományos működést kényszeríti.
    """
    if str(config.get("grass_model", "auto")).strip().lower() == "hsv":
        return "hsv"
    lo, hi = config.get("grass_lower"), config.get("grass_upper")
    if lo is None or hi is None:
        return "auto"
    lo, hi = np.asarray(lo), np.asarray(hi)
    if is_full_range(lo, hi) or (np.array_equal(lo, DEFAULT_GRASS_HSV[0]) and np.array_equal(hi, DEFAULT_GRASS_HSV[1])):
        return "auto"
    return "hsv"


def _features(bgr):
    """(..., 3) uint8 BGR -> (..., 3) float32: log-színarány (G/R, G/B) és log-világosság.
    A két színarány független a megvilágítás erősségétől (árnyék = szorzó), a színezet (pl. kékes
    árnyék, meleg reflektorfény) csak eltolja őket."""
    f = _LOG_LUT[bgr]
    gray = (0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]).astype(np.uint8)
    out = np.empty(bgr.shape[:-1] + (3,), np.float32)
    out[..., 0] = f[..., 1] - f[..., 2]
    out[..., 1] = f[..., 1] - f[..., 0]
    out[..., 2] = _LOG_LUT[gray]
    return out


class _Mode:
    """Egy fű-módus (pl. napos / árnyékos fű): 2D Gauss a színarányokon + világosság-tartomány."""
    __slots__ = ("mu", "cov", "icov", "lv", "slv", "w", "bgr")

    def __init__(self, mu, cov, lv, slv, w, bgr):
        self.mu = np.asarray(mu, np.float64)
        self.cov = np.asarray(cov, np.float64)
        self.lv, self.slv, self.w = float(lv), float(slv), float(w)
        self.bgr = np.asarray(bgr, np.float64)
        self._inv()

    def _inv(self):
        c = self.cov + np.eye(2) * GrassModel.CHROMA_FLOOR ** 2
        self.icov = np.linalg.inv(c)

    def d2(self, F):
        dx = F[..., 0] - self.mu[0]
        dy = F[..., 1] - self.mu[1]
        a, b, c = self.icov[0, 0], self.icov[0, 1], self.icov[1, 1]
        return a * dx * dx + 2 * b * dx * dy + c * dy * dy

    def blend(self, other, a):
        self.mu = (1 - a) * self.mu + a * other.mu
        self.cov = (1 - a) * self.cov + a * other.cov
        self.lv = (1 - a) * self.lv + a * other.lv
        self.slv = (1 - a) * self.slv + a * other.slv
        self.w = (1 - a) * self.w + a * other.w
        self.bgr = (1 - a) * self.bgr + a * other.bgr
        self._inv()


class GrassModel:
    """
    Videónként tanult fűmodell a fix HSV-doboz helyett.

    * Jellemzők: log(G/R), log(G/B) színarány (a fényerőtől független) + log-világosság.
    * Tanulás (observe): a kép alsó-középső részéből (ott szinte biztosan fű van) robusztus kezdő
      becslés, majd a kép alsó 65%-ának "zöldes" pixeleire legfeljebb 2 módusú Gauss-keverék (EM):
      napos és árnyékos fű (az árnyékos módus színaránya a napostól legfeljebb MODE_CHROMA_GAP-re
      lehet – így a fák / reklámtábla nem lesznek "fű"). Később lassan frissül (felejtő átlag), és
      ha új, elég nagy árnyékos / megvilágított rész jelenik meg, a második módus akkor is létrejön.
    * Maszk (mask): a pixel fű, ha valamelyik módus színarány-Mahalanobis távolsága < D2_MAX és a
      világossága a módus tartományában van; vagy ha a napos módus "kifakult" (csillanás /
      fátyolfény: a fehér felé eltolódott, de világosabb) változata. Utána zárás, mint a régi
      build_grass_mask-nál (a fehér vonalak és a lábak árnyéka kitöltődik).
    * Kimenetek a többi lépésnek: mask() (foot_on_grass, labdakövető), is_grass_pixels() (mezszín
      fű-kiszűrése), hsv_bounds() (HSV-tartományt váró kód, pl. a pályakövető), illumination_gain()
      (a játékos körüli fű alapján megvilágítás-korrekció a mezszínhez).
    Tanulás előtt (ready == False) a mask() a fix alap HSV tartományra esik vissza.
    """

    CHROMA_FLOOR = 0.05      # a színarány-szórás alsó korlátja (zaj, tömörítés)
    D2_MAX = 9.0             # ~3 szigma
    LV_TOL = 3.0             # világosság-tűrés (szigma)
    LV_TOL_MIN = 0.45        # ... de legalább ennyi (log; ~x1.6)
    LV_TOL_DARK = 1.0        # sötétebb irányban legalább ennyi (log; ~x0,37: az alakok saját árnyéka is fű)
    MODE_CHROMA_GAP = 0.45   # a második módus színaránya legfeljebb ennyire térhet el az elsőtől
    MODE_LV_GAP = 0.25       # ... és világosságban legalább ennyire különbözzön (vagy színarányban 2,5 szigmára)
    MIN_MODE_W = 0.1         # a második módus minimális súlya
    SHADOW_CHROMA_GAP = 0.25  # árnyék-módus (megvilágítás-korrekcióhoz): legfeljebb ennyi színarány-eltérés ...
    SHADOW_LV_GAP = 0.25      # ... és legalább ennyivel sötétebb (log; ~x0,78)
    DENS_WIN, DENS_MIN = 5, 0.5   # többségi szűrő: ablakméret (px) és a fű-pixelek minimális aránya
    EDGE_GROW = 0            # a kész maszk tágítása (px, a maszk felbontásán; 0 = nincs)
    D2_WEAK = 36.0           # gyenge jelölt színarány-tűrése (~6 szigma)
    WEAK = 4                 # a gyenge jelölt címkéje
    FADED_T_MIN = 0.2        # a "kifakult" (csillanás) fű minimális megmaradt színessége a napos módushoz képest

    def __init__(self, update_alpha=0.15, max_samples=24000, learn_frames=3, learn_scale=0.5, seed=0):
        self.modes = []
        self.learn_scale = float(learn_scale)
        self.learn_frames = max(1, int(learn_frames))
        self.update_alpha = float(update_alpha)
        self.max_samples = int(max_samples)
        self.rng = np.random.default_rng(seed)
        self.n_obs = 0
        self.n_fits = 0
        self._hsv_bounds = None
        self._pending = []          # a kezdeti tanulás mintái (több képkocka)
        self.history = []           # (megfigyelés sorszáma, módusok száma, árnyék-arány) – naplóhoz

    @property
    def ready(self):
        return bool(self.modes)

    # ------------------------------------------------------------------ tanulás
    def _candidates(self, frame):
        """
        A tanulás mintái egy képkockából, a maszk felbontásán (learn_scale):
        (mag-pixelek, mag-jellemzők), (pálya-pixelek, pálya-jellemzők). Pálya-pixel: durván zöldes,
        nem túl sötét, nem beégett, a referencia (napos) fű színarányához közeli, ÉS a kép alsó-középső
        részéhez (ahol biztosan fű van) összefüggő területen van – így a fasor, a távoli bokrok a
        reklámtábla / salakpálya mögött nem kerülnek a mintába (a második, árnyékos módus sem tanulhat rájuk).
        """
        sc = self.learn_scale
        small = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc != 1.0 else frame
        h, w = small.shape[:2]
        F = _features(small)
        green = (F[..., 0] > 0.04) & (F[..., 1] > -0.15) & (F[..., 2] > np.log(14.0)) & (small.max(axis=2) < 253)
        sy0, sy1, sx0, sx1 = int(h * 0.60), int(h * 0.97), int(w * 0.15), int(w * 0.85)
        seed_zone = np.zeros((h, w), bool)
        seed_zone[sy0:sy1, sx0:sx1] = True
        ref = self.modes[0].mu if self.ready else None
        if ref is None:
            sf = F[seed_zone & green]
            if len(sf) < 50:
                e = np.zeros((0, 3), np.float32)
                return (e.astype(np.uint8), e), (e.astype(np.uint8), e)
            ref = np.median(sf[:, :2], axis=0)
        near = green & (np.hypot(F[..., 0] - ref[0], F[..., 1] - ref[1]) < self.MODE_CHROMA_GAP + 0.1)
        dens = cv2.blur(near.astype(np.float32), (5, 5)) >= 0.5
        n, cc = cv2.connectedComponents(dens.astype(np.uint8), connectivity=8)
        if n > 1:
            counts = np.bincount(cc[seed_zone].ravel(), minlength=n)
            counts[0] = 0
            keep = counts >= max(20, 0.05 * counts.sum())
            region = keep[cc] & near
        else:
            region = near
        seed = region & seed_zone
        return (small[seed], F[seed]), (small[region], F[region])

    def _fit(self, seed_px, seed_F, px, F, init=None):
        """Legfeljebb 2 módusú, korlátozott EM. Visszatér: módus-lista vagy None."""
        if len(seed_F) < 50 and init is None:
            return None
        if len(F) > self.max_samples:
            idx = self.rng.choice(len(F), self.max_samples, replace=False)
            F, px = F[idx], px[idx]
        if init is None:
            med = np.median(seed_F, axis=0)
            mad = np.median(np.abs(seed_F - med), axis=0) * 1.4826 + 1e-3
            core = np.all(np.abs(seed_F - med) < 3 * mad, axis=1)
            sf = seed_F[core] if core.sum() >= 30 else seed_F
            A = _Mode(sf[:, :2].mean(0), np.cov(sf[:, :2].T), sf[:, 2].mean(), max(0.08, sf[:, 2].std()), 1.0,
                      seed_px[core].mean(0) if core.sum() >= 30 else seed_px.mean(0))
            modes = [A]
        else:
            modes = [_Mode(m.mu, m.cov, m.lv, m.slv, m.w, m.bgr) for m in init]
        A = modes[0]
        # csak a napos módus közelében lévő (színarányban) pixelek számítanak – fák, tábla, salak kiesnek
        near = np.hypot(F[:, 0] - A.mu[0], F[:, 1] - A.mu[1]) < self.MODE_CHROMA_GAP + 0.1
        F, px = F[near], px[near]
        if len(F) < 50:
            return modes if init is not None else None
        # második módus keresése: a napos módushoz nem illő, de közeli színarányú, elég sok pixel
        if len(modes) == 1:
            dA = A.d2(F)
            lvok = np.abs(F[:, 2] - A.lv) <= max(self.LV_TOL * A.slv, self.LV_TOL_MIN)
            out = ~((dA < self.D2_MAX) & lvok)
            if out.mean() >= self.MIN_MODE_W:
                Fo = F[out]
                modes.append(_Mode(np.median(Fo[:, :2], 0), np.cov(Fo[:, :2].T), np.median(Fo[:, 2]),
                                   max(0.08, Fo[:, 2].std()), out.mean(), px[out].mean(0)))
        # EM (3D Gauss: színarány + világosság), néhány lépés
        for _ in range(8):
            ll = []
            for m in modes:
                d = m.d2(F) + ((F[:, 2] - m.lv) / max(m.slv, 0.05)) ** 2
                det = np.linalg.det(m.cov + np.eye(2) * self.CHROMA_FLOOR ** 2) * max(m.slv, 0.05) ** 2
                ll.append(np.log(max(m.w, 1e-3)) - 0.5 * d - 0.5 * np.log(max(det, 1e-12)))
            ll = np.stack(ll, 1)
            ll -= ll.max(1, keepdims=True)
            R = np.exp(ll)
            R /= R.sum(1, keepdims=True)
            for k, m in enumerate(modes):
                r = R[:, k]
                s = r.sum()
                if s < 20:
                    continue
                mu = (F[:, :2] * r[:, None]).sum(0) / s
                X = F[:, :2] - mu
                cov = (X * r[:, None]).T @ X / s
                lv = float((F[:, 2] * r).sum() / s)
                slv = float(np.sqrt(((F[:, 2] - lv) ** 2 * r).sum() / s))
                m.mu, m.cov, m.lv, m.slv, m.w = mu, cov, lv, max(0.06, slv), float(s / len(F))
                m.bgr = (px * r[:, None]).sum(0) / s
                m._inv()
        # épség: a második módus a napostól nem túl távoli színarányú és elég nagy; különben eldobjuk
        if len(modes) == 2:
            a, b = modes
            gap = float(np.hypot(*(a.mu - b.mu)))
            separate = abs(a.lv - b.lv) >= self.MODE_LV_GAP or b.d2(a.mu[None, :])[0] > 6.0
            if b.w < self.MIN_MODE_W or gap > self.MODE_CHROMA_GAP or not separate:
                modes = [a]
                a.w = 1.0
            # a világosabb módus legyen az első ("napos" referencia)
            if len(modes) == 2 and modes[1].lv > modes[0].lv:
                modes = [modes[1], modes[0]]
        return modes

    def observe(self, frame):
        """Egy képkocka megfigyelése (tanulás / lassú frissítés). Visszatér: True, ha a modell változott.
        Az első illesztés learn_frames képkocka mintáiból készül (lásd még flush())."""
        (sp, sF), (wp, wF) = self._candidates(frame)
        self.n_obs += 1
        if not self.ready:
            self._pending.append((sp, sF, wp, wF))
            if len(self._pending) < self.learn_frames:
                return False
            return self.flush()
        return self._update(sp, sF, wp, wF)

    def flush(self):
        """A kezdeti tanulás lezárása az eddig gyűjtött mintákból. Visszatér: True, ha kész a modell."""
        if self.ready or not self._pending:
            return self.ready
        sp = np.concatenate([p[0] for p in self._pending])
        sF = np.concatenate([p[1] for p in self._pending])
        wp = np.concatenate([p[2] for p in self._pending])
        wF = np.concatenate([p[3] for p in self._pending])
        modes = self._fit(sp, sF, wp, wF)
        if modes is None:
            self._pending = self._pending[-self.learn_frames:]
            return False
        self.modes = modes
        self._pending = []
        self.n_fits += 1
        self._update_hsv(wp, wF)
        self.history.append((self.n_obs, len(self.modes), self.shadow_weight()))
        return True

    def _update(self, sp, sF, wp, wF):
        # lassú frissítés: a jelenlegi modellből indított EM, felejtő átlaggal
        new = self._fit(sp, sF, wp, wF, init=self.modes if len(self.modes) == 2 else None)
        if new is None and len(self.modes) == 1:
            new = self._fit(sp, sF, wp, wF, init=self.modes)
        if not new:
            return False
        a = self.update_alpha
        # az új módusok párosítása a meglévőkhöz (színarány + világosság szerint)
        used = set()
        for m in self.modes:
            dist = [float(np.hypot(*(m.mu - n.mu)) + abs(m.lv - n.lv)) if j not in used else 1e9
                    for j, n in enumerate(new)]
            j = int(np.argmin(dist)) if dist else -1
            if j >= 0 and dist[j] < 0.6:
                used.add(j)
                m.blend(new[j], a)
            else:
                m.w *= (1 - a)          # most nem látszik: lassan felejtjük
        for j, n in enumerate(new):
            if j not in used and len(self.modes) < 2 and n.w >= self.MIN_MODE_W:
                # új (pl. árnyékos) rész jelent meg: a második módus azonnal felvéve, kisebb súllyal
                n.w = max(self.MIN_MODE_W, n.w * 0.5)
                self.modes.append(n)
        if len(self.modes) == 2 and min(m.w for m in self.modes) < self.MIN_MODE_W * 0.5:
            self.modes = [max(self.modes, key=lambda m: m.w)]
        self.modes.sort(key=lambda m: -m.lv)     # a világosabb a "napos" referencia
        self.n_fits += 1
        self._update_hsv(wp, wF)
        self.history.append((self.n_obs, len(self.modes), self.shadow_weight()))
        return True

    def shadow_like(self):
        """Igaz, ha a második módus valódi (szorzó jellegű) árnyék: hasonló színarány, jóval sötétebb.
        A csillanás / fátyolfény (hozzáadódó fény, fakóbb szín) ilyenkor NEM árnyék – arra a
        szorzós megvilágítás-korrekció rossz volna."""
        if len(self.modes) < 2:
            return False
        a, b = self.modes
        return float(np.hypot(*(a.mu - b.mu))) < self.SHADOW_CHROMA_GAP and (a.lv - b.lv) >= self.SHADOW_LV_GAP

    def shadow_weight(self):
        return 0.0 if len(self.modes) < 2 else float(self.modes[1].w / (self.modes[0].w + self.modes[1].w))

    # ------------------------------------------------------------------ osztályozás
    def classify_features(self, F):
        """F: (..., 3) jellemzők -> címke: 0 = nem fű, 1 = napos módus, 2 = második módus, 3 = kifakult
        (csillanás), 4 (WEAK) = gyenge jelölt (csak biztos fű mellett fű)."""
        lab = np.zeros(F.shape[:-1], np.uint8)
        if not self.ready:
            return lab
        best = None
        for k, m in enumerate(self.modes):
            # a világosság-tűrés lefelé nagyobb: a játékosok / labda saját árnyéka is fű (a színarány marad)
            dlv = F[..., 2] - m.lv
            tol_up = max(self.LV_TOL * m.slv, self.LV_TOL_MIN)
            tol_dn = max(self.LV_TOL * m.slv, self.LV_TOL_DARK)
            # a módusnál sötétebb (árnyékos) pixel színaránya zajosabb és kissé a 0 felé húz: nagyobb tűrés
            d2 = m.d2(F) / (1.0 + (np.minimum(dlv, 0.0) / 0.4) ** 2)
            lvok = (dlv <= tol_up) & (dlv >= -tol_dn)
            ok = (d2 < self.D2_MAX) & lvok
            weak = (d2 < self.D2_WEAK) & lvok
            weak_any = weak if k == 0 else (weak_any | weak)
            score = np.where(ok, d2 + (dlv / max(m.slv, 0.1)) ** 2, np.inf)
            if best is None:
                best = score
                lab[ok] = k + 1
            else:
                closer = score < best
                lab[closer] = k + 1
                best = np.minimum(best, score)
        # félárnyék (az árnyék éle): a két módus keveréke – a kettőt összekötő szakasz közelében
        if len(self.modes) == 2:
            A, B = self.modes
            d = np.array([B.mu[0] - A.mu[0], B.mu[1] - A.mu[1], B.lv - A.lv])
            nd = float(d @ d)
            if nd > 1e-6:
                X0, X1, X2 = F[..., 0] - A.mu[0], F[..., 1] - A.mu[1], F[..., 2] - A.lv
                t = (X0 * d[0] + X1 * d[1] + X2 * d[2]) / nd
                tc = np.clip(t, 0.0, 1.0)
                e0, e1, e2 = X0 - tc * d[0], X1 - tc * d[1], X2 - tc * d[2]
                sig = max(self.CHROMA_FLOOR, float(np.sqrt((np.trace(A.cov) + np.trace(B.cov)) / 4.0)))
                mix = (t > 0) & (t < 1) & (np.hypot(e0, e1) < 2.5 * sig) & \
                      (np.abs(e2) < max(A.slv, B.slv) * 2.0) & (lab == 0)
                lab[mix] = 2
        # csillanás / fátyolfény: a módus színaránya a fehér (0, 0) felé "kifakulva", de világosabban
        # (napos módus: fátyolfény; árnyékos módus: az árnyékba eső, elmosott fehér vonal is ilyen)
        for k, m in enumerate(self.modes):
            n2 = float(m.mu @ m.mu)
            if n2 <= 0.04:
                continue
            t = (F[..., 0] * m.mu[0] + F[..., 1] * m.mu[1]) / n2
            perp = np.hypot(F[..., 0] - t * m.mu[0], F[..., 1] - t * m.mu[1])
            sig = max(self.CHROMA_FLOOR, float(np.sqrt(np.trace(m.cov) / 2.0)))
            t_min = self.FADED_T_MIN if k == 0 else 0.2
            faded = (t >= t_min) & (t < 1.0) & (perp < 3.0 * sig * np.maximum(t, 0.5)) & \
                    (F[..., 2] >= m.lv - 0.15) & (lab == 0)
            lab[faded] = 3 if k == 0 else 2   # az árnyékos módus "kifakult" pixele (pl. vonal) árnyékos fű
        # gyenge jelölt: a színarány tágabb tűrésen belül (a játékosok körüli színszivárgás, tömörítés);
        # a maszkban csak a biztos fű közvetlen szomszédságában számít fűnek
        lab[weak_any & (lab == 0)] = self.WEAK
        return lab

    LUT_BITS = 6     # a gyors osztályozó táblázat csatornánkénti felbontása (6 bit = 4-es lépés)

    def labels(self, bgr):
        """
        Gyors osztályozás táblázatból: (..., 3) uint8 BGR -> címke (mint a classify_features). A
        táblázat (2^18 szín, a 4-es rács középpontjaival) a modell minden változásakor újraépül.
        """
        bgr = np.asarray(bgr, np.uint8)
        if not self.ready:
            return np.zeros(bgr.shape[:-1], np.uint8)
        key = (self.n_fits, len(self.modes))
        if getattr(self, "_lut_key", None) != key:
            n = 1 << self.LUT_BITS
            step = 256 // n
            c = (np.arange(n) * step + step // 2).astype(np.uint8)
            B, G, R = np.meshgrid(c, c, c, indexing="ij")
            grid = np.stack([B, G, R], -1).reshape(-1, 3)
            self._lut = self.classify_features(_features(grid)).astype(np.uint8)
            self._lut_key = key
        sh = 8 - self.LUT_BITS
        b = bgr[..., 0].astype(np.uint32) >> sh
        g = bgr[..., 1].astype(np.uint32) >> sh
        r = bgr[..., 2].astype(np.uint32) >> sh
        return self._lut[(b << (2 * self.LUT_BITS)) | (g << self.LUT_BITS) | r]

    def is_grass_pixels(self, bgr_pixels):
        """(N, 3) uint8 BGR -> bool (N,): fű-e a pixel (pl. a mezszín-leíróhoz)."""
        px = np.asarray(bgr_pixels, np.uint8).reshape(-1, 3)
        lab = self.labels(px)
        return (lab > 0) & (lab < self.WEAK)

    def mask(self, frame_bgr, scale=0.5):
        """Fűmaszk csökkentett felbontáson (mint a build_grass_mask). Visszatér: (maszk, scale)."""
        if not self.ready:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            return build_grass_mask(hsv, *DEFAULT_GRASS_HSV, scale=scale)
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        lab = self.labels(small)
        strong = (lab > 0) & (lab < self.WEAK)
        # csak a SŰRŰ biztos fű mellett (a fák / bokrok elszórt "fűszínű" foltjai mellett nem)
        solid = (lab == 1) | (lab == 2)    # a kifakult (csillanó) rész mellett nem: ott minden szürkés-zöld
        dense = solid & (cv2.blur(solid.astype(np.float32), (5, 5)) >= 0.7)
        near = cv2.dilate(dense.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return self._clean_mask(strong, extra=(lab == self.WEAK) & near), scale

    def _clean_mask(self, raw, extra=None):
        """
        A pixelenkénti döntés tisztítása: (1) többségi szűrő (a fák / bokrok foltjai közül csak
        néhány pixel "fűszínű", a fű sűrű), (2) zárás (fehér vonalak, árnyékok), (3) csak a kép alsó
        részébe nyúló vagy nagyon nagy összefüggő részek maradnak (a fasor, a távoli zöld felületek
        nem a pálya). extra: további (gyenge) fű-pixelek – az összefüggőség-vizsgálatban nem
        vesznek részt (nem köthetik össze a fasort a pályával), csak a végső maszkban.
        """
        h, w = raw.shape[:2]
        dens = cv2.blur(raw.astype(np.float32), (self.DENS_WIN, self.DENS_WIN))
        # a vékony (1–2 px) fehér vonalak ne vágják szét a füvet, a vastagabb tábla / salaksáv igen
        rawd = cv2.dilate(raw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        core = np.where((dens >= self.DENS_MIN) & rawd, 255, 0).astype(np.uint8)
        # az összefüggő részek a zárás ELŐTT: a reklámtábla / salaksáv még elválasztja a fasort a pályától
        n, cc, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        if n > 2:
            low = int(h * 0.55)
            keep = np.zeros(n, bool)
            for i in range(1, n):
                y0, hh, area = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
                keep[i] = (y0 + hh > low and area >= 0.002 * h * w) or area >= 0.2 * h * w
            core = np.where(keep[cc], 255, 0).astype(np.uint8)
        # a többségi szűrő a széleken "rág" (a játékosok körül): a megtartott részek közvetlen
        # környezetében a pixelenkénti döntés számít
        zone = cv2.dilate(core, np.ones((5, 5), np.uint8)) > 0
        full = raw if extra is None else (raw | extra)
        mask = np.where(full & zone, 255, 0).astype(np.uint8)
        k = max(5, (int(h / 60) // 2) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if self.EDGE_GROW > 0:
            # a fű széle kicsit kifelé (a láb / cipő / saját árnyék / színszivárgás a fűhöz tartozik)
            g = 2 * self.EDGE_GROW + 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
        return mask

    def label_map(self, frame_bgr, scale=0.5):
        """Módus-címkék képe (0 nem fű, 1 napos, 2 árnyékos/második, 3 kifakult) – minőségméréshez, rajzoláshoz."""
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame_bgr
        lab = self.labels(small)
        lab[lab == self.WEAK] = 0
        return lab

    # ------------------------------------------------------------------ kimenetek más lépéseknek
    def _update_hsv(self, px, F):
        lab = self.classify_features(F)
        g = px[(lab > 0) & (lab < self.WEAK)]
        if len(g) < 50:
            return
        hsv = cv2.cvtColor(g.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        lo = np.array([np.percentile(hsv[:, 0], 0.5) - 5, np.percentile(hsv[:, 1], 0.5) - 25,
                       np.percentile(hsv[:, 2], 0.5) - 20])
        hi = np.array([np.percentile(hsv[:, 0], 99.5) + 5, 255, 255])
        lo = np.clip(np.round(lo), [0, 10, 10], [179, 255, 255]).astype(int)
        hi = np.clip(np.round(hi), [0, 0, 0], [179, 255, 255]).astype(int)
        self._hsv_bounds = (lo, hi)

    def hsv_bounds(self, union_default=False):
        """A tanult fű (mindkét módus) HSV-burka (lower, upper) a HSV tartományt váró kódnak
        (pl. pitch_tracker.PitchTracker); tanulás előtt az alap tartomány. union_default: az alap
        tartománnyal egyesítve (sosem szűkebb a réginél – a meglévő hangolás megmarad)."""
        if self._hsv_bounds is None:
            return DEFAULT_GRASS_HSV[0].copy(), DEFAULT_GRASS_HSV[1].copy()
        lo, hi = self._hsv_bounds[0].copy(), self._hsv_bounds[1].copy()
        if union_default:
            lo, hi = np.minimum(lo, DEFAULT_GRASS_HSV[0]), np.maximum(hi, DEFAULT_GRASS_HSV[1])
        return lo, hi

    def reference_bgr(self):
        """A napos (referencia) fű átlagszíne BGR-ben, vagy None."""
        return None if not self.ready else self.modes[0].bgr.copy()

    def illumination_gain(self, frame_bgr, bbox, min_pixels=12):
        """
        Megvilágítás-korrekció a játékos körüli fű alapján (von Kries fehéregyensúly): a láb körüli
        sáv fűpixelei melyik módusba tartoznak; az árnyékos módus arányában a csatornánkénti szorzó
        a napos / árnyékos fű átlagszínének aránya (a módusok átlagából – nem zajos, és a nyírási
        csíkok sem zavarják). Visszatér: (szorzó BGR float32[3] vagy None, árnyékos arány 0..1).
        """
        if not self.shadow_like():
            return None, 0.0
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = max(2, x2 - x1), max(4, y2 - y1)
        rx1, rx2 = max(0, x1 - bw), min(W, x2 + bw)
        ry1, ry2 = max(0, y2 - int(0.25 * bh)), min(H, y2 + int(0.25 * bh) + 1)
        if rx2 <= rx1 or ry2 <= ry1:
            return None, 0.0
        step = max(1, int(min(rx2 - rx1, ry2 - ry1) / 24))
        patch = frame_bgr[ry1:ry2:step, rx1:rx2:step]
        inner = np.ones(patch.shape[:2], bool)
        cx1, cx2 = (x1 - rx1) // step, (x2 - rx1 + step - 1) // step
        cy1 = (y1 - ry1) // step
        inner[max(0, cy1):, max(0, cx1):max(0, cx2)] = False      # maga a játékos kimarad
        lab = self.labels(patch)[inner]
        n1, n2 = int(np.count_nonzero(lab == 1)), int(np.count_nonzero(lab == 2))
        if n1 + n2 < min_pixels:
            return None, 0.0
        frac = n2 / float(n1 + n2)
        if frac < 0.15:
            return None, frac
        ratio = np.clip(self.modes[0].bgr / np.maximum(self.modes[1].bgr, 1.0), 0.4, 4.0)
        gain = 1.0 + frac * (ratio - 1.0)
        return gain.astype(np.float32), frac

    def describe(self):
        if not self.ready:
            return "Adaptív fűmodell: még nincs tanulva (alap HSV tartomány)"
        parts = []
        for k, m in enumerate(self.modes):
            b, g, r = [int(v) for v in m.bgr]
            parts.append(f"{'napos' if k == 0 else 'árnyékos/második'}: RGB=({r},{g},{b}), súly {m.w:.2f}, "
                         f"színarány ({m.mu[0]:.2f}, {m.mu[1]:.2f}), világosság e^{m.lv:.2f}")
        lo, hi = self.hsv_bounds()
        return (f"Adaptív fűmodell ({len(self.modes)} módus, {self.n_fits} illesztés): " + "; ".join(parts)
                + f"; HSV-burok {lo.tolist()}–{hi.tolist()}")

    @classmethod
    def learn_from_video(cls, path, max_frames=8, step=6, **kw):
        """Előtanulás a videó első képkockáiból (minden step-edikből, legfeljebb max_frames darab)."""
        kw.setdefault("learn_frames", max_frames)
        model = cls(**kw)
        cap = cv2.VideoCapture(path)
        try:
            idx = used = 0
            while used < max_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    model.observe(fr)
                    used += 1
                idx += 1
            model.flush()
        finally:
            cap.release()
        return model

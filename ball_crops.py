"""
Natív felbontású labda-kivágás (LABDA-KIVÁGÁS) – a kis labda felismeréséhez.

Gond: a Veo 1920x1080-as képén a labda ~6–12 px átmérőjű; a felismerő a teljes képet imgsz=640-re
kicsinyíti (1/3), így a labda 2–4 px lesz – a COCO modell ezt alig látja (valódi felvételen 3,8%-os
labda-felismerés). Megoldás: a labda környékén NATÍV felbontású (pl. 640x640-es) kivágáson külön
labda-felismerés, a dobozokat visszavetítjük a teljes képre, és jelöltként adjuk a labdakövetőnek.

Melyik kivágás (plan):
  * ha a labdakövető tudja, hol a labda (detektált / rövid ideig becsült állapot): egy kivágás a
    következő képkockára becsült hely (pozíció + sebesség) köré – ha a teljes képes menet az előző
    képkockán magabiztosan megtalálta, csak minden `track_every`. képkockán (költségkorlát);
  * ha a labda elveszett: a pálya csempéin (a kép alsó ~70%-a, átfedéssel) körbeforgó keresés,
    `tile_every` képkockánként egy csempe; a sorrend a legutóbbi ismert hely (ha friss), különben a
    képközép (a Veo a labdát követi) közelségét követi.
A kivágásokat egy hívásban (batch) futtatjuk. Csak numpy – a felismerő hívását a hívó adja (detect_fn).
"""
import math

import numpy as np


class BallCropSearch:
    def __init__(self, frame_size, crop=640, tile_every=2, track_every=2, roi_top=0.3, overlap=0.15,
                 recent_sec=2.0, fps=25.0):
        self.W, self.H = int(frame_size[0]), int(frame_size[1])
        self.crop = int(max(32, min(crop, self.W, self.H)))
        self.tile_every = max(1, int(tile_every))
        self.track_every = max(1, int(track_every))
        self.recent_frames = int(round(recent_sec * fps))
        self.tiles = self._make_tiles(roi_top, overlap)
        self._order = None
        self._ptr = 0
        self._lost_since = None
        self.last_known = None      # (x, y, képkocka)
        self.stats = {"kivágás-futtatás": 0, "követő-kivágás": 0, "csempe": 0,
                      "kivágás-találat": 0, "teljes-kép-találat": 0, "csak-kivágás-találat": 0}

    # ------------------------------------------------------------------ csempék
    def _make_tiles(self, roi_top, overlap):
        c, W, H = self.crop, self.W, self.H
        y_lo = int(H * roi_top)
        step = max(1, int(c * (1.0 - overlap)))

        def starts(lo, hi):
            span = hi - lo
            if span <= c:
                return [max(0, hi - c)]
            n = int(math.ceil((span - c) / float(step))) + 1
            return [int(round(lo + k * (span - c) / float(n - 1))) for k in range(n)]
        tiles = []
        for y0 in starts(y_lo, H):
            for x0 in starts(0, W):
                tiles.append((x0, y0, x0 + c, y0 + c))
        return tiles

    def window_at(self, x, y):
        """A (x, y) köré igazított, képen belülre tolt kivágás (x0, y0, x1, y1)."""
        c = self.crop
        x0 = int(round(min(max(0.0, x - c / 2.0), self.W - c)))
        y0 = int(round(min(max(0.0, y - c / 2.0), self.H - c)))
        return (x0, y0, x0 + c, y0 + c)

    def _tile_order(self, frame_idx):
        if self.last_known is not None and frame_idx - self.last_known[2] <= self.recent_frames:
            px, py = self.last_known[0], self.last_known[1]
        else:
            px, py = self.W / 2.0, self.H * 0.6
        d = [math.hypot((t[0] + t[2]) / 2.0 - px, (t[1] + t[3]) / 2.0 - py) for t in self.tiles]
        return [self.tiles[i] for i in np.argsort(d, kind="stable")]

    # ------------------------------------------------------------------ tervezés
    def plan(self, frame_idx, state=None, full_frame_hit=False):
        """
        A képkocka kivágásai [(x0, y0, x1, y1), ...]. state: a labdakövető legutóbbi állapota
        (ball_tracker.BallState vagy None); full_frame_hit: az előző képkockán a teljes képes menet
        magabiztosan megtalálta-e a labdát (akkor ritkábban kell kivágás).
        """
        if state is not None and getattr(state, "found", False):
            self._lost_since = None
            self.last_known = (state.x, state.y, frame_idx)
            if full_frame_hit and frame_idx % self.track_every != 0:
                return []
            nx = state.x + (state.vx or 0.0)
            ny = state.y + (state.vy or 0.0)
            self.stats["követő-kivágás"] += 1
            return [self.window_at(nx, ny)]
        if self._lost_since is None:
            self._lost_since = frame_idx
            self._order = self._tile_order(frame_idx)
            self._ptr = 0
            return [self._next_tile()]
        if (frame_idx - self._lost_since) % self.tile_every != 0:
            return []
        if self._ptr >= len(self._order):
            self._order = self._tile_order(frame_idx)
            self._ptr = 0
        return [self._next_tile()]

    def _next_tile(self):
        t = self._order[self._ptr % len(self._order)]
        self._ptr += 1
        self.stats["csempe"] += 1
        return t

    # ------------------------------------------------------------------ futtatás
    def run(self, frame, windows, detect_fn):
        """
        A kivágások felismerése egy hívásban. detect_fn(list[kép]) -> list[[(xyxy, conf), ...]]
        (kivágás-koordinátában). Visszatér: [(xyxy a teljes képen, conf), ...].
        """
        if not windows:
            return []
        crops = [frame[y0:y1, x0:x1] for (x0, y0, x1, y1) in windows]
        self.stats["kivágás-futtatás"] += 1
        res = detect_fn(crops) or []
        out = []
        for (x0, y0, _x1, _y1), boxes in zip(windows, res):
            for xyxy, conf in boxes or []:
                bx1, by1, bx2, by2 = [float(v) for v in xyxy[:4]]
                out.append(((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0), float(conf)))
        return out

    def merge(self, full_boxes, crop_boxes, count_conf=0.0):
        """
        A teljes képes és a kivágásos labda-dobozok egyesítése (a duplikátumokból a nagyobb pontszámú
        marad). Visszatér: az egyesített [(xyxy, conf)] lista. A statisztika a count_conf feletti
        dobozokat számolja (képkockánként: volt-e labda a teljes képen / a kivágáson).
        """
        full_boxes = list(full_boxes or [])
        f_hit = any(c > count_conf for _b, c in full_boxes)
        c_hit = any(c > count_conf for _b, c in (crop_boxes or []))
        self.stats["teljes-kép-találat"] += int(f_hit)
        self.stats["kivágás-találat"] += int(c_hit)
        self.stats["csak-kivágás-találat"] += int(c_hit and not f_hit)
        out = list(full_boxes)
        for cb, cc in crop_boxes or []:
            ccx, ccy = (cb[0] + cb[2]) / 2.0, (cb[1] + cb[3]) / 2.0
            cs = max(4.0, 0.5 * ((cb[2] - cb[0]) + (cb[3] - cb[1])))
            dup = None
            for k, (fb, fc) in enumerate(out):
                if math.hypot((fb[0] + fb[2]) / 2.0 - ccx, (fb[1] + fb[3]) / 2.0 - ccy) <= cs:
                    dup = k
                    break
            if dup is None:
                out.append((tuple(cb), cc))
            elif cc > out[dup][1]:
                out[dup] = (tuple(cb), cc)
        return out

    def describe(self):
        s = self.stats
        return (f"Labda-kivágás ({self.crop}px, {len(self.tiles)} csempe, csempe minden {self.tile_every}. képkockán): "
                f"futtatás {s['kivágás-futtatás']} (követő-kivágás {s['követő-kivágás']}, csempe {s['csempe']}); "
                f"képkocka labdával – teljes kép: {s['teljes-kép-találat']}, kivágás: {s['kivágás-találat']}, "
                f"csak a kivágás találta: {s['csak-kivágás-találat']}")

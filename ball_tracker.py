"""
Labdakövetés a képtérben (a YOLO labda-jelöltjeiből egyetlen, folytonos labdapálya).

Eddig a main.py képkockánként egyszerűen az utolsó, küszöb feletti 'ball' detektálást vette
labdának: több jelöltnél véletlenszerű volt a választás, kihagyásnál (takarás, elmosódás)
pedig nem volt labda. Ez a modul:

  * pontozza a jelölteket (konfidencia, méret-konzisztencia, füvön van-e, távolság a
    becsült pozíciótól),
  * állandó sebességű Kálmán-szűrőt futtat KÉPPONT-térben (x, y, vx, vy),
  * kamera-mozgás kompenzációt fogad: képkockák közti eltolás (dx, dy) vagy 3x3 homográfia
    (előző kép -> aktuális kép), ezzel tolja el a becslést a forgó (Veo) kameránál,
  * kapuz (gating): a becsléstől túl messzi jelöltet kidobja (kiugró érték), de ha egy
    kapun kívüli jelölt több képkockán át következetesen jelen van, átvált rá (újraindítás),
  * rövid kimaradásnál előrejelzett pozíciót ad 'predicted' jelzéssel (alapból max. 0,24 s-ig;
    a követés belül 0,5 s-ig él tovább, hogy a visszatérő labdát a kapuzás még elfogadja –
    a szintetikus mérés szerint 6 képkocka után a becslés hibája már gyakran > 15 px),
  * opcionálisan pályakoordinátát számol a kép -> pálya homográfiával,
  * téves jelöltek elleni priorok (pályán van-e, a H-ból várható méret, fej / felsőtest,
    álló folt) és mag-követés: elvesztés után csak 2–3 egymást követő, következetes
    detektálásból indul újra (lásd BallTracker).

Csak numpy + OpenCV (a rajzoláshoz / homográfiához) kell, torch / ultralytics nem.
"""
import math

import numpy as np

try:  # a rajzolás és a homográfia OpenCV-t használ, a szűrő maga nem
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

STATUS_DETECTED = "detected"
STATUS_PREDICTED = "predicted"
STATUS_LOST = "lost"
STATUS_COASTING = "coasting"   # belül még követjük (kapuzáshoz), de a becslés már túl bizonytalan a kiadáshoz


class BallState:
    """Egy képkocka labda-becslése."""
    __slots__ = ("frame_idx", "status", "x", "y", "vx", "vy", "conf", "bbox", "score", "gap_frames", "n_candidates")

    def __init__(self, frame_idx, status, x=None, y=None, vx=0.0, vy=0.0, conf=0.0, bbox=None, score=0.0,
                 gap_frames=0, n_candidates=0):
        self.frame_idx = frame_idx
        self.status = status
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.conf = conf
        self.bbox = bbox
        self.score = score
        self.gap_frames = gap_frames
        self.n_candidates = n_candidates

    @property
    def found(self):
        return self.status in (STATUS_DETECTED, STATUS_PREDICTED) and self.x is not None

    @property
    def predicted(self):
        return self.status == STATUS_PREDICTED

    @property
    def pixel(self):
        """Egész képpont (x, y) vagy None, ha nincs labda."""
        if not self.found:
            return None
        return int(round(self.x)), int(round(self.y))

    def to_dict(self):
        return {"frame": self.frame_idx, "status": self.status,
                "pixel": None if not self.found else [round(self.x, 1), round(self.y, 1)],
                "velocity": [round(self.vx, 2), round(self.vy, 2)], "conf": round(self.conf, 3),
                "score": round(self.score, 3), "gap_frames": self.gap_frames}

    def __repr__(self):
        return f"BallState({self.frame_idx}, {self.status}, {self.pixel}, conf={self.conf:.2f})"


def _cand_center_size(c):
    """Jelölt: {'xyxy': (x1, y1, x2, y2), 'conf': float} vagy (x1, y1, x2, y2, conf) -> (cx, cy, méret, conf, doboz)."""
    if isinstance(c, dict):
        x1, y1, x2, y2 = c["xyxy"]
        conf = float(c.get("conf", 0.5))
    else:
        x1, y1, x2, y2, conf = c[:5]
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    size = max(1.0, 0.5 * ((x2 - x1) + (y2 - y1)))
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), size, float(conf), (x1, y1, x2, y2)


class BallTracker:
    """
    Online labdakövető. Képkockánként (vagy minden detektált képkockán) hívandó:
        state = tracker.update(jelöltek, camera_shift=(dx, dy), grass_mask=..., grass_scale=...,
                               pitch_H=H_kép->pálya (ha megbízható), person_boxes=[(x1,y1,x2,y2), ...],
                               dt_frames=az előző hívás óta eltelt képkockák)

    Téves labda-jelöltek elleni előzetes (prior) szorzók a pontszámon (0..1):
      * pályán / a pálya fölött: a jelölt középpontja a pályára vetítve legfeljebb pitch_margin_m-re
        lehet kint (a magasan szálló labda vetülete messzebb esik, ezért ez csak puha büntetés);
        a horizont fölé vetülő jelölt szinte kizárva,
      * méret: a H-ból várható labdaátmérő (0,22 m) az adott képponton; a doboz hosszabb oldala
        nem lehet sokkal kisebb, a rövidebb oldala nem lehet sokkal nagyobb (az elmosódás csak
        a hosszabb oldalt nyújtja),
      * fej / felsőtest: a játékos-doboz felső ~40%-ában lévő jelölt (gyakori téves: fej) büntetve,
      * álló folt: a kamera-kompenzált képen több képkockán át egy helyben lévő jelölt (folt,
        vonal) büntetve, ha messze van a becsléstől.
    Indulás / újraindulás elvesztés után: legalább init_frames (erős, jó prior) ill.
    init_frames_weak egymást követő, mozgásban következetes detektálás kell (mag-követés).
    """

    def __init__(self, fps=25.0, frame_size=None, max_gap_sec=0.5, max_output_gap_sec=0.24,
                 min_conf=0.15, init_conf=0.35,
                 accel_sigma_px=6.0, meas_sigma_px=2.5, gate_base_px=None, gate_grow_px=None,
                 min_score=0.25, reinit_frames=3, velocity_damping=0.97,
                 w_conf=0.40, w_dist=0.35, w_size=0.15, w_grass=0.10,
                 init_frames=2, init_frames_weak=3, ball_diameter_m=0.22, pitch_margin_m=3.0, air_margin_m=7.0,
                 head_penalty=0.35, static_penalty=0.35, static_frames=3, seed_min_prior=0.4):
        self.fps = float(fps or 25.0)
        self.frame_size = frame_size
        diag = math.hypot(*frame_size) if frame_size else 1468.0
        self.max_gap_frames = max(1, int(round(max_gap_sec * self.fps)))
        self.max_output_gap_frames = min(self.max_gap_frames, max(0, int(round(max_output_gap_sec * self.fps))))
        self.min_conf = float(min_conf)
        self.init_conf = float(init_conf)
        self.q = float(accel_sigma_px) ** 2
        self.r = float(meas_sigma_px) ** 2
        # kapu: alap sugár + kimaradásonként növekvő rész (képátló arányában, 720p-n kb. 35 + 25 px)
        self.gate_base = gate_base_px if gate_base_px is not None else 0.024 * diag
        self.gate_grow = gate_grow_px if gate_grow_px is not None else 0.017 * diag
        self.min_score = float(min_score)
        self.reinit_frames = int(reinit_frames)
        self.damping = float(velocity_damping)
        self.w = (w_conf, w_dist, w_size, w_grass)
        # téves jelöltek elleni priorok + mag-követés (indulás több egymást követő detektálásból)
        self.init_frames = max(1, int(init_frames))
        self.init_frames_weak = max(self.init_frames, int(init_frames_weak))
        self.ball_diameter = float(ball_diameter_m)
        self.pitch_margin = float(pitch_margin_m)
        self.air_margin = float(air_margin_m)
        self.head_penalty = float(head_penalty)
        self.static_penalty = float(static_penalty)
        self.static_frames = int(static_frames)
        self.seed_min_prior = float(seed_min_prior)
        self.reset()

    # ------------------------------------------------------------ állapot
    def reset(self):
        self.x = None              # [x, y, vx, vy]
        self.P = None
        self.gap = 0               # kimaradt képkockák száma (mérés nélkül)
        self.size_ema = None
        self.last_meas = None      # (frame_idx, x, y)
        self.challenger = None     # kapun kívüli, következetes jelölt: {'x','y','n','frame'}
        self.seeds = []            # elveszett állapotban: induló labda-hipotézisek (mag-követés)
        self.hist = []             # az előző hívások jelöltjeinek képpontjai (álló folt felismeréshez)
        self.frame_idx = -1
        self.stats = {"frames": 0, "detected": 0, "predicted": 0, "coasting": 0, "lost": 0, "rejected": 0, "reinit": 0,
                      "multi_candidate_frames": 0, "prior_penalized": 0, "seed_started": 0}

    @property
    def active(self):
        return self.x is not None

    # ------------------------------------------------------------ kamera-kompenzáció
    @staticmethod
    def _warp(pts, camera_shift=None, camera_H=None):
        """[(x, y), ...] eltolása a kamera-mozgással (homográfia vagy eltolás)."""
        if not pts:
            return pts
        if camera_H is not None and cv2 is not None:
            q = cv2.perspectiveTransform(np.array([pts], np.float64), np.asarray(camera_H, np.float64))[0]
            return [(float(a), float(b)) for a, b in q]
        if camera_shift is not None:
            dx, dy = float(camera_shift[0]), float(camera_shift[1])
            return [(x + dx, y + dy) for x, y in pts]
        return pts

    def _compensate(self, camera_shift=None, camera_H=None):
        # mag-hipotézisek és a jelölt-előzmény (álló folt) mindig együtt mozognak a képpel
        if self.seeds:
            moved = self._warp([(s["x"], s["y"]) for s in self.seeds], camera_shift, camera_H)
            moved_v = self._warp([(s["x"] + s["vx"], s["y"] + s["vy"]) for s in self.seeds], camera_shift, camera_H)
            for s, (x, y), (xv, yv) in zip(self.seeds, moved, moved_v):
                s["x"], s["y"], s["vx"], s["vy"] = x, y, xv - x, yv - y
        self.hist = [self._warp(h, camera_shift, camera_H) for h in self.hist]
        if self.x is None:
            return
        if camera_H is not None and cv2 is not None:
            Hm = np.asarray(camera_H, np.float64)
            p = np.array([[[self.x[0], self.x[1]], [self.x[0] + self.x[2], self.x[1] + self.x[3]]]], np.float64)
            q = cv2.perspectiveTransform(p, Hm)[0]
            self.x[0], self.x[1] = q[0]
            self.x[2], self.x[3] = q[1] - q[0]
            if self.last_meas is not None:
                lm = cv2.perspectiveTransform(np.array([[[self.last_meas[1], self.last_meas[2]]]], np.float64), Hm)[0][0]
                self.last_meas = (self.last_meas[0], float(lm[0]), float(lm[1]))
        elif camera_shift is not None:
            dx, dy = float(camera_shift[0]), float(camera_shift[1])
            self.x[0] += dx
            self.x[1] += dy
            if self.last_meas is not None:
                self.last_meas = (self.last_meas[0], self.last_meas[1] + dx, self.last_meas[2] + dy)
        if self.challenger is not None:
            # a kapun kívüli jelöltet is a kamera-mozgással toljuk
            (cx, cy), = self._warp([(self.challenger["x"], self.challenger["y"])], camera_shift, camera_H)
            self.challenger["x"], self.challenger["y"] = cx, cy

    # ------------------------------------------------------------ Kálmán
    def _predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
        q = self.q
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = G @ G.T * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _innovation_cov(self, r):
        return self.P[:2, :2] + np.eye(2) * r

    def _update(self, zx, zy, r):
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float64)
        S = Hm @ self.P @ Hm.T + np.eye(2) * r
        K = self.P @ Hm.T @ np.linalg.inv(S)
        y = np.array([zx, zy]) - Hm @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ Hm) @ self.P

    def _init_track(self, cx, cy, size, vx=0.0, vy=0.0):
        self.x = np.array([cx, cy, vx, vy], np.float64)
        self.P = np.diag([self.r * 4, self.r * 4, 400.0, 400.0])
        self.size_ema = size
        self.gap = 0
        self.challenger = None
        self.seeds = []

    # ------------------------------------------------------------ pontozás
    def _grass_score(self, cx, cy, grass_mask, grass_scale):
        if grass_mask is None:
            return 1.0
        mh, mw = grass_mask.shape[:2]
        gx, gy = int(cx * grass_scale), int(cy * grass_scale)
        if not (0 <= gx < mw and 0 <= gy < mh):
            return 0.0
        # a labda maga nem zöld: a környezetét nézzük (3x3-as szomszédság a csökkentett maszkon)
        patch = grass_mask[max(0, gy - 2):gy + 3, max(0, gx - 2):gx + 3]
        return float(np.count_nonzero(patch)) / max(1, patch.size)

    def _size_score(self, size):
        if self.size_ema is None:
            return 1.0
        return math.exp(-abs(math.log(size / max(self.size_ema, 1e-3))) / 0.5)

    def _score(self, conf, dist_term, size, grass):
        wc, wd, ws, wg = self.w
        return wc * min(1.0, conf / 0.7) + wd * dist_term + ws * self._size_score(size) + wg * grass

    @staticmethod
    def _geometry(pitch_H, frame_size):
        """pitch_H: 3x3 kép -> pálya H, vagy kész PitchGeometry-szerű objektum (to_pitch / pixels_per_meter)."""
        if pitch_H is None:
            return None
        if hasattr(pitch_H, "to_pitch"):
            return pitch_H
        if not frame_size:
            return None
        from pitch_filters import PitchGeometry   # késleltetett import (a követő pitch_filters nélkül is fut)
        return PitchGeometry(pitch_H, frame_size)

    def _priors(self, cands, geom, person_boxes):
        """Jelöltenként: (pálya + méret szorzó, fej/felsőtest-e, hány korábbi hívásban volt ott álló jelölt)."""
        out = []
        for cx, cy, size, conf, box in cands:
            f = 1.0
            bw, bh = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
            if geom is not None:
                xy = geom.to_pitch(cx, cy)
                if xy is None:
                    f *= 0.1                                    # a horizont fölé vetül (lelátó, ég)
                else:
                    # a szálló labda talajra vetülete a kamerától TÁVOLABB esik (a kapuk / a túlsó
                    # oldalvonal mögé), ezért arra nagyobb a tűrés; a közeli oldalvonalon túl nem
                    d_away = math.hypot(max(0.0, -xy[0], xy[0] - 105.0), max(0.0, -xy[1]))
                    d_near = max(0.0, xy[1] - 68.0)
                    if d_near > self.pitch_margin:
                        f *= max(0.15, math.exp(-(d_near - self.pitch_margin) / 6.0))
                    if d_away > self.air_margin:
                        f *= max(0.15, math.exp(-(d_away - self.air_margin) / 6.0))
                    ppm = geom.pixels_per_meter(xy[0], xy[1])
                    if ppm:
                        D = self.ball_diameter * ppm
                        lo, hi = max(bw, bh) / D, min(bw, bh) / D
                        if lo < 0.8:
                            f *= math.exp(-(0.8 - lo) / 0.15)    # túl kicsi (fej / cipő / zaj)
                        elif hi > 3.5:
                            f *= math.exp(-(hi - 3.5))           # túl nagy (a szálló labda nagyobb lehet)
            head = False
            for px1, py1, px2, py2 in person_boxes or ():
                pw, ph = px2 - px1, py2 - py1
                if ph <= 0:
                    continue
                if px1 - 0.1 * pw <= cx <= px2 + 0.1 * pw and py1 - 0.08 * ph <= cy <= py1 + 0.42 * ph:
                    head = True
                    break
            rad = max(1.5, 0.35 * size)
            n_static = sum(1 for h in self.hist if any(math.hypot(cx - x, cy - y) <= rad for x, y in h))
            out.append((f, head, n_static))
        return out

    def _seed_factor(self, prior):
        f, head, n_static = prior
        if head:
            f *= self.head_penalty
        if n_static >= self.static_frames:
            f *= self.static_penalty
        return f

    # ------------------------------------------------------------ fő lépés
    def update(self, candidates, camera_shift=None, camera_H=None, grass_mask=None, grass_scale=1.0,
               frame_idx=None, dt_frames=1, pitch_H=None, person_boxes=None):
        """
        candidates: [{'xyxy': (x1,y1,x2,y2), 'conf': c}, ...] – a labda osztály jelöltjei ezen a képkockán
        camera_shift: (dx, dy) képpont – a képtartalom elmozdulása az előző hívás óta (pl. CameraMotion)
        camera_H: 3x3 homográfia (előző kép -> aktuális kép); ha meg van adva, ez az erősebb
        dt_frames: hány képkocka telt el az előző hívás óta (gyors mód: detect_every_n)
        pitch_H: a képkocka kép -> pálya H-ja (csak ha megbízható!) – pálya- és méret-prior
        person_boxes: játékos / bíró dobozok – fej / felsőtest-prior
        """
        dt_frames = max(1, int(dt_frames or 1))
        self.frame_idx = self.frame_idx + dt_frames if frame_idx is None else frame_idx
        self.stats["frames"] += 1
        cands = [_cand_center_size(c) for c in (candidates or [])]
        cands = [c for c in cands if c[3] >= self.min_conf]
        if len(cands) > 1:
            self.stats["multi_candidate_frames"] += 1

        self._compensate(camera_shift, camera_H)
        priors = self._priors(cands, self._geometry(pitch_H, self.frame_size), person_boxes)
        try:
            if self.x is None:
                return self._try_init(cands, priors, grass_mask, grass_scale, dt_frames)
            return self._step(cands, priors, grass_mask, grass_scale, dt_frames)
        finally:
            self.hist = (self.hist + [[(c[0], c[1]) for c in cands]])[-(self.static_frames + 1):]

    def _step(self, cands, priors, grass_mask, grass_scale, dt_frames):
        self._predict(float(dt_frames))
        gap_after = self.gap + dt_frames
        gate = self.gate_base + self.gate_grow * min(gap_after, self.max_gap_frames)
        px, py = self.x[0], self.x[1]
        best, best_score = None, -1.0
        outside = []
        for c, pr in zip(cands, priors):
            cx, cy, size, conf, box = c
            d = math.hypot(cx - px, cy - py)
            # csak a (kimaradással növő) kapu számít: korábban a Mahalanobis-feltétel (m2 <= 16) hosszú
            # kimaradás után a nagy kovariancia miatt a kapun kívüli, messzi téves jelöltet is beengedte
            if d > gate:
                outside.append((c, pr))
                continue
            dist_term = math.exp(-0.5 * (d / (0.5 * gate)) ** 2)
            f, head, n_static = pr
            if head:
                f *= 0.6 if dist_term > 0.6 else self.head_penalty   # a becslés mellett (fejelés) enyhébb
            if n_static >= self.static_frames and dist_term < 0.3:
                f *= self.static_penalty                             # álló folt a becsléstől messze
            if f < 0.999:
                self.stats["prior_penalized"] += 1
            sc = f * self._score(conf, dist_term, size, self._grass_score(cx, cy, grass_mask, grass_scale))
            if sc > best_score:
                best, best_score = c, sc
        self.stats["rejected"] += len(outside)

        if best is not None and best_score >= self.min_score:
            cx, cy, size, conf, box = best
            d = math.hypot(cx - px, cy - py)
            if d > 0.5 * gate and self.last_meas is not None:
                # hirtelen irányváltás (rúgás): a sebességet az utolsó mérésből újrabecsüljük
                fr = max(1, self.frame_idx - self.last_meas[0])
                self.x[2] = (cx - self.last_meas[1]) / fr
                self.x[3] = (cy - self.last_meas[2]) / fr
                self.P[2:, 2:] = np.eye(2) * 100.0
                self.P[:2, :2] += np.eye(2) * d
            r = self.r * (1.0 + max(0.0, size - 6.0) / 6.0)
            self._update(cx, cy, r)
            self.size_ema = size if self.size_ema is None else 0.8 * self.size_ema + 0.2 * size
            self.gap = 0
            self.last_meas = (self.frame_idx, cx, cy)
            self.challenger = None
            self.stats["detected"] += 1
            return BallState(self.frame_idx, STATUS_DETECTED, float(self.x[0]), float(self.x[1]),
                             float(self.x[2]), float(self.x[3]), conf, box, best_score, 0, len(cands))

        # nincs elfogadható mérés: előrejelzés, közben figyeljük a kapun kívüli következetes jelöltet
        self.gap = gap_after
        if self._challenge(outside):
            return self._emit_reinit(len(cands))
        self.x[2] *= self.damping ** dt_frames
        self.x[3] *= self.damping ** dt_frames
        if self.gap > self.max_gap_frames or not self._in_frame(self.x[0], self.x[1]):
            self.x = None
            self.P = None
            self.stats["lost"] += 1
            return BallState(self.frame_idx, STATUS_LOST, gap_frames=self.gap, n_candidates=len(cands))
        status = STATUS_PREDICTED if self.gap <= self.max_output_gap_frames else STATUS_COASTING
        self.stats["predicted" if status == STATUS_PREDICTED else "coasting"] += 1
        return BallState(self.frame_idx, status, float(self.x[0]), float(self.x[1]),
                         float(self.x[2]), float(self.x[3]), 0.0, None, 0.0, self.gap, len(cands))

    def _in_frame(self, x, y):
        if not self.frame_size:
            return True
        w, h = self.frame_size
        m = 0.05 * w
        return -m <= x <= w + m and -m <= y <= h + m

    def _challenge(self, outside):
        """Kapun kívüli, erős, jó priorú jelölt, ami több képkockán át folytonosan jelen van -> átváltás."""
        strong = [c for c, pr in outside if c[3] >= self.init_conf and self._seed_factor(pr) >= 0.5]
        if not strong:
            if self.challenger is not None and self.frame_idx - self.challenger["frame"] > 2:
                self.challenger = None
            return False
        ch = self.challenger
        if ch is not None:
            near = min(strong, key=lambda c: math.hypot(c[0] - ch["x"], c[1] - ch["y"]))
            if math.hypot(near[0] - ch["x"], near[1] - ch["y"]) <= self.gate_base + self.gate_grow * 2:
                ch.update({"x": near[0], "y": near[1], "n": ch["n"] + 1, "frame": self.frame_idx, "c": near})
                return ch["n"] >= self.reinit_frames
        best = max(strong, key=lambda c: c[3])
        self.challenger = {"x": best[0], "y": best[1], "n": 1, "frame": self.frame_idx, "c": best}
        return False

    def _emit_reinit(self, n_cands):
        cx, cy, size, conf, box = self.challenger["c"]
        self._init_track(cx, cy, size)
        self.last_meas = (self.frame_idx, cx, cy)
        self.stats["reinit"] += 1
        self.stats["detected"] += 1
        return BallState(self.frame_idx, STATUS_DETECTED, cx, cy, 0.0, 0.0, conf, box, conf, 0, n_cands)

    def _try_init(self, cands, priors, grass_mask, grass_scale, dt_frames=1):
        """
        Elveszett állapot – mag-követés: minden jó priorú jelöltből induló hipotézis (mag) lesz, amit
        a következő detektálások mozgásban következetesen folytatnak. A követés akkor indul, ha egy
        mag elég hosszú: init_frames egymást követő detektálás (átlagosan erős konfidencia és jó
        prior mellett), különben init_frames_weak. Egy detektálásnyi kimaradás megengedett.
        """
        max_seed_gap = max(2, 2 * dt_frames)
        self.seeds = [s for s in self.seeds if self.frame_idx - s["frame"] <= max_seed_gap]
        usable = [(c, self._seed_factor(pr)) for c, pr in zip(cands, priors)]
        usable = [(c, f) for c, f in usable if f >= self.seed_min_prior]
        used = set()
        for s in sorted(self.seeds, key=lambda s: -s["n"]):
            gap = max(1, self.frame_idx - s["frame"])
            if s["n"] >= 2:
                px, py = s["x"] + s["vx"] * gap, s["y"] + s["vy"] * gap
                rad = self.gate_base * (0.5 + 0.25 * (gap - 1))
            else:   # a sebesség még ismeretlen: a labda egy képkocka alatt kb. a kapu-sugárnyit mozdulhat
                px, py = s["x"], s["y"]
                rad = self.gate_base * gap
            bi, bd = None, None
            for k, (c, f) in enumerate(usable):
                if k in used:
                    continue
                d = math.hypot(c[0] - px, c[1] - py)
                if d <= rad and (bd is None or d < bd):
                    bi, bd = k, d
            if bi is None:
                continue
            used.add(bi)
            c, f = usable[bi]
            vx, vy = (c[0] - s["x"]) / gap, (c[1] - s["y"]) / gap
            if s["n"] >= 2:
                vx, vy = 0.5 * (vx + s["vx"]), 0.5 * (vy + s["vy"])
            s.update({"x": c[0], "y": c[1], "vx": vx, "vy": vy, "n": s["n"] + 1, "frame": self.frame_idx,
                      "conf": s["conf"] + c[3], "fmin": min(s["fmin"], f), "c": c, "f": f})
        for k, (c, f) in enumerate(usable):
            if k not in used:
                self.seeds.append({"x": c[0], "y": c[1], "vx": 0.0, "vy": 0.0, "n": 1, "frame": self.frame_idx,
                                   "conf": c[3], "fmin": f, "c": c, "f": f})
        self.seeds = sorted(self.seeds, key=lambda s: (-s["n"], -s["conf"]))[:16]

        ready = []
        for s in self.seeds:
            if s["frame"] != self.frame_idx:
                continue
            strong = s["conf"] / s["n"] >= self.init_conf and s["fmin"] >= 0.6
            if s["n"] >= (self.init_frames if strong else self.init_frames_weak):
                ready.append(s)
        if not ready:
            self.stats["lost"] += 1
            return BallState(self.frame_idx, STATUS_LOST, n_candidates=len(cands))
        s = max(ready, key=lambda s: (s["n"], s["conf"] * s["fmin"]))
        cx, cy, size, conf, box = s["c"]
        vx, vy = s["vx"], s["vy"]
        self._init_track(cx, cy, size, vx, vy)
        self.last_meas = (self.frame_idx, cx, cy)
        self.stats["detected"] += 1
        self.stats["seed_started"] += 1
        grass = self._grass_score(cx, cy, grass_mask, grass_scale)
        sc = s["f"] * self._score(conf, 1.0, size, grass)
        return BallState(self.frame_idx, STATUS_DETECTED, cx, cy, vx, vy, conf, box, sc, 0, len(cands))

    # ------------------------------------------------------------ segédek
    def summary(self):
        s = self.stats
        n = max(1, s["frames"])
        return (f"Labdakövető: {s['frames']} képkocka, detektált {s['detected']} ({100 * s['detected'] / n:.1f}%), "
                f"becsült {s['predicted']} ({100 * s['predicted'] / n:.1f}%), "
                f"nincs labda {s['lost'] + s['coasting']} (ebből belső követés {s['coasting']}), "
                f"kidobott jelölt {s['rejected']}, újraindítás {s['reinit']}, "
                f"több jelöltes képkocka {s['multi_candidate_frames']}, "
                f"prior-büntetés {s['prior_penalized']}, mag-követésből indult {s['seed_started']}")


def pitch_position(state, H):
    """A labda képpontja -> pálya (m) a kép->pálya homográfiával; None, ha nincs labda vagy H."""
    if state is None or not state.found or H is None or cv2 is None:
        return None
    pt = np.array([[[state.x, state.y]]], np.float64)
    q = cv2.perspectiveTransform(pt, np.asarray(H, np.float64))[0][0]
    return float(q[0]), float(q[1])


def interpolate_gaps(points, max_gap):
    """
    Utólagos (offline) hézagkitöltés: points = [(x, y) vagy None, ...] képkockánként.
    A legfeljebb max_gap hosszú None-sorozatokat két mért pont között lineárisan kitölti.
    Visszatér: (pontok, jelzők) ahol a jelző 'measured' / 'interpolated' / None.
    """
    out = list(points)
    flags = ["measured" if p is not None else None for p in points]
    last = None
    for i, p in enumerate(points):
        if p is None:
            continue
        if last is not None and 1 < i - last <= max_gap + 1:
            x0, y0 = points[last]
            x1, y1 = p
            n = i - last
            for k in range(1, n):
                a = k / n
                out[last + k] = (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
                flags[last + k] = "interpolated"
        last = i
    return out, flags


def draw_ball_state(frame, state, color=(0, 255, 255)):
    """Detektált labda: teli kör; előrejelzett (becsült) labda: üres kör."""
    if state is None or not state.found or cv2 is None:
        return
    p = state.pixel
    if state.predicted:
        cv2.circle(frame, p, 7, (0, 0, 0), 3)
        cv2.circle(frame, p, 7, color, 1)
    else:
        cv2.circle(frame, p, 5, color, -1)

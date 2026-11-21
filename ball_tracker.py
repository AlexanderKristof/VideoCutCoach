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
  * opcionálisan pályakoordinátát számol a kép -> pálya homográfiával.

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
        state = tracker.update(jelöltek, camera_shift=(dx, dy), grass_mask=..., grass_scale=...)
    """

    def __init__(self, fps=25.0, frame_size=None, max_gap_sec=0.5, max_output_gap_sec=0.24,
                 min_conf=0.15, init_conf=0.35,
                 accel_sigma_px=6.0, meas_sigma_px=2.5, gate_base_px=None, gate_grow_px=None,
                 min_score=0.25, reinit_frames=3, velocity_damping=0.97,
                 w_conf=0.40, w_dist=0.35, w_size=0.15, w_grass=0.10):
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
        self.reset()

    # ------------------------------------------------------------ állapot
    def reset(self):
        self.x = None              # [x, y, vx, vy]
        self.P = None
        self.gap = 0               # kimaradt képkockák száma (mérés nélkül)
        self.size_ema = None
        self.last_meas = None      # (frame_idx, x, y)
        self.challenger = None     # kapun kívüli, következetes jelölt: {'x','y','n','frame'}
        self.pending = None        # elveszett állapotban: előző képkocka gyenge jelöltjei
        self.frame_idx = -1
        self.stats = {"frames": 0, "detected": 0, "predicted": 0, "coasting": 0, "lost": 0, "rejected": 0, "reinit": 0,
                      "multi_candidate_frames": 0}

    @property
    def active(self):
        return self.x is not None

    # ------------------------------------------------------------ kamera-kompenzáció
    def _compensate(self, camera_shift=None, camera_H=None):
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
        if self.challenger is not None and camera_H is not None and cv2 is not None:
            # a kapun kívüli jelöltet is a kamera-homográfiával toljuk (eddig csak eltolásnál mozdult)
            c = cv2.perspectiveTransform(np.array([[[self.challenger["x"], self.challenger["y"]]]], np.float64),
                                         np.asarray(camera_H, np.float64))[0][0]
            self.challenger["x"], self.challenger["y"] = float(c[0]), float(c[1])
        elif self.challenger is not None and camera_shift is not None:
            self.challenger["x"] += float(camera_shift[0])
            self.challenger["y"] += float(camera_shift[1])

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

    # ------------------------------------------------------------ fő lépés
    def update(self, candidates, camera_shift=None, camera_H=None, grass_mask=None, grass_scale=1.0,
               frame_idx=None, dt_frames=1):
        """
        candidates: [{'xyxy': (x1,y1,x2,y2), 'conf': c}, ...] – a labda osztály jelöltjei ezen a képkockán
        camera_shift: (dx, dy) képpont – a képtartalom elmozdulása az előző hívás óta (pl. CameraMotion)
        camera_H: 3x3 homográfia (előző kép -> aktuális kép); ha meg van adva, ez az erősebb
        dt_frames: hány képkocka telt el az előző hívás óta (gyors mód: detect_every_n)
        """
        self.frame_idx = self.frame_idx + dt_frames if frame_idx is None else frame_idx
        self.stats["frames"] += 1
        cands = [_cand_center_size(c) for c in (candidates or [])]
        cands = [c for c in cands if c[3] >= self.min_conf]
        if len(cands) > 1:
            self.stats["multi_candidate_frames"] += 1

        self._compensate(camera_shift, camera_H)

        if self.x is None:
            return self._try_init(cands, grass_mask, grass_scale, camera_shift, camera_H)

        self._predict(float(dt_frames))
        gap_after = self.gap + dt_frames
        gate = self.gate_base + self.gate_grow * min(gap_after, self.max_gap_frames)
        S = self._innovation_cov(self.r)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.eye(2) / max(self.r, 1.0)

        px, py = self.x[0], self.x[1]
        best, best_score = None, -1.0
        outside = []
        for c in cands:
            cx, cy, size, conf, box = c
            dvec = np.array([cx - px, cy - py])
            d = float(np.hypot(*dvec))
            m2 = float(dvec @ S_inv @ dvec)
            if d > gate and m2 > 16.0:
                outside.append(c)
                continue
            dist_term = math.exp(-0.5 * (d / (0.5 * gate)) ** 2)
            sc = self._score(conf, dist_term, size, self._grass_score(cx, cy, grass_mask, grass_scale))
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
        """Kapun kívüli, erős jelölt, ami több képkockán át egy helyen / folytonosan jelen van -> átváltás."""
        strong = [c for c in outside if c[3] >= self.init_conf]
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

    def _try_init(self, cands, grass_mask, grass_scale, camera_shift, camera_H=None):
        """Elveszett állapot: erős jelöltből azonnal, gyengébből két egymást követő, közeli jelöltből indulunk."""
        pending = self.pending
        if pending and camera_H is not None and cv2 is not None:
            q = cv2.perspectiveTransform(np.array([[(c[0], c[1]) for c in pending]], np.float64),
                                         np.asarray(camera_H, np.float64))[0]
            pending = [(float(q[k][0]), float(q[k][1])) + tuple(c[2:]) for k, c in enumerate(pending)]
        elif pending is not None and camera_shift is not None:
            pending = [(c[0] + camera_shift[0], c[1] + camera_shift[1]) + tuple(c[2:]) for c in pending]
        best, best_score = None, -1.0
        for c in cands:
            cx, cy, size, conf, box = c
            grass = self._grass_score(cx, cy, grass_mask, grass_scale)
            confirm = 0.0
            if pending:
                dmin = min(math.hypot(cx - p[0], cy - p[1]) for p in pending)
                confirm = math.exp(-0.5 * (dmin / self.gate_base) ** 2)
            if conf < self.init_conf and confirm < 0.3:
                continue
            sc = self._score(conf, confirm, size, grass)
            if sc > best_score:
                best, best_score = c, sc
        self.pending = cands
        if best is None:
            self.stats["lost"] += 1
            return BallState(self.frame_idx, STATUS_LOST, n_candidates=len(cands))
        cx, cy, size, conf, box = best
        vx = vy = 0.0
        if pending:
            p = min(pending, key=lambda p: math.hypot(cx - p[0], cy - p[1]))
            if math.hypot(cx - p[0], cy - p[1]) < self.gate_base * 2:
                vx, vy = cx - p[0], cy - p[1]
        self._init_track(cx, cy, size, vx, vy)
        self.last_meas = (self.frame_idx, cx, cy)
        self.pending = None
        self.stats["detected"] += 1
        return BallState(self.frame_idx, STATUS_DETECTED, cx, cy, vx, vy, conf, box, best_score, 0, len(cands))

    # ------------------------------------------------------------ segédek
    def summary(self):
        s = self.stats
        n = max(1, s["frames"])
        return (f"Labdakövető: {s['frames']} képkocka, detektált {s['detected']} ({100 * s['detected'] / n:.1f}%), "
                f"becsült {s['predicted']} ({100 * s['predicted'] / n:.1f}%), "
                f"nincs labda {s['lost'] + s['coasting']} (ebből belső követés {s['coasting']}), "
                f"kidobott jelölt {s['rejected']}, újraindítás {s['reinit']}, "
                f"több jelöltes képkocka {s['multi_candidate_frames']}")


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

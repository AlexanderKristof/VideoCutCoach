"""
Saját többobjektumos követő (ByteTrack-szerű algoritmus, SAJÁT, nulláról írt kód).

Az algoritmus ötlete a ByteTrack cikkből (Zhang et al., 2022, "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box") és annak MIT licencű referencia-implementációjából ismert; ez a
modul nem tartalmaz onnan (vagy az AGPL-3.0 ultralytics-ből) átmásolt kódot.

Lépések képkockánként:
  1. minden követés Kálmán-előrejelzése (állapot: középpont x, y, oldalarány a = w/h, magasság h
     és ezek sebessége), opcionálisan a kamera-mozgás kompenzálásával (3x3 képtér-homográfia:
     előző kép -> aktuális kép, pl. a PitchTracker.frame_motion()-ből);
  2. ERŐS detektálások (conf >= high_thresh) párosítása az aktív + elveszett követésekkel
     (költség: 1 - IoU, opcionálisan a detektálás pontszámával súlyozva), Magyar-módszerrel;
  3. GYENGE detektálások (low_thresh <= conf < high_thresh) párosítása a még párosítatlan, az
     előző képkockán aktív követésekkel (szigorúbb IoU) – így a takarásban elhalványuló alak
     sem veszik el;
  4. a megerősítetlen (előző képkockán született) követések párosítása a maradék erős
     detektálásokkal; ami nem párosul, törlődik;
  5. új követés a maradék erős detektálásokból (conf >= new_track_thresh) – csak a következő
     párosításnál kerül a kimenetbe (az első képkockát kivéve);
  6. az elveszett követés track_buffer képkockáig (frame_rate szerint skálázva) visszatérhet
     ugyanazzal az azonosítóval.

A labdát NEM ez követi: a labda minden nyers jelöltje a BallTracker-be megy (ball_tracker.py).
Csak numpy kell hozzá (a scipy linear_sum_assignment-et használja, ha telepítve van).
"""
import numpy as np

from .detector_base import box_iou

STATE_NEW, STATE_TRACKED, STATE_LOST, STATE_REMOVED = 0, 1, 2, 3


# ====================================================================== párosítás
def _hungarian_min(cost):
    """Minimális költségű hozzárendelés (sorok <= oszlopok), potenciálos Magyar-módszer, O(n^2 m)."""
    n, m = cost.shape
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, np.int64)     # p[j] = az oszlophoz rendelt sor (1-től), 0 = szabad
    way = np.zeros(m + 1, np.int64)
    a = np.zeros((n + 1, m + 1))
    a[1:, 1:] = cost
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used
            free[0] = False
            cur = a[i0] - u[i0] - v
            upd = free & (cur < minv)
            minv[upd] = cur[upd]
            way[upd] = j0
            cand = np.where(free, minv, INF)
            j1 = int(np.argmin(cand))
            delta = cand[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] > 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


try:  # a scipy (BSD) gyorsabb; ha nincs, a saját megvalósítás fut
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover - környezetfüggő
    _lsa = None


def linear_assignment(cost, thresh, use_scipy=True):
    """
    Hozzárendelés küszöbbel: a thresh-nél drágább párok nem jöhetnek létre.
    Visszatér: (párok [(sor, oszlop)], párosítatlan sorok, párosítatlan oszlopok).
    """
    cost = np.asarray(cost, np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    big = thresh + 1e4
    c = np.where(cost > thresh, big, cost)
    if use_scipy and _lsa is not None:
        r, k = _lsa(c)
    elif n <= m:
        r, k = _hungarian_min(c)
    else:
        k, r = _hungarian_min(c.T)
    pairs = [(int(i), int(j)) for i, j in zip(r, k) if cost[i, j] <= thresh]
    mr = {i for i, _ in pairs}
    mc = {j for _, j in pairs}
    return pairs, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


# ====================================================================== Kálmán-szűrő
class KalmanXYAH:
    """Állandó sebességű Kálmán-szűrő (cx, cy, a, h, vcx, vcy, va, vh); a zaj a doboz magasságával arányos."""

    def __init__(self, std_pos=1.0 / 20, std_vel=1.0 / 160):
        self.std_pos, self.std_vel = std_pos, std_vel
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.Hm = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
               self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        S = self.Hm @ cov @ self.Hm.T + R
        K = np.linalg.solve(S, (cov @ self.Hm.T).T).T
        innov = z - self.Hm @ mean
        mean = mean + K @ innov
        cov = cov - K @ S @ K.T
        return mean, cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], max(1e-3, b[3] - b[1])
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w / h, h], np.float64)


def xyah_to_xyxy(m):
    w = m[2] * m[3]
    return np.array([m[0] - w / 2, m[1] - m[3] / 2, m[0] + w / 2, m[1] + m[3] / 2], np.float64)


def _apply_h(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def warp_state(mean, cov, H):
    """
    Kamera-mozgás kompenzáció: a követés állapotát (előző kép koordinátái) az aktuális képre viszi
    a H homográfiával. A középpontot vetítjük, a w / h-t a doboz oldalfelező pontjainak képéből
    számoljuk, a sebességet és a kovarianciát a középpontbeli Jacobi-mátrixszal forgatjuk / skálázzuk.
    """
    cx, cy, a, h = mean[:4]
    w = a * h
    pts = np.array([[cx, cy], [cx, cy - h / 2], [cx, cy + h / 2], [cx - w / 2, cy], [cx + w / 2, cy],
                    [cx + 1.0, cy], [cx, cy + 1.0]])
    q = _apply_h(H, pts)
    if not np.all(np.isfinite(q)):
        return mean, cov
    nh = float(np.linalg.norm(q[2] - q[1]))
    nw = float(np.linalg.norm(q[4] - q[3]))
    if nh <= 1e-3 or nw <= 1e-3:
        return mean, cov
    J = np.stack([q[5] - q[0], q[6] - q[0]], 1)   # 2x2: d(kép)/d(előző kép) a középpontban
    s = nh / max(h, 1e-6)
    new = mean.copy()
    new[0], new[1] = q[0]
    new[3] = nh
    new[2] = nw / nh
    new[4:6] = J @ mean[4:6]
    new[7] = mean[7] * s
    T = np.eye(8)
    T[0:2, 0:2] = J
    T[3, 3] = s
    T[4:6, 4:6] = J
    T[7, 7] = s
    return new, T @ cov @ T.T


# ====================================================================== követés
class Track:
    __slots__ = ("id", "mean", "cov", "score", "cls", "state", "activated", "frame_id", "start_frame",
                 "hits", "det_box", "det_index")

    def __init__(self, box, score, cls, kf, frame_id):
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        self.id = 0
        self.score = float(score)
        self.cls = int(cls)
        self.state = STATE_NEW
        self.activated = False
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.hits = 1
        self.det_box = np.asarray(box, np.float64)
        self.det_index = -1

    @property
    def xyxy(self):
        return xyah_to_xyxy(self.mean)

    def __repr__(self):
        return f"Track(id={self.id}, state={self.state}, box={np.round(self.xyxy, 1).tolist()})"


class ByteTracker:
    """
    Paraméterek (az ultralytics bytetrack.yaml alapértékeihez igazítva, hogy a viselkedés hasonló legyen):
      high_thresh 0.25, low_thresh 0.1, new_track_thresh 0.25, match_thresh 0.8 (költség, azaz IoU >= 0.2),
      low_match_thresh 0.5, unconfirmed_thresh 0.7, track_buffer 30 (képkocka @ 30 FPS), fuse_score,
      class_aware: False – osztályfüggetlen párosítás (a kapus / játékos osztálytévesztés ne törjön azonosítót),
      output: "kalman" (simított doboz, mint a ByteTrack) vagy "detection" (a párosított detektálás doboza).
    """

    def __init__(self, high_thresh=0.25, low_thresh=0.1, new_track_thresh=0.25, match_thresh=0.8,
                 low_match_thresh=0.5, unconfirmed_thresh=0.7, track_buffer=30, frame_rate=30,
                 fuse_score=True, class_aware=False, output="kalman", min_box_area=0.0):
        self.high_thresh, self.low_thresh, self.new_track_thresh = high_thresh, low_thresh, new_track_thresh
        self.match_thresh, self.low_match_thresh, self.unconfirmed_thresh = match_thresh, low_match_thresh, unconfirmed_thresh
        self.max_time_lost = max(1, int(round(frame_rate / 30.0 * track_buffer)))
        self.fuse_score, self.class_aware, self.output = fuse_score, class_aware, output
        self.min_box_area = float(min_box_area)
        self.kf = KalmanXYAH()
        self.reset()

    def reset(self):
        self.tracked, self.lost = [], []
        self.frame_id = 0
        self._next_id = 1
        self.stats = {"új": 0, "visszatért": 0, "gyenge párosítás": 0, "törölt": 0}

    # ------------------------------------------------------------------ segédek
    def _cost(self, tracks, boxes, scores, classes, fuse):
        if not tracks or len(boxes) == 0:
            return np.zeros((len(tracks), len(boxes)))
        tb = np.array([t.xyxy for t in tracks])
        iou = box_iou(tb, boxes).astype(np.float64)
        if fuse:
            iou = iou * scores[None, :]
        cost = 1.0 - iou
        if self.class_aware:
            tc = np.array([t.cls for t in tracks])
            cost[tc[:, None] != classes[None, :]] = 1.0 + 1e-3
        return cost

    def _activate(self, t):
        t.id = self._next_id
        self._next_id += 1
        self.stats["új"] += 1

    def _update_track(self, t, box, score, cls, idx):
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(box))
        t.score, t.cls = float(score), int(cls)
        t.det_box = np.asarray(box, np.float64)
        t.det_index = int(idx)
        t.frame_id = self.frame_id
        t.hits += 1
        if t.state == STATE_LOST:
            self.stats["visszatért"] += 1
        t.state = STATE_TRACKED
        t.activated = True

    # ------------------------------------------------------------------ fő lépés
    def update(self, xyxy, conf, cls=None, motion=None):
        """
        xyxy (N,4), conf (N,), cls (N,) – a képkocka (labda NÉLKÜLI) detektálásai.
        motion: 3x3 homográfia az ELŐZŐ update óta (előző kép -> aktuális kép) vagy None.
        Visszatér: dict numpy tömbökkel: xyxy, conf, cls, id, det_index – a ezen a képkockán
        párosított, megerősített követések (mint az ultralytics model.track() kimenete).
        """
        self.frame_id += 1
        xyxy = np.asarray(xyxy, np.float64).reshape(-1, 4)
        conf = np.asarray(conf, np.float64).reshape(-1)
        cls = np.zeros(len(xyxy), np.int64) if cls is None else np.asarray(cls, np.int64).reshape(-1)

        hi_idx = np.where(conf >= self.high_thresh)[0]
        lo_idx = np.where((conf >= self.low_thresh) & (conf < self.high_thresh))[0]

        unconfirmed = [t for t in self.tracked if not t.activated]
        active = [t for t in self.tracked if t.activated]
        pool = active + self.lost

        H = None
        if motion is not None:
            H = np.asarray(motion, np.float64).reshape(3, 3)
            if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
                H = None
        for t in pool + unconfirmed:
            if t.state != STATE_TRACKED:
                t.mean[7] = 0.0       # elveszett követés magassága ne "fusson el"
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            if H is not None:
                t.mean, t.cov = warp_state(t.mean, t.cov, H)
            t.det_index = -1

        # 1. erős detektálások <-> aktív + elveszett követések
        hb, hs, hc = xyxy[hi_idx], conf[hi_idx], cls[hi_idx]
        cost = self._cost(pool, hb, hs, hc, self.fuse_score)
        pairs, rest_t, rest_d = linear_assignment(cost, self.match_thresh)
        for ti, di in pairs:
            self._update_track(pool[ti], hb[di], hs[di], hc[di], hi_idx[di])

        # 2. gyenge detektálások <-> a maradék, előző képkockán még aktív követések
        r_tracks = [pool[i] for i in rest_t if pool[i].state == STATE_TRACKED]
        lb, ls, lc = xyxy[lo_idx], conf[lo_idx], cls[lo_idx]
        cost = self._cost(r_tracks, lb, ls, lc, False)
        pairs2, rest_t2, _ = linear_assignment(cost, self.low_match_thresh)
        for ti, di in pairs2:
            self._update_track(r_tracks[ti], lb[di], ls[di], lc[di], lo_idx[di])
            self.stats["gyenge párosítás"] += 1
        for i in rest_t2:
            r_tracks[i].state = STATE_LOST

        # 3. megerősítetlen követések <-> a maradék erős detektálások
        rem = np.asarray(rest_d, np.int64)
        cost = self._cost(unconfirmed, hb[rem], hs[rem], hc[rem], self.fuse_score)
        pairs3, rest_u, rest_d3 = linear_assignment(cost, self.unconfirmed_thresh)
        for ti, di in pairs3:
            t = unconfirmed[ti]
            self._update_track(t, hb[rem[di]], hs[rem[di]], hc[rem[di]], hi_idx[rem[di]])
            self._activate(t)
        for i in rest_u:
            unconfirmed[i].state = STATE_REMOVED

        # 4. új követések
        new_tracks = []
        for k in rest_d3:
            di = rem[k]
            if hs[di] < self.new_track_thresh:
                continue
            b = hb[di]
            if (b[2] - b[0]) * (b[3] - b[1]) < self.min_box_area:
                continue
            t = Track(b, hs[di], hc[di], self.kf, self.frame_id)
            t.state = STATE_TRACKED
            t.det_index = int(hi_idx[di])
            if self.frame_id == 1:
                t.activated = True
                self._activate(t)
            new_tracks.append(t)

        # 5. listák karbantartása
        tracked = [t for t in self.tracked + self.lost + new_tracks if t.state == STATE_TRACKED]
        lost = [t for t in self.tracked + self.lost if t.state == STATE_LOST]
        keep_lost = []
        for t in lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.state = STATE_REMOVED
                self.stats["törölt"] += 1
            else:
                keep_lost.append(t)
        tracked, keep_lost = self._dedupe(tracked, keep_lost)
        self.tracked, self.lost = tracked, keep_lost

        out = [t for t in self.tracked if t.activated and t.frame_id == self.frame_id]
        boxes = np.array([t.xyxy if self.output == "kalman" else t.det_box for t in out], np.float32).reshape(-1, 4)
        return {
            "xyxy": boxes,
            "conf": np.array([t.score for t in out], np.float32),
            "cls": np.array([t.cls for t in out], np.int64),
            "id": np.array([t.id for t in out], np.int64),
            "det_index": np.array([t.det_index for t in out], np.int64),
        }

    def _dedupe(self, tracked, lost):
        """Az aktív és az elveszett lista közti majdnem azonos (IoU > 0.85) párokból a rövidebb életű kiesik."""
        act = [t for t in tracked if t.activated]
        if not act or not lost:
            return tracked, lost
        iou = box_iou(np.array([t.xyxy for t in act]), np.array([t.xyxy for t in lost]))
        drop_t, drop_l = set(), set()
        for i, j in zip(*np.where(iou > 0.85)):
            a, b = act[i], lost[j]
            if a.frame_id - a.start_frame >= b.frame_id - b.start_frame:
                drop_l.add(j)
            else:
                drop_t.add(id(a))
        return ([t for t in tracked if id(t) not in drop_t],
                [t for k, t in enumerate(lost) if k not in drop_l])

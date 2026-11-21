"""
Ál-YOLO a végponttól végpontig tartó tesztekhez (torch / ultralytics nélkül).

A synth_match.py ground-truth JSON-jából állít elő "detektálásokat" valószerű hibákkal:
  * dobozremegés (a szélek véletlen elmozdulása a doboz méretéhez képest),
  * kihagyott detektálás: játékosnál ~10% (csomósan, takarásnál több), labdánál ~35%
    (takarásnál / elmosódásnál / nagyon kicsi labdánál több),
  * téves pozitívok: nézők / edzők / cserék 'player'-ként, labda-tévesztés a játékosok lábánál,
    a fehér vonalakon és a nézőknél,
  * osztálytévesztés (kapus -> játékos, bíró -> játékos),
  * a detektálás azonosító nélküli; a track() egy egyszerű, ByteTrack-szerű IoU követővel ad
    azonosítót – és az ultralytics-hez hasonlóan a nem követett (meg nem erősített) dobozokat
    a track() eredményéből elhagyja (ez a gyors, ritkán detektált labdát is érinti!).
  * callback-mechanizmus az ultralytics mintájára (model.add_callback / model.callbacks): a
    követést az első track() hívásnál regisztrált 'on_predict_postprocess_end' callback végzi,
    a korábban regisztrált (saját) callbackek még a NYERS predictor.results-ot látják
    (lásd main.py, NYERS LABDA-DETEKTÁLÁSOK, és docs/MODELL_BEALLITAS.md).

Az interfész a main.py által használt részhalmaz:
    model.names, model.track(frame, persist=True, tracker=..., device=..., verbose=False),
    model.predict(...), model(...), model.predictor.trackers[i].reset(),
    model.add_callback(event, func), model.callbacks, predictor.results (a callbackekben)
    results[i].boxes: iterálható; box.cls[0], box.conf[0], box.xyxy[0], box.id (None vagy [id]),
    valamint boxes.id (None, ha nincs követés), boxes.cls / .conf / .xyxy tömbök, len(boxes).

A képkocka -> ground-truth hozzárendelés hívásszámlálóval történik (a 0. hívás = frame_offset).
"""
import json
import math

import numpy as np

FOOTBALL_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
COCO_NAMES = {0: "person", 32: "sports ball"}

DEFAULT_PROFILE = {
    "player_miss": 0.10,          # átlagos kihagyási arány játékosnál
    "player_miss_burst": 0.5,     # P(kihagyás | előző képkockán is kimaradt)
    "occluded_miss": 0.40,
    "tiny_px": 14.0,              # ennél kisebb alak: +tiny_miss
    "tiny_miss": 0.35,
    "box_jitter": 0.04,           # a doboz széleinek szórása a dobozmagasság arányában
    "keeper_as_player": 0.15,
    "referee_as_player": 0.10,
    "spectator_detect": 0.65,     # néző / edző / csere 'player'-ként
    "ball_miss": 0.35,
    "ball_miss_burst": 0.55,
    "ball_occluded_miss": 0.85,
    "ball_blur_miss": 0.20,       # erős elmosódásnál ennyivel több kihagyás
    "ball_jitter_px": 0.8,
    "ball_fp_rate": 0.35,         # téves labda-jelölt / képkocka (Poisson átlag)
    "ball_conf": (0.25, 0.80),
    "ball_fp_conf": (0.10, 0.50),
}


class _Arr(np.ndarray):
    """numpy tömb .item()/.cpu()/.numpy()/.tolist() hívásokkal, mint a torch tenzor."""

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def int(self):
        return self.astype(np.int64).view(_Arr)


def _arr(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype).view(_Arr)


class FakeBox:
    __slots__ = ("cls", "conf", "xyxy", "id")

    def __init__(self, cls, conf, xyxy, tid=None):
        self.cls = _arr([cls])
        self.conf = _arr([conf])
        self.xyxy = _arr([xyxy])
        self.id = None if tid is None else _arr([tid], np.float32)


class FakeBoxes:
    def __init__(self, boxes, with_ids):
        self._boxes = boxes
        n = len(boxes)
        self.cls = _arr([b.cls[0] for b in boxes]) if n else _arr(np.zeros(0))
        self.conf = _arr([b.conf[0] for b in boxes]) if n else _arr(np.zeros(0))
        self.xyxy = _arr([b.xyxy[0] for b in boxes]).reshape(-1, 4) if n else _arr(np.zeros((0, 4)))
        self.id = _arr([b.id[0] for b in boxes]) if (with_ids and n) else None

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, i):
        return self._boxes[i]


class FakeResult:
    def __init__(self, boxes, with_ids, names):
        self.boxes = FakeBoxes(boxes, with_ids)
        self.names = names


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


class SimpleByteTrack:
    """
    Egyszerűsített ByteTrack: két lépcsős (erős / gyenge detektálás) IoU párosítás állandó
    sebességű doboz-előrejelzéssel, új követés megerősítése a következő képkockán,
    elveszett követés megtartása 'buffer' képkockáig. Csak a megerősített, ezen a képkockán
    párosított követések kerülnek a kimenetbe (mint az ultralytics model.track()-nél).
    """

    def __init__(self, high=0.25, low=0.10, new=0.25, match_iou=0.2, low_iou=0.5, buffer=30):
        self.high, self.low, self.new = high, low, new
        self.match_iou, self.low_iou, self.buffer = match_iou, low_iou, buffer
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.frame = 0

    @staticmethod
    def _pred(t):
        dx, dy = t["v"]
        b = t["box"]
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)

    def _match(self, tracks, dets, idxs, thr):
        pairs = []
        for ti, t in enumerate(tracks):
            pb = self._pred(t)
            for di in idxs:
                if dets[di][2] != t["group"]:
                    continue
                v = _iou(pb, dets[di][0])
                if v >= thr:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d, out = set(), set(), []
        for v, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            out.append((ti, di))
        return out, used_t, used_d

    def _apply(self, t, det):
        b = det[0]
        ob = t["box"]
        gap = max(1, self.frame - t["last"])
        nv = (((b[0] + b[2]) - (ob[0] + ob[2])) / 2.0 / gap, ((b[1] + b[3]) - (ob[1] + ob[3])) / 2.0 / gap)
        t["v"] = (0.5 * t["v"][0] + 0.5 * nv[0], 0.5 * t["v"][1] + 0.5 * nv[1])
        t["box"] = b
        t["last"] = self.frame

    def update(self, dets):
        """dets: [(xyxy, conf, group)] -> [(det_index, track_id)] a kimenetbe kerülő párok."""
        first = self.frame == 0 and not self.tracks
        high = [i for i, d in enumerate(dets) if d[1] >= self.high]
        low = [i for i, d in enumerate(dets) if self.low <= d[1] < self.high]
        confirmed = [t for t in self.tracks if t["active"]]
        unconfirmed = [t for t in self.tracks if not t["active"]]
        out = []
        m1, ut1, ud1 = self._match(confirmed, dets, high, self.match_iou)
        for ti, di in m1:
            self._apply(confirmed[ti], dets[di])
            out.append((di, confirmed[ti]["id"]))
        rest_t = [t for k, t in enumerate(confirmed) if k not in ut1 and self.frame - t["last"] <= 1]
        m2, ut2, ud2 = self._match(rest_t, dets, low, self.low_iou)
        for ti, di in m2:
            self._apply(rest_t[ti], dets[di])
            out.append((di, rest_t[ti]["id"]))
        rem_high = [i for i in high if i not in ud1]
        m3, ut3, ud3 = self._match(unconfirmed, dets, rem_high, 0.3)
        for ti, di in m3:
            t = unconfirmed[ti]
            self._apply(t, dets[di])
            t["active"] = True
            out.append((di, t["id"]))
        # meg nem erősített, párosítatlan követések törlése
        matched_unconf = {unconfirmed[ti]["id"] for ti, _ in m3}
        self.tracks = [t for t in self.tracks if t["active"] or t["id"] in matched_unconf]
        for di in rem_high:
            if di in ud3 or dets[di][1] < self.new:
                continue
            t = {"id": self.next_id, "box": dets[di][0], "v": (0.0, 0.0), "last": self.frame,
                 "active": first, "group": dets[di][2]}
            self.next_id += 1
            self.tracks.append(t)
            if first:
                out.append((di, t["id"]))
        self.tracks = [t for t in self.tracks if self.frame - t["last"] <= self.buffer]
        self.frame += 1
        return out


class _Predictor:
    def __init__(self, owner):
        self.trackers = [_TrackerProxy(owner)]
        self.results = []      # ultralytics: a postprocess utáni eredmények (a callbackek ezt látják)
        self.callbacks = owner.callbacks   # ultralytics: a Model és a predictor ugyanazt a dict-et használja


class _TrackerProxy:
    """A model_config.reset_tracker() ezen keresztül törli a követőt (és a képkocka-számlálót)."""

    def __init__(self, owner):
        self.owner = owner

    def reset(self):
        self.owner.reset()

    def reset_id(self):
        self.owner.tracker.next_id = 1


class FakeYOLO:
    def __init__(self, gt=None, seed=0, profile=None, names="football", drop_untracked=True):
        self.names = dict(FOOTBALL_NAMES if names == "football" else COCO_NAMES)
        self.coco = names != "football"
        self.seed = int(seed)
        self.profile = dict(DEFAULT_PROFILE)
        self.profile.update(profile or {})
        self.drop_untracked = drop_untracked
        self.tracker = SimpleByteTrack()
        # ultralytics: Model.callbacks (esemény -> függvénylista, regisztrálási sorrendben fut)
        self.callbacks = {"on_predict_start": [], "on_predict_postprocess_end": []}
        self._tracker_registered = False
        self._last_frame = None    # (képkocka objektum id, idx, dets) – az ugyanarra a képre hívott predict()-hez
        self.predictor = _Predictor(self)
        self.gt = None
        self.frame_offset = 0
        self.calls = 0
        self.stats = {"frames": 0, "ball_gt_visible": 0, "ball_detected": 0, "ball_fp": 0,
                      "ball_dropped_by_tracker": 0, "person_detected": 0, "spectator_fp": 0}
        if gt is not None:
            self.load_ground_truth(gt)

    # -------------------------------------------------------------- beállítás
    def load_ground_truth(self, gt, frame_offset=0):
        if isinstance(gt, str):
            with open(gt, "r", encoding="utf-8") as f:
                gt = json.load(f)
        self.gt = gt
        self.roles = {p["id"]: p["role"] for p in gt["people"]}
        self.frame_offset = int(frame_offset)
        self.reset()
        return self

    def reset(self):
        self.tracker.reset()
        self.calls = 0
        self._last_frame = None
        self._miss_state = {}
        self._ball_missed = False
        for k in self.stats:
            self.stats[k] = 0

    def _cls(self, role):
        if self.coco:
            return 32 if role == "ball" else 0
        return {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}[role]

    # -------------------------------------------------------------- detektálás
    def _frame_rng(self, idx):
        return np.random.default_rng([self.seed, idx])

    def detect_frame(self, idx, img_shape=None):
        """A ground truth idx. képkockájából zajos detektálások: [(xyxy, conf, cls, group)]."""
        if self.gt is None:
            return []
        frames = self.gt["frames"]
        if idx < 0 or idx >= len(frames):
            return []
        fr = frames[idx]
        pr = self.profile
        rng = self._frame_rng(idx)
        W_, H_ = self.gt["width"], self.gt["height"]
        out = []
        self.stats["frames"] += 1
        persons = []
        for o in fr["objs"]:
            pid, x1, y1, x2, y2, occ = o[0], o[5], o[6], o[7], o[8], o[9]
            role = self.roles.get(pid, "spectator")
            h = max(1.0, y2 - y1)
            w = max(1.0, x2 - x1)
            persons.append((x1, y1, x2, y2, h))
            if role in ("spectator", "coach", "sub"):
                if rng.random() > pr["spectator_detect"]:
                    continue
                det_role, conf = "player", rng.uniform(0.3, 0.75)
                self.stats["spectator_fp"] += 1
            else:
                prev_miss = self._miss_state.get(pid, False)
                base = pr["player_miss"]
                p_hm = base * (1 - pr["player_miss_burst"]) / max(1e-6, 1 - base)
                p_miss = pr["player_miss_burst"] if prev_miss else p_hm
                if occ > 0.5:
                    p_miss = max(p_miss, pr["occluded_miss"])
                if h < pr["tiny_px"]:
                    p_miss = min(0.95, p_miss + pr["tiny_miss"])
                miss = rng.random() < p_miss
                self._miss_state[pid] = miss
                if miss:
                    continue
                det_role = role
                if role == "goalkeeper" and rng.random() < pr["keeper_as_player"]:
                    det_role = "player"
                elif role == "referee" and rng.random() < pr["referee_as_player"]:
                    det_role = "player"
                conf = float(np.clip(rng.normal(0.82, 0.08) - (0.25 if h < 30 else 0.0), 0.2, 0.97))
                self.stats["person_detected"] += 1
            j = pr["box_jitter"] * h
            bx1 = x1 + rng.normal(0, j * 0.7)
            bx2 = x2 + rng.normal(0, j * 0.7)
            by1 = y1 + rng.normal(0, j)
            by2 = y2 + rng.normal(0, j * 0.6)
            if bx2 - bx1 < 2 or by2 - by1 < 3:
                continue
            bx1, bx2 = max(0.0, bx1), min(W_ - 1.0, bx2)
            by1, by2 = max(0.0, by1), min(H_ - 1.0, by2)
            if bx2 <= bx1 or by2 <= by1:
                continue
            out.append(((bx1, by1, bx2, by2), conf, self._cls(det_role), "person"))

        # labda
        b = fr["ball"]
        r_ball = 4.0
        if b.get("vis"):
            self.stats["ball_gt_visible"] += 1
            r = b.get("r", 3.0)
            r_ball = r
            p_hm = pr["ball_miss"] * (1 - pr["ball_miss_burst"]) / max(1e-6, 1 - pr["ball_miss"])
            p_miss = pr["ball_miss_burst"] if self._ball_missed else p_hm
            blur = b.get("blur", 0.0)
            if b.get("occ", 0) > 0.5:
                p_miss = max(p_miss, pr["ball_occluded_miss"])
            if blur > 4 * r:
                p_miss = min(0.95, p_miss + pr["ball_blur_miss"])
            if r < 1.6:
                p_miss = min(0.95, p_miss + 0.1)
            miss = rng.random() < p_miss
            self._ball_missed = miss
            if not miss:
                x1, y1, x2, y2 = b["bbox"]
                cx = (x1 + x2) / 2 + rng.normal(0, pr["ball_jitter_px"])
                cy = (y1 + y2) / 2 + rng.normal(0, pr["ball_jitter_px"])
                bw = (x2 - x1) * rng.uniform(0.85, 1.15) + 1.0
                bh = (y2 - y1) * rng.uniform(0.85, 1.15) + 1.0
                lo, hi = pr["ball_conf"]
                conf = rng.uniform(lo, hi) * (0.75 if blur > 4 * r else 1.0)
                out.append(((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), float(conf), self._cls("ball"), "ball"))
                self.stats["ball_detected"] += 1
        else:
            self._ball_missed = False
        n_fp = rng.poisson(pr["ball_fp_rate"])
        for _ in range(n_fp):
            kind = rng.random()
            if persons and kind < 0.5:
                x1, y1, x2, y2, h = persons[rng.integers(len(persons))]
                cx = rng.uniform(x1, x2)
                cy = y2 - rng.uniform(0.0, 0.12) * h
                s = max(2.0, h * 0.06)
            elif persons and kind < 0.7:
                x1, y1, x2, y2, h = persons[rng.integers(len(persons))]
                cx = rng.uniform(x1, x2)
                cy = y1 + rng.uniform(0, 0.15) * h
                s = max(2.0, h * 0.06)
            else:
                cx = rng.uniform(0, W_)
                cy = rng.uniform(H_ * 0.3, H_)
                s = max(2.0, 2 * r_ball * rng.uniform(0.7, 1.4))
            lo, hi = pr["ball_fp_conf"]
            out.append(((cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2), float(rng.uniform(lo, hi)),
                         self._cls("ball"), "ball"))
            self.stats["ball_fp"] += 1
        return out

    # -------------------------------------------------------------- ultralytics-szerű API
    def _next_idx(self):
        idx = self.frame_offset + self.calls
        self.calls += 1
        return idx

    def add_callback(self, event, func):
        """ultralytics Model.add_callback: a függvény a lista VÉGÉRE kerül (regisztrálási sorrend)."""
        self.callbacks.setdefault(event, []).append(func)

    def _run_callbacks(self, event):
        for cb in list(self.callbacks.get(event, [])):
            cb(self.predictor)

    def _tracker_postprocess(self, predictor):
        """
        Az ultralytics trackers/track.py on_predict_postprocess_end megfelelője: a követő a
        predictor.results dobozait frissíti, és az eredményt a követett dobozokra cseréli
        (a meg nem erősített dobozok – köztük a legtöbb labda – kiesnek).
        """
        dets = predictor._dets
        pairs = self.tracker.update([(d[0], d[1], d[3]) for d in dets])
        if not pairs:
            # ultralytics: ha egyetlen követés sincs, az eredmény változatlan (azonosító nélkül)
            return
        ids = dict(pairs)
        boxes = []
        for i, d in enumerate(dets):
            if i in ids:
                boxes.append(FakeBox(d[2], d[1], d[0], ids[i]))
            elif not self.drop_untracked:
                boxes.append(FakeBox(d[2], d[1], d[0], -1))
            elif d[3] == "ball":
                self.stats["ball_dropped_by_tracker"] += 1
        predictor.results[0] = FakeResult(boxes, True, self.names)

    def track(self, frame=None, persist=True, **kwargs):
        if not self._tracker_registered:
            # ultralytics Model.track(): register_tracker() az ELSŐ track() hívásnál fűzi a
            # követő callbackjét a listához – az addig regisztrált callbackek előtte futnak
            self.add_callback("on_predict_postprocess_end", self._tracker_postprocess)
            self._tracker_registered = True
        idx = self._next_idx()
        dets = self.detect_frame(idx, getattr(frame, "shape", None))
        self._last_frame = (id(frame), idx, dets)
        self._run_callbacks("on_predict_start")
        # nyers (követés előtti) eredmény: minden detektálás, azonosító nélkül
        self.predictor._dets = dets
        self.predictor.results = [FakeResult([FakeBox(d[2], d[1], d[0]) for d in dets], False, self.names)]
        self._run_callbacks("on_predict_postprocess_end")
        return self.predictor.results

    def predict(self, frame=None, classes=None, conf=0.0, **kwargs):
        # Ugyanarra a képkockára hívott predict() (pl. külön labda-menet) ugyanazt a GT képkockát kapja
        if self._last_frame is not None and frame is not None and self._last_frame[0] == id(frame):
            dets = self._last_frame[2]
        else:
            idx = self._next_idx()
            dets = self.detect_frame(idx, getattr(frame, "shape", None))
        boxes = [FakeBox(d[2], d[1], d[0]) for d in dets
                 if d[1] >= conf and (classes is None or d[2] in classes)]
        return [FakeResult(boxes, False, self.names)]

    __call__ = predict


class FakeDetector:
    """
    A detection/detector_base.py Detector-interfésze a FakeYOLO zajos ground-truth detektálásaival
    (követés és callbackek nélkül) – a saját követő (detection/tracker.py) végponttól végpontig
    tartó teszteléséhez. Ugyanazt a hívásszámlálót és véletlen-sorozatot használja, mint a FakeYOLO,
    így egy újratöltött (load_ground_truth) FakeYOLO-val képkockára azonos detektálásokat ad.
    """

    def __init__(self, fake_yolo):
        self.src = fake_yolo
        self.names = fake_yolo.names

    def detect(self, frame):
        from detection.detector_base import Detections
        idx = self.src._next_idx()
        dets = self.src.detect_frame(idx, getattr(frame, "shape", None))
        if not dets:
            return Detections(names=self.names)
        return Detections([d[0] for d in dets], [d[1] for d in dets], [d[2] for d in dets], self.names)

    def detect_batch(self, frames):
        return [self.detect(f) for f in frames]

    def describe(self):
        return "ál-detektor (FakeDetector, ground truth + zaj)"

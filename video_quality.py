"""
Videóminőség-figyelő: a valódi amatőr (Veo) felvételek tipikus hibáinak mérése, hogy az elemzés
"tudja", mikor kevésbé megbízható.

Mintavételezett képkockánként (alap: 0,5 mp-enként) mért jellemzők (frame_metrics):
  sharpness   – élesség: a Laplace-szűrt log-világosság szórásnégyzete a füves részen (rögzített,
                640 px-es feldolgozási szélességen – így felbontás-független; a log miatt az árnyék
                nem csökkenti), enyhe előszűréssel (a zaj ne tűnjön élességnek);
                edge_sharpness / motion_ratio: a legerősebb élek (vonalak, alakok) gradiense, illetve a
                vízszintes / függőleges élesség aránya (a forgó kamera mozgási elmosódása irányított)
  clipped     – beégett (≥ 250) képpontok aránya, crushed – bebukott (≤ 10) képpontok aránya
  contrast    – RMS kontraszt (szórás / átlag) a füvön, brightness – átlagos világosság
  noise       – zajbecslés (Immerkær-féle Laplace-különbség, robusztus medián), eredeti felbontáson
  blockiness  – 8x8-as blokkhatárokon mért ugrás / a blokkon belüli szomszéd-különbség (kockásodás)
  shadow      – árnyék-megosztottság: a fű kétmódusú (napos / árnyékos) világossága (az adaptív
                fűmodell módusaiból, vagy Otsu-küszöbbel), shadow_frac és shadow_ratio
  glare       – napfény / csillanás: nagy kiterjedésű, világos, fakó (alacsony telítettségű) képrész
                és a "kifakult" fű aránya

Ezekből hibánként súlyosság (0..1), címke (ha ≥ 0,5): "elmosódott", "beégett / napfény",
"erős árnyék", "kockás tömörítés", "sötét / zajos"; és képkockánkénti megbízhatóság
(reliability 0..1, 1 = jó kép). A summary() a teljes klipre: osztályzat, az egyes hibák
idő-aránya, a legrosszabb szakaszok időponttal.

Csak numpy + OpenCV.
"""
import math

import cv2
import numpy as np

PROC_WIDTH = 640

ISSUES = ("blur", "glare", "shadow", "blocking", "dark_noise")
ISSUE_LABELS = {
    "blur": "elmosódott",
    "glare": "beégett / napfény",
    "shadow": "erős árnyék",
    "blocking": "kockás tömörítés",
    "dark_noise": "sötét / zajos",
}
# a megbízhatóság csökkenése teljes súlyosságnál (szorzódnak)
ISSUE_WEIGHTS = {"blur": 0.45, "glare": 0.40, "shadow": 0.20, "blocking": 0.35, "dark_noise": 0.40}
FLAG_SEVERITY = 0.5

GRADES = ((0.85, "A", "jó"), (0.70, "B", "közepes"), (0.50, "C", "gyenge"), (0.0, "D", "rossz"))


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def _gray_proc(frame):
    h, w = frame.shape[:2]
    s = PROC_WIDTH / float(w)
    small = cv2.resize(frame, (PROC_WIDTH, max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA) if s < 1 else frame
    return small, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _noise_sigma(gray_full):
    """Immerkær-féle zajbecslés (σ, 0–255 skála) robusztus mediánnal, a sima területeken."""
    g = gray_full.astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    # élek (vonalak, alakok) kizárása: csak a kis gradiensű pixelek
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
    mag = np.abs(gx) + np.abs(gy)
    flat = mag < np.percentile(mag, 60)
    v = r[flat] if np.count_nonzero(flat) > 100 else r.ravel()
    return float(np.median(v) / (6.0 * 0.6745))


def _blockiness(gray_full):
    """
    Kockásodás: a 8 px-es blokkhatáron mért ugrás és a két szomszédos (blokkon belüli) különbség
    aránya, csak ott, ahol a blokkhatár két oldala sima (a textúra / élek elfednék). 1 = nincs.
    """
    g = gray_full.astype(np.float32)
    h, w = g.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 24 or w8 < 24:
        return 1.0

    def stat(d, cols):
        b = d[:, cols]
        n = 0.5 * (d[:, cols - 1] + d[:, cols + 1])
        flat = (d[:, cols - 1] < 6) & (d[:, cols + 1] < 6)
        if np.count_nonzero(flat) < 100:
            return None
        return (b[flat].mean() + 0.3) / (n[flat].mean() + 0.3)

    def one(img):
        d = np.abs(np.diff(img, axis=1))                 # d[:, x]: x és x+1 oszlop között
        at_edge = stat(d, np.arange(7, d.shape[1] - 1, 8))     # blokkhatárok
        control = stat(d, np.arange(3, d.shape[1] - 1, 8))     # ugyanez a blokkok belsejében (zaj-torzítás ellen)
        if at_edge is None or control is None:
            return 1.0
        return float(at_edge / control)
    g = g[:h8, :w8]
    return 0.5 * (one(g) + one(g.T.copy()))


def frame_metrics(frame, grass_model=None, grass_mask=None):
    """
    Egy képkocka minőség-jellemzői (lásd a modul leírását). grass_model: adaptív fűmodell
    (pitch_filters.GrassModel; árnyék- és csillanás-méréshez), grass_mask: kész fűmaszk (bármilyen
    felbontású, 255 = fű) – ha nincs, a modellből / a kép alsó 60%-ából számol.
    """
    small, gray = _gray_proc(frame)
    h, w = gray.shape
    lab_map = None
    if grass_model is not None and getattr(grass_model, "ready", False):
        lab_map = grass_model.label_map(small, scale=1.0)
        gm = (lab_map > 0)
    elif grass_mask is not None:
        gm = cv2.resize(grass_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        gm = np.zeros((h, w), bool)
        gm[int(h * 0.4):] = True
    if np.count_nonzero(gm) < 0.05 * h * w:
        gm = np.zeros((h, w), bool)
        gm[int(h * 0.4):] = True
    gm_er = cv2.erode(gm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if np.count_nonzero(gm_er) < 0.03 * h * w:
        gm_er = gm

    g32 = gray.astype(np.float32)
    # log-világosságon: az árnyék / megvilágítás (szorzó) nem változtatja az élességet
    pre = cv2.GaussianBlur(np.log(g32 + 8.0) * 64.0, (0, 0), 0.8)
    lap = cv2.Laplacian(pre, cv2.CV_32F, ksize=3)
    sharp = float(lap[gm_er].var())
    # élek élessége irányonként (a vonalak / alakok legerősebb gradiensei): a forgó kamera mozgási
    # elmosódása a vízszintes gradienseket tompítja, a függőlegeseket alig
    zone = np.zeros((h, w), bool)
    zone[int(h * 0.4):] = True          # rögzített képrész (a tábla / horizont élei ne számítsanak)
    gx = np.abs(cv2.Sobel(pre, cv2.CV_32F, 1, 0, ksize=3))[zone]
    gy = np.abs(cv2.Sobel(pre, cv2.CV_32F, 0, 1, ksize=3))[zone]
    px_, py_ = float(np.percentile(gx, 99.5)), float(np.percentile(gy, 99.5))
    edge_sharp = float(math.hypot(px_, py_))
    motion_ratio = (min(px_, py_) + 1.0) / (max(px_, py_) + 1.0)   # 1 = izotróp, kicsi = irányított elmosódás

    clipped = float(np.count_nonzero(small.max(axis=2) >= 250)) / (h * w)
    crushed = float(np.count_nonzero(gray <= 10)) / (h * w)
    gvals = g32[gm]
    brightness = float(gvals.mean()) if gvals.size else float(g32.mean())
    contrast = float(gvals.std() / max(1.0, gvals.mean())) if gvals.size else 0.0

    # zaj és kockásodás eredeti felbontáson, a kép alsó-középső részén (ott jellemzően sima fű van)
    H, W = frame.shape[:2]
    y0, y1 = (int(H * 0.55) // 8) * 8, int(H * 0.95)
    x0, x1 = (int(W * 0.2) // 8) * 8, int(W * 0.8)
    crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    noise = _noise_sigma(crop)
    blockiness = _blockiness(crop)

    # árnyék-megosztottság
    shadow_frac, shadow_ratio = 0.0, 1.0
    if lab_map is not None and grass_model.shadow_like():
        n1 = np.count_nonzero(lab_map == 1)
        n2 = np.count_nonzero(lab_map == 2)
        if n1 + n2 > 0:
            shadow_frac = n2 / float(n1 + n2)
            a, b = grass_model.modes
            shadow_ratio = float(math.exp(b.lv - a.lv))
    elif gvals.size > 200:
        v8 = np.clip(gvals, 0, 255).astype(np.uint8).reshape(-1, 1)
        thr, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = gvals[gvals <= thr], gvals[gvals > thr]
        if lo.size and hi.size:
            shadow_frac = lo.size / float(gvals.size)
            # csak valódi kétmódusúság: a két csoport elválik (a szórásokhoz képest)
            sep = (hi.mean() - lo.mean()) / (lo.std() + hi.std() + 1e-3)
            shadow_ratio = float(lo.mean() / max(1.0, hi.mean())) if sep > 2.0 else 1.0

    # napfény / csillanás: nagy, világos, fakó terület (a vékony fehér vonalak / mezek kiszűrve)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    bright_flat = ((hsv[..., 2] >= 215) & (hsv[..., 1] <= 50)).astype(np.uint8)
    bright_flat = cv2.morphologyEx(bright_flat, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    veil = float(np.count_nonzero(bright_flat[int(h * 0.25):])) / (h * w * 0.75)   # az égbolt nélkül
    faded = float(np.count_nonzero(lab_map == 3)) / max(1, np.count_nonzero(lab_map > 0)) if lab_map is not None else 0.0
    # fakóság: a fű telítettsége a fűmodell referenciájához képest
    sat_grass = float(hsv[..., 1][gm].mean()) if np.count_nonzero(gm) else 0.0

    return {
        "sharpness": round(sharp, 2), "edge_sharpness": round(edge_sharp, 1), "motion_ratio": round(motion_ratio, 3),
        "clipped": round(clipped, 4), "crushed": round(crushed, 4),
        "brightness": round(brightness, 1), "contrast": round(contrast, 3),
        "noise": round(noise, 2), "blockiness": round(blockiness, 3),
        "shadow_frac": round(shadow_frac, 3), "shadow_ratio": round(shadow_ratio, 3),
        "glare_area": round(veil, 4), "faded_grass": round(faded, 3), "grass_sat": round(sat_grass, 1),
    }


def severities(m, sharp_ref=None, sat_ref=None, ratio_ref=None, pan_px=None):
    """Hibánkénti súlyosság (0..1) a mért jellemzőkből. sharp_ref / sat_ref / ratio_ref: a videó saját
    (futó) referenciája – az élesség, a fű telítettsége és az élek irányaránya ehhez képest számít.
    pan_px: a kamera elfordulása képpontban / képkocka (1280 széles képre vetítve), ha ismert – a
    forgásból eredő elmosódás csak mozgó kameránál jelölhető (álló kameránál az irányarány a jelenettől függ)."""
    sev = {}
    # elmosódás: irányított (a vízszintes és függőleges élek élességének aránya) vagy általános
    # (az élek élessége a videó saját, futó referenciájához képest – csillanásnál ez nem számít,
    # mert ott a kontraszt csökken, nem az élesség)
    # (a jelenet saját irányaránya a vonalak / alakok miatt 0,6–0,95 között mozog: a futó referenciához mérve)
    s_dir = _clip01(((ratio_ref if ratio_ref else 0.85) - 0.12 - m["motion_ratio"]) / 0.2)
    if pan_px is not None:
        s_dir *= _clip01((pan_px - 2.0) / 4.0)
    rel = m["edge_sharpness"] / max(1e-3, sharp_ref) if sharp_ref else 1.0
    s_rel = _clip01((0.75 - rel) / 0.35)
    glary = max(m["glare_area"] / 0.08, m["clipped"] / 0.06) >= 0.5
    # (az általános élesség-csökkenés a jelenettől is függ – egymagában nem jelöl, csak ront a megbízhatóságon)
    sev["blur"] = _clip01(max(s_dir, 0.0 if glary else 0.45 * s_rel))
    # napfény: beégett + fakó terület + kifakult fű + a fű telítettségének esése
    s_sat = _clip01((0.85 - m["grass_sat"] / sat_ref) / 0.35) if sat_ref else 0.0
    sev["glare"] = _clip01(max(m["glare_area"] / 0.08, m["clipped"] / 0.06, m["faded_grass"] / 0.25, s_sat))
    # erős árnyék: a fű kétmódusú, a módusok elég nagyok és a világosság-arány kicsi
    f = m["shadow_frac"]
    balance = _clip01(min(f, 1.0 - f) / 0.25)   # nagy kiterjedésű árnyék (a játékosok saját árnyéka nem)
    sev["shadow"] = _clip01((0.8 - m["shadow_ratio"]) / 0.3) * balance
    sev["blocking"] = _clip01((m["blockiness"] - 1.7) / 0.8)
    sev["dark_noise"] = _clip01(max((75.0 - m["brightness"]) / 40.0, (m["noise"] - 3.5) / 5.0, m["crushed"] / 0.2))
    return {k: round(v, 3) for k, v in sev.items()}


def reliability_from(sev):
    r = 1.0
    for k, v in sev.items():
        r *= 1.0 - ISSUE_WEIGHTS.get(k, 0.3) * v
    return round(max(0.0, r), 3)


def grade_of(rel):
    for thr, letter, label in GRADES:
        if rel >= thr:
            return letter, label
    return GRADES[-1][1], GRADES[-1][2]


class VideoQualityMonitor:
    """
    Használat a feldolgozó ciklusban:
        qm = VideoQualityMonitor(fps, grass_model=gm)
        if qm.due(frame_idx): sample = qm.update(frame, t)
        qm.reliability_at(t)  -> 0..1 (a legközelebbi minták legrosszabbja ±window mp-en belül)
        qm.summary()          -> az events JSON 'video_quality' blokkja
    """

    def __init__(self, fps, sample_sec=0.5, grass_model=None, ref_window=40):
        self.fps = float(fps or 25.0)
        self.step = max(1, int(round(sample_sec * self.fps)))
        self.sample_sec = self.step / self.fps
        self.grass_model = grass_model
        self.samples = []
        self.ref_window = int(ref_window)
        self._sharp_hist = []
        self._sat_hist = []
        self._ratio_hist = []

    def due(self, frame_idx):
        return frame_idx % self.step == 0

    def update(self, frame, t, grass_mask=None, pan_px=None):
        """Egy mintavételezett képkocka mérése. pan_px: a kamera elfordulása (px / képkocka, a képszélesség
        szerint 1280-ra vetítve; pl. pitch_filters.CameraMotion eltolásából) – ha ismert, az irányított
        elmosódás csak mozgó kameránál jelölhető."""
        m = frame_metrics(frame, self.grass_model, grass_mask)
        if pan_px is not None:
            m["pan_px"] = round(float(pan_px), 2)
        # futó referencia: a nem gyanúsan homályos minták mediánja (a 75. percentilis) – a videó saját "éles" szintje
        self._sharp_hist.append(m["edge_sharpness"])
        self._sat_hist.append(m["grass_sat"])
        self._ratio_hist.append(m["motion_ratio"])
        rhist = self._ratio_hist[-self.ref_window:]
        ratio_ref = float(np.percentile(rhist, 75)) if len(rhist) >= 4 else None
        hist = self._sharp_hist[-self.ref_window:]
        sharp_ref = float(np.percentile(hist, 75)) if len(hist) >= 4 else None
        shist = self._sat_hist[-self.ref_window:]
        sat_ref = float(np.percentile(shist, 75)) if len(shist) >= 4 else None
        sev = severities(m, sharp_ref, sat_ref, ratio_ref, pan_px)
        rel = reliability_from(sev)
        flags = [k for k in ISSUES if sev.get(k, 0.0) >= FLAG_SEVERITY]
        sample = {"t": round(float(t), 2), "metrics": m, "severity": sev, "reliability": rel, "flags": flags}
        self.samples.append(sample)
        return sample

    def reliability_at(self, t, window=None):
        """A megbízhatóság a t időpontban: a ±window (alap: 1 mintaköz) közötti minták legrosszabbja."""
        if not self.samples:
            return None
        window = self.sample_sec * 1.01 if window is None else window
        vals = [s["reliability"] for s in self.samples if abs(s["t"] - t) <= window]
        if not vals:
            near = min(self.samples, key=lambda s: abs(s["t"] - t))
            return near["reliability"]
        return min(vals)

    @staticmethod
    def describe_sample(s):
        m = s["metrics"]
        lab = ", ".join(ISSUE_LABELS[k] for k in s["flags"]) or "rendben"
        return (f"MINŐSÉG: megbízhatóság={s['reliability']:.2f} [{lab}] élesség={m['sharpness']:.1f} "
                f"(élek {m['edge_sharpness']:.0f}, irányarány {m['motion_ratio']:.2f}), beégett={100 * m['clipped']:.1f}%, fény={m['brightness']:.0f}, "
                f"zaj={m['noise']:.1f}, kockásodás={m['blockiness']:.2f}, árnyék={100 * m['shadow_frac']:.0f}% "
                f"(arány {m['shadow_ratio']:.2f}), csillanás={100 * m['glare_area']:.1f}%")

    def _segments(self):
        """Összefüggő szakaszok hibánként: [(hiba, kezdet, vég, átlagos súlyosság, min. megbízhatóság)]."""
        segs = []
        half = self.sample_sec / 2.0
        for k in ISSUES:
            cur = None
            for s in self.samples:
                on = k in s["flags"]
                if on:
                    if cur is None:
                        cur = {"issue": k, "start": s["t"], "end": s["t"], "sev": [], "rel": []}
                    cur["end"] = s["t"]
                    cur["sev"].append(s["severity"][k])
                    cur["rel"].append(s["reliability"])
                elif cur is not None:
                    segs.append(cur)
                    cur = None
            if cur is not None:
                segs.append(cur)
        out = []
        for c in segs:
            out.append({"issue": c["issue"], "label": ISSUE_LABELS[c["issue"]],
                        "start_sec": round(max(0.0, c["start"] - half), 2), "end_sec": round(c["end"] + half, 2),
                        "severity": round(float(np.mean(c["sev"])), 2),
                        "min_reliability": round(float(np.min(c["rel"])), 2)})
        return out

    def summary(self, max_segments=6):
        if not self.samples:
            return {"grade": None, "label": "nincs mérés", "samples": 0}
        rel = np.array([s["reliability"] for s in self.samples])
        mean_rel = float(rel.mean())
        letter, label = grade_of(mean_rel)
        n = len(self.samples)
        pct = {ISSUE_LABELS[k]: round(100.0 * sum(k in s["flags"] for s in self.samples) / n, 1) for k in ISSUES}
        segs = self._segments()
        segs.sort(key=lambda s: -(s["end_sec"] - s["start_sec"]) * (1.0 - s["min_reliability"] + 0.1 * s["severity"]))
        med = {}
        for key in ("sharpness", "edge_sharpness", "motion_ratio", "brightness", "noise", "blockiness", "clipped", "shadow_frac", "glare_area"):
            med[key] = round(float(np.median([s["metrics"][key] for s in self.samples])), 3)
        main_issues = [k for k, v in sorted(pct.items(), key=lambda kv: -kv[1]) if v >= 10.0]
        return {
            "grade": letter, "label": label,
            "mean_reliability": round(mean_rel, 3), "min_reliability": round(float(rel.min()), 3),
            "samples": n, "sample_sec": round(self.sample_sec, 3),
            "issue_time_pct": pct,
            "main_issues": main_issues,
            "worst_segments": segs[:max_segments],
            "median_metrics": med,
            "summary_text": (f"Videóminőség: {letter} ({label}), átlagos megbízhatóság {mean_rel:.2f}"
                             + (f"; fő gondok: " + ", ".join(f"{k} ({pct[k]:.0f}% idő)" for k in main_issues)
                                if main_issues else "; jelentős képhiba nem volt")),
        }


def annotate_event_reliability(events, quality, window_sec=1.0):
    """
    Az események 'reliability' (0..1) mezője az esemény idején mért képminőségből
    (a ±window_sec közötti minták legrosszabbja), és 'quality_flags': az akkori képhibák (magyar
    címkék). Utófeldolgozás – az events.py belseje nem változik.
    """
    if quality is None or not quality.samples:
        return events
    for ev in events:
        t = ev.get("time")
        if t is None:
            continue
        rel = quality.reliability_at(float(t), window=window_sec)
        if rel is None:
            continue
        ev["reliability"] = round(float(rel), 2)
        flags = sorted({f for s in quality.samples if abs(s["t"] - float(t)) <= window_sec for f in s["flags"]})
        if flags:
            ev["quality_flags"] = [ISSUE_LABELS[f] for f in flags]
    return events

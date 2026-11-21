"""
Pályakövetés mozgó (Veo) kamerához: képkockánkénti homográfia.

A Veo virtuális kamerája a labdát követi: forog (pásztáz), néha zoomol,
és egyszerre nagyjából fél pályát mutat. Az egyszer, a klip elején
kiszámolt homográfia (kép -> pálya méter) ezért az első pásztázás után
már hibás. Ez a modul minden képkockára saját homográfiát ad.

Fizikai kameramodell ('kamera' üzemmód, alap): a Veo egy árbocon áll, a helye
a felvétel alatt nem változik. A kalibrációból (kézi pontok vagy automatikus)
a kamera helyét (C) és állását becsüljük; képkockánként csak az állás
    x = (pan, tilt, roll [fok], 100·ln f)
változik (4 szabadsági fok a 8 helyett). Így a homográfia nem "vándorolhat el"
lehetetlen (nyírt, az égbe forduló) alakba, és a sodródás is sokkal kisebb.

1. Terjesztés: két képkocka között a kép a K2·R2·R1^T·K1^-1 alakú (tiszta
   forgatás + zoom) homográfiával mozog. A sarokpontokat (GFTT) csak a
   talajsíkon (a pálya + néhány méter, a jelenlegi becslésből; a fűszínű
   fasor / lelátó / képre égetett felirat kizárva, a játékos-dobozok
   kitakarva) keressük, piramisos LK követi őket – a kezdőpont az állandó
   szögsebességű előrejelzés, így a gyors pásztázás sem szökik ki –, majd a
   4 paramétert robosztusan (Cauchy-súlyozás) illesztjük. A becslés
   kovarianciáját is visszük (folyamatzaj a mozgás nagyságával nő).

2. Vonal-illesztés és önellenőrzés (refine_every képkockánként, illetve gyanús
   állapotban minden képkockán): a modell összes szabványos vonalát a képre
   vetítjük, és az állást MAP-becsléssel finomítjuk: a vonalmaszk
   távolság-transzformáltján mért robosztus költség + a mozgásbecslés mint
   előzetes. A rosszul meghatározott irányokban (pl. csak egy vonal látszik) a
   predikció marad, más vonalra ugrani csak erős alátámasztással lehet.
   Minőségmérték: a látható modellpontok hány %-a esik vonalpixelre
   (pontosság) és a látható modellvonalak hány %-át támasztják alá
   vonalpixelek (alátámasztás); ebből simított bizalom (conf).

3. Kamerahely-pontosítás: a jól illeszkedő, eltérő irányú képkockák
   kulcsképek; ezekből a kamerahelyet közösen becsüljük (bundle_adjust_camera)
   – egy képből a kamerahely és a fókusz/dőlés nehezen választható szét,
   több irányból viszont jól meghatározott.

4. Státusz (csak 'ok' esetén ad vissza H-t, különben None):
   'ok' – a vonalak megerősítik a becslést; 'bizonytalan' – a bizalom alacsony,
   rég nem volt megerősítés, vagy a becsült helyzethiba nagy; 'elveszett' –
   tartósan rossz illesztés / sikertelen mozgásbecslés: ilyenkor
   újrapozicionálás a rögzített kamerahellyel, (pan, tilt, zoom) rácskeresés a
   predikció körül; ha ez többször sem sikerül, teljes automatikus kalibráció;
   'kalibráció szükséges' – nincs (elfogadható) kalibráció.

5. Automatikus kezdeti kalibráció (best effort): ha nincs kézi kalibráció,
   kamera-hipotézisek rácsát (pásztázási szög, dőlés, fókusz, kamera
   távolsága és magassága; a kamera a felezővonal vonalában, a közeli
   oldalvonal mögött) pontozzuk ugyanazzal a távolság-transzformált
   mértékkel, a legjobbakat finomítjuk, és csak akkor fogadjuk el, ha a
   vonalegyezés (pontosság és lefedettség) eléri a küszöböt.

Tartalék ('homográfia' üzemmód): ha a kalibráció nem írható le rögzített helyű
lyukkamerával (pl. kézi, mozgó kamera), a régi módszer fut: a képkockák közti
általános homográfia (RANSAC), H_t = H_{t-1} · inv(F_t), és 8 paraméteres
korrekció a vonalakra (refine_homography).

Jelölés: F_t a (t-1). kép pixeleit viszi a t. kép pixeleire (x_t = F_t · x_{t-1});
belül G = inv(H) (pálya -> kép) a tárolt alak.

Koordináta-konvenció (a TacticalEngine.compute_homography-val egyezik):
x a pálya hossza mentén (0..105), y a szélesség mentén (0..68); a kép
bal felső sarkához a (0, 0) pálya-sarok tartozik, a kép alja (a kamera
felé eső oldalvonal) az y = 68.

Csak OpenCV + numpy, önállóan tesztelhető (tests/test_pitch_tracker.py).
"""
import math

import cv2
import numpy as np

from pitch_filters import build_grass_mask, is_full_range

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
DEFAULT_GRASS_HSV = (np.array([30, 40, 40]), np.array([85, 255, 255]))

STATUS_OK = "ok"
STATUS_LOST = "elveszett"
STATUS_UNSURE = "bizonytalan"
STATUS_NEED_CALIB = "kalibráció szükséges"


# ---------------------------------------------------------------------------
# Pályamodell
# ---------------------------------------------------------------------------

def pitch_polylines(step=0.5, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """
    A szabványos (FIFA) pályavonalak töréspontos vonalakként, méterben.
    Visszatér: [np.ndarray (N, 2), ...] – minden elem egy összefüggő vonal.
    """
    lines = []

    def seg(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        lines.append(a + (b - a) * t)

    def arc(cx, cy, r, a0, a1):
        n = max(3, int(math.ceil(abs(a1 - a0) * r / step)) + 1)
        t = np.linspace(a0, a1, n)
        lines.append(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1))

    L, W = length, width
    cy = W / 2.0
    seg((0, 0), (L, 0))            # túlsó oldalvonal
    seg((0, W), (L, W))            # közeli oldalvonal
    seg((0, 0), (0, W))            # bal alapvonal
    seg((L, 0), (L, W))            # jobb alapvonal
    seg((L / 2, 0), (L / 2, W))    # felezővonal
    arc(L / 2, cy, 9.15, 0.0, 2 * math.pi)  # kezdőkör
    arc_a = math.acos(5.5 / 9.15)  # a 16-os ív a büntetőpontból, a 16-oson kívüli része
    for x0, s in ((0.0, 1.0), (L, -1.0)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):   # 16-os és 5-ös
            y1, y2 = cy - half, cy + half
            seg((x0, y1), (x0 + s * depth, y1))
            seg((x0 + s * depth, y1), (x0 + s * depth, y2))
            seg((x0 + s * depth, y2), (x0, y2))
        if s > 0:
            arc(11.0, cy, 9.15, -arc_a, arc_a)
        else:
            arc(L - 11.0, cy, 9.15, math.pi - arc_a, math.pi + arc_a)
    return lines


def pitch_model_points(step=0.5):
    """Az összes vonal mintapontja egy tömbben (M, 2), méterben."""
    return np.concatenate(pitch_polylines(step), axis=0)


def pitch_model_tangents(step=0.5):
    """A pitch_model_points pontjaihoz tartozó egységnyi érintőirányok (M, 2)."""
    out = []
    for poly in pitch_polylines(step):
        t = np.gradient(poly, axis=0)
        out.append(t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Geometriai segédfüggvények
# ---------------------------------------------------------------------------

def project_points(G, pts):
    """Pálya -> kép vetítés G-vel. Visszatér: (uv (N,2), w (N,)) – w <= 0: a kamera mögött."""
    X = G[:, :2] @ pts.T + G[:, 2:3]
    w = X[2]
    safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return (X[:2] / safe).T, w


def scale_matrix(s, sy=None):
    """
    Teljes felbontás -> feldolgozási felbontás pixelkoordináta. A pixelközéppont-
    konvenció miatt (cv2.resize) u_s = (u + 0.5)·s - 0.5, nem egyszerűen u·s –
    a fél pixel eltérés a távoli vonalaknál fél méteres hibát is okozna.
    """
    sy = s if sy is None else sy
    return np.array([[s, 0, 0.5 * s - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]], dtype=np.float64)


def homography_from_points(img_pts, pitch_pts):
    """Kép -> pálya homográfia legalább 4 pontpárból (legkisebb négyzetek, nem RANSAC)."""
    img = np.asarray(img_pts, np.float64).reshape(-1, 2)
    pit = np.asarray(pitch_pts, np.float64).reshape(-1, 2)
    if len(img) < 4 or len(img) != len(pit):
        return None
    H, _ = cv2.findHomography(img, pit, 0)
    return H


def parse_calib_points(calib_str, width, height, pitch_points_str=None):
    """
    A kliens kalibrációja: calib_str = "rx,ry;rx,ry;..." normalizált (0..1) képpontok.
    pitch_points_str üres: a régi négysarkos sorrend (bal-felső, bal-alsó,
    jobb-alsó, jobb-felső = (0,0), (0,68), (105,68), (105,0)).
    Megadva: "x,y;x,y;..." méterben, ugyanannyi pont, mint a képpont (>= 4) –
    így fél pályás képen is lehet kalibrálni (pl. 16-os sarkai, felezővonal végei).
    Visszatér: H (kép pixel -> pálya méter) vagy None.
    """
    if not calib_str:
        return None
    img = []
    for p in calib_str.split(';'):
        p = p.strip()
        if p:
            rx, ry = map(float, p.split(','))
            img.append([rx * width, ry * height])
    if pitch_points_str:
        pit = []
        for p in pitch_points_str.split(';'):
            p = p.strip()
            if p:
                pit.append(list(map(float, p.split(','))))
    else:
        pit = [[0.0, 0.0], [0.0, PITCH_WIDTH], [PITCH_LENGTH, PITCH_WIDTH], [PITCH_LENGTH, 0.0]]
    if len(img) < 4 or len(img) != len(pit):
        return None
    return homography_from_points(img, pit)


def camera_rotation(pan_deg, tilt_deg, roll_deg=0.0):
    """
    A kamera forgatási mátrixa (sorai: a kép jobbra, lefelé és előre mutató tengelye a világban).
    Konvenció: lásd camera_ground_homography. (Zárt alak: előre = (sp·ct, -cp·ct, st),
    jobbra = előre × fel = (cp, sp, 0), le = előre × jobbra = (-st·sp, st·cp, ct).)
    """
    p = math.radians(pan_deg)
    t = math.radians(tilt_deg)
    sp, cp, st, ct = math.sin(p), math.cos(p), math.sin(t), math.cos(t)
    fwd = (sp * ct, -cp * ct, st)
    right = (cp, sp, 0.0)
    down = (-st * sp, st * cp, ct)
    if roll_deg:
        r = math.radians(roll_deg)
        cr, sr = math.cos(r), math.sin(r)
        right, down = (tuple(cr * a + sr * b for a, b in zip(right, down)),
                       tuple(-sr * a + cr * b for a, b in zip(right, down)))
    return np.array([right, down, fwd])


def camera_K(focal_px, img_w, img_h):
    return np.array([[focal_px, 0, img_w / 2.0], [0, focal_px, img_h / 2.0], [0, 0, 1.0]])


def camera_ground_homography(pan_deg, tilt_deg, focal_px, img_w, img_h, cam_pos, roll_deg=0.0):
    """
    Lyukkamera pálya-sík -> kép homográfia (G).
    cam_pos = (x, y, z) méterben; a világ z tengelye LEFELÉ mutat (a pálya-
    koordináták x jobbra, y a kamera felé – így jobbsodrású), tehát a magasság z = -h.
    pan = 0: a kamera a túlsó oldalvonal felé (-y) néz; pozitív pan: jobbra (+x felé).
    tilt: lefelé dőlés fokban.
    """
    R = camera_rotation(pan_deg, tilt_deg, roll_deg)
    C = np.asarray(cam_pos, float)
    K = camera_K(focal_px, img_w, img_h)
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def orient_G(G, img_w, img_h):
    """
    G (pálya -> kép) normalizálása: egységnyi norma, és olyan előjel, hogy a
    látható pályapontoknál w > 0 legyen (a homográfia előjele tetszőleges,
    pl. a cv2.findHomography eredményénél; a 'kamera előtt' vizsgálat erre épül).
    """
    G = np.asarray(G, np.float64)
    G = G / np.linalg.norm(G)
    try:
        H = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return G
    X = H @ np.array([img_w * 0.5, img_h * 0.9, 1.0])
    if abs(X[2]) < 1e-12:
        return G
    xy = X[:2] / X[2]
    w = G[2, 0] * xy[0] + G[2, 1] * xy[1] + G[2, 2]
    return G if w > 0 else -G


def _bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = x - x0
    fy = y - y0
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# ---------------------------------------------------------------------------
# Vonalpixelek
# ---------------------------------------------------------------------------

def _odd(v):
    v = int(v)
    return v if v % 2 == 1 else v + 1


def _boxes_mask(shape, boxes, scale, pad_ratio=0.15, pad_px=3):
    """Dobozok (teljes felbontású x1,y1,x2,y2) kitakaró maszkja a feldolgozási felbontáson (255 = doboz)."""
    m = np.zeros(shape[:2], np.uint8)
    if not boxes:
        return m
    h, w = shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = [float(v) * scale for v in b[:4]]
        px = (x2 - x1) * pad_ratio + pad_px
        py = (y2 - y1) * pad_ratio + pad_px
        cv2.rectangle(m, (max(0, int(x1 - px)), max(0, int(y1 - py))),
                      (min(w - 1, int(x2 + px)), min(h - 1, int(y2 + py))), 255, -1)
    return m


def detect_line_mask(small_bgr, grass_mask=None, box_mask=None, hsv=None, min_tophat=22, max_sat=140):
    """
    Fehér pályavonal-pixelek: világos, alacsony telítettségű, vékony struktúra a füvön.
    - a színcsatornák minimuma (fehérnél mind magas, fűnél a piros/kék alacsony),
    - top-hat (a kernelnél vékonyabb világos részek maradnak meg),
    - telítettség-küszöb, fűzóna (tágítva, hogy a szélső vonal is benne legyen),
    - a játékos-/labdadobozok kitakarva.
    """
    h, w = small_bgr.shape[:2]
    b, g, r = cv2.split(small_bgr)
    mn = cv2.min(cv2.min(b, g), r)
    k = _odd(max(7, w / 55))     # a legvastagabb (közeli) vonalnál szélesebb legyen
    th = cv2.morphologyEx(mn, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if hsv is None:
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    mask = ((th >= min_tophat) & (hsv[:, :, 1] <= max_sat)).astype(np.uint8) * 255
    if grass_mask is not None:
        kd = _odd(max(3, w / 300))
        zone = cv2.dilate(grass_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        mask &= zone
    if box_mask is not None:
        mask[box_mask > 0] = 0
    mask = _coherent(mask, sigma=max(1.5, w / 400.0))
    return _keep_thin_components(mask, min_len=max(6, w // 80), max_half_width=max(3.0, w / 110.0))


def _coherent(mask, sigma=2.0, min_coherence=0.45):
    """
    Csak az egyértelmű irányú (vonalszerű) pixelek maradnak: a maszk
    struktúratenzorának koherenciája ((λ1-λ2)/(λ1+λ2))^2. Egy vonal (vagy a
    kezdőkör egy íve) helyben egyenes -> magas; a lelátó / reklámtábla /
    közönség szórt zaja irány nélküli -> alacsony.
    """
    if not np.any(mask):
        return mask
    f = mask.astype(np.float32) / 255.0
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    kb = 2 * int(round(1.5 * sigma)) + 1          # dobozszűrő (gyorsabb, mint a Gauss)
    jxx = cv2.boxFilter(gx * gx, -1, (kb, kb))
    jyy = cv2.boxFilter(gy * gy, -1, (kb, kb))
    jxy = cv2.boxFilter(gx * gy, -1, (kb, kb))
    # csak a maszk pixeleinél számolunk (a maszk ritka – így sokkal gyorsabb)
    ys, xs = np.nonzero(mask)
    a, b, c = jxx[ys, xs], jyy[ys, xs], jxy[ys, xs]
    tr = a + b
    det_term = np.sqrt((a - b) ** 2 + 4 * c * c)
    coh = (det_term / np.maximum(tr, 1e-6)) ** 2
    out = mask.copy()
    bad = coh < min_coherence
    out[ys[bad], xs[bad]] = 0
    return out


def _keep_thin_components(mask, min_len=8, max_half_width=4.5):
    """
    Csak a vonalszerű összefüggő részek maradnak: a szórt zaj (lelátó,
    reklámtábla-felirat, mezek) apró foltokat ad, a nagy világos felületek
    (fehér mez közelről) pedig vastagok. Vastagság: a komponensen belüli
    távolság-transzformált maximuma (fél szélesség).
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    bw = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    bh = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    length = np.hypot(bw, bh)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    thick = np.zeros(n, bool)
    thick[np.unique(lab[dt > max_half_width])] = True
    keep = (length >= min_len) & ~thick
    keep[0] = False
    return np.where(keep[lab], 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Vonal-illesztés (finomítás)
# ---------------------------------------------------------------------------

class _LineField:
    """A vonalmaszk távolság-transzformáltja és gradiense (egy képkockára)."""

    def __init__(self, line_mask):
        self.mask = line_mask
        self.h, self.w = line_mask.shape[:2]
        inv = cv2.compare(line_mask, 0, cv2.CMP_EQ)       # 255, ahol nincs vonal
        self.D = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
        self.gx = cv2.Sobel(self.D, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        self.gy = cv2.Sobel(self.D, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        self.n_line = int(np.count_nonzero(line_mask))
        self._long = None

    def _long_components(self):
        # A lefedettséghez csak a hosszú (biztosan vonal) komponensek számítanak:
        # a lelátó / reklámtábla zaja rövid darabokra esik szét (csak igény esetén számoljuk)
        if self._long is None:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(self.mask, connectivity=8)
            length = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
            keep = length >= max(20.0, self.w / 30.0)
            keep[0] = False
            long_mask = np.where(keep[lab], 255, 0).astype(np.uint8)
            self._long = (long_mask, int(np.count_nonzero(long_mask)))
        return self._long

    @property
    def long_mask(self):
        return self._long_components()[0]

    @property
    def n_long(self):
        return self._long_components()[1]

    def inside(self, uv, w, margin=1.0):
        return ((w > 1e-6) & (uv[:, 0] >= margin) & (uv[:, 0] <= self.w - 1 - margin)
                & (uv[:, 1] >= margin) & (uv[:, 1] <= self.h - 1 - margin))

    def dist(self, uv):
        return _bilinear(self.D, uv[:, 0], uv[:, 1])


def _cost(field, G, P, tau, wts=None):
    """Csonkolt négyzetes költség rögzített pontkészleten; a képen kívülre eső pont tau-t ér."""
    uv, w = project_points(G, P)
    ins = field.inside(uv, w)
    d = np.full(len(P), tau, np.float64)
    if ins.any():
        d[ins] = np.minimum(field.dist(uv[ins]), tau)
    if wts is None:
        return float(np.mean(d * d)), d
    return float(np.sum(wts * d * d) / np.sum(wts)), d


LINE_WIDTH_M = 0.12


def _point_weights(G, P, T, step):
    """
    Pontonkénti súly az illesztéshez:
    - képbeli mintatávolság (px): a 0.5 m-es minták a távoli vonalakon sűrűn
      esnek a képre – súlyozás nélkül ezek uralnák az illesztést;
    - a vonal látható szélessége (px): a távoli, erősen rövidülő (pl. a túlsó
      oldalvonal) vonal 1 px-nél vékonyabb, alig látszik, és mellette a lelátó /
      reklámtábla zaja van – ezek kis súlyt kapnak.
    T: a pontok érintőirányai (None esetén csak a mintatávolság számít).
    """
    eps = 0.05
    uv, _ = project_points(G, P)
    ux, _ = project_points(G, P + np.array([eps, 0.0]))
    uy, _ = project_points(G, P + np.array([0.0, eps]))
    Jx = (ux - uv) / eps            # d(kép)/d(pálya x), px/m
    Jy = (uy - uv) / eps
    if T is None:
        spacing = 0.5 * (np.linalg.norm(Jx, axis=1) + np.linalg.norm(Jy, axis=1)) * step
        return np.clip(spacing, 0.1, 4.0)
    tang = Jx * T[:, :1] + Jy * T[:, 1:2]
    norm = Jx * (-T[:, 1:2]) + Jy * T[:, :1]
    spacing = np.linalg.norm(tang, axis=1) * step
    # a normális irányú nagyítás a vonal irányára merőleges komponense ~ a látható szélesség
    tl = np.maximum(np.linalg.norm(tang, axis=1), 1e-9)
    cross = np.abs(tang[:, 0] * norm[:, 1] - tang[:, 1] * norm[:, 0]) / tl
    width_px = cross * LINE_WIDTH_M
    vis = np.clip(width_px / 1.5, 0.05, 1.0) ** 2
    return np.clip(spacing, 0.1, 4.0) * vis


def refine_homography(field, G, pts, taus=(8.0, 4.0, 2.0), iters=6, prior=2e-2, min_points=40, step=0.5,
                      eig_rel=1e-3, tangents=None):
    """
    G (pálya -> feldolgozási kép) finomítása: a vetített modellpontok
    távolság-transzformált értékének minimalizálása.
    Paraméterezés: 8 paraméteres korrekció normalizált képkoordinátákban,
    G' = N^-1 · C(δ) · N · G, C(δ) = I + [[δ0 δ1 δ2] [δ3 δ4 δ5] [δ6 δ7 0]].
    Levenberg–Marquardt csillapítás + kis prior (alulhatározott esetben, pl.
    ha csak egy vonal látszik, ne csússzon el).
    Visszatér: (G_új, info) vagy (None, info), ha kevés a látható pont.
    """
    h, w = field.h, field.w
    fn = 0.5 * w
    N = np.array([[1 / fn, 0, -0.5 * w / fn], [0, 1 / fn, -0.5 * h / fn], [0, 0, 1]])
    Ninv = np.linalg.inv(N)
    uv, wz = project_points(G, pts)
    vis = field.inside(uv, wz, margin=2.0)
    P = pts[vis]
    info = {"n_visible": int(len(P))}
    if len(P) < min_points:
        return None, info
    G = G.copy()
    Wp = _point_weights(G, P, None if tangents is None else tangents[vis], step)
    tau0 = max(taus)
    info["cost_before"], d0 = _cost(field, G, P, tau0, Wp)
    info["precision_before"] = float(np.sum(Wp * (d0 < 2.0)) / np.sum(Wp))
    for tau in taus:
        lam = 1e-3
        cost, _ = _cost(field, G, P, tau, Wp)
        for _ in range(iters):
            uv, wz = project_points(G, P)
            ins = field.inside(uv, wz)
            uv, wp = uv[ins], Wp[ins]
            if len(uv) < min_points:
                break
            d = field.dist(uv)
            sel = d < tau
            if np.count_nonzero(sel) < min_points // 2:
                break
            uv, d, wp = uv[sel], d[sel], wp[sel]
            gxs = _bilinear(field.gx, uv[:, 0], uv[:, 1]) * fn
            gys = _bilinear(field.gy, uv[:, 0], uv[:, 1]) * fn
            q = (uv - np.array([0.5 * w, 0.5 * h])) / fn
            x, y = q[:, 0], q[:, 1]
            z = np.zeros_like(x)
            o = np.ones_like(x)
            Ju = np.stack([x, y, o, z, z, z, -x * x, -x * y], axis=1)
            Jv = np.stack([z, z, z, x, y, o, -x * y, -y * y], axis=1)
            J = gxs[:, None] * Ju + gys[:, None] * Jv
            s = 0.5 * tau
            wgt = wp / (1.0 + (d / s) ** 2)          # Cauchy-súly (robosztus) × képbeli sűrűség
            A = J.T @ (J * wgt[:, None]) / np.sum(wp)
            b = J.T @ (d * wgt) / np.sum(wp)
            # Csak a mérésekkel jól meghatározott irányokban lépünk: a gyengén
            # meghatározott (pl. csak egy vonal + kör látszik) irányokban a
            # terjesztett becslés marad (különben a zajra illesztene).
            ev, V = np.linalg.eigh(A)
            strong = ev >= eig_rel * max(ev.max(), 1e-12)
            Vs = V[:, strong]
            As = Vs.T @ A @ Vs
            bs = Vs.T @ b
            improved = False
            for _ in range(4):
                Ad = As + lam * np.diag(np.diag(As)) + prior * np.eye(len(bs)) * (np.trace(A) / 8.0 + 1e-9)
                try:
                    delta = -Vs @ np.linalg.solve(Ad, bs)
                except np.linalg.LinAlgError:
                    break
                C = np.array([[1 + delta[0], delta[1], delta[2]],
                              [delta[3], 1 + delta[4], delta[5]],
                              [delta[6], delta[7], 1.0]])
                G_new = Ninv @ C @ N @ G
                cost_new, _ = _cost(field, G_new, P, tau, Wp)
                if cost_new < cost:
                    G, cost = G_new, cost_new
                    lam = max(lam / 3.0, 1e-6)
                    improved = True
                    break
                lam *= 10.0
            if not improved:
                break
    info["cost_after"], d1 = _cost(field, G, P, tau0, Wp)
    info["precision_after"] = float(np.sum(Wp * (d1 < 2.0)) / np.sum(Wp))
    return G, info


def line_recall(field, G, tol_px=3):
    """A felismert vonalpixelek mekkora része esik a vetített modellvonalak közelébe (0..1)."""
    if field.n_long == 0:
        return 0.0
    model = np.zeros((field.h, field.w), np.uint8)
    for poly in pitch_polylines(0.5):
        uv, wz = project_points(G, poly)
        _draw_projected(model, uv, wz, 255, 1)
    model = cv2.dilate(model, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1)))
    return float(np.count_nonzero(model[field.long_mask > 0])) / field.n_long


def _draw_projected(img, uv, wz, color, thickness, lineType=cv2.LINE_8, shift_scale=1.0):
    """Vetített vonal kirajzolása; a kamera mögé eső pontoknál a vonal megszakad."""
    ok = wz > 1e-6
    lim = 1e5
    run = []
    for i in range(len(uv)):
        if ok[i] and abs(uv[i, 0]) < lim and abs(uv[i, 1]) < lim:
            run.append(uv[i])
        else:
            if len(run) >= 2:
                cv2.polylines(img, [np.round(np.array(run) * shift_scale).astype(np.int32)], False,
                              color, thickness, lineType)
            run = []
    if len(run) >= 2:
        cv2.polylines(img, [np.round(np.array(run) * shift_scale).astype(np.int32)], False,
                      color, thickness, lineType)


def evaluate_fit(field, G, pts, tangents=None, step=0.5):
    """Illeszkedés: (pontosság, lefedettség, látható pontok). Pontosság: a látható modellpontok
    (súlyozott) hány %-a van 2 px-en belül vonalpixelhez; lefedettség: lásd line_recall."""
    uv, wz = project_points(G, pts)
    vis = field.inside(uv, wz, margin=2.0)
    n = int(np.count_nonzero(vis))
    if n == 0:
        return 0.0, 0.0, 0
    wp = _point_weights(G, pts[vis], None if tangents is None else tangents[vis], step)
    prec = float(np.sum(wp * (field.dist(uv[vis]) < 2.0)) / max(np.sum(wp), 1e-9))
    return prec, line_recall(field, G), n


# ---------------------------------------------------------------------------
# Automatikus kezdeti kalibráció
# ---------------------------------------------------------------------------

def auto_calibrate(field, img_w, img_h, pts_coarse, pts_fine, max_candidates=12,
                   pans=None, tilts=None, hfovs=None, cam_positions=None, min_visible=60, px=1.0,
                   tangents=None):
    """
    Kamera-hipotézisek rácskeresése a vonalmaszk távolság-transzformáltján.
    img_w, img_h: a feldolgozási kép mérete (field mérete).
    Visszatér: (G, info) – G None, ha egyik hipotézis sem elég jó.
    Feltevés: a kamera a felezővonal vonalában áll, a közeli (y = 68) oldalvonal
    mögött (Veo ajánlott elhelyezése); a pálya 180°-os szimmetriája miatt a
    túloldali kamera ugyanígy írható le.
    """
    pans = np.arange(-60.0, 60.1, 2.0) if pans is None else np.asarray(pans, float)
    tilts = np.arange(4.0, 32.1, 2.0) if tilts is None else np.asarray(tilts, float)
    hfovs = (85.0, 72.0, 62.0, 53.0, 45.0, 37.0) if hfovs is None else hfovs
    cam_positions = ((PITCH_LENGTH / 2, PITCH_WIDTH + 4.0, -6.0),
                     (PITCH_LENGTH / 2, PITCH_WIDTH + 10.0, -9.0)) if cam_positions is None else cam_positions
    hyps = []
    Gs = []
    for cp in cam_positions:
        for hf in hfovs:
            f = 0.5 * img_w / math.tan(math.radians(hf) / 2)
            for t in tilts:
                for p in pans:
                    Gs.append(camera_ground_homography(p, t, f, img_w, img_h, cp))
                    hyps.append((float(p), float(t), float(hf), cp))
    Gs = np.array(Gs)                                  # (K, 3, 3)
    # A durva pontozás is súlyoz: képbeli mintatávolság × a vonal látható szélessége
    # (lásd _point_weights), különben a nagy látószögű, távoli vonalakat mutató
    # hipotézisek nyernének. A pontosság mellett a 'lefedettség' (a felismert
    # vonalhossz mekkora részét magyarázza a hipotézis) is számít.
    scores = _score_hypotheses(field, Gs, pts_coarse, img_w, img_h, px=px, min_visible=min_visible)
    order = np.argsort(-scores)
    chosen = []
    for k in order:
        if scores[k] <= 0 or len(chosen) >= max_candidates:
            break
        p, t, hf, cp = hyps[k]
        if any(abs(p - c[0]) <= 4 and abs(t - c[1]) <= 4 and c[3] == cp for c in (hyps[j] for j in chosen)):
            continue
        chosen.append(k)
    best = None
    for k in chosen:
        p, t, hf, cp = hyps[k]
        f0 = 0.5 * img_w / math.tan(math.radians(hf) / 2)
        theta0 = np.array([p, t, 0.0, f0, cp[0], cp[1], cp[2]], np.float64)
        G1, _ = refine_camera(field, theta0, img_w, img_h, pts_fine, tangents,
                              taus=tuple(v * px for v in (16.0, 8.0, 4.0)), iters=12)
        if G1 is None:
            continue
        G2, _ = refine_homography(field, G1, pts_fine, taus=tuple(v * px for v in (4.0, 2.0)), iters=8,
                                  tangents=tangents)
        if G2 is not None:
            G1 = G2
        prec, rec, n = evaluate_fit(field, G1, pts_fine, tangents)
        fit = 2 * prec * rec / max(prec + rec, 1e-9)
        if best is None or fit > best[0]:
            best = (fit, G1, prec, rec, n, hyps[k], float(scores[k]))
    info = {"hypotheses": len(Gs), "candidates": len(chosen)}
    if best is None:
        return None, info
    fit, G1, prec, rec, n, hyp, cs = best
    info.update({"fit": fit, "precision": prec, "recall": rec, "n_visible": n,
                 "pan": hyp[0], "tilt": hyp[1], "hfov": hyp[2], "coarse_score": cs})
    if not _plausible(G1, img_w, img_h):
        info["reason"] = "valószínűtlen geometria"
        return None, info
    return G1, info


CAMERA_PRIOR = {
    # Veo ajánlás: a felezővonal vonalában, az oldalvonal mögött, 3–10 m magasan
    "x": (PITCH_LENGTH / 2, 12.0),      # (várható érték, szórás) méterben
    "y_min": PITCH_WIDTH + 1.0, "y_max": PITCH_WIDTH + 25.0,
    "h_min": 2.5, "h_max": 15.0,
}


def refine_camera(field, theta, img_w, img_h, pts, tangents=None, taus=(16.0, 8.0, 4.0, 2.0), iters=8,
                  step=0.5, min_points=40, prior_weight=0.05):
    """
    Fizikai kameramodell illesztése (pan, tilt, roll [fok], fókusz [px], kamera x, y, z [m]):
    7 szabadsági fok, négyzetes pixel, középen lévő optikai tengely. Az általános
    8 paraméteres homográfiánál kevésbé tud a vonal nélküli képrészeken torzulni,
    és a kamera helyére előzetes tudást (CAMERA_PRIOR) is adhatunk.
    Numerikus Jacobi-mátrix, Levenberg–Marquardt. Visszatér: (G, theta) vagy (None, theta).
    """
    def G_of(th):
        return camera_ground_homography(th[0], th[1], th[3], img_w, img_h, (th[4], th[5], th[6]), roll_deg=th[2])

    def prior_res(th):
        cp = CAMERA_PRIOR
        r = [(th[4] - cp["x"][0]) / cp["x"][1], th[2] / 3.0]
        y, hgt = th[5], -th[6]
        r.append(max(0.0, cp["y_min"] - y, y - cp["y_max"]) / 2.0)
        r.append(max(0.0, cp["h_min"] - hgt, hgt - cp["h_max"]) / 1.0)
        return np.array(r)

    th = np.asarray(theta, np.float64).copy()
    G = G_of(th)
    uv, wz = project_points(G, pts)
    vis = field.inside(uv, wz, margin=2.0)
    if np.count_nonzero(vis) < min_points:
        return None, th
    P = pts[vis]
    Wp = _point_weights(G, P, None if tangents is None else tangents[vis], step)
    Wn = Wp / np.sum(Wp)
    eps = np.array([0.05, 0.05, 0.05, 0.002 * th[3], 0.05, 0.05, 0.05])

    def dists(th_, tau):
        uv_, wz_ = project_points(G_of(th_), P)
        ins = field.inside(uv_, wz_)
        d = np.full(len(P), tau, np.float64)
        if ins.any():
            d[ins] = np.minimum(field.dist(uv_[ins]), tau)
        return d

    def total(th_, tau):
        d = dists(th_, tau)
        return float(np.sum(Wn * d * d) / (tau * tau) + prior_weight * np.sum(prior_res(th_) ** 2))

    for tau in taus:
        lam = 1e-2
        cost = total(th, tau)
        for _ in range(iters):
            d = dists(th, tau)
            J = np.zeros((len(P), 7))
            Jp = np.zeros((4, 7))
            pr = prior_res(th)
            for j in range(7):
                t2 = th.copy()
                t2[j] += eps[j]
                J[:, j] = (dists(t2, tau) - d) / eps[j]
                Jp[:, j] = (prior_res(t2) - pr) / eps[j]
            s = 0.5 * tau
            wgt = Wn * (d < tau) / (1.0 + (d / s) ** 2) / (tau * tau)
            A = J.T @ (J * wgt[:, None]) + prior_weight * Jp.T @ Jp
            b = J.T @ (d * wgt) + prior_weight * Jp.T @ pr
            improved = False
            for _ in range(5):
                Ad = A + lam * np.diag(np.diag(A) + 1e-12)
                try:
                    delta = -np.linalg.solve(Ad, b)
                except np.linalg.LinAlgError:
                    break
                t2 = th + delta
                c2 = total(t2, tau)
                if c2 < cost:
                    th, cost = t2, c2
                    lam = max(lam / 3.0, 1e-6)
                    improved = True
                    break
                lam *= 10.0
            if not improved:
                break
    return G_of(th), th


def _coarse_step(pts):
    """A mintapontok lépésköze (a pitch_model_points hívásból visszafejtve)."""
    d = np.linalg.norm(np.diff(pts[:40], axis=0), axis=1)
    return float(np.median(d)) if len(d) else 1.0


def _plausible(G, img_w, img_h):
    """A kép alsó felének pontjai a kamera előtt legyenek és a pálya közelébe essenek."""
    try:
        H = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return False
    probe = np.array([[img_w * a, img_h * b] for a in (0.1, 0.5, 0.9) for b in (0.6, 0.95)])
    X = H[:, :2] @ probe.T + H[:, 2:3]
    if np.any(np.abs(X[2]) < 1e-12):
        return False
    xy = (X[:2] / X[2]).T
    # A G a pálya pontjait a kamera elé vetíti (w > 0); a kép pontjainak visszavetítése is legyen konzisztens
    _, wz = project_points(G, xy)
    if np.any(wz <= 0):
        return False
    return bool(np.all((xy[:, 0] > -60) & (xy[:, 0] < PITCH_LENGTH + 60)
                       & (xy[:, 1] > -60) & (xy[:, 1] < PITCH_WIDTH + 60)))


# ---------------------------------------------------------------------------
# Fizikai kameramodell: rögzített kamerahely, képkockánként pan / tilt / roll / fókusz
# ---------------------------------------------------------------------------
#
# A Veo kamera egy árbocon áll, a helye a felvétel alatt nem változik; a virtuális
# kamera csak forog (pásztáz, dől) és zoomol. A képkockánkénti állapot ezért
#     x = (pan [fok], tilt [fok], roll [fok], 100·ln f)      (f: fókusz, teljes felbontású px)
# a kamera helye (C, méter, z lefelé) pedig egyetlen, lassan pontosított becslés.
# A képkockák közti mozgás így a K2·R2·R1^T·K1^-1 alakú (tiszta forgatás + zoom) homográfia –
# 8 helyett 4 szabadsági fok, ami a sodródást és a "lehetetlen" (az égbe forduló, nyírt)
# homográfiákat nagyrészt kizárja.

def x_focal(x):
    return math.exp(float(x[3]) / 100.0)


def x_to_G(x, C, img_w, img_h):
    """Állapot + kamerahely -> pálya -> (teljes felbontású) kép homográfia."""
    return camera_ground_homography(x[0], x[1], x_focal(x), img_w, img_h, C, roll_deg=x[2])


def rotation_motion(x1, x2, img_w, img_h):
    """Két kameraállás közti képtér-mozgás (teljes felbontás): az x1 kép pixelét az x2 kép pixelébe viszi."""
    K1 = camera_K(x_focal(x1), img_w, img_h)
    K2 = camera_K(x_focal(x2), img_w, img_h)
    R1 = camera_rotation(x1[0], x1[1], x1[2])
    R2 = camera_rotation(x2[0], x2[1], x2[2])
    return K2 @ R2 @ R1.T @ np.linalg.inv(K1)


def _rotation_to_angles(R):
    """camera_rotation inverze: (pan, tilt, roll) fokban."""
    fwd = R[2] / np.linalg.norm(R[2])
    pan = math.degrees(math.atan2(fwd[0], -fwd[1]))
    tilt = math.degrees(math.asin(float(np.clip(fwd[2], -1.0, 1.0))))
    right0 = np.cross(fwd, np.array([0.0, 0.0, -1.0]))
    nr = np.linalg.norm(right0)
    if nr < 1e-9:
        return pan, tilt, 0.0
    right0 /= nr
    down0 = np.cross(fwd, right0)
    roll = math.degrees(math.atan2(float(R[0] @ down0), float(R[0] @ right0)))
    return pan, tilt, roll


def _transform(F, p):
    X = p @ F[:, :2].T + F[:, 2]
    w = X[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return X[:, :2] / w


def _lm(res_fn, x0, eps, iters=30, lam=1e-3, tol=1e-10):
    """Kis Levenberg–Marquardt numerikus Jacobi-mátrixszal (néhány paraméterhez)."""
    x = np.asarray(x0, np.float64).copy()
    r = res_fn(x)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            x2 = x.copy()
            x2[j] += eps[j]
            J[:, j] = (res_fn(x2) - r) / eps[j]
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(8):
            try:
                dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            except np.linalg.LinAlgError:
                break
            x2 = x + dx
            r2 = res_fn(x2)
            c2 = float(r2 @ r2)
            if np.isfinite(c2) and c2 < cost:
                done = cost - c2 < tol * max(cost, 1e-12)
                x, r, cost = x2, r2, c2
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved or done:
            break
    return x, cost


def decompose_camera(G, img_w, img_h):
    """
    Zárt alakú felbontás: G (pálya -> teljes kép) = K·[r1 r2 -R·C], négyzetes pixel, a kép
    közepén lévő optikai tengely. Visszatér: (x, C) vagy None.
    """
    G = orient_G(G, img_w, img_h)
    Kc = np.array([[1.0, 0, img_w / 2.0], [0, 1.0, img_h / 2.0], [0, 0, 1.0]])
    M = np.linalg.inv(Kc) @ G
    g1, g2 = M[:, 0], M[:, 1]
    # r1 ⟂ r2 és |r1| = |r2|:  a·w + b = 0, w = 1/f²
    a = np.array([g1[0] * g2[0] + g1[1] * g2[1], g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2])
    b = np.array([g1[2] * g2[2], g1[2] ** 2 - g2[2] ** 2])
    if float(a @ a) < 1e-24:
        return None
    w = -float(a @ b) / float(a @ a)
    if not np.isfinite(w) or w <= 0:
        return None
    f = 1.0 / math.sqrt(w)
    Mn = np.diag([1.0 / f, 1.0 / f, 1.0]) @ M
    lam = 0.5 * (np.linalg.norm(Mn[:, 0]) + np.linalg.norm(Mn[:, 1]))
    r1, r2, t = Mn[:, 0] / lam, Mn[:, 1] / lam, Mn[:, 2] / lam
    Rc = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(Rc)
    Rc = U @ Vt
    if np.linalg.det(Rc) < 0:
        return None
    C = -Rc.T @ t
    pan, tilt, roll = _rotation_to_angles(Rc)
    return np.array([pan, tilt, roll, 100.0 * math.log(f)]), C


def fit_camera_to_homography(H_full, img_w, img_h):
    """
    Fizikai kamera (pan, tilt, roll, fókusz, kamerahely) illesztése egy kép -> pálya homográfiához
    (pl. a kézi kalibrációhoz). A képen látható pályaterület rácspontjain a kamera-vetítés és a
    homográfia eltérését minimalizálja. Visszatér: (x, C, info) vagy None.
    """
    try:
        G = orient_G(np.linalg.inv(np.asarray(H_full, np.float64)), img_w, img_h)
    except np.linalg.LinAlgError:
        return None
    uu, vv = np.meshgrid(np.linspace(0.03, 0.97, 17) * img_w, np.linspace(0.03, 0.97, 11) * img_h)
    img = np.column_stack([uu.ravel(), vv.ravel()])
    Hn = np.linalg.inv(G)
    X = Hn[:, :2] @ img.T + Hn[:, 2:3]
    ok = np.abs(X[2]) > 1e-12
    pit = (X[:2] / np.where(ok, X[2], 1.0)).T
    _, wz = project_points(G, pit)
    ok &= wz > 0
    inside = ok & (pit[:, 0] > -3) & (pit[:, 0] < PITCH_LENGTH + 3) & (pit[:, 1] > -3) & (pit[:, 1] < PITCH_WIDTH + 3)
    if inside.sum() < 12:
        inside = ok & (np.abs(pit[:, 0] - PITCH_LENGTH / 2) < 120) & (np.abs(pit[:, 1] - PITCH_WIDTH / 2) < 100)
    if inside.sum() < 8:
        return None
    img, pit = img[inside], pit[inside]
    dec = decompose_camera(G, img_w, img_h)
    if dec is None:
        return None
    x0, C0 = dec
    y0 = np.concatenate([x0, C0])
    sc = img_w / 1280.0

    def res(y):
        uv, w_ = project_points(x_to_G(y[:4], y[4:], img_w, img_h), pit)
        r = ((uv - img) / sc).ravel()
        r = np.where(np.isfinite(r), r, 1e3)
        # gyenge előzetes: kis roll, a kamera a pálya fölött (nem alatta)
        return np.concatenate([r, [y[2] / 10.0, max(0.0, y[6] + 1.0) * 10.0]])

    y, _ = _lm(res, y0, eps=np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]), iters=50)
    rr = res(y)[:-2].reshape(-1, 2)
    resid = np.linalg.norm(rr, axis=1)
    info = {"resid_med_px": float(np.median(resid)), "resid_max_px": float(resid.max()),
            "height_m": float(-y[6]), "focal_px": x_focal(y[:4])}
    return y[:4], y[4:], info


def camera_plausible(x, C, info=None):
    """A kamerahely és -állás fizikailag értelmes-e (Veo: az oldalvonal mögött, 1–40 m magasan)."""
    h = -float(C[2])
    d = math.hypot(float(C[0]) - PITCH_LENGTH / 2, float(C[1]) - PITCH_WIDTH / 2)
    ok = 1.0 <= h <= 40.0 and d <= 150.0 and -5.0 <= x[1] <= 70.0 and abs(x[2]) <= 15.0
    if info is not None:
        ok = ok and info.get("resid_med_px", 0.0) <= 25.0   # zajos (néhány pontos) kézi kalibráció is mehet
    return bool(ok)


def model_line_ids(step=0.5):
    """A pitch_model_points pontjaihoz tartozó vonal-sorszámok (melyik modellvonal pontja)."""
    return np.concatenate([np.full(len(p), i, np.int32) for i, p in enumerate(pitch_polylines(step))])


def refine_pose_lines(field, y_pred, Pinv, G_of, pts, tangents, eps, px=1.0, taus=(8.0, 4.0, 2.0),
                      iters=5, n_eff=30.0, min_points=40, step=0.5, line_ids=None):
    """
    MAP-illesztés a vonalakra: a vetített modellpontok (robosztus, Cauchy) távolság-költsége +
    a mozgásbecslésből jövő előzetes (y_pred, Pinv információs mátrix). A rosszul meghatározott
    irányokban (pl. csak egy vonal látszik) a predikció marad.
    G_of(y): paraméterek -> pálya -> vonal-felbontású kép homográfia.
    n_eff: a vonal-mérés 'effektív pontszáma' (a pontok erősen korreláltak); px: pixel-lépték.
    Visszatér: (y_új, info) vagy (None, info), ha kevés a látható pont.
    """
    y_pred = np.asarray(y_pred, np.float64)
    y = y_pred.copy()
    G = G_of(y)
    uv, wz = project_points(G, pts)
    vis = field.inside(uv, wz, margin=2.0)
    P = pts[vis]
    info = {"n_visible": int(len(P))}
    if len(P) < min_points:
        return None, info
    T = None if tangents is None else tangents[vis]
    Wp = _point_weights(G, P, T, step)
    Wn = Wp / np.sum(Wp)
    k = n_eff / (px * px)

    def dists(G_, tau):
        uv_, wz_ = project_points(G_, P)
        ins = field.inside(uv_, wz_)
        d = np.full(len(P), tau, np.float64)
        if ins.any():
            d[ins] = np.minimum(field.dist(uv_[ins]), tau)
        return d, uv_, ins

    def total(y_, tau):
        d, _, _ = dists(G_of(y_), tau)
        s = 0.5 * tau
        dy = y_ - y_pred
        return float(k * np.sum(Wn * s * s * np.log1p((d / s) ** 2)) + dy @ Pinv @ dy)

    d0, _, _ = dists(G, max(taus))
    info["precision_before"] = float(np.sum(Wn * (d0 < 2.0 * px)))
    A = Pinv.copy()
    for tau in taus:
        tau = tau * px
        lam = 1e-3
        cost = total(y, tau)
        for _ in range(iters):
            Gc = G_of(y)
            d, uv, ins = dists(Gc, tau)
            sel = ins & (d < tau)
            if np.count_nonzero(sel) < min_points // 2:
                break
            Ju = np.empty((len(P), len(y)))
            Jv = np.empty((len(P), len(y)))
            for j in range(len(y)):
                y2 = y.copy()
                y2[j] += eps[j]
                uv2, _ = project_points(G_of(y2), P)
                Ju[:, j] = (uv2[:, 0] - uv[:, 0]) / eps[j]
                Jv[:, j] = (uv2[:, 1] - uv[:, 1]) / eps[j]
            us, vs = uv[sel, 0], uv[sel, 1]
            gx = _bilinear(field.gx, us, vs)
            gy = _bilinear(field.gy, us, vs)
            J = gx[:, None] * Ju[sel] + gy[:, None] * Jv[sel]
            s = 0.5 * tau
            wc = Wn[sel] / (1.0 + (d[sel] / s) ** 2)
            A = k * (J.T @ (J * wc[:, None])) + Pinv
            b = k * (J.T @ (d[sel] * wc)) + Pinv @ (y - y_pred)
            improved = False
            for _ in range(5):
                try:
                    dy = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), b)
                except np.linalg.LinAlgError:
                    break
                y2 = y + dy
                c2 = total(y2, tau)
                if c2 < cost:
                    y, cost = y2, c2
                    lam = max(lam / 3.0, 1e-7)
                    improved = True
                    break
                lam *= 10.0
            if not improved:
                break
    G1 = G_of(y)
    d1, _, ins1 = dists(G1, max(taus) * px)
    good = d1 < 2.0 * px
    info["precision"] = float(np.sum(Wn * good))
    info["mean_dist"] = float(np.sum(Wn * d1))
    info["info_matrix"] = A
    if line_ids is not None:
        # a látható modellvonalak (elég hosszú látható résszel) hány %-át támasztják alá vonalpixelek
        ids = line_ids[vis]
        sup, tot = 0.0, 0.0
        for i in np.unique(ids):
            m = ids == i
            wsum = float(np.sum(Wp[m]))
            if wsum < 3.0 * px:      # (képbeli hossz px-ben ~ a súlyok összege)
                continue
            tot += wsum
            if np.sum(Wp[m] * good[m]) >= 0.4 * wsum:
                sup += wsum
        info["line_support"] = sup / tot if tot > 0 else 0.0
        info["visible_len"] = tot
    return y, info


class _KeyField:
    """Kulcskép a kamerahely-becsléshez: csak a vonalmaszk távolság-transzformáltja (kevés memória)."""

    def __init__(self, field):
        self.D = field.D
        self.h, self.w = field.h, field.w

    def inside(self, uv, w, margin=1.0):
        return ((w > 1e-6) & (uv[:, 0] >= margin + 1) & (uv[:, 0] <= self.w - 2 - margin)
                & (uv[:, 1] >= margin + 1) & (uv[:, 1] <= self.h - 2 - margin))

    def dist(self, uv):
        return _bilinear(self.D, uv[:, 0], uv[:, 1])

    def grad(self, uv):
        u, v = uv[:, 0], uv[:, 1]
        gx = 0.5 * (_bilinear(self.D, u + 1.0, v) - _bilinear(self.D, u - 1.0, v))
        gy = 0.5 * (_bilinear(self.D, u, v + 1.0) - _bilinear(self.D, u, v - 1.0))
        return gx, gy


def bundle_adjust_camera(keyframes, C0, C_prior_info, C_center, img_w, img_h, M_line, pts, tangents, px=1.0,
                         taus=(6.0, 3.0, 1.5), iters=8, n_eff=30.0, x_sigma=(3.0, 2.0, 1.0, 15.0), step=0.5):
    """
    A kamerahely (C) közös becslése több kulcsképből (eltérő pásztázási szögek): a vonalak
    távolság-költségének összege minden kulcsképen, kulcsképenként saját (pan, tilt, roll, fókusz)
    állással. Egyetlen képből a kamerahely és a fókusz/dőlés nehezen választható szét
    (hátrébb lépés ~ rázoomolás); több irányból nézve viszont jól meghatározott.
    Gauss–Newton, Levenberg–Marquardt csillapítással; a kulcsképek állását Schur-komplementtel
    elimináljuk (3x3-as rendszer a kamerahelyre).
    keyframes: [{"field": _KeyField, "x": állás}, ...]. Visszatér: (C, [x_k], info).
    """
    K = len(keyframes)
    C = np.asarray(C0, np.float64).copy()
    xs = [np.asarray(kf["x"], np.float64).copy() for kf in keyframes]
    x0s = [x.copy() for x in xs]
    Px = np.diag(1.0 / np.asarray(x_sigma, np.float64) ** 2)
    k = n_eff / (px * px)
    eps = 1e-3
    data = []
    for kf, x in zip(keyframes, xs):
        f = kf["field"]
        G = M_line @ x_to_G(x, C, img_w, img_h)
        uv, wz = project_points(G, pts)
        vis = f.inside(uv, wz, margin=3.0)
        P = pts[vis]
        Wp = _point_weights(G, P, None if tangents is None else tangents[vis], step) if len(P) else np.zeros(0)
        data.append((f, P, Wp / max(np.sum(Wp), 1e-9)))

    def frame_cost(i, x, Cc, tau):
        f, P, Wn = data[i]
        if len(P) == 0:
            return 0.0
        uv, wz = project_points(M_line @ x_to_G(x, Cc, img_w, img_h), P)
        ins = f.inside(uv, wz)
        d = np.full(len(P), tau)
        if ins.any():
            d[ins] = np.minimum(f.dist(uv[ins]), tau)
        s = 0.5 * tau
        dx = x - x0s[i]
        return float(k * np.sum(Wn * s * s * np.log1p((d / s) ** 2)) + dx @ Px @ dx)

    def total(xs_, Cc, tau):
        dC = Cc - C_center
        return sum(frame_cost(i, xs_[i], Cc, tau) for i in range(K)) + float(dC @ C_prior_info @ dC)

    info = {"keyframes": K}
    for tau in taus:
        tau *= px
        lam = 1e-3
        cost = total(xs, C, tau)
        for _ in range(iters):
            S = C_prior_info.copy()
            r = C_prior_info @ (C - C_center)
            blocks = []
            for i in range(K):
                f, P, Wn = data[i]
                if len(P) < 30:
                    blocks.append(None)
                    continue
                y = np.concatenate([xs[i], C])
                uv, wz = project_points(M_line @ x_to_G(y[:4], y[4:], img_w, img_h), P)
                ins = f.inside(uv, wz)
                d = np.full(len(P), tau)
                d[ins] = f.dist(uv[ins])
                sel = ins & (d < tau)
                if np.count_nonzero(sel) < 20:
                    blocks.append(None)
                    continue
                Ju = np.empty((len(P), 7))
                Jv = np.empty((len(P), 7))
                for j in range(7):
                    y2 = y.copy()
                    y2[j] += eps
                    uv2, _ = project_points(M_line @ x_to_G(y2[:4], y2[4:], img_w, img_h), P)
                    Ju[:, j] = (uv2[:, 0] - uv[:, 0]) / eps
                    Jv[:, j] = (uv2[:, 1] - uv[:, 1]) / eps
                gx, gy = f.grad(uv[sel])
                J = gx[:, None] * Ju[sel] + gy[:, None] * Jv[sel]
                s = 0.5 * tau
                wc = k * Wn[sel] / (1.0 + (d[sel] / s) ** 2)
                A = J.T @ (J * wc[:, None])
                b = J.T @ (d[sel] * wc)
                Axx = A[:4, :4] + Px
                bx = b[:4] + Px @ (xs[i] - x0s[i])
                AxC, ACC, bC = A[:4, 4:], A[4:, 4:], b[4:]
                try:
                    Axx_inv = np.linalg.inv(Axx)
                except np.linalg.LinAlgError:
                    blocks.append(None)
                    continue
                S += ACC - AxC.T @ Axx_inv @ AxC
                r += bC - AxC.T @ Axx_inv @ bx
                blocks.append((Axx, AxC, bx))
            improved = False
            for _ in range(5):
                try:
                    Sd = S + lam * np.diag(np.diag(S) + 1e-9)
                    dC = -np.linalg.solve(Sd, r)
                except np.linalg.LinAlgError:
                    break
                xs2 = []
                for i in range(K):
                    bl = blocks[i]
                    if bl is None:
                        xs2.append(xs[i])
                        continue
                    Axx, AxC, bx = bl
                    Ad = Axx + lam * np.diag(np.diag(Axx))
                    xs2.append(xs[i] - np.linalg.solve(Ad, bx + AxC @ dC))
                C2 = C + dC
                c2 = total(xs2, C2, tau)
                if c2 < cost:
                    xs, C, cost = xs2, C2, c2
                    lam = max(lam / 3.0, 1e-7)
                    improved = True
                    break
                lam *= 10.0
            if not improved:
                break
        info["S"] = S
    info["cost"] = cost
    return C, xs, info


def _score_hypotheses(field, Gs, pts_coarse, img_w, img_h, px=1.0, min_visible=60):
    """
    Kamera-hipotézisek (G-k, pálya -> vonal-felbontású kép) gyors pontozása a vonalmaszk közelségi
    térképén: a vetített modellpontok súlyozott pontossága és a felismert vonalhossz lefedettsége
    (F1). Visszatér: pontszámok (K,).
    """
    tau = 10.0 * px
    Q = np.maximum(0.0, 1.0 - field.D / tau).astype(np.float32)   # közelségi térkép (1 = vonalon)
    step_c = _coarse_step(pts_coarse)
    tan_c = pitch_model_tangents(step_c)
    nrm_c = np.column_stack([-tan_c[:, 1], tan_c[:, 0]])
    d_off = 0.25
    Pc = np.column_stack([pts_coarse, np.ones(len(pts_coarse))]).T                    # (3, M)
    Pt = np.column_stack([pts_coarse + tan_c * d_off, np.ones(len(pts_coarse))]).T
    Pn = np.column_stack([pts_coarse + nrm_c * d_off, np.ones(len(pts_coarse))]).T
    if field.n_long > 0:
        inner = cv2.distanceTransform(field.long_mask, cv2.DIST_L2, 3)
        thickness = max(1.0, 4.0 * float(inner[field.long_mask > 0].mean()))
        line_len = field.n_long / thickness
    else:
        line_len = 1.0
    scores = np.zeros(len(Gs))
    chunk = 300

    def proj(Gc, P3):
        X = Gc @ P3
        wz = X[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return X[:, 0] / wz, X[:, 1] / wz, wz

    for i in range(0, len(Gs), chunk):
        Gc = Gs[i:i + chunk]
        u, v, wz = proj(Gc, Pc)
        ut, vt, _ = proj(Gc, Pt)
        un, vn, _ = proj(Gc, Pn)
        vis = (wz > 1e-6) & (u >= 0) & (u <= img_w - 1) & (v >= 0) & (v <= img_h - 1)
        ui = np.where(vis, u, 0).astype(np.int32)
        vi = np.where(vis, v, 0).astype(np.int32)
        q = np.where(vis, Q[vi, ui], 0.0)
        tx, ty = (ut - u) / d_off, (vt - v) / d_off
        nx, ny = (un - u) / d_off, (vn - v) / d_off
        tl = np.sqrt(tx * tx + ty * ty)
        width_px = np.abs(tx * ny - ty * nx) / np.maximum(tl, 1e-9) * LINE_WIDTH_M
        wgt = np.clip(np.nan_to_num(tl * step_c), 0.1, 4.0 * step_c / 0.5) * \
            np.clip(np.nan_to_num(width_px) / 1.5, 0.05, 1.0) ** 2
        wgt = np.where(vis, wgt, 0.0)
        nvis = vis.sum(axis=1)
        matched = (wgt * q).sum(axis=1)
        prec = matched / np.maximum(wgt.sum(axis=1), 1e-9)
        rec = np.minimum(1.0, matched / line_len)
        s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        s[nvis < min_visible] = 0
        scores[i:i + chunk] = s
    return scores


def search_pose(field, x_center, C, img_w, img_h, M_line, pts_coarse, px=1.0, pan_half=30.0, pan_step=1.5,
                tilt_half=5.0, tilt_step=1.0, zooms=(0.7, 0.8, 0.9, 1.0, 1.12, 1.25, 1.4), top_k=6):
    """
    Újrapozicionálás rögzített kamerahellyel: a (pan, tilt, zoom) rács pontozása a predikció
    körül (kis keresési tér – gyors). Visszatér: a legjobb, egymástól eltérő hipotézisek listája
    [(pontszám, x), ...].
    """
    x_center = np.asarray(x_center, np.float64)
    pans = x_center[0] + np.arange(-pan_half, pan_half + 1e-9, pan_step)
    tilts = x_center[1] + np.arange(-tilt_half, tilt_half + 1e-9, tilt_step)
    tilts = tilts[(tilts > 0.5) & (tilts < 60)]
    xs, Gs = [], []
    for z in zooms:
        lf = x_center[3] + 100.0 * math.log(z)
        for t in tilts:
            for p in pans:
                x = np.array([p, t, x_center[2], lf])
                xs.append(x)
                Gs.append(M_line @ x_to_G(x, C, img_w, img_h))
    scores = _score_hypotheses(field, np.array(Gs), pts_coarse, field.w, field.h, px=px)
    order = np.argsort(-scores)
    out = []
    for kk in order:
        if scores[kk] <= 0 or len(out) >= top_k:
            break
        x = xs[kk]
        if any(abs(x[0] - o[1][0]) <= 3 * pan_step and abs(x[1] - o[1][1]) <= 2 * tilt_step for o in out):
            continue
        out.append((float(scores[kk]), x))
    return out


# ---------------------------------------------------------------------------
# Képkockák közötti kameramozgás
# ---------------------------------------------------------------------------

class MotionEstimator:
    """
    F_t becslése (előző kép pixel -> aktuális kép pixel), a feldolgozási felbontáson.
    GFTT a statikus háttérmaszkon (fű/vonal, dobozok nélkül), LK előre-hátra ellenőrzéssel,
    RANSAC homográfia, majd elfogadási vizsgálatok.
    """

    def __init__(self, max_corners=500, min_inliers=25, ransac_px=1.5, max_scale_step=0.12,
                 max_shear=0.06, max_persp=0.1, max_reproj=1.2):
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.ransac_px = ransac_px
        self.max_scale_step = max_scale_step
        self.max_shear = max_shear
        self.max_persp = max_persp
        self.max_reproj = max_reproj
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def track(self, prev_gray, gray, mask=None, F_init=None):
        """
        Sarokpontok követése: GFTT az előző képen, piramisos LK előre-hátra ellenőrzéssel.
        F_init: a mozgás előrejelzése (előző -> aktuális kép, feldolgozási felbontás) – gyors
        pásztázásnál ebből indul az LK (cv2.OPTFLOW_USE_INITIAL_FLOW), így a nagy elmozdulás sem
        szökik ki a piramis befogási tartományából. Visszatér: (p0, p1, info) vagy (None, None, info).
        """
        h, w = gray.shape[:2]
        pts = cv2.goodFeaturesToTrack(prev_gray, self.max_corners, 0.003, max(6, w // 90),
                                      mask=mask, blockSize=5)
        if (pts is None or len(pts) < self.min_inliers * 2) and mask is not None:
            # Egyszínű fű: tartalék – az egész kép (a dobozok nélkül nem tudjuk, de a RANSAC kiszűri)
            pts = cv2.goodFeaturesToTrack(prev_gray, self.max_corners, 0.003, max(6, w // 90), blockSize=5)
        info = {"features": 0 if pts is None else int(len(pts)), "inliers": 0, "reason": ""}
        if pts is None or len(pts) < self.min_inliers:
            info["reason"] = "kevés jellemzőpont"
            return None, None, info
        pts = pts.astype(np.float32)
        good = None
        if F_init is not None:
            init = cv2.perspectiveTransform(pts, F_init).astype(np.float32)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, init.copy(),
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, pts.copy(),
                                                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
            if good.sum() < max(self.min_inliers, 0.3 * len(pts)):
                good = None      # rossz előrejelzés: előrejelzés nélkül újra
        if good is None:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **self.lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **self.lk)
            fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
        p0 = pts.reshape(-1, 2)[good].astype(np.float64)
        p1 = nxt.reshape(-1, 2)[good].astype(np.float64)
        if len(p0) < self.min_inliers:
            info["reason"] = "kevés követett pont"
            return None, None, info
        return p0, p1, info

    def estimate(self, prev_gray, gray, mask=None, F_init=None):
        h, w = gray.shape[:2]
        p0, p1, info = self.track(prev_gray, gray, mask, F_init)
        if p0 is None:
            return None, info
        F, inl = cv2.findHomography(p0, p1, cv2.RANSAC, self.ransac_px, maxIters=2000, confidence=0.995)
        if F is None:
            info["reason"] = "RANSAC sikertelen"
            return None, info
        inl = inl.reshape(-1).astype(bool)
        info["inliers"] = int(inl.sum())
        info["inlier_ratio"] = float(inl.mean())
        if inl.sum() < self.min_inliers or inl.mean() < 0.3:
            info["reason"] = "kevés inlier"
            return None, info
        # Finomítás az inliereken (legkisebb négyzetek)
        F2, _ = cv2.findHomography(p0[inl], p1[inl], 0)
        if F2 is not None:
            F = F2
        F = F / F[2, 2]
        pr = cv2.perspectiveTransform(p0[inl].reshape(-1, 1, 2), F).reshape(-1, 2)
        err = float(np.median(np.linalg.norm(pr - p1[inl], axis=1)))
        info["reproj"] = err
        # Normalizált koordinátákban vizsgáljuk a skála/nyírás/perspektíva ugrását
        fn = 0.5 * w
        N = np.array([[1 / fn, 0, -0.5 * w / fn], [0, 1 / fn, -0.5 * h / fn], [0, 0, 1]])
        Fn = N @ F @ np.linalg.inv(N)
        Fn = Fn / Fn[2, 2]
        sv = np.linalg.svd(Fn[:2, :2], compute_uv=False)
        scale = math.sqrt(max(sv[0] * sv[1], 1e-12))
        shear = sv[0] / max(sv[1], 1e-12) - 1.0
        persp = float(np.abs(Fn[2, :2]).max())
        info.update({"scale": scale, "shear": shear})
        if abs(scale - 1.0) > self.max_scale_step:
            info["reason"] = f"nagy skálaváltozás ({scale:.3f})"
            return None, info
        if shear > self.max_shear:
            info["reason"] = f"nagy nyírás ({shear:.3f})"
            return None, info
        if persp > self.max_persp:
            info["reason"] = "nagy perspektíva-változás"
            return None, info
        if err > self.max_reproj:
            info["reason"] = f"nagy visszavetítési hiba ({err:.2f} px)"
            return None, info
        return F, info


class RotationMotionEstimator:
    """
    Képkockák közti mozgás rögzített helyű, forgó + zoomoló kamerára: az új állapot
    (pan, tilt, roll, fókusz) 4 paraméteres robosztus illesztése a követett sarokpontokra
    (x_t = K_t·R_t·R_{t-1}^T·K_{t-1}^-1 · x_{t-1}). A mozgó tárgyak (játékosok, nézők) a
    Cauchy-súlyozással kiesnek; az előrejelzés (állandó szögsebesség) az LK kezdőpontja.
    """

    def __init__(self, tracker=None, min_inliers=25, inlier_px=2.0, max_resid=1.6):
        self.tracker = tracker or MotionEstimator()
        self.min_inliers = min_inliers
        self.inlier_px = inlier_px          # teljes felbontású px (1280 széles képre vetítve)
        self.max_resid = max_resid

    def estimate(self, prev_gray, gray, mask, x_prev, x_pred, img_w, img_h, S):
        Sinv = np.linalg.inv(S)
        F_pred = S @ rotation_motion(x_prev, x_pred, img_w, img_h) @ Sinv
        p0, p1, info = self.tracker.track(prev_gray, gray, mask, F_pred / F_pred[2, 2])
        if p0 is None:
            return None, info
        p0f = _transform(Sinv, p0)
        p1f = _transform(Sinv, p1)
        sc = img_w / 1280.0
        K1inv = np.linalg.inv(camera_K(x_focal(x_prev), img_w, img_h))
        R1t = camera_rotation(x_prev[0], x_prev[1], x_prev[2]).T
        B = R1t @ K1inv           # az előző kép pixeléből világ-irány

        def pred(x2):
            M = camera_K(x_focal(x2), img_w, img_h) @ camera_rotation(x2[0], x2[1], x2[2]) @ B
            return _transform(M, p0f)

        x = np.asarray(x_pred, np.float64).copy()
        eps = np.array([1e-3, 1e-3, 1e-3, 1e-3])
        r = None
        for scale in (8.0, 4.0, 2.0, 1.0, 1.0):
            s = scale * sc
            for _ in range(2):
                uv = pred(x)
                r = (uv - p1f).ravel()
                J = np.empty((len(r), 4))
                for j in range(4):
                    x2 = x.copy()
                    x2[j] += eps[j]
                    J[:, j] = ((pred(x2) - p1f).ravel() - r) / eps[j]
                e = np.repeat(np.linalg.norm((uv - p1f), axis=1), 2)
                wgt = 1.0 / (1.0 + (e / s) ** 2)
                A = J.T @ (J * wgt[:, None])
                b = J.T @ (r * wgt)
                try:
                    dx = -np.linalg.solve(A + 1e-6 * np.diag(np.diag(A) + 1e-9), b)
                except np.linalg.LinAlgError:
                    break
                x = x + dx
        e = np.linalg.norm(pred(x) - p1f, axis=1) / sc
        inl = e < self.inlier_px
        info["inliers"] = int(inl.sum())
        info["inlier_ratio"] = float(inl.mean())
        info["reproj"] = float(np.median(e[inl])) if inl.any() else float("inf")
        if inl.sum() < self.min_inliers or inl.mean() < 0.3:
            # Rossz kezdőpont (hirtelen irányváltás): RANSAC homográfia az inlierekhez, azokon újraillesztés
            F, m = cv2.findHomography(p0, p1, cv2.RANSAC, 1.5, maxIters=2000, confidence=0.995)
            if F is None or m is None or m.sum() < self.min_inliers:
                info["reason"] = "kevés inlier"
                return None, info
            m = m.reshape(-1).astype(bool)
            q0, q1 = p0f[m], p1f[m]

            def res(x2):
                M = camera_K(x_focal(x2), img_w, img_h) @ camera_rotation(x2[0], x2[1], x2[2]) @ B
                return (_transform(M, q0) - q1).ravel()
            x, _ = _lm(res, np.asarray(x_prev, np.float64), eps, iters=20)
            e = np.linalg.norm(pred(x) - p1f, axis=1) / sc
            inl = e < self.inlier_px
            info["inliers"] = int(inl.sum())
            info["inlier_ratio"] = float(inl.mean())
            info["reproj"] = float(np.median(e[inl])) if inl.any() else float("inf")
            info["fallback"] = True
            if inl.sum() < self.min_inliers or inl.mean() < 0.25:
                info["reason"] = "kevés inlier"
                return None, info
        if info["reproj"] > self.max_resid:
            info["reason"] = f"nagy visszavetítési hiba ({info['reproj']:.2f} px)"
            return None, info
        d = x - np.asarray(x_prev)
        if abs(d[0]) > 8.0 or abs(d[1]) > 4.0 or abs(d[2]) > 2.0 or abs(d[3]) > 15.0:
            info["reason"] = "valószínűtlen ugrás"
            return None, info
        return x, info


# ---------------------------------------------------------------------------
# A követő
# ---------------------------------------------------------------------------

class PitchTracker:
    """
    Képkockánkénti homográfia mozgó kamerához.

    Használat:
        tracker = PitchTracker((width, height), grass_lower, grass_upper, H0=H)   # H0 lehet None
        for frame in ...:
            H_t, q = tracker.update(frame, boxes)   # boxes: [(x1, y1, x2, y2), ...] teljes felbontásban
            if H_t is not None: ...

    A visszaadott H_t teljes felbontású kép pixel -> pálya méter (mint a régi statikus H). Csak
    megbízható ('ok' státuszú) képkockán ad H-t; 'bizonytalan', 'elveszett' és 'kalibráció
    szükséges' esetén None (a belső becslés a current_H()-val ekkor is lekérdezhető).
    q: minőség-szótár (status, conf, inliers, lost, refined, source, err_m, ...).

    Két üzemmód:
      * 'kamera' (alap): rögzített helyű, forgó + zoomoló kamera (Veo). A kalibrációból a kamera
        helyét (C) és állását (pan, tilt, roll, fókusz) becsüljük; képkockánként csak az
        állás változik (4 paraméter), a kamerahelyet a jól illeszkedő képkockák lassan pontosítják.
      * 'homográfia' (tartalék): ha a kalibráció nem írható le ilyen kamerával (pl. nem Veo,
        hanem mozgó kézi kamera), az általános 8 paraméteres homográfiát követjük (régi módszer).
    """

    def __init__(self, frame_size, grass_lower=None, grass_upper=None, H0=None, proc_width=640,
                 line_width=960, refine_every=3, auto_init=True, auto_init_every=50, auto_init_max_every=250, max_lost=15,
                 relocalize_after=75, min_conf=0.35, refine_gain=0.5):
        self.W, self.H = int(frame_size[0]), int(frame_size[1])
        self.s = min(1.0, float(proc_width) / max(1, self.W))
        self.ws, self.hs = int(round(self.W * self.s)), int(round(self.H * self.s))
        # a tényleges (egész pixelre kerekített) méretarány tengelyenként
        self.S = scale_matrix(self.ws / float(self.W), self.hs / float(self.H))
        # A vonalkereséshez nagyobb felbontás kell (a távoli vonal 1 px körüli)
        self.sl = min(1.0, float(max(line_width, proc_width)) / max(1, self.W))
        self.wl, self.hl = int(round(self.W * self.sl)), int(round(self.H * self.sl))
        self.SL = scale_matrix(self.wl / float(self.W), self.hl / float(self.H))
        self.P2L = self.SL @ np.linalg.inv(self.S)      # feldolgozási -> vonal-felbontás
        self.L2P = np.linalg.inv(self.P2L)
        self.px = self.wl / 640.0                          # pixel-küszöbök léptéke
        if grass_lower is None or grass_upper is None or is_full_range(np.asarray(grass_lower), np.asarray(grass_upper)):
            grass_lower, grass_upper = DEFAULT_GRASS_HSV
        self.grass_lower = np.asarray(grass_lower)
        self.grass_upper = np.asarray(grass_upper)
        self.refine_every = max(1, int(refine_every))
        self.auto_init = auto_init
        self.auto_init_every = max(1, int(auto_init_every))
        self.auto_init_max_every = max(self.auto_init_every, int(auto_init_max_every))
        self._auto_wait = self.auto_init_every
        self.max_lost = max_lost
        self.relocalize_after = relocalize_after
        self.min_conf = min_conf
        self.refine_gain = refine_gain
        self.motion = MotionEstimator()
        self.rot_motion = RotationMotionEstimator(self.motion)
        self.pts_fine = pitch_model_points(0.5)
        self.tan_fine = pitch_model_tangents(0.5)
        self.ids_fine = model_line_ids(0.5)
        self.pts_coarse = pitch_model_points(1.5)
        # --- önértékelés / fizikai modell hangolása ---
        self.conf_ok = 0.6           # ennél kisebb (simított) vonalegyezés: 'bizonytalan'
        self.conf_lost = 0.3         # ennél kisebb: újrapozicionálás
        self.err_ok_m = 2.5          # a becsült (kovariancia alapú) helyzethiba felső határa 'ok'-hoz
        self.stale_frames = 75       # ennyi képkocka vonal-megerősítés nélkül: 'bizonytalan'
        self.pose_floor = (0.25, 0.25, 0.25, 1.5)   # (fok, fok, fok, 100·ln f) – lásd _check_pose
        self.keyframe_gap = 25       # legalább ennyi képkocka két kulcskép között (kamerahely-becslés)
        self.max_keyframes = 10
        self.reloc_every = 8         # újrapozicionálási kísérletek gyakorisága
        self.reloc_max_fails = 4     # ennyi sikertelen újrapozicionálás után teljes automatikus kalibráció
        self.reloc_fails = 0
        # --- állapot ---
        self.mode = None             # 'kamera' | 'homográfia' | None
        self.x = None                # kameraállás (pan, tilt, roll, 100·ln f)
        self.C = None                # kamerahely (m, z lefelé)
        self.P = None                # az állás kovarianciája (4x4)
        self.v = np.zeros(4)         # állandó-szögsebesség előrejelzés
        self.G = None                # pálya -> feldolgozási kép
        self.source = None
        self.frame_idx = -1
        self.prev_gray = None
        self.last_F = None       # utolsó képkockák közti mozgás (feldolgozási felbontás), lásd frame_motion()
        self.prev_mask = None
        self.lost = 0            # egymás utáni sikertelen mozgásbecslések / rossz illesztések
        self.since_refine = 0
        self.since_verified = 0  # képkockák a legutóbbi jó vonal-megerősítés óta
        self.keyframes = []      # kulcsképek a kamerahely becsléséhez
        self.last_keyframe = -10 ** 9
        self.C_calib = None      # a kalibrációból becsült kamerahely (az előzetes középpontja)
        self.C_prior_info = None
        self.bad_streak = 0      # egymás utáni rossz vonal-ellenőrzések
        self.last_reloc_try = -10 ** 9
        self.last_auto_try = None
        self.conf = 0.0
        self.last_info = {}
        if H0 is not None:
            self.set_calibration(H0)

    # --- kalibráció -------------------------------------------------------
    def set_calibration(self, H_full, source="kézi"):
        """Kézi (vagy külső) kalibráció beállítása az aktuális képkockára: H teljes felbontású kép -> pálya."""
        if H_full is None:
            return
        H_full = np.asarray(H_full, np.float64)
        self.source = source
        self.lost = 0
        self.bad_streak = 0
        self.conf = max(self.conf, 0.6)
        self.since_refine = self.refine_every   # az első képkockán rögtön finomítunk
        self.since_verified = 0
        fit = fit_camera_to_homography(H_full, self.W, self.H)
        if fit is not None and camera_plausible(fit[0], fit[1], fit[2]):
            x, C, info = fit
            # a kézi kalibráció pontatlansága: a kamerahely néhány méteres, az állás tized fokos hibájú lehet
            self._set_pose(x, C, P=np.diag([0.3, 0.3, 0.3, 2.0]) ** 2, C_sigma=5.0)
            self.last_info = {"camera_fit": info}
        else:
            self.mode = "homográfia"
            self.x = None
            self.C = None
            G_full = np.linalg.inv(H_full)
            self.G = orient_G(self.S @ G_full, self.ws, self.hs)

    def _set_pose(self, x, C, P=None, C_sigma=None):
        self.mode = "kamera"
        self.x = np.asarray(x, np.float64).copy()
        self.C = np.asarray(C, np.float64).copy()
        if P is not None:
            self.P = np.asarray(P, np.float64)
        if C_sigma is not None:
            self.C_calib = self.C.copy()
            self.C_prior_info = np.eye(3) / C_sigma ** 2
            self.keyframes = []
        self.v = np.zeros(4)
        self._update_G()

    def _update_G(self):
        if self.mode == "kamera" and self.x is not None:
            self.G = orient_G(self.S @ x_to_G(self.x, self.C, self.W, self.H), self.ws, self.hs)

    def frame_motion(self):
        """
        A legutóbbi update() képkockák közti mozgása teljes felbontásban: 3x3 homográfia, ami az
        ELŐZŐ kép pixelét az AKTUÁLIS kép pixelébe viszi (a vonal-finomítás korrekciója nélkül –
        tisztán a képtartalom mozgása; a forgó kameránál a levegőben lévő labdára is érvényes).
        None, ha nincs előző kép vagy a mozgásbecslés sikertelen volt.
        """
        F = getattr(self, "last_F", None)
        if F is None:
            return None
        Ff = np.linalg.inv(self.S) @ F @ self.S
        return Ff / Ff[2, 2] if abs(Ff[2, 2]) > 1e-12 else Ff

    def current_H(self):
        """Teljes felbontású kép -> pálya homográfia (vagy None) – a belső becslés, státusztól függetlenül."""
        if self.G is None:
            return None
        Gf = np.linalg.inv(self.S) @ self.G
        H = np.linalg.inv(Gf)
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    def camera_state(self):
        """A fizikai kameramodell állapota (vagy None): pan, tilt, roll (fok), fókusz (px), kamerahely (m)."""
        if self.mode != "kamera" or self.x is None:
            return None
        return {"pan": float(self.x[0]), "tilt": float(self.x[1]), "roll": float(self.x[2]),
                "focal_px": x_focal(self.x), "position": [float(v) for v in self.C],
                "height_m": float(-self.C[2])}

    # --- előfeldolgozás ---------------------------------------------------
    def _prepare(self, frame):
        small = cv2.resize(frame, (self.ws, self.hs), interpolation=cv2.INTER_AREA) if self.s < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        grass, _ = build_grass_mask(hsv, self.grass_lower, self.grass_upper, scale=1.0)
        return small, gray, hsv, grass

    def _line_field(self, frame, small, hsv, grass, box_mask, boxes):
        """Vonalmaszk + távolság-transzformált a vonal-felbontáson."""
        if (self.wl, self.hl) == (self.ws, self.hs):
            return _LineField(detect_line_mask(small, grass, box_mask, hsv))
        img = cv2.resize(frame, (self.wl, self.hl), interpolation=cv2.INTER_AREA) if self.sl < 1.0 else frame
        g = None if grass is None else cv2.resize(grass, (self.wl, self.hl), interpolation=cv2.INTER_NEAREST)
        bm = _boxes_mask(img.shape, boxes, self.sl)
        return _LineField(detect_line_mask(img, g, bm))

    def _feature_mask(self, grass, box_mask):
        if grass is None:
            m = np.full((self.hs, self.ws), 255, np.uint8)
        else:
            kd = _odd(max(9, self.ws / 30))
            m = cv2.dilate(grass, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        if self.mode == "kamera" and self.G is not None:
            # Csak a talajsík a pálya körül: a fűszínű fasor / lelátó, a képre égetett feliratok
            # (eredményjelző) és a horizont feletti részek nem a kamera forgásával mozognak
            g = self._ground_region_mask()
            if g is not None:
                m2 = m & g
                if np.count_nonzero(m2) > 0.05 * m2.size:
                    m = m2
        m[box_mask > 0] = 0
        return m

    def _ground_region_mask(self, margin=4.0, cell=4):
        """A pálya (+ margin m) talajsík-tartománya a feldolgozási képen (255), a jelenlegi G alapján."""
        gw, gh = (self.ws + cell - 1) // cell, (self.hs + cell - 1) // cell
        key = (gw, gh)
        if getattr(self, "_grid_key", None) != key:
            uu, vv = np.meshgrid((np.arange(gw) + 0.5) * cell, (np.arange(gh) + 0.5) * cell)
            self._grid_pts = np.stack([uu.ravel(), vv.ravel(), np.ones(gw * gh)])
            self._grid_key = key
        try:
            Hn = np.linalg.inv(self.G)
        except np.linalg.LinAlgError:
            return None
        X = Hn @ self._grid_pts
        w = X[2]
        ok = np.abs(w) > 1e-12
        xy = X[:2] / np.where(ok, w, 1.0)
        # a talajpont a kamera előtt van: G·(x, y, 1) harmadik komponense pozitív (orient_G)
        wz = self.G[2, 0] * xy[0] + self.G[2, 1] * xy[1] + self.G[2, 2]
        ok &= (wz > 0) & (xy[0] > -margin) & (xy[0] < PITCH_LENGTH + margin) & \
            (xy[1] > -margin) & (xy[1] < PITCH_WIDTH + margin)
        m = (ok.reshape(gh, gw) * 255).astype(np.uint8)
        return cv2.resize(m, (self.ws, self.hs), interpolation=cv2.INTER_NEAREST)

    # --- fő lépés -----------------------------------------------------------
    def update(self, frame, boxes=None):
        """Egy képkocka feldolgozása. Visszatér: (H_t vagy None, minőség-szótár)."""
        self.frame_idx += 1
        small, gray, hsv, grass = self._prepare(frame)
        box_mask = _boxes_mask(gray.shape, boxes, self.s)
        q = {"frame": self.frame_idx, "refined": False, "motion_ok": None, "refine_tried": False}

        # 1) Terjesztés a képkockák közti mozgással
        motion_failed = False
        self.last_F = None     # a labdakövetőnek: képtér-mozgás (előző -> aktuális kép), lásd frame_motion()
        if self.prev_gray is not None:
            if self.mode == "kamera":
                motion_failed = not self._propagate_pose(gray, q)
            else:
                motion_failed = not self._propagate_homography(gray, q)
        self.prev_gray = gray
        self.prev_mask = self._feature_mask(grass, box_mask)

        field = None
        # 2) Sodródás-javítás és önellenőrzés a vonalakkal
        if self.G is not None:
            self.since_refine += 1
            self.since_verified += 1
            need = (self.since_refine >= self.refine_every or motion_failed or self.lost > 0
                    or self.conf < max(self.min_conf, self.conf_ok if self.mode == "kamera" else -1.0)
                    or self.bad_streak > 0)
            if need:
                field = self._line_field(frame, small, hsv, grass, box_mask, boxes)
                if self.mode == "kamera":
                    self._check_pose(field, q)
                else:
                    self._refine(field, q, wide=self.lost > 0)
            if self.mode == "kamera" and (self.bad_streak >= 4 or self.lost > self.max_lost) \
                    and self.frame_idx - self.last_reloc_try >= self.reloc_every:
                if field is None:
                    field = self._line_field(frame, small, hsv, grass, box_mask, boxes)
                self._relocalize(field, q)
            if self.lost > self.relocalize_after or (self.mode == "kamera" and self.bad_streak > self.relocalize_after):
                # Tartósan elveszett: újra kell kalibrálni (automatikusan, ha engedélyezett)
                self._reset()

        # 3) Automatikus kezdeti kalibráció
        if self.G is None and self.auto_init and (
                self.last_auto_try is None or self.frame_idx - self.last_auto_try >= self._auto_wait):
            self.last_auto_try = self.frame_idx
            if field is None:
                field = self._line_field(frame, small, hsv, grass, box_mask, boxes)
            if not self.try_auto_calibrate(field, q):
                # Sikertelen próbálkozás után ritkábban próbáljuk (drága: ~1–2 mp)
                self._auto_wait = min(self.auto_init_max_every, int(self._auto_wait * 1.5))
            else:
                self._auto_wait = self.auto_init_every

        q["lost"] = self.lost
        q["conf"] = round(float(self.conf), 3)
        q["source"] = self.source
        q["mode"] = self.mode
        if self.G is None:
            q["status"] = STATUS_NEED_CALIB
            return None, q
        if self.lost > self.max_lost:
            q["status"] = STATUS_LOST
            return None, q
        if self.mode == "kamera":
            err_m = self._pose_error_m()
            q["err_m"] = round(err_m, 2)
            if self.bad_streak >= 4 and self.conf < self.conf_lost:
                q["status"] = STATUS_LOST
                return None, q
            if self.conf < self.conf_ok or err_m > self.err_ok_m or self.since_verified > self.stale_frames:
                q["status"] = STATUS_UNSURE
                return None, q
        q["status"] = STATUS_OK
        return self.current_H(), q

    def _reset(self):
        self.G = None
        self.x = None
        self.C = None
        self.mode = None
        self.source = None
        self.conf = 0.0
        self.lost = 0
        self.bad_streak = 0
        self.last_auto_try = None       # azonnal próbálkozhat az automatikus kalibrációval
        self._auto_wait = self.auto_init_every

    # --- fizikai kameramodell ---------------------------------------------
    def _propagate_pose(self, gray, q):
        x_prev = self.x
        x_pred = x_prev + self.v
        x_new, minfo = self.rot_motion.estimate(self.prev_gray, gray, self.prev_mask, x_prev, x_pred,
                                                self.W, self.H, self.S)
        q["inliers"] = minfo.get("inliers", 0)
        if x_new is not None:
            q["motion_ok"] = True
            d = x_new - x_prev
            self.v = d
            F_full = rotation_motion(x_prev, x_new, self.W, self.H)
            F = self.S @ F_full @ np.linalg.inv(self.S)
            self.last_F = F / F[2, 2]
            # folyamatzaj: gyors mozgásnál és nagy maradéknál a becslés bizonytalanabb
            res = min(2.0, float(minfo.get("reproj", 1.0)))
            sa = 0.004 + 0.02 * np.abs(d[:3]) + 0.01 * res
            sf = 0.02 + 0.03 * abs(d[3]) + 0.02 * res
            Q = np.diag(np.concatenate([sa, [sf]]) ** 2)
            self.x = x_new
            ok = True
        else:
            q["motion_ok"] = False
            q["motion_reason"] = minfo.get("reason", "")
            self.x = x_pred
            Q = np.diag(np.concatenate([0.1 + np.abs(self.v[:3]), [0.5 + abs(self.v[3])]]) ** 2)
            self.v = 0.5 * self.v
            self.lost += 1
            ok = False
        self.P = self.P + Q
        self._update_G()
        return ok

    def _pose_error_m(self):
        """A kovarianciából becsült helyzethiba (m) a kép alsó kétharmadának néhány pontján."""
        if self.P is None or self.x is None:
            return float("inf")
        pts_img = np.array([[self.W * a, self.H * b] for a in (0.25, 0.5, 0.75) for b in (0.45, 0.75)])
        H0 = np.linalg.inv(x_to_G(self.x, self.C, self.W, self.H))

        def back(x):
            Hx = np.linalg.inv(x_to_G(x, self.C, self.W, self.H))
            X = Hx[:, :2] @ pts_img.T + Hx[:, 2:3]
            return (X[:2] / X[2]).T

        base = back(self.x)
        X0 = H0[:, :2] @ pts_img.T + H0[:, 2:3]
        ok = (X0[2] * np.sign((H0 @ np.array([self.W / 2, self.H - 1, 1.0]))[2]) > 0) & \
             (base[:, 0] > -5) & (base[:, 0] < PITCH_LENGTH + 5) & (base[:, 1] > -5) & (base[:, 1] < PITCH_WIDTH + 5)
        if not ok.any():
            return 0.0
        var = np.zeros(len(pts_img))
        L = np.linalg.cholesky(self.P + 1e-12 * np.eye(4))
        for j in range(4):
            dx = L[:, j]
            p = back(self.x + dx)
            var += np.sum((p - base) ** 2, axis=1)
        return float(np.sqrt(np.median(var[ok])))

    def _G_line_of(self, C=None):
        C = self.C if C is None else C
        SL, W, H = self.SL, self.W, self.H

        def G_of(y):
            return SL @ x_to_G(y, C, W, H)
        return G_of

    def _check_pose(self, field, q):
        """Vonal-illesztés (MAP, a mozgás-előrejelzéssel mint előzetessel) + önellenőrzés."""
        self.since_refine = 0
        q["refine_tried"] = True
        sig = np.sqrt(np.diag(self.P))
        wide = sig[0] > 0.4 or sig[1] > 0.25 or self.bad_streak > 0
        taus = (16.0, 8.0, 4.0, 2.0) if wide else (8.0, 4.0, 2.0)
        # A vonal-mérés felé engedékeny előzetes: a mozgásbecslés kovarianciája alulbecsüli a lassú
        # (zoom-) sodródást, ezért legalább pose_floor szórást engedünk – a vonalakkal rosszul
        # meghatározott irányokban így is a predikció marad
        Pinv = np.linalg.inv(self.P + np.diag(self.pose_floor) ** 2)
        y, info = refine_pose_lines(field, self.x, Pinv, self._G_line_of(), self.pts_fine, self.tan_fine,
                                    eps=np.full(4, 1e-3), px=self.px, taus=taus, line_ids=self.ids_fine)
        n_vis = info.get("n_visible", 0)
        q["n_visible"] = n_vis
        if y is None:
            # nincs mit ellenőrizni (kevés látható vonal): a bizonytalanság nő, a bizalom lassan csökken
            self.conf = 0.97 * self.conf
            return False
        prec, sup = info["precision"], info.get("line_support", 0.0)
        dy = y - self.x
        maha = float(dy @ Pinv @ dy)
        q["precision"] = round(prec, 3)
        q["support"] = round(sup, 3)
        q["innovation"] = round(maha, 2)
        accepted = (prec >= info["precision_before"] - 0.02 and
                    _plausible(self._G_line_of()(y), self.wl, self.hl))
        # nagy ugrást (más vonalra ugrás) csak erős alátámasztással fogadunk el
        if accepted and maha > 30.0 and not (prec >= 0.7 and sup >= 0.7):
            accepted = False
            q["refine_reason"] = "nagy ugrás, gyenge alátámasztás"
        if accepted:
            self.x = y
            A = info["info_matrix"]
            try:
                P_post = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                P_post = self.P
            # ne legyen túl magabiztos (a pontok korreláltak, a modell nem tökéletes)
            self.P = P_post + np.diag([0.01, 0.01, 0.01, 0.05]) ** 2
            q["refined"] = True
            self._update_G()
        m = prec if accepted else info["precision_before"]
        weight = min(1.0, n_vis / 300.0) * (0.5 + 0.5 * min(1.0, info.get("visible_len", 0.0) / (150.0 * self.px)))
        quality = 0.5 * m + 0.5 * sup if accepted else 0.5 * m
        # erős megerősítés (pl. takarás / elvesztés után) gyorsabban emeli a bizalmat
        a = (0.6 if quality >= 0.8 and quality > self.conf else 0.35) * weight
        self.conf = (1 - a) * self.conf + a * quality
        good = accepted and prec >= 0.5 and sup >= 0.5 and n_vis >= 80
        if good:
            self.since_verified = 0
            self.lost = 0
            self.bad_streak = 0
            if prec >= 0.6 and sup >= 0.6 and n_vis >= 150:
                self._maybe_add_keyframe(field, q)
        elif n_vis >= 80 and quality < 0.4:
            self.bad_streak += 1
        return accepted

    def _maybe_add_keyframe(self, field, q):
        """
        Jól illeszkedő képkocka felvétele kulcsképnek (eltérő irányokból), és a kamerahely közös
        újrabecslése a kulcsképekből (bundle_adjust_camera).
        """
        if self.frame_idx - self.last_keyframe < self.keyframe_gap:
            return
        x = self.x
        for i, kf in enumerate(self.keyframes):
            if abs(kf["x"][0] - x[0]) < 3.0 and abs(kf["x"][1] - x[1]) < 1.5 and abs(kf["x"][3] - x[3]) < 8.0:
                if self.frame_idx - kf["frame"] < 500:
                    return                     # ilyen nézetünk már van
                self.keyframes.pop(i)          # régi, hasonló nézet: frissítjük
                break
        self.keyframes.append({"field": _KeyField(field), "x": x.copy(), "frame": self.frame_idx})
        self.last_keyframe = self.frame_idx
        if len(self.keyframes) > self.max_keyframes:
            # a legkevésbé eltérő (legközelebbi pásztázási szögű) kulcskép kiesik
            pans = np.array([kf["x"][0] for kf in self.keyframes])
            dmin = [np.min(np.abs(np.delete(pans, i) - pans[i])) for i in range(len(pans) - 1)]
            self.keyframes.pop(int(np.argmin(dmin)))
        pans = [kf["x"][0] for kf in self.keyframes]
        if len(self.keyframes) < 3 or max(pans) - min(pans) < 10.0:
            return
        C_new, xs, info = bundle_adjust_camera(self.keyframes, self.C, self.C_prior_info, self.C_calib,
                                               self.W, self.H, self.SL, self.pts_fine, self.tan_fine, px=self.px)
        q["camera_ba"] = {"keyframes": len(self.keyframes), "shift_m": round(float(np.linalg.norm(C_new - self.C)), 2)}
        if not camera_plausible(self.x, C_new) or np.linalg.norm(C_new - self.C_calib) > 15.0:
            return
        for kf, xk in zip(self.keyframes, xs):
            kf["x"] = xk
        G_old = x_to_G(self.x, self.C, self.W, self.H)
        self.C = C_new
        self.x = self._fit_pose_to_G(G_old, self.x)
        self.P = self.P + np.diag([0.02, 0.02, 0.02, 0.1]) ** 2
        self._update_G()
        q["camera_refined"] = True

    def _fit_pose_to_G(self, G_full, x0):
        """Az állás (4 paraméter) illesztése egy adott G-hez a jelenlegi kamerahellyel (a kép ne ugorjon)."""
        uu, vv = np.meshgrid(np.linspace(0.05, 0.95, 9) * self.W, np.linspace(0.3, 0.95, 6) * self.H)
        img = np.column_stack([uu.ravel(), vv.ravel()])
        Hn = np.linalg.inv(G_full)
        X = Hn[:, :2] @ img.T + Hn[:, 2:3]
        pit = (X[:2] / X[2]).T
        _, wz = project_points(G_full, pit)
        ok = (wz > 0) & (np.abs(pit[:, 0] - PITCH_LENGTH / 2) < 90) & (np.abs(pit[:, 1] - PITCH_WIDTH / 2) < 70)
        if ok.sum() < 6:
            return x0
        img, pit = img[ok], pit[ok]
        C, W, H = self.C, self.W, self.H

        def res(x):
            uv, _ = project_points(x_to_G(x, C, W, H), pit)
            return (uv - img).ravel()
        x, _ = _lm(res, x0, eps=np.full(4, 1e-3), iters=15)
        return x

    def _relocalize(self, field, q):
        """Elveszett / rossz illesztés: (pan, tilt, zoom) keresés a predikció körül, rögzített kamerahellyel."""
        self.last_reloc_try = self.frame_idx
        wide = self.bad_streak > 20 or self.lost > 3 * self.max_lost
        cands = search_pose(field, self.x, self.C, self.W, self.H, self.SL, self.pts_coarse, px=self.px,
                            pan_half=60.0 if wide else 25.0, pan_step=2.0 if wide else 1.5,
                            tilt_half=6.0 if wide else 4.0)
        best = None
        weak = np.diag([1.0 / 20.0 ** 2, 1.0 / 5.0 ** 2, 1.0 / 2.0 ** 2, 1.0 / 20.0 ** 2])
        for score, x0 in cands:
            Pinv = weak
            y, info = refine_pose_lines(field, x0, Pinv, self._G_line_of(), self.pts_fine, self.tan_fine,
                                        eps=np.full(4, 1e-3), px=self.px, taus=(16.0, 8.0, 4.0, 2.0), iters=6,
                                        line_ids=self.ids_fine)
            if y is None:
                continue
            fit = 0.5 * info["precision"] + 0.5 * info.get("line_support", 0.0)
            if best is None or fit > best[0]:
                best = (fit, y, info)
        q["relocalize"] = {"candidates": len(cands), "accepted": False}
        ok = False
        if best is not None:
            fit, y, info = best
            G = self._G_line_of()(y)
            rec = line_recall(field, G)
            q["relocalize"].update({"fit": round(fit, 3), "recall": round(rec, 3)})
            ok = (info["precision"] >= 0.7 and info.get("line_support", 0) >= 0.7 and rec >= 0.4
                  and info["n_visible"] >= 100 and _plausible(G, self.wl, self.hl))
        if not ok:
            self.reloc_fails += 1
            if self.reloc_fails >= self.reloc_max_fails and self.auto_init:
                # A rögzített kamerahellyel nem található jó állás: lehet, hogy maga a kamerahely
                # (a kalibráció) rossz – teljes automatikus kalibráció (kamerahely-hipotézisekkel)
                self.reloc_fails = 0
                q["relocalize"]["auto"] = self.try_auto_calibrate(field, q)
                return q["relocalize"]["auto"]
            return False
        self.reloc_fails = 0
        self.x = y
        self.v = np.zeros(4)
        self.P = np.diag([0.1, 0.1, 0.1, 0.5]) ** 2
        self.conf = max(self.conf, 0.5 * info["precision"] + 0.5 * info.get("line_support", 0))
        self.lost = 0
        self.bad_streak = 0
        self.since_verified = 0
        self._update_G()
        q["relocalize"]["accepted"] = True
        return True

    # --- általános homográfia (tartalék üzemmód) ---------------------------
    def _propagate_homography(self, gray, q):
        F_init = None if self.last_F_h is None else self.last_F_h
        F, minfo = self.motion.estimate(self.prev_gray, gray, self.prev_mask, F_init=F_init)
        q["inliers"] = minfo.get("inliers", 0)
        self.last_F = F
        self.last_F_h = F
        if F is not None:
            q["motion_ok"] = True
            if self.G is not None:
                self.G = F @ self.G
                self.G /= np.linalg.norm(self.G)
            return True
        q["motion_ok"] = False
        q["motion_reason"] = minfo.get("reason", "")
        if self.G is not None:
            self.lost += 1
        return False

    last_F_h = None

    def _refine(self, field, q, wide=False):
        taus = tuple(t * self.px for t in ((16.0, 8.0, 4.0, 2.0) if wide else (6.0, 3.0, 1.5)))
        G_cur = self.P2L @ self.G
        q["refine_tried"] = True
        G_new, info = refine_homography(field, G_cur, self.pts_fine, taus=taus, tangents=self.tan_fine)
        self.since_refine = 0
        q["n_visible"] = info.get("n_visible", 0)
        if G_new is None:
            return False
        pb, pa = info["precision_before"], info["precision_after"]
        accepted = (info["cost_after"] < info["cost_before"] * 0.995 and pa >= pb - 0.01
                    and _plausible(G_new, self.wl, self.hl))
        if accepted:
            # Túl nagy ugrást csak elveszett állapotban engedünk
            uv0, _ = project_points(G_cur, self.pts_fine)
            uv1, _ = project_points(G_new, self.pts_fine)
            vis = field.inside(uv0, np.ones(len(uv0)))
            shift = float(np.median(np.linalg.norm(uv1[vis] - uv0[vis], axis=1))) if vis.any() else 0.0
            q["refine_shift_px"] = round(shift, 2)
            if shift > (40.0 if wide else 15.0) * self.px:
                accepted = False
        q["precision"] = round(pa if accepted else pb, 3)
        if accepted:
            G_new = self.L2P @ G_new
            if not wide and self.refine_gain < 1.0:
                # Részleges korrekció (komplementer szűrő): a terjesztés rövid távon
                # pontosabb, a vonalillesztés hosszú távon – a zajos illesztést átlagoljuk
                C = G_new @ np.linalg.inv(self.G)
                C /= C[2, 2]
                C = np.eye(3) + self.refine_gain * (C - np.eye(3))
                G_new = C @ self.G
            self.G = G_new / np.linalg.norm(G_new)
            q["refined"] = True
            pa_eff = pa
        else:
            pa_eff = pb
        # Bizalom: a vonalegyezés (pontosság) simítva; kevés látható vonalnál nem csökkentjük nagyon
        if info.get("n_visible", 0) >= 80:
            self.conf = 0.6 * self.conf + 0.4 * pa_eff
        if pa_eff >= 0.5 and info.get("n_visible", 0) >= 80:
            self.lost = 0        # a vonalak megerősítették a helyzetet
        return accepted

    def try_auto_calibrate(self, field, q=None):
        G, info = auto_calibrate(field, self.wl, self.hl, self.pts_coarse, self.pts_fine, px=self.px,
                                  tangents=self.tan_fine)
        self.last_info = info
        ok = G is not None and info.get("precision", 0) >= 0.6 and info.get("recall", 0) >= 0.45 \
            and info.get("n_visible", 0) >= 100
        if q is not None:
            q["auto_init"] = {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in info.items() if k in ("fit", "precision", "recall", "n_visible", "pan", "tilt", "hfov")}
            q["auto_init"]["accepted"] = bool(ok)
        if ok:
            G_full = np.linalg.inv(self.SL) @ G
            self.source = "automatikus"
            self.conf = info["precision"]
            self.lost = 0
            self.bad_streak = 0
            self.since_refine = 0
            self.since_verified = 0
            fit = fit_camera_to_homography(np.linalg.inv(G_full), self.W, self.H)
            if fit is not None and camera_plausible(fit[0], fit[1], fit[2]):
                self._set_pose(fit[0], fit[1], P=np.diag([0.2, 0.2, 0.2, 1.5]) ** 2, C_sigma=5.0)
            else:
                self.mode = "homográfia"
                self.x = None
                self.C = None
                self.G = orient_G(self.L2P @ G, self.ws, self.hs)
        return ok

    # --- visszafelé terjesztés (kalibráció a klip közepén) ----------------
    def backpropagate(self, frames, H_at_last):
        """
        Ha a kliens nem a klip első képkockáján kalibrált: a frames bejárható a
        klip elejétől a kalibrált képkockáig (azt is beleértve). Az összesített
        mozgás M = F_k · ... · F_1 (0. kép -> k. kép), így
            H_0 = H_k · M      (mert H_{t-1} = H_t · F_t).
        Állandó memóriában fut (csak az előző képet tartja). Visszatér: (H_0, sikertelen lépések száma).
        A kezdő H-t az előre haladó követés vonal-finomítása utána pontosítja.
        """
        M = np.eye(3)
        prev = None
        fails = 0
        for frame in frames:
            _, gray, _, grass = self._prepare(frame)
            if prev is not None:
                F, _ = self.motion.estimate(prev[0], gray, prev[1])
                if F is None:
                    fails += 1
                else:
                    M = F @ M
                    M /= M[2, 2]
            prev = (gray, self._feature_mask(grass, np.zeros_like(gray)))
        Ms = np.linalg.inv(self.S) @ M @ self.S        # teljes felbontásra
        H0 = np.asarray(H_at_last, np.float64) @ Ms
        return H0 / H0[2, 2], fails


# ---------------------------------------------------------------------------
# Rajzolás
# ---------------------------------------------------------------------------

_OVERLAY_POLYS = None


def draw_pitch_overlay(frame, H, color=(255, 255, 0), alpha=0.45, thickness=1):
    """A vetített pályavonalak halvány (alapból cián) kirajzolása, helyben. Ha a cián vonal
    a valódi fehér vonalon fut, jó a kalibráció; ha mellette, látszik az eltérés."""
    global _OVERLAY_POLYS
    if H is None:
        return frame
    if _OVERLAY_POLYS is None:
        _OVERLAY_POLYS = pitch_polylines(0.5)
    try:
        G = np.linalg.inv(np.asarray(H, np.float64))
    except np.linalg.LinAlgError:
        return frame
    h, w = frame.shape[:2]
    # a homográfia előjele tetszőleges (pl. H[2,2] = 1-re normálva): a kamera előtti pontokhoz w > 0 kell
    G = orient_G(G, w, h)
    mask = np.zeros((h, w), np.uint8)
    for poly in _OVERLAY_POLYS:
        uv, wz = project_points(G, poly)
        _draw_projected(mask, uv, wz, 255, thickness, cv2.LINE_AA)
    sel = mask > 0
    if np.any(sel):
        a = (mask[sel].astype(np.float32) / 255.0 * alpha)[:, None]
        frame[sel] = (frame[sel].astype(np.float32) * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
    return frame

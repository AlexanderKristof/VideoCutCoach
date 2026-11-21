"""
Pályakövetés mozgó (Veo) kamerához: képkockánkénti homográfia.

A Veo virtuális kamerája a labdát követi: forog (pásztáz), néha zoomol,
és egyszerre nagyjából fél pályát mutat. Az egyszer, a klip elején
kiszámolt homográfia (kép -> pálya méter) ezért az első pásztázás után
már hibás. Ez a modul minden képkockára saját homográfiát ad:

1. Terjesztés (propagation): két egymást követő képkocka között a kamera
   mozgását a statikus háttérből becsüljük (fű + pályavonalak, a játékos-
   és labdadobozok kitakarva): GFTT sarokpontok + Lucas–Kanade optikai
   áramlás csökkentett felbontású szürke képen, majd RANSAC-os
   homográfia (cv2.findHomography).
   Jelölés: F_t a (t-1). kép pixeleit viszi a t. kép pixeleire
   (x_t = F_t · x_{t-1}). Ha H_{t-1} a (t-1). kép -> pálya leképezés, akkor
       pálya = H_{t-1} · x_{t-1} = H_{t-1} · inv(F_t) · x_t
   tehát  H_t = H_{t-1} · inv(F_t).
   Belül a fordítottját tároljuk (G = inv(H): pálya -> kép), ekkor
   G_t = F_t · G_{t-1}  (ugyanaz, csak invertálás nélkül).
   A rossz becslést (kevés inlier, nagy skála/nyírás-ugrás, nagy
   visszavetítési hiba) eldobjuk, ilyenkor az előző H marad és nő az
   'elveszett' számláló.

2. Sodródás-javítás a pályavonalakkal: időnként (refine_every képkockánként,
   vagy ha a mozgásbecslés hibázott) a modell összes szabványos vonalát
   (oldal- és alapvonalak, felezővonal, 16-osok, 5-ösök, kezdőkör,
   16-os ív) a képre vetítjük, és a homográfiát úgy finomítjuk, hogy a
   vetített pontok a felismert fehér vonalpixeleken legyenek. A vonalmaszk
   távolság-transzformáltját mintavételezzük, és egy 8 paraméteres
   korrekciós homográfiát Levenberg–Marquardt / Gauss–Newton lépésekkel
   optimalizálunk (csak numpy, scipy nem kell). A finomítást csak akkor
   fogadjuk el, ha a költség javul.

3. Automatikus kezdeti kalibráció (best effort): ha nincs kézi kalibráció,
   kamera-hipotézisek rácsát (pásztázási szög, dőlés, fókusz, kamera
   távolsága és magassága; a kamera a felezővonal vonalában, a közeli
   oldalvonal mögött) pontozzuk ugyanazzal a távolság-transzformált
   mértékkel, a legjobbakat finomítjuk, és csak akkor fogadjuk el, ha a
   vonalegyezés (pontosság és lefedettség) eléri a küszöböt. Különben a
   státusz 'kalibráció szükséges'.

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


def camera_ground_homography(pan_deg, tilt_deg, focal_px, img_w, img_h, cam_pos, roll_deg=0.0):
    """
    Lyukkamera pálya-sík -> kép homográfia (G).
    cam_pos = (x, y, z) méterben; a világ z tengelye LEFELÉ mutat (a pálya-
    koordináták x jobbra, y a kamera felé – így jobbsodrású), tehát a magasság z = -h.
    pan = 0: a kamera a túlsó oldalvonal felé (-y) néz; pozitív pan: jobbra (+x felé).
    tilt: lefelé dőlés fokban.
    """
    p = math.radians(pan_deg)
    t = math.radians(tilt_deg)
    fwd = np.array([math.sin(p) * math.cos(t), -math.cos(p) * math.cos(t), math.sin(t)])
    up = np.array([0.0, 0.0, -1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    if roll_deg:
        r = math.radians(roll_deg)
        right, down = (math.cos(r) * right + math.sin(r) * down,
                       -math.sin(r) * right + math.cos(r) * down)
    R = np.stack([right, down, fwd])
    C = np.asarray(cam_pos, float)
    K = np.array([[focal_px, 0, img_w / 2.0], [0, focal_px, img_h / 2.0], [0, 0, 1]])
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
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    tr = jxx + jyy
    det_term = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy)
    coh = (det_term / np.maximum(tr, 1e-6)) ** 2
    out = mask.copy()
    out[coh < min_coherence] = 0
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
        inv = np.where(line_mask > 0, 0, 255).astype(np.uint8)
        self.D = cv2.distanceTransform(inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float32)
        self.gx = cv2.Sobel(self.D, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        self.gy = cv2.Sobel(self.D, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        self.n_line = int(np.count_nonzero(line_mask))
        # A lefedettséghez csak a hosszú (biztosan vonal) komponensek számítanak:
        # a lelátó / reklámtábla zaja rövid darabokra esik szét
        n, lab, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
        length = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        keep = length >= max(20.0, self.w / 30.0)
        keep[0] = False
        self.long_mask = np.where(keep[lab], 255, 0).astype(np.uint8)
        self.n_long = int(np.count_nonzero(self.long_mask))

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
    tau = 10.0 * px
    Q = np.maximum(0.0, 1.0 - field.D / tau).astype(np.float32)   # közelségi térkép (1 = vonalon)
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
    tan_c = pitch_model_tangents(_coarse_step(pts_coarse))
    nrm_c = np.column_stack([-tan_c[:, 1], tan_c[:, 0]])
    step_c = _coarse_step(pts_coarse)
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

    def estimate(self, prev_gray, gray, mask=None):
        h, w = gray.shape[:2]
        pts = cv2.goodFeaturesToTrack(prev_gray, self.max_corners, 0.003, max(6, w // 90),
                                      mask=mask, blockSize=5)
        if (pts is None or len(pts) < self.min_inliers * 2) and mask is not None:
            # Egyszínű fű: tartalék – az egész kép (a dobozok nélkül nem tudjuk, de a RANSAC kiszűri)
            pts = cv2.goodFeaturesToTrack(prev_gray, self.max_corners, 0.003, max(6, w // 90), blockSize=5)
        info = {"features": 0 if pts is None else int(len(pts)), "inliers": 0, "reason": ""}
        if pts is None or len(pts) < self.min_inliers:
            info["reason"] = "kevés jellemzőpont"
            return None, info
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **self.lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **self.lk)
        fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
        p0 = pts.reshape(-1, 2)[good]
        p1 = nxt.reshape(-1, 2)[good]
        if len(p0) < self.min_inliers:
            info["reason"] = "kevés követett pont"
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

    A visszaadott H_t teljes felbontású kép pixel -> pálya méter (mint a régi statikus H).
    q: minőség-szótár (status, conf, inliers, lost, refined, source, ...).
    """

    def __init__(self, frame_size, grass_lower=None, grass_upper=None, H0=None, proc_width=640,
                 line_width=960, refine_every=10, auto_init=True, auto_init_every=50, auto_init_max_every=250, max_lost=15,
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
        self.pts_fine = pitch_model_points(0.5)
        self.tan_fine = pitch_model_tangents(0.5)
        self.pts_coarse = pitch_model_points(1.5)
        self.G = None            # pálya -> feldolgozási kép
        self.source = None
        self.frame_idx = -1
        self.prev_gray = None
        self.last_F = None       # utolsó képkockák közti mozgás (feldolgozási felbontás), lásd frame_motion()
        self.prev_mask = None
        self.lost = 0            # egymás utáni sikertelen mozgásbecslések / rossz illesztések
        self.since_refine = 0
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
        G_full = np.linalg.inv(np.asarray(H_full, np.float64))
        self.G = orient_G(self.S @ G_full, self.ws, self.hs)
        self.source = source
        self.lost = 0
        self.conf = max(self.conf, 0.5)
        self.since_refine = self.refine_every   # az első képkockán rögtön finomítunk

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
        """Teljes felbontású kép -> pálya homográfia (vagy None)."""
        if self.G is None:
            return None
        Gf = np.linalg.inv(self.S) @ self.G
        H = np.linalg.inv(Gf)
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

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
        m[box_mask > 0] = 0
        return m

    # --- fő lépés -----------------------------------------------------------
    def update(self, frame, boxes=None):
        """Egy képkocka feldolgozása. Visszatér: (H_t vagy None, minőség-szótár)."""
        self.frame_idx += 1
        small, gray, hsv, grass = self._prepare(frame)
        box_mask = _boxes_mask(gray.shape, boxes, self.s)
        q = {"frame": self.frame_idx, "refined": False, "motion_ok": None}

        # 1) Terjesztés: G_t = F_t · G_{t-1}  (azaz H_t = H_{t-1} · inv(F_t))
        motion_failed = False
        self.last_F = None     # a labdakövetőnek: képtér-mozgás (előző -> aktuális kép), lásd frame_motion()
        if self.prev_gray is not None:
            F, minfo = self.motion.estimate(self.prev_gray, gray, self.prev_mask)
            q["inliers"] = minfo.get("inliers", 0)
            self.last_F = F
            if F is not None:
                q["motion_ok"] = True
                if self.G is not None:
                    self.G = F @ self.G
                    self.G /= np.linalg.norm(self.G)
            else:
                q["motion_ok"] = False
                q["motion_reason"] = minfo.get("reason", "")
                motion_failed = True
                if self.G is not None:
                    self.lost += 1
        self.prev_gray = gray
        self.prev_mask = self._feature_mask(grass, box_mask)

        field = None
        # 2) Sodródás-javítás a vonalakkal
        if self.G is not None:
            self.since_refine += 1
            if self.since_refine >= self.refine_every or motion_failed or self.lost > 0 or self.conf < self.min_conf:
                field = self._line_field(frame, small, hsv, grass, box_mask, boxes)
                self._refine(field, q, wide=self.lost > 0)
            if self.lost > self.relocalize_after:
                # Tartósan elveszett: újra kell kalibrálni (automatikusan, ha engedélyezett)
                self.G = None
                self.source = None
                self.conf = 0.0
                self.lost = 0

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
        if self.G is None:
            q["status"] = STATUS_NEED_CALIB
            return None, q
        if self.lost > self.max_lost:
            q["status"] = STATUS_LOST
            return None, q
        q["status"] = STATUS_OK
        return self.current_H(), q

    def _refine(self, field, q, wide=False):
        taus = tuple(t * self.px for t in ((16.0, 8.0, 4.0, 2.0) if wide else (6.0, 3.0, 1.5)))
        G_cur = self.P2L @ self.G
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
            self.G = orient_G(self.L2P @ G, self.ws, self.hs)
            self.source = "automatikus"
            self.conf = info["precision"]
            self.lost = 0
            self.since_refine = 0
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
    mask = np.zeros((h, w), np.uint8)
    for poly in _OVERLAY_POLYS:
        uv, wz = project_points(G, poly)
        _draw_projected(mask, uv, wz, 255, thickness, cv2.LINE_AA)
    sel = mask > 0
    if np.any(sel):
        a = (mask[sel].astype(np.float32) / 255.0 * alpha)[:, None]
        frame[sel] = (frame[sel].astype(np.float32) * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
    return frame

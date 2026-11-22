#!/usr/bin/env python3
"""
Kattintós pályakalibráció (asztali OpenCV ablak).

    python tools/kalibracio.py MECCS.mp4 --ido 10:00
    python tools/kalibracio.py MECCS.mp4 --ido 50:30 --felido 2          (ha a szünetben odébb tették a kamerát)
    python tools/kalibracio.py MECCS.mp4 --palya 100x64                   (nem szabványos pályaméret)
    python tools/kalibracio.py MECCS.mp4 --elonezet kep.jpg               (ablak nélkül: a mentett kalibráció képe)

Menete: bal oldalt a videó képe, jobb oldalt a pálya felülnézeti rajza a számozott, jellegzetes
pontokkal (szögletek, 16-os és 5-ös sarkai, felezővonal végei, kezdőpont, a kezdőkör és a felezővonal
metszéspontjai, büntetőpontok, a 16-os ív és a 16-os vonal metszéspontjai).
  1. kattints egy pontra a videón (görgővel nagyítható; a nagyító a kurzor körüli részt mutatja),
  2. majd a hozzá tartozó pontra a rajzon (a legközelebbi jelölt pontra ugrik) – vagy fordított sorrendben,
  3. legalább 4 pár kell (ajánlott 5–8, a képen minél szétszórtabban). 4 pártól a program a videóra
     rajzolja (cián színnel) a kiszámolt pályavonalakat és pontonként a hibát – ha a cián vonalak a
     valódi fehér vonalakon futnak, jó a kalibráció.
Billentyűk: Enter = kész (mentés), Backspace / u = utolsó pár törlése, r = újrakezdés,
a / d = ±1 mp, A / D (vagy j / l) = ±10 mp (időváltáskor a pontok törlődnek), görgő = nagyítás,
f = nagyítás vissza, h = súgó ki/be, Esc / q = kilépés mentés nélkül.

Kimenet: a kiírt, bemásolható elemzes.py-kapcsolók
    --kalibracio "x,y;..." --kalibracio-pontok "X,Y;..." --kalibracio-ido MP
(képkoordináták 0..1 arányban, pályapontok méterben), és a <videó>.kalibracio.json fájl: az
elemzes.py ezt magától használja, ha nincs --kalibracio kapcsoló. A --felido 2 a 2. félidő
kalibrációját ugyanebbe a fájlba írja (a pipeline mindkét félidőben a sajátját használja).
"""
import argparse
import json
import math
import os
import sys
import unicodedata

import numpy as np

try:
    import cv2
except ImportError:          # a tiszta (rajzolás nélküli) függvények teszteléséhez nem kell
    cv2 = None

STD_LENGTH, STD_WIDTH = 105.0, 68.0
JSON_SUFFIX = ".kalibracio.json"
JSON_VERSION = 1


# ---------------------------------------------------------------------------
# Tiszta függvények (tesztelhetők, ablak nélkül)
# ---------------------------------------------------------------------------

def parse_time(text):
    """'10:00', '1:02:03', '600' vagy '600.5' -> másodperc (float). Hibás szövegnél ValueError."""
    t = str(text).strip().replace(",", ".")
    if not t:
        raise ValueError("üres időpont")
    parts = t.split(":")
    if len(parts) > 3:
        raise ValueError(f"érvénytelen időpont: {text!r}")
    sec = 0.0
    for p in parts:
        sec = sec * 60.0 + float(p)
    if sec < 0:
        raise ValueError(f"negatív időpont: {text!r}")
    return sec


def format_time(sec):
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    frac = sec - int(sec)
    base = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return base + (f".{int(round(frac * 10)) % 10}" if frac >= 0.05 else "")


def parse_pitch_size(text):
    """'100x64' (vagy '100*64', '100,64') -> (hossz, szélesség) méterben."""
    t = str(text).lower().replace("*", "x").replace(",", "x").replace("×", "x")
    try:
        a, b = [float(v) for v in t.split("x")]
    except ValueError:
        raise ValueError(f"a pályaméret formája HOSSZxSZÉLESSÉG, pl. 100x64 (kapott: {text!r})")
    if a < b:
        a, b = b, a
    if not (40.0 <= a <= 130.0 and 25.0 <= b <= 100.0):
        raise ValueError(f"valószínűtlen pályaméret: {a:g} x {b:g} m")
    return a, b


def landmarks(length=STD_LENGTH, width=STD_WIDTH):
    """
    A kattintható pályapontok: [(sorszám, név, x, y), ...] méterben (x a hossz mentén 0..length,
    y a szélesség mentén 0..width; y = 0 a túlsó, y = width a kamera felőli oldalvonal).
    A 16-os, az 5-ös, a kezdőkör és a büntetőpont a szabály szerinti (fix) méretű.
    """
    L, W = float(length), float(width)
    cy = W / 2.0
    pts = []

    def add(name, x, y):
        pts.append((len(pts) + 1, name, float(x), float(y)))

    add("bal felső szöglet", 0, 0)
    add("felezővonal felső vége", L / 2, 0)
    add("jobb felső szöglet", L, 0)
    add("bal alsó szöglet", 0, W)
    add("felezővonal alsó vége", L / 2, W)
    add("jobb alsó szöglet", L, W)
    add("kezdőpont", L / 2, cy)
    add("kezdőkör × felezővonal (felső)", L / 2, cy - 9.15)
    add("kezdőkör × felezővonal (alsó)", L / 2, cy + 9.15)
    add("kezdőkör bal széle", L / 2 - 9.15, cy)
    add("kezdőkör jobb széle", L / 2 + 9.15, cy)
    arc_dy = math.sqrt(9.15 ** 2 - 5.5 ** 2)
    for side, x0, s in (("bal", 0.0, 1.0), ("jobb", L, -1.0)):
        add(f"{side} 16-os felső sarka (alapvonal)", x0, cy - 20.16)
        add(f"{side} 16-os felső sarka", x0 + s * 16.5, cy - 20.16)
        add(f"{side} 16-os alsó sarka", x0 + s * 16.5, cy + 20.16)
        add(f"{side} 16-os alsó sarka (alapvonal)", x0, cy + 20.16)
        add(f"{side} 5-ös felső sarka (alapvonal)", x0, cy - 9.16)
        add(f"{side} 5-ös felső sarka", x0 + s * 5.5, cy - 9.16)
        add(f"{side} 5-ös alsó sarka", x0 + s * 5.5, cy + 9.16)
        add(f"{side} 5-ös alsó sarka (alapvonal)", x0, cy + 9.16)
        add(f"{side} büntetőpont", x0 + s * 11.0, cy)
        add(f"{side} 16-os ív × 16-os vonal (felső)", x0 + s * 16.5, cy - arc_dy)
        add(f"{side} 16-os ív × 16-os vonal (alsó)", x0 + s * 16.5, cy + arc_dy)
    return pts


def pitch_lines(length=STD_LENGTH, width=STD_WIDTH, step=0.5):
    """A pályavonalak töréspontos vonalakként (a pitch_tracker modelljével azonos, tetszőleges mérettel)."""
    L, W = float(length), float(width)
    cy = W / 2.0
    out = []

    def seg(a, b):
        n = max(2, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        out.append(np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t)

    def arc(cx, cyy, r, a0, a1):
        n = max(3, int(math.ceil(abs(a1 - a0) * r / step)) + 1)
        t = np.linspace(a0, a1, n)
        out.append(np.stack([cx + r * np.cos(t), cyy + r * np.sin(t)], axis=1))

    seg((0, 0), (L, 0))
    seg((0, W), (L, W))
    seg((0, 0), (0, W))
    seg((L, 0), (L, W))
    seg((L / 2, 0), (L / 2, W))
    arc(L / 2, cy, 9.15, 0.0, 2 * math.pi)
    a = math.acos(5.5 / 9.15)
    for x0, s in ((0.0, 1.0), (L, -1.0)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            seg((x0, cy - half), (x0 + s * depth, cy - half))
            seg((x0 + s * depth, cy - half), (x0 + s * depth, cy + half))
            seg((x0 + s * depth, cy + half), (x0, cy + half))
        if s > 0:
            arc(11.0, cy, 9.15, -a, a)
        else:
            arc(L - 11.0, cy, 9.15, math.pi - a, math.pi + a)
    return out


def nearest_landmark(lms, x, y, max_dist=None):
    """A (x, y) pálya-ponthoz (m) legközelebbi jelölt pont, vagy None (ha max_dist-nél messzebb van)."""
    best = None
    for lm in lms:
        d = math.hypot(lm[2] - x, lm[3] - y)
        if best is None or d < best[0]:
            best = (d, lm)
    if best is None or (max_dist is not None and best[0] > max_dist):
        return None
    return best[1]


def _apply(H, pts):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    X = H[:, :2] @ p.T + H[:, 2:3]
    w = np.where(np.abs(X[2]) < 1e-12, 1e-12, X[2])
    return (X[:2] / w).T


def homography_from_pairs(img_pts, pitch_pts):
    """
    Kép (px) -> pálya (m) homográfia legalább 4 pontpárból (legkisebb négyzetek, mint a
    pitch_tracker.parse_calib_points). Visszatér: (H, hibák) vagy (None, None).
    hibák: pontonként {"px": visszavetítési hiba a képen, "m": eltérés a pályán}.
    """
    img = np.asarray(img_pts, np.float64).reshape(-1, 2)
    pit = np.asarray(pitch_pts, np.float64).reshape(-1, 2)
    if len(img) < 4 or len(img) != len(pit):
        return None, None
    # 3 pont egy egyenesen (pl. csak a felezővonal pontjai) -> elfajult
    if np.linalg.matrix_rank(np.column_stack([pit, np.ones(len(pit))]), tol=1e-6) < 3 or \
            np.linalg.matrix_rank(np.column_stack([img, np.ones(len(img))]), tol=1e-6) < 3:
        return None, None
    if cv2 is not None:
        H, _ = cv2.findHomography(img, pit, 0)
    else:
        H = _dlt(img, pit)
    if H is None or not np.all(np.isfinite(H)):
        return None, None
    try:
        G = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, None
    err_px = np.linalg.norm(_apply(G, pit) - img, axis=1)
    err_m = np.linalg.norm(_apply(H, img) - pit, axis=1)
    return H / H[2, 2], [{"px": float(a), "m": float(b)} for a, b in zip(err_px, err_m)]


def _dlt(img, pit):
    """Normalizált DLT (OpenCV nélküli tartalék)."""
    def norm(p):
        c = p.mean(axis=0)
        s = math.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    T1, T2 = norm(img), norm(pit)
    a = _apply(T1, img)
    b = _apply(T2, pit)
    A = []
    for (x, y), (u, v) in zip(a, b):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    return np.linalg.inv(T2) @ Hn @ T1


def format_args(pairs, width, height, time_sec):
    """
    pairs: [((u, v) képpont px, (x, y) pálya m), ...]. Visszatér: dict a kapcsolók értékeivel és a
    bemásolható parancssori szöveggel ("args").
    """
    img = ";".join(f"{u / float(width):.4f},{v / float(height):.4f}" for (u, v), _ in pairs)
    pit = ";".join(f"{_num(x)},{_num(y)}" for _, (x, y) in pairs)
    t = round(float(time_sec), 2)
    return {"kalibracio": img, "kalibracio_pontok": pit, "kalibracio_ido": t,
            "args": f'--kalibracio "{img}" --kalibracio-pontok "{pit}" --kalibracio-ido {_num(t)}'}


def _num(v):
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def json_path_for(video):
    return video + JSON_SUFFIX


def make_half_entry(pairs, width, height, time_sec, errors=None, names=None):
    a = format_args(pairs, width, height, time_sec)
    entry = {"kalibracio": a["kalibracio"], "kalibracio_pontok": a["kalibracio_pontok"],
             "kalibracio_ido": a["kalibracio_ido"], "ido_szoveg": format_time(time_sec),
             "kepmeret": [int(width), int(height)], "pontok": []}
    for i, ((u, v), (x, y)) in enumerate(pairs):
        p = {"kep_px": [round(float(u), 1), round(float(v), 1)], "palya_m": [float(x), float(y)]}
        if names and i < len(names) and names[i]:
            p["nev"] = names[i]
        if errors and i < len(errors):
            p["hiba_px"] = round(errors[i]["px"], 2)
            p["hiba_m"] = round(errors[i]["m"], 3)
        entry["pontok"].append(p)
    return entry


def save_calibration_json(path, half, entry, pitch_size=(STD_LENGTH, STD_WIDTH), video=None):
    """A félidő kalibrációjának mentése (a másik félidő meglévő bejegyzése megmarad). Visszatér: a teljes dict."""
    data = load_calibration_json(path) or {}
    data["verzio"] = JSON_VERSION
    if video:
        data["video"] = os.path.basename(video)
    data["palya"] = [float(pitch_size[0]), float(pitch_size[1])]
    data.setdefault("felidok", {})[str(int(half))] = entry
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return data


def load_calibration_json(path):
    """A kalibrációs JSON beolvasása (None, ha nincs / hibás)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("felidok"), dict):
        return None
    return data


def calibration_options(data):
    """
    A JSON -> elemzes.py / pipeline opciók: az 1. félidőé a calib_points / calib_pitch_points /
    calib_time_sec, a 2. félidőé a calib2_* kulcsokba kerül. Visszatér: dict (üres, ha nincs adat).
    """
    out = {}
    if not data:
        return out
    halves = data.get("felidok") or {}
    for half, prefix in (("1", "calib"), ("2", "calib2")):
        e = halves.get(half)
        if not e or not e.get("kalibracio"):
            continue
        out[f"{prefix}_points"] = e["kalibracio"]
        out[f"{prefix}_pitch_points"] = e.get("kalibracio_pontok", "")
        if e.get("kalibracio_ido") is not None:
            out[f"{prefix}_time_sec"] = float(e["kalibracio_ido"])
    return out


# ---------------------------------------------------------------------------
# Rajzolás (OpenCV, ablak nélkül is – a --elonezet és a tesztek ezt használják)
# ---------------------------------------------------------------------------

CYAN = (255, 255, 0)
YELLOW = (0, 230, 255)
GREEN = (80, 220, 80)
RED = (60, 60, 255)
WHITE = (255, 255, 255)


class DiagramView:
    """A pálya felülnézeti rajza: méter <-> rajz-pixel átváltás és kirajzolás."""

    def __init__(self, length, width, panel_w, panel_h, margin=26):
        self.L, self.W = float(length), float(width)
        self.pw, self.ph = int(panel_w), int(panel_h)
        s = min((self.pw - 2 * margin) / self.L, (self.ph - 2 * margin) / self.W)
        self.s = s
        self.ox = (self.pw - self.L * s) / 2.0
        self.oy = (self.ph - self.W * s) / 2.0
        self.lms = landmarks(self.L, self.W)
        self.lines = pitch_lines(self.L, self.W)

    def to_px(self, x, y):
        return self.ox + x * self.s, self.oy + y * self.s

    def to_m(self, px, py):
        return (px - self.ox) / self.s, (py - self.oy) / self.s

    def pick(self, px, py, max_px=40):
        x, y = self.to_m(px, py)
        return nearest_landmark(self.lms, x, y, max_dist=max_px / self.s)

    def render(self, used=(), pending=None, hover=None):
        img = np.full((self.ph, self.pw, 3), (40, 95, 45), np.uint8)
        for k in range(0, int(self.L // 5) + 1, 2):       # kaszálási sávok
            x0, _ = self.to_px(k * 5, 0)
            x1, _ = self.to_px(min(self.L, (k + 1) * 5), 0)
            cv2.rectangle(img, (int(x0), int(self.oy)), (int(x1), int(self.oy + self.W * self.s)), (45, 105, 50), -1)
        for poly in self.lines:
            p = np.array([self.to_px(x, y) for x, y in poly], np.float64)
            cv2.polylines(img, [np.round(p * 4).astype(np.int32)], False, WHITE, 1, cv2.LINE_AA, 2)
        used_ids = {u: i for i, u in enumerate(used)}
        fs = 0.36 if self.pw < 600 else 0.42
        for num, _name, x, y in self.lms:
            px, py = self.to_px(x, y)
            c = (int(round(px)), int(round(py)))
            if num in used_ids:
                cv2.circle(img, c, 7, GREEN, -1, cv2.LINE_AA)
                cv2.putText(img, str(used_ids[num] + 1), (c[0] - 4, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 0, 0), 1, cv2.LINE_AA)
            else:
                col = YELLOW if pending == num else (0, 200, 255)
                cv2.circle(img, c, 5 if pending == num else 3, col, -1, cv2.LINE_AA)
            # a felirat a pályán kívülre / a pont külső oldalára kerüljön (ne takarják egymást)
            (tw, th), _ = cv2.getTextSize(str(num), cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            left = x > self.L / 2 + 1e-6 if x not in (0.0, self.L) else x == 0.0
            tx = c[0] - tw - 5 if left else c[0] + 5
            ty = c[1] - 5 if y <= self.W / 2 + 1e-6 else c[1] + th + 5
            cv2.putText(img, str(num), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(num), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, 1, cv2.LINE_AA)
        if hover is not None:
            px, py = self.to_px(hover[2], hover[3])
            cv2.circle(img, (int(px), int(py)), 10, YELLOW, 1, cv2.LINE_AA)
            _label(img, f"{hover[0]}. {hover[1]}", (8, self.ph - 10))
        return img


def ascii_text(text):
    """Ékezetek nélkül (az OpenCV beépített betűtípusa a legtöbb változatban nem rajzol ékezetet)."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")


def _label(img, text, org, scale=0.5, color=WHITE):
    text = ascii_text(text)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_projection(frame, H, length=STD_LENGTH, width=STD_WIDTH, color=CYAN, thickness=2):
    """A kalibrációból vetített pályavonalak a (teljes felbontású) képre, helyben."""
    try:
        G = np.linalg.inv(np.asarray(H, np.float64))
    except np.linalg.LinAlgError:
        return frame
    h, w = frame.shape[:2]
    # előjel: a kép alsó-közepi pontja legyen a kamera előtt
    q = H @ np.array([w / 2.0, h * 0.9, 1.0])
    ref = G @ (q / q[2])
    sgn = 1.0 if ref[2] > 0 else -1.0
    for poly in pitch_lines(length, width):
        X = G[:, :2] @ poly.T + G[:, 2:3]
        ok = X[2] * sgn > 1e-9
        uv = (X[:2] / np.where(np.abs(X[2]) < 1e-12, 1e-12, X[2])).T
        ok &= np.all(np.abs(uv) < 1e5, axis=1)
        run = []
        for i in range(len(uv)):
            if ok[i]:
                run.append(uv[i])
            elif run:
                if len(run) >= 2:
                    cv2.polylines(frame, [np.round(np.array(run) * 4).astype(np.int32)], False, color, thickness,
                                  cv2.LINE_AA, 2)
                run = []
        if len(run) >= 2:
            cv2.polylines(frame, [np.round(np.array(run) * 4).astype(np.int32)], False, color, thickness,
                          cv2.LINE_AA, 2)
    return frame


def annotate_frame(frame, pairs, H=None, errors=None, pending_img=None, length=STD_LENGTH, width=STD_WIDTH,
                   lm_nums=None):
    """A videókép teljes felbontásban: vetített vonalak (cián), a kattintott pontok sorszámmal és hibával."""
    img = frame.copy()
    if H is not None:
        draw_projection(img, H, length, width)
    sc = max(1.0, img.shape[1] / 1280.0)
    for i, ((u, v), _xy) in enumerate(pairs):
        c = (int(round(u)), int(round(v)))
        cv2.circle(img, c, int(7 * sc), GREEN, 2, cv2.LINE_AA)
        cv2.line(img, (c[0] - int(12 * sc), c[1]), (c[0] + int(12 * sc), c[1]), GREEN, 1, cv2.LINE_AA)
        cv2.line(img, (c[0], c[1] - int(12 * sc)), (c[0], c[1] + int(12 * sc)), GREEN, 1, cv2.LINE_AA)
        txt = f"{i + 1}" + (f" ({lm_nums[i]})" if lm_nums and i < len(lm_nums) else "")
        if errors and i < len(errors):
            txt += f"  {errors[i]['px']:.1f} px / {errors[i]['m']:.2f} m"
        _label(img, txt, (c[0] + 10 * sc, c[1] - 10 * sc), scale=0.5 * sc)
    if pending_img is not None:
        c = (int(round(pending_img[0])), int(round(pending_img[1])))
        cv2.circle(img, c, int(9 * sc), YELLOW, 2, cv2.LINE_AA)
    return img


def compose(video_panel, diagram_panel, status_lines=()):
    """A két panel egymás mellett + állapotsor alul (egyetlen ablak képe)."""
    h = max(video_panel.shape[0], diagram_panel.shape[0])
    right = np.full((h, diagram_panel.shape[1], 3), 30, np.uint8)
    right[:diagram_panel.shape[0]] = diagram_panel
    y = diagram_panel.shape[0] + 22
    for line in status_lines:
        if y > h - 4:
            break
        cv2.putText(right, ascii_text(line), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        y += 20
    left = np.full((h, video_panel.shape[1], 3), 0, np.uint8)
    left[:video_panel.shape[0]] = video_panel
    return np.hstack([left, np.full((h, 3, 3), 80, np.uint8), right])


# ---------------------------------------------------------------------------
# Interaktív eszköz
# ---------------------------------------------------------------------------

HELP = [
    "Kattints a videón, majd a rajzon a hozzá tartozó pontra.",
    "Legalább 4 pár (ajánlott 5-8, szétszórva).",
    "Enter: kész  Backspace/u: vissza  r: újra",
    "a/d: -/+1 mp   A/D vagy j/l: -/+10 mp",
    "görgő: nagyítás  f: teljes kép  h: súgó",
    "Esc/q: kilépés mentés nélkül",
]


class FrameSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"a videó nem nyitható meg: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.duration = n / self.fps if n > 0 else None

    def read(self, t):
        if self.duration is not None:
            t = min(max(0.0, t), max(0.0, self.duration - 1.0 / self.fps))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = self.cap.read()
        if not ok:
            return None, t
        pos = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        # a pozíció a beolvasott kép UTÁNI időt mutatja egyes kodekeknél: a kért időt tartjuk meg
        return frame, (t if not pos or pos <= 0 else max(0.0, pos / 1000.0 - 1.0 / self.fps))

    def release(self):
        self.cap.release()


class CalibrationTool:
    """Az ablak állapota: képkocka, nagyítás, pontpárok. A render() ablak nélkül is működik."""

    def __init__(self, frame, time_sec, pitch_size=(STD_LENGTH, STD_WIDTH), view_w=1120, diag_w=480):
        self.L, self.W = pitch_size
        self.diag = DiagramView(self.L, self.W, diag_w, int(diag_w * (self.W + 8) / (self.L + 8)))
        self.view_w = int(view_w)
        self.pairs = []          # [((u, v), (x, y)), ...]
        self.lm_nums = []
        self.pending_img = None
        self.pending_lm = None
        self.hover_lm = None
        self.mouse = None        # (x, y) az ablakban
        self.show_help = True
        self.message = ""
        self.set_frame(frame, time_sec)

    # --- állapot ---
    def set_frame(self, frame, time_sec):
        self.frame = frame
        self.t = float(time_sec)
        self.fh, self.fw = frame.shape[:2]
        self.base = self.view_w / float(self.fw)        # teljes kép -> panel
        self.view_h = int(round(self.fh * self.base))
        self.zoom = 1.0
        self.cx, self.cy = self.fw / 2.0, self.fh / 2.0  # a nézet középpontja (teljes kép px)
        self.pairs, self.lm_nums = [], []
        self.pending_img = self.pending_lm = None

    def _scale(self):
        return self.base * self.zoom

    def view_to_img(self, x, y):
        s = self._scale()
        return self.cx + (x - self.view_w / 2.0) / s, self.cy + (y - self.view_h / 2.0) / s

    def img_to_view(self, u, v):
        s = self._scale()
        return (u - self.cx) * s + self.view_w / 2.0, (v - self.cy) * s + self.view_h / 2.0

    def _clamp_center(self):
        s = self._scale()
        hw, hh = self.view_w / 2.0 / s, self.view_h / 2.0 / s
        self.cx = min(max(self.cx, hw), self.fw - hw) if hw < self.fw / 2 else self.fw / 2.0
        self.cy = min(max(self.cy, hh), self.fh - hh) if hh < self.fh / 2 else self.fh / 2.0

    def zoom_at(self, x, y, factor):
        u, v = self.view_to_img(x, y)
        self.zoom = float(min(8.0, max(1.0, self.zoom * factor)))
        s = self._scale()
        self.cx = u - (x - self.view_w / 2.0) / s
        self.cy = v - (y - self.view_h / 2.0) / s
        self._clamp_center()

    def solve(self):
        if len(self.pairs) < 4:
            return None, None
        return homography_from_pairs([p[0] for p in self.pairs], [p[1] for p in self.pairs])

    def _try_pair(self):
        if self.pending_img is not None and self.pending_lm is not None:
            lm = self.pending_lm
            if lm[0] in self.lm_nums:
                i = self.lm_nums.index(lm[0])        # ugyanaz a pálya-pont újra: lecseréljük
                self.pairs[i] = (self.pending_img, (lm[2], lm[3]))
            else:
                self.pairs.append((self.pending_img, (lm[2], lm[3])))
                self.lm_nums.append(lm[0])
            self.message = f"{len(self.pairs)}. pár: {lm[0]}. {lm[1]}"
            self.pending_img = self.pending_lm = None

    def click(self, x, y):
        if x < self.view_w and y < self.view_h:
            self.pending_img = self.view_to_img(x, y)
            self._try_pair()
        elif x > self.view_w + 3 and y < self.diag.ph:
            lm = self.diag.pick(x - self.view_w - 3, y)
            if lm is not None:
                self.pending_lm = lm
                self._try_pair()

    def undo(self):
        if self.pending_img is not None or self.pending_lm is not None:
            self.pending_img = self.pending_lm = None
        elif self.pairs:
            self.pairs.pop()
            self.lm_nums.pop()

    def reset(self):
        self.pairs, self.lm_nums = [], []
        self.pending_img = self.pending_lm = None

    # --- rajz ---
    def render(self):
        H, errs = self.solve()
        full = annotate_frame(self.frame, self.pairs, H, errs, self.pending_img, self.L, self.W, self.lm_nums)
        s = self._scale()
        M = np.array([[s, 0, self.view_w / 2.0 - self.cx * s], [0, s, self.view_h / 2.0 - self.cy * s]])
        view = cv2.warpAffine(full, M, (self.view_w, self.view_h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if self.mouse is not None and self.mouse[0] < self.view_w and self.mouse[1] < self.view_h:
            self._magnifier(view, full)
        _label(view, f"{format_time(self.t)}  nagyitas {self.zoom:.1f}x", (8, 20))
        pend = None if self.pending_lm is None else self.pending_lm[0]
        dview = self.diag.render(self.lm_nums, pending=pend, hover=self.hover_lm or self.pending_lm)
        status = [f"Parok: {len(self.pairs)}" + ("  (meg kell legalabb 4)" if len(self.pairs) < 4 else "")]
        if errs:
            status.append(f"Hiba: atlag {np.mean([e['px'] for e in errs]):.1f} px, "
                          f"max {max(e['m'] for e in errs):.2f} m")
        elif len(self.pairs) >= 4:
            status.append("A pontok elfajultak (egy egyenesen?)")
        if self.pending_img is not None:
            status.append("Most kattints a rajzon a parjara!")
        elif self.pending_lm is not None:
            status.append("Most kattints a videon a parjara!")
        if self.message:
            status.append(self.message)
        if self.show_help:
            status += [""] + HELP
        return compose(view, dview, status)

    def _magnifier(self, view, full, size=180, factor=4.0):
        u, v = self.view_to_img(*self.mouse)
        half = size / 2.0 / (factor * self.base)
        M = np.array([[factor * self.base, 0, size / 2.0 - u * factor * self.base],
                      [0, factor * self.base, size / 2.0 - v * factor * self.base]])
        mag = cv2.warpAffine(self.frame, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        c = size // 2
        cv2.line(mag, (c - 12, c), (c - 3, c), RED, 1)
        cv2.line(mag, (c + 3, c), (c + 12, c), RED, 1)
        cv2.line(mag, (c, c - 12), (c, c - 3), RED, 1)
        cv2.line(mag, (c, c + 3), (c, c + 12), RED, 1)
        cv2.rectangle(mag, (0, 0), (size - 1, size - 1), WHITE, 1)
        mx, my = self.mouse
        x0 = int(mx + 25) if mx + 25 + size < self.view_w else int(mx - 25 - size)
        y0 = int(min(max(0, my - size / 2), self.view_h - size))
        if half > 0 and 0 <= x0 and x0 + size <= view.shape[1] and y0 + size <= view.shape[0]:
            view[y0:y0 + size, x0:x0 + size] = mag


def display_available():
    """Van-e grafikus megjelenítés (OpenCV GUI-val, kijelzővel)? Visszatér: (bool, magyar indoklás)."""
    if cv2 is None:
        return False, "nincs telepítve az OpenCV (pip install opencv-python)"
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, "nincs grafikus kijelző (DISPLAY) – ezt a gépet csak parancssorból érjük el"
    try:
        info = cv2.getBuildInformation()
    except Exception:
        info = ""
    gui = ""
    for line in info.splitlines():
        if line.strip().startswith("GUI:"):
            gui = line.split(":", 1)[1].strip()
    if gui.upper() in ("NONE", "NO"):
        return False, ("az OpenCV ablakkezelés nélküli (headless) változata van telepítve – "
                       "telepítsd az opencv-python csomagot (pip install opencv-python)")
    return True, ""


KEY_BACKSPACE = (8, 127)
KEY_ENTER = (10, 13)
KEY_ESC = 27


def run_interactive(video, t0, half, pitch_size, json_path):
    src = FrameSource(video)
    frame, t = src.read(t0)
    if frame is None:
        src.release()
        print(f"Hiba: nem sikerült képkockát olvasni a(z) {format_time(t0)} időpontnál.")
        return 2
    tool = CalibrationTool(frame, t, pitch_size)
    win = "Kalibracio - kattints a videon, majd a rajzon"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    except cv2.error as ex:
        src.release()
        print(f"Hiba: nem nyitható meg ablak ({ex}). Grafikus kijelzővel rendelkező gépen futtasd.")
        return 3

    def on_mouse(event, x, y, flags, _param):
        tool.mouse = (x, y)
        if x > tool.view_w + 3:
            tool.hover_lm = tool.diag.pick(x - tool.view_w - 3, y) if y < tool.diag.ph else None
        else:
            tool.hover_lm = None
        if event == cv2.EVENT_LBUTTONDOWN:
            tool.click(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL and x < tool.view_w:
            tool.zoom_at(x, y, 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8)

    cv2.setMouseCallback(win, on_mouse)
    result = None
    try:
        while True:
            cv2.imshow(win, tool.render())
            k = cv2.waitKeyEx(30)
            if k == -1:
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break                       # az ablakot bezárták
                except cv2.error:
                    break
                continue
            ch = k & 0xFF
            step = None
            if ch in (ord("a"),) or k == 2424832:
                step = -1.0
            elif ch in (ord("d"),) or k == 2555904:
                step = 1.0
            elif ch in (ord("A"), ord("j")) or k == 2162688:
                step = -10.0
            elif ch in (ord("D"), ord("l")) or k == 2228224:
                step = 10.0
            if step is not None:
                fr, nt = src.read(tool.t + step)
                if fr is not None:
                    had = len(tool.pairs)
                    tool.set_frame(fr, nt)
                    tool.message = "Új képkocka" + (" – a pontok törölve" if had else "")
                continue
            if ch in KEY_ENTER:
                if len(tool.pairs) < 4:
                    tool.message = "Legalább 4 pár kell!"
                    continue
                H, errs = tool.solve()
                if H is None:
                    tool.message = "A pontokból nem számolható kalibráció (egy egyenesen vannak?)"
                    continue
                result = (list(tool.pairs), errs, tool.t, (tool.fw, tool.fh), list(tool.lm_nums))
                break
            if ch in KEY_BACKSPACE or ch == ord("u"):
                tool.undo()
            elif ch == ord("r"):
                tool.reset()
            elif ch == ord("f"):
                tool.zoom, tool.cx, tool.cy = 1.0, tool.fw / 2.0, tool.fh / 2.0
            elif ch == ord("h"):
                tool.show_help = not tool.show_help
            elif ch in (KEY_ESC, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        src.release()
    if result is None:
        print("Kilépés mentés nélkül.")
        return 1
    pairs, errs, t, (w, h), nums = result
    names = {n: nm for n, nm, _x, _y in landmarks(*pitch_size)}
    finish(video, json_path, half, pitch_size, pairs, errs, t, w, h, [names.get(n) for n in nums])
    return 0


def finish(video, json_path, half, pitch_size, pairs, errs, t, w, h, names=None):
    """Kiírás + JSON mentés (a GUI és a tesztek közös része)."""
    entry = make_half_entry(pairs, w, h, t, errs, names)
    save_calibration_json(json_path, half, entry, pitch_size, video)
    a = format_args(pairs, w, h, t)
    print(f"\nKalibráció kész ({half}. félidő, {format_time(t)}, {len(pairs)} pontpár).")
    if errs:
        print(f"Visszavetítési hiba: átlag {np.mean([e['px'] for e in errs]):.1f} px, "
              f"legnagyobb {max(e['m'] for e in errs):.2f} m")
        if max(e['m'] for e in errs) > 1.5:
            print("Figyelem: valamelyik pont 1,5 m-nél jobban eltér – ellenőrizd, jó pontot választottál-e a rajzon.")
    print(f"Mentve: {json_path}  (az elemzes.py magától használja, ha nincs --kalibracio kapcsoló)")
    print("\nAz elemzes.py kapcsolói (bemásolhatók):")
    print(f"  {a['args']}")
    if tuple(pitch_size) != (STD_LENGTH, STD_WIDTH):
        print(f"\nMegjegyzés: a pálya {pitch_size[0]:g} x {pitch_size[1]:g} m – az elemző motor egyelőre a "
              "105 x 68 m-es modellel számol (lásd README), a pontok a valódi méretben kerültek mentésre.")


def render_preview_from_json(video, json_path, half, out_path, t_override=None):
    """Ablak nélküli előnézet: a mentett kalibráció vonalai a hozzá tartozó képkockán (JPG)."""
    data = load_calibration_json(json_path)
    e = (data or {}).get("felidok", {}).get(str(int(half)))
    if not e:
        print(f"Hiba: nincs mentett kalibráció a {half}. félidőhöz ({json_path}).")
        return 2
    L, W = (data.get("palya") or [STD_LENGTH, STD_WIDTH])[:2]
    src = FrameSource(video)
    frame, t = src.read(t_override if t_override is not None else float(e.get("kalibracio_ido") or 0.0))
    src.release()
    if frame is None:
        print("Hiba: nem sikerült képkockát olvasni.")
        return 2
    h, w = frame.shape[:2]
    pairs = [((p["kep_px"][0] * w / e["kepmeret"][0], p["kep_px"][1] * h / e["kepmeret"][1]),
              tuple(p["palya_m"])) for p in e.get("pontok", [])]
    tool = CalibrationTool(frame, t, (L, W))
    nums = []
    for _, (x, y) in pairs:
        lm = nearest_landmark(tool.diag.lms, x, y, max_dist=0.05)
        nums.append(lm[0] if lm else 0)
    tool.pairs, tool.lm_nums = pairs, nums
    tool.show_help = False
    img = tool.render()
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Előnézet mentve: {out_path}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="kalibracio.py",
        description="Kattintós pályakalibráció: pontpárok a videó képén és a pálya rajzán; az eredmény "
                    "az elemzes.py kapcsolói és a <videó>.kalibracio.json fájl.")
    p.add_argument("video", help="a meccs videója")
    p.add_argument("--ido", default="10:00", metavar="IDŐ",
                   help="melyik időpont képén kalibrálunk (pp:mm, ó:pp:mm vagy mp; alap: 10:00) – "
                        "a/d és A/D billentyűkkel lehet léptetni, hogy sok vonal látsszon")
    p.add_argument("--felido", type=int, default=1, choices=(1, 2),
                   help="melyik félidő kalibrációja (alap: 1; ha a szünetben odébb tették a kamerát, "
                        "a 2. félidőt is kalibráld --felido 2-vel)")
    p.add_argument("--palya", default=None, metavar="HOSSZxSZÉL",
                   help="a pálya mérete méterben, pl. 100x64 (alap: 105x68, vagy a JSON-ban mentett)")
    p.add_argument("--json", default=None, metavar="FÁJL",
                   help="a kalibrációs fájl útvonala (alap: <videó>.kalibracio.json)")
    p.add_argument("--elonezet", default=None, metavar="KÉP.jpg",
                   help="ablak nélkül: a mentett kalibráció vetített vonalait rajzolja a képkockára és elmenti")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.video):
        print(f"Hiba: a videófájl nem található: {args.video}")
        return 2
    json_path = args.json or json_path_for(args.video)
    try:
        t0 = parse_time(args.ido)
        if args.palya:
            size = parse_pitch_size(args.palya)
        else:
            old = load_calibration_json(json_path)
            size = tuple((old or {}).get("palya") or (STD_LENGTH, STD_WIDTH))
    except ValueError as ex:
        print(f"Hiba: {ex}")
        return 2
    if cv2 is None:
        print("Hiba: nincs telepítve az OpenCV (pip install opencv-python).")
        return 3
    if args.elonezet:
        return render_preview_from_json(args.video, json_path, args.felido, args.elonezet)
    ok, why = display_available()
    if not ok:
        print(f"Hiba: az eszköz ablakot nyitna, de {why}.\n"
              "Tipp: futtasd a saját gépeden (Windows: python tools\\kalibracio.py MECCS.mp4 --ido 10:00).")
        return 3
    return run_interactive(args.video, t0, args.felido, size, json_path)


if __name__ == "__main__":
    sys.exit(main())

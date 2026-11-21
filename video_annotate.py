"""
Rajzos videóexport a taktikai tábla "Videó elemzés" módjához, és a szerveren lévő
videók listája.

A tábla a videóra rajzolt elemeket "rajz-megállókhoz" köti: {t, items, hold_sec, note}.
- t: a videó ideje másodpercben (a böngésző video.currentTime értéke)
- items: rajzelemek, a koordináták a videókép arányában (0–1; x a szélesség, y a
  magasság mentén) – így minden felbontáson ugyanott vannak
- hold_sec: a kimerevítés hossza az exportban (alapértelmezés: 3 mp)
- note: rövid cím / megjegyzés (a kimerevített kép bal felső sarkába kerül)

Rajzelem-típusok (ugyanaz, mint a táblán): arrow (nyíl), run (szaggatott nyíl – futás),
line (vonal), circle (karika: pts[0] a közép, pts[1] a kerület egy pontja),
zone (félig átlátszó téglalap), free (szabadkézi vonal), text (felirat, pts[0] a közepe),
spot (kiemelés: fényes gyűrű, körülötte elsötétített kép; pts mint a karikánál).

Méretezés (a böngészővel egyezően): egy "egység" a videókép szélességének 1/80-a
(VUNIT); a vonalvastagság 0,35, a nyílhegy 1,6, a felirat 2,6 egység.

Export (burn_stops): a videó a megállóknál kettévágva, közéjük a rajzos állókép
(hold_sec hosszan) kerül. Csak a kimerevített részekhez kell képet rajzolni; a
mozgó szakaszokat a clip_export.cut_clip vágja képkocka-pontosan (újrakódolással),
a darabokat a clip_export.concat_clips fűzi össze (concat demuxer, újrakódolás nélkül).
A kész videó hossza: az eredeti + a kimerevítések összege.

Képkocka-hozzárendelés: a böngésző a t időpontban a floor(t * fps) sorszámú képkockát
mutatja; a k. képkocka a (k - 0,25) / fps időtől kezdve "pontos kereséssel" vágható ki.

Felirat: Pillow + TrueType betű (DejaVu / Liberation / Arial), így az ékezetek (ő, ű)
is helyesek. Pillow nélkül OpenCV rajzol, a feliratból ekkor az ékezetek elmaradnak.
"""
import os
import re
import math
import time
import shutil
import tempfile
import unicodedata
import subprocess
from urllib.parse import quote, unquote

import cv2
import numpy as np

import clip_export as ce

try:  # opcionális: szebb (élsimított) rajz és ékezetes felirat
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:  # pragma: no cover - Pillow nélküli gépen
    HAVE_PIL = False

VUNIT = 80.0                 # egy egység = a képszélesség 1/80-a
DEFAULT_HOLD_SEC = 3.0
MIN_HOLD_SEC = 0.2
MAX_HOLD_SEC = 60.0
MAX_STOPS = 200
MAX_ITEMS_PER_STOP = 300
ITEM_TYPES = ("arrow", "run", "line", "circle", "zone", "free", "text", "spot")
VIDEO_EXTS = (".mp4",)
_SKIP_PREFIXES = ("temp_cut_",)
_HEX_RE = re.compile(r"^#([0-9a-fA-F]{6})$")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "DejaVuSans-Bold.ttf", "arialbd.ttf",   # a Pillow a saját keresési útvonalán is megnézi
]


# ---------------------------------------------------------------------------
# Videók listája (GET /videos)
# ---------------------------------------------------------------------------
_probe_cache = {}


def probe_video(path):
    """{fps, frames, width, height, duration_sec} OpenCV-vel (None, ha nem olvasható). Gyorsítótárazva."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    if st.st_size < 1024:          # ennyiben nincs videó (félig írt / sérült fájl)
        return None
    key = (os.path.abspath(path), st.st_mtime, st.st_size)
    if key in _probe_cache:
        return _probe_cache[key]
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            info = None
        else:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if not (0 < fps < 400):
                fps = 0.0
            info = {"fps": round(fps, 3) if fps else None, "frames": frames or None, "width": w or None,
                    "height": h or None,
                    "duration_sec": round(frames / fps, 3) if fps and frames else None}
    finally:
        cap.release()
    if len(_probe_cache) > 2000:
        _probe_cache.clear()
    _probe_cache[key] = info
    return info


def _video_kind(name):
    low = name.lower()
    for prefix, kind in (("raw_", "raw"), ("match_", "match"), ("half1_", "half"), ("half2_", "half"),
                         ("rajzos_", "export"), ("burned_", "export")):
        if low.startswith(prefix):
            return kind
    return "other"


def list_videos(clips_dir, url_prefix="/clips", probe=True):
    """
    A clips mappa (nem rekurzívan) .mp4 fájljai és az elemzések JSON fájljai.
    Visszatér: {"videos": [{name, url, size, mtime, kind, fps?, duration_sec?, width?, height?, events_url?}],
                "events": [{name, url, size, mtime}]}, mindkettő a legújabb elöl.
    events_url: a raw_<id>.mp4 klip elemzése (events_<id>.json), ha megvan.
    """
    videos, events = [], []
    try:
        names = os.listdir(clips_dir)
    except OSError:
        return {"videos": [], "events": []}
    name_set = set(names)
    for fn in names:
        path = os.path.join(clips_dir, fn)
        if not os.path.isfile(path):
            continue
        low = fn.lower()
        try:
            st = os.stat(path)
        except OSError:
            continue
        base = {"name": fn, "url": f"{url_prefix}/{quote(fn)}", "size": st.st_size,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)), "_ts": st.st_mtime}
        if low.endswith(VIDEO_EXTS) and not low.startswith(_SKIP_PREFIXES) and st.st_size > 0:
            item = dict(base, kind=_video_kind(fn))
            m = re.match(r"^raw_([A-Za-z0-9_-]+)\.mp4$", fn)
            if m and f"events_{m.group(1)}.json" in name_set:
                item["events_url"] = f"{url_prefix}/events_{m.group(1)}.json"
            if probe:
                info = probe_video(path) or {}
                for k in ("fps", "duration_sec", "width", "height"):
                    if info.get(k):
                        item[k] = info[k]
            videos.append(item)
        elif re.match(r"^(match_)?events_[A-Za-z0-9_-]+\.json$", fn):
            events.append(base)
    for lst in (videos, events):
        lst.sort(key=lambda d: (d["_ts"], d["name"]), reverse=True)
        for d in lst:
            d.pop("_ts", None)
    return {"videos": videos, "events": events}


def resolve_clip_url(url, clips_dir, url_prefix="/clips"):
    """A '/clips/<fájlnév>.mp4' URL → abszolút útvonal a clips mappában. Hibánál ValueError (magyarul)."""
    if not isinstance(url, str) or not url:
        raise ValueError("Hiányzik a videó címe (video_url).")
    path_part = url.split("?", 1)[0].split("#", 1)[0]
    prefix = url_prefix.rstrip("/") + "/"
    if not path_part.startswith(prefix):
        raise ValueError("Csak a szerver clips mappájában lévő videó exportálható.")
    name = unquote(path_part[len(prefix):])
    if not name or "/" in name or "\\" in name or name.startswith(".") or "\x00" in name:
        raise ValueError("Érvénytelen videónév.")
    if not name.lower().endswith(VIDEO_EXTS):
        raise ValueError("Csak .mp4 videó exportálható.")
    root = os.path.abspath(clips_dir)
    path = os.path.abspath(os.path.join(root, name))
    if os.path.dirname(path) != root or not os.path.isfile(path):
        raise ValueError(f"A videó nem található a szerveren: {name}")
    return path


# ---------------------------------------------------------------------------
# Megállók ellenőrzése
# ---------------------------------------------------------------------------
def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clean_item(it):
    if not isinstance(it, dict) or it.get("type") not in ITEM_TYPES:
        return None
    pts = []
    for p in it.get("pts") or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = _finite(p[0]), _finite(p[1])
            if x is not None and y is not None:
                pts.append((max(-1.0, min(2.0, x)), max(-1.0, min(2.0, y))))
        if len(pts) >= 5000:
            break
    need = 1 if it["type"] in ("text", "free") else 2
    if len(pts) < need:
        return None
    color = it.get("color") if isinstance(it.get("color"), str) and _HEX_RE.match(it.get("color")) else "#ffd400"
    out = {"type": it["type"], "pts": pts, "color": color}
    if it["type"] == "text":
        text = str(it.get("text") or "").strip()[:80]
        if not text:
            return None
        out["text"] = text
    return out


def normalize_stops(stops, fps, frames=None, default_hold=DEFAULT_HOLD_SEC, allow_empty=False):
    """
    A kliens megállói → [{frame, t, hold_sec, items, note}], képkocka szerint rendezve.
    Az azonos képkockára eső megállók összevonva (a rajzok együtt, a hosszabb kitartás).
    Hibánál ValueError (magyar üzenettel).
    """
    if stops is None and allow_empty:
        stops = []
    if not isinstance(stops, list) or (not stops and not allow_empty):
        raise ValueError("Nincs rajz-megálló: előbb rajzolj a videóra.")
    if len(stops) > MAX_STOPS:
        raise ValueError(f"Túl sok rajz-megálló (legfeljebb {MAX_STOPS}).")
    if not fps or fps <= 0:
        raise ValueError("A videó képkockasebessége nem állapítható meg.")
    by_frame = {}
    for i, st in enumerate(stops, 1):
        if not isinstance(st, dict):
            raise ValueError(f"A(z) {i}. megálló hibás.")
        t = _finite(st.get("t"))
        if t is None or t < 0:
            raise ValueError(f"A(z) {i}. megálló ideje hibás.")
        hold = _finite(st.get("hold_sec"))
        hold = default_hold if hold is None or hold <= 0 else hold
        hold = max(MIN_HOLD_SEC, min(MAX_HOLD_SEC, hold))
        items = [c for c in (_clean_item(it) for it in (st.get("items") or [])[:MAX_ITEMS_PER_STOP]) if c]
        note = re.sub(r"[\x00-\x1f\x7f]", " ", str(st.get("note") or "")).strip()[:120]
        if not items and not note:
            continue
        frame = int(math.floor(t * fps + 1e-3))
        if frames:
            frame = max(0, min(int(frames) - 1, frame))
        cur = by_frame.get(frame)
        if cur:
            cur["items"] += items
            cur["hold_sec"] = max(cur["hold_sec"], hold)
            cur["note"] = " · ".join(x for x in (cur["note"], note) if x)
        else:
            by_frame[frame] = {"frame": frame, "t": round(t, 4), "hold_sec": hold, "items": items, "note": note}
    if not by_frame and not allow_empty:
        raise ValueError("A rajz-megállókon nincs rajz.")
    return [by_frame[k] for k in sorted(by_frame)]


# ---------------------------------------------------------------------------
# Rajzolás egy képkockára
# ---------------------------------------------------------------------------
def _rgb(hex_color):
    m = _HEX_RE.match(hex_color or "")
    if not m:
        return (255, 212, 0)
    v = m.group(1)
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))


_font_cache = {}


def find_font_path():
    """Az első betöltető TrueType betű útvonala (ékezetekkel), vagy None."""
    if "path" in _font_cache:
        return _font_cache["path"]
    found = None
    if HAVE_PIL:
        for cand in FONT_CANDIDATES:
            try:
                ImageFont.truetype(cand, 20)
                found = cand
                break
            except Exception:
                continue
    _font_cache["path"] = found
    return found


def _font(size):
    path = find_font_path()
    if not path:
        return None
    key = (path, int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, int(size))
    return _font_cache[key]


def strip_accents(text):
    """OpenCV-hez: ő → o, ű → u stb. (a Hershey betűkben nincs ékezet)."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")


def _dash_segments(p0, p1, dash, gap):
    """Szaggatott vonal darabjai p0 → p1 között [(a, b), ...]."""
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return []
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    out, pos = [], 0.0
    while pos < length:
        e = min(length, pos + dash)
        out.append(((x0 + ux * pos, y0 + uy * pos), (x0 + ux * e, y0 + uy * e)))
        pos = e + gap
    return out


class _Geom:
    """A böngészővel egyező méretek egy W×H képen (képpontban)."""

    def __init__(self, w, h, ss=1):
        self.w, self.h, self.ss = w, h, ss
        self.unit = w / VUNIT * ss
        self.lw = max(2.0 * ss, 0.35 * self.unit)
        self.head = max(10.0 * ss, 1.6 * self.unit)

    def pt(self, p):
        return (p[0] * self.w * self.ss, p[1] * self.h * self.ss)


def _arrow_geom(g, d):
    P = [g.pt(p) for p in d["pts"]]
    (x1, y1), (x2, y2) = P[0], P[-1]
    head = 0 if d["type"] == "line" else g.head
    a = math.atan2(y2 - y1, x2 - x1)
    end = (x2 - head * 0.7 * math.cos(a), y2 - head * 0.7 * math.sin(a))
    tri = None
    if head:
        tri = [(x2, y2), (x2 - head * math.cos(a - 0.45), y2 - head * math.sin(a - 0.45)),
               (x2 - head * math.cos(a + 0.45), y2 - head * math.sin(a + 0.45))]
    return (x1, y1), end, tri


def _render_pil(frame_bgr, items, note):
    h, w = frame_bgr.shape[:2]
    ss = 2 if max(w, h) <= 2200 else 1          # élsimítás: kétszeres felbontáson rajzolunk
    g = _Geom(w, h, ss)
    W2, H2 = w * ss, h * ss
    layer = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    lw = max(1, int(round(g.lw)))

    def round_line(pts, color, width):
        if len(pts) < 2:
            return
        dr.line(pts, fill=color, width=width, joint="curve")
        r = width / 2.0
        for x, y in (pts[0], pts[-1]):
            dr.ellipse([x - r, y - r, x + r, y + r], fill=color)

    spots = [d for d in items if d["type"] == "spot"]
    if spots:
        # elsötétítés a kiemelt körökön kívül
        mask = Image.new("L", (W2, H2), int(255 * 0.55))
        md = ImageDraw.Draw(mask)
        for d in spots:
            (cx, cy), (ex, ey) = g.pt(d["pts"][0]), g.pt(d["pts"][1])
            r = math.hypot(ex - cx, ey - cy)
            md.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
        layer.paste(Image.new("RGBA", (W2, H2), (0, 0, 0, 255)), (0, 0), mask)

    for d in items:
        col = _rgb(d["color"]) + (255,)
        t = d["type"]
        if t in ("arrow", "run", "line"):
            a, end, tri = _arrow_geom(g, d)
            if t == "run":
                for s0, s1 in _dash_segments(a, end, 1.2 * g.unit, 0.9 * g.unit):
                    round_line([s0, s1], col, lw)
            else:
                round_line([a, end], col, lw)
            if tri:
                dr.polygon(tri, fill=col)
        elif t in ("circle", "spot"):
            (cx, cy), (ex, ey) = g.pt(d["pts"][0]), g.pt(d["pts"][1])
            r = math.hypot(ex - cx, ey - cy)
            width = lw if t == "circle" else max(1, int(round(g.lw * 1.3)))
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=width)
            if t == "spot":
                r2 = r + width * 1.4
                dr.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=(255, 255, 255, 110),
                           width=max(1, width // 2))
        elif t == "zone":
            (x0, y0), (x1, y1) = g.pt(d["pts"][0]), g.pt(d["pts"][1])
            box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            fill = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
            ImageDraw.Draw(fill).rectangle(box, fill=_rgb(d["color"]) + (int(255 * 0.22),))
            layer.alpha_composite(fill)
            dr = ImageDraw.Draw(layer)
            corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]), (box[0], box[1])]
            for i in range(4):
                for s0, s1 in _dash_segments(corners[i], corners[i + 1], 0.8 * g.unit, 0.6 * g.unit):
                    dr.line([s0, s1], fill=col, width=lw)
        elif t == "free":
            round_line([g.pt(p) for p in d["pts"]], col, lw)
        elif t == "text":
            size = max(14 * ss, 2.6 * g.unit)
            font = _font(size)
            x, y = g.pt(d["pts"][0])
            if font is not None:
                dr.text((x, y), d["text"], font=font, fill=col, anchor="mm",
                        stroke_width=max(1, int(round(2 * ss * max(1.0, g.unit / 12)))), stroke_fill=(0, 0, 0, 170))
    if note:
        size = max(14 * ss, 2.0 * g.unit)
        font = _font(size)
        if font is not None:
            pad = size * 0.45
            x0, y0 = 1.0 * g.unit, 1.0 * g.unit
            bbox = dr.textbbox((x0 + pad, y0 + pad), note, font=font)
            dr.rounded_rectangle([x0, y0, bbox[2] + pad, bbox[3] + pad], radius=pad * 0.8, fill=(0, 0, 0, 150))
            dr.text((x0 + pad, y0 + pad), note, font=font, fill=(255, 255, 255, 255))
    if ss != 1:
        layer = layer.resize((w, h), Image.LANCZOS)
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    base.alpha_composite(layer)
    out = cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)
    # a Pillow nélküli felirat-ág (nincs TrueType betű): OpenCV-vel, ékezet nélkül
    if find_font_path() is None:
        _cv_texts(out, items, note, _Geom(w, h))
    return out


def _cv_texts(img, items, note, g):
    for d in items:
        if d["type"] != "text":
            continue
        text = strip_accents(d["text"])
        scale = max(0.5, 2.6 * g.unit / 30.0)
        thick = max(1, int(round(scale * 2)))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        x, y = g.pt(d["pts"][0])
        org = (int(x - tw / 2), int(y + th / 2))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        r, gg, b = _rgb(d["color"])
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (b, gg, r), thick, cv2.LINE_AA)
    if note:
        text = strip_accents(note)
        scale = max(0.5, 2.0 * g.unit / 30.0)
        thick = max(1, int(round(scale * 2)))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        x0 = y0 = int(g.unit)
        pad = int(th * 0.5)
        sub = img[y0:y0 + th + 2 * pad, x0:x0 + tw + 2 * pad]
        sub[:] = (sub * 0.4).astype(img.dtype)
        cv2.putText(img, text, (x0 + pad, y0 + pad + th), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick,
                    cv2.LINE_AA)


def _render_cv(frame_bgr, items, note):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    g = _Geom(w, h)
    lw = max(1, int(round(g.lw)))
    ip = lambda p: (int(round(p[0])), int(round(p[1])))
    spots = [d for d in items if d["type"] == "spot"]
    if spots:
        mask = np.zeros((h, w), np.uint8)
        for d in spots:
            c, e = g.pt(d["pts"][0]), g.pt(d["pts"][1])
            cv2.circle(mask, ip(c), int(math.hypot(e[0] - c[0], e[1] - c[1])), 255, -1)
        dark = (img * 0.45).astype(img.dtype)
        img[mask == 0] = dark[mask == 0]
    for d in items:
        r, gg, b = _rgb(d["color"])
        col = (b, gg, r)
        t = d["type"]
        if t in ("arrow", "run", "line"):
            a, end, tri = _arrow_geom(g, d)
            segs = _dash_segments(a, end, 1.2 * g.unit, 0.9 * g.unit) if t == "run" else [(a, end)]
            for s0, s1 in segs:
                cv2.line(img, ip(s0), ip(s1), col, lw, cv2.LINE_AA)
            if tri:
                cv2.fillPoly(img, [np.array([ip(p) for p in tri], np.int32)], col, cv2.LINE_AA)
        elif t in ("circle", "spot"):
            c, e = g.pt(d["pts"][0]), g.pt(d["pts"][1])
            cv2.circle(img, ip(c), int(math.hypot(e[0] - c[0], e[1] - c[1])), col,
                       lw if t == "circle" else max(1, int(round(g.lw * 1.3))), cv2.LINE_AA)
        elif t == "zone":
            (x0, y0), (x1, y1) = g.pt(d["pts"][0]), g.pt(d["pts"][1])
            p0, p1 = ip((min(x0, x1), min(y0, y1))), ip((max(x0, x1), max(y0, y1)))
            over = img.copy()
            cv2.rectangle(over, p0, p1, col, -1)
            cv2.addWeighted(over, 0.22, img, 0.78, 0, img)
            cv2.rectangle(img, p0, p1, col, lw, cv2.LINE_AA)
        elif t == "free":
            pts = np.array([ip(g.pt(p)) for p in d["pts"]], np.int32)
            cv2.polylines(img, [pts], False, col, lw, cv2.LINE_AA)
    _cv_texts(img, items, note, g)
    return img


def render_overlay(frame_bgr, items, note=None):
    """A rajzelemek (normalizált koordinátákkal) ráégetése egy BGR képkockára. Új képet ad vissza."""
    clean = [c for c in (_clean_item(it) for it in (items or [])) if c]
    if HAVE_PIL:
        return _render_pil(frame_bgr, clean, note)
    return _render_cv(frame_bgr, clean, note)


# ---------------------------------------------------------------------------
# Export ffmpeg-gel
# ---------------------------------------------------------------------------
def _has_audio(video_path):
    """Van-e hangsáv (ffmpeg -i kimenetéből; ffprobe nem kell)."""
    try:
        res = subprocess.run([ce.FFMPEG, "-hide_banner", "-i", video_path], capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(f"Az ffmpeg nem található ({ce.FFMPEG}). Állítsd be az FFMPEG_PATH környezeti változót.")
    return re.search(r"Stream #\d+:\d+.*?: Audio:", res.stderr.decode("utf-8", "replace")) is not None


def _frame_time(k, fps):
    """
    A k. képkocka "pontos kereséshez" (negyed képkockával előtte): így a (k-1). már kimarad, és a
    kimenet állandó képkockasebességű rácsán a k. kocka a 0. helyre kerül (fél kockánál az ffmpeg
    egy hellyel későbbre kerekítene, és a részlet utolsó kockája elveszne).
    """
    return 0.0 if k <= 0 else (k - 0.25) / fps


def extract_frame(video_path, k, fps, out_png):
    ce._run([ce.FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{_frame_time(k, fps):.4f}",
             "-i", video_path, "-frames:v", "1", out_png])
    img = cv2.imread(out_png)
    if img is None:
        raise RuntimeError(f"A(z) {k}. képkocka nem olvasható ki.")
    return img


def encode_still(png_path, hold_sec, fps, out_path, audio):
    """Állókép-szakasz: a clip_export.cut_clip kódolási beállításaival, hogy a concat újrakódolás nélkül fűzzön."""
    n = max(1, int(round(hold_sec * fps)))
    cmd = [ce.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-loop", "1", "-framerate", f"{fps:.6f}", "-i", png_path]
    if audio:
        cmd += ["-f", "lavfi", "-t", f"{n / fps:.4f}", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-map", "0:v:0"] + (["-map", "1:a:0"] if audio else [])
    cmd += ["-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-r", f"{fps:.6f}"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
    cmd += ["-movflags", "+faststart", out_path]
    ce._run(cmd)
    return n


def burn_stops(video_path, stops, out_path, default_hold=DEFAULT_HOLD_SEC, progress_cb=None, keep_temp=False,
               start_sec=None, end_sec=None):
    """
    A rajzos export elkészítése. stops: a kliens megállói (lásd a modul leírását).
    start_sec / end_sec: csak ez a részlet (a benne lévő megállókkal); None = a teljes videó.
    A részlet határa is képkocka-pontos: floor(start * fps) .. floor(end * fps) (kizárólag).
    progress_cb(kész_lépés, összes_lépés, felirat) – kivételét továbbengedi (megszakítás).
    Visszatér: {"status": "success", "local_path", "duration_sec", "stops", "hold_total_sec", "fps", "frames"}.
    Hibánál ValueError (rossz bemenet) vagy RuntimeError (ffmpeg).
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError("A videófájl nem található.")
    info = probe_video(video_path) or {}
    fps, frames = info.get("fps"), info.get("frames")
    if not fps:
        raise ValueError("A videó nem olvasható (képkockasebesség ismeretlen).")
    first = 0 if start_sec is None else max(0, int(math.floor(float(start_sec) * fps + 1e-3)))
    last = frames if end_sec is None else int(math.floor(float(end_sec) * fps + 1e-3))
    if frames:
        last = min(frames, last) if last is not None else frames
    if last is not None and last - first < 1:
        raise ValueError("A részlet üres: a vége legyen a kezdete után.")
    ranged = start_sec is not None or end_sec is not None
    plan = normalize_stops(stops, fps, frames, default_hold, allow_empty=ranged)
    plan = [st for st in plan if st["frame"] >= first and (last is None or st["frame"] < last)]
    if not plan and not ranged:
        raise ValueError("Nincs rajz-megálló: előbb rajzolj a videóra.")
    audio = _has_audio(video_path)
    tmp = tempfile.mkdtemp(prefix="rajzos_", dir=os.path.dirname(os.path.abspath(out_path)))
    parts, pos, hold_total = [], first, 0.0
    steps = 2 * len(plan) + 2
    done = 0

    def step(label):
        nonlocal done
        done += 1
        if progress_cb is not None:
            progress_cb(done, steps, label)

    try:
        for i, st in enumerate(plan, 1):
            k = st["frame"]
            if k > pos:
                seg = os.path.join(tmp, f"seg_{i:03d}.mp4")
                ce.cut_clip(video_path, _frame_time(pos, fps), _frame_time(k, fps), seg, accurate=True)
                parts.append(seg)
            step(f"{i}/{len(plan)}. szakasz kivágva")
            png = os.path.join(tmp, f"still_{i:03d}.png")
            frame = extract_frame(video_path, k, fps, png)
            cv2.imwrite(png, render_overlay(frame, st["items"], st["note"]))
            still = os.path.join(tmp, f"still_{i:03d}.mp4")
            n = encode_still(png, st["hold_sec"], fps, still, audio)
            hold_total += n / fps
            parts.append(still)
            pos = k
            step(f"{i}/{len(plan)}. rajzos állókép kész")
        if last is None or pos < last:
            seg = os.path.join(tmp, "seg_end.mp4")
            if last is not None and (end_sec is not None):
                end = _frame_time(last, fps)
            else:
                end = (info.get("duration_sec") or (pos / fps)) + 10.0
            ce.cut_clip(video_path, _frame_time(pos, fps), end, seg, accurate=True)
            parts.append(seg)
        step("Utolsó szakasz kivágva")
        if len(parts) == 1:
            shutil.copyfile(parts[0], out_path)
        else:
            ce.concat_clips(parts, out_path)
        step("Összefűzve")
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("Az ffmpeg nem hozta létre a kész videót.")
        out_info = probe_video(out_path) or {}
        return {"status": "success", "local_path": os.path.abspath(out_path),
                "duration_sec": out_info.get("duration_sec"), "stops": len(plan),
                "hold_total_sec": round(hold_total, 3), "fps": fps, "frames": out_info.get("frames")}
    finally:
        if not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)


def render_still_png(video_path, t, items, note, out_png):
    """Egyetlen rajzos képkocka PNG-be (a "Kép mentése" szerveres változatához / teszthez)."""
    info = probe_video(video_path) or {}
    fps = info.get("fps") or 25.0
    k = int(math.floor(float(t) * fps + 1e-3))
    if info.get("frames"):
        k = max(0, min(info["frames"] - 1, k))
    frame = extract_frame(video_path, k, fps, out_png)
    cv2.imwrite(out_png, render_overlay(frame, items, note))
    return out_png

"""
Hőtérképek és játékos-statisztikák a követett pályakoordinátákból.

Bemenet: a TacticalEngine.player_trajectories szerkezete
    {track_id: [(x_m, y_m, t_sec), ...]}  (105 x 68 m-es pálya)
és követésenkénti csapat/kapus adat (lásd TrackMetaCollector).

- Csapat- és játékos-hőtérkép: idővel súlyozott foglaltsági rács (alapból 1 m-es
  cellák), Gauss-simítással, 0..1-re normálva. A súly a következő mintáig eltelt
  idő; a hosszú kimaradásokat (> max_gap_sec) nem számoljuk teljes hosszal, így a
  hiányzó felismerések nem „égetnek be” egy pontot.
- Második félidő tükrözése (flip_at_sec): az ennél későbbi mintáknál az x tengelyt
  tükrözzük, így a két félidő egy irányba támadva összeadható.
- Megjelenítés: a review_export.draw_minimap stílusú tiszta pályán, színskálás,
  átlátszó (ahol nincs adat, ott a pálya látszik) rétegként, PNG-be.
- Statisztika: megtett táv (a > 10 m/s-os ugrásokat zajnak vesszük), átlagos
  pozíció, követett idő.

Csak numpy + OpenCV – a torch/ultralytics nélkül is használható és tesztelhető.
"""
import os
import re
import json
import uuid
from collections import Counter, defaultdict

import cv2
import numpy as np

from review_export import draw_minimap, _label, _ascii

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

DEFAULT_BIN_M = 1.0
DEFAULT_SIGMA_M = 2.0
DEFAULT_MAX_GAP_SEC = 1.0
MAX_SPEED_MPS = 10.0
MIN_PLAYER_SEC = 20.0
# A megtett táv számolásánál ennyi időnként veszünk mintát, hogy a 25 fps-es
# homográfia-remegés ne adódjon össze
DISTANCE_MIN_STEP_SEC = 0.2


# ---------------------------------------------------------------------------
# Követésenkénti csapat / kapus adat és tömörített mentés
# ---------------------------------------------------------------------------

class TrackMetaCollector:
    """
    A feldolgozó ciklusban gyűjti, hogy melyik követés (track_id) melyik csapatba
    sorolódott a leggyakrabban, és volt-e valaha kapusnak jelölve.
    """

    def __init__(self):
        self._teams = defaultdict(Counter)
        self._keeper = {}

    def add(self, players):
        for p in players or []:
            tid = p.get('track_id')
            if tid is None:
                continue
            if p.get('team_id') is not None:
                self._teams[tid][int(p['team_id'])] += 1
            self._keeper[tid] = self._keeper.get(tid, False) or bool(p.get('is_keeper'))

    def summary(self):
        """{track_id: {"team": 1/2/None, "keeper": bool}}"""
        out = {}
        for tid in set(self._teams) | set(self._keeper):
            counts = self._teams.get(tid)
            team = counts.most_common(1)[0][0] if counts else None
            out[tid] = {"team": team, "keeper": self._keeper.get(tid, False)}
        return out


def compact_trajectories(trajectories, track_meta=None, hz=5.0, decimals=1):
    """
    Tömör, JSON-ba írható pályák: hz-re ritkítva, a koordináták `decimals`
    tizedesre kerekítve.
    Kimenet: {"hz": 5.0, "tracks": {"12": {"team": 1, "keeper": false,
              "points": [[x, y, t], ...]}, ...}}
    """
    track_meta = track_meta or {}
    min_dt = 1.0 / hz if hz and hz > 0 else 0.0
    tracks = {}
    for tid, pts in (trajectories or {}).items():
        kept = []
        last_t = None
        for x, y, t in sorted(pts, key=lambda p: p[2]):
            if last_t is not None and (t - last_t) < min_dt - 1e-6:
                continue
            kept.append([round(float(x), decimals), round(float(y), decimals), round(float(t), 2)])
            last_t = t
        if not kept:
            continue
        meta = track_meta.get(tid, {})
        tracks[str(tid)] = {
            "team": meta.get("team"),
            "keeper": bool(meta.get("keeper", False)),
            "points": kept,
        }
    return {"hz": hz, "tracks": tracks}


def load_trajectories(data):
    """
    Az események JSON-jából (dict) visszaadja: (pályák, meta)
        pályák: {track_id(int): [(x, y, t), ...]}
        meta:   {track_id(int): {"team": ..., "keeper": ...}}
    """
    block = (data or {}).get("trajectories") or {}
    tracks = block.get("tracks", {}) if isinstance(block, dict) else {}
    trajectories, meta = {}, {}
    for key, tr in tracks.items():
        try:
            tid = int(key)
        except (TypeError, ValueError):
            continue
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in tr.get("points", []) if len(p) >= 3]
        if pts:
            trajectories[tid] = pts
            meta[tid] = {"team": tr.get("team"), "keeper": bool(tr.get("keeper", False))}
    return trajectories, meta


# ---------------------------------------------------------------------------
# Számítás
# ---------------------------------------------------------------------------

def _prepare(points, flip_at_sec=None):
    """Időrendbe tett numpy tömb (N x 3), a második félidő x-tükrözésével."""
    if not points:
        return np.zeros((0, 3), np.float64)
    arr = np.array(sorted(points, key=lambda p: p[2]), dtype=np.float64)[:, :3]
    if flip_at_sec is not None:
        second = arr[:, 2] >= float(flip_at_sec)
        arr[second, 0] = PITCH_LENGTH - arr[second, 0]
    return arr


def sample_weights(times, max_gap_sec=DEFAULT_MAX_GAP_SEC):
    """
    Mintánkénti idősúly (mp): a következő mintáig eltelt idő. Ha ez nagyobb
    max_gap_sec-nél (kimaradt felismerés), a követés tipikus (medián) lépésközét
    használjuk; az utolsó minta is ezt kapja.
    """
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([min(0.04, max_gap_sec)])
    dt = np.diff(times)
    valid = dt[(dt > 0) & (dt <= max_gap_sec)]
    typical = float(np.median(valid)) if len(valid) else min(0.04, max_gap_sec)
    w = np.empty(n)
    w[:-1] = np.where((dt > 0) & (dt <= max_gap_sec), dt, np.where(dt > max_gap_sec, typical, 0.0))
    w[-1] = typical
    return w


def occupancy_grid(arr, weights, bin_m=DEFAULT_BIN_M):
    """Idősúlyozott foglaltsági rács (sor = y, oszlop = x), mp-ben."""
    nx = int(round(PITCH_LENGTH / bin_m))
    ny = int(round(PITCH_WIDTH / bin_m))
    if len(arr) == 0:
        return np.zeros((ny, nx), np.float64)
    x = np.clip(arr[:, 0], 0.0, PITCH_LENGTH - 1e-6)
    y = np.clip(arr[:, 1], 0.0, PITCH_WIDTH - 1e-6)
    grid, _, _ = np.histogram2d(y, x, bins=[ny, nx], range=[[0, PITCH_WIDTH], [0, PITCH_LENGTH]], weights=weights)
    return grid


def smooth_normalize(grid, bin_m=DEFAULT_BIN_M, sigma_m=DEFAULT_SIGMA_M):
    """Gauss-simítás és 0..1-re normálás (a csúcs = 1)."""
    g = grid.astype(np.float32)
    if sigma_m and sigma_m > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma_m / bin_m, sigmaY=sigma_m / bin_m,
                             borderType=cv2.BORDER_CONSTANT)
    peak = float(g.max()) if g.size else 0.0
    if peak <= 0:
        return np.zeros_like(g)
    return np.clip(g / peak, 0.0, 1.0)


def build_heatmap(point_lists, flip_at_sec=None, bin_m=DEFAULT_BIN_M, sigma_m=DEFAULT_SIGMA_M,
                  max_gap_sec=DEFAULT_MAX_GAP_SEC):
    """
    Több követés (pl. egy csapat összes játékosa) közös, normált hőtérképe.
    point_lists: [[(x, y, t), ...], ...]
    Visszatér: (normált rács, összes idő mp-ben)
    """
    total = None
    total_sec = 0.0
    for pts in point_lists:
        arr = _prepare(pts, flip_at_sec)
        if len(arr) == 0:
            continue
        w = sample_weights(arr[:, 2], max_gap_sec)
        g = occupancy_grid(arr, w, bin_m)
        total = g if total is None else total + g
        total_sec += float(w.sum())
    if total is None:
        total = occupancy_grid(np.zeros((0, 3)), np.zeros(0), bin_m)
    return smooth_normalize(total, bin_m, sigma_m), total_sec


def player_stats(points, flip_at_sec=None, max_gap_sec=DEFAULT_MAX_GAP_SEC, max_speed_mps=MAX_SPEED_MPS,
                 min_step_sec=DISTANCE_MIN_STEP_SEC):
    """
    Egy követés statisztikája:
      distance_m: megtett táv; a min_step_sec-nél sűrűbb mintákat összevonjuk,
                  a max_speed_mps-nél gyorsabb lépést (ugrás/azonosító-csere) kihagyjuk
      avg_x, avg_y: idővel súlyozott átlagpozíció (méter)
      time_sec: követett idő (a kimaradásokat nem számolva teljes hosszal)
    """
    arr = _prepare(points, flip_at_sec)
    if len(arr) == 0:
        return {"distance_m": 0.0, "avg_x": None, "avg_y": None, "time_sec": 0.0, "samples": 0}
    w = sample_weights(arr[:, 2], max_gap_sec)
    total_w = float(w.sum())
    if total_w > 0:
        avg_x = float(np.average(arr[:, 0], weights=w))
        avg_y = float(np.average(arr[:, 1], weights=w))
    else:
        avg_x, avg_y = float(arr[:, 0].mean()), float(arr[:, 1].mean())

    distance = 0.0
    last = arr[0]
    for p in arr[1:]:
        dt = p[2] - last[2]
        if dt < min_step_sec:
            continue
        step = float(np.hypot(p[0] - last[0], p[1] - last[1]))
        if step / dt <= max_speed_mps:
            distance += step
        last = p
    return {
        "distance_m": round(distance, 1),
        "avg_x": round(avg_x, 1),
        "avg_y": round(avg_y, 1),
        "time_sec": round(total_w, 1),
        "samples": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# Megjelenítés
# ---------------------------------------------------------------------------

def render_heatmap(heat, title=None, width=840, max_alpha=0.75, colormap=cv2.COLORMAP_JET, min_level=0.02):
    """
    Normált rácsot (0..1) rajzol a tiszta pályára. Ahol a hőérték < min_level,
    ott átlátszó (a zöld pálya látszik); a pályavonalak a réteg fölé kerülnek.
    """
    base = draw_minimap([], None, width=width)
    h, w = base.shape[:2]
    heat_img = cv2.resize(np.asarray(heat, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat_img = np.clip(heat_img, 0.0, 1.0)
    colored = cv2.applyColorMap((heat_img * 255).astype(np.uint8), colormap)

    alpha = np.where(heat_img >= min_level, max_alpha * np.sqrt(heat_img), 0.0)[..., None]
    out = (base.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha).astype(np.uint8)

    # Pályavonalak visszarajzolása a réteg fölé (ahol az alap nem fűszínű)
    grass = np.array([60, 130, 60], np.uint8)
    line_mask = np.any(base != grass, axis=2)
    out[line_mask] = base[line_mask]

    if title:
        header = np.zeros((34, w, 3), np.uint8)
        header[:] = (40, 40, 40)
        _label(header, _ascii(title), (8, 23), 0.6)
        out = np.vstack([header, out])
    return out


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "x"


def generate_heatmaps(events_data, out_dir, prefix, team="all", player_id=None, flip_at_sec=None,
                      min_player_sec=MIN_PLAYER_SEC, url_prefix="/clips", width=840):
    """
    Az események JSON-jából (dict) PNG hőtérképeket készít az out_dir mappába.
    team: "1", "2" vagy "all"; player_id: csak ennek a követésnek a hőtérképe.
    Visszatér: JSON-ba írható eredmény (lásd /heatmap végpont).
    """
    trajectories, meta = load_trajectories(events_data)
    if not trajectories:
        return {"status": "error",
                "message": "Az eseményfájlban nincsenek pályakoordináták (kalibráció nélkül futott az elemzés?)."}

    team = str(team or "all").strip().lower()
    if team not in ("1", "2", "all"):
        return {"status": "error", "message": f"Érvénytelen csapat: {team} (1, 2 vagy all)."}
    teams = (1, 2) if team == "all" else (int(team),)
    os.makedirs(out_dir, exist_ok=True)
    half_note = " (2. felido tukrozve)" if flip_at_sec is not None else ""

    def save(img, name):
        cv2.imwrite(os.path.join(out_dir, name), img)
        return f"{url_prefix}/{name}"

    team_heatmaps = {}
    if player_id is None:
        for t in teams:
            lists = [pts for tid, pts in trajectories.items() if meta.get(tid, {}).get("team") == t]
            heat, total = build_heatmap(lists, flip_at_sec)
            if not lists:
                team_heatmaps[f"team{t}"] = None
                continue
            img = render_heatmap(heat, f"Csapat {t} hoterkep - {len(lists)} kovetes, {total / 60:.1f} jatekos-perc{half_note}",
                                 width=width)
            team_heatmaps[f"team{t}"] = save(img, f"heatmap_{prefix}_team{t}.png")

    if player_id is not None:
        if int(player_id) not in trajectories:
            return {"status": "error", "message": f"A(z) #{player_id} követés nem szerepel az adatok között."}
        candidates = [int(player_id)]
    else:
        candidates = [tid for tid in trajectories if meta.get(tid, {}).get("team") in teams]

    player_heatmaps, stats_list = [], []
    for tid in sorted(candidates):
        pts = trajectories[tid]
        st = player_stats(pts, flip_at_sec)
        m = meta.get(tid, {})
        entry = {"track_id": tid, "team": m.get("team"), "keeper": m.get("keeper", False), **st}
        stats_list.append(entry)
        if player_id is None and st["time_sec"] < min_player_sec:
            continue
        heat, _ = build_heatmap([pts], flip_at_sec)
        role = " kapus" if m.get("keeper") else ""
        title = (f"#{tid} (csapat {m.get('team')}{role}) - {st['time_sec']:.0f} mp, "
                 f"{st['distance_m']:.0f} m{half_note}")
        url = save(render_heatmap(heat, title, width=width), f"heatmap_{prefix}_p{_slug(tid)}.png")
        player_heatmaps.append({"track_id": tid, "team": m.get("team"), "url": url,
                                "distance_m": st["distance_m"], "time_sec": st["time_sec"]})

    return {
        "status": "success",
        "team_heatmaps": team_heatmaps,
        "player_heatmaps": player_heatmaps,
        "player_stats": stats_list,
    }


def generate_heatmaps_from_file(events_path, out_dir, **kwargs):
    """Fájlból olvasó változat (a végpont ezt hívja a háttérszálon)."""
    if not events_path or not os.path.exists(events_path):
        return {"status": "error", "message": "Az eseményfájl nem található."}
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as ex:
        return {"status": "error", "message": f"Az eseményfájl nem olvasható: {ex}"}
    # Egyedi előtag: az ismételt hívás (pl. más tükrözéssel) ne írja felül a kliens által gyorsítótárazott képet
    prefix = kwargs.pop("prefix", None) or (
        f"{_slug(os.path.splitext(os.path.basename(events_path))[0])}_{uuid.uuid4().hex[:6]}")
    return generate_heatmaps(data, out_dir, prefix, **kwargs)

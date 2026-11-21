"""
A pályakövető (pitch_tracker.py) tesztjei szintetikus jeleneten.
Futtatás: python3 tests/test_pitch_tracker.py
(pytest nem kell; torch/ultralytics nélkül fut, a main.py-t nem importálja)

A jelenet: felülnézeti pályatextúra (csíkos, zajos fű, valós méretű fehér
vonalak, lelátó a pálya körül), ezt egy ismert, mozgó kamerával (felezővonal
vonalában, 6 m-re az oldalvonal mögött, 7 m magas árbocon; pásztázás ±25°,
dőlés- és zoomváltozás) a képre vetítjük, és mozgó játékos-foltokat rajzolunk rá.
"""
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

import pitch_tracker as pt

PPM = 20.0          # textúra: pixel / méter
MARGIN = 20.0       # a pálya körüli sáv (m)
CAM = (pt.PITCH_LENGTH / 2, pt.PITCH_WIDTH + 6.0, -7.0)


# ---------------------------------------------------------------------------
# Szintetikus jelenet
# ---------------------------------------------------------------------------

def make_texture(seed=0):
    rng = np.random.default_rng(seed)
    tw = int((pt.PITCH_LENGTH + 2 * MARGIN) * PPM)
    th = int((pt.PITCH_WIDTH + 2 * MARGIN) * PPM)
    tex = np.zeros((th, tw, 3), np.float32)
    xs = (np.arange(tw) / PPM) - MARGIN
    ys = (np.arange(th) / PPM) - MARGIN
    X, Y = np.meshgrid(xs, ys)
    # Fű: 5.25 m-es nyírási csíkok + több léptékű zaj (textúra a követéshez)
    stripe = ((np.floor(X / 5.25) % 2) * 2 - 1)[..., None]
    base = np.array([55, 135, 60], np.float32)
    tex[:] = base + stripe * np.array([4, 12, 5], np.float32)
    for sigma, amp in ((40, 10), (8, 8), (1.5, 9)):
        n = rng.standard_normal((th, tw)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= n.std() + 1e-6
        tex += (n * amp)[..., None] * np.array([0.5, 1.0, 0.6], np.float32)
    # Lelátó / környezet a pálya körül (4 m-nél távolabb): szürkés, színes foltok
    outside = (X < -4) | (X > pt.PITCH_LENGTH + 4) | (Y < -4) | (Y > pt.PITCH_WIDTH + 4)
    stand = rng.integers(40, 200, size=(th // 8 + 1, tw // 8 + 1, 3)).astype(np.float32)
    stand = cv2.resize(stand, (tw, th), interpolation=cv2.INTER_NEAREST)
    stand = cv2.GaussianBlur(stand, (0, 0), 1.0)
    tex[outside] = stand[outside] * 0.6 + np.array([90, 90, 95], np.float32) * 0.4
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    # Vonalak (12 cm), élsimítva
    thick = max(1, int(round(0.12 * PPM)))
    for poly in pt.pitch_polylines(0.25):
        p = np.round((poly + MARGIN) * PPM * 4).astype(np.int32)
        cv2.polylines(tex, [p], False, (235, 238, 235), thick, cv2.LINE_AA, shift=2)
    return tex


def tex_from_pitch():
    """Pálya méter -> textúra pixel."""
    return np.array([[PPM, 0, MARGIN * PPM], [0, PPM, MARGIN * PPM], [0, 0, 1]], np.float64)


def camera_path(n, w=1280, h=720):
    """Ismert kamerapálya: (pan, tilt, f) képkockánként – ±25° pásztázás, zoom, dőlés."""
    out = []
    for i in range(n):
        t = i / max(1, n - 1)
        pan = -25.0 * math.cos(2 * math.pi * t * 1.0)          # -25 -> +25 -> -25
        tilt = 17.0 + 2.5 * math.sin(2 * math.pi * t * 1.5)
        f = w * (0.80 + 0.12 * math.sin(2 * math.pi * t * 0.75))  # enyhe zoom (hfov ~ 62..75°)
        out.append((pan, tilt, f))
    return out


class Players:
    def __init__(self, n=16, seed=1):
        rng = np.random.default_rng(seed)
        self.pos = np.column_stack([rng.uniform(5, 100, n), rng.uniform(5, 63, n)])
        self.vel = rng.normal(0, 3.0, (n, 2))
        cols = [(40, 40, 220), (220, 80, 40), (235, 235, 235)]   # piros, kék, FEHÉR mez (a vonalszűrőt zavarná)
        self.col = [cols[i % 3] for i in range(n)]

    def step(self, dt=0.04):
        self.pos += self.vel * dt
        for k in range(2):
            lim = pt.PITCH_LENGTH if k == 0 else pt.PITCH_WIDTH
            bad = (self.pos[:, k] < 1) | (self.pos[:, k] > lim - 1)
            self.vel[bad, k] *= -1


def render_frame(tex, pan, tilt, f, players=None, w=1280, h=720, rng=None):
    G = pt.camera_ground_homography(pan, tilt, f, w, h, CAM)
    A = G @ np.linalg.inv(tex_from_pitch())
    img = cv2.warpPerspective(tex, A, (w, h), flags=cv2.INTER_LINEAR, borderValue=(110, 110, 115))
    # Ég a horizont fölött (ahol a képpont nem a talajsíkra vetül)
    Hg = np.linalg.inv(G)
    uu, vv = _grid(w, h)
    wz = Hg[2, 0] * uu + Hg[2, 1] * vv + Hg[2, 2]
    # a talajon lévő pontoknál a visszavetítés iránya egyezik a kamera előtti iránnyal
    ref = Hg[2] @ np.array([w / 2, h - 1, 1.0])
    img[(wz * np.sign(ref)) <= 1e-9] = (200, 170, 140)
    boxes = []
    if players is not None:
        # Teljes kamera (talppont és fej) a doboz méretéhez
        for (x, y), col in zip(players.pos, players.col):
            foot, wf = pt.project_points(G, np.array([[x, y]]))
            if wf[0] <= 0:
                continue
            # 1.8 m magas alak: fej pontja a z = -1.8 síkon (a világ z lefelé mutat)
            head = _project3d(pan, tilt, f, w, h, (x, y, -1.8))
            if head is None:
                continue
            fx, fy = foot[0]
            hh = fy - head[1]
            if hh < 3 or not (-50 < fx < w + 50 and -50 < fy < h + 200):
                continue
            ww = hh * 0.35
            cv2.ellipse(img, (int(fx), int(fy - hh / 2)), (max(1, int(ww / 2)), max(1, int(hh / 2))), 0, 0, 360, col, -1)
            boxes.append((int(fx - ww / 2), int(fy - hh), int(fx + ww / 2), int(fy)))
    if rng is not None:
        img = np.clip(img.astype(np.int16) + _noise(img.shape, rng), 0, 255).astype(np.uint8)
    return img, np.linalg.inv(G), boxes


_GRID = {}
_NOISE = {}


def _grid(w, h):
    if (w, h) not in _GRID:
        vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
        _GRID[(w, h)] = (uu, vv)
    return _GRID[(w, h)]


def _noise(shape, rng):
    """Szenzorzaj (σ = 3): néhány előre generált zajkép közül választ (gyorsabb)."""
    if shape not in _NOISE:
        g = np.random.default_rng(99)
        _NOISE[shape] = [g.normal(0, 3, shape).astype(np.int16) for _ in range(6)]
    return _NOISE[shape][int(rng.integers(0, 6))]


def _project3d(pan, tilt, f, w, h, Xw):
    p, t = math.radians(pan), math.radians(tilt)
    fwd = np.array([math.sin(p) * math.cos(t), -math.cos(p) * math.cos(t), math.sin(t)])
    right = np.cross(fwd, [0, 0, -1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    Xc = R @ (np.asarray(Xw, float) - np.asarray(CAM, float))
    if Xc[2] <= 0:
        return None
    return np.array([f * Xc[0] / Xc[2] + w / 2, f * Xc[1] / Xc[2] + h / 2])


def reprojection_error_m(H_est, H_true, w=1280, h=720, max_dist=None):
    """Átlagos és medián hiba méterben a képen látható pályaterület rácspontjain."""
    if H_est is None:
        return float("inf"), float("inf")
    uu, vv = np.meshgrid(np.arange(20, w, 40), np.arange(20, h, 40))
    p = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)

    def apply(H):
        X = H[:, :2] @ p.T + H[:, 2:3]
        return (X[:2] / X[2]).T, X[2]

    t, wt = apply(H_true)
    ref = (H_true @ np.array([w / 2, h - 1, 1.0]))[2]
    ok = (wt * np.sign(ref) > 0) & (t[:, 0] >= 0) & (t[:, 0] <= pt.PITCH_LENGTH) & \
         (t[:, 1] >= 0) & (t[:, 1] <= pt.PITCH_WIDTH)
    if max_dist is not None:
        ok &= np.hypot(t[:, 0] - CAM[0], t[:, 1] - CAM[1]) <= max_dist
    if not ok.any():
        return float("nan"), float("nan")
    e, _ = apply(H_est)
    d = np.linalg.norm(e[ok] - t[ok], axis=1)
    return float(d.mean()), float(np.median(d))


def perturb_H(H_true, rng, px=4.0, w=1280, h=720):
    """Kézi kalibráció pontatlansága: 4 képpont ±px hibával."""
    img = np.array([[w * 0.2, h * 0.5], [w * 0.8, h * 0.5], [w * 0.15, h * 0.9], [w * 0.85, h * 0.9]])
    X = H_true[:, :2] @ img.T + H_true[:, 2:3]
    pitch = (X[:2] / X[2]).T
    noisy = img + rng.uniform(-px, px, img.shape)
    return pt.homography_from_points(noisy, pitch)


# ---------------------------------------------------------------------------
# Tesztek
# ---------------------------------------------------------------------------

def test_composition_direction():
    """H_t = H_{t-1} · inv(F_t), ahol F_t az előző kép pixeleit viszi az aktuálisra."""
    w, h = 1280, 720
    G0 = pt.camera_ground_homography(-10, 15, 1000, w, h, CAM)
    G1 = pt.camera_ground_homography(-8, 15.5, 1030, w, h, CAM)
    H0, H1 = np.linalg.inv(G0), np.linalg.inv(G1)
    F = G1 @ np.linalg.inv(G0)            # x_1 = F x_0
    H1_est = H0 @ np.linalg.inv(F)
    a = H1_est / H1_est[2, 2]
    b = H1 / H1[2, 2]
    assert np.allclose(a, b, atol=1e-9 * np.abs(b).max()), (a, b)
    # Belső tárolás: G_t = F_t · G_{t-1}
    assert np.allclose(pt.orient_G(F @ G0, w, h), pt.orient_G(G1, w, h), atol=1e-9)


def test_calibration_parsing_and_convention():
    """A régi négysarkos kalibráció és a pálya-pontos változat ugyanazt adja; x=0 a kép bal oldalán."""
    w, h = 1280, 720
    G = pt.camera_ground_homography(-35, 18, 1000, w, h, CAM)
    # A bal alapvonal (x=0) a kép bal felén látszik, ha balra pásztázunk
    uv, wz = pt.project_points(G, np.array([[0.0, 34.0], [30.0, 34.0]]))
    assert wz.min() > 0 and uv[0, 0] < uv[1, 0]
    # Pálya-pontos kalibráció (a 16-os 4 sarka)
    pitch = np.array([[0, 13.84], [16.5, 13.84], [16.5, 54.16], [0, 54.16]])
    uv, _ = pt.project_points(G, pitch)
    s = ";".join(f"{u / w:.6f},{v / h:.6f}" for u, v in uv)
    ps = ";".join(f"{x},{y}" for x, y in pitch)
    H = pt.parse_calib_points(s, w, h, ps)
    err, _ = reprojection_error_m(H, np.linalg.inv(G))
    assert err < 0.05, err
    assert pt.parse_calib_points("", w, h) is None
    assert pt.parse_calib_points("0.1,0.1;0.2,0.2", w, h) is None


def test_orient_sign_invariance():
    w, h = 640, 360
    G = pt.camera_ground_homography(50, 15, 400, w, h, CAM)   # erős pásztázás: a (0,0) sarok a kamera mögött
    assert np.allclose(pt.orient_G(-G, w, h), pt.orient_G(G, w, h))


def _run_sequence(n, refine, H_init, tex, rng_seed=3, w=1280, h=720, timing=False, auto_init=False):
    rng = np.random.default_rng(rng_seed)
    players = Players()
    path = camera_path(n, w, h)
    tracker = pt.PitchTracker((w, h), H0=H_init, refine_every=(10 if refine else 10 ** 9),
                              auto_init=auto_init)
    if not refine:
        tracker.min_conf = -1.0     # csak terjesztés: a bizalom miatt se finomítson
    errs, med, times, lost_frames = [], [], [], 0
    for i, (pan, tilt, f) in enumerate(path):
        players.step()
        frame, H_true, boxes = render_frame(tex, pan, tilt, f, players, w, h, rng)
        t0 = time.perf_counter()
        H_est, q = tracker.update(frame, boxes)
        times.append(time.perf_counter() - t0)
        if H_est is None:
            lost_frames += 1
            H_est = tracker.current_H()
        e, m = reprojection_error_m(H_est, H_true)
        errs.append(e)
        med.append(m)
    return np.array(errs), np.array(med), np.array(times), lost_frames


def test_tracking_with_drift_correction(n=300):
    tex = make_texture()
    w, h = 1280, 720
    pan, tilt, f = camera_path(n, w, h)[0]
    _, H_true0, _ = render_frame(tex, pan, tilt, f, None, w, h)
    H_init = perturb_H(H_true0, np.random.default_rng(7), px=4.0)
    e0, _ = reprojection_error_m(H_init, H_true0)

    t0 = time.perf_counter()
    e_ref, m_ref, times, lost = _run_sequence(n, True, H_init, tex)
    t_ref = time.perf_counter() - t0
    e_prop, m_prop, _, _ = _run_sequence(n, False, H_init, tex)
    print(f"  kezdeti (kézi, ±4 px) hiba: {e0:.2f} m")
    print(f"  csak terjesztés:     átlag {np.mean(e_prop):.2f} m, max {np.max(e_prop):.2f} m, "
          f"utolsó 50 átlag {np.mean(e_prop[-50:]):.2f} m")
    print(f"  vonal-finomítással:  átlag {np.mean(e_ref):.2f} m, max {np.max(e_ref):.2f} m, "
          f"medián {np.median(m_ref):.2f} m, utolsó 50 átlag {np.mean(e_ref[-50:]):.2f} m, elveszett: {lost}")
    print(f"  idő/képkocka (1280x720): átlag {times.mean() * 1000:.1f} ms, "
          f"p95 {np.percentile(times, 95) * 1000:.1f} ms (teljes futás renderrel {t_ref:.1f} s)")
    assert lost == 0, lost
    assert np.mean(e_ref) < 1.0, np.mean(e_ref)
    assert np.mean(e_ref) < np.mean(e_prop), (np.mean(e_ref), np.mean(e_prop))
    assert np.mean(e_ref[-50:]) < 0.8 * np.mean(e_prop[-50:]) or np.mean(e_prop[-50:]) < 0.3


def fast_camera_path(n, w=1280):
    """
    Nehéz (Veo-szerű, gyors) kamerapálya: két gyors pásztázás ±45°-ig (csúcson ~2°/képkocka),
    közben erős zoom (hfov ~ 70° -> 40°) és dőlésváltozás – a régi követő ezen elsodródott.
    """
    out = []
    for i in range(n):
        t = i / max(1, n - 1)
        # sima "lépcsős" pásztázás: -10° -> +45° -> -45° -> 0°
        def ease(a, b, t0, t1):
            s = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
            return a + (b - a) * (3 * s * s - 2 * s * s * s)
        pan = ease(-10, 45, 0.05, 0.3) + ease(0, -90, 0.4, 0.7) + ease(0, 45, 0.8, 0.95)
        zoom = 1.0 + 0.9 * math.sin(math.pi * min(1.0, max(0.0, (t - 0.25) / 0.5))) ** 2
        f = 0.85 * w * zoom
        tilt = 13.0 + 3.0 * math.sin(2 * math.pi * t * 1.3) + 0.25 * (zoom - 1.0) * 4.0
        out.append((pan, tilt, f))
    return out


def landmark_H(H_true, rng, sigma=1.5, w=1280, h=720, max_points=6):
    """Valószerű kézi kalibráció: a képen látható pályapontok (16-os sarkai, kezdőkör, felezővonal ...) zajjal."""
    marks = [(52.5, 0.0), (52.5, 68.0), (52.5, 24.85), (52.5, 43.15), (43.35, 34.0), (61.65, 34.0),
             (16.5, 13.84), (16.5, 54.16), (0.0, 13.84), (0.0, 54.16), (88.5, 13.84), (88.5, 54.16),
             (105.0, 13.84), (105.0, 54.16), (0.0, 0.0), (105.0, 0.0), (0.0, 68.0), (105.0, 68.0)]
    G = pt.orient_G(np.linalg.inv(H_true), w, h)
    uv, wz = pt.project_points(G, np.array(marks))
    ok = (wz > 0) & (uv[:, 0] > 12) & (uv[:, 0] < w - 12) & (uv[:, 1] > 12) & (uv[:, 1] < h - 12)
    idx = np.nonzero(ok)[0][:max_points]
    assert len(idx) >= 4, idx
    img = uv[idx] + rng.normal(0, sigma, (len(idx), 2))
    return pt.homography_from_points(img, np.array(marks)[idx])


def _run_fast(path, H_init, tex, seed=3, w=1280, h=720):
    rng = np.random.default_rng(seed)
    players = Players(n=20, seed=seed)
    tracker = pt.PitchTracker((w, h), H0=H_init, auto_init=False)
    errs, status = [], []
    for pan, tilt, f in path:
        players.step()
        frame, H_true, boxes = render_frame(tex, pan, tilt, f, players, w, h, rng)
        _, q = tracker.update(frame, boxes)
        errs.append(reprojection_error_m(tracker.current_H(), H_true)[1])
        status.append(q["status"])
    return np.array(errs), status


def test_fast_pan_zoom_regression(n=240):
    """
    Gyors pásztázás + erős zoom, valószerű kézi kalibrációval (látható pontok, 1,5 px zaj):
    a hiba minden szakaszban kicsi marad, és a nagy hibájú képkockákat a követő nem jelzi 'ok'-nak.
    (A régi, 8 paraméteres homográfia-követő itt 5–30 m-re sodródott, végig 'ok' státusszal.)
    """
    tex = make_texture(2)
    w, h = 1280, 720
    path = fast_camera_path(n, w)
    pan, tilt, f = path[0]
    _, H_true0, _ = render_frame(tex, pan, tilt, f, None, w, h)
    H_init = landmark_H(H_true0, np.random.default_rng(1))
    errs, status = _run_fast(path, H_init, tex)
    seg = [float(np.median(errs[k * n // 5:(k + 1) * n // 5])) for k in range(5)]
    ok = np.array([s == pt.STATUS_OK for s in status])
    bad = errs > 5.0
    print(f"  szakasz-mediánok (m): {', '.join(f'{v:.2f}' for v in seg)}; 'ok': {ok.mean() * 100:.0f}%, "
          f"> 5 m: {bad.sum()} (ebből 'ok': {(bad & ok).sum()})")
    assert max(seg) < 1.5, seg
    assert (bad & ok).sum() <= max(2, 0.2 * bad.sum()), (bad.sum(), (bad & ok).sum())
    assert ok.mean() > 0.7, ok.mean()


def test_wrong_calibration_is_flagged_or_fixed(n=90):
    """
    Rossz kézi kalibráció (a pálya 8 m-rel eltolva): a követő vagy kijavítja (vonalak alapján
    újrapozicionál), vagy nem jelzi 'ok'-nak – rossz homográfiát nem ad vissza csendben.
    """
    tex = make_texture(3)
    w, h = 1280, 720
    path = fast_camera_path(n * 2, w)[:n]
    pan, tilt, f = path[0]
    _, H_true0, _ = render_frame(tex, pan, tilt, f, None, w, h)
    T = np.array([[1, 0, 8.0], [0, 1, 0], [0, 0, 1]])        # 8 m eltolás a pálya hossza mentén
    errs, status = _run_fast(path, T @ H_true0, tex, seed=5)
    ok = np.array([s == pt.STATUS_OK for s in status])
    bad = errs > 3.0
    print(f"  rossz kalibráció: 'ok' és > 3 m hibás: {(ok & bad).sum()} / {n}, utolsó 30 medián hiba "
          f"{np.median(errs[-30:]):.2f} m, utolsó státusz: {status[-1]}")
    assert (ok & bad).sum() <= 0.1 * n, (ok & bad).sum()


def test_backpropagate():
    """Kalibráció a klip közepén: visszafelé terjesztés a klip elejére."""
    tex = make_texture()
    w, h = 1280, 720
    rng = np.random.default_rng(5)
    path = camera_path(300, w, h)[:40]
    frames, Hs = [], []
    for pan, tilt, f in path:
        fr, Ht, _ = render_frame(tex, pan, tilt, f, None, w, h, rng)
        frames.append(fr)
        Hs.append(Ht)
    tracker = pt.PitchTracker((w, h), auto_init=False)
    H0, fails = tracker.backpropagate(frames, Hs[-1])
    err, _ = reprojection_error_m(H0, Hs[0])
    print(f"  visszafelé terjesztés 40 képkockán: {err:.2f} m hiba, sikertelen lépés: {fails}")
    assert fails == 0 and err < 1.0, (err, fails)


def test_auto_calibration():
    """Automatikus kalibráció kalibrációs pontok nélkül, néhány nézetre."""
    tex = make_texture()
    w, h = 1280, 720
    rng = np.random.default_rng(11)
    views = [(-30.0, 18.0, 0.80 * w), (28.0, 16.0, 0.85 * w), (-22.0, 20.0, 0.75 * w), (0.0, 16.0, 0.9 * w)]
    ok_count = 0
    for pan, tilt, f in views:
        frame, H_true, boxes = render_frame(tex, pan, tilt, f, Players(seed=2), w, h, rng)
        tracker = pt.PitchTracker((w, h), auto_init=True)
        t0 = time.perf_counter()
        H, q = tracker.update(frame, boxes)
        dt = time.perf_counter() - t0
        err, med = reprojection_error_m(H, H_true) if H is not None else (float("inf"), float("inf"))
        ai = q.get("auto_init", {})
        print(f"  nézet pan={pan:+.0f}° tilt={tilt:.0f}°: {q['status']}, hiba {err:.2f} m (medián {med:.2f}), "
              f"pontosság {ai.get('precision')}, lefedettség {ai.get('recall')}, {dt * 1000:.0f} ms")
        if H is not None:
            # Ha elfogadta, legyen jó (rossz kalibrációt ne adjon vissza)
            assert err < 4.0, err
            ok_count += 1
    assert ok_count >= 2, ok_count


def test_auto_init_then_tracking(n=120):
    """Kalibráció nélkül: automatikus indulás, majd követés vonal-finomítással."""
    tex = make_texture()
    e, m, _, lost = _run_sequence(n, True, None, tex, rng_seed=4, auto_init=True)
    ok = np.isfinite(e)
    first = int(np.argmax(ok)) if ok.any() else -1
    print(f"  automatikus kalibráció az {first}. képkockán; hiba utána: átlag {np.mean(e[ok]):.2f} m, "
          f"utolsó 50 átlag {np.mean(e[-50:]):.2f} m, H nélküli képkocka: {lost}")
    assert 0 <= first <= 50, first
    assert np.mean(e[-50:]) < 1.0, np.mean(e[-50:])


def test_lost_and_recover():
    """Használhatatlan képek (pl. takarás) alatt 'elveszett', utána a vonalakból visszatalál."""
    tex = make_texture()
    w, h = 1280, 720
    pan, tilt, f = -10.0, 17.0, 1050.0
    frame, H_true, _ = render_frame(tex, pan, tilt, f, None, w, h)
    tracker = pt.PitchTracker((w, h), H0=H_true, auto_init=False)
    for _ in range(3):
        H, q = tracker.update(frame, [])
    assert q["status"] == pt.STATUS_OK
    gray = np.full_like(frame, 128)
    for _ in range(tracker.max_lost + 2):
        H, q = tracker.update(gray, [])
    assert H is None and q["status"] == pt.STATUS_LOST, q
    for _ in range(3):
        H, q = tracker.update(frame, [])
    err, _ = reprojection_error_m(H, H_true)
    print(f"  visszatalálás után: {q['status']}, hiba {err:.2f} m")
    assert q["status"] == pt.STATUS_OK and err < 1.0, (q, err)


def test_auto_calibration_rejects_empty_grass():
    """Vonal nélküli kép (csak fű): ne találjon ki kalibrációt."""
    w, h = 1280, 720
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (55, 135, 60)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    tracker = pt.PitchTracker((w, h), auto_init=True)
    H, q = tracker.update(img, [])
    assert H is None and q["status"] == pt.STATUS_NEED_CALIB


def test_frame_motion_maps_prev_to_current():
    """frame_motion(): az ELŐZŐ kép pixelét az AKTUÁLIS kép pixelébe viszi (a labdakövető kamera-kompenzációja)."""
    tex = make_texture(4)
    w, h = 1280, 720
    tracker = pt.PitchTracker((w, h), auto_init=False)
    assert tracker.frame_motion() is None
    poses = [(-10.0, 17.0, 1000.0), (-9.4, 17.1, 1004.0)]
    Gs = []
    for pan, tilt, f in poses:
        frame, H_true, _ = render_frame(tex, pan, tilt, f, None, w, h, np.random.default_rng(1))
        tracker.update(frame)
        Gs.append(np.linalg.inv(H_true))
    F = tracker.frame_motion()
    assert F is not None
    F_true = Gs[1] @ np.linalg.inv(Gs[0])
    pts = np.array([[[300.0, 500.0], [900.0, 450.0], [640.0, 650.0]]])
    d = np.linalg.norm(cv2.perspectiveTransform(pts, F)[0] - cv2.perspectiveTransform(pts, F_true)[0], axis=1)
    assert d.max() < 1.5, d


def test_overlay_draws_inside_frame():
    w, h = 1280, 720
    G = pt.camera_ground_homography(-20, 17, 1000, w, h, CAM)
    frame = np.zeros((h, w, 3), np.uint8)
    pt.draw_pitch_overlay(frame, np.linalg.inv(G))
    assert np.count_nonzero(frame) > 1000
    pt.draw_pitch_overlay(frame, None)   # nincs H: nem rajzol, nem hibázik


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    t_all = time.perf_counter()
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"OK   {name} ({time.perf_counter() - t0:.1f} s)")
        except AssertionError as ex:
            failed += 1
            print(f"HIBA {name}: {ex!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} teszt sikeres, {time.perf_counter() - t_all:.1f} s")
    sys.exit(1 if failed else 0)

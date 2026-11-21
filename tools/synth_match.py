#!/usr/bin/env python3
"""
Szintetikus focimeccs-videó generátor (tesztekhez, valódi Veo felvétel nélkül).

Mit készít:
  * Forgatókönyv alapú 2D meccs-szimuláció a 105x68 m-es pályán: 22 játékos (két mezszín,
    két eltérő színű kapus), egy fekete mezes bíró, passzok, lövések (kapura / mellé), védés,
    gól, kirúgás, kapus ívelés és rövid kidobás – mind valós (ground truth) időbélyeggel.
  * Megjelenítés: a félpályavonalnál álló, kb. 7 m magas árbocra szerelt perspektív kamera,
    ami késleltetve (Veo-szerűen) követi a labdát, enyhe zoom-változással. A pálya (csíkos fű,
    zajtextúra, fehér FIFA vonalak) homográfiás vetítéssel, a játékosok egyszerű álló figurák
    (mez, nadrág, lábak, fej, árnyék), a labda kicsi, fehér, gyors mozgásnál elmosódott.
    Az oldalvonalon kívül nézők / edzők / cserék (egy részük füvön, egy részük nem füvön),
    néhányan sétálnak.
  * Opcionális bemelegítés, rövid félidei szünet és sípszó-hang (3,2 kHz) a kezdőrúgásnál,
    gólnál, félidő végén és a lefújáskor – így a match_segmenter is végig tesztelhető.

Kimenet:
  * MP4 (H.264, AAC hang) – ffmpeg kell hozzá (PATH-on 'ffmpeg' vagy 'ffmpeg.exe').
  * Ground-truth JSON: képkockánkénti kamera-homográfia (kép -> pálya), a játékosok / labda
    pozíciója pályaméterben és képpontban (befoglaló dobozzal), az események listája,
    valamint az első képkocka kalibrációs pontjai (a /cut-raw calib_points formátumában).

Példa:
    python3 tools/synth_match.py --out synth.mp4 --duration 60 --seed 7
    python3 tools/synth_match.py --out full.mp4 --segments "warmup:20,half1:90,break:30,half2:90"

Csak numpy + OpenCV + ffmpeg (subprocess) kell hozzá.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
import cv2

L, W = 105.0, 68.0
POST_LO, POST_HI = W / 2 - 3.66, W / 2 + 3.66
GOAL_H = 2.44
BALL_R = 0.11
PERSON_H = 1.80

# Kamera: a közeli oldalvonaltól 6 m-re, a félpályavonalnál, 7 m magasan
CAM_POS = np.array([L / 2, W + 6.0, 7.0])

# Textúra-tartomány (méter) és felbontás (pixel / méter)
TEX_X0, TEX_X1, TEX_Y0, TEX_Y1 = -25.0, 130.0, -30.0, 85.0
TEX_PPM = 20.0
GRASS_X0, GRASS_X1, GRASS_Y0, GRASS_Y1 = -4.0, L + 4.0, -3.5, W + 3.0
BOARD_Y = -4.2

# Színek (BGR)
KITS = {
    "team1": {"shirt": (40, 40, 205), "shorts": (235, 235, 235), "socks": (40, 40, 200)},
    "team2": {"shirt": (190, 90, 30), "shorts": (60, 35, 20), "socks": (190, 90, 30)},
    "keeper1": {"shirt": (30, 215, 240), "shorts": (25, 25, 25), "socks": (30, 215, 240)},
    "keeper2": {"shirt": (165, 45, 150), "shorts": (165, 45, 150), "socks": (25, 25, 25)},
    "referee": {"shirt": (22, 22, 22), "shorts": (22, 22, 22), "socks": (22, 22, 22)},
}
SKINS = [(150, 180, 225), (120, 150, 200), (90, 120, 170), (60, 85, 125)]
CASUAL = [(90, 90, 90), (60, 60, 60), (140, 120, 100), (50, 70, 110), (200, 200, 200), (40, 90, 60),
          (100, 60, 130), (30, 30, 120), (160, 140, 60), (20, 20, 20), (0, 140, 255)]
GRASS_A = np.array([52, 142, 62], np.float32)
GRASS_B = np.array([44, 124, 54], np.float32)
TRACK = np.array([62, 84, 150], np.float32)     # salakos / rekortán sáv (nem fű)
PATH = np.array([120, 125, 128], np.float32)    # szürke gyalogút
ROUGH = np.array([50, 110, 95], np.float32)     # gondozatlan fű a távolban
LINE = (228, 232, 228)

# 4-4-2 alapfelállás a jobbra (x=105 felé) támadó csapatnak: (x, y)
FORMATION = [(3.5, 34.0),
             (20.0, 10.0), (18.0, 27.0), (18.0, 41.0), (20.0, 58.0),
             (37.0, 12.0), (35.0, 28.0), (35.0, 40.0), (37.0, 56.0),
             (48.0, 27.0), (49.0, 41.0)]


# ============================================================ kamera
class Camera:
    """Pan/tilt/zoom tűlyuk-kamera. Világ: x a pálya hossza, y a szélessége (a kamera y=74-nél), z felfelé."""

    def __init__(self, width, height, pos=CAM_POS):
        self.w, self.h = width, height
        self.cx, self.cy = width / 2.0, height / 2.0
        self.C = np.asarray(pos, np.float64)
        self.set(0.0, 0.2, width)

    def set(self, pan, tilt, f):
        self.pan, self.tilt, self.f = float(pan), float(tilt), float(f)
        hdir = np.array([math.sin(pan), -math.cos(pan), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        self.fw = math.cos(tilt) * hdir + math.sin(tilt) * down
        self.r = np.array([math.cos(pan), math.sin(pan), 0.0])
        up_cam = math.cos(tilt) * np.array([0.0, 0.0, 1.0]) + math.sin(tilt) * hdir
        self.d = -up_cam
        self.R = np.stack([self.r, self.d, self.fw])           # sorok: kamera x, y, z tengely
        self.K = np.array([[f, 0, self.cx], [0, f, self.cy], [0, 0, 1.0]])

    def aim_at(self, ax, ay, f):
        v = np.array([ax - self.C[0], ay - self.C[1], -self.C[2]])
        pan = math.atan2(v[0], -v[1])
        tilt = math.atan2(self.C[2], math.hypot(v[0], v[1]))
        self.set(pan, tilt, f)

    def project(self, pts):
        """pts: (N,3) világpontok -> (uv (N,2), mélység (N,))"""
        pc = (np.asarray(pts, np.float64) - self.C) @ self.R.T
        z = pc[:, 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.cx + self.f * pc[:, 0] / zs
        v = self.cy + self.f * pc[:, 1] / zs
        return np.stack([u, v], 1), z

    def H_pitch2img(self):
        t = -self.R @ self.C
        M = np.stack([self.R[:, 0], self.R[:, 1], t], 1)
        H = self.K @ M
        return H / H[2, 2]

    def H_img2pitch(self):
        H = np.linalg.inv(self.H_pitch2img())
        return H / H[2, 2]


# ============================================================ pálya textúra
def _smooth_noise(rng, h, w, cell):
    small = rng.standard_normal((max(2, h // cell), max(2, w // cell))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def build_pitch_texture(rng, ppm=TEX_PPM):
    tw = int((TEX_X1 - TEX_X0) * ppm)
    th = int((TEX_Y1 - TEX_Y0) * ppm)
    xs = TEX_X0 + (np.arange(tw) + 0.5) / ppm
    ys = TEX_Y0 + (np.arange(th) + 0.5) / ppm
    X, Y = np.meshgrid(xs, ys)

    tex = np.empty((th, tw, 3), np.float32)
    tex[:] = TRACK
    grass = (X >= GRASS_X0) & (X <= GRASS_X1) & (Y >= GRASS_Y0) & (Y <= GRASS_Y1)
    stripe = (np.floor(X / (L / 18.0)).astype(int) % 2) == 0
    tex[grass & stripe] = GRASS_A
    tex[grass & ~stripe] = GRASS_B
    path = (Y < BOARD_Y - 1.5) & (Y > -11.0)
    tex[path] = PATH
    tex[Y <= -11.0] = ROUGH

    lo = _smooth_noise(rng, th, tw, int(ppm * 6)) * 0.07 + _smooth_noise(rng, th, tw, int(ppm * 1.2)) * 0.05
    hi = rng.standard_normal((th, tw)).astype(np.float32) * 0.05
    tex *= (1.0 + lo + hi)[..., None]
    # kopott fű a kapuk előtt és a középkörben
    for cx_, cy_, rad in ((5.0, 34.0, 7.0), (L - 5.0, 34.0, 7.0), (L / 2, W / 2, 6.0)):
        d = np.hypot(X - cx_, Y - cy_)
        wear = np.clip(1.0 - d / rad, 0, 1) * 0.25 * (0.6 + 0.4 * _smooth_noise(rng, th, tw, int(ppm)))
        tex = tex * (1 - wear[..., None]) + np.array([70, 120, 120], np.float32) * wear[..., None]
    tex = np.clip(tex, 0, 255).astype(np.uint8)

    def P(x, y):
        return (int(round((x - TEX_X0) * ppm * 4)), int(round((y - TEX_Y0) * ppm * 4)))

    th_px = max(1, int(round(0.15 * ppm)))
    sh = 2

    def line(a, b):
        cv2.line(tex, P(*a), P(*b), LINE, th_px, cv2.LINE_AA, sh)

    def arc(c, r, a0, a1):
        cv2.ellipse(tex, P(*c), (int(r * ppm * 4), int(r * ppm * 4)), 0, a0, a1, LINE, th_px, cv2.LINE_AA, sh)

    line((0, 0), (L, 0)); line((L, 0), (L, W)); line((L, W), (0, W)); line((0, W), (0, 0))
    line((L / 2, 0), (L / 2, W))
    arc((L / 2, W / 2), 9.15, 0, 360)
    cv2.circle(tex, P(L / 2, W / 2), int(0.2 * ppm * 4), LINE, -1, cv2.LINE_AA, sh)
    for gx, s in ((0.0, 1), (L, -1)):
        # tizenhatos, ötös, büntetőpont, félkör
        line((gx, W / 2 - 20.16), (gx + s * 16.5, W / 2 - 20.16))
        line((gx + s * 16.5, W / 2 - 20.16), (gx + s * 16.5, W / 2 + 20.16))
        line((gx + s * 16.5, W / 2 + 20.16), (gx, W / 2 + 20.16))
        line((gx, W / 2 - 9.16), (gx + s * 5.5, W / 2 - 9.16))
        line((gx + s * 5.5, W / 2 - 9.16), (gx + s * 5.5, W / 2 + 9.16))
        line((gx + s * 5.5, W / 2 + 9.16), (gx, W / 2 + 9.16))
        cv2.circle(tex, P(gx + s * 11.0, W / 2), int(0.2 * ppm * 4), LINE, -1, cv2.LINE_AA, sh)
        ang = math.degrees(math.acos(5.5 / 9.15))
        if s == 1:
            arc((gx + 11.0, W / 2), 9.15, -ang, ang)
        else:
            arc((gx - 11.0, W / 2), 9.15, 180 - ang, 180 + ang)
    for cx_, cy_, a0 in ((0, 0, 0), (L, 0, 90), (L, W, 180), (0, W, 270)):
        arc((cx_, cy_), 1.0, a0, a0 + 90)
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    M = np.array([[1.0 / ppm, 0, TEX_X0], [0, 1.0 / ppm, TEX_Y0], [0, 0, 1.0]])
    return tex, M


# ============================================================ szimuláció
class Person:
    __slots__ = ("id", "role", "team", "number", "pos", "vel", "max_speed", "base", "kit", "skin",
                 "run_until", "run_target", "wob", "walk", "height")

    def __init__(self, pid, role, team, number, pos, kit, skin, max_speed=7.0, base=None, height=PERSON_H):
        self.id, self.role, self.team, self.number = pid, role, team, number
        self.pos = np.array(pos, np.float64)
        self.vel = np.zeros(2)
        self.max_speed = max_speed
        self.base = np.array(base if base is not None else pos, np.float64)
        self.kit, self.skin = kit, skin
        self.run_until, self.run_target = -1.0, None
        self.wob = None
        self.walk = None
        self.height = height


def _team_key(team):
    return f"team{team}"


class MatchSim:
    """Forgatókönyv-vezérelt meccs-szimuláció. A run() képkockánkénti állapotokat ad vissza."""

    def __init__(self, fps, segments, rng):
        self.fps = fps
        self.dt = 1.0 / fps
        self.rng = rng
        self.segments = segments
        self.events = []
        self.whistles = []
        self.periods = []
        self.passes = {"team1": 0, "team2": 0}
        self.people = []
        self._build_people()
        self.ball = np.array([L / 2, W / 2, 0.0])
        self.ball_vis = True
        self.mode = "dead"
        self.mode_data = {}
        self.t = 0.0
        self.attack_right = {1: True, 2: False}
        self.phase_list = []
        self.phase_idx = 0
        self.phase_passes = 0
        self.formation_mode = "play"
        self.half = 0

    # -------------------------------------------------- szereplők
    def _build_people(self):
        rng = self.rng
        pid = 0
        for team in (1, 2):
            for i, (bx, by) in enumerate(FORMATION):
                keeper = i == 0
                kit = KITS[f"keeper{team}"] if keeper else KITS[f"team{team}"]
                base = (bx, by)
                pos = (bx, by) if team == 1 else (L - bx, by)
                p = Person(pid, "goalkeeper" if keeper else "player", team, 1 if keeper else i + 1,
                           pos, kit, SKINS[rng.integers(len(SKINS))],
                           max_speed=6.0 if keeper else rng.uniform(6.8, 8.2), base=base,
                           height=rng.uniform(1.72, 1.92))
                p.wob = (rng.uniform(0.15, 0.4), rng.uniform(0, 6.28), rng.uniform(0.1, 0.3), rng.uniform(0, 6.28))
                self.people.append(p)
                pid += 1
        ref = Person(pid, "referee", 0, 0, (L / 2 - 8, W / 2 + 6), KITS["referee"], SKINS[0], max_speed=6.5)
        self.people.append(ref)
        self.ref = ref
        pid += 1
        self.n_match = pid
        # nézők, edzők, cserék
        specs = []
        for _ in range(22):   # túlsó oldal, a reklámtáblák mögött (nem fű)
            specs.append(("spectator", (rng.uniform(5, 100), rng.uniform(-9.5, -6.0)), None))
        for _ in range(5):    # túlsó oldal, a fű szélén (cserék, melegítők)
            specs.append(("sub", (rng.uniform(30, 75), rng.uniform(-3.2, -1.6)), None))
        for _ in range(6):    # kapuk mögött (salakon)
            gx = rng.choice([-7.0, L + 7.0])
            specs.append(("spectator", (gx + rng.uniform(-2, 2), rng.uniform(18, 50)), None))
        for x in (44.0, 46.0, 59.0, 61.5):   # közeli oldal, technikai zóna (fű széle)
            specs.append(("coach", (x, W + rng.uniform(1.2, 2.4)), None))
        for _ in range(3):    # sétáló nézők a túlsó oldalon
            x0 = rng.uniform(10, 90)
            specs.append(("spectator", (x0, rng.uniform(-8.5, -7.0)), (x0, rng.uniform(15, 30), rng.choice([-1, 1]))))
        specs.append(("spectator", (rng.uniform(20, 80), -2.6), (rng.uniform(20, 80), 20.0, 1)))  # füvön sétáló
        for role, pos, walk in specs:
            if role == "coach":
                kit = {"shirt": (35, 35, 45), "shorts": (35, 35, 45), "socks": (30, 30, 30)}
            elif role == "sub":
                kit = {"shirt": (0, 150, 255), "shorts": (60, 35, 20), "socks": (40, 40, 40)}
            else:
                c = CASUAL[rng.integers(len(CASUAL))]
                kit = {"shirt": c, "shorts": CASUAL[rng.integers(len(CASUAL))], "socks": (40, 40, 40)}
            p = Person(pid, role, 0, 0, pos, kit, SKINS[rng.integers(len(SKINS))], max_speed=1.4,
                       height=rng.uniform(1.6, 1.9))
            if walk is not None:
                p.walk = {"x0": walk[0], "span": walk[1], "dir": walk[2], "y": pos[1]}
            self.people.append(p)
            pid += 1

    def players(self, team=None):
        return [p for p in self.people[:22] if team is None or p.team == team]

    def keeper(self, team):
        return self.people[0 if team == 1 else 11]

    def attack_goal_x(self, team):
        return L if self.attack_right[team] else 0.0

    def own_goal_x(self, team):
        return 0.0 if self.attack_right[team] else L

    def to_team(self, team, x):
        """Világ x -> a csapat 'jobbra támad' koordinátája."""
        return x if self.attack_right[team] else L - x

    def from_team(self, team, x):
        return x if self.attack_right[team] else L - x

    # -------------------------------------------------- események / sípszó
    def emit(self, typ, team=None, t=None, **kw):
        ev = {"type": typ, "time": round(self.t if t is None else t, 3)}
        if team is not None:
            ev["team"] = _team_key(team)
        ev.update(kw)
        self.events.append(ev)

    def whistle(self, kind, t=None, blasts=1):
        t = self.t if t is None else t
        if blasts == 1:
            self.whistles.append({"time": round(t, 3), "duration": 0.6, "kind": kind})
        else:
            for k, d in zip(range(blasts), (0.6, 0.6, 1.0)):
                self.whistles.append({"time": round(t + 0.9 * k, 3), "duration": d, "kind": kind})

    # -------------------------------------------------- forgatókönyv
    def _half_script(self, kick_team):
        A, B = kick_team, 3 - kick_team
        return [("kickoff", A), ("attack", A, "shot_off"), ("goal_kick", B),
                ("attack", B, "turnover"), ("attack", A, "shot_save"), ("keeper_throw", B),
                ("attack", B, "shot_goal"), ("kickoff", A),
                ("attack", A, "backpass"), ("keeper_long", A), ("attack", A, "turnover"),
                ("attack", B, "shot_save_late"), ("keeper_throw", A),
                ] + [("attack", A if k % 2 == 0 else B, "free") for k in range(40)]

    @property
    def phase(self):
        if self.phase_idx < len(self.phase_list):
            return self.phase_list[self.phase_idx]
        return ("attack", 1, "free")

    def next_phase(self):
        self.phase_idx += 1
        self.phase_passes = 0

    # -------------------------------------------------- labda akciók
    def _release(self, kind, p_from, target, speed, height, receiver=None, outcome=None, aerial=False):
        p0 = np.array(self.ball[:2])
        p1 = np.array(target, np.float64)
        dist = float(np.linalg.norm(p1 - p0))
        T = max(0.25, dist / speed)
        self.mode = "flight"
        self.mode_data = {"kind": kind, "p0": p0, "p1": p1, "t0": self.t, "T": T, "h": height,
                          "z0": float(self.ball[2]), "receiver": receiver, "outcome": outcome,
                          "aerial": aerial, "from": p_from, "crossed": False}
        if receiver is not None:
            receiver.run_target = p1.copy()
            receiver.run_until = self.t + T + 0.3
        return T

    def _pass(self, owner, receiver, kind="pass", speed=None, height=0.0, aerial=False, lead=True):
        rng = self.rng
        rp = receiver.pos + (receiver.vel * 0.6 if lead else 0) + rng.normal(0, 1.0, 2)
        rp = np.clip(rp, [1.0, 1.0], [L - 1.0, W - 1.0])
        d = float(np.linalg.norm(rp - self.ball[:2]))
        if speed is None:
            speed = float(np.clip(6.0 + d * 0.25, 7.0, 14.0)) * rng.uniform(0.9, 1.1)
        if kind == "pass" and receiver.team == owner.team:
            self.passes[_team_key(owner.team)] += 1
        return self._release(kind, owner, rp, speed, height, receiver=receiver, aerial=aerial)

    def _shoot(self, owner, outcome):
        rng = self.rng
        team = owner.team
        gx = self.attack_goal_x(team)
        s = 1.0 if gx == L else -1.0
        keeper = self.keeper(3 - team)
        if outcome == "goal":
            ty = rng.choice([rng.uniform(POST_LO + 0.5, POST_LO + 1.6), rng.uniform(POST_HI - 1.6, POST_HI - 0.5)])
            target = (gx + s * 1.6, ty)
            h = rng.uniform(0.3, 1.2)
        elif outcome == "save":
            ty = float(np.clip(keeper.pos[1] + rng.uniform(-1.5, 1.5), POST_LO + 1.0, POST_HI - 1.0))
            target = (float(np.clip(keeper.pos[0], min(gx, gx - s * 5.0), max(gx, gx - s * 5.0))), ty)
            h = rng.uniform(0.4, 1.0)
        else:  # mellé
            ty = rng.choice([POST_LO - rng.uniform(2.0, 5.0), POST_HI + rng.uniform(2.0, 5.0)])
            target = (gx + s * 3.5, ty)
            h = rng.uniform(0.8, 2.5)
        speed = rng.uniform(22.0, 28.0)
        T = self._release("shot", owner, target, speed, h, outcome=outcome, aerial=True)
        on_target = outcome in ("goal", "save")
        self.emit("shot", team, on_target=bool(on_target), outcome=outcome,
                  distance_m=round(float(np.linalg.norm(np.array(target) - self.ball[:2])), 1))
        if outcome == "save":
            keeper.run_target = np.array(target, np.float64)
            keeper.run_until = self.t + T + 0.2
        elif outcome == "goal":
            # a kapus elkésve vetődik
            keeper.run_target = np.array([gx - s * 1.0, (keeper.pos[1] + ty) / 2.0])
            keeper.run_until = self.t + T + 0.5

    def _flight_dist(self):
        md = self.mode_data
        return round(float(np.linalg.norm(md["p1"] - md["p0"])), 1)

    def _choose_teammate(self, owner, forward_bias=1.0, min_d=7.0, max_d=32.0, exclude_keeper=True, back=False):
        rng = self.rng
        team = owner.team
        ox = self.to_team(team, owner.pos[0])
        cands = []
        for p in self.players(team):
            if p is owner or (exclude_keeper and p.role == "goalkeeper"):
                continue
            d = float(np.linalg.norm(p.pos - owner.pos))
            if d < min_d or d > max_d:
                continue
            fx = self.to_team(team, p.pos[0]) - ox
            w = math.exp((-fx if back else fx) * 0.06 * forward_bias) * (1.0 / (1 + abs(d - 16) / 10))
            cands.append((w, p))
        if not cands:
            others = [p for p in self.players(team) if p is not owner and p.role != "goalkeeper"]
            return min(others, key=lambda p: np.linalg.norm(p.pos - owner.pos))
        ws = np.array([c[0] for c in cands])
        return cands[rng.choice(len(cands), p=ws / ws.sum())][1]

    def _forward_teammate(self, owner, max_d=36.0):
        """Előre passz: a legelőretoltabb elérhető csapattárs (kis véletlennel)."""
        team = owner.team
        ox = self.to_team(team, owner.pos[0])
        cands = [p for p in self.players(team) if p is not owner and p.role != "goalkeeper"
                 and 6.0 < np.linalg.norm(p.pos - owner.pos) < max_d and self.to_team(team, p.pos[0]) > ox]
        if not cands:
            return None
        return max(cands, key=lambda p: self.to_team(team, p.pos[0]) + self.rng.uniform(0, 8))

    def _nearest_opponent(self, owner, ahead=12.0):
        team = owner.team
        s = 1.0 if self.attack_right[team] else -1.0
        pt = owner.pos + np.array([s * ahead, 0.0])
        opps = [p for p in self.players(3 - team) if p.role != "goalkeeper"]
        return min(opps, key=lambda p: np.linalg.norm(p.pos - pt))

    def _owner_action(self, owner):
        """A labdát birtokló játékos döntése a forgatókönyv fázisa szerint."""
        rng = self.rng
        ph = self.phase
        kind = ph[0]
        if kind in ("keeper_long", "keeper_throw") and owner.role == "goalkeeper" and owner.team == ph[1]:
            if kind == "keeper_long":
                fw = [p for p in self.players(owner.team) if p.role != "goalkeeper"]
                far = [p for p in fw if np.linalg.norm(p.pos - owner.pos) >= 42.0] or fw
                rec = max(far, key=lambda p: self.to_team(owner.team, p.pos[0]) + rng.uniform(0, 8))
                self._pass(owner, rec, kind="gk_long", speed=rng.uniform(19, 23), height=rng.uniform(9, 13),
                           aerial=True)
                self.emit("gk_long_ball", owner.team, keeper_id=owner.id, distance_m=self._flight_dist())
            else:
                rec = self._choose_teammate(owner, forward_bias=-0.3, min_d=10.0, max_d=26.0)
                self._pass(owner, rec, kind="gk_throw", speed=rng.uniform(10, 13), height=1.4, aerial=True)
                self.emit("gk_short_pass", owner.team, keeper_id=owner.id, distance_m=self._flight_dist())
            self.next_phase()
            return
        if kind != "attack":
            # váratlan helyzet: egyszerű passz
            self._pass(owner, self._choose_teammate(owner))
            return
        team, goal = ph[1], ph[2]
        if owner.team != team:
            # labdavesztés a fázis csapata javára
            rec = min([p for p in self.players(team) if p.role != "goalkeeper"],
                      key=lambda p: np.linalg.norm(p.pos - owner.pos) + rng.uniform(0, 6))
            self._pass(owner, rec, kind="pass")
            return
        gx = self.attack_goal_x(team)
        dist_goal = math.hypot(owner.pos[0] - gx, owner.pos[1] - W / 2)
        if goal.startswith("shot_"):
            if goal == "shot_save_late" and self.phase_passes < 3:
                self.phase_passes += 1
                self._pass(owner, self._choose_teammate(owner, forward_bias=0.5))
                return
            if dist_goal <= 27.0 and owner.role != "goalkeeper":
                outcome = {"shot_off": "off", "shot_save": "save", "shot_save_late": "save",
                           "shot_goal": "goal"}[goal]
                self._shoot(owner, outcome)
                self.next_phase()
                return
            fwd = self._forward_teammate(owner) if self.phase_passes >= 2 else None
            if dist_goal < 36.0 or (self.phase_passes >= 2 and fwd is None):
                # cselezés kapu felé
                self.mode_data.update({"hold_until": self.t + rng.uniform(1.0, 1.6), "dribble": 6.5})
                return
            self.phase_passes += 1
            self._pass(owner, fwd if fwd is not None else self._choose_teammate(owner, forward_bias=2.5))
            return
        if goal == "turnover":
            if self.phase_passes >= 1:
                self.next_phase()
                opp = self._nearest_opponent(owner)
                self._pass(owner, opp, kind="pass", lead=False)
                return
            self.phase_passes += 1
            self._pass(owner, self._choose_teammate(owner, forward_bias=0.8))
            return
        if goal == "backpass":
            k = self.keeper(team)
            ox = self.to_team(team, owner.pos[0])
            if self.phase_passes >= 1 and ox < 45.0 and owner.role != "goalkeeper":
                self._pass(owner, k, kind="backpass", lead=False)
                self.next_phase()
                return
            self.phase_passes += 1
            self._pass(owner, self._choose_teammate(owner, back=True, forward_bias=1.5))
            return
        # free
        if rng.random() < 0.15 and self.phase_passes >= 2:
            self.next_phase()
            self._pass(owner, self._nearest_opponent(owner), lead=False)
            return
        self.phase_passes += 1
        if self.phase_passes > 6:
            self.next_phase()
        self._pass(owner, self._choose_teammate(owner, forward_bias=0.6))

    def _set_owned(self, owner, hold=None):
        self.mode = "owned"
        hold = self.rng.uniform(0.6, 1.4) if hold is None else hold
        self.mode_data = {"owner": owner, "hold_until": self.t + hold, "dribble": 3.5}
        owner.run_until = -1

    def _set_dead(self, pos, duration, then, visible=True):
        self.mode = "dead"
        self.ball[:2] = pos
        self.ball[2] = 0.0
        self.ball_vis = visible
        self.mode_data = {"until": self.t + duration, "then": then}

    # -------------------------------------------------- labda frissítés
    def _arrive(self):
        md = self.mode_data
        kind, outcome = md["kind"], md["outcome"]
        rec = md["receiver"]
        if kind == "shot":
            team = md["from"].team
            gx = self.attack_goal_x(team)
            s = 1.0 if gx == L else -1.0
            if outcome == "save":
                k = self.keeper(3 - team)
                self.emit("gk_save", 3 - team, keeper_id=k.id)
                self._set_owned(k, hold=2.2)
                self.ball[2] = 1.0
                return
            if outcome == "goal":
                self._set_dead(self.ball[:2].copy(), 2.0, ("carry_center", 3 - team))
                return
            # mellé: alapvonalon túl, kirúgás következik
            gk_spot = (gx - s * 5.0, W / 2 + self.rng.choice([-5.0, 5.0]))
            self._set_dead(self.ball[:2].copy(), 1.5, ("carry_goal_kick", gk_spot))
            return
        if rec is not None:
            self._set_owned(rec)
            return
        self._set_dead(self.ball[:2].copy(), 1.0, None)

    def _update_ball(self):
        md = self.mode_data
        if self.mode == "flight":
            u = (self.t - md["t0"]) / md["T"]
            if u >= 1.0:
                self.ball[:2] = md["p1"]
                self.ball[2] = 0.0 if md["outcome"] != "save" else 1.0
                self._arrive()
                return
            g = u if md["aerial"] else (u - 0.3 * u * u) / 0.7
            self.ball[:2] = md["p0"] + (md["p1"] - md["p0"]) * g
            z = 4.0 * md["h"] * u * (1 - u) + md["z0"] * (1 - u)
            self.ball[2] = z
            if md["kind"] == "shot" and md["outcome"] == "goal" and not md["crossed"]:
                gx = self.attack_goal_x(md["from"].team)
                if (gx == L and self.ball[0] >= L) or (gx == 0.0 and self.ball[0] <= 0.0):
                    md["crossed"] = True
                    team = md["from"].team
                    k = self.keeper(3 - team)
                    self.emit("goal", team, goal_x=gx)
                    self.emit("gk_conceded", 3 - team, keeper_id=k.id, goal_x=gx)
                    self.whistle("goal", t=self.t + 0.8)
        elif self.mode == "owned":
            o = md["owner"]
            if o.role == "goalkeeper" and self.ball[2] > 0.5:
                self.ball[:2] = o.pos + np.array([0.0, 0.25])
                self.ball[2] = 1.0
            else:
                v = o.vel
                sp = float(np.linalg.norm(v))
                dirv = v / sp if sp > 0.3 else np.array([1.0 if self.attack_right.get(o.team, True) else -1.0, 0])
                foot = o.pos + dirv * 0.55
                self.ball[:2] += (foot - self.ball[:2]) * 0.35
                self.ball[2] = 0.0
            if self.t >= md["hold_until"]:
                self._owner_action(o)
        elif self.mode == "dead":
            if md.get("until") is not None and self.t >= md["until"]:
                self._restart(md.get("then"))

    def _restart(self, then):
        rng = self.rng
        if then is None:
            self.mode_data = {"until": None}
            return
        kind = then[0]
        if kind == "carry_center":
            self.ball_vis = False
            self._set_dead((L / 2, W / 2), 4.0, ("kickoff_ready",), visible=False)
            self.formation_mode = "kickoff"
            return
        if kind == "kickoff_ready":
            self.ball_vis = True
            self._set_dead((L / 2, W / 2), 2.5, ("kickoff_go",))
            return
        if kind == "kickoff_go":
            ph = self.phase
            team = ph[1] if ph[0] == "kickoff" else 1
            self.whistle("kickoff")
            mids = [p for p in self.players(team) if p.role != "goalkeeper"]
            back = [p for p in mids if 6.0 < np.linalg.norm(p.pos - np.array([L / 2, W / 2])) < 22.0
                    and self.to_team(team, p.pos[0]) < L / 2]
            rec = back[rng.integers(len(back))] if back else mids[5]
            kicker = min(mids, key=lambda p: np.linalg.norm(p.pos - np.array([L / 2, W / 2])))
            self.formation_mode = "play"
            if ph[0] == "kickoff":
                self.next_phase()
            self._pass(kicker, rec, lead=False)
            return
        if kind == "carry_goal_kick":
            self.ball_vis = False
            self._set_dead(then[1], 2.5, ("goal_kick_ready",), visible=False)
            self.formation_mode = "goal_kick"
            return
        if kind == "goal_kick_ready":
            self.ball_vis = True
            self._set_dead(self.ball[:2].copy(), 3.0, ("goal_kick_go",))
            return
        if kind == "goal_kick_go":
            ph = self.phase
            team = ph[1] if ph[0] == "goal_kick" else (1 if self.ball[0] < L / 2 else 2)
            k = self.keeper(team)
            self.formation_mode = "play"
            fw = [p for p in self.players(team) if p.role != "goalkeeper"]
            far = [p for p in fw if np.linalg.norm(p.pos - self.ball[:2]) >= 42.0] or fw
            rec = max(far, key=lambda p: self.to_team(team, p.pos[0]) + rng.uniform(-10, 10) - abs(p.pos[1] - W / 2) * 0.2)
            if ph[0] == "goal_kick":
                self.next_phase()
            self._pass(k, rec, kind="goal_kick", speed=rng.uniform(20, 24), height=rng.uniform(10, 14), aerial=True)
            d = self._flight_dist()
            self.emit("gk_goal_kick", team, keeper_id=k.id, is_long=d > 35.0, distance_m=d)
            return

    # -------------------------------------------------- játékos mozgás
    def _targets(self, seg_kind, seg_t, seg_len):
        rng = self.rng
        t = self.t
        bx, by = float(self.ball[0]), float(self.ball[1])
        owner = self.mode_data.get("owner") if self.mode == "owned" else None
        poss_team = owner.team if owner is not None else (
            self.mode_data["from"].team if self.mode == "flight" else None)
        targets = {}
        speeds = {}
        if seg_kind == "warmup":
            for p in self.players():
                if p.run_until < t or p.run_target is None:
                    half_lo = 5.0 if self.to_team(p.team, 10) < L / 2 else L / 2 + 3
                    xr = rng.uniform(8, 45)
                    p.run_target = np.array([self.from_team(p.team, xr), rng.uniform(8, 60)])
                    p.run_until = t + rng.uniform(3, 7)
                targets[p.id] = p.run_target
                speeds[p.id] = 3.0
            targets[self.ref.id] = np.array([L / 2, W + 1.5])
            speeds[self.ref.id] = 1.5
            return targets, speeds
        if seg_kind == "break":
            frac = seg_t / max(seg_len, 1e-6)
            for p in self.players() + [self.ref]:
                if frac < 0.7:
                    bench_x = 44.0 if p.team == 1 else (61.0 if p.team == 2 else L / 2)
                    targets[p.id] = np.array([bench_x + (p.id % 6) * 0.9 - 2.5, W + 1.0 + (p.id % 3) * 0.7])
                else:
                    targets[p.id] = self._kickoff_pos(p)
                speeds[p.id] = 2.8 if frac < 0.7 else 6.0
            return targets, speeds

        ball_team_x = {1: self.to_team(1, bx), 2: self.to_team(2, bx)}
        for p in self.players():
            team = p.team
            if self.formation_mode == "kickoff":
                targets[p.id] = self._kickoff_pos(p)
                speeds[p.id] = 4.0
                continue
            if p.role == "goalkeeper":
                gx = self.own_goal_x(team)
                s = 1.0 if gx == 0.0 else -1.0
                adv = float(np.clip((L - abs(bx - gx)) / L * 12.0 - 4.0, 0.5, 6.0))
                tgt = np.array([gx + s * adv, W / 2 + float(np.clip((by - W / 2) * 0.22, -3.0, 3.0))])
                if self.formation_mode == "goal_kick" and abs(bx - gx) < 12.0 and self.mode == "dead":
                    tgt = self.ball[:2] + np.array([-s * 1.2, 0.3])
                targets[p.id] = tgt
                speeds[p.id] = 4.5
            else:
                base = p.base
                shift = 0.0 if self.formation_mode == "goal_kick" else 0.48 * (ball_team_x[team] - 45.0)
                if poss_team == team:
                    shift += 6.0
                elif poss_team is not None:
                    shift -= 3.0
                xt = float(np.clip(base[0] + shift, 8.0, 92.0))
                if self.formation_mode == "goal_kick":
                    xt = float(np.clip(xt, 18.0, 88.0))
                yt = base[1] + (by - W / 2) * 0.35
                a1, ph1, a2, ph2 = p.wob
                xt += 2.5 * math.sin(a1 * t + ph1)
                yt += 2.5 * math.sin(a2 * t + ph2)
                tgt = np.array([self.from_team(team, xt), float(np.clip(yt, 2.0, W - 2.0))])
                speeds[p.id] = 4.0
                # kiugrás (befutás) a támadó csapatban
                if poss_team == team and p.number >= 9 and rng.random() < 0.004:
                    p.run_target = np.array([self.from_team(team, min(95.0, xt + 12.0)), tgt[1] + rng.uniform(-8, 8)])
                    p.run_until = t + 2.5
                targets[p.id] = tgt
            if p.run_until >= t and p.run_target is not None:
                targets[p.id] = p.run_target
                speeds[p.id] = p.max_speed
        # letámadás: a labdához legközelebbi ellenfél
        if owner is not None and owner.role != "goalkeeper":
            opps = [p for p in self.players(3 - owner.team) if p.role != "goalkeeper"]
            opps.sort(key=lambda p: np.linalg.norm(p.pos - self.ball[:2]))
            gx = self.own_goal_x(3 - owner.team)
            for k, p in enumerate(opps[:2]):
                towards = np.array([gx, W / 2]) - self.ball[:2]
                towards /= max(1e-6, np.linalg.norm(towards))
                targets[p.id] = self.ball[:2] + towards * (2.2 + 4.0 * k)
                speeds[p.id] = 5.5
        if owner is not None:
            md = self.mode_data
            if owner.role == "goalkeeper":
                targets[owner.id] = owner.pos.copy()
                speeds[owner.id] = 0.5
            else:
                gx = self.attack_goal_x(owner.team)
                dirv = np.array([gx, W / 2 + (owner.id % 5 - 2) * 3.0]) - owner.pos
                dirv /= max(1e-6, np.linalg.norm(dirv))
                targets[owner.id] = owner.pos + dirv * 6.0
                speeds[owner.id] = md.get("dribble", 3.5)
        # bíró: átlósan követi a labdát
        targets[self.ref.id] = np.array([float(np.clip(bx - 9.0, 15, 90)), float(np.clip(by + 9.0, 8, 62))])
        speeds[self.ref.id] = 5.0
        return targets, speeds

    def _kickoff_pos(self, p):
        if p.role == "referee":
            return np.array([L / 2 - 6.0, W / 2 + 10.0])
        bx, by = p.base
        if p.role == "goalkeeper":
            xt = 4.0
        else:
            xt = min(bx * 0.95, L / 2 - 1.5)
            kicking = self.phase[0] == "kickoff" and self.phase[1] == p.team
            if kicking and p.number in (10, 11):
                xt, by = L / 2 - 0.8, W / 2 + (1.0 if p.number == 10 else -6.0)
            elif math.hypot(xt - L / 2, by - W / 2) < 9.5:
                xt = L / 2 - 10.0
        return np.array([self.from_team(p.team, xt), by])

    def _move(self, targets, speeds):
        dt = self.dt
        movers = [p for p in self.people[:self.n_match]]
        for p in movers:
            tgt = targets.get(p.id)
            if tgt is None:
                continue
            delta = tgt - p.pos
            d = float(np.linalg.norm(delta))
            sp = min(speeds.get(p.id, 4.0), p.max_speed, d * 1.6)
            desired = delta / d * sp if d > 1e-6 else np.zeros(2)
            dv = desired - p.vel
            amax = 4.5 * dt
            nv = float(np.linalg.norm(dv))
            if nv > amax:
                dv *= amax / nv
            p.vel += dv
        # egyszerű taszítás (ne fedjék egymást teljesen)
        pos = np.array([p.pos for p in movers])
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + np.eye(len(movers)) * 99
        close = dist < 1.1
        if close.any():
            push = (diff / dist[..., None] * ((1.1 - dist) * close)[..., None]).sum(1)
            for i, p in enumerate(movers):
                p.vel += push[i] * 1.5 * dt * 10
        for p in movers:
            p.pos += p.vel * dt
            p.pos[0] = float(np.clip(p.pos[0], -3.0, L + 3.0))
            p.pos[1] = float(np.clip(p.pos[1], -3.0, W + 3.0))
        # nézők
        for p in self.people[self.n_match:]:
            if p.walk is None:
                p.vel[:] = 0
                continue
            wk = p.walk
            x = p.pos[0] + wk["dir"] * 1.2 * dt
            if abs(x - wk["x0"]) > wk["span"]:
                wk["dir"] *= -1
            p.vel = np.array([wk["dir"] * 1.2, 0.0])
            p.pos = np.array([x, wk["y"]])

    # -------------------------------------------------- futtatás
    def run(self):
        frames = []
        t_seg0 = 0.0
        for name, seg_len in self.segments:
            n = int(round(seg_len * self.fps))
            kind = "half" if name.startswith("half") else name
            if kind == "half":
                self.half += 1
                self.attack_right = {1: self.half % 2 == 1, 2: self.half % 2 == 0}
                kick_team = 1 if self.half % 2 == 1 else 2
                self.phase_list = self._half_script(kick_team)
                self.phase_idx = 0
                self.phase_passes = 0
                self.formation_mode = "kickoff"
                for p in self.players():
                    p.pos = self._kickoff_pos(p) + self.rng.normal(0, 0.3, 2)
                    p.vel[:] = 0
                    p.run_until = -1
                self.ref.pos = self._kickoff_pos(self.ref)
                self.ball_vis = True
                self._set_dead((L / 2, W / 2), 1.2, ("kickoff_go",))
                self.periods.append({"half": self.half, "start": round(t_seg0 + 1.2, 3),
                                     "end": round(t_seg0 + seg_len, 3),
                                     "team1_attacks_right": self.attack_right[1]})
            elif kind == "warmup":
                self.formation_mode = "play"
                self._set_dead((L / 2, W / 2), 0.0, None)
                self.mode_data = {"until": None}
            elif kind == "break":
                self.mode = "dead"
                self.mode_data = {"until": None}
                self.ball[:] = (L / 2, W / 2, 0.0)
            for i in range(n):
                self.t = t_seg0 + i * self.dt
                seg_t = i * self.dt
                if kind == "half" and seg_t >= seg_len - 2.5 and self.mode != "dead_end":
                    # lefújás / félidő vége: a labda megáll, 3 sípszó
                    self.whistle("half_end" if self.half % 2 == 1 else "full_time", blasts=3)
                    self.periods[-1]["end"] = round(self.t, 3)
                    self.mode = "dead_end"
                    self.mode_data = {}
                    self.ball[2] = 0.0
                if kind == "warmup":
                    self._warmup_ball()
                elif kind == "half" and self.mode != "dead_end":
                    self._update_ball()
                targets, speeds = self._targets("half" if kind == "half" else kind, seg_t, seg_len)
                self._move(targets, speeds)
                frames.append({
                    "t": self.t,
                    "pos": np.array([p.pos for p in self.people], np.float32),
                    "vel": np.array([p.vel for p in self.people], np.float32),
                    "ball": self.ball.copy(),
                    "ball_vis": bool(self.ball_vis),
                    "seg": name,
                })
            t_seg0 += n * self.dt
        return frames

    def _warmup_ball(self):
        """Bemelegítés: az 1. csapat játékosai passzolgatnak (nem számít eseménynek)."""
        rng = self.rng
        if self.mode == "flight":
            md = self.mode_data
            u = (self.t - md["t0"]) / md["T"]
            if u >= 1:
                self.mode = "owned"
                self.mode_data = {"owner": md["receiver"], "hold_until": self.t + rng.uniform(1, 2)}
            else:
                g = (u - 0.3 * u * u) / 0.7
                self.ball[:2] = md["p0"] + (md["p1"] - md["p0"]) * g
            return
        if self.mode == "owned":
            o = self.mode_data["owner"]
            self.ball[:2] += (o.pos - self.ball[:2]) * 0.3
            if self.t >= self.mode_data["hold_until"]:
                rec = self._choose_teammate(o, forward_bias=0.0, min_d=5.0, max_d=25.0)
                self._release("warm", o, rec.pos.copy(), 9.0, 0.0, receiver=rec)
            return
        self.mode = "owned"
        self.mode_data = {"owner": self.players(1)[6], "hold_until": self.t + 1.0}


# ============================================================ hang
def make_audio(path, total_sec, whistles, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    n = int(total_sec * sr)
    noise = rng.normal(0, 0.05, n).astype(np.float32)
    noise = np.convolve(noise, np.ones(8) / 8, mode="same").astype(np.float32) * 2.0
    for w in whistles:
        i0 = int(w["time"] * sr)
        i1 = min(n, int((w["time"] + w["duration"]) * sr))
        if i1 <= i0 or i0 < 0:
            continue
        t = np.arange(i1 - i0) / sr
        freq = 3200 + 120 * np.sin(2 * np.pi * 35 * t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        env = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
        noise[i0:i1] += (0.3 * env * np.sin(phase)).astype(np.float32)
    pcm = (np.clip(noise, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ============================================================ megjelenítés
class Renderer:
    def __init__(self, width, height, rng):
        self.w, self.h = width, height
        self.rng = rng
        self.tex, self.M_tex = build_pitch_texture(rng)
        self.cam = Camera(width, height)
        # égbolt: függőleges színátmenet
        grad = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        top = np.array([230, 190, 150], np.float32)
        bot = np.array([235, 225, 215], np.float32)
        self.sky = np.repeat((top * (1 - grad) + bot * grad)[:, None, :], width, axis=1).astype(np.uint8)
        # fasor-textúra (azimut szerint görgetve)
        self.tree_w = 6000
        tn = (_smooth_noise(rng, height, self.tree_w, 6) * 0.25 + rng.standard_normal((height, self.tree_w)).astype(np.float32) * 0.08)
        base = np.array([40, 75, 45], np.float32)
        self.trees = np.clip(base * (1 + tn[..., None]), 0, 255).astype(np.uint8)
        az = np.linspace(-math.pi, math.pi, 3600)
        th = 7.0 + 2.0 * np.sin(az * 7.0) + 1.5 * np.sin(az * 23.0 + 1.0) + 1.0 * np.sin(az * 61.0)
        self.tree_h = th.astype(np.float32)
        # szenzorzaj
        self.noise_p = np.clip(np.abs(rng.normal(0, 3.0, (height + 32, width + 32, 3))), 0, 30).astype(np.uint8)
        self.noise_n = np.clip(np.abs(rng.normal(0, 3.0, (height + 32, width + 32, 3))), 0, 30).astype(np.uint8)
        # reklámtáblák
        self.boards = []
        x = 8.0
        k = 0
        while x < L - 8.0:
            c = CASUAL[(k * 7 + 3) % len(CASUAL)]
            self.boards.append((x, min(x + 6.0, L - 8.0), tuple(int(v) for v in c)))
            x += 6.0
            k += 1

    # --- háttér
    def _background(self, frame):
        cam = self.cam
        w, h = self.w, self.h
        us = np.arange(w, dtype=np.float64)
        hdir = np.array([math.sin(cam.pan), -math.cos(cam.pan)])
        rr = np.array([math.cos(cam.pan), math.sin(cam.pan)])
        # a horizont magasságában a sugár vízszintes összetevője: hdir / cos(tilt) + r * (u - cx) / f
        dirs = hdir[None, :] / math.cos(cam.tilt) + rr[None, :] * ((us - cam.cx) / cam.f)[:, None]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        C = cam.C[:2]
        # metszés a textúra határával
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dirs[:, 0] > 0, (TEX_X1 - C[0]) / dirs[:, 0], (TEX_X0 - C[0]) / dirs[:, 0])
            ty = np.where(dirs[:, 1] > 0, (TEX_Y1 - C[1]) / dirs[:, 1], (TEX_Y0 - C[1]) / dirs[:, 1])
        tx = np.where(np.isfinite(tx) & (tx > 0), tx, 1e9)
        ty = np.where(np.isfinite(ty) & (ty > 0), ty, 1e9)
        D = np.minimum(tx, ty) - 0.5
        az = np.arctan2(dirs[:, 0], -dirs[:, 1])
        ai = ((az + math.pi) / (2 * math.pi) * 3599).astype(int).clip(0, 3599)
        Ht = self.tree_h[ai]
        base_pts = np.stack([C[0] + dirs[:, 0] * D, C[1] + dirs[:, 1] * D, np.zeros(w)], 1)
        top_pts = base_pts.copy()
        top_pts[:, 2] = Ht
        uvb, _ = cam.project(base_pts)
        uvt, _ = cam.project(top_pts)
        vb = uvb[:, 1]
        vt = uvt[:, 1]
        rows = np.arange(h, dtype=np.float64)[:, None]
        tree_mask = rows < vb[None, :]
        if tree_mask.any():
            shift = int(cam.pan * cam.f) % self.tree_w
            cols = (np.arange(w) + shift) % self.tree_w
            trees = self.trees[:, cols]
            sky_mask = rows < vt[None, :]
            bg = np.where(sky_mask[..., None], self.sky, trees)
            np.copyto(frame, bg, where=tree_mask[..., None])

    # --- alakok
    def _draw_person(self, frame, foot, head, kit, skin, phase, moving, items_bbox):
        a = np.asarray(foot, np.float64)
        b = np.asarray(head, np.float64)
        hpx = a[1] - b[1]
        if hpx < 3:
            return None
        e = b - a

        def P(s, dx=0.0):
            p = a + e * s
            return (int(round((p[0] + dx * hpx) * 4)), int(round(p[1] * 4)))

        sh = 2
        lw = max(1, int(round(hpx * 0.075)))
        swing = math.sin(phase) * 0.09 * moving
        # árnyék (a nap jobbról-hátulról)
        x0 = int(a[0] - hpx * 0.1); x1 = int(a[0] + hpx * 0.55)
        y0 = int(a[1] - hpx * 0.06); y1 = int(a[1] + hpx * 0.06) + 1
        H_, W_ = frame.shape[:2]
        if x1 > 0 and y1 > 0 and x0 < W_ and y0 < H_:
            xa, xb, ya, yb = max(0, x0), min(W_, x1), max(0, y0), min(H_, y1)
            if xb > xa and yb > ya:
                roi = frame[ya:yb, xa:xb]
                m = np.zeros(roi.shape[:2], np.uint8)
                cv2.ellipse(m, (int(a[0] + hpx * 0.22) - xa, int(a[1]) - ya),
                            (max(1, int(hpx * 0.3)), max(1, int(hpx * 0.045))), 0, 0, 360, 255, -1)
                sel = m > 0
                roi[sel] = (roi[sel] * 0.55).astype(np.uint8)
        # lábak
        for side, sw in ((-1, swing), (1, -swing)):
            hip = P(0.47, side * 0.045)
            footp = P(0.0, side * 0.05 + sw)
            knee = P(0.24, side * 0.05 + sw * 0.5)
            cv2.line(frame, hip, knee, skin, lw, cv2.LINE_AA, sh)
            cv2.line(frame, knee, footp, kit["socks"], lw, cv2.LINE_AA, sh)
            cv2.circle(frame, footp, max(1, lw // 2) * 4, (25, 25, 25), -1, cv2.LINE_AA, sh)
        # nadrág
        pts = np.array([P(0.42, -0.11), P(0.42, 0.11), P(0.57, 0.12), P(0.57, -0.12)], np.int32)
        cv2.fillConvexPoly(frame, pts, kit["shorts"], cv2.LINE_AA, sh)
        # karok
        for side, sw in ((-1, -swing), (1, swing)):
            sh_pt = P(0.80, side * 0.13)
            el = P(0.64, side * 0.16 + sw * 0.6)
            hand = P(0.50, side * 0.15 + sw)
            cv2.line(frame, sh_pt, el, kit["shirt"], max(1, int(lw * 0.9)), cv2.LINE_AA, sh)
            cv2.line(frame, el, hand, skin, max(1, int(lw * 0.8)), cv2.LINE_AA, sh)
        # törzs
        pts = np.array([P(0.55, -0.11), P(0.55, 0.11), P(0.83, 0.14), P(0.83, -0.14)], np.int32)
        cv2.fillConvexPoly(frame, pts, kit["shirt"], cv2.LINE_AA, sh)
        # fej
        hr = max(1.0, hpx * 0.068)
        hc = P(0.92)
        cv2.circle(frame, hc, int(hr * 4), skin, -1, cv2.LINE_AA, sh)
        cv2.circle(frame, (hc[0], hc[1] - int(hr * 1.6)), int(hr * 0.75 * 4), (30, 30, 40), -1, cv2.LINE_AA, sh)
        xs = [a[0], b[0]]
        x_lo = min(xs) - hpx * (0.16 + abs(swing))
        x_hi = max(xs) + hpx * (0.16 + abs(swing))
        y_lo = (a + e * 0.92)[1] - hr
        y_hi = a[1] + lw * 0.5
        return [x_lo, y_lo, x_hi, y_hi]

    def _draw_ball(self, frame, c, c_prev, r):
        H_, W_ = frame.shape[:2]
        L_ = math.hypot(c[0] - c_prev[0], c[1] - c_prev[1])
        pad = int(r + L_ + 3)
        x0, y0 = int(min(c[0], c_prev[0]) - pad), int(min(c[1], c_prev[1]) - pad)
        x1, y1 = int(max(c[0], c_prev[0]) + pad), int(max(c[1], c_prev[1]) + pad)
        xa, xb, ya, yb = max(0, x0), min(W_, x1), max(0, y0), min(H_, y1)
        if xb <= xa or yb <= ya:
            return None
        roi = frame[ya:yb, xa:xb]
        m = np.zeros(roi.shape[:2], np.uint8)
        p1 = (int(round((c[0] - xa) * 4)), int(round((c[1] - ya) * 4)))
        p0 = (int(round((c_prev[0] - xa) * 4)), int(round((c_prev[1] - ya) * 4)))
        th = max(1, int(round(2 * r)))
        if L_ > 1.0:
            cv2.line(m, p0, p1, 255, th, cv2.LINE_AA, 2)
            alpha = min(1.0, (2 * r + 1.0) / (L_ + 2 * r)) * 0.9 + 0.1
        else:
            cv2.circle(m, p1, max(2, int(r * 4)), 255, -1, cv2.LINE_AA, 2)
            alpha = 1.0
        a = (m.astype(np.float32) / 255.0 * alpha)[..., None]
        col = np.array([238, 240, 240], np.float32)
        roi[:] = (roi.astype(np.float32) * (1 - a) + col * a).astype(np.uint8)
        hw = max(r, th / 2.0)
        return [min(c[0], c_prev[0]) - hw, min(c[1], c_prev[1]) - hw,
                max(c[0], c_prev[0]) + hw, max(c[1], c_prev[1]) + hw]

    def render(self, st, st_prev, cam_params, people, gait):
        cam = self.cam
        cam.set(*cam_params)
        w, h = self.w, self.h
        Hp2i = cam.H_pitch2img()
        Mw = Hp2i @ self.M_tex
        frame = cv2.warpPerspective(self.tex, Mw, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(50, 90, 60))
        self._background(frame)

        items = []  # (mélység, típus, adat)
        # reklámtáblák
        for x0, x1, col in self.boards:
            q = np.array([[x0, BOARD_Y, 0], [x1, BOARD_Y, 0], [x1, BOARD_Y, 0.9], [x0, BOARD_Y, 0.9]])
            uv, z = cam.project(q)
            if (z > 0.5).all():
                items.append((float(z.mean()), "board", (uv, col)))
        # kapuk
        for gx, s in ((0.0, -1.0), (L, 1.0)):
            pts = np.array([[gx, POST_LO, 0], [gx, POST_LO, GOAL_H], [gx, POST_HI, GOAL_H], [gx, POST_HI, 0],
                            [gx + s * 1.8, POST_LO, 0], [gx + s * 1.0, POST_LO, 2.0],
                            [gx + s * 1.0, POST_HI, 2.0], [gx + s * 1.8, POST_HI, 0]])
            uv, z = cam.project(pts)
            if (z > 0.5).all():
                items.append((float(z[:4].mean()) - 0.01, "goal", (uv, s)))
        # emberek
        pos = st["pos"]
        foot3 = np.concatenate([pos, np.zeros((len(pos), 1))], 1)
        heights = np.array([p.height for p in people])
        head3 = foot3.copy()
        head3[:, 2] = heights
        uvf, zf = cam.project(foot3)
        uvh, zh = cam.project(head3)
        speed = np.linalg.norm(st["vel"], axis=1)
        for i, p in enumerate(people):
            if zf[i] < 1.0 or zh[i] < 1.0:
                continue
            u, v = uvf[i]
            hp = uvf[i][1] - uvh[i][1]
            if u < -hp or u > w + hp or v < -5 or uvh[i][1] > h + 5:
                continue
            items.append((float(zf[i]), "person", i))
        # labda
        b = st["ball"]
        bc = np.array([[b[0], b[1], b[2] + BALL_R]])
        uvb, zb = cam.project(bc)
        ball_info = None
        if st["ball_vis"] and zb[0] > 1.0:
            bp = st_prev["ball"] if st_prev is not None else b
            mid = 0.5 * (bp + b)   # kb. fél képkockás expozíció
            uvp, _ = cam.project(np.array([[mid[0], mid[1], mid[2] + BALL_R]]))
            r = max(1.2, cam.f * BALL_R / zb[0])
            items.append((float(zb[0]), "ball", (uvb[0], uvp[0], r)))
            # labdaárnyék
            uvs, zs = cam.project(np.array([[b[0] + 0.15 * b[2], b[1], 0.0]]))
            if zs[0] > 1.0:
                items.append((float(zs[0]) + 0.05, "ball_shadow", (uvs[0], r)))
        items.sort(key=lambda it: -it[0])

        boxes = {}
        ball_box = None
        for depth, kind, data in items:
            if kind == "board":
                uv, col = data
                cv2.fillConvexPoly(frame, (uv * 4).astype(np.int32), col, cv2.LINE_AA, 2)
                cv2.polylines(frame, [(uv * 4).astype(np.int32)], True, (230, 230, 230), 1, cv2.LINE_AA, 2)
            elif kind == "goal":
                uv, s = data
                q = (uv * 4).astype(np.int32)
                net = (205, 205, 205)
                for a_, b_ in ((4, 5), (5, 6), (6, 7), (0, 4), (3, 7), (1, 5), (2, 6), (4, 7)):
                    cv2.line(frame, _pt(q[a_]), _pt(q[b_]), net, 1, cv2.LINE_AA, 2)
                for k in range(1, 8):
                    f_ = k / 8.0
                    top = q[1] + (q[2] - q[1]) * f_
                    bot = q[4] + (q[7] - q[4]) * f_
                    cv2.line(frame, _pt(top), _pt(bot), net, 1, cv2.LINE_AA, 2)
                pw = max(1, int(round(cam.f * 0.12 / max(depth, 1.0))))
                for a_, b_ in ((0, 1), (1, 2), (2, 3)):
                    cv2.line(frame, _pt(q[a_]), _pt(q[b_]), (245, 245, 245), pw, cv2.LINE_AA, 2)
            elif kind == "person":
                i = data
                p = people[i]
                moving = min(1.0, speed[i] / 3.0)
                bb = self._draw_person(frame, uvf[i], uvh[i], p.kit, p.skin, gait[i], moving, None)
                if bb is not None:
                    boxes[i] = (bb, float(zf[i]), (float(uvf[i][0]), float(uvf[i][1])))
            elif kind == "ball_shadow":
                uv, r = data
                cv2.ellipse(frame, (int(uv[0] * 4), int(uv[1] * 4)), (max(1, int(r * 4 * 1.1)), max(1, int(r * 4 * 0.5))),
                            0, 0, 360, (30, 70, 35), -1, cv2.LINE_AA, 2)
            elif kind == "ball":
                c, cp, r = data
                ball_box = self._draw_ball(frame, c, cp, r)
                ball_info = {"uv": [float(c[0]), float(c[1])], "r": float(r), "depth": float(depth),
                             "blur_px": float(math.hypot(c[0] - cp[0], c[1] - cp[1])) * 2.0}
        frame = cv2.GaussianBlur(frame, (3, 3), 0.7)
        ox, oy = int(self.rng.integers(0, 32)), int(self.rng.integers(0, 32))
        cv2.add(frame, self.noise_p[oy:oy + h, ox:ox + w], dst=frame)
        cv2.subtract(frame, self.noise_n[32 - oy:32 - oy + h, 32 - ox:32 - ox + w], dst=frame)
        return frame, Hp2i, boxes, ball_box, ball_info


def _pt(a):
    return (int(a[0]), int(a[1]))


def _iou_cover(a, b):
    """Az 'a' doboz mekkora részét takarja 'b' (0..1)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return (x2 - x1) * (y2 - y1) / area


# ============================================================ kamera pálya (Veo-szerű követés)
# Kameraprofilok: omega – a követés sajátfrekvenciája (1/s, nagyobb = gyorsabb pásztázás),
# zoom – a fókusz tartománya f0 szorzójaként, zoom_tau – a zoom időállandója (s),
# zoom_pulse – időnkénti hirtelen rázoomolás amplitúdója (0 = nincs)
CAMERA_PROFILES = {
    "alap": {"omega": 1.7, "zoom": (0.9, 1.45), "zoom_tau": 1.2, "zoom_pulse": 0.0},
    "nehez": {"omega": 2.8, "zoom": (0.8, 1.9), "zoom_tau": 0.5, "zoom_pulse": 0.35},
}


def camera_path(states, fps, width, profile="alap", seed=0):
    prof = CAMERA_PROFILES[profile] if isinstance(profile, str) else profile
    dt = 1.0 / fps
    omega = prof["omega"]
    zlo, zhi = prof["zoom"]
    prng = np.random.default_rng(seed + 12345)
    pulse, pulse_left = 0.0, 0
    f0 = (width / 2.0) / math.tan(math.radians(26.0))
    b0 = states[0]["ball"]
    aim = np.array([b0[0], b0[1]], np.float64)
    vel = np.zeros(2)
    f = f0
    last_target = aim.copy()
    cam = Camera(width, int(width * 9 / 16))
    out = []
    for st in states:
        if st["ball_vis"]:
            last_target = np.array([st["ball"][0], st["ball"][1]])
        tgt = last_target.copy()
        tgt[0] = float(np.clip(tgt[0], -2.0, L + 2.0))
        tgt[1] = float(np.clip(tgt[1], 6.0, W - 4.0))
        acc = omega * omega * (tgt - aim) - 2.0 * omega * vel
        vel += acc * dt
        aim += vel * dt
        dist = math.hypot(aim[0] - CAM_POS[0], aim[1] - CAM_POS[1])
        f_t = f0 * float(np.clip((dist / 40.0) ** 0.5, zlo, zhi))
        if prof["zoom_pulse"] > 0:
            # hirtelen rázoomolás (pl. a labda közelében), majd vissza
            if pulse_left <= 0 and prng.random() < dt / 6.0:
                pulse, pulse_left = float(prng.uniform(0.5, 1.0)) * prof["zoom_pulse"], int(prng.integers(25, 75))
            if pulse_left > 0:
                pulse_left -= 1
                f_t *= 1.0 + pulse
        f += (f_t - f) * min(1.0, dt / prof["zoom_tau"])
        # az optikai tengely kicsit a labda elé (a kamera felé) néz: a labda a kép közepe fölé kerül,
        # a túlsó oldalvonal a kép felső negyedébe (Veo-szerű kivágás)
        to_cam = np.array([CAM_POS[0] - aim[0], CAM_POS[1] - aim[1]])
        off = to_cam / max(1e-6, np.linalg.norm(to_cam)) * min(4.0, 0.1 * dist)
        cam.aim_at(aim[0] + off[0], aim[1] + off[1], f)
        out.append((cam.pan, cam.tilt, cam.f))
    return out


# ============================================================ fő folyamat
def find_ffmpeg():
    return os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def parse_segments(spec, duration):
    if not spec:
        return [("half1", float(duration))]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, sec = part.partition(":")
        name = name.strip().lower()
        if name not in ("warmup", "half1", "half2", "break"):
            raise ValueError(f"Ismeretlen szakasz: {name!r} (lehet: warmup, half1, break, half2)")
        out.append((name, float(sec) if sec else float(duration)))
    return out


def _r(v, n=2):
    return round(float(v), n)


def generate(out_path, gt_path=None, duration=60.0, width=1280, height=720, fps=25, seed=7,
             segments=None, audio=True, crf=23, preset="veryfast", progress=True, save_frames=None,
             camera_profile="alap"):
    """
    A szintetikus meccs legenerálása. Visszatér: a ground-truth dict (és kiírja a JSON-t is).
    save_frames: {képkocka index: jpg útvonal} – mintaképek mentése.
    camera_profile: a kamera mozgásprofilja (CAMERA_PROFILES: "alap" vagy "nehez" – gyorsabb
    pásztázás, nagyobb és gyorsabb zoom, hirtelen rázoomolások).
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("Nem található ffmpeg (PATH vagy FFMPEG_PATH)")
    t_start = time.time()
    rng = np.random.default_rng(seed)
    segs = segments if isinstance(segments, list) else parse_segments(segments, duration)
    sim = MatchSim(fps, segs, rng)
    states = sim.run()
    n_frames = len(states)
    total_sec = n_frames / fps
    if progress:
        print(f"Szimuláció kész: {n_frames} képkocka, {len(sim.events)} esemény ({time.time() - t_start:.1f} s)")

    cams = camera_path(states, fps, width, camera_profile, seed)
    renderer = Renderer(width, height, np.random.default_rng(seed + 1))
    people = sim.people

    gt_path = gt_path or os.path.splitext(out_path)[0] + "_gt.json"
    tmpdir = tempfile.mkdtemp(prefix="synth_")
    wav = os.path.join(tmpdir, "audio.wav")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
    if audio:
        make_audio(wav, total_sec + 1.0, sim.whistles, seed=seed)
        cmd += ["-i", wav, "-c:a", "aac", "-b:a", "64k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-g", str(fps * 2), out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    gait = np.zeros(len(people))
    frames_gt = []
    try:
        for fi, st in enumerate(states):
            gait += np.linalg.norm(st["vel"], axis=1) / fps * 2.6
            frame, Hp2i, boxes, ball_box, ball_info = renderer.render(
                st, states[fi - 1] if fi > 0 else None, cams[fi], people, gait)
            proc.stdin.write(frame.tobytes())
            if save_frames and fi in save_frames:
                cv2.imwrite(save_frames[fi], frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            Hi2p = np.linalg.inv(Hp2i)
            Hi2p /= Hi2p[2, 2]
            # takarás: egy közelebbi alak doboza fedi-e
            vis = []
            for i, (bb, z, fuv) in boxes.items():
                if bb[2] < 0 or bb[0] > width or bb[3] < 0 or bb[1] > height:
                    continue
                vis.append((i, bb, z, fuv))
            objs = []
            for i, bb, z, fuv in vis:
                occ = max([_iou_cover(bb, bb2) for j, bb2, z2, _f in vis if j != i and z2 < z] or [0.0])
                x, y = st["pos"][i]
                objs.append([i, _r(x), _r(y), _r(fuv[0], 1), _r(fuv[1], 1),
                             _r(bb[0], 1), _r(bb[1], 1), _r(bb[2], 1), _r(bb[3], 1), _r(occ)])
            b = st["ball"]
            ball = {"xyz": [_r(b[0]), _r(b[1]), _r(b[2])], "vis": False, "on": bool(st["ball_vis"])}
            if ball_info is not None and ball_box is not None:
                u, v = ball_info["uv"]
                inside = 0 <= u < width and 0 <= v < height
                bocc = 0.0
                for i, bb, z, _f in vis:
                    if z < ball_info["depth"] and bb[0] <= u <= bb[2] and bb[1] <= v <= bb[3]:
                        bocc = 1.0
                ball.update({"vis": bool(inside), "uv": [_r(u, 1), _r(v, 1)],
                             "bbox": [_r(x, 1) for x in ball_box], "r": _r(ball_info["r"]),
                             "blur": _r(ball_info["blur_px"], 1), "occ": bocc})
            frames_gt.append({
                "t": _r(st["t"], 3), "seg": st["seg"],
                "H": [float(f"{v:.6g}") for v in Hi2p.reshape(-1)],
                "cam": [_r(cams[fi][0], 5), _r(cams[fi][1], 5), _r(cams[fi][2], 1)],
                "ball": ball, "objs": objs,
                "xy": [[_r(p[0]), _r(p[1])] for p in st["pos"][:sim.n_match]],
            })
            if progress and fi % 250 == 0:
                print(f"  render {fi}/{n_frames} ({time.time() - t_start:.1f} s)")
    finally:
        proc.stdin.close()
        proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)

    # első képkocka kalibráció: a pálya 4 sarka (BF, BA, JA, JF) relatív képkoordinátában
    H0 = np.linalg.inv(np.array(frames_gt[0]["H"]).reshape(3, 3))
    H0 /= H0[2, 2]   # a (0,0) sarok a kamera előtt van -> pozitív mélység-skála
    corners = np.array([[0, 0], [0, W], [L, W], [L, 0]], np.float64)
    ph = np.concatenate([corners, np.ones((4, 1))], 1) @ H0.T
    in_front = bool((ph[:, 2] > 0).all())
    uv = ph[:, :2] / ph[:, 2:3]
    calib_str = ";".join(f"{u / width:.6f},{v / height:.6f}" for u, v in uv)

    gt = {
        "version": 1,
        "video": os.path.basename(out_path),
        "fps": fps, "width": width, "height": height, "n_frames": n_frames,
        "duration": round(total_sec, 3), "seed": seed,
        "pitch": {"length": L, "width": W},
        "camera": {"position": CAM_POS.tolist(), "model": "tűlyuk, pan/tilt/zoom, torzítás nélkül",
                   "profile": camera_profile if isinstance(camera_profile, str) else "egyedi"},
        "segments": [{"name": n, "duration": d} for n, d in segs],
        "periods": sim.periods,
        "whistles": sim.whistles,
        "passes": sim.passes,
        "people": [{"id": p.id, "role": p.role, "team": p.team, "number": p.number} for p in people],
        "events": sorted(sim.events, key=lambda e: e["time"]),
        "calibration_first_frame": {"corners_img": uv.round(2).tolist(), "calib_points": calib_str,
                                    "all_in_front_of_camera": in_front},
        "frame_fields": {
            "H": "3x3 homográfia sorfolytonosan: kép (px) -> pálya (m)",
            "cam": "[pan rad, tilt rad, fókusz px]",
            "objs": "[person_id, x_m, y_m, láb_u, láb_v, x1, y1, x2, y2, takarás 0..1] – csak a képen látható alakok",
            "xy": "minden játékos + bíró pálya-pozíciója (people sorrendjében, az első n_match elem)",
            "ball": "xyz: pálya m + magasság; on: játékban / látható helyen van (nem viszik); uv: képpont; vis: a képen látszik; occ: takarja egy alak",
        },
        "frames": frames_gt,
    }
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, separators=(",", ":"))
    if progress:
        print(f"Kész: {out_path} + {gt_path} ({time.time() - t_start:.1f} s)")
    return gt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Szintetikus focimeccs-videó és ground-truth JSON generálása tesztekhez.")
    ap.add_argument("--out", default="synth_match.mp4", help="kimeneti MP4 fájl (alap: synth_match.mp4)")
    ap.add_argument("--gt", default=None, help="ground-truth JSON útvonala (alap: <out>_gt.json)")
    ap.add_argument("--duration", type=float, default=60.0, help="a játékrész hossza mp-ben, ha nincs --segments (alap: 60)")
    ap.add_argument("--width", type=int, default=1280, help="képszélesség (alap: 1280)")
    ap.add_argument("--height", type=int, default=720, help="képmagasság (alap: 720)")
    ap.add_argument("--fps", type=int, default=25, help="képkocka / mp (alap: 25)")
    ap.add_argument("--seed", type=int, default=7, help="véletlen mag (alap: 7)")
    ap.add_argument("--segments", default=None,
                    help='szakaszok, pl. "warmup:20,half1:90,break:30,half2:90" (alap: egy félidő --duration hosszal)')
    ap.add_argument("--no-audio", action="store_true", help="hangsáv (sípszavak) nélkül")
    ap.add_argument("--crf", type=int, default=23, help="H.264 minőség (alap: 23)")
    ap.add_argument("--camera", default="alap", choices=sorted(CAMERA_PROFILES),
                    help="kamera mozgásprofil: alap vagy nehez (gyorsabb pásztázás, több zoom; alap: alap)")
    ap.add_argument("--save-frames", default=None,
                    help="mintaképek mentése: 'index1,index2,...:könyvtár' (pl. '0,300,900:/tmp/kepek')")
    args = ap.parse_args(argv)
    save = None
    if args.save_frames:
        idx, _, d = args.save_frames.partition(":")
        os.makedirs(d or ".", exist_ok=True)
        save = {int(i): os.path.join(d or ".", f"frame_{int(i):05d}.jpg") for i in idx.split(",") if i.strip()}
    generate(args.out, args.gt, args.duration, args.width, args.height, args.fps, args.seed,
             args.segments, audio=not args.no_audio, crf=args.crf, save_frames=save,
             camera_profile=args.camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())

import cv2
import numpy as np
import math

from events import EventDetector, EventConfig, GK_EVENT_TYPES, annotate_categories

class TacticalEngine:
    def __init__(self, pitch_length=105.0, pitch_width=68.0, team1_attacks_right=True, event_config=None):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Valós FIFA pálya sarok- és kapupontjai (méterben)
        # Saját kapu középpontja: (0.0, 34.0), kapufák: (0.0, 30.34) és (0.0, 37.66)
        self.goal_left_post = np.array([0.0, 30.34])
        self.goal_right_post = np.array([0.0, 37.66])
        self.goal_center = np.array([0.0, 34.0])

        self.player_trajectories = {}
        self.ball_trajectory = []

        # Esemény-detektor (lövés, gól, kapus események) – lásd events.py
        cfg = event_config or EventConfig()
        cfg.team1_attacks_right = bool(team1_attacks_right)
        self.event_detector = EventDetector(cfg, pitch_length, pitch_width)

        # A statisztikák ugyanazok a dict objektumok, mint a detektorban (nincs külön szinkron). A passzok /
        # szerelések stb. a detektor KIADOTT eseményeit számolják (korábban itt egy külön, naiv számlálás
        # futott, ami nem egyezett az eseménylistával).
        self.stats = self.event_detector.stats

    @property
    def events(self):
        """A detektor által eddig talált események (idő szerint rendezve)."""
        return sorted(self.event_detector.events, key=lambda e: e["time"])

    def set_attack_direction(self, team1_attacks_right):
        """Félidőnként fordítható: True esetén az 1. csapat az x=105 kapura támad."""
        self.event_detector.set_attack_direction(team1_attacks_right)

    def finalize(self, timestamp=None):
        """A feldolgozás végén lezárja a függő eseményeket (pl. megerősítetlen védés)."""
        return self.event_detector.finalize(timestamp)

    def keeper_team_near(self, coords):
        """A megadott pálya-pozícióhoz legközelebbi kapus csapata ("team1"/"team2"), vagy None."""
        return self.event_detector.keeper_team_near(coords)

    def merge_timeline(self, external_events):
        """Összefésüli a külső (pl. main.py helyezkedési hiba) eseményeket a detektor eseményeivel,
        idő szerint rendezve, és a külső kapus eseményeket is beszámítja a statisztikába."""
        for ev in external_events:
            team = ev.get("team")
            if ev.get("type") in GK_EVENT_TYPES and team in self.stats["gk_events"]:
                bucket = self.stats["gk_events"][team]
                bucket[ev["type"]] = bucket.get(ev["type"], 0) + 1
        annotate_categories(external_events)
        return sorted(list(external_events) + list(self.event_detector.events), key=lambda e: e["time"])

    def compute_homography(self, src_points):
        """
        src_points: 4 képpont (pixel): Bal-felső, Bal-alsó, Jobb-alsó, Jobb-felső
        Leképezi a pontokat a 105x68 méteres valós koordinátákra.
        """
        if len(src_points) != 4:
            return None

        dst_points = np.array([
            [0.0, 0.0],
            [0.0, self.pitch_width],
            [self.pitch_length, self.pitch_width],
            [self.pitch_length, 0.0]
        ], dtype=np.float32)

        src_pts = np.array(src_points, dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_points, cv2.RANSAC, 5.0)
        return H

    def transform_to_pitch(self, foot_x, foot_y, H):
        """Átváltja a képi talppontot méter alapú koordinátára. 
        A vonalon túli nézőket/edzőket automatikusan kiszűri (None értékkel tér vissza)."""
        if H is None:
            return None
        pt = np.array([[[float(foot_x), float(foot_y)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)
        real_x = float(transformed[0][0][0])
        real_y = float(transformed[0][0][1])

        # 2.0 méteres tűréshatár az oldalvonalon kívül a bedobások és szögletek miatt
        if real_x < -2.0 or real_x > self.pitch_length + 2.0 or real_y < -2.0 or real_y > self.pitch_width + 2.0:
            return None
        return real_x, real_y

    # LABDA-ÁTHELYEZÉS: a hálóban nyugvó labda (gól után; x a [-2,5; 0] / [105; 107,5] sávban, y a
    # kapufák ±1 m-én) érvényes labda-pozíció. A labda KÖZÉPPONTJÁNAK vetülete ráadásul a kamerától
    # távolabb esik (a szintetikus mérésben kb. 1 m), ezért a régi ±2 m-es doboz a hálóban lévő labdát
    # szinte mindig kidobta (a gól-megerősítés emiatt maradt el). A hálón kívül marad a régi 2 m: a kapus
    # kezében (~1 m magasan) tartott labda vetülete pl. az alapvonal mögé, a kapun kívülre esik, és az
    # events.py ezt (a labda "eltűnését" a kapusnál) védésként értelmezi.
    BALL_NET_TOL_X_M = 1.5     # a háló mélységén túl ennyi vetítési tűrés
    BALL_NET_TOL_Y_M = 0.5     # a kapufák ±1 m-én túl ennyi

    def ball_in_net_zone(self, x, y):
        """A gólvonal mögött, a kapuhálóban (vetítési tűréssel) van-e a pont."""
        lo, hi = 30.34 - 1.0 - self.BALL_NET_TOL_Y_M, 37.66 + 1.0 + self.BALL_NET_TOL_Y_M
        depth = 2.5 + self.BALL_NET_TOL_X_M
        return lo <= y <= hi and (-depth <= x <= 0.0 or self.pitch_length <= x <= self.pitch_length + depth)

    def transform_ball_to_pitch(self, u, v, H):
        """A labda képpontja -> pálya (m): mint transform_to_pitch, de a kapuhálóban lévő labdát is elfogadja."""
        if H is None:
            return None
        pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
        q = cv2.perspectiveTransform(pt, H)[0][0]
        x, y = float(q[0]), float(q[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if self.ball_in_net_zone(x, y):
            return x, y
        if x < -2.0 or x > self.pitch_length + 2.0 or y < -2.0 or y > self.pitch_width + 2.0:
            return None
        return x, y

    def update_tracks(self, frame_idx, timestamp, detections, ball_pos, H):
        current_frame_players = []

        for det in detections:
            tid = det.get('track_id')
            if tid is None:
                continue

            fx, fy = det['screen_foot']
            real_coords = self.transform_to_pitch(fx, fy, H)
            if real_coords is None:
                continue  # Pályán kívüli néző: végleg kizárva

            rx, ry = real_coords
            det['pitch_coords'] = (rx, ry)
            current_frame_players.append(det)

            if tid not in self.player_trajectories:
                self.player_trajectories[tid] = []
            self.player_trajectories[tid].append((rx, ry, timestamp))

        real_ball = None
        if ball_pos is not None and H is not None:
            real_ball_coords = self.transform_ball_to_pitch(ball_pos[0], ball_pos[1], H)
            if real_ball_coords is not None:
                real_ball = real_ball_coords
                self.ball_trajectory.append((real_ball[0], real_ball[1], timestamp))

        # Esemény-detektálás minden képkockán (játékosok nélkül is: a lövés/gól a labdán múlik)
        self.event_detector.update(timestamp, real_ball, current_frame_players)

        return current_frame_players, real_ball

    def analyze_goalkeeper(self, gk_coords, ball_coords):
        """Kiszámítja a kapus szögfelezőjét és az ideális pozíciótól mért eltérést méterben."""
        if gk_coords is None or ball_coords is None:
            return None

        gx, gy = gk_coords
        bx, by = ball_coords

        # A kapus a hozzá közelebbi kaput védi (bal: x=0, jobb: x=pálya hossza)
        if gx <= self.pitch_length / 2.0:
            post_a, post_b = self.goal_left_post, self.goal_right_post
            dist_from_goal = gx
        else:
            post_a = np.array([self.pitch_length, self.goal_left_post[1]])
            post_b = np.array([self.pitch_length, self.goal_right_post[1]])
            dist_from_goal = self.pitch_length - gx

        vec_left = post_a - np.array([bx, by])
        vec_right = post_b - np.array([bx, by])

        norm_l = vec_left / np.linalg.norm(vec_left)
        norm_r = vec_right / np.linalg.norm(vec_right)
        bisector = (norm_l + norm_r) / np.linalg.norm(norm_l + norm_r)

        line_pt1 = np.array([bx, by])
        line_pt2 = np.array([bx, by]) + bisector * 50.0

        p1_to_gk = np.array([gx, gy]) - line_pt1
        p1_to_p2 = line_pt2 - line_pt1
        # 2D vektoriális szorzat kézzel (a numpy 2.x np.cross már nem fogad 2 elemű vektorokat)
        cross_prod = p1_to_p2[0] * p1_to_gk[1] - p1_to_p2[1] * p1_to_gk[0]
        offset_meters = cross_prod / np.linalg.norm(p1_to_p2)

        return {
            "dist_from_goal_m": round(float(dist_from_goal), 2),
            "bisector_offset_m": round(float(offset_meters), 2),
            "gk_coords": (gx, gy),
            "ball_coords": (bx, by)
        }
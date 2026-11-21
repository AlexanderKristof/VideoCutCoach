import cv2
import numpy as np
import math

from events import EventDetector, EventConfig, GK_EVENT_TYPES

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
        self.possession_history = []

        # Esemény-detektor (lövés, gól, kapus események) – lásd events.py
        cfg = event_config or EventConfig()
        cfg.team1_attacks_right = bool(team1_attacks_right)
        self.event_detector = EventDetector(cfg, pitch_length, pitch_width)

        # A lövés/gól/kapus statisztikák ugyanazok a dict objektumok, mint a detektorban (nincs külön szinkron)
        det_stats = self.event_detector.stats
        self.stats = {
            "passes": {"team1": 0, "team2": 0},
            "tackles": {"team1": 0, "team2": 0},
            "shots": det_stats["shots"],
            "shots_on_target": det_stats["shots_on_target"],
            "goals": det_stats["goals"],
            "gk_events": det_stats["gk_events"]
        }

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
            real_ball_coords = self.transform_to_pitch(ball_pos[0], ball_pos[1], H)
            if real_ball_coords is not None:
                real_ball = real_ball_coords
                self.ball_trajectory.append((real_ball[0], real_ball[1], timestamp))

        if real_ball and current_frame_players:
            self._evaluate_possession_and_events(real_ball, current_frame_players, timestamp)

        # Esemény-detektálás minden képkockán (játékosok nélkül is: a lövés/gól a labdán múlik)
        self.event_detector.update(timestamp, real_ball, current_frame_players)

        return current_frame_players, real_ball

    def _evaluate_possession_and_events(self, ball_xy, players, timestamp):
        bx, by = ball_xy
        closest_player = None
        min_dist = 999.0

        for p in players:
            px, py = p['pitch_coords']
            dist = math.hypot(px - bx, py - by)
            if dist < min_dist:
                min_dist = dist
                closest_player = p

        # Labdabirtoklás küszöb: 1.5 méter
        if closest_player and min_dist <= 1.5:
            tid = closest_player['track_id']
            team = f"team{closest_player['team_id']}"

            if not self.possession_history:
                self.possession_history.append((tid, team, timestamp))
            else:
                last_tid, last_team, last_time = self.possession_history[-1]
                if last_team == team and last_tid != tid:
                    if (timestamp - last_time) > 0.5:
                        self.stats["passes"][team] += 1
                        self.possession_history.append((tid, team, timestamp))
                elif last_team != team:
                    self.stats["tackles"][team] += 1
                    self.possession_history.append((tid, team, timestamp))

        # A lövés/gól/kapus események detektálása az EventDetector-ban történik (events.py)

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
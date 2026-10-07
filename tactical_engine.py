import cv2
import numpy as np
import math

class TacticalEngine:
    def __init__(self, pitch_length=105.0, pitch_width=68.0):
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

        self.stats = {
            "passes": {"team1": 0, "team2": 0},
            "tackles": {"team1": 0, "team2": 0},
            "shots": {"team1": 0, "team2": 0},
            "shots_on_target": {"team1": 0, "team2": 0}
        }

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

        # Lövés detektálása a labda sebességvektora alapján (> 45 km/h)
        if len(self.ball_trajectory) >= 3:
            prev_b = self.ball_trajectory[-3]
            dt = timestamp - prev_b[2]
            if dt > 0:
                speed_mps = math.hypot(bx - prev_b[0], by - prev_b[1]) / dt
                if (speed_mps * 3.6) > 45.0:
                    if (prev_b[0] - bx) > 2.0:
                        slope = (by - prev_b[1]) / (bx - prev_b[0] + 1e-5)
                        intersect_y = by - slope * bx
                        if 30.0 <= intersect_y <= 38.0:
                            self.stats["shots_on_target"]["team2"] += 1
                        else:
                            self.stats["shots"]["team2"] += 1

    def analyze_goalkeeper(self, gk_coords, ball_coords):
        """Kiszámítja a kapus szögfelezőjét és az ideális pozíciótól mért eltérést méterben."""
        if gk_coords is None or ball_coords is None:
            return None

        gx, gy = gk_coords
        bx, by = ball_coords

        vec_left = self.goal_left_post - np.array([bx, by])
        vec_right = self.goal_right_post - np.array([bx, by])

        norm_l = vec_left / np.linalg.norm(vec_left)
        norm_r = vec_right / np.linalg.norm(vec_right)
        bisector = (norm_l + norm_r) / np.linalg.norm(norm_l + norm_r)

        line_pt1 = np.array([bx, by])
        line_pt2 = np.array([bx, by]) + bisector * 50.0

        p1_to_gk = np.array([gx, gy]) - line_pt1
        p1_to_p2 = line_pt2 - line_pt1
        cross_prod = np.cross(p1_to_p2, p1_to_gk)
        offset_meters = cross_prod / np.linalg.norm(p1_to_p2)

        return {
            "dist_from_goal_m": round(float(gx), 2),
            "bisector_offset_m": round(float(offset_meters), 2),
            "gk_coords": (gx, gy),
            "ball_coords": (bx, by)
        }
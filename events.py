"""
Mérkőzés-esemény detektálás (lövés, gól, kapus események) tisztán Python alapon.

Bemenet képkockánként: időbélyeg (mp), a labda pálya-koordinátája méterben (vagy None),
valamint a játékosok listája (track_id, team_id, is_keeper, pitch_coords).
Koordináta-rendszer: 105x68 m-es pálya, x a hossz mentén (bal kapu x=0, jobb kapu x=105),
kapufák: y = 30.34 .. 37.66.

FONTOS (egykamerás korlát): a labda magasságát nem ismerjük. A levegőben lévő labda
talajra vetített pozíciója torzul, ezért a sebesség/irány becslés ívelt labdáknál pontatlan,
és az ívelés/rövid passz megkülönböztetése KIZÁRÓLAG a megtett távolság alapján történik.

A modul nem függ torch/ultralytics/cv2 csomagtól, így önállóan tesztelhető.
"""

import math
from collections import deque

TEAM_LABELS = {"team1": "1. csapat", "team2": "2. csapat"}
KEEPER_LABELS = {"team1": "Hazai kapus", "team2": "Vendég kapus"}

GK_EVENT_TYPES = (
    "gk_save", "gk_claim_cross", "gk_goal_kick", "gk_long_ball",
    "gk_short_pass", "gk_conceded", "gk_sweeper", "gk_positioning_error",
)

_NEG_INF = float("-inf")


def other_team(team):
    return "team2" if team == "team1" else "team1"


def _team_label(team):
    return TEAM_LABELS.get(team, str(team))


def _keeper_label(team, tid=None):
    base = KEEPER_LABELS.get(team, "Kapus")
    return f"{base} (ID #{tid})" if tid is not None else base


def fit_velocity(samples):
    """Legkisebb négyzetes egyenes illesztés (t, x, y) mintákra -> (vx, vy) m/s, vagy None."""
    n = len(samples)
    if n < 2:
        return None
    tm = sum(s[0] for s in samples) / n
    xm = sum(s[1] for s in samples) / n
    ym = sum(s[2] for s in samples) / n
    den = sum((s[0] - tm) ** 2 for s in samples)
    if den <= 1e-9:
        return None
    vx = sum((s[0] - tm) * (s[1] - xm) for s in samples) / den
    vy = sum((s[0] - tm) * (s[2] - ym) for s in samples) / den
    return vx, vy


class EventConfig:
    """Az eseménydetektálás hangolható paraméterei. Bármelyik felülírható kulcsszavas argumentummal."""

    def __init__(self, **overrides):
        # --- Labda sebesség becslés ---
        self.vel_window_s = 0.3            # simító ablak (mp) a sebesség-regresszióhoz
        self.vel_min_samples = 3           # minimum minta a becsléshez
        self.vel_max_span_s = 0.6          # ritka minták esetén ennyi mp-nél hosszabb ablakot nem használunk
        self.max_ball_speed_mps = 40.0     # efölötti ugrás téves labda-detektálásnak minősül (~144 km/h)

        # --- Lövés ---
        self.shot_speed_kmh = 45.0         # lövés minimális labdasebessége
        self.shot_max_distance_m = 35.0    # ennél messzebbről a gólvonaltól nem tekintjük lövésnek
        self.shot_min_x_ratio = 0.5        # |vx|/sebesség minimum (beadások kiszűrése, ~60° max. szög)
        self.post_tolerance_m = 0.5        # kapufa-tűrés a kaput eltaláló lövéshez
        self.off_target_band_m = 12.0      # kapu mellé tartó lövés: max. ennyi m-re a kapu közepétől
        self.shot_debounce_s = 1.5         # egy lövés után ennyi ideig nincs új lövés
        self.team1_attacks_right = True    # alapértelmezés: az 1. csapat a x=105 kapura támad

        # --- Gól ---
        self.goal_debounce_s = 10.0
        self.goal_line_margin_m = 0.3      # ennyivel a gólvonal mögött kell lennie a labdának

        # --- Labdabirtoklás ---
        self.possession_radius_m = 1.5
        self.possession_max_ball_speed_mps = 10.0
        self.possession_memory_s = 4.0     # ennél régebbi birtoklás alapján nem írunk jóvá lövést

        # --- Védés ---
        self.save_contact_radius_m = 2.0
        self.save_loose_radius_m = 3.0     # ezen belül megállás/irányváltás is védésnek számít
        self.save_confirm_s = 2.0          # ennyi ideig nem eshet gól a kontakt után
        self.shot_resolve_window_s = 3.0   # a lövés eddig "él" (gól/védés hozzárendelés)

        # --- Beadás elfogása ---
        self.wide_zone_y = 14.0            # szélső sáv: y < 14 vagy y > 54
        self.cross_window_s = 3.0
        self.cross_max_line_dist_m = 40.0  # a beadás a gólvonaltól ennyin belül induljon

        # --- Kirúgás (kapukirúgás) ---
        self.goal_kick_still_s = 1.0
        self.goal_kick_still_radius_m = 0.6
        self.goal_kick_min_speed_mps = 8.0
        self.goal_kick_launch_dist_m = 3.0
        self.goal_kick_out_memory_s = 45.0

        # --- Kapus labdakihozatal ---
        self.long_ball_m = 35.0            # ennél hosszabb út = ívelés (csak távolság alapján!)
        self.min_distribution_m = 4.0
        self.distribution_timeout_s = 6.0
        self.gk_keep_radius_m = 2.5
        self.gk_release_radius_m = 3.0

        # --- Kifutás ---
        self.sweeper_speed_mps = 4.0
        self.sweeper_min_line_dist_m = 12.0
        self.sweeper_debounce_s = 6.0
        self.keeper_max_speed_mps = 10.0   # efölött track-ID csere / zaj

        # --- Klip javaslat ---
        self.clip_pre_s = 4.0
        self.clip_post_s = 3.0

        for k, v in overrides.items():
            if not hasattr(self, k):
                raise ValueError(f"Ismeretlen esemény-konfigurációs kulcs: {k}")
            setattr(self, k, v)


class EventDetector:
    """Állapotgépes eseménydetektor. Képkockánként hívandó: update(t, ball_xy, players)."""

    def __init__(self, config=None, pitch_length=105.0, pitch_width=68.0):
        self.cfg = config or EventConfig()
        self.L = float(pitch_length)
        self.W = float(pitch_width)
        self.post_lo = self.W / 2.0 - 3.66
        self.post_hi = self.W / 2.0 + 3.66

        self.events = []
        self.stats = {
            "shots": {"team1": 0, "team2": 0},            # összes lövés (kaput eltalálók is)
            "shots_on_target": {"team1": 0, "team2": 0},
            "goals": {"team1": 0, "team2": 0},
            "gk_events": {"team1": {}, "team2": {}},
        }

        self._ball = deque()             # (t, x, y)
        self._ball_rejects = 0
        self._keepers = {}               # tid -> {team, hist, pos, last_t, last_sweep}
        self._poss = None                # (tid, team, is_keeper, t)
        self._last_shot_t = _NEG_INF
        self._shot_dir = None            # az utolsó lövés iránya, amíg a labda gyors marad
        self._last_goal_t = _NEG_INF
        self._pending_shot = None
        self._still = None
        self._gk_armed = None
        self._byline_out = {}            # goal_x -> t
        self._last_wide = None           # (t, x, y)
        self._gk_poss = None
        self._dist = None
        self._new = []
        self._last_t = None

    # ------------------------------------------------------------------ API
    def set_attack_direction(self, team1_attacks_right):
        """Félidőnként fordítható támadási irány (csak akkor számít, ha a kapusok nem azonosíthatók)."""
        self.cfg.team1_attacks_right = bool(team1_attacks_right)

    def update(self, timestamp, ball_xy, players):
        t = float(timestamp)
        self._new = []
        players = [p for p in (players or []) if p.get("pitch_coords") is not None]

        self._update_keepers(t, players)

        ball = None
        if ball_xy is not None:
            bx, by = float(ball_xy[0]), float(ball_xy[1])
            if self._add_ball(t, bx, by):
                ball = (bx, by)

        vel = self._ball_velocity(t)
        speed = math.hypot(vel[0], vel[1]) if vel else None
        closest, cdist = self._closest_player(ball, players)

        if ball is not None:
            self._update_possession(t, ball, speed, closest, cdist)
            self._track_zones(t, ball)
            self._check_goal(t, ball)
            if vel is not None:
                self._check_shot(t, ball, vel, speed)

        self._resolve_shot(t, ball, vel, speed)
        self._check_goal_kick(t, ball, players)
        self._update_gk_possession(t, ball, speed, closest, cdist)
        self._check_sweeper(t)

        self._last_t = t
        return self._new

    def finalize(self, timestamp=None):
        """A klip végén lezárja a függő eseményeket (megerősítetlen védés, folyamatban lévő kihozatal)."""
        self._new = []
        t = timestamp if timestamp is not None else (self._last_t or 0.0)
        ps = self._pending_shot
        if ps and ps["contact"] is not None:
            self._emit_save(ps)
        self._pending_shot = None
        if self._dist is not None:
            self._finish_distribution()
        self._gk_poss = None
        self._last_t = t
        return self._new

    def defending_team(self, goal_x, t=None):
        """Az adott kaput védő csapat: elsődlegesen a kapus-trackek alapján, különben a konfigurált irány szerint."""
        votes = {}
        for k in self._keepers.values():
            if t is not None and t - k["last_t"] > 60.0:
                continue
            if self._keeper_goal(k) == goal_x:
                votes[k["team"]] = votes.get(k["team"], 0) + len(k["hist"])
        if votes:
            return max(votes.items(), key=lambda kv: kv[1])[0]
        left_defender = "team1" if self.cfg.team1_attacks_right else "team2"
        return left_defender if goal_x == 0.0 else other_team(left_defender)

    def keeper_team_near(self, coords, t=None, max_dist=5.0):
        """A megadott pozícióhoz legközelebbi ismert kapus csapata (vagy None)."""
        if coords is None:
            return None
        best, best_d = None, max_dist
        for k in self._keepers.values():
            if k.get("pos") is None:
                continue
            if t is not None and t - k["last_t"] > 1.0:
                continue
            d = math.hypot(k["pos"][0] - coords[0], k["pos"][1] - coords[1])
            if d <= best_d:
                best, best_d = k["team"], d
        return best

    # ------------------------------------------------------------ segédek
    def _emit(self, ev):
        t = ev["time"]
        ev["time"] = round(float(t), 2)
        ev.setdefault("start_sec", round(max(0.0, t - self.cfg.clip_pre_s), 2))
        ev.setdefault("end_sec", round(t + self.cfg.clip_post_s, 2))
        self.events.append(ev)
        self._new.append(ev)
        team = ev.get("team")
        if ev["type"].startswith("gk_") and team in self.stats["gk_events"]:
            bucket = self.stats["gk_events"][team]
            bucket[ev["type"]] = bucket.get(ev["type"], 0) + 1
        return ev

    def _goal_xs(self):
        return (0.0, self.L)

    def _in_penalty_area(self, p, goal_x):
        return abs(p[0] - goal_x) <= 16.5 and abs(p[1] - self.W / 2.0) <= 20.16

    def _goal_area_of(self, p):
        for gx in self._goal_xs():
            if abs(p[0] - gx) <= 5.5 and abs(p[1] - self.W / 2.0) <= 9.16:
                return gx
        return None

    def _in_wide_zone(self, p):
        return p[1] < self.cfg.wide_zone_y or p[1] > self.W - self.cfg.wide_zone_y

    def _add_ball(self, t, x, y):
        cfg = self.cfg
        if self._ball:
            pt, px, py = self._ball[-1]
            dt = t - pt
            if dt <= 0:
                return False
            if dt < 1.0 and math.hypot(x - px, y - py) / dt > cfg.max_ball_speed_mps:
                # Valószínűleg téves labda-detektálás; 3 egymás utáni eltérés után elfogadjuk az új pozíciót
                self._ball_rejects += 1
                if self._ball_rejects < 3:
                    return False
                self._ball.clear()
        self._ball_rejects = 0
        self._ball.append((t, x, y))
        while self._ball and t - self._ball[0][0] > 5.0:
            self._ball.popleft()
        return True

    def _ball_velocity(self, t):
        cfg = self.cfg
        if not self._ball or t - self._ball[-1][0] > 0.5:
            return None
        t_last = self._ball[-1][0]
        win = [s for s in self._ball if t_last - s[0] <= cfg.vel_window_s + 1e-9]
        if len(win) < cfg.vel_min_samples:
            win = list(self._ball)[-cfg.vel_min_samples:]
            if len(win) < cfg.vel_min_samples or win[-1][0] - win[0][0] > cfg.vel_max_span_s:
                return None
        return fit_velocity(win)

    def _closest_player(self, ball, players):
        if ball is None:
            return None, None
        best, best_d = None, None
        for p in players:
            px, py = p["pitch_coords"]
            d = math.hypot(px - ball[0], py - ball[1])
            if best_d is None or d < best_d:
                best, best_d = p, d
        return best, best_d

    def _update_keepers(self, t, players):
        for p in players:
            if not p.get("is_keeper"):
                continue
            tid = p.get("track_id")
            if tid is None:
                continue
            x, y = float(p["pitch_coords"][0]), float(p["pitch_coords"][1])
            k = self._keepers.get(tid)
            if k is None:
                k = {"hist": deque(), "last_sweep": _NEG_INF}
                self._keepers[tid] = k
            k["team"] = f"team{p.get('team_id', 1)}"
            k["hist"].append((t, x, y))
            while k["hist"] and t - k["hist"][0][0] > 10.0:
                k["hist"].popleft()
            k["pos"] = (x, y)
            k["last_t"] = t

    def _keeper_goal(self, k):
        hist = k["hist"]
        if not hist:
            return None
        mean_x = sum(h[1] for h in hist) / len(hist)
        return 0.0 if mean_x < self.L / 2.0 else self.L

    def _active_keepers(self, t, goal_x=None):
        out = []
        for tid, k in self._keepers.items():
            if t - k["last_t"] > 0.5:
                continue
            if goal_x is not None and self._keeper_goal(k) != goal_x:
                continue
            out.append((tid, k))
        return out

    def _nearest_keeper_tid(self, t, goal_x, pos):
        best, best_d = None, None
        for tid, k in self._active_keepers(t, goal_x):
            d = math.hypot(k["pos"][0] - pos[0], k["pos"][1] - pos[1])
            if best_d is None or d < best_d:
                best, best_d = tid, d
        return best

    # --------------------------------------------------------- birtoklás
    def _update_possession(self, t, ball, speed, closest, cdist):
        cfg = self.cfg
        if closest is None or cdist > cfg.possession_radius_m:
            return
        if speed is not None and speed > cfg.possession_max_ball_speed_mps:
            return
        team = f"team{closest.get('team_id', 1)}"
        is_keeper = bool(closest.get("is_keeper"))
        self._poss = (closest.get("track_id"), team, is_keeper, t)
        # Ha mezőnyjátékos a szélső sávon kívül átveszi a labdát, az már nem közvetlen beadás
        if not is_keeper and not self._in_wide_zone(ball):
            self._last_wide = None

    def _track_zones(self, t, ball):
        cfg = self.cfg
        x, y = ball
        if self._in_wide_zone(ball):
            self._last_wide = (t, x, y)
        in_posts = self.post_lo <= y <= self.post_hi
        if not in_posts:
            if x < -cfg.goal_line_margin_m:
                self._byline_out[0.0] = t
            elif x > self.L + cfg.goal_line_margin_m:
                self._byline_out[self.L] = t
        for gx in list(self._byline_out.keys()):
            if abs(x - gx) > 20.0:
                del self._byline_out[gx]

    # --------------------------------------------------------------- gól
    def _check_goal(self, t, ball):
        cfg = self.cfg
        x, y = ball
        if not (self.post_lo <= y <= self.post_hi):
            return
        if x < -cfg.goal_line_margin_m:
            goal_x = 0.0
        elif x > self.L + cfg.goal_line_margin_m:
            goal_x = self.L
        else:
            return
        if t - self._last_goal_t < cfg.goal_debounce_s:
            return
        self._last_goal_t = t

        defending = self.defending_team(goal_x, t)
        scoring = other_team(defending)

        ps = self._pending_shot
        if ps is not None and ps["goal_x"] == goal_x and t - ps["t"] <= cfg.shot_resolve_window_s:
            shot_ev = ps["ev"]
            if not shot_ev["on_target"]:
                # A gól utólag igazolja, hogy a lövés kaput talált
                shot_ev["on_target"] = True
                shot_ev["desc"] = shot_ev["desc"].replace("Kapu mellé tartó lövés", "Kaput eltaláló lövés")
                self.stats["shots_on_target"][shot_ev["team"]] += 1
        else:
            # Gól észlelt lövés nélkül (pl. lassú befejezés): pótlólag lövésként is rögzítjük
            self.stats["shots"][scoring] += 1
            self.stats["shots_on_target"][scoring] += 1
            self._emit({
                "type": "shot", "time": t, "team": scoring, "on_target": True, "inferred": True,
                "goal_x": goal_x,
                "desc": f"Kaput eltaláló lövés – {_team_label(scoring)} (a gól alapján utólag rögzítve)",
            })
        self._pending_shot = None

        self.stats["goals"][scoring] += 1
        self._emit({
            "type": "goal", "time": t, "team": scoring, "goal_x": goal_x,
            "start_sec": round(max(0.0, t - 8.0), 2), "end_sec": round(t + 4.0, 2),
            "desc": f"Valószínű GÓL – {_team_label(scoring)} (a labda a kapufák között átjutott a gólvonalon)",
        })
        ktid = self._nearest_keeper_tid(t, goal_x, ball)
        self._emit({
            "type": "gk_conceded", "time": t, "team": defending, "keeper_track_id": ktid, "goal_x": goal_x,
            "start_sec": round(max(0.0, t - 8.0), 2), "end_sec": round(t + 3.0, 2),
            "desc": f"Kapott gól – {_keeper_label(defending, ktid)}",
        })
        # A gól minden folyamatban lévő kapus-szekvenciát lezár
        self._gk_poss = None
        self._dist = None
        self._gk_armed = None

    # ------------------------------------------------------------ lövés
    def _check_shot(self, t, ball, vel, speed):
        cfg = self.cfg
        vx, vy = vel
        if speed * 3.6 < cfg.shot_speed_kmh:
            self._shot_dir = None
            return
        if self._shot_dir is not None and t - self._last_shot_t > 4.0:
            self._shot_dir = None
        if self._shot_dir is not None and vx * self._shot_dir > 0:
            return  # ugyanaz a lövés még repül (nem számoljuk többször)
        self._shot_dir = None
        if t - self._last_shot_t < cfg.shot_debounce_s:
            return
        bx, by = ball
        if bx < 0.0 or bx > self.L:
            return
        if abs(vx) / speed < cfg.shot_min_x_ratio:
            return
        goal_x = self.L if vx > 0 else 0.0
        line_dist = abs(goal_x - bx)
        if line_dist > cfg.shot_max_distance_m:
            return
        t_cross = (goal_x - bx) / vx
        if t_cross <= 0:
            return
        y_cross = by + vy * t_cross
        on_target = (self.post_lo - cfg.post_tolerance_m) <= y_cross <= (self.post_hi + cfg.post_tolerance_m)
        if not on_target and abs(y_cross - self.W / 2.0) > cfg.off_target_band_m:
            return  # valószínűleg hosszú passz / beadás, nem lövés

        poss = self._poss
        shooter = None
        if poss is not None and t - poss[3] <= cfg.possession_memory_s:
            team, shooter = poss[1], poss[0]
        else:
            team = other_team(self.defending_team(goal_x, t))

        self._last_shot_t = t
        self._shot_dir = 1.0 if vx > 0 else -1.0
        self.stats["shots"][team] += 1
        if on_target:
            self.stats["shots_on_target"][team] += 1
        kmh = speed * 3.6
        kind = "Kaput eltaláló lövés" if on_target else "Kapu mellé tartó lövés"
        ev = self._emit({
            "type": "shot", "time": t, "team": team, "on_target": bool(on_target),
            "speed_kmh": round(kmh, 1), "goal_x": goal_x, "shooter_track_id": shooter,
            "distance_m": round(math.hypot(goal_x - bx, self.W / 2.0 - by), 1),
            "desc": f"{kind} – {_team_label(team)}, {kmh:.0f} km/h, "
                    f"{math.hypot(goal_x - bx, self.W / 2.0 - by):.0f} m-ről",
        })
        self._pending_shot = {
            "ev": ev, "goal_x": goal_x, "t": t, "contact": None,
            "dir": 1.0 if vx > 0 else -1.0, "defending": self.defending_team(goal_x, t),
        }

    def _resolve_shot(self, t, ball, vel, speed):
        cfg = self.cfg
        ps = self._pending_shot
        if ps is None:
            return
        if ps["contact"] is None:
            if t - ps["t"] > cfg.shot_resolve_window_s:
                self._pending_shot = None
                return
            if ball is None or not ps["ev"]["on_target"]:
                return
            best = None
            for tid, k in self._active_keepers(t, ps["goal_x"]):
                d = math.hypot(k["pos"][0] - ball[0], k["pos"][1] - ball[1])
                if best is None or d < best[0]:
                    best = (d, tid, k["team"])
            if best is None:
                return
            d, tid, team = best
            reversed_dir = vel is not None and vel[0] * ps["dir"] < 0
            stopped = speed is not None and speed < 3.0
            if d <= cfg.save_contact_radius_m or (d <= cfg.save_loose_radius_m and (reversed_dir or stopped)):
                ps["contact"] = (t, tid, team)
        elif t - ps["contact"][0] >= cfg.save_confirm_s:
            self._emit_save(ps)
            self._pending_shot = None

    def _emit_save(self, ps):
        ct, tid, team = ps["contact"]
        shot_ev = ps["ev"]
        self._emit({
            "type": "gk_save", "time": ct, "team": team, "keeper_track_id": tid,
            "shot_time": shot_ev["time"], "goal_x": ps["goal_x"],
            "desc": f"Kapus védés – {_keeper_label(team, tid)}, "
                    f"{_team_label(shot_ev['team'])} lövése ({shot_ev.get('speed_kmh', '?')} km/h)",
        })

    # -------------------------------------------------------- kirúgás
    def _check_goal_kick(self, t, ball, players):
        cfg = self.cfg
        if ball is None:
            return
        armed = self._gk_armed
        if armed is not None:
            d = math.hypot(ball[0] - armed["x"], ball[1] - armed["y"])
            if d >= cfg.goal_kick_launch_dist_m:
                dt = max(t - armed["t_still"], 1e-3)
                self._gk_armed = None
                self._still = None
                if d / dt >= cfg.goal_kick_min_speed_mps:
                    self._start_goal_kick(t, armed)
                return
            if d > cfg.goal_kick_still_radius_m:
                if t - armed["t_still"] > 2.0:
                    self._gk_armed = None  # lassan elgurult, nem kirúgás
                return  # elindult, várjuk a 3 m-es távolságot
            armed["t_still"] = t
            return

        st = self._still
        if st is not None and math.hypot(ball[0] - st["x"], ball[1] - st["y"]) <= cfg.goal_kick_still_radius_m:
            st["t_last"] = t
        else:
            self._still = st = {"t0": t, "t_last": t, "x": ball[0], "y": ball[1]}

        if st["t_last"] - st["t0"] < cfg.goal_kick_still_s:
            return
        goal_x = self._goal_area_of((st["x"], st["y"]))
        if goal_x is None:
            return
        recent_out = t - self._byline_out.get(goal_x, _NEG_INF) <= cfg.goal_kick_out_memory_s
        player_near = any(
            math.hypot(p["pitch_coords"][0] - st["x"], p["pitch_coords"][1] - st["y"]) <= 1.5
            for p in players
        )
        # A kapus kezében tartott labdát (közel a kapushoz, nem volt alapvonalon túl) nem tekintjük kirúgásnak
        if recent_out or not player_near:
            self._gk_armed = {"goal_x": goal_x, "x": st["x"], "y": st["y"], "t_still": st["t_last"]}

    def _start_goal_kick(self, t, armed):
        goal_x = armed["goal_x"]
        team = self.defending_team(goal_x, t)
        tid = self._nearest_keeper_tid(t, goal_x, (armed["x"], armed["y"]))
        self._byline_out.pop(goal_x, None)
        self._gk_poss = None
        self._dist = {
            "kind": "goal_kick", "tid": tid, "team": team, "goal_x": goal_x,
            "t0": armed["t_still"], "p0": (armed["x"], armed["y"]), "max_d": 0.0,
            "last_t": t, "still_t": None,
        }

    # ------------------------------------------- kapus birtoklás / kihozatal
    def _update_gk_possession(self, t, ball, speed, closest, cdist):
        cfg = self.cfg
        if self._dist is not None:
            self._advance_distribution(t, ball, speed, closest, cdist)

        gp = self._gk_poss
        if ball is None:
            if gp is not None and t - gp["t_last"] > 1.5:
                self._gk_poss = None
            return

        slow = speed is None or speed <= cfg.possession_max_ball_speed_mps
        if closest is not None and closest.get("is_keeper") and cdist <= cfg.possession_radius_m and slow:
            tid = closest.get("track_id")
            if gp is None or gp["tid"] != tid:
                k = self._keepers.get(tid)
                goal_x = self._keeper_goal(k) if k else (0.0 if ball[0] < self.L / 2 else self.L)
                team = k["team"] if k else f"team{closest.get('team_id', 1)}"
                d = self._dist
                if d is not None and d["kind"] == "gk" and d["tid"] == tid:
                    self._dist = None  # a kapus visszaszerezte, nem volt valódi kihozatal
                self._gk_poss = {"tid": tid, "team": team, "goal_x": goal_x,
                                 "t_start": t, "t_last": t, "ball": ball}
                self._check_cross_claim(t, ball, tid, team, goal_x)
            else:
                gp["t_last"] = t
                gp["ball"] = ball
            return

        if gp is None:
            return
        k = self._keepers.get(gp["tid"])
        ref = k["pos"] if (k and t - k["last_t"] <= 0.5) else gp["ball"]
        dk = math.hypot(ball[0] - ref[0], ball[1] - ref[1])
        if dk <= cfg.gk_keep_radius_m:
            gp["t_last"] = t
            gp["ball"] = ball
        elif dk > cfg.gk_release_radius_m:
            self._gk_poss = None
            if self._dist is None:
                self._dist = {
                    "kind": "gk", "tid": gp["tid"], "team": gp["team"], "goal_x": gp["goal_x"],
                    "t0": gp["t_last"], "p0": gp["ball"], "max_d": 0.0, "last_t": t, "still_t": None,
                }
                self._advance_distribution(t, ball, speed, closest, cdist)

    def _advance_distribution(self, t, ball, speed, closest, cdist):
        cfg = self.cfg
        d = self._dist
        if ball is not None:
            dd = math.hypot(ball[0] - d["p0"][0], ball[1] - d["p0"][1])
            d["max_d"] = max(d["max_d"], dd)
            d["last_t"] = t
            slow = speed is None or speed <= cfg.possession_max_ball_speed_mps
            if (closest is not None and cdist <= cfg.possession_radius_m and slow
                    and closest.get("track_id") != d["tid"] and t - d["t0"] > 0.2):
                self._finish_distribution()
                return
            if speed is not None and speed < 1.0:
                if d["still_t"] is None:
                    d["still_t"] = t
                elif t - d["still_t"] >= 0.5:
                    self._finish_distribution()
                    return
            else:
                d["still_t"] = None
        if t - d["t0"] > cfg.distribution_timeout_s or (ball is None and t - d["last_t"] > 1.5):
            self._finish_distribution()

    def _finish_distribution(self):
        cfg = self.cfg
        d = self._dist
        self._dist = None
        dist = d["max_d"]
        if d["kind"] == "gk" and dist < cfg.min_distribution_m:
            return
        # Egykamerás rendszer: nincs labdamagasság, ezért az ívelés/rövid passz csak a távolságból dől el
        is_long = dist > cfg.long_ball_m
        team, tid = d["team"], d["tid"]
        if d["kind"] == "goal_kick":
            kind = "hosszú" if is_long else "rövid"
            self._emit({
                "type": "gk_goal_kick", "time": d["t0"], "team": team, "keeper_track_id": tid,
                "distance_m": round(dist, 1), "is_long": is_long, "goal_x": d["goal_x"],
                "desc": f"Kirúgás ({kind}, {dist:.0f} m) – {_keeper_label(team, tid)}",
            })
        elif is_long:
            self._emit({
                "type": "gk_long_ball", "time": d["t0"], "team": team, "keeper_track_id": tid,
                "distance_m": round(dist, 1), "is_long": True, "goal_x": d["goal_x"],
                "desc": f"Kapus ívelés ({dist:.0f} m) – {_keeper_label(team, tid)}",
            })
        else:
            self._emit({
                "type": "gk_short_pass", "time": d["t0"], "team": team, "keeper_track_id": tid,
                "distance_m": round(dist, 1), "is_long": False, "goal_x": d["goal_x"],
                "desc": f"Kapus rövid passz / kidobás ({dist:.0f} m) – {_keeper_label(team, tid)}",
            })

    def _check_cross_claim(self, t, ball, tid, team, goal_x):
        cfg = self.cfg
        lw = self._last_wide
        if lw is None or t - lw[0] > cfg.cross_window_s:
            return
        if not self._in_penalty_area(ball, goal_x) or self._in_wide_zone(ball):
            return
        if abs(lw[1] - goal_x) > cfg.cross_max_line_dist_m:
            return
        self._last_wide = None
        self._emit({
            "type": "gk_claim_cross", "time": t, "team": team, "keeper_track_id": tid, "goal_x": goal_x,
            "desc": f"Beadás elfogása – {_keeper_label(team, tid)}",
        })

    # ------------------------------------------------------------ kifutás
    def _check_sweeper(self, t):
        cfg = self.cfg
        for tid, k in self._keepers.items():
            if k["last_t"] != t:
                continue
            win = [h for h in k["hist"] if t - h[0] <= 0.6 + 1e-9]
            if len(win) < 4:
                continue
            vel = fit_velocity(win)
            if vel is None:
                continue
            sp = math.hypot(vel[0], vel[1])
            if sp < cfg.sweeper_speed_mps or sp > cfg.keeper_max_speed_mps:
                continue
            goal_x = self._keeper_goal(k)
            x, y = k["pos"]
            if abs(x - goal_x) < cfg.sweeper_min_line_dist_m:
                continue
            rx, ry = x - goal_x, y - self.W / 2.0
            rn = math.hypot(rx, ry)
            if rn <= 1e-6 or (vel[0] * rx + vel[1] * ry) / rn < 0.3 * sp:
                continue  # nem a kaputól távolodik (pl. visszafut)
            if t - k["last_sweep"] < cfg.sweeper_debounce_s:
                continue
            k["last_sweep"] = t
            self._emit({
                "type": "gk_sweeper", "time": t, "team": k["team"], "keeper_track_id": tid,
                "goal_x": goal_x, "speed_mps": round(sp, 1),
                "distance_from_goal_m": round(abs(x - goal_x), 1),
                "desc": f"Kapus kifutás – {_keeper_label(k['team'], tid)}, "
                        f"{abs(x - goal_x):.0f} m-re a gólvonaltól, {sp:.1f} m/s",
            })

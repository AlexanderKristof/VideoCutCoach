"""
Szintetikus labda/játékos pályákkal tesztelt eseménydetektálás.
Futtatás: python3 tests/test_events.py   (pytest nem szükséges, de azzal is fut)
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from events import EventDetector, EventConfig  # noqa: E402

FPS = 25.0
DT = 1.0 / FPS


def P(tid, team, x, y, keeper=False):
    return {"track_id": tid, "team_id": team, "is_keeper": keeper, "pitch_coords": (float(x), float(y))}


class Sim:
    def __init__(self, **cfg):
        self.det = EventDetector(EventConfig(**cfg))
        self.t = 0.0

    def step(self, ball, players):
        self.det.update(self.t, ball, players)
        self.t += DT

    def hold(self, seconds, ball, players):
        """Álló labda / játékosok adott ideig."""
        for _ in range(int(round(seconds * FPS))):
            self.step(ball, players)

    def move(self, start, end, speed, players_fn=None, stop_at_end=True):
        """A labda egyenes vonalon, állandó sebességgel start -> end. Visszaadja a végpontot."""
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        n = max(1, int(math.ceil(dist / speed * FPS)))
        for i in range(1, n + 1):
            f = i / n
            b = (start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f)
            self.step(b, players_fn(b) if players_fn else [])
        return end

    def finalize(self):
        self.det.finalize(self.t)

    def of_type(self, typ):
        return [e for e in self.det.events if e["type"] == typ]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ----------------------------------------------------------------- lövések
def test_shot_on_target_right_goal():
    s = Sim()
    att = P(7, 1, 84.5, 34)
    s.hold(1.0, (85.0, 34.0), [att])
    s.move((85.0, 34.0), (104.0, 35.0), 25.0, lambda b: [att])
    s.step(None, [att])
    shots = s.of_type("shot")
    check(len(shots) == 1, f"pontosan 1 lövés kell, kaptunk: {shots}")
    check(shots[0]["team"] == "team1" and shots[0]["on_target"], shots[0])
    check(s.det.stats["shots"]["team1"] == 1 and s.det.stats["shots_on_target"]["team1"] == 1, s.det.stats)
    check(shots[0]["start_sec"] < shots[0]["time"] < shots[0]["end_sec"], shots[0])


def test_shot_on_target_left_goal():
    s = Sim()
    att = P(9, 2, 20.5, 36)
    s.hold(1.0, (20.0, 36.0), [att])
    s.move((20.0, 36.0), (1.0, 33.0), 28.0, lambda b: [att])
    shots = s.of_type("shot")
    check(len(shots) == 1, shots)
    check(shots[0]["team"] == "team2" and shots[0]["on_target"] and shots[0]["goal_x"] == 0.0, shots[0])


def test_shot_off_target():
    s = Sim()
    att = P(7, 1, 84.5, 34)
    s.hold(1.0, (85.0, 34.0), [att])
    s.move((85.0, 34.0), (104.0, 43.5), 25.0, lambda b: [att])
    shots = s.of_type("shot")
    check(len(shots) == 1 and not shots[0]["on_target"], shots)
    check(s.det.stats["shots"]["team1"] == 1 and s.det.stats["shots_on_target"]["team1"] == 0, s.det.stats)


def test_long_pass_not_shot():
    s = Sim()
    p1 = P(4, 1, 29.5, 34)
    s.hold(1.0, (30.0, 34.0), [p1])
    s.move((30.0, 34.0), (75.0, 10.0), 25.0)          # hosszú passz a saját térfélről
    s.hold(1.0, (75.0, 10.0), [P(11, 1, 75.0, 10.5)])
    s.move((75.0, 10.0), (100.0, 62.0), 25.0)         # keresztlabda a túlsó szélre
    s.hold(1.0, (100.0, 62.0), [])
    s.move((88.0, 5.0), (100.0, 33.0), 20.0)          # beadás a szélről (szög miatt nem lövés)
    check(not s.of_type("shot"), s.of_type("shot"))


def test_shot_debounce_and_two_shots():
    s = Sim()
    att = P(7, 1, 84.5, 34)
    s.hold(1.0, (85.0, 34.0), [att])
    s.move((85.0, 34.0), (100.0, 34.8), 25.0, lambda b: [att])
    # kipattanó, majd azonnali újabb lövés 1.5 mp-en belül -> nem új lövés (debounce)
    s.move((100.0, 34.8), (92.0, 34.0), 20.0)
    s.move((92.0, 34.0), (104.0, 36.0), 25.0)
    check(len(s.of_type("shot")) == 1, s.of_type("shot"))
    # 3 mp múlva új lövés -> 2 lövés
    s.hold(3.0, (88.0, 30.0), [P(8, 1, 88.5, 30)])
    s.move((88.0, 30.0), (104.0, 33.0), 25.0)
    check(len(s.of_type("shot")) == 2, s.of_type("shot"))


def test_slow_long_shot_counted_once():
    # 35 km/h feletti, de hosszan repülő lövés (2+ mp) -> csak egyszer számít
    s = Sim()
    att = P(7, 1, 71.5, 34)
    s.hold(1.0, (72.0, 34.0), [att])
    s.move((72.0, 34.0), (104.0, 34.0), 14.0, lambda b: [att])
    check(len(s.of_type("shot")) == 1, s.of_type("shot"))


def test_attack_direction_fallback_and_flip():
    # Nincs birtoklás és nincs kapus -> irány alapján: alapból az 1. csapat támad x=105 felé
    s = Sim()
    s.hold(0.5, (80.0, 34.0), [])
    s.move((80.0, 34.0), (104.0, 34.0), 25.0)
    check(s.of_type("shot")[0]["team"] == "team1", s.of_type("shot"))
    s2 = Sim(team1_attacks_right=False)
    s2.hold(0.5, (80.0, 34.0), [])
    s2.move((80.0, 34.0), (104.0, 34.0), 25.0)
    check(s2.of_type("shot")[0]["team"] == "team2", s2.of_type("shot"))
    s3 = Sim()
    s3.det.set_attack_direction(False)
    s3.hold(0.5, (25.0, 34.0), [])
    s3.move((25.0, 34.0), (1.0, 34.0), 25.0)
    # megfordított irány: az 1. csapat a bal (x=0) kapura támad
    check(s3.of_type("shot")[0]["team"] == "team1", s3.of_type("shot"))


# ------------------------------------------------------------------- gól
def test_goal():
    s = Sim()
    att = P(7, 1, 87.5, 34)
    gk = P(1, 2, 104.0, 41.0, keeper=True)
    s.hold(1.0, (88.0, 34.0), [att, gk])
    s.move((88.0, 34.0), (106.0, 34.5), 25.0, lambda b: [att, gk])
    s.hold(3.0, (106.0, 34.5), [att, gk])
    s.finalize()
    goals = s.of_type("goal")
    check(len(goals) == 1 and goals[0]["team"] == "team1", goals)
    conc = s.of_type("gk_conceded")
    check(len(conc) == 1 and conc[0]["team"] == "team2" and conc[0]["keeper_track_id"] == 1, conc)
    check(not s.of_type("gk_save"), s.of_type("gk_save"))
    check(s.det.stats["goals"]["team1"] == 1 and s.det.stats["shots_on_target"]["team1"] == 1, s.det.stats)
    check(len(s.of_type("shot")) == 1, s.of_type("shot"))


def test_clearance_to_corner_is_not_goal():
    """Felszabadítás szögletre (a valós felvételen gólnak vette): a labda a kapufa mellett, a vonal
    mögé megy, majd a szögletzászlónál áll meg – nem gól, nem kapott gól."""
    s = Sim()
    defender = P(4, 2, 95.0, 30.0)
    gk = P(1, 2, 103.0, 40.0, keeper=True)
    s.hold(1.0, (95.0, 30.0), [defender, gk])
    # lassú, kapufa közelébe (y≈30.5, a kapufa 30.34) tartó labda a vonal mögé – nincs "lövés" előtte
    s.move((95.0, 30.0), (106.0, 30.6), 9.0, lambda b: [defender, gk])
    s.hold(2.0, (106.0, 30.6), [defender, gk])
    s.move((106.0, 30.6), (104.6, 0.6), 6.0, lambda b: [defender, gk])   # a labdát a sarokhoz viszik
    s.hold(3.0, (104.6, 0.6), [defender, gk])
    s.finalize()
    check(not s.of_type("goal"), s.of_type("goal"))
    check(not s.of_type("gk_conceded"), s.of_type("gk_conceded"))


def test_ball_behind_line_without_shot_needs_kickoff():
    """Lövés nélkül a vonal mögött álló labda nem megerősített gól; középkezdés után igen."""
    s = Sim()
    gk = P(1, 2, 98.0, 20.0, keeper=True)
    s.hold(1.0, (100.0, 34.0), [gk])
    s.move((100.0, 34.0), (105.8, 34.0), 3.0, lambda b: [gk])    # lassan begurul (nem lövés-sebesség)
    s.hold(3.0, (105.8, 34.0), [gk])
    s.move((105.8, 34.0), (52.5, 34.0), 20.0, lambda b: [gk])
    s.hold(2.0, (52.5, 34.0), [gk])                               # középkezdés
    s.finalize()
    goals = s.of_type("goal")
    check(len(goals) == 1 and goals[0]["confirmation"] == "kickoff", goals)


# ------------------------------------------------------------ kapus események
def test_save():
    s = Sim()
    att = P(7, 1, 87.5, 34)
    gk = P(1, 2, 104.0, 35.0, keeper=True)
    s.hold(1.0, (88.0, 34.0), [att, gk])
    s.move((88.0, 34.0), (103.4, 35.0), 25.0, lambda b: [att, gk])
    s.hold(3.0, (103.6, 35.0), [att, gk])   # a kapus megfogja
    saves = s.of_type("gk_save")
    check(len(saves) == 1 and saves[0]["team"] == "team2" and saves[0]["keeper_track_id"] == 1, saves)
    check(not s.of_type("goal"), s.of_type("goal"))
    check(not s.of_type("gk_goal_kick"), "a kézben tartott labda nem kirúgás")
    check(s.det.stats["gk_events"]["team2"].get("gk_save") == 1, s.det.stats)


def test_cross_claim():
    s = Sim()
    winger = P(11, 1, 88.0, 5.5)
    gk = P(1, 2, 100.5, 32.5, keeper=True)
    s.hold(1.0, (88.0, 5.0), [winger, gk])
    s.move((88.0, 5.0), (100.2, 32.4), 18.0, lambda b: [winger, gk])
    s.hold(1.0, (100.2, 32.4), [winger, gk])
    claims = s.of_type("gk_claim_cross")
    check(len(claims) == 1 and claims[0]["team"] == "team2", claims)
    check(not s.of_type("shot"), s.of_type("shot"))


def test_goal_kick():
    s = Sim()
    gk = P(1, 2, 101.8, 40.0, keeper=True)
    s.hold(1.0, (100.0, 50.0), [gk])
    s.move((100.0, 50.0), (106.0, 50.0), 5.0, lambda b: [gk])   # alapvonalon túl, kapun kívül
    s.hold(3.0, None, [gk])
    s.hold(2.0, (103.0, 40.0), [gk])                            # labda lehelyezve az ötösön
    recv = P(6, 2, 55.0, 40.0)
    s.move((103.0, 40.0), (55.0, 40.0), 22.0, lambda b: [gk, recv])
    s.hold(1.0, (55.0, 40.0), [gk, recv])
    s.finalize()
    gks = s.of_type("gk_goal_kick")
    check(len(gks) == 1 and gks[0]["team"] == "team2" and gks[0]["is_long"], gks)
    check(45.0 < gks[0]["distance_m"] < 50.0, gks)
    check(not s.of_type("gk_long_ball") and not s.of_type("gk_short_pass"), "nincs dupla kihozatal")


def _distribution(end, speed):
    s = Sim()
    gk = P(1, 1, 3.0, 34.0, keeper=True)
    s.hold(2.0, (3.5, 34.0), [gk])
    recv = P(5, 1, end[0], end[1])
    s.move((3.5, 34.0), end, speed, lambda b: [gk, recv])
    s.hold(1.0, end, [gk, recv])
    s.finalize()
    return s


def test_long_vs_short_distribution():
    s = _distribution((50.0, 30.0), 20.0)
    lb = s.of_type("gk_long_ball")
    check(len(lb) == 1 and lb[0]["team"] == "team1" and lb[0]["distance_m"] > 35, lb)
    check(not s.of_type("gk_short_pass") and not s.of_type("gk_goal_kick") and not s.of_type("shot"),
          s.det.events)
    s = _distribution((15.0, 20.0), 8.0)
    sp = s.of_type("gk_short_pass")
    check(len(sp) == 1 and sp[0]["team"] == "team1" and sp[0]["distance_m"] < 35, sp)
    check(not s.of_type("gk_long_ball"), s.det.events)


def test_sweeper():
    s = Sim()
    x = 8.0
    while x < 26.0:
        s.step((40.0, 30.0), [P(1, 1, x, 34.0, keeper=True)])
        x += 6.0 * DT
    sw = s.of_type("gk_sweeper")
    check(len(sw) == 1 and sw[0]["team"] == "team1", sw)


# ------------------------------------------------ zajtűrés / regressziók
def test_goal_kick_not_from_ball_behind_goal_line():
    # Gól után a labda a gólvonal mögött (a hálóban) pihen, majd eltűnik és a középponton jelenik meg:
    # ez NEM kirúgás (régi hiba: a vonal mögötti labdát "az ötösben állónak" vette)
    s = Sim()
    att = P(7, 1, 87.5, 34)
    gk = P(1, 2, 103.0, 41.0, keeper=True)
    s.hold(1.0, (88.0, 34.0), [att, gk])
    s.move((88.0, 34.0), (106.0, 34.5), 25.0, lambda b: [att, gk])
    s.hold(2.0, (106.0, 34.5), [att, gk])
    s.hold(3.0, None, [att, gk])
    s.hold(2.5, (52.5, 34.0), [P(9, 2, 53.5, 34.0), gk])        # középkezdés
    s.move((52.5, 34.0), (45.0, 30.0), 10.0, lambda b: [gk])
    s.hold(1.0, (45.0, 30.0), [gk])
    s.finalize()
    check(len(s.of_type("goal")) == 1, s.of_type("goal"))
    check(not s.of_type("gk_goal_kick"), s.of_type("gk_goal_kick"))
    check(not s.of_type("gk_long_ball") and not s.of_type("gk_short_pass"), s.det.events)


def test_goal_kick_timed_at_kick():
    # Alapvonalon túl -> a labda 3 mp-ig áll az ötösben -> elrúgják: az esemény ideje az elrúgás
    s = Sim()
    gk = P(1, 2, 101.0, 39.0, keeper=True)
    s.hold(1.0, (95.0, 50.0), [gk])
    s.move((95.0, 50.0), (106.5, 52.0), 12.0, lambda b: [gk])
    s.hold(2.0, None, [gk])
    s.hold(3.0, (100.0, 39.0), [gk])
    t_kick = s.t
    recv = P(6, 2, 52.0, 30.0)
    s.move((100.0, 39.0), (52.0, 30.0), 22.0, lambda b: [gk, recv])
    s.hold(1.0, (52.0, 30.0), [gk, recv])
    s.finalize()
    gks = s.of_type("gk_goal_kick")
    check(len(gks) == 1 and gks[0]["team"] == "team2", gks)
    check(abs(gks[0]["time"] - t_kick) <= 0.2, (gks, t_kick))


def test_keeper_held_ball_projected_behind_line_is_save():
    # Egykamerás vetítés: a kapus kezében ~1 m magasan tartott labda talajra vetített képe a gólvonal
    # mögé, a kapufák közé esik. Ez védés, nem gól (a kapus utána kidobja a labdát).
    s = Sim()
    att = P(7, 1, 82.5, 36.0)
    gk = P(1, 2, 98.5, 37.0, keeper=True)
    s.hold(1.0, (83.0, 36.0), [att, gk])
    s.move((83.0, 36.0), (98.0, 36.8), 24.0, lambda b: [att, gk])
    # a kamera (52.5, 74, 7) felől nézve az 1 m magasan tartott labda vetülete: ~(106.2, 30.8)
    s.hold(2.0, (106.2, 31.0), [att, gk])
    recv = P(4, 2, 85.0, 50.0)
    s.move((97.0, 38.0), (85.0, 50.0), 12.0, lambda b: [gk, recv])
    s.hold(1.0, (85.0, 50.0), [gk, recv])
    s.finalize()
    check(not s.of_type("goal"), s.of_type("goal"))
    saves = s.of_type("gk_save")
    check(len(saves) == 1 and saves[0]["team"] == "team2", saves)
    check(len(s.of_type("shot")) == 1, s.of_type("shot"))


def test_goal_with_keeper_near_confirmed_by_kickoff():
    # A kapus a labda mellett vetődik (a gólvonalnál): a gólt csak a középkezdés erősíti meg
    s = Sim()
    att = P(7, 1, 87.5, 34)
    gk = P(1, 2, 104.2, 35.5, keeper=True)
    s.hold(1.0, (88.0, 34.0), [att, gk])
    s.move((88.0, 34.0), (105.8, 35.0), 25.0, lambda b: [att, gk])
    s.hold(0.5, (105.8, 35.0), [att, gk])
    s.hold(6.0, None, [att, gk])
    check(not s.of_type("goal"), "középkezdés nélkül még nincs megerősítve")
    s.hold(2.0, (52.5, 34.0), [P(9, 2, 53.5, 34.0), gk])
    goals = s.of_type("goal")
    check(len(goals) == 1 and goals[0]["team"] == "team1", goals)
    check(goals[0]["confirmation"] == "kickoff" and goals[0]["confidence"] >= 0.9, goals)
    check(abs(goals[0]["time"] - 1.7) < 0.3, goals)       # az időpont a gólvonal átlépése
    check(not s.of_type("gk_save"), s.of_type("gk_save"))


def test_noisy_shot_and_blips():
    # Zajos labdapozíció (±0,5 m), kiesések és egy téves felvillanás: egyetlen lövés, nincs gól
    import random
    rng = random.Random(3)
    s = Sim()
    att = P(7, 1, 84.5, 34)
    gk = P(1, 2, 103.5, 34.0, keeper=True)
    for _ in range(25):
        s.step((85.0 + rng.gauss(0, 0.5), 34.0 + rng.gauss(0, 0.5)), [att, gk])
    n = 30
    for i in range(1, n + 1):
        x, y = 85.0 + 18.0 * i / n, 34.0 + 6.5 * i / n
        if i in (5, 6, 12, 13, 14):
            ball = None                        # kiesés
        elif i == 9:
            ball = (60.0, 10.0)                # téves felvillanás
        else:
            ball = (x + rng.gauss(0, 0.5), y + rng.gauss(0, 0.5))
        s.step(ball, [att, gk])
    s.hold(2.0, None, [att, gk])
    s.finalize()
    check(len(s.of_type("shot")) == 1, s.of_type("shot"))
    check(not s.of_type("goal"), s.of_type("goal"))


def test_backpass_not_shot():
    # A védekező csapat játékosa a saját kapusához passzol gyorsan: nem lövés
    s = Sim()
    defender = P(3, 1, 25.5, 34.0)
    gk = P(1, 1, 3.0, 34.0, keeper=True)
    s.hold(1.0, (25.0, 34.0), [defender, gk])
    s.move((25.0, 34.0), (3.5, 34.0), 16.0, lambda b: [defender, gk])
    s.hold(1.0, (3.5, 34.0), [gk])
    s.finalize()
    check(not s.of_type("shot"), s.of_type("shot"))
    check(not s.of_type("gk_save"), s.of_type("gk_save"))


def test_engine_integration():
    try:
        import numpy as np
        from tactical_engine import TacticalEngine
    except ImportError as ex:  # pragma: no cover
        print(f"  (kihagyva: {ex})")
        return
    eng = TacticalEngine()
    H = np.eye(3, dtype=np.float64)  # pixel == méter
    t = 0.0
    for _ in range(25):
        eng.update_tracks(0, t, [{"track_id": 7, "team_id": 2, "is_keeper": False, "screen_foot": (20.5, 36)}],
                          (20.0, 36.0), H)
        t += DT
    for i in range(1, 18):
        eng.update_tracks(0, t, [], (20.0 - i * 1.1, 36.0 - i * 0.1), H)
        t += DT
    eng.finalize()
    check(eng.stats["shots"]["team2"] == 1 and eng.stats["shots_on_target"]["team2"] == 1, eng.stats)
    merged = eng.merge_timeline([{"type": "gk_positioning_error", "time": 0.1, "team": "team1", "desc": "x"}])
    check(merged[0]["type"] == "gk_positioning_error" and merged[-1]["type"] == "shot", merged)
    check(eng.stats["gk_events"]["team1"].get("gk_positioning_error") == 1, eng.stats)
    r = eng.analyze_goalkeeper((103.0, 34.0), (90.0, 34.0))
    check(r["dist_from_goal_m"] == 2.0 and abs(r["bisector_offset_m"]) < 0.01, r)


def test_positioning_monitor():
    from events import PositioningMonitor
    m = PositioningMonitor(offset_m=2.5, min_s=1.5, ball_max_dist_m=30.0, debounce_s=20.0)

    def ev(off, bx=90.0, gx=103.0):
        return {"bisector_offset_m": off, "gk_coords": (gx, 34.0), "ball_coords": (bx, 30.0), "dist_from_goal_m": 2.0}

    out = []
    t = 0.0
    for _ in range(25):          # 1 mp nagy eltérés: még nem esemény
        out.append(m.update(t, ev(3.0), "team2")); t += 0.04
    for _ in range(10):          # rövid szünet (a kapus visszalép)
        out.append(m.update(t, ev(0.5), "team2")); t += 0.04
    check(not any(out), "a rövid kilengés nem esemény")
    out = []
    for _ in range(60):          # 2.4 mp tartós eltérés
        out.append(m.update(t, ev(-3.4), "team2")); t += 0.04
    evs = [e for e in out if e]
    check(len(evs) == 1 and evs[0]["team"] == "team2" and evs[0]["offset_m"] == -3.4, evs)
    # a labda a túlsó térfélen: nem számít
    m2 = PositioningMonitor()
    res = [m2.update(i * 0.04, ev(5.0, bx=20.0), "team2") for i in range(100)]
    check(not any(res), "távoli labdánál nincs helyezkedési hiba")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Exception as ex:  # noqa: BLE001
            failed += 1
            print(f"HIBA  {name}: {type(ex).__name__}: {ex}")
    print(f"\n{len(tests) - failed}/{len(tests)} teszt sikeres")
    sys.exit(1 if failed else 0)

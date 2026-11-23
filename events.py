"""
Mérkőzés-esemény detektálás (lövés, gól, kapus események) tisztán Python alapon.

Bemenet képkockánként: időbélyeg (mp), a labda pálya-koordinátája méterben (vagy None),
valamint a játékosok listája (track_id, team_id, is_keeper, pitch_coords).
Koordináta-rendszer: 105x68 m-es pálya, x a hossz mentén (bal kapu x=0, jobb kapu x=105),
kapufák: y = 30.34 .. 37.66.

FONTOS (egykamerás korlát): a labda magasságát nem ismerjük. A levegőben lévő labda
talajra vetített pozíciója torzul (a kamerától távolodva "elcsúszik"), ezért a sebesség/irány
becslés ívelt labdáknál pontatlan, és az ívelés/rövid passz megkülönböztetése KIZÁRÓLAG a
megtett távolság alapján történik. A kapus kezében ~1 m magasan tartott labda vetülete a
gólvonal mögé eshet – ezért a gólt nem egyetlen képkocka dönti el (lásd lent).

Zajtűrés (valódi detektor + homográfia kimenetéhez):
  * labda: kapuzás (fizikailag lehetetlen ugrások és rövid téves felvillanások eldobása, új
    labda-track csak több egybehangzó minta után), rövid csúszóablakos egyenes-illesztés a
    pálya-síkban (simított pozíció + sebesség), a sebességet nem számoljuk ~0,4 mp-nél
    hosszabb kiesésen át, és a lövéshez több egybehangzó, megbízható minta kell;
  * kapusok: csapatonként követve (track-ID csere / új ID nem számít), simított pozíció,
    ~1 m-es remegést tűrő birtoklási sugarak;
  * gól: a kapufák közötti átjutás csak JELÖLT; megerősítés: a labda a vonal mögött marad
    (ha nincs ott a kapus), vagy középkezdés (álló labda a középponton) következik. Ha a kapus
    a labdánál van (közel / a kamera-sugár szerint a kezében tartja), csak középkezdés erősít
    meg, a kapus labdakihozatala pedig elveti (ilyenkor védés);
  * kirúgás: alapvonalon túlra került labda (nem gól) -> a labda az ötösön belül ≥ 1 mp-ig
    áll -> elrúgják; az esemény időpontja az elrúgás.

Labdás mezőny-események (1. szakasz: minden labdás esemény csapatonként, kalapokba sorolva –
EVENT_CATEGORIES): passz / sikertelen passz / labdaszerzés / szerelés / labdavezetés a megerősített
birtoklási szakaszok egymásutánjából; beadás a szélső sávból a 16-osba; szöglet / bedobás /
szabadrúgás / kezdőrúgás az álló labda helyéből és elrúgásából (bedobás álló labda nélkül is:
oldalvonalon túl -> holtidő -> a vonal mellől a pályára). Kétes helyzetben nincs esemény; a valós
felvételen még bizonytalan típusok "experimental": True jelölést kapnak (EXPERIMENTAL_TYPES).

A modul nem függ torch/ultralytics/cv2 csomagtól, így önállóan tesztelhető.
"""

import math
from collections import deque

TEAM_LABELS = {"team1": "1. csapat", "team2": "2. csapat"}
KEEPER_LABELS = {"team1": "1. csapat kapusa", "team2": "2. csapat kapusa"}

GK_EVENT_TYPES = (
    "gk_save", "gk_claim_cross", "gk_goal_kick", "gk_long_ball",
    "gk_short_pass", "gk_conceded", "gk_sweeper", "gk_positioning_error",
)

# Labdás (nem kapus) események – a meccs teljes eseménylistájához (1. szakasz: minden labdás esemény
# felismerése és kategorizálása csapatonként)
PLAY_EVENT_TYPES = (
    "pass", "pass_failed", "cross", "throw_in", "corner", "free_kick", "kickoff", "tackle", "interception",
    "dribble",
)

# Eseménytípus -> "kalap" (a felület ezek szerint csoportosít). Egyetlen helyen tartjuk.
EVENT_CATEGORIES = {
    "corner": "rogzitett", "free_kick": "rogzitett", "throw_in": "rogzitett", "kickoff": "rogzitett",
    "gk_goal_kick": "rogzitett",
    "shot": "befejezes", "goal": "befejezes",
    "cross": "beadas",
    "pass": "passz", "pass_failed": "passz",
    "tackle": "labdaszerzes", "interception": "labdaszerzes",
    "dribble": "labdavezetes",
}
for _t in GK_EVENT_TYPES:
    EVENT_CATEGORIES.setdefault(_t, "kapus")

CATEGORY_LABELS = {
    "rogzitett": "Rögzített helyzetek", "befejezes": "Befejezések", "beadas": "Beadások",
    "passz": "Passzok", "labdaszerzes": "Labdaszerzések", "labdavezetes": "Labdavezetés",
    "kapus": "Kapus események", "egyeb": "Egyéb",
}

EVENT_LABELS = {
    "shot": "Lövés", "goal": "Gól", "pass": "Passz", "pass_failed": "Sikertelen passz", "cross": "Beadás",
    "throw_in": "Bedobás", "corner": "Szöglet", "free_kick": "Szabadrúgás", "kickoff": "Kezdőrúgás",
    "tackle": "Szerelés", "interception": "Labdaszerzés (elfogott passz)", "dribble": "Labdavezetés / cselezés",
    "gk_save": "Kapus védés", "gk_claim_cross": "Beadás elfogása", "gk_goal_kick": "Kirúgás",
    "gk_long_ball": "Kapus ívelés", "gk_short_pass": "Kapus rövid passz", "gk_conceded": "Kapott gól",
    "gk_sweeper": "Kapus kifutás", "gk_positioning_error": "Kapus helyezkedési hiba",
}

# Kísérleti (a valós, zajos felvételen még nem elég megbízható) típusok: az esemény "experimental": True
# jelölést kap, a felület alapból elrejtheti / külön jelölheti. Lásd tools/replay_events.py.
EXPERIMENTAL_TYPES = frozenset({"pass_failed", "tackle", "interception", "cross", "dribble"})

# Típusonkénti klip-ablak (mp az esemény előtt, után) – ahol eltér az alapértelmezéstől
_CLIP_WINDOWS = {
    "pass": (2.0, 2.0), "pass_failed": (2.0, 3.0), "tackle": (3.0, 3.0), "interception": (3.0, 3.0),
    "cross": (3.0, 4.0), "throw_in": (2.0, 5.0), "corner": (2.0, 7.0), "free_kick": (3.0, 7.0),
    "kickoff": (1.0, 5.0), "dribble": (1.0, 2.0),
}


def event_category(typ):
    """Az eseménytípus kalapja (EVENT_CATEGORIES), ismeretlen típusnál "egyeb"."""
    return EVENT_CATEGORIES.get(typ, "kapus" if str(typ).startswith("gk_") else "egyeb")


def annotate_categories(events):
    """A 'category' mező pótlása (külső / régi eseménylistákhoz). Helyben módosít, a listát adja vissza."""
    for ev in events or []:
        if isinstance(ev, dict) and ev.get("type"):
            ev.setdefault("category", event_category(ev["type"]))
    return events


# Eseménytípus -> a stats kulcsa (a statisztika a kiadott eseményeket számolja)
_STAT_KEYS = {
    "pass": "passes", "pass_failed": "passes_failed", "cross": "crosses", "tackle": "tackles",
    "interception": "interceptions", "throw_in": "throw_ins", "corner": "corners",
    "free_kick": "free_kicks", "kickoff": "kickoffs", "dribble": "dribbles",
}

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
    fit = fit_line(samples)
    return None if fit is None else (fit[3], fit[4])


def fit_line(samples):
    """Egyenes illesztés (t, x, y) mintákra.
    Visszatér: (t_átlag, x_átlag, y_átlag, vx, vy, rms_maradék_m, sebesség_szórás_mps) vagy None."""
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
    sse = 0.0
    for s in samples:
        dx = s[1] - (xm + vx * (s[0] - tm))
        dy = s[2] - (ym + vy * (s[0] - tm))
        sse += dx * dx + dy * dy
    rms = math.sqrt(sse / n)
    # a meredekség szórása (tengelyenként): sigma / sqrt(sum (t - tm)^2), sigma ~ maradék
    dof = max(1, n - 2)
    sigma = math.sqrt(sse / (2.0 * dof))
    sv = sigma / math.sqrt(den)
    return tm, xm, ym, vx, vy, rms, sv


def _seg_dist(p, a, b):
    """Pont - szakasz távolság (2D)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    u = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + u * dx), p[1] - (ay + u * dy))


def load_tuning(path):
    """
    A hangolt küszöbök (tools/hangolas.py -> esemeny_hangolas.json) beolvasása: {param: érték}. Hiányzó vagy
    hibás fájlnál üres dict; ismeretlen paraméter kimarad (egy régebbi / újabb verzió fájlja sem okoz hibát).
    """
    import json as _json
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = _json.load(f)
    except (OSError, ValueError):
        return {}
    over = data.get("overrides") if isinstance(data, dict) else None
    if not isinstance(over, dict) or data.get("enabled") is False:
        return {}
    ref = EventConfig()
    out = {}
    for k, v in over.items():
        if k == "team1_attacks_right" or not hasattr(ref, k):
            continue
        cur = getattr(ref, k)
        if isinstance(cur, bool) or not isinstance(cur, (int, float)) or not isinstance(v, (int, float)):
            continue
        out[k] = type(cur)(v) if isinstance(cur, int) and float(v).is_integer() else float(v)
    return out


class EventConfig:
    """Az eseménydetektálás hangolható paraméterei. Bármelyik felülírható kulcsszavas argumentummal."""

    def __init__(self, **overrides):
        # --- Labda szűrés / sebesség becslés ---
        self.vel_window_s = 0.4            # csúszóablak (mp) a simításhoz / sebesség-regresszióhoz
        self.vel_min_samples = 4           # minimum minta a megbízható sebességhez
        self.vel_min_span_s = 0.1          # a megbízható becslés mintái legalább ennyi időt fedjenek le
        self.vel_max_span_s = 0.6          # ritka minták esetén ennyi mp-nél hosszabb ablakot nem használunk
        self.vel_max_resid_m = 1.5         # efölötti illesztési maradék: az ablakban irányváltás / téves minta volt
        self.ball_gap_s = 0.4              # ennél hosszabb kiesésen át nem számolunk sebességet
        self.max_ball_speed_mps = 55.0     # efölötti ugrás téves labda-detektálásnak minősül (a levegőben lévő
                                           # labda vetülete a valódi sebességnél jóval gyorsabban "mozoghat")
        self.ball_gate_m = 2.0             # a sebesség-kapu mellett ennyi pozíciózajt tűrünk
        self.ball_reacquire_s = 1.0        # ennél hosszabb kiesés után bármely új labda-pozíciót elfogadunk
        self.ball_switch_samples = 4       # ennyi egybehangzó "kapun kívüli" minta után új labda-trackre váltunk

        # --- Lövés ---
        self.shot_speed_kmh = 45.0         # lövés minimális labdasebessége
        self.shot_speed_sigma_k = 1.0      # a becsült sebesség - k * szórás érje el a küszöböt
        self.shot_confirm_samples = 2      # ennyi egymást követő megbízható becslés kell a lövéshez
        self.shot_max_distance_m = 35.0    # ennél messzebbről a gólvonaltól nem tekintjük lövésnek
        self.shot_min_x_ratio = 0.5        # |vx|/sebesség minimum (beadások kiszűrése, ~60° max. szög)
        self.post_tolerance_m = 0.5        # kapufa-tűrés a kaput eltaláló lövéshez
        self.off_target_band_m = 12.0      # kapu mellé tartó lövés: max. ennyi m-re a kapu közepétől
        self.shot_debounce_s = 1.5         # egy lövés után ennyi ideig nincs új lövés
        self.shot_validate_s = 3.0         # eddig várunk a lövés igazolására (kapu elérése / kapus / kiesés)
        self.shot_valid_line_m = 2.5       # a gólvonaltól ennyin belülre érő labda igazolja a lövést
        self.team1_attacks_right = True    # alapértelmezés: az 1. csapat a x=105 kapura támad

        # --- Gól ---
        self.goal_debounce_s = 10.0
        self.goal_line_margin_m = 0.5      # ennyivel a gólvonal mögött kell lennie a (simított) labdának
        self.goal_min_samples = 2          # ennyi egymást követő minta a vonal mögött -> gól-jelölt
        self.goal_persist_s = 1.0          # megerősítés: ennyi ideig a vonal mögött marad (kapus nincs ott)
        self.goal_persist_samples = 5
        self.goal_kickoff_window_s = 60.0  # középkezdés ennyin belül a jelölt után -> megerősítés
        self.goal_kickoff_still_s = 1.0    # a középkezdéshez ennyi ideig álljon a labda a középponton
        self.goal_kickoff_radius_m = 2.5
        self.goal_keeper_radius_m = 2.0    # kapus ennyin belül a labdánál az átjutáskor -> valószínű védés
        self.goal_keeper_line_m = 3.0      # ... és a kapus is a gólvonal közelében van
        self.goal_post_inner_m = 0.4       # a "vonal mögött maradt" megerősítéshez a labda a kapufáktól
                                           # ennyivel BEJJEBB jusson át (a kapufa melletti, szögletre / alapvonalra
                                           # menő labda vetülete a kalibráció ~1 m-es hibájával könnyen "bent" van)
        self.goal_corner_radius_m = 4.0    # a jelölt után a szögletzászlónál megálló labda: szöglet, nem gól
        self.goal_emit_unconfirmed = True   # megerősítetlen, de valószínű gól kiadása ("Valószínű GÓL",
                                            # confirmed=False, confidence=0.5) – lásd _likely_unconfirmed

        # --- Kamera (a kapus kezében tartott labda vetületéhez) ---
        # "auto": szokásos Veo-elhelyezés a félpályavonalnál, a közeli (y = W) oldalvonal mögött
        # 6 m-re, 7 m magasan; None: kikapcsolva; vagy (x, y, magasság) méterben.
        self.camera_pos = "auto"
        self.held_min_height_m = 0.4
        self.held_max_height_m = 1.6
        self.held_tolerance_m = 1.3

        # --- Labdabirtoklás ---
        self.possession_radius_m = 1.5
        self.possession_max_ball_speed_mps = 10.0
        self.possession_memory_s = 4.0     # ennél régebbi birtoklás alapján nem írunk jóvá lövést
        self.keeper_jitter_m = 1.0         # a kapus-pozíció ennyit remeghet (a sugarak ezzel bővülnek)
        self.keeper_smooth_s = 0.3

        # --- Védés ---
        self.save_contact_radius_m = 2.0
        self.save_loose_radius_m = 3.0     # ezen belül megállás/irányváltás is védésnek számít
        self.save_confirm_s = 2.0          # ennyi ideig nem eshet gól a kontakt után
        self.save_target_margin_m = 4.0    # a kapufákon kívül ennyin belül tartó lövés is lehet védés
        self.save_vanish_s = 0.5           # a kapusnál eltűnő labda (kézben, vetítve) ennyi idő után kontakt
        self.shot_resolve_window_s = 3.0   # a lövés eddig "él" (gól/védés hozzárendelés)

        # --- Beadás elfogása ---
        self.wide_zone_y = 14.0            # szélső sáv: y < 14 vagy y > 54
        self.cross_window_s = 3.0
        self.cross_max_line_dist_m = 40.0  # a beadás a gólvonaltól ennyin belül induljon

        # --- Kirúgás (kapukirúgás) ---
        self.goal_kick_still_s = 1.0
        self.goal_kick_still_radius_m = 1.5   # álló labda: a minták ennyin belül (pozíciózajjal együtt)
        self.goal_kick_min_speed_mps = 8.0
        self.goal_kick_launch_dist_m = 3.0
        self.goal_kick_out_memory_s = 45.0
        self.goal_kick_area_tol_m = 1.5    # az ötös határán ennyi pozíciózajt tűrünk

        # --- Kapus labdakihozatal ---
        self.long_ball_m = 35.0            # ennél hosszabb út = ívelés (csak távolság alapján!)
        self.min_distribution_m = 4.0
        self.distribution_timeout_s = 6.0
        self.distribution_lost_s = 3.5     # ennyi ideig várunk a levegőben "eltűnt" labdára
        self.distribution_return_m = 15.0  # a legtávolabbi ponthoz képest ennyivel visszajött labda: a kihozatal vége
        self.gk_keep_radius_m = 2.5
        self.gk_release_radius_m = 3.0
        self.gk_hold_max_s = 8.0           # a kézben (láthatatlanul) tartott labda max. ideje
        self.gk_throw_speed_mps = 12.0     # kidobás feltételezett sebessége a kiadás idejének becsléséhez

        # --- Kifutás ---
        self.sweeper_speed_mps = 4.0
        self.sweeper_min_line_dist_m = 12.0
        self.sweeper_debounce_s = 6.0
        self.sweeper_max_ball_dist_m = 30.0  # a kifutás a labda felé történik: a labda ennyin belül legyen
        self.sweeper_toward_ball_cos = 0.5   # a kapus mozgásiránya és a labda iránya közti szög koszinusza
        self.keeper_max_speed_mps = 10.0   # efölött track-ID csere / zaj
        # --- Kapus szabadrúgás (pl. les után): a kapus az álló labdához megy és elrúgja ---
        self.gk_free_kick_keeper_radius_m = 3.0
        self.gk_free_kick_max_line_dist_m = 35.0

        # --- Labdabirtoklási szakaszok (passz / sikertelen passz / szerelés / labdaszerzés) ---
        # Birtoklási minta: a labda (simított pozíció) a legközelebbi játékostól ctl_radius_m-en belül,
        # lassú, a pályán belül, és a legközelebbi ellenfél nincs majdnem ugyanolyan közel (vitatott labda).
        self.ctl_radius_m = 2.0
        self.ctl_max_ball_speed_mps = 8.0
        self.ctl_contest_margin_m = 0.4    # ennyin belül egyformán közeli ellenfél: vitatott, nem számít
        self.ctl_min_samples = 3           # egy birtoklás ennyi mintától "megerősített"
        self.ctl_min_span_s = 0.2          # ... és legalább ennyi ideig tart
        self.ctl_gap_s = 1.0               # a birtokláson belül ennyi labda-kiesést tűrünk
        self.ctl_same_player_m = 3.0       # ID-csere: ha a labda ennyin belül marad, ugyanaz a játékos vezeti
        self.pass_min_m = 6.0              # a passz minimális hossza (rövidebb: cselezés / zaj)
        self.pass_min_gap_s = 0.2
        self.pass_max_gap_s = 4.0          # a passzadás és az átvétel között legfeljebb ennyi idő
        self.pass_max_speed_mps = 40.0     # efölötti átlagsebesség: téves labda-ugrás
        self.pass_dribble_m = 2.5          # a passzadó még a labdánál: cselezés, nem passz
        self.pass_max_unseen_s = 1.5       # ennél hosszabb labda-kiesés a két birtoklás között: nem tudjuk, mi történt
        self.pass_receive_s = 1.2          # az átvétel legfeljebb ennyivel azután, hogy a labda az átvétel helyére ért
        self.pass_long_m = 30.0
        self.pass_progressive_m = 10.0
        self.pass_failed_min_m = 8.0       # ennél hosszabb úton az ellenfélhez kerülő labda: sikertelen passz
        self.tackle_max_m = 4.0            # ennél rövidebb úton, közvetlen kontaktussal: szerelés
        self.tackle_contact_m = 2.5        # a szerelő játékos ennyire volt a labdától az utolsó birtoklási mintánál
        self.dribble_min_m = 10.0          # labdavezetés: a birtoklás alatt ennyit haladt a labda
        self.dribble_min_s = 1.5

        # --- Beadás (szélről a 16-os felé) ---
        self.cross_final_third_m = 35.0    # a beadás a gólvonaltól ennyin belül induljon
        self.cross_box_tol_m = 1.0         # a 16-os határán ennyi pozíciózajt tűrünk
        self.cross_min_m = 8.0
        self.cross_window_flight_s = 4.0   # az indítás után eddig kell a 16-osba érnie

        # --- Rögzített helyzetek (szöglet, bedobás, szabadrúgás, kezdőrúgás) ---
        self.restart_launch_dist_m = 3.0   # az álló labda ennyit mozdul el -> elrúgták / bedobták
        self.restart_min_speed_mps = 5.0
        self.restart_debounce_s = 3.0
        self.restart_team_wait_s = 4.0     # ismeretlen végrehajtó csapat: eddig várunk a következő birtoklásra
        self.restart_kicker_m = 2.5        # a végrehajtó játékos ennyin belül legyen az álló labdától
        self.corner_radius_m = 3.5         # a szögletzászlótól ennyin belül álló labda: szöglet
        self.corner_still_s = 1.0
        self.touch_zone_m = 2.5            # az oldalvonaltól ennyin belül (vagy kívül) álló labda: bedobás
        self.touch_out_margin_m = 0.5      # ennyivel az oldalvonalon túl: a labda elhagyta a pályát
        self.throw_in_still_s = 0.8
        self.throw_in_rest_alone_s = 1.5   # oldalvonalon túlra kerülés nélkül ennyi ideig álljon a labda
        self.throw_in_min_dead_s = 3.0     # a pályára azonnal visszagurulás nem bedobás
        self.throw_in_out_memory_s = 60.0
        self.throw_in_max_dx_m = 25.0      # a bedobás helye a kimenetel helyétől
        self.free_kick_still_s = 1.5
        self.free_kick_dead_s = 3.0        # ennyi állás már biztosan holt labda (rövidebbnél más jel is kell)
        self.free_kick_players_m = 15.0
        self.free_kick_min_players = 2
        self.free_kick_dangerous_m = 35.0  # ennyin belül az ellenfél kapujától: veszélyes szabadrúgás
        self.kickoff_still_s = 1.0

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
        cam = self.cfg.camera_pos
        if cam == "auto":
            cam = (self.L / 2.0, self.W + 6.0, 7.0)
        self._cam = tuple(float(c) for c in cam) if cam else None

        self.events = []
        self.stats = {
            "shots": {"team1": 0, "team2": 0},            # összes lövés (kaput eltalálók is)
            "shots_on_target": {"team1": 0, "team2": 0},
            "goals": {"team1": 0, "team2": 0},
            "gk_events": {"team1": {}, "team2": {}},
        }
        # a labdás események darabszáma csapatonként (pontosan a kiadott eseményeket számolja)
        for key in _STAT_KEYS.values():
            self.stats[key] = {"team1": 0, "team2": 0}

        self._raw = deque()              # elfogadott nyers labdaminták (t, x, y)
        self._cand = None                # alternatív labda-track jelölt (téves felvillanás vagy új labda)
        self._bst = None                 # simított labdaállapot: {t, pos, vel, speed, sv, reliable}
        self._keepers = {}               # csapat -> {team, tid, hist, pos, last_t, last_sweep}
        # a mezőnyjátékosok átlagos x-e csapatonként (melyik csapat melyik kaput védi); {team: [átlag, n]}
        self._side_x = {"team1": [0.0, 0], "team2": [0.0, 0]}
        self._side_left = None           # a legutóbb eldöntött bal kaput védő csapat
        self._poss = None                # (tid, team, is_keeper, t)
        self._poss_hist = deque()        # (t, team, tid) – birtoklási minták az utolsó 3 mp-ből
        self._last_shot_t = _NEG_INF
        self._shot_dir = None            # az utolsó lövés iránya, amíg a labda gyors marad
        self._shot_streak = None
        self._last_goal_t = _NEG_INF
        self._pending_shot = None
        self._goal_cand = None
        self._goal_block = set()         # elvetett jelölt után: addig nincs új, amíg a labda vissza nem jön
        self._last_in_field = None
        self._beyond = None              # (goal_x, n, t_first, t_in) – egymást követő minták a vonal mögött
        self._out_seq = None             # (goal_x, n) – egymást követő minták az alapvonalon túl (kapun kívül)
        self._out = {}                   # goal_x -> t (labda az alapvonalon túl, nem gól)
        self._out_clear = {}             # goal_x -> egymást követő minták messze a kaputól
        self._lost_checked = None
        self._reappear = None            # (t, kiesés hossza, utolsó pozíció, utolsó sebesség)
        self._rest = None                # álló labda: {t0, t_last, sx, sy, n, miss}
        self._last_kickoff = None        # (t0, t_last) – álló labda a középponton
        self._gk_armed = None
        self._last_wide = None           # (t, x, y)
        self._gk_poss = None
        self._held_seq = None            # (team, n, t) – a kapus kamera-sugarán látott labda
        self._dist = None
        self._dist_started = []          # (t, team) – kapus-kihozatalok indulása (gól-jelölt elvetéséhez)
        # --- labdabirtoklási szakaszok (passz / szerelés) ---
        self._tid_team = {}              # track_id -> [1. csapat szavazat, 2. csapat szavazat]
        self._ctl = None                 # az aktuális (esetleg még meg nem erősített) birtoklás
        self._ctl_last = None            # az utolsó megerősített birtoklás
        self._ctl_break_t = _NEG_INF     # játékmegszakítás / lövés / rögzített helyzet ideje (passz-lánc vége)
        self._ctl_foreign_t = _NEG_INF   # utolsó meg nem erősített ellenfél-érintés ideje
        self._flight = None              # a labda útja az utolsó megerősített birtoklás vége óta (beadáshoz)
        self._out_play_n = 0             # egymást követő, a pályán kívüli labdaminták
        # --- rögzített helyzetek ---
        self._touch_seq = None           # (oldal_y, n, t_első, x) – egymást követő minták az oldalvonalon túl
        self._line_out = None            # oldalvonalon túlra került labda: {side, t, x, last_team}
        self._sp_armed = None            # álló labda, végrehajtásra várva: {kind, x, y, t_still, ...}
        self._sp_wait = None             # kiadott rögzített helyzet, a csapata még ismeretlen
        self._sp_follow = None           # az utolsó rögzített helyzet (a következő birtoklás kimenetéhez)
        self._last_restart_t = _NEG_INF
        self._rest_free = None           # (rest objektum, szabad minták, összes) – a labda mellett nincs játékos
        self._prev_ball = None           # az előző elfogadott labdaminta (t, x, y)
        self._ball_switch_t = _NEG_INF   # az utolsó labda-track váltás ideje (_feed_candidate)
        self._unseen_max = 0.0           # a leghosszabb labda-kiesés az utolsó megerősített birtoklás óta
        self._last_goal_team = None
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
            if math.isfinite(bx) and math.isfinite(by) and self._add_ball(t, bx, by):
                self._bst = self._ball_state()
                ball = self._bst["pos"]

        st = self._bst if (self._bst is not None and t - self._bst["t"] <= 0.5) else None
        vel = st["vel"] if (st is not None and ball is not None) else None
        speed = st["speed"] if vel is not None else None
        closest, cdist = self._closest_player(ball, players)

        if ball is not None:
            self._update_possession(t, ball, speed, closest, cdist)
            self._track_zones(t, ball)
            self._update_rest(t, ball)
            self._check_goal(t, ball)
            if vel is not None and st["reliable"]:
                self._check_shot(t, ball, vel, speed, st["sv"])
        else:
            self._check_lost_out(t)

        self._validate_shot(t, ball, vel, speed, closest, cdist)
        self._resolve_shot(t, ball, vel, speed)
        self._update_goal_cand(t, ball)
        self._check_goal_kick(t, ball, speed)
        self._update_gk_possession(t, ball, speed, closest, cdist)
        self._check_sweeper(t)
        self._update_restarts(t, ball, speed, closest, cdist, players)
        self._update_control(t, ball, speed, closest, cdist, players)
        if ball is not None:
            self._prev_ball = (t, ball[0], ball[1])

        self._last_t = t
        return self._new

    def finalize(self, timestamp=None):
        """A klip végén lezárja a függő eseményeket (megerősítetlen védés, folyamatban lévő kihozatal)."""
        self._new = []
        t = timestamp if timestamp is not None else (self._last_t or 0.0)
        c = self._goal_cand
        if c is not None:
            self._finish_goal_cand(c, t, final=True)
        ps = self._pending_shot
        if ps is not None:
            if not ps["emitted"] and ps["speed"] * 3.6 >= 2.0 * self.cfg.shot_speed_kmh:
                self._emit_shot(ps)   # nagyon gyors, de még igazolatlan lövés a klip végén
            if ps["emitted"] and ps["contact"] is not None:
                self._emit_save(ps)
        self._pending_shot = None
        if self._dist is not None:
            self._finish_distribution()
        self._gk_poss = None
        self._close_flight(t, None, final=True)
        if self._ctl_last is not None:
            self._check_dribble(self._ctl_last)
        if self._sp_wait is not None:
            self._flush_restart(None)
        self._last_t = t
        return self._new

    def defending_team(self, goal_x, t=None):
        """
        Az adott kaput védő csapat: a mezőnyjátékosok átlagos helyzetéből (a bal kaput az a csapat védi,
        amelyiknek a játékosai átlagosan balrább állnak), elég minta híján a konfigurált irány szerint.
        A kapus mezszíne itt NEM számít: a kapus meze gyakran az ellenfél színéhez hasonlít, ezért a
        mezszín szerinti besorolás a valós felvételen rossz csapatot adott a kapusnak.
        """
        left = self._left_team()
        return left if goal_x == 0.0 else other_team(left)

    def _left_team(self):
        (m1, n1), (m2, n2) = self._side_x["team1"], self._side_x["team2"]
        if n1 >= 50 and n2 >= 50 and abs(m1 - m2) >= 2.0:
            self._side_left = "team1" if m1 < m2 else "team2"
        if self._side_left is not None:
            return self._side_left
        return "team1" if self.cfg.team1_attacks_right else "team2"

    def _update_sides(self, players):
        """A mezőnyjátékosok átlagos x-e csapatonként (lassan felejtő átlag: kb. az utolsó pár perc)."""
        for p in players:
            if p.get("is_keeper") or not p.get("team_known", True):
                continue
            team = f"team{p.get('team_id')}"
            st = self._side_x.get(team)
            if st is None:
                continue
            x = float(p["pitch_coords"][0])
            if not (0.0 <= x <= self.L):
                continue
            st[1] += 1
            a = 1.0 / st[1] if st[1] < 2000 else 1.0 / 2000
            st[0] += a * (x - st[0])

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
        typ = ev["type"]
        ev["time"] = round(float(t), 2)
        pre, post = _CLIP_WINDOWS.get(typ, (self.cfg.clip_pre_s, self.cfg.clip_post_s))
        ev.setdefault("start_sec", round(max(0.0, t - pre), 2))
        ev.setdefault("end_sec", round(t + post, 2))
        ev.setdefault("category", event_category(typ))
        if typ in EXPERIMENTAL_TYPES:
            ev.setdefault("experimental", True)
        self.events.append(ev)
        self._new.append(ev)
        team = ev.get("team")
        if typ.startswith("gk_") and team in self.stats["gk_events"]:
            bucket = self.stats["gk_events"][team]
            bucket[typ] = bucket.get(typ, 0) + 1
        key = _STAT_KEYS.get(typ)
        if key is not None and team in self.stats[key]:
            self.stats[key][team] += 1
        if typ == "goal":
            self._last_goal_team = team
        if typ in ("shot", "goal", "gk_goal_kick", "gk_long_ball", "gk_short_pass", "gk_save",
                   "gk_claim_cross", "gk_conceded"):
            # a lövés / gól / kapus-kihozatal megszakítja a mezőny-passzláncot
            self._ctl_break_t = max(self._ctl_break_t, float(t))
        return ev

    def _goal_xs(self):
        return (0.0, self.L)

    def _line_dist(self, p, goal_x):
        """Távolság a gólvonaltól a pálya felől (a vonal mögött 0)."""
        return max(0.0, p[0] - goal_x) if goal_x == 0.0 else max(0.0, goal_x - p[0])

    def _in_penalty_area(self, p, goal_x):
        return abs(p[0] - goal_x) <= 16.5 and abs(p[1] - self.W / 2.0) <= 20.16

    def _goal_area_of(self, p, tol=0.0):
        """Az ötös (kapuelőtér) – csak a pálya felőli oldalon (a gólvonal mögötti labda NEM az ötösben van)."""
        for gx in self._goal_xs():
            inside = (p[0] - gx) if gx == 0.0 else (gx - p[0])
            if -0.5 <= inside <= 5.5 + tol and abs(p[1] - self.W / 2.0) <= 9.16 + tol:
                return gx
        return None

    def _in_wide_zone(self, p):
        return p[1] < self.cfg.wide_zone_y or p[1] > self.W - self.cfg.wide_zone_y

    # --------------------------------------------------------- labda szűrés
    def _add_ball(self, t, x, y):
        """Kapuzás: a fizikailag lehetetlen ugrásokat (téves felvillanás) eldobjuk; ha a "kapun kívüli"
        minták egymással egybehangzók (új labda / elvesztett követés), néhány minta után átváltunk."""
        cfg = self.cfg
        raw = self._raw
        if raw:
            pt, px, py = raw[-1]
            dt = t - pt
            if dt <= 0:
                return False
            if dt <= cfg.ball_reacquire_s:
                gate = cfg.max_ball_speed_mps * dt + cfg.ball_gate_m
                d = math.hypot(x - px, y - py)
                st = self._bst
                if d > gate and st is not None and st["vel"] is not None:
                    ex = st["pos"][0] + st["vel"][0] * (t - st["t"])
                    ey = st["pos"][1] + st["vel"][1] * (t - st["t"])
                    d = min(d, math.hypot(x - ex, y - ey))
                if d > gate:
                    return self._feed_candidate(t, x, y)
        self._cand = None
        if raw and t - raw[-1][0] >= 2.0:
            # hosszabb kiesés utáni újra-megjelenés (pl. a labdát lehelyezték): a kiesés előtti állapot
            st = self._bst
            self._reappear = (t, t - raw[-1][0], st["pos"] if st else (raw[-1][1], raw[-1][2]),
                              st["vel"] if st else None)
        raw.append((t, x, y))
        while raw and t - raw[0][0] > 5.0:
            raw.popleft()
        return True

    def _feed_candidate(self, t, x, y):
        cfg = self.cfg
        c = self._cand
        if c and t - c[-1][0] <= 0.5 and math.hypot(x - c[-1][1], y - c[-1][2]) <= \
                cfg.max_ball_speed_mps * (t - c[-1][0]) + cfg.ball_gate_m:
            c.append((t, x, y))
        else:
            self._cand = c = [(t, x, y)]
        if len(c) >= cfg.ball_switch_samples:
            self._raw.clear()
            self._raw.extend(c)
            self._cand = None
            self._bst = None
            self._ball_switch_t = t   # új labda-track: a korábbi álló labdához kötött állapot érvénytelen
            return True
        return False

    def _ball_state(self):
        """Simított pozíció + sebesség az utolsó (kiesésmentes) szakasz csúszóablakos egyenes-illesztéséből."""
        cfg = self.cfg
        raw = self._raw
        last = raw[-1]
        seg = [last]
        for s in reversed(raw):
            if s is last:
                continue
            if seg[-1][0] - s[0] > cfg.ball_gap_s or last[0] - s[0] > cfg.vel_max_span_s:
                break
            seg.append(s)
        seg.reverse()
        win = [s for s in seg if last[0] - s[0] <= cfg.vel_window_s + 1e-9]
        if len(win) < cfg.vel_min_samples:
            win = seg[-cfg.vel_min_samples:]
        st = {"t": last[0], "pos": (last[1], last[2]), "vel": None, "speed": None, "sv": None,
              "reliable": False, "n": len(win)}
        if len(win) >= 3:
            fit = fit_line(win)
            if fit is not None:
                tm, xm, ym, vx, vy, rms, sv = fit
                if rms <= cfg.vel_max_resid_m:
                    xs = [s[1] for s in win]
                    ys = [s[2] for s in win]
                    px = min(max(xm + vx * (last[0] - tm), min(xs)), max(xs))
                    py = min(max(ym + vy * (last[0] - tm), min(ys)), max(ys))
                    st["pos"] = (px, py)
                    st["vel"] = (vx, vy)
                    st["speed"] = math.hypot(vx, vy)
                    st["sv"] = sv
                    st["reliable"] = (len(win) >= cfg.vel_min_samples
                                      and win[-1][0] - win[0][0] >= cfg.vel_min_span_s - 1e-9)
                elif rms <= 2.0 * cfg.vel_max_resid_m:
                    # irányváltás az ablakban: a sebesség csak tájékoztató, a pozíció a nyers utolsó
                    st["vel"] = (vx, vy)
                    st["speed"] = math.hypot(vx, vy)
                    st["sv"] = sv
        return st

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

    # ------------------------------------------------------------ kapusok
    def _update_keepers(self, t, players):
        cfg = self.cfg
        self._update_sides(players)
        by_team = {}
        for p in players:
            if not p.get("is_keeper"):
                continue
            # a kapus csapata a védett kapuból (pályaoldalból) dől el, nem a mezszínéből
            side_goal = 0.0 if float(p["pitch_coords"][0]) < self.L / 2.0 else self.L
            by_team.setdefault(self.defending_team(side_goal), []).append(p)
        for team, cands in by_team.items():
            k = self._keepers.get(team)
            if k is None:
                k = {"team": team, "hist": deque(), "last_sweep": _NEG_INF, "pos": None, "tid": None}
                self._keepers[team] = k
            p = cands[0]
            if len(cands) > 1 and k["pos"] is not None:
                p = min(cands, key=lambda c: math.hypot(c["pitch_coords"][0] - k["pos"][0],
                                                        c["pitch_coords"][1] - k["pos"][1]))
            x, y = float(p["pitch_coords"][0]), float(p["pitch_coords"][1])
            k["tid"] = p.get("track_id")
            k["hist"].append((t, x, y))
            while k["hist"] and t - k["hist"][0][0] > 10.0:
                k["hist"].popleft()
            win = [h for h in k["hist"] if t - h[0] <= cfg.keeper_smooth_s + 1e-9]
            k["pos"] = (sum(h[1] for h in win) / len(win), sum(h[2] for h in win) / len(win))
            k["last_t"] = t

    def _keeper_goal(self, k):
        hist = k["hist"]
        if not hist:
            return None
        mean_x = sum(h[1] for h in hist) / len(hist)
        return 0.0 if mean_x < self.L / 2.0 else self.L

    def _active_keepers(self, t, goal_x=None, max_age=0.5):
        out = []
        for k in self._keepers.values():
            if k["pos"] is None or t - k["last_t"] > max_age:
                continue
            if goal_x is not None and self._keeper_goal(k) != goal_x:
                continue
            out.append((k["tid"], k))
        return out

    def _defending_keeper(self, t, goal_x, max_age=1.0):
        ks = self._active_keepers(t, goal_x, max_age)
        if not ks:
            return None
        team = self.defending_team(goal_x, t)
        for tid, k in ks:
            if k["team"] == team:
                return k
        return ks[0][1]

    def _nearest_keeper_tid(self, t, goal_x, pos):
        best, best_d = None, None
        for tid, k in self._active_keepers(t, goal_x, 1.0):
            d = math.hypot(k["pos"][0] - pos[0], k["pos"][1] - pos[1])
            if best_d is None or d < best_d:
                best, best_d = tid, d
        if best is None:
            k = self._keepers.get(self.defending_team(goal_x, t))
            if k is not None:
                best = k["tid"]
        return best

    def _held_dist(self, ball, kpos):
        """A labda távolsága a kapus kezében (held_min..max magasan) tartott labda talajra vetített
        pontjaitól (a kamera-sugár mentén). Kamera nélkül: egyszerű távolság a kapustól."""
        d0 = math.hypot(ball[0] - kpos[0], ball[1] - kpos[1])
        cam = self._cam
        if cam is None:
            return d0
        cx, cy, cz = cam
        pts = []
        for h in (self.cfg.held_min_height_m, self.cfg.held_max_height_m):
            if h >= cz - 0.2:
                return d0
            k = cz / (cz - h)
            pts.append((cx + (kpos[0] - cx) * k, cy + (kpos[1] - cy) * k))
        # a kapus-pozíció hibája a vetítéssel arányosan nő: távoli vetületnél tágabb tűrés (a hívó
        # fix tűréssel hasonlít, ezért a távolságot csökkentjük)
        ds = _seg_dist(ball, pts[0], pts[1])
        return min(d0, max(0.0, ds - 0.08 * d0))

    def _keeper_has_ball(self, ball, k, goal_x, speed=None):
        """A kapusnál van-e a labda (közel, vagy – lassú labdánál – a kamera-sugár szerint a kezében)."""
        cfg = self.cfg
        d = math.hypot(ball[0] - k["pos"][0], ball[1] - k["pos"][1])
        if d <= cfg.goal_keeper_radius_m + 0.5 * cfg.keeper_jitter_m:
            return True
        if speed is not None and speed > cfg.possession_max_ball_speed_mps:
            return False   # a gyors labda nem lehet a kapus kezében (a sugár-egybeesés csak véletlen)
        return (abs(k["pos"][0] - goal_x) <= 9.0
                and self._held_dist(ball, k["pos"]) <= cfg.held_tolerance_m + 0.3 * cfg.keeper_jitter_m)

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
        self._poss_hist.append((t, team, closest.get("track_id")))
        while self._poss_hist and t - self._poss_hist[0][0] > 3.0:
            self._poss_hist.popleft()
        # Ha mezőnyjátékos a szélső sávon kívül átveszi a labdát, az már nem közvetlen beadás
        if not is_keeper and not self._in_wide_zone(ball):
            self._last_wide = None

    def _track_zones(self, t, ball):
        cfg = self.cfg
        x, y = ball
        if self._in_wide_zone(ball):
            self._last_wide = (t, x, y)
        # Alapvonalon túl, a kapun kívül (több egymást követő minta) -> kirúgás / szöglet várható
        gx = None
        if not (self.post_lo - 0.3 <= y <= self.post_hi + 0.3):
            if x < -cfg.goal_line_margin_m:
                gx = 0.0
            elif x > self.L + cfg.goal_line_margin_m:
                gx = self.L
        if gx is not None:
            n = self._out_seq[1] + 1 if (self._out_seq and self._out_seq[0] == gx) else 1
            self._out_seq = (gx, n)
            if n >= cfg.goal_min_samples:
                self._out[gx] = t
        else:
            self._out_seq = None
        for g in list(self._out.keys()):
            if abs(x - g) > 30.0:
                self._out_clear[g] = self._out_clear.get(g, 0) + 1
                if self._out_clear[g] >= 3:
                    del self._out[g]
                    self._out_clear.pop(g, None)
            else:
                self._out_clear.pop(g, None)
        for g in list(self._goal_block):
            if self._line_dist(ball, g) > 3.0:
                self._goal_block.discard(g)

    def _check_lost_out(self, t):
        """A gólvonal felé tartó, ott eltűnő labda (pl. magasan mellé szálló lövés vetülete a pályán
        kívülre esik) -> alapvonalon túl, ha a kapus nincs a labdánál és nem lett gól-jelölt."""
        st = self._bst
        if st is None or st["vel"] is None or self._lost_checked is st:
            return
        if t - st["t"] < 1.0:
            return
        self._lost_checked = st
        vx = st["vel"][0]
        for gx in self._goal_xs():
            if self._line_dist(st["pos"], gx) > 4.0:
                continue
            toward = vx > 0 if gx == self.L else vx < 0
            if not toward or abs(vx) < 4.0:
                continue
            if self._goal_cand is not None and self._goal_cand["goal_x"] == gx:
                continue
            k = self._defending_keeper(st["t"], gx)
            if k is not None and self._keeper_has_ball(st["pos"], k, gx, st["speed"]):
                continue
            if self.post_lo - 0.5 <= st["pos"][1] + st["vel"][1] * (abs(st["pos"][0] - gx) / abs(vx)) \
                    <= self.post_hi + 0.5 and self.post_lo <= st["pos"][1] <= self.post_hi:
                continue   # a kapu felé tartott: lehet gól vagy védés, nem jelöljük "kint"-nek
            self._out[gx] = st["t"]

    def _update_rest(self, t, ball):
        """Álló labda követése (kirúgás-lehelyezés, középkezdés); egy kiugró mintát tűr."""
        cfg = self.cfg
        r = self._rest
        if r is not None:
            cx, cy = r["sx"] / r["n"], r["sy"] / r["n"]
            d = math.hypot(ball[0] - cx, ball[1] - cy)
            if d <= cfg.goal_kick_still_radius_m:
                r["sx"] += ball[0]
                r["sy"] += ball[1]
                r["n"] += 1
                r["t_last"] = t
                r["miss"] = 0
            elif r["miss"] < 3 and d <= 3.0 * cfg.goal_kick_still_radius_m:
                r["miss"] += 1
            else:
                r = None
        if r is None:
            r = {"t0": t, "t_last": t, "sx": ball[0], "sy": ball[1], "n": 1, "miss": 0}
        self._rest = r
        if r["t_last"] - r["t0"] >= cfg.goal_kick_still_s and r["n"] >= 3:
            cx, cy = r["sx"] / r["n"], r["sy"] / r["n"]
            if (r["t_last"] - r["t0"] >= cfg.goal_kickoff_still_s
                    and math.hypot(cx - self.L / 2.0, cy - self.W / 2.0) <= cfg.goal_kickoff_radius_m):
                self._last_kickoff = (r["t0"], r["t_last"])

    def _near_centre(self, p):
        return math.hypot(p[0] - self.L / 2.0, p[1] - self.W / 2.0) <= self.cfg.goal_kickoff_radius_m + 0.5

    def _rest_point(self, t):
        r = self._rest
        if r is None or r["n"] < 3 or r["t_last"] - r["t0"] < self.cfg.goal_kick_still_s:
            return None
        return r["sx"] / r["n"], r["sy"] / r["n"]

    # --------------------------------------------------------------- gól
    def _check_goal(self, t, ball):
        cfg = self.cfg
        x, y = ball
        gx = None
        if self.post_lo - 0.2 <= y <= self.post_hi + 0.2:
            if x < -cfg.goal_line_margin_m:
                gx = 0.0
            elif x > self.L + cfg.goal_line_margin_m:
                gx = self.L
        if gx is None:
            if self._line_dist(ball, 0.0) > 0 and self._line_dist(ball, self.L) > 0:
                self._beyond = None
                self._last_in_field = t
            return
        b = self._beyond
        if b is not None and b[0] == gx:
            self._beyond = b = (gx, b[1] + 1, b[2], b[3])
        else:
            self._beyond = b = (gx, 1, t, self._last_in_field)
        if b[1] < cfg.goal_min_samples:
            return
        if self._goal_cand is not None or gx in self._goal_block:
            return
        if t - self._last_goal_t < cfg.goal_debounce_s:
            return
        t_first, t_in = b[2], b[3]
        t_cross = (t_in + t_first) / 2.0 if (t_in is not None and 0 < t_first - t_in <= 0.5) else t_first

        k = self._defending_keeper(t, gx)
        keeper_near = False
        ktid = k["tid"] if k is not None else None
        if k is not None:
            d = math.hypot(ball[0] - k["pos"][0], ball[1] - k["pos"][1])
            near_line = abs(k["pos"][0] - gx) <= cfg.goal_keeper_line_m + 0.5 * cfg.keeper_jitter_m
            if d <= cfg.goal_keeper_radius_m + 0.5 * cfg.keeper_jitter_m and near_line:
                keeper_near = True
            else:
                st = self._bst
                bspeed = st["speed"] if (st is not None and st["reliable"]) else None
                keeper_near = self._keeper_has_ball(ball, k, gx, bspeed)
        gp = self._gk_poss
        if gp is not None and gp["goal_x"] == gx and t - gp["t_last"] <= 1.0:
            keeper_near = True
        ps = self._pending_shot
        if ps is not None and (ps["goal_x"] != gx or t - ps["t"] > cfg.shot_resolve_window_s + 1.0):
            ps = None
        self._goal_cand = {
            "t": t_cross, "goal_x": gx, "keeper_near": keeper_near, "ktid": ktid,
            "keeper_team": k["team"] if k is not None else self.defending_team(gx, t),
            "behind_n": b[1], "behind_last": t, "returned": 0, "shot": ps, "ys": [y],
            "k0": k["pos"] if k is not None else None,
        }
        if ps is not None:
            ps["goal_cand"] = self._goal_cand

    def _update_goal_cand(self, t, ball):
        cfg = self.cfg
        c = self._goal_cand
        if c is None:
            return
        gx = c["goal_x"]
        if ball is not None and t > c["behind_last"]:
            beyond = (ball[0] < -0.2 if gx == 0.0 else ball[0] > self.L + 0.2)
            if beyond and self.post_lo - 1.0 <= ball[1] <= self.post_hi + 1.0:
                c["behind_n"] += 1
                c["behind_last"] = t
                c.setdefault("ys", []).append(ball[1])
            elif self._line_dist(ball, gx) > 1.5:
                c["returned"] += 1
                c.setdefault("first_return_t", t)
        if c["keeper_near"] and ball is not None and c["k0"] is not None:
            # A hálóban álló labdától eltávolodó kapus: nem a kezében volt (a vetület csak egybeesett)
            k = self._keepers.get(c["keeper_team"])
            r = self._rest
            if (k is not None and r is not None and r["t0"] <= c["t"] + 0.5 and r["t_last"] - r["t0"] >= 0.8
                    and math.hypot(ball[0] - r["sx"] / r["n"], ball[1] - r["sy"] / r["n"]) <= 1.0
                    and math.hypot(k["pos"][0] - c["k0"][0], k["pos"][1] - c["k0"][1]) >= 1.5
                    and self._held_dist(ball, k["pos"]) > cfg.held_tolerance_m):
                c["keeper_near"] = False
                gp = self._gk_poss
                if gp is not None and gp["team"] == c["keeper_team"]:
                    self._gk_poss = None
        ko = self._last_kickoff
        if ko is not None and c["t"] + 1.0 < ko[0] <= c["t"] + cfg.goal_kickoff_window_s:
            self._confirm_goal(c, "kickoff")
            return
        # Elvetés: a labda a szögletzászlónál áll meg (szöglet lesz belőle, nem gól)
        r = self._rest
        if (r is not None and r["t0"] > c["t"] and r["n"] >= 3 and r["t_last"] - r["t0"] >= 0.8
                and self._near_corner((r["sx"] / r["n"], r["sy"] / r["n"]), gx)):
            self._reject_goal(c, keeper=False)
            return
        if (not c["keeper_near"] and self._persist_ok(c) and c["returned"] < 3):
            self._confirm_goal(c, "persist")
            return
        # Elvetés: a játék közben folytatódott (új lövés, vagy a labda a túlsó térfélen jár) – a
        # későbbi középkezdés már nem ehhez a jelölthöz tartozik
        if ball is not None and self._line_dist(ball, gx) > self.L / 2.0 + 5.0:
            c["far_n"] = c.get("far_n", 0) + 1
        ps = self._pending_shot
        if c.get("far_n", 0) >= 5 or (ps is not None and ps["t"] > c["t"] + 1.0):
            self._finish_goal_cand(c, t, live_play=True)
            return
        # Elvetés: a védő kapus kihozza a labdát (védés volt), vagy kirúgás következik
        if any(ts > c["t"] and team == c["keeper_team"] for ts, team in self._dist_started):
            self._reject_goal(c, keeper=True)
            return
        if self._gk_armed is not None and self._gk_armed["goal_x"] == gx and self._gk_armed["t_still"] > c["t"]:
            self._reject_goal(c, keeper=c["keeper_near"])
            return
        # Lövés nélkül, de a kapuban maradt labda: a középkezdést várjuk (a labdát közben kiviszik a
        # kapuból – ez itt nem elvetés); a szöglet / kirúgás / folytatódó játék fent elveti.
        waiting_kickoff = c.get("shot") is None and not c["keeper_near"] and self._persisted_inside(c)
        if not c["keeper_near"] and c["returned"] >= 8 and not waiting_kickoff:
            if self._likely_unconfirmed(c):
                self._confirm_goal(c, "unconfirmed")
            else:
                self._reject_goal(c, keeper=False)
            return
        if t - c["t"] > cfg.goal_kickoff_window_s:
            self._finish_goal_cand(c, t)

    def _finish_goal_cand(self, c, t, final=False, live_play=False):
        cfg = self.cfg
        if c["keeper_near"] or (live_play and c["behind_last"] - c["t"] < cfg.goal_persist_s):
            self._reject_goal(c, keeper=True)
        elif self._persist_ok(c):
            self._confirm_goal(c, "persist")
        elif self._likely_unconfirmed(c):
            self._confirm_goal(c, "unconfirmed")
        else:
            self._reject_goal(c, keeper=False)

    def _persist_ok(self, c):
        """
        "A labda a gólvonal mögött maradt" megerősítés. Csak ha volt hozzá lövés (a lövés nélküli,
        a vonal mögött álló labda sokkal gyakrabban szöglet / kirúgás előtti labda, vagy a kalibráció
        hibája), és a labda a kapufák között, azoktól kissé beljebb jutott át. Különben a középkezdés
        erősítheti meg, vagy "Valószínű gól" lehet belőle.
        """
        return c.get("shot") is not None and self._persisted_inside(c)

    def _persisted_inside(self, c):
        """A labda elég ideig a vonal mögött volt, a kapufák között (azoktól kissé beljebb)."""
        cfg = self.cfg
        if c["behind_n"] < cfg.goal_persist_samples or c["behind_last"] - c["t"] < cfg.goal_persist_s:
            return False
        ys = sorted(c.get("ys") or [])
        if ys:
            ym = ys[len(ys) // 2]
            if not (self.post_lo + cfg.goal_post_inner_m <= ym <= self.post_hi - cfg.goal_post_inner_m):
                return False
        return True

    def _near_corner(self, p, gx):
        r = self.cfg.goal_corner_radius_m
        return (abs(p[0] - gx) <= r and (p[1] <= r or p[1] >= self.W - r))

    def _likely_unconfirmed(self, c):
        """Megerősítetlen, de valószínű gól (csak goal_emit_unconfirmed=True esetén): a labda több mintán
        át a vonal mögött volt, a kapus nem volt ott, és utána legalább 3 mp-ig nem tért vissza a pályára
        (nem kipattanó). Tipikus eset: a hálóban / középen álló labdát a detektor nem látja."""
        if not self.cfg.goal_emit_unconfirmed or c["keeper_near"] or c["behind_n"] < 3:
            return False
        ys = sorted(c.get("ys") or [])
        if ys and not (self.post_lo + self.cfg.goal_post_inner_m <= ys[len(ys) // 2]
                       <= self.post_hi - self.cfg.goal_post_inner_m):
            return False
        fr = c.get("first_return_t")
        return fr is None or fr - c["behind_last"] >= 3.0

    def _reject_goal(self, c, keeper):
        self._goal_cand = None
        gx = c["goal_x"]
        self._goal_block.add(gx)
        ps = c.get("shot")
        if keeper and ps is not None and self._pending_shot is ps:
            # A kapus kezében tartott labda vetülete esett a vonal mögé: védés
            if ps["contact"] is None:
                ps["contact"] = (c["t"], c["ktid"], c["keeper_team"])
            if not ps["emitted"]:
                self._emit_shot(ps)
            self._emit_save(ps)
            self._pending_shot = None
        elif not keeper:
            self._out[gx] = c["t"]   # nem gól -> a labda alapvonalon túl (kirúgás jöhet)

    def _confirm_goal(self, c, how):
        cfg = self.cfg
        self._goal_cand = None
        gx = c["goal_x"]
        t = c["t"]
        self._last_goal_t = t
        defending = c["keeper_team"] if c["keeper_team"] else self.defending_team(gx, t)
        scoring = other_team(defending)
        ps = c.get("shot")
        if ps is not None and self._pending_shot is ps:
            if not ps["emitted"]:
                ps["on_target"] = True
                self._emit_shot(ps)
            shot_ev = ps["ev"]
            if not shot_ev["on_target"]:
                # A gól utólag igazolja, hogy a lövés kaput talált
                shot_ev["on_target"] = True
                shot_ev["desc"] = shot_ev["desc"].replace("Kapu mellé tartó lövés", "Kaput eltaláló lövés")
                self.stats["shots_on_target"][shot_ev["team"]] += 1
            scoring = shot_ev["team"] if shot_ev["team"] != defending else scoring
        else:
            # Gól észlelt lövés nélkül (pl. lassú befejezés): pótlólag lövésként is rögzítjük
            self.stats["shots"][scoring] += 1
            self.stats["shots_on_target"][scoring] += 1
            ts = t
            for pt, pteam, ptid in reversed(self._poss_hist):
                if pteam == scoring and 0.0 <= t - pt <= 3.0:
                    ts = pt
                    break
            self._emit({
                "type": "shot", "time": ts, "team": scoring, "on_target": True, "inferred": True,
                "goal_x": gx,
                "desc": f"Kaput eltaláló lövés – {_team_label(scoring)} (a gól alapján utólag rögzítve)",
            })
        self._pending_shot = None

        confidence = {"kickoff": 0.95, "persist": 0.8, "unconfirmed": 0.5}[how]
        why = {"kickoff": "középkezdés követte", "persist": "a labda a gólvonal mögött maradt",
               "unconfirmed": "megerősítés nélkül"}[how]
        prefix = "GÓL" if how != "unconfirmed" else "Valószínű GÓL"
        self.stats["goals"][scoring] += 1
        self._emit({
            "type": "goal", "time": t, "team": scoring, "goal_x": gx,
            "confidence": confidence, "confirmed": how != "unconfirmed", "confirmation": how,
            "start_sec": round(max(0.0, t - 8.0), 2), "end_sec": round(t + 4.0, 2),
            "desc": f"{prefix} – {_team_label(scoring)} (a labda a kapufák között átjutott a gólvonalon; {why})",
        })
        ktid = c["ktid"] if c["ktid"] is not None else self._nearest_keeper_tid(t, gx, (gx, self.W / 2.0))
        self._emit({
            "type": "gk_conceded", "time": t, "team": defending, "keeper_track_id": ktid, "goal_x": gx,
            "confidence": confidence,
            "start_sec": round(max(0.0, t - 8.0), 2), "end_sec": round(t + 3.0, 2),
            "desc": f"Kapott gól – {_keeper_label(defending, ktid)}",
        })
        # A gól minden folyamatban lévő kapus-szekvenciát lezár
        self._gk_poss = None
        self._dist = None
        self._gk_armed = None
        self._out.pop(gx, None)
        self._goal_block.add(gx)

    # ------------------------------------------------------------ lövés
    def _check_shot(self, t, ball, vel, speed, sv):
        cfg = self.cfg
        vx, vy = vel
        speed_lo = speed - cfg.shot_speed_sigma_k * (sv or 0.0)
        if speed_lo * 3.6 < cfg.shot_speed_kmh:
            if speed * 3.6 < cfg.shot_speed_kmh:
                self._shot_dir = None
            self._shot_streak = None
            return
        if self._shot_dir is not None and t - self._last_shot_t > 4.0:
            self._shot_dir = None
        if self._shot_dir is not None and vx * self._shot_dir > 0:
            return  # ugyanaz a lövés még repül (nem számoljuk többször)
        if t - self._last_shot_t < cfg.shot_debounce_s:
            return
        bx, by = ball
        if bx < 0.0 or bx > self.L or abs(vx) / speed < cfg.shot_min_x_ratio:
            self._shot_streak = None
            return
        goal_x = self.L if vx > 0 else 0.0
        line_dist = abs(goal_x - bx)
        t_cross = (goal_x - bx) / vx
        if line_dist > cfg.shot_max_distance_m or t_cross <= 0:
            self._shot_streak = None
            return
        y_cross = by + vy * t_cross
        on_target = (self.post_lo - cfg.post_tolerance_m) <= y_cross <= (self.post_hi + cfg.post_tolerance_m)
        if not on_target and abs(y_cross - self.W / 2.0) > cfg.off_target_band_m:
            self._shot_streak = None
            return  # valószínűleg hosszú passz / beadás, nem lövés

        defending = self.defending_team(goal_x, t)
        poss = self._poss
        shooter = None
        team = other_team(defending)
        if poss is not None and t - poss[3] <= cfg.possession_memory_s:
            # A labdabirtokló csapata a lövés előtti utolsó ~1,5 mp birtoklási mintáiból (többségi
            # szavazás: zajos pozícióknál egy-egy közeli ellenfél ne "vegye el" a labdát)
            votes = {}
            for pt, pteam, ptid in self._poss_hist:
                if poss[3] - pt <= 1.5:
                    votes[pteam] = votes.get(pteam, 0) + 1
            if votes:
                best = max(votes.items(), key=lambda kv: kv[1])
                total = sum(votes.values())
                if best[0] == defending:
                    if best[1] >= 3 and best[1] >= 0.75 * total:
                        self._shot_streak = None
                        return  # a védekező csapat a saját kapuja felé: visszapassz / felszabadítás
                else:
                    team = best[0]
                    shooter = poss[0] if poss[1] == team else None

        sk = self._shot_streak
        if sk is not None and sk["goal_x"] == goal_x and t - sk["t_last"] <= 0.3:
            sk["n"] += 1
            sk["t_last"] = t
        else:
            self._shot_streak = sk = {"t0": t, "t_last": t, "n": 1, "goal_x": goal_x, "ball": ball}
        if sk["n"] < cfg.shot_confirm_samples:
            return
        self._shot_streak = None
        t0 = sk["t0"]
        p0 = sk["ball"]
        # A lövés pillanata: a lövő csapat utolsó labdaérintése, ha az közvetlenül a gyors szakasz előtt volt
        # (a kiesések / simítás miatt a gyors labdát gyakran csak később látjuk)
        for pt, pteam, ptid in reversed(self._poss_hist):
            if pteam == team and 0.0 <= t0 - pt <= 1.2:
                t0 = pt
                break

        self._last_shot_t = t
        self._shot_dir = 1.0 if vx > 0 else -1.0
        dist = math.hypot(goal_x - p0[0], self.W / 2.0 - p0[1])
        self._pending_shot = {
            "t": t0, "t_detect": t, "goal_x": goal_x, "team": team, "shooter": shooter,
            "on_target": bool(on_target), "speed": speed, "dist": dist,
            # a vetítési torzítás miatt a "mellé" becslés bizonytalan: a védés-vizsgálathoz tágabb sáv
            "toward_goal": bool(on_target) or abs(y_cross - self.W / 2.0) <= 3.66 + cfg.save_target_margin_m,
            "dir": 1.0 if vx > 0 else -1.0, "defending": defending,
            "contact": None, "emitted": False, "ev": None, "goal_cand": None,
        }

    def _emit_shot(self, ps):
        if ps["emitted"]:
            return ps["ev"]
        ps["emitted"] = True
        team = ps["team"]
        on_target = ps["on_target"]
        self.stats["shots"][team] += 1
        if on_target:
            self.stats["shots_on_target"][team] += 1
        kmh = ps["speed"] * 3.6
        kind = "Kaput eltaláló lövés" if on_target else "Kapu mellé tartó lövés"
        ps["ev"] = self._emit({
            "type": "shot", "time": ps["t"], "team": team, "on_target": bool(on_target),
            "speed_kmh": round(kmh, 1), "goal_x": ps["goal_x"], "shooter_track_id": ps["shooter"],
            "distance_m": round(ps["dist"], 1),
            "desc": f"{kind} – {_team_label(team)}, {kmh:.0f} km/h, {ps['dist']:.0f} m-ről",
        })
        return ps["ev"]

    def _drop_shot(self):
        """A lövés-jelölt passznak bizonyult: törlés, és a lövés-tiltás feloldása."""
        self._pending_shot = None
        self._last_shot_t = _NEG_INF
        self._shot_dir = None

    def _validate_shot(self, t, ball, vel, speed, closest, cdist):
        """A gyors, kapu felé tartó labda csak akkor lövés, ha eléri a kaput / a kapust / a gólvonalnál
        tűnik el, vagy visszapattan – ha egy csapattárs veszi át messze a kaputól, passz volt."""
        cfg = self.cfg
        ps = self._pending_shot
        if ps is None or ps["emitted"]:
            return
        gx = ps["goal_x"]
        if ball is not None:
            ld = self._line_dist(ball, gx)
            if ld <= cfg.shot_valid_line_m:
                self._emit_shot(ps)
                return
            if vel is not None and vel[0] * ps["dir"] < -3.0 and ld <= 12.0 and t - ps["t_detect"] > 0.1:
                self._emit_shot(ps)   # visszapattanó / blokkolt lövés a kapu előtt
                return
            slow = speed is None or speed <= cfg.possession_max_ball_speed_mps
            if closest is not None and cdist <= cfg.possession_radius_m and slow and t - ps["t_detect"] > 0.1:
                cteam = f"team{closest.get('team_id', 1)}"
                if closest.get("is_keeper") and cteam == ps["defending"]:
                    self._emit_shot(ps)
                    return
                if cteam == ps["team"] and ld > 4.0:
                    self._drop_shot()
                    return
                if cteam != ps["team"] and ld > 11.0:
                    self._drop_shot()
                    return
            if speed is not None and speed < 2.0 and ld > 8.0 and t - ps["t_detect"] > 0.5:
                self._drop_shot()
                return
        else:
            st = self._bst
            if st is not None and st["vel"] is not None and t - st["t"] >= 0.6:
                ld = self._line_dist(st["pos"], gx)
                vxs = st["vel"][0] * ps["dir"]
                if vxs > 3.0 and ld <= 14.0 and st["t"] + ld / vxs <= t:
                    self._emit_shot(ps)   # a kapu felé repülő labda eltűnt (vetület a pályán kívül / a kapusnál)
                    return
        if t - ps["t"] > cfg.shot_validate_s:
            self._drop_shot()

    def _resolve_shot(self, t, ball, vel, speed):
        cfg = self.cfg
        ps = self._pending_shot
        if ps is None:
            return
        if ps["contact"] is None:
            if t - ps["t"] > cfg.shot_resolve_window_s:
                if ps["emitted"] and ps.get("goal_cand") is not None and self._goal_cand is ps["goal_cand"]:
                    return   # a gól-jelölt dönt
                self._pending_shot = None
                return
            gx = ps["goal_x"]
            k = self._defending_keeper(t, gx)
            if k is None:
                return
            if ball is not None:
                d = math.hypot(k["pos"][0] - ball[0], k["pos"][1] - ball[1])
                reversed_dir = vel is not None and vel[0] * ps["dir"] < 0
                stopped = speed is not None and speed < 3.0
                jit = 0.5 * cfg.keeper_jitter_m
                in_posts = self.post_lo - 0.5 <= ball[1] <= self.post_hi + 0.5
                behind = (ball[0] < -0.5) if gx == 0.0 else (ball[0] > self.L + 0.5)
                if ps["toward_goal"]:
                    held = (abs(k["pos"][0] - gx) <= 9.0 and d > 1.0
                            and self._held_dist(ball, k["pos"]) <= cfg.held_tolerance_m)
                    slowed = speed is not None and speed < 0.5 * ps["speed"]
                    if (not behind or in_posts) and (
                            d <= 1.2
                            or (d <= cfg.save_contact_radius_m + jit and (reversed_dir or stopped or slowed))
                            or (d <= cfg.save_loose_radius_m + jit and (reversed_dir or stopped))
                            or (held and (stopped or slowed or speed is None))):
                        ps["contact"] = (t, k["tid"], k["team"])
                elif not behind and d <= cfg.save_contact_radius_m and (reversed_dir or stopped or d <= 1.2):
                    # kapu mellé tartó lövés: csak egyértelmű érintés (a pálya felőli oldalon)
                    ps["contact"] = (t, k["tid"], k["team"])
            else:
                # A labda a kapusnál tűnt el (kézben tartva a vetülete a pályán kívülre esik)
                st = self._bst
                if st is not None and t - st["t"] >= cfg.save_vanish_s and st["t"] >= ps["t"]:
                    d = math.hypot(k["pos"][0] - st["pos"][0], k["pos"][1] - st["pos"][1])
                    slow_last = st["speed"] is not None and st["speed"] <= cfg.possession_max_ball_speed_mps
                    near_r = (cfg.save_loose_radius_m if ps["toward_goal"] else cfg.save_contact_radius_m) \
                        + 0.5 * cfg.keeper_jitter_m
                    heading = False
                    if st["vel"] is not None and st["speed"] and d <= 9.0:
                        # a kapus felé repülő labda tűnt el (elkapta; a kézben tartott labda vetülete kiesik)
                        ux, uy = st["vel"][0] / st["speed"], st["vel"][1] / st["speed"]
                        rx, ry = k["pos"][0] - st["pos"][0], k["pos"][1] - st["pos"][1]
                        along = rx * ux + ry * uy
                        heading = 0.0 < along <= 8.0 and abs(rx * uy - ry * ux) <= 4.5
                    if d <= near_r or heading or (ps["toward_goal"] and
                            slow_last and abs(k["pos"][0] - gx) <= 9.0
                            and self._held_dist(st["pos"], k["pos"]) <= cfg.held_tolerance_m):
                        ps["contact"] = (st["t"], k["tid"], k["team"])
            if ps["contact"] is not None:
                if not ps["emitted"]:
                    self._emit_shot(ps)
                # A kapus megfogta / magánál tartja: kapus-birtoklás (akkor is, ha a labda nem látszik)
                if self._gk_poss is None:
                    kp = k["pos"]
                    self._gk_poss = {"tid": k["tid"], "team": k["team"], "goal_x": gx,
                                     "t_start": ps["contact"][0], "t_last": ps["contact"][0],
                                     "ball": kp, "in_hands": True}
        else:
            c = ps.get("goal_cand")
            if c is not None and self._goal_cand is c:
                return   # a gól-jelölt dönt (gól vagy védés)
            if t - ps["contact"][0] >= cfg.save_confirm_s:
                self._emit_save(ps)
                self._pending_shot = None

    def _emit_save(self, ps):
        if ps.get("saved"):
            return
        ps["saved"] = True
        if not ps["emitted"]:
            self._emit_shot(ps)
        ct, tid, team = ps["contact"]
        shot_ev = ps["ev"]
        self._emit({
            "type": "gk_save", "time": ct, "team": team, "keeper_track_id": tid,
            "shot_time": shot_ev["time"], "goal_x": ps["goal_x"],
            "desc": f"Kapus védés – {_keeper_label(team, tid)}, "
                    f"{_team_label(shot_ev['team'])} lövése ({shot_ev.get('speed_kmh', '?')} km/h)",
        })

    # -------------------------------------------------------- kirúgás
    def _check_goal_kick(self, t, ball, speed):
        """Kirúgás: alapvonalon túlra került labda (nem gól) után a labda az ötösben ≥ goal_kick_still_s
        ideig áll, majd elrúgják. Az esemény ideje az elrúgás (az utolsó álló minta)."""
        cfg = self.cfg
        if ball is None:
            return
        armed = self._gk_armed
        if armed is not None:
            d = math.hypot(ball[0] - armed["x"], ball[1] - armed["y"])
            dt = max(t - armed["t_still"], 1e-3)
            if d >= cfg.goal_kick_launch_dist_m:
                self._gk_armed = None
                self._rest = None
                fast = d / dt >= cfg.goal_kick_min_speed_mps
                # magas (vetítve a pályán kívülre eső) kirúgás: a labda csak a földet érés körül látszik újra
                if fast and (dt <= 1.5 or (dt <= 4.0 and d >= 15.0)):
                    self._start_goal_kick(t, armed)
                return
            if d > 1.5 * cfg.goal_kick_still_radius_m:
                if dt > 2.0:
                    self._gk_armed = None  # lassan elgurult, nem kirúgás
                return
            armed["t_still"] = t
            return

        rp = self._rest_point(t)
        if rp is None:
            return
        goal_x = self._goal_area_of(rp, cfg.goal_kick_area_tol_m)
        if goal_x is None:
            fk = self._keeper_free_kick(rp)
            if fk is not None:
                self._gk_armed = {"goal_x": fk, "x": rp[0], "y": rp[1], "t_still": self._rest["t_last"],
                                  "kind": "free_kick"}
            return
        out_t = self._out.get(goal_x)
        cand = self._goal_cand
        if cand is not None and cand["goal_x"] == goal_x and not cand["keeper_near"]:
            out_t = max(out_t if out_t is not None else _NEG_INF, cand["t"])
        if out_t is None or t - out_t > cfg.goal_kick_out_memory_s or self._rest["t0"] < out_t:
            if not self._goal_kick_fallback(t, goal_x):
                return
        self._gk_armed = {"goal_x": goal_x, "x": rp[0], "y": rp[1], "t_still": self._rest["t_last"]}

    def _keeper_free_kick(self, rp):
        """
        Kapus szabadrúgás (pl. les után): az álló labda a saját térfelén, a kapujához közel van, és
        a saját kapusa áll mellette. Visszatér: a kapu x-e, vagy None.
        """
        cfg = self.cfg
        for k in self._keepers.values():
            gx = self._keeper_goal(k)
            if self._line_dist(rp, gx) > cfg.gk_free_kick_max_line_dist_m:
                continue
            if math.hypot(k["pos"][0] - rp[0], k["pos"][1] - rp[1]) <= cfg.gk_free_kick_keeper_radius_m:
                return gx
        return None

    def _goal_kick_fallback(self, t, goal_x):
        """Ha az alapvonalon túlra kerülést nem láttuk (pl. a magasan mellé szálló labda vetülete a
        pályán kívülre esett), akkor is kirúgás: a labda hosszabb kiesés után már állva jelenik meg az
        ötösben (lehelyezték), a kiesés előtt e kapu térfelén / felé mozgott, és nem a kapus kezében van."""
        cfg = self.cfg
        ra = self._reappear
        r = self._rest
        if ra is None or r is None or abs(r["t0"] - ra[0]) > 1.0 or r["t_last"] - r["t0"] < 1.5:
            return False
        if t - ra[0] > cfg.goal_kick_out_memory_s:
            return False
        gp = self._gk_poss
        if gp is not None and gp.get("in_hands"):
            return False
        ps = self._pending_shot
        if ps is not None and ps["contact"] is not None:
            return False
        last_pos, last_vel = ra[2], ra[3]
        toward = last_vel is not None and ((last_vel[0] > 2.0) if goal_x == self.L else (last_vel[0] < -2.0))
        return self._line_dist(last_pos, goal_x) <= 25.0 and (toward or self._line_dist(last_pos, goal_x) <= 6.0)

    def _start_goal_kick(self, t, armed):
        goal_x = armed["goal_x"]
        team = self.defending_team(goal_x, t)
        tid = self._nearest_keeper_tid(t, goal_x, (armed["x"], armed["y"]))
        self._out.pop(goal_x, None)
        self._gk_poss = None
        self._dist = {
            # a kapus szabadrúgása kihozatalként (rövid passz / ívelés) számít, nem kirúgásként
            "kind": "free_kick" if armed.get("kind") == "free_kick" else "goal_kick",
            "tid": tid, "team": team, "goal_x": goal_x,
            "t0": armed["t_still"], "p0": (armed["x"], armed["y"]), "max_d": 0.0,
            "last_t": armed["t_still"], "still_t": None,
        }
        self._dist_started.append((armed["t_still"], team))
        self._ctl_break_t = max(self._ctl_break_t, armed["t_still"])
        self._ctl_last = None
        self._sp_armed = None
        self._line_out = None
        self._last_restart_t = armed["t_still"]
        if armed.get("kind") == "free_kick":
            # A kapus szabadrúgása rögzített helyzetként is megjelenik (a kihozatal gk_* eseménye mellett)
            self._emit_free_kick(armed["t_still"], (armed["x"], armed["y"]), team, by_keeper=True, keeper_tid=tid)

    # ------------------------------------------- kapus birtoklás / kihozatal
    def _update_gk_possession(self, t, ball, speed, closest, cdist):
        cfg = self.cfg
        if self._dist is not None:
            self._advance_distribution(t, ball, speed, closest, cdist)
        self._dist_started = [d for d in self._dist_started if t - d[0] <= 120.0]

        gp = self._gk_poss
        if ball is None:
            if gp is not None:
                limit = cfg.gk_hold_max_s if gp.get("in_hands") else 3.0
                if t - gp["t_last"] > limit:
                    self._gk_poss = None
            return

        slow = speed is None or speed <= cfg.possession_max_ball_speed_mps
        if ball[0] < -0.5 or ball[0] > self.L + 0.5:
            # a gólvonal mögötti labda nincs játékban: nem kapus-birtoklás, és nem is kiadás
            # (kivéve a kézben tartott labda vetületét – az a kapus kamera-sugarán van)
            if gp is None or not gp.get("in_hands"):
                return
        # Birtokló kapus: a legközelebbi játékos a kapus, vagy a kapus (remegést tűrve) a labdánál /
        # a kamera-sugár szerint a kezében tartja
        holder = None
        for tid, k in self._active_keepers(t):
            d = math.hypot(k["pos"][0] - ball[0], k["pos"][1] - ball[1])
            gx = self._keeper_goal(k)
            near = d <= cfg.possession_radius_m or (
                d <= cfg.possession_radius_m + 0.5 * cfg.keeper_jitter_m
                and (closest is None or closest.get("is_keeper") or cdist > 0.6 * d))
            held = False
            if not near and abs(k["pos"][0] - gx) <= 16.5 and \
                    self._held_dist(ball, k["pos"]) <= cfg.held_tolerance_m:
                hs = self._held_seq
                if gp is not None and gp["team"] == k["team"]:
                    held = True
                else:
                    n = hs[1] + 1 if (hs is not None and hs[0] == k["team"] and t - hs[2] <= 0.5) else 1
                    self._held_seq = (k["team"], n, t)
                    held = n >= 3
            if slow and (near or held):
                if holder is None or d < holder[0]:
                    holder = (d, tid, k, gx, held)
        if holder is not None:
            d, tid, k, goal_x, held = holder
            team = k["team"]
            if gp is None or gp["team"] != team:
                dd = self._dist
                if dd is not None and dd["kind"] == "gk" and dd["team"] == team:
                    self._dist = None  # a kapus visszaszerezte, nem volt valódi kihozatal
                self._gk_poss = {"tid": tid, "team": team, "goal_x": goal_x,
                                 "t_start": t, "t_last": t, "ball": k["pos"] if held else ball,
                                 "in_hands": held, "ball_obs": (t, ball[0], ball[1])}
                self._check_cross_claim(t, ball if not held else k["pos"], tid, team, goal_x)
            else:
                gp["t_last"] = t
                gp["tid"] = tid
                gp["ball"] = k["pos"] if held else ball
                gp["in_hands"] = gp.get("in_hands") or held
                gp["ball_obs"] = (t, ball[0], ball[1])
            return

        if gp is None:
            return
        k = self._keepers.get(gp["team"])
        ref = k["pos"] if (k and t - k["last_t"] <= 0.5) else gp["ball"]
        dk = math.hypot(ball[0] - ref[0], ball[1] - ref[1])
        if dk <= cfg.gk_keep_radius_m:
            gp["t_last"] = t
            gp["ball"] = ball
            gp["ball_obs"] = (t, ball[0], ball[1])
        elif dk > cfg.gk_release_radius_m + 0.5 * cfg.keeper_jitter_m:
            self._gk_poss = None
            obs = gp.get("ball_obs")
            if speed is not None:
                standing = speed < 1.5
            else:
                standing = obs is not None and t - obs[0] <= 1.0 and math.hypot(ball[0] - obs[1], ball[1] - obs[2]) < 1.0
            if standing or (self._near_centre(ball) and dk > 20.0 and (speed is None or speed < 3.0)):
                return   # a labda áll (a kapus távolodott el tőle / középkezdésre vár): nem kiadás
            if gp["t_last"] - gp["t_start"] < 0.4 and t - gp["t_last"] < 0.5:
                return   # csak érintette (kiütés / lepattanó), nem birtokolta: nem kiadás
            if self._dist is None:
                t0 = gp["t_last"]
                if t - t0 > 1.0:
                    # a labda nem látszott a kiadáskor (kézben / magasan): becsült kiadási idő
                    t0 = max(t0, t - dk / cfg.gk_throw_speed_mps)
                self._dist = {
                    "kind": "gk", "tid": gp["tid"], "team": gp["team"], "goal_x": gp["goal_x"],
                    "t0": t0, "p0": gp["ball"], "max_d": 0.0, "last_t": t, "still_t": None,
                }
                self._dist_started.append((t0, gp["team"]))
                self._advance_distribution(t, ball, speed, closest, cdist)

    def _advance_distribution(self, t, ball, speed, closest, cdist):
        cfg = self.cfg
        d = self._dist
        if d is None:
            return
        if ball is not None:
            dd = math.hypot(ball[0] - d["p0"][0], ball[1] - d["p0"][1])
            if dd > d["max_d"] + 3.0:
                d["max_n"] = 1
            elif dd >= d["max_d"] - 3.0:
                d["max_n"] = d.get("max_n", 0) + 1
            d["max_d"] = max(d["max_d"], dd)
            if (d.get("max_n", 0) >= 3 and d["max_d"] >= cfg.min_distribution_m
                    and dd < d["max_d"] - cfg.distribution_return_m):
                # A labda (a nem látott átvétel után) jóval a legtávolabbi pontja elől jön vissza: a kihozatal
                # hossza a legtávolabbi (legalább három mintával alátámasztott) pont, nem a mostani helyzet
                # (valós hiba: az 50 m-es kapus-ívelés 31 m-es rövid passznak látszott).
                d["end_d"] = d["max_d"]
                self._finish_distribution()
                return
            d["end_d"] = dd
            d["last_t"] = t
            slow = speed is None or speed <= cfg.possession_max_ball_speed_mps
            own_keeper = (closest is not None and closest.get("is_keeper")
                          and f"team{closest.get('team_id', 1)}" == d["team"] and dd < 6.0)
            if self._near_centre(ball) and (speed is None or speed < 3.0) and dd > 20.0:
                # a labda a középkezdés helyére került (gól után): nem kapus-kihozatal
                self._dist = None
                return
            if (closest is not None and cdist <= cfg.possession_radius_m and slow and speed is not None
                    and not own_keeper and t - d["t0"] > 0.2 and dd >= 2.0):
                self._finish_distribution()
                return
            if speed is not None and speed < 1.5:
                if d["still_t"] is None:
                    d["still_t"] = t
                elif t - d["still_t"] >= 0.5:
                    self._finish_distribution()
                    return
            else:
                d["still_t"] = None
        if t - d["t0"] > cfg.distribution_timeout_s or (ball is None and t - d["last_t"] > cfg.distribution_lost_s):
            self._finish_distribution()

    def _finish_distribution(self):
        cfg = self.cfg
        d = self._dist
        self._dist = None
        # A megtett út: a labda utolsó (átvételi / megállási) pontjáig mért távolság. A legnagyobb
        # távolság helyett, mert a levegőben lévő labda vetülete messzebbre "esik" a valódinál.
        dist = d.get("end_d", d["max_d"])
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
        if self._pending_shot is not None:
            return   # lövés után a megfogás védés, nem beadás-elfogás
        self._last_wide = None
        self._emit({
            "type": "gk_claim_cross", "time": t, "team": team, "keeper_track_id": tid, "goal_x": goal_x,
            "desc": f"Beadás elfogása – {_keeper_label(team, tid)}",
        })

    # ------------------------------------------------------------ kifutás
    def _sweeper_context_ok(self, t, k, vx, vy, sp):
        """
        Valódi kifutás: a kapus a (mozgó, nem a nála lévő) labda felé fut. Nem kifutás (valós próbán
        tévesen az lett): a labdát kézben vivő kapus kifut a 16-os széléig, hogy kirúgja; a kapus
        az álló labdához megy (szabadrúgás, kirúgás); követési zaj, amikor a labda messze van.
        """
        cfg = self.cfg
        gp = self._gk_poss
        if gp is not None and gp["team"] == k["team"] and t - gp["t_last"] <= 2.0:
            return False                         # nála van a labda (kézben / lábnál)
        d = self._dist
        if d is not None and d["team"] == k["team"]:
            return False                         # éppen kihozza / elrúgja a labdát
        st = self._bst
        if st is None or t - st["t"] > 0.5:
            return False                         # labda nélkül nem dönthető el
        r = self._rest
        if r is not None and r["t_last"] >= t - 0.2 and r["t_last"] - r["t0"] >= 0.8:
            return False                         # álló labda: rögzített helyzet, nem kifutás
        bx, by = st["pos"]
        kx, ky = k["pos"]
        dbx, dby = bx - kx, by - ky
        dist = math.hypot(dbx, dby)
        if dist > cfg.sweeper_max_ball_dist_m:
            return False
        if dist > 1.0 and (vx * dbx + vy * dby) / (dist * sp) < cfg.sweeper_toward_ball_cos:
            return False                         # nem a labda felé fut
        return True

    def _check_sweeper(self, t):
        cfg = self.cfg
        for k in self._keepers.values():
            if k["last_t"] != t:
                continue
            win = [h for h in k["hist"] if t - h[0] <= 1.0 + 1e-9]
            if len(win) < 8 or win[-1][0] - win[0][0] < 0.5:
                continue
            fit = fit_line(win)
            if fit is None:
                continue
            vx, vy, sv = fit[3], fit[4], fit[6]
            sp = math.hypot(vx, vy)
            if sp - sv < cfg.sweeper_speed_mps or sp > cfg.keeper_max_speed_mps:
                continue
            goal_x = self._keeper_goal(k)
            x, y = k["pos"]
            if abs(x - goal_x) < cfg.sweeper_min_line_dist_m:
                continue
            rx, ry = x - goal_x, y - self.W / 2.0
            rn = math.hypot(rx, ry)
            if rn <= 1e-6 or (vx * rx + vy * ry) / rn < 0.3 * sp:
                continue  # nem a kaputól távolodik (pl. visszafut)
            if not self._sweeper_context_ok(t, k, vx, vy, sp):
                continue
            if t - k["last_sweep"] < cfg.sweeper_debounce_s:
                continue
            k["last_sweep"] = t
            self._emit({
                "type": "gk_sweeper", "time": t, "team": k["team"], "keeper_track_id": k["tid"],
                "goal_x": goal_x, "speed_mps": round(sp, 1),
                "distance_from_goal_m": round(abs(x - goal_x), 1),
                "desc": f"Kapus kifutás – {_keeper_label(k['team'], k['tid'])}, "
                        f"{abs(x - goal_x):.0f} m-re a gólvonaltól, {sp:.1f} m/s",
            })

    # ================================================== labdás mezőny-események
    # Birtoklási szakaszok: egy szakasz ugyanannak a csapatnak (és – ID-cserét tűrve – ugyanannak a
    # játékosnak) az egymást követő birtoklási mintái. Egy szakasz csak ctl_min_samples minta után
    # "megerősített"; két egymást követő megerősített szakaszból dől el a passz / sikertelen passz /
    # szerelés. A zajos (egykamerás) bemenet miatt a pontosság fontosabb a felidézésnél: kétes
    # helyzetben (vitatott labda, közbeeső ellenfél-érintés, cselezés, játékmegszakítás) nincs esemény.

    def _attack_goal(self, team):
        """A csapat által támadott kapu x-e (a mezőnyjátékosok átlagos helyzetéből, lásd defending_team)."""
        return self.L if self.defending_team(0.0) == team else 0.0

    def _in_box(self, p, goal_x, tol=0.0):
        return abs(p[0] - goal_x) <= 16.5 + tol and abs(p[1] - self.W / 2.0) <= 20.16 + tol

    def _zone_of(self, p, team):
        """A pont a csapat saját vagy az ellenfél térfelén van (a csapat támadási iránya szerint)."""
        return "ellenfel_terfel" if self._line_dist(p, self._attack_goal(team)) < self.L / 2.0 else "sajat_terfel"

    def _player_team(self, p):
        """A játékos csapata a követés eddigi többségi besorolásából (a képkockánkénti mezszín-besorolás
        egy-egy kockán téveszthet); kevés / bizonytalan szavazatnál a képkocka szerinti besorolás.
        A kapus csapata – mint a _update_keepers-ben – a védett kapuból (pályaoldalból) dől el."""
        if p.get("is_keeper"):
            return self.defending_team(0.0 if float(p["pitch_coords"][0]) < self.L / 2.0 else self.L)
        tid = p.get("track_id")
        v = self._tid_team.get(tid) if tid is not None else None
        if v is not None and v[0] + v[1] >= 5:
            if v[0] >= 2.0 * v[1]:
                return "team1"
            if v[1] >= 2.0 * v[0]:
                return "team2"
        return f"team{p.get('team_id', 1)}"

    def _vote_teams(self, players):
        for p in players:
            tid = p.get("track_id")
            try:
                tm = int(p.get("team_id"))
            except (TypeError, ValueError):
                continue
            if tid is None or tm not in (1, 2) or not p.get("team_known", True):
                continue
            v = self._tid_team.get(tid)
            if v is None:
                v = self._tid_team[tid] = [0.0, 0.0]
            v[tm - 1] += 1.0
            if v[0] + v[1] > 60.0:      # lassan felejt (újrahasznosított / cserélt ID)
                v[0] *= 0.5
                v[1] *= 0.5

    def _update_control(self, t, ball, speed, closest, cdist, players):
        cfg = self.cfg
        self._vote_teams(players)
        if ball is None:
            return
        pb = self._prev_ball
        if pb is not None:
            self._unseen_max = max(self._unseen_max, t - pb[0])
        self._update_flight(t, ball)
        x, y = ball
        m = cfg.touch_out_margin_m
        if x < -m or x > self.L + m or y < -m or y > self.W + m:
            self._out_play_n += 1
            if self._out_play_n >= 2:   # a labda elhagyta a pályát: a passzlánc megszakad
                self._ctl_break_t = max(self._ctl_break_t, t)
                self._close_flight(t, None)
            return
        self._out_play_n = 0
        if closest is None or cdist > cfg.ctl_radius_m:
            return
        if speed is not None and speed > cfg.ctl_max_ball_speed_mps:
            return
        team = self._player_team(closest)
        snap = []
        for p in players:
            px, py = p["pitch_coords"]
            d = math.hypot(px - x, py - y)
            if d > 12.0:
                continue
            pteam = self._player_team(p)
            if p is not closest and pteam != team and d <= cdist + cfg.ctl_contest_margin_m:
                return   # vitatott labda: egy ellenfél majdnem ugyanolyan közel
            snap.append((p.get("track_id"), pteam, px, py))
        tid = closest.get("track_id")
        keeper = 1 if closest.get("is_keeper") else 0
        ep = self._ctl
        if ep is not None and ep["team"] == team and t - ep["t1"] <= cfg.ctl_gap_s and (
                tid == ep["tid"] or math.hypot(x - ep["p1"][0], y - ep["p1"][1]) <= cfg.ctl_same_player_m):
            ep["t1"] = t
            ep["p1"] = ball
            ep["n"] += 1
            ep["tid"] = tid
            ep["keeper_n"] += keeper
            if ep is self._ctl_last:
                self._unseen_max = 0.0
        else:
            last = self._ctl_last
            if ep is not None and not ep["conf"] and last is not None and ep["team"] != last["team"]:
                self._ctl_foreign_t = ep["t1"]   # rövid ellenfél-érintés (nem lett birtoklás)
            ep = self._ctl = {"team": team, "tid": tid, "tid0": tid, "t0": t, "t1": t, "p0": ball, "p1": ball,
                              "n": 1, "keeper_n": keeper, "conf": False}
        ep["snap"] = snap
        if not ep["conf"] and ep["n"] >= cfg.ctl_min_samples and ep["t1"] - ep["t0"] >= cfg.ctl_min_span_s - 1e-9:
            ep["conf"] = True
            prev = self._ctl_last
            self._ctl_last = ep
            ep["unseen_before"] = self._unseen_max
            self._unseen_max = 0.0
            self._on_new_control(t, prev, ep, players)

    def _on_new_control(self, t, A, B, players):
        """Új megerősített birtoklás (B) az előző (A) után: passz / sikertelen passz / szerelés / beadás."""
        cfg = self.cfg
        f = self._sp_follow
        if f is not None:
            self._sp_follow = None
            if f.get("team") and B["t0"] - f["time"] <= 8.0:
                f["retained"] = B["team"] == f["team"]   # a rögzített helyzet után a csapatnál maradt-e
        if self._sp_wait is not None:
            self._flush_restart(B["team"])
        if A is None:
            return
        self._check_dribble(A)
        fl = self._flight
        if fl is not None and fl["ep"] is A and self._close_flight(t, B):
            return   # beadás volt (a passz helyett)
        if self._ctl_break_t >= A["t1"] - 0.5:
            return   # közben játékmegszakítás / lövés / rögzített helyzet volt
        if any(A["t1"] - 1.0 <= ts <= B["t0"] for ts, _ in self._dist_started):
            return   # kapus-kihozatal (gk_* esemény)
        ps = self._pending_shot
        if (ps is not None and ps["t"] >= A["t1"] - 1.0) or self._goal_cand is not None:
            return
        gap = B["t0"] - A["t1"]
        if gap > cfg.pass_max_gap_s or B.get("unseen_before", 0.0) > cfg.pass_max_unseen_s:
            return
        p1, p0 = A["p1"], B["p0"]
        d = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
        if d / max(gap, 1e-3) > cfg.pass_max_speed_mps:
            return   # téves labda-ugrás
        t_arr = None
        for rt, rx, ry in self._raw:
            if rt > A["t1"] and math.hypot(rx - p0[0], ry - p0[1]) <= 3.0:
                t_arr = rt
                break
        if t_arr is not None and B["t0"] - t_arr > cfg.pass_receive_s:
            return   # a labda már korábban odaért, és ott állt / ott volt (nem követett játékosnál?): kétes
        if A["tid"] is not None and A["tid"] in (B["tid0"], B["tid"]):
            return   # ugyanaz a játékos (csapat-besorolás villogása / cselezés kieséssel)
        if 2 * A["keeper_n"] > A["n"]:
            return   # a kapus labdája: a kihozatalt a gk_* események írják le
        if B["team"] == A["team"]:
            if d < cfg.pass_min_m or gap < cfg.pass_min_gap_s or self._ctl_foreign_t > A["t1"] - 0.1:
                return
            for stid, steam, sx, sy in A["snap"]:
                if stid is not None and stid == B["tid0"] and math.hypot(sx - p1[0], sy - p1[1]) <= cfg.pass_dribble_m:
                    return   # az "átvevő" már a passz pillanatában a labdánál volt: cselezés
            for p in players:
                if p.get("track_id") == A["tid"]:
                    px, py = p["pitch_coords"]
                    if math.hypot(px - B["p1"][0], py - B["p1"][1]) <= cfg.pass_dribble_m:
                        return   # a "passzadó" még mindig a labdánál: cselezés
            self._emit_pass(A, B, d)
        elif d >= cfg.pass_failed_min_m and gap >= cfg.pass_min_gap_s:
            self._emit_pass_failed(A, B, d)
        elif d <= cfg.tackle_max_m:
            contact = any(steam == B["team"] and math.hypot(sx - p1[0], sy - p1[1]) <= cfg.tackle_contact_m
                          for stid, steam, sx, sy in A["snap"])
            if contact:
                self._emit_tackle(A, B, d)

    def _check_dribble(self, A):
        """Labdavezetés: egy megerősített birtoklásban a labda legalább dribble_min_m-t haladt (kezdő- és
        végpont között), legalább dribble_min_s ideig, mezőnyjátékosnál. Kísérleti (lásd EXPERIMENTAL_TYPES)."""
        cfg = self.cfg
        if A.get("dribble_checked"):
            return
        A["dribble_checked"] = True
        if 2 * A["keeper_n"] > A["n"] or A["t1"] - A["t0"] < cfg.dribble_min_s:
            return
        d = math.hypot(A["p1"][0] - A["p0"][0], A["p1"][1] - A["p0"][1])
        if d < cfg.dribble_min_m or A["n"] < 3 * cfg.ctl_min_samples:
            return
        team = A["team"]
        atk = self._attack_goal(team)
        gain = self._line_dist(A["p0"], atk) - self._line_dist(A["p1"], atk)
        self._emit({
            "type": "dribble", "time": A["t0"], "team": team, "player_track_id": A["tid0"],
            "from_xy": [round(A["p0"][0], 1), round(A["p0"][1], 1)], "to_xy": [round(A["p1"][0], 1), round(A["p1"][1], 1)],
            "distance_m": round(d, 1), "duration_s": round(A["t1"] - A["t0"], 1), "gain_m": round(gain, 1),
            "progressive": gain >= cfg.pass_progressive_m, "zone": self._zone_of(A["p0"], team),
            "end_sec": round(A["t1"] + 2.0, 2),
            "desc": f"Labdavezetés – {_team_label(team)}, {d:.0f} m, {A['t1'] - A['t0']:.1f} mp"
                    + (" (előre)" if gain >= cfg.pass_progressive_m else ""),
        })

    def _pass_fields(self, A, B, d, team):
        atk = self._attack_goal(team)
        p0, p1 = A["p1"], B["p0"]
        gain = self._line_dist(p0, atk) - self._line_dist(p1, atk)
        return {
            "from_xy": [round(p0[0], 1), round(p0[1], 1)], "to_xy": [round(p1[0], 1), round(p1[1], 1)],
            "distance_m": round(d, 1), "gain_m": round(gain, 1), "zone": self._zone_of(p0, team),
            "is_long": d > self.cfg.pass_long_m, "progressive": gain >= self.cfg.pass_progressive_m,
            "into_box": self._in_box(p1, atk) and not self._in_box(p0, atk),
            "player_track_id": A["tid"], "receive_time": round(B["t0"], 2), "attack_goal_x": atk,
        }

    def _emit_pass(self, A, B, d):
        team = A["team"]
        ev = {"type": "pass", "time": A["t1"], "team": team}
        ev.update(self._pass_fields(A, B, d, team))
        ev["receiver_track_id"] = B["tid0"]
        ev["end_sec"] = round(B["t0"] + 2.0, 2)
        tags = [t for t, on in (("hosszú", ev["is_long"]), ("előre játszott", ev["progressive"]),
                                ("a 16-osba", ev["into_box"])) if on]
        ev["desc"] = (f"Passz – {_team_label(team)}, {d:.0f} m" + (f" ({', '.join(tags)})" if tags else "")
                      + (", saját térfélről" if ev["zone"] == "sajat_terfel" else ", az ellenfél térfelén"))
        self._emit(ev)

    def _emit_pass_failed(self, A, B, d):
        team = A["team"]
        ev = {"type": "pass_failed", "time": A["t1"], "team": team, "won_by": B["team"]}
        ev.update(self._pass_fields(A, B, d, team))
        ev["receiver_track_id"] = B["tid0"]
        ev["end_sec"] = round(B["t0"] + 3.0, 2)
        ev["desc"] = (f"Sikertelen passz – {_team_label(team)}, {d:.0f} m, "
                      f"az ellenfél ({_team_label(B['team'])}) szerezte meg")
        self._emit(ev)
        self._emit({
            "type": "interception", "time": B["t0"], "team": B["team"], "player_track_id": B["tid0"],
            "lost_by": team, "from_xy": ev["from_xy"], "to_xy": ev["to_xy"], "distance_m": ev["distance_m"],
            "zone": self._zone_of(B["p0"], B["team"]), "pass_time": ev["time"],
            "desc": f"Labdaszerzés (elfogott passz) – {_team_label(B['team'])}",
        })

    def _emit_tackle(self, A, B, d):
        team = B["team"]
        p = B["p0"]
        self._emit({
            "type": "tackle", "time": A["t1"], "team": team, "player_track_id": B["tid0"],
            "lost_by": A["team"], "lost_by_track_id": A["tid"], "from_xy": [round(p[0], 1), round(p[1], 1)],
            "distance_m": round(d, 1), "zone": self._zone_of(p, team),
            "end_sec": round(B["t0"] + 3.0, 2),
            "desc": f"Szerelés / labdaszerzés párharcban – {_team_label(team)}, "
                    f"{'az ellenfél' if self._zone_of(p, team) == 'ellenfel_terfel' else 'a saját'} térfelén",
        })

    # ---------------------------------------------------------------- beadás
    def _cross_origin(self, A):
        """A birtoklás vége beadás-helyről (szélső sáv a 16-os szélességén kívül, utolsó harmad) történt-e."""
        if 2 * A["keeper_n"] > A["n"]:
            return False
        p = A["p1"]
        atk = self._attack_goal(A["team"])
        return (0.0 <= p[0] <= self.L and 0.0 <= p[1] <= self.W and abs(p[1] - self.W / 2.0) > 20.16
                and self._line_dist(p, atk) <= self.cfg.cross_final_third_m)

    def _update_flight(self, t, ball):
        A = self._ctl_last
        if A is None:
            self._flight = None
            return
        fl = self._flight
        if fl is None or fl["ep"] is not A:
            fl = self._flight = {"ep": A, "box_t": None, "box_pt": None}
        if t <= A["t1"] + 1e-6 or fl["box_t"] is not None:
            return
        if t - A["t1"] > self.cfg.cross_window_flight_s:
            return
        if not self._cross_origin(A):
            return
        atk = self._attack_goal(A["team"])
        if (self._in_box(ball, atk, self.cfg.cross_box_tol_m)
                and math.hypot(ball[0] - A["p1"][0], ball[1] - A["p1"][1]) >= self.cfg.cross_min_m
                and abs(ball[1] - self.W / 2.0) < abs(A["p1"][1] - self.W / 2.0) - 3.0):
            fl["box_t"] = t
            fl["box_pt"] = ball

    def _close_flight(self, t, B, final=False):
        """A labda útja az A birtoklás után lezárul (új birtoklás B, labda a pályán kívül, klip vége).
        Ha szélről a 16-os felé indult és oda is ért: beadás. Visszatér: volt-e beadás."""
        cfg = self.cfg
        fl = self._flight
        if fl is None or (B is not None and fl["ep"] is B):
            return False
        A = fl["ep"]
        self._flight = None
        if not self._cross_origin(A):
            return False
        atk = self._attack_goal(A["team"])
        box_pt = fl["box_pt"]
        if box_pt is None and B is not None and B["t0"] - A["t1"] <= cfg.cross_window_flight_s:
            p = B["p0"]
            if (self._in_box(p, atk, cfg.cross_box_tol_m)
                    and math.hypot(p[0] - A["p1"][0], p[1] - A["p1"][1]) >= cfg.cross_min_m
                    and abs(p[1] - self.W / 2.0) < abs(A["p1"][1] - self.W / 2.0) - 3.0):
                box_pt = p
        if box_pt is None:
            return False
        shot_after = False
        for ev in reversed(self.events[-12:]):
            if ev["type"] == "shot" and ev.get("team") == A["team"] and ev["time"] >= A["t1"] - 0.6:
                if ev["time"] <= A["t1"] + 0.6:
                    return False   # maga a "beadás" volt a lövés
                shot_after = True
        if B is not None:
            completed = B["team"] == A["team"] or shot_after
        elif shot_after:
            completed = True
        else:
            completed = None if final else False
        team = A["team"]
        p0 = A["p1"]
        side = "kozeli" if p0[1] > self.W / 2.0 else "tavoli"
        res = {True: "sikeres", False: "sikertelen", None: "kimenetel ismeretlen"}[completed]
        ev = {
            "type": "cross", "time": A["t1"], "team": team, "player_track_id": A["tid"],
            "from_xy": [round(p0[0], 1), round(p0[1], 1)], "to_xy": [round(box_pt[0], 1), round(box_pt[1], 1)],
            "completed": completed, "side": side, "attack_goal_x": atk,
            "desc": f"Beadás – {_team_label(team)} ({'közeli' if side == 'kozeli' else 'túlsó'} szélről, {res})",
        }
        if B is not None:
            ev["receiver_track_id"] = B["tid0"]
            ev["won_by"] = B["team"]
        self._emit(ev)
        return True

    # =================================================== rögzített helyzetek
    # Álló labda (self._rest) a pálya jellegzetes pontján -> felfegyverzés -> a labda elmozdul
    # (restart_launch_dist_m, restart_min_speed_mps) -> esemény a végrehajtás pillanatában.
    #   * középpont: kezdőrúgás; szögletzászló: szöglet; oldalvonal: bedobás; ötös: kirúgás (lásd fent);
    #   * egyéb álló labda: szabadrúgás, ha valóban holt labda (elég hosszú állás / újra-megjelenés / nincs
    #     a labdánál játékos) és vannak körülötte játékosok.
    # A bedobás álló labda nélkül is felismerhető: oldalvonalon túlra kerülés -> legalább
    # throw_in_min_dead_s holtidő -> a labda az oldalvonal mellől indulva a pályára kerül.

    def _update_restarts(self, t, ball, speed, closest, cdist, players):
        self._track_touch_out(t, ball)
        if ball is not None:
            r = self._rest
            if r is not None:
                rf = self._rest_free
                if rf is None or rf[0] is not r:
                    rf = self._rest_free = [r, 0, 0]
                rf[2] += 1
                if cdist is None or cdist > 1.2:
                    rf[1] += 1
            if self._sp_armed is not None:
                self._advance_restart(t, ball, speed, closest, cdist, players)
            else:
                self._arm_restart(t, players)
                if self._sp_armed is None:
                    self._check_throw_in_reappear(t, ball)
        w = self._sp_wait
        if w is not None and t - w["t_launch"] > self.cfg.restart_team_wait_s:
            self._flush_restart(None)

    def _track_touch_out(self, t, ball):
        if ball is None:
            return
        cfg = self.cfg
        x, y = ball
        m = cfg.touch_out_margin_m
        side = 0.0 if y < -m else (self.W if y > self.W + m else None)
        if side is None or not (-1.0 <= x <= self.L + 1.0):
            self._touch_seq = None
            lo = self._line_out
            if lo is not None and t - lo["t"] > cfg.throw_in_out_memory_s:
                self._line_out = None
            return
        ts = self._touch_seq
        ts = (side, ts[1] + 1, ts[2], ts[3]) if (ts is not None and ts[0] == side) else (side, 1, t, x)
        self._touch_seq = ts
        if ts[1] < 2:
            return
        lo = self._line_out
        if lo is None or lo["side"] != side or t - lo["t_last"] > 5.0:
            last = self._ctl_last
            last_team = last["team"] if (last is not None and 0.0 <= ts[2] - last["t1"] <= 4.0) else None
            self._line_out = {"side": side, "t": ts[2], "t_last": t, "x": ts[3], "last_team": last_team}
        else:
            lo["t_last"] = t

    def _restart_zone(self, p):
        cfg = self.cfg
        x, y = p
        if math.hypot(x - self.L / 2.0, y - self.W / 2.0) <= cfg.goal_kickoff_radius_m:
            return "kickoff", None
        for gx in self._goal_xs():
            for cy in (0.0, self.W):
                if math.hypot(x - gx, y - cy) <= cfg.corner_radius_m:
                    return "corner", gx
        if x < -0.5 or x > self.L + 0.5:
            return None, None
        if y <= cfg.touch_zone_m or y >= self.W - cfg.touch_zone_m:
            return "throw_in", None
        if self._goal_area_of(p, cfg.goal_kick_area_tol_m) is not None:
            return None, None   # kirúgás (lásd _check_goal_kick)
        return "free_kick", None

    def _arm_restart(self, t, players):
        cfg = self.cfg
        r = self._rest
        if r is None or r["n"] < 3 or r["t_last"] < t - 1e-6:
            return
        dur = r["t_last"] - r["t0"]
        if dur < min(cfg.throw_in_still_s, cfg.corner_still_s, cfg.kickoff_still_s):
            return
        if self._gk_armed is not None or self._dist is not None or self._goal_cand is not None:
            return
        gp = self._gk_poss
        if gp is not None and t - gp["t_last"] <= 1.0:
            return   # a kapusnál (kézben / lábnál) lévő labda
        if r["t0"] <= self._last_restart_t + 0.5:
            return   # ezt az álló labdát már végrehajtottnak vettük
        cx, cy = r["sx"] / r["n"], r["sy"] / r["n"]
        kind, gx = self._restart_zone((cx, cy))
        if kind is None:
            return
        if kind == "kickoff" and dur < cfg.kickoff_still_s:
            return
        if kind == "corner" and dur < cfg.corner_still_s:
            return
        if kind == "throw_in":
            lo = self._line_out
            if dur < cfg.throw_in_still_s or (lo is None and dur < cfg.throw_in_rest_alone_s):
                return
        if kind == "free_kick":
            if dur < cfg.free_kick_still_s:
                return
            ra = self._reappear
            rf = self._rest_free
            dead = (dur >= cfg.free_kick_dead_s
                    or (ra is not None and abs(r["t0"] - ra[0]) <= 1.0)
                    or (rf is not None and rf[0] is r and rf[2] >= 5 and rf[1] >= 0.6 * rf[2]))
            if not dead:
                return
            near = sum(1 for p in players if math.hypot(p["pitch_coords"][0] - cx, p["pitch_coords"][1] - cy)
                       <= cfg.free_kick_players_m)
            if near < cfg.free_kick_min_players:
                return
        self._sp_armed = {"kind": kind, "x": cx, "y": cy, "goal_x": gx, "t_still": r["t_last"], "votes": {},
                          "kicker": None}

    def _advance_restart(self, t, ball, speed, closest, cdist, players):
        cfg = self.cfg
        a = self._sp_armed
        if self._gk_armed is not None or self._ball_switch_t >= a["t_still"] - 1e-6:
            # a kapus végzi el (kapus-szabadrúgás / kirúgás), vagy a labdakövetés másik labdára váltott
            # (valós hiba: a lassan sodródó téves "álló labda" után a követő átugrott a valódira -> "elrúgás")
            self._sp_armed = None
            return
        d = math.hypot(ball[0] - a["x"], ball[1] - a["y"])
        dt = max(t - a["t_still"], 1e-3)
        if d >= cfg.restart_launch_dist_m:
            self._sp_armed = None
            fast = d / dt >= cfg.restart_min_speed_mps or (speed is not None and speed >= cfg.restart_min_speed_mps)
            # végrehajtó játékos kell az álló labdánál (a mellette senki nélkül álló "labda" többnyire téves
            # felismerés: kúp, tartaléklabda, vonal-csillanás)
            kicker = bool(a["votes"]) or any(
                math.hypot(p["pitch_coords"][0] - a["x"], p["pitch_coords"][1] - a["y"]) <= cfg.restart_kicker_m
                for p in players)
            if fast and kicker and (dt <= 1.5 or (dt <= 4.0 and d >= 15.0)):
                self._launch_restart(a["t_still"], a)
            return
        if d > 1.5 * cfg.goal_kick_still_radius_m:
            if dt > 2.0:
                self._sp_armed = None   # lassan elgurult: nem rögzített helyzet
            return
        a["t_still"] = t
        if closest is not None and cdist is not None and cdist <= cfg.restart_kicker_m:
            tm = self._player_team(closest)
            a["votes"][tm] = a["votes"].get(tm, 0) + 1
            a["kicker"] = closest.get("track_id")

    def _check_throw_in_reappear(self, t, ball):
        """Bedobás álló labda nélkül: oldalvonalon túl -> holtidő -> az oldalvonal mellől a pályára."""
        cfg = self.cfg
        lo = self._line_out
        if lo is None:
            return
        side = lo["side"]
        inside = ball[1] if side == 0.0 else self.W - ball[1]
        if inside <= 4.0:
            return
        self._line_out = None
        if t - lo["t"] < cfg.throw_in_min_dead_s:
            return   # azonnal visszajött (vagy a levegőben lévő labda vetülete esett kívülre): nem bedobás
        pb = self._prev_ball
        if pb is None or t - pb[0] > 1.5:
            return
        pin = pb[2] if side == 0.0 else self.W - pb[2]
        if pin > cfg.touch_zone_m + 1.0 or abs(pb[1] - lo["x"]) > cfg.throw_in_max_dx_m:
            return   # a bedobás helyét nem láttuk
        if math.hypot(ball[0] - pb[1], ball[1] - pb[2]) / max(t - pb[0], 1e-3) < cfg.restart_min_speed_mps * 0.6:
            return
        self._line_out = lo   # a _launch_restart innen veszi az utolsó érintő csapatát
        self._launch_restart(pb[0], {"kind": "throw_in", "x": pb[1], "y": pb[2], "goal_x": None, "votes": {},
                                     "kicker": None})

    def _launch_restart(self, t0, a):
        kind = a["kind"]
        if t0 - self._last_restart_t < self.cfg.restart_debounce_s:
            return
        self._last_restart_t = t0
        self._ctl_break_t = max(self._ctl_break_t, t0)
        self._ctl_last = None
        self._ctl = None
        self._flight = None
        votes = a.get("votes") or {}
        vote_team = None
        if votes:
            best = max(votes.items(), key=lambda kv: kv[1])
            if best[1] >= 2 and best[1] >= 0.6 * sum(votes.values()):
                vote_team = best[0]
        team, how = None, None
        gx = a.get("goal_x")
        if kind == "corner":
            team, how = other_team(self.defending_team(gx, t0)), "palyaoldal"
            self._out.pop(gx, None)
        elif kind == "throw_in":
            lo = self._line_out
            if lo is not None and lo.get("last_team"):
                team, how = other_team(lo["last_team"]), "utolso_erintes"
            elif vote_team:
                team, how = vote_team, "vegrehajto"
            self._line_out = None
        elif kind == "kickoff":
            if self._last_goal_team and t0 - self._last_goal_t <= 120.0:
                team, how = other_team(self._last_goal_team), "kapott_gol"
            elif vote_team:
                team, how = vote_team, "vegrehajto"
        elif vote_team:
            team, how = vote_team, "vegrehajto"
        self._sp_wait = {"kind": kind, "t": t0, "pos": (a["x"], a["y"]), "goal_x": gx, "team": team,
                         "how": how, "kicker": a.get("kicker"), "t_launch": self._last_t if self._last_t else t0}
        if team is not None:
            self._flush_restart(None)

    def _flush_restart(self, fallback_team):
        w = self._sp_wait
        self._sp_wait = None
        if w is None:
            return
        team, how = w["team"], w["how"]
        if team is None and fallback_team is not None:
            team, how = fallback_team, "kovetkezo_birtoklas"
        kind = w["kind"]
        if team is None and kind != "kickoff":
            return   # ismeretlen végrehajtó csapat: inkább kihagyjuk
        x, y = w["pos"]
        side = "kozeli" if y > self.W / 2.0 else "tavoli"
        base = {"time": w["t"], "team": team, "from_xy": [round(x, 1), round(y, 1)], "team_source": how,
                "player_track_id": w["kicker"]}
        tl = _team_label(team) if team else "ismeretlen csapat"
        if kind == "free_kick":
            ev = self._emit_free_kick(w["t"], (x, y), team, extra=base)
        elif kind == "corner":
            base.update({"type": "corner", "goal_x": w["goal_x"], "side": side,
                         "desc": f"Szöglet – {tl} ({'közeli' if side == 'kozeli' else 'túlsó'} oldal)"})
            ev = self._emit(base)
        elif kind == "throw_in":
            base.update({"type": "throw_in", "side": side, "zone": self._zone_of((x, y), team),
                         "desc": f"Bedobás – {tl} ({'közeli' if side == 'kozeli' else 'túlsó'} oldalvonal, "
                                 f"{'az ellenfél' if self._zone_of((x, y), team) == 'ellenfel_terfel' else 'a saját'} "
                                 f"térfelén)"})
            ev = self._emit(base)
        else:
            base.update({"type": "kickoff", "desc": f"Kezdőrúgás – {tl}"})
            ev = self._emit(base)
        self._sp_follow = ev

    def _emit_free_kick(self, t, pos, team, by_keeper=False, keeper_tid=None, extra=None):
        cfg = self.cfg
        x, y = pos
        atk = self._attack_goal(team)
        dist_goal = math.hypot(atk - x, self.W / 2.0 - y)
        dangerous = dist_goal <= cfg.free_kick_dangerous_m
        penalty = abs(self._line_dist(pos, atk) - 11.0) <= 1.5 and abs(y - self.W / 2.0) <= 1.5
        zone = self._zone_of(pos, team)
        ev = dict(extra or {"time": t, "team": team, "from_xy": [round(x, 1), round(y, 1)]})
        ev.update({"type": "free_kick", "zone": zone, "dangerous": dangerous, "is_penalty": penalty,
                   "distance_to_goal_m": round(dist_goal, 1), "by_keeper": bool(by_keeper)})
        if keeper_tid is not None:
            ev["keeper_track_id"] = keeper_tid
        kind = "Büntető" if penalty else ("Veszélyes szabadrúgás" if dangerous else "Szabadrúgás")
        who = f" (kapus: {_keeper_label(team, keeper_tid)})" if by_keeper else ""
        ev["desc"] = (f"{kind} – {_team_label(team)}{who}, {dist_goal:.0f} m-re az ellenfél kapujától, "
                      f"{'az ellenfél' if zone == 'ellenfel_terfel' else 'a saját'} térfelén")
        return self._emit(ev)


class PositioningMonitor:
    """
    Kapus helyezkedési hiba – csak a valóban értelmezhető esetek: a labda a kapus kapujához közel
    (veszélyes zónában) van, a kapus a labda és a kapu közepe közti szögfelezőtől TARTÓSAN és
    jelentősen eltér (a kamera képéből számolt pozíció képkockánként 1–2 m-t is remeghet).
    Kapusonként (csapatonként) legfeljebb debounce_s-enként egy esemény.
    Korábban minden 1,2 m-nél nagyobb, egyetlen képkockás eltérés esemény lett (2 perc alatt 8 db,
    a pálya túlsó végén lévő labdánál is) – ez használhatatlan volt.
    """

    def __init__(self, offset_m=2.5, min_s=1.5, ball_max_dist_m=30.0, debounce_s=20.0,
                 gap_s=0.4, pitch_length=105.0):
        self.offset_m = offset_m
        self.min_s = min_s
        self.ball_max_dist_m = ball_max_dist_m
        self.debounce_s = debounce_s
        self.gap_s = gap_s
        self.L = pitch_length
        self._run = {}        # team -> {"t0", "t_last", "max", "emitted"}
        self._last_emit = {}  # team -> t

    def update(self, t, gk_eval, team):
        """gk_eval: TacticalEngine.analyze_goalkeeper eredménye (vagy None). Visszatér: esemény dict vagy None."""
        if not gk_eval or team is None:
            return None
        gx = gk_eval["gk_coords"][0]
        bx = gk_eval["ball_coords"][0]
        ball_dist = bx if gx <= self.L / 2.0 else self.L - bx
        off = float(gk_eval["bisector_offset_m"])
        bad = abs(off) >= self.offset_m and 0.0 <= ball_dist <= self.ball_max_dist_m
        run = self._run.get(team)
        if not bad:
            if run is not None and t - run["t_last"] > self.gap_s:
                self._run.pop(team, None)
            return None
        if run is None or t - run["t_last"] > self.gap_s:
            run = self._run[team] = {"t0": t, "t_last": t, "max": off, "emitted": False}
        run["t_last"] = t
        if abs(off) > abs(run["max"]):
            run["max"] = off
        if run["emitted"] or t - run["t0"] < self.min_s:
            return None
        if t - self._last_emit.get(team, -1e9) < self.debounce_s:
            return None
        run["emitted"] = True
        self._last_emit[team] = t
        t0 = run["t0"]
        return {
            "type": "gk_positioning_error",
            "time": round(t0, 2),
            "team": team,
            "start_sec": round(max(0.0, t0 - 3.0), 2),
            "end_sec": round(t + 2.0, 2),
            "offset_m": round(run["max"], 2),
            "desc": f"Kapus helyezkedési hiba: {abs(run['max']):.1f} m-re a szögfelezőtől, "
                    f"{t - t0:.1f} mp-en át ({'labda felőli' if run['max'] > 0 else 'túlsó'} oldal)",
        }

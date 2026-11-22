"""
A teljes meccs elemzése egy lépésben (HTTP nélkül is futtatható folyamat).

Lépések (zárójelben a haladásból kapott súly):
  a) félidők felismerése – match_segmenter.detect_match_halves          (10)
  b) a meccs kivágása a szünet nélkül – cut_match_halves (kihagyható)    (5)
     képkocka-pontos smart cut (fast_cut.py): percek alatt kész, a kész fájlok azonnal
     megjelennek a job részeredményében (result.kesz_reszek, result.reszleges = true)
  c) mindkét félidő elemzése – main.cut_and_analyze_sync                 (68)
     (a 2. félidőben a team1_attacks_right megfordítva – térfélcsere)
  d) a két félidő eseményeinek összefésülése egy meccs-JSON-ba            (1)
  e) hőtérképek félidőnként és összesítve                                 (4)
  f) klip-összeállítások (kapusonként + összefoglaló: gólok, lövések)      (12)

A modul importálásakor NEM töltődik be a main.py (ott a YOLO modell import közben
töltődik be, ami torch nélkül nem megy) – a nehéz függvényeket a `funcs` paraméteren
kapja (a szerver a saját main moduljából adja át, a tesztek hamis függvényeket adnak),
alapból pedig a default_funcs() csak futáskor importálja őket.
"""
import os
import json
import time
import uuid

from jobs import JobCancelled

# ------------------------------------------------------------------
# A /cut-raw hangolási paraméterei, ugyanazokkal az alapértékekkel.
# Új paraméter: elég ide felvenni (+ build_analysis_config, ha a config-ba másképp kerül);
# a tests/test_jobs.py ellenőrzi, hogy ez egyezik-e a main.py /cut-raw végpontjával.
# ------------------------------------------------------------------
ANALYSIS_DEFAULTS = {
    "detect_mode": "team",
    "calib_points": "",
    "roi_top_pct": 0.28,
    "roi_bottom_pct": 0.10,
    "min_player_height_pct": 0.022,  # a képmagasság arányában (720p: ~16 px, 1080p: ~24 px), fix px korlát nélkül
    "use_static_roi": False,
    "grass_min_ratio": 0.3,
    "stationary_window_sec": 4.0,
    "stationary_max_move": 0.5,
    "grass_hsv_lower": "30,40,40", "grass_hsv_upper": "85,255,255",
    "ref_hsv_lower": "0,0,0", "ref_hsv_upper": "179,255,255",
    "team1_hsv_lower": "0,0,0", "team1_hsv_upper": "179,255,255",
    "team2_hsv_lower": "0,0,0", "team2_hsv_upper": "179,255,255",
    "keeper1_hsv_lower": "0,0,0", "keeper1_hsv_upper": "179,255,255",
    "keeper2_hsv_lower": "0,0,0", "keeper2_hsv_upper": "179,255,255",
    "team1_attacks_right": True,     # az 1. félidőre vonatkozik, a 2.-ban megfordul
    "review_export": True,
    "auto_teams": True,              # automatikus csapatba sorolás mezszín alapján
    "swap_teams": False,             # az 1. félidőre vonatkozik, a 2.-ban megfordul
    "dynamic_homography": True,      # képkockánkénti pályakövetés (forgó Veo kamera)
    "draw_pitch_overlay": True,
    "calib_time_sec": None,          # a kézi kalibráció képének ideje (eredeti videóban)
    "calib_pitch_points": "",
    "pitch_margin_m": 2.5,           # pályahatár-szűrő: ennyi m-rel a vonalon túl még elfogadott (0 = ki)
    "grass_model": "auto",           # fűmaszk: "auto" = videónként tanult fűmodell, "hsv" = fix HSV tartomány
    "output_scale": 1.0,             # az annotált elemzés-videó méretaránya (0.5 = fél felbontás, gyorsabb)
}

# A teljes folyamat saját beállításai
PIPELINE_DEFAULTS = {
    "expected_half_min": 45.0,
    # a 2. félidő saját kézi kalibrációja (ha a szünetben odébb tették a kamerát; tools/kalibracio.py --felido 2)
    "calib2_points": "", "calib2_pitch_points": "", "calib2_time_sec": None,
    "use_person_count": False,
    "skip_cut": False,
    "padding_sec": 5.0,
    # kézi félidőhatárok (mp vagy pp:mm / ó:pp:mm); ha mind a négy megvan, nincs felismerés
    "first_start": "", "first_end": "", "second_start": "", "second_end": "",
    "heatmaps": True,
    # vesszővel elválasztva: goalkeepers (kapusonként egy videó), highlights (gólok + lövések); üres = nincs
    "compilations": "goalkeepers,highlights",
    "clip_pre_sec": 5.0,
    "clip_post_sec": 4.0,
    "clip_accurate": True,
}

ALL_DEFAULTS = {**ANALYSIS_DEFAULTS, **PIPELINE_DEFAULTS}

COMPILATION_KINDS = ("goalkeepers", "highlights")

# Egyezik a main.py GK_TIMELINE_TYPES / GK_TEAM_NAMES értékeivel (a teszt ellenőrzi)
GK_TIMELINE_TYPES = ("gk_save", "gk_claim_cross", "gk_goal_kick", "gk_long_ball", "gk_short_pass",
                     "gk_conceded", "gk_sweeper", "gk_positioning_error")
GK_TEAM_NAMES = {"team1": "Hazai Kapus", "team2": "Vendég Kapus"}
HIGHLIGHT_TYPES = ("goal", "shot")

# A 2. félidő követési azonosítóinak eltolása az összesített pályákban
# (a követő félidőnként újraindul, az azonosítók ütköznének)
SECOND_HALF_TRACK_OFFSET = 100000

PHASE_WEIGHTS = {"detect": 10.0, "cut": 5.0, "analyze": 68.0, "merge": 1.0, "heatmaps": 4.0, "clips": 12.0}

HALF_LABELS = {"first_half": "1. félidő", "second_half": "2. félidő"}


# ------------------------------------------------------------------
# Paraméterek
# ------------------------------------------------------------------

def _to_bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "igen", "yes", "on", "i"):
        return True
    if s in ("0", "false", "nem", "no", "off", "n", ""):
        return False
    raise ValueError(f"Érvénytelen logikai érték: {value!r}")


def coerce_options(raw, defaults=None):
    """
    A kapott (pl. form) értékekből az ismert kulcsokat az alapérték típusára alakítja,
    a hiányzókat az alapértékkel tölti ki. Ismeretlen kulcsokat figyelmen kívül hagy.
    Hibás értéknél ValueError (magyar üzenettel).
    """
    defaults = ALL_DEFAULTS if defaults is None else defaults
    raw = raw or {}
    out = {}
    for key, default in defaults.items():
        if key not in raw or raw[key] is None:
            out[key] = default
            continue
        value = raw[key]
        try:
            if default is None:  # opcionális szám (pl. calib_time_sec); üres = nincs megadva
                out[key] = None if str(value).strip() == "" else float(value)
            elif isinstance(default, bool):
                out[key] = _to_bool(value)
            elif isinstance(default, float):
                out[key] = float(value)
            elif isinstance(default, int):
                out[key] = int(value)
            else:
                out[key] = str(value)
        except (TypeError, ValueError):
            raise ValueError(f"Érvénytelen érték a(z) '{key}' mezőben: {value!r}")
    return out


def _hsv(text):
    return [int(x) for x in str(text).split(",")]


def build_analysis_config(opts):
    """A cut_and_analyze_sync config dict-je – ugyanaz, mint amit a /cut-raw végpont épít."""
    o = coerce_options(opts, ANALYSIS_DEFAULTS)
    config = {
        "team1_attacks_right": o["team1_attacks_right"],
        "review_export": o["review_export"],
        "auto_teams": o["auto_teams"], "swap_teams": o["swap_teams"],
        "dynamic_homography": o["dynamic_homography"], "draw_pitch_overlay": o["draw_pitch_overlay"],
        "calib_time_sec": o["calib_time_sec"], "calib_pitch_points": o["calib_pitch_points"],
        "pitch_margin_m": o["pitch_margin_m"], "grass_model": o["grass_model"],
        "output_scale": o["output_scale"],
        "roi_top_pct": o["roi_top_pct"], "roi_bottom_pct": o["roi_bottom_pct"],
        "min_player_height_pct": o["min_player_height_pct"],
        "use_static_roi": o["use_static_roi"], "grass_min_ratio": o["grass_min_ratio"],
        "stationary_window_sec": o["stationary_window_sec"], "stationary_max_move": o["stationary_max_move"],
    }
    for k in ("grass", "ref", "team1", "team2", "keeper1", "keeper2"):
        config[f"{k}_lower"] = _hsv(o[f"{k}_hsv_lower"])
        config[f"{k}_upper"] = _hsv(o[f"{k}_hsv_upper"])
    return config


def parse_compilations(value):
    kinds = [k.strip().lower() for k in str(value or "").replace(";", ",").split(",") if k.strip()]
    bad = [k for k in kinds if k not in COMPILATION_KINDS]
    if bad:
        raise ValueError(f"Ismeretlen összeállítás: {', '.join(bad)} (lehetséges: {', '.join(COMPILATION_KINDS)})")
    return kinds


# ------------------------------------------------------------------
# Az alapértelmezett (valódi) függvények – lusta importtal
# ------------------------------------------------------------------

def default_funcs(main_module=None):
    """
    A valódi munkafüggvények. main_module: a már betöltött main modul (a szerver adja át
    önmagát, így nem töltődik be másodszor); ha None, itt importáljuk (parancssor).
    """
    import match_segmenter
    import heatmap
    import clip_export

    def _main():
        nonlocal main_module
        if main_module is None:
            import main as m  # YOLO + torch betöltése – csak itt, futáskor
            main_module = m
        return main_module

    def detect_halves(video_path, expected_half_min, use_person_count, progress_cb=None):
        counter = _main()._segmenter_person_counter if use_person_count else None
        return match_segmenter.detect_match_halves(video_path, expected_half_min=expected_half_min,
                                                   person_counter=counter, progress_cb=progress_cb)

    def analyze(video_path, start_sec, end_sec, calib_points, config, detect_mode="team", progress=None):
        return _main().cut_and_analyze_sync(video_path, start_sec, end_sec, True, detect_mode,
                                            calib_points, config, progress=progress)

    return {
        "detect_halves": detect_halves,
        "cut_match": match_segmenter.cut_match_halves,
        "analyze": analyze,
        "heatmaps": heatmap.generate_heatmaps_from_file,
        "export_clips": clip_export.export_clips,
        "parse_time": match_segmenter.parse_time_value,
        "format_hms": match_segmenter.format_hms,
        "clips_dir": lambda: _main().CLIPS_DIR,
    }


# ------------------------------------------------------------------
# Segédek
# ------------------------------------------------------------------

class _NullProgress:
    def report(self, pct=None, phase=None):
        pass

    __call__ = report

    def check_cancel(self):
        pass

    def sub(self, start, end, label=None):
        return self


def _sub(progress, start, end, label):
    if hasattr(progress, "sub"):
        return progress.sub(start, end, label)

    def fn(pct=None, phase=None):  # sima függvény (pct, phase) is elég haladásjelzőnek
        g = None if pct is None else start + (end - start) * min(100.0, max(0.0, float(pct))) / 100.0
        progress(g, f"{label}: {phase}" if phase else label)
    return fn


def _check(progress):
    if hasattr(progress, "check_cancel"):
        progress.check_cancel()


def _publish(progress, result):
    """Részeredmény közzététele (jobs.ProgressReporter.publish): a kliens a kész részeket
    (pl. a vágott meccsvideót) már a teljes folyamat vége előtt megkapja."""
    pub = getattr(progress, "publish", None)
    if callable(pub):
        pub(json.loads(json.dumps(dict(result, reszleges=True), default=str)))


def _accepts(func, name):
    """Elfogad-e a függvény ilyen nevű kulcsszavas paramétert (a tesztek hamis függvényeinél nem)."""
    import inspect
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _url(path, clips_dir, url_prefix="/clips"):
    """Ha a fájl a kiszolgált clips mappában van, a /clips/... URL-je, különben None."""
    if not path:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(clips_dir))
    except ValueError:  # más meghajtó (Windows)
        return None
    if rel.startswith(".."):
        return None
    return f"{url_prefix}/{rel.replace(os.sep, '/')}"


def _fmt_clock(sec):
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def resolve_halves(opts, funcs, progress, notes):
    """A félidők határai: kézi értékek, ahol meg vannak adva, a többi felismeréssel."""
    parse_time = funcs.get("parse_time") or (lambda v: float(v) if str(v).strip() else None)
    overrides = [parse_time(opts[k]) for k in ("first_start", "first_end", "second_start", "second_end")]
    detection = None
    if any(v is None for v in overrides):
        detection = funcs["detect_halves"](opts["video_path"], opts["expected_half_min"],
                                           opts["use_person_count"], progress_cb=_segmenter_cb(progress))
        notes += list(detection.get("notes") or [])
    else:
        notes.append("Kézzel megadott félidőhatárok – automatikus felismerés nélkül.")
    halves = {}
    for key, vals in (("first_half", overrides[:2]), ("second_half", overrides[2:])):
        h = dict((detection or {}).get(key) or {})
        if vals[0] is not None:
            h["start_sec"] = vals[0]
        if vals[1] is not None:
            h["end_sec"] = vals[1]
        if "start_sec" in h and "end_sec" in h and float(h["end_sec"]) > float(h["start_sec"]):
            halves[key] = {"start_sec": round(float(h["start_sec"]), 2), "end_sec": round(float(h["end_sec"]), 2)}
            if funcs.get("format_hms"):
                halves[key]["start_hms"] = funcs["format_hms"](halves[key]["start_sec"])
                halves[key]["end_hms"] = funcs["format_hms"](halves[key]["end_sec"])
        elif h:
            notes.append(f"{'Az 1.' if key == 'first_half' else 'A 2.'} félidő határai hibásak, kihagyva.")
    if "first_half" not in halves and "second_half" in halves:
        halves = {"first_half": halves["second_half"]}
        notes.append("Csak egy érvényes félidő van – 1. félidőként elemzem.")
    if not halves:
        raise RuntimeError("Nincs érvényes félidő az elemzéshez.")
    progress(100.0, "kész")
    return halves, detection


def _segmenter_cb(progress):
    """match_segmenter progress_cb(fázis, százalék) → a mi haladásunk (hang: 0–10, kép: 10–100)."""
    def cb(kind, pct):
        if kind == "hang":
            progress(10.0, "hangsáv (sípszavak)")
        else:
            progress(10.0 + 0.9 * float(pct), "képi jelek mintavételezése")
    return cb


def _clip_cb(progress, lo, hi, label):
    """clip_export progress_cb(kész, összes) → a [lo, hi] sáv."""
    def cb(done, total):
        progress(lo + (hi - lo) * done / max(1, total), f"{label} ({done}/{total} klip)")
    return cb


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_half_events(half_data, halves, expected_half_min, cut_offsets=None):
    """
    A félidők eseményeinek összefésülése.
    half_data: {"first_half": events-JSON dict, "second_half": ...}
    Minden eseményhez: half (1/2), half_label, time_in_half, time_in_match (a szünet nélküli
    játékidő), match_clock (hivatalos óra: a 2. félidő 45:00-tól), és ha van vágott meccsvideó,
    full_match_sec (az időpont a szünet nélküli meccsvideóban).
    A 'time' / 'start_sec' / 'end_sec' az EREDETI videó ideje marad (a klipvágáshoz).
    """
    events = []
    first = halves.get("first_half")
    first_len = (first["end_sec"] - first["start_sec"]) if first else 0.0
    for idx, key in enumerate(("first_half", "second_half"), 1):
        data = half_data.get(key)
        if not data:
            continue
        h = halves[key]
        match_offset = 0.0 if idx == 1 else first_len
        clock_offset = 0.0 if idx == 1 else float(expected_half_min) * 60.0
        for ev in data.get("events") or []:
            e = dict(ev)
            t = float(e.get("time", 0.0))
            in_half = max(0.0, t - h["start_sec"])
            e["half"] = idx
            e["half_label"] = HALF_LABELS[key]
            e["time_in_half"] = round(in_half, 2)
            e["time_in_match"] = round(match_offset + in_half, 2)
            e["match_clock"] = _fmt_clock(clock_offset + in_half)
            if cut_offsets and cut_offsets.get(key) is not None:
                e["full_match_sec"] = round(cut_offsets[key] + (t - h["start_sec"]), 2)
            events.append(e)
    events.sort(key=lambda e: (e["half"], float(e.get("time", 0.0))))
    return events


def merge_trajectories(half_data):
    """
    A félidők pályáinak egyesítése; a 2. félidő azonosítói SECOND_HALF_TRACK_OFFSET-tel eltolva.
    A labda útja ("ball": [[x, y, t], ...]) a két félidőből összefűzve, idő szerint rendezve
    (az idők az eredeti videó másodpercei, így a félidők nem fedik egymást).
    """
    tracks, hz, ball, ball_hz = {}, None, [], None
    for idx, key in enumerate(("first_half", "second_half")):
        block = ((half_data.get(key) or {}).get("trajectories") or {})
        hz = hz or block.get("hz")
        ball_hz = ball_hz or block.get("ball_hz")
        for p in block.get("ball") or []:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                ball.append(list(p[:3]))
        for tid, tr in (block.get("tracks") or {}).items():
            try:
                new_id = int(tid) + (SECOND_HALF_TRACK_OFFSET if idx == 1 else 0)
            except (TypeError, ValueError):
                continue
            tracks[str(new_id)] = dict(tr, half=idx + 1, original_track_id=int(tid))
    out = {"hz": hz or 5.0, "tracks": tracks}
    if ball:
        ball.sort(key=lambda p: float(p[2]))
        out["ball_hz"] = ball_hz or 10.0
        out["ball"] = ball
    return out


def merge_team_colors(half_data):
    """
    A tanult mezszínek ([r, g, b]) a meccs-összesítőbe: az 1. félidőé, ha van, különben a 2.-é.
    A pipeline a 2. félidőben megfordítja a csapatsorrendet (swap_teams), így a "team1" mindkét
    félidőben ugyanaz a csapat – a színek is egyeznek.
    """
    for key in ("first_half", "second_half"):
        colors = (half_data.get(key) or {}).get("team_colors")
        if isinstance(colors, dict) and colors.get("team1") and colors.get("team2"):
            return colors
    return None


def _cut_offsets(halves, padding_sec, duration_sec=None):
    """A félidők kezdetének helye a vágott (szünet nélküli) meccsvideóban – a cut_match_halves logikája szerint."""
    offsets, pos = {}, 0.0
    for key in ("first_half", "second_half"):
        h = halves.get(key)
        if not h:
            continue
        s = max(0.0, h["start_sec"] - padding_sec)
        e = h["end_sec"] + padding_sec
        if duration_sec:
            e = min(e, float(duration_sec))
        if e <= s:
            continue
        offsets[key] = pos + (h["start_sec"] - s)
        pos += e - s
    return offsets


def _clip_items(events, types=None, team=None):
    items = []
    for ev in events:
        if types and ev.get("type") not in types:
            continue
        if team is not None and (ev.get("team") or "ismeretlen") != team:
            continue
        if ev.get("start_sec") is not None and ev.get("end_sec") is not None:
            items.append({"start_sec": ev["start_sec"], "end_sec": ev["end_sec"]})
        elif ev.get("time") is not None:
            items.append({"time": ev["time"]})
    return items


# ------------------------------------------------------------------
# A teljes folyamat
# ------------------------------------------------------------------

def run_match_pipeline(video_path, options=None, progress=None, funcs=None, out_dir=None):
    """
    A teljes meccs elemzése. Visszatér: JSON-ba írható eredmény dict (lásd docs/HATTERFELADATOK.md).
    progress: jobs.ProgressReporter (vagy bármi report(pct, phase) / sub() / check_cancel() felülettel,
              vagy egy sima (pct, phase) függvény); megszakításkor JobCancelled-et dobhat.
    funcs: a munkafüggvények (alap: default_funcs()); a tesztek hamis függvényeket adnak.
    out_dir: a kimeneti mappa (alap: a main.CLIPS_DIR) – az elemzés (cut_and_analyze_sync) mindig a
             main.CLIPS_DIR-be ír, a többi lépés ide.
    """
    t0 = time.monotonic()
    progress = progress or _NullProgress()
    funcs = funcs or default_funcs()
    opts = coerce_options(options or {})
    opts["video_path"] = video_path
    compilations = parse_compilations(opts["compilations"])
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"A videófájl nem található: {video_path}")
    if out_dir is None:
        clips_dir = funcs.get("clips_dir")
        out_dir = clips_dir() if callable(clips_dir) else (clips_dir or "clips")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]

    W = dict(PHASE_WEIGHTS)
    if opts["skip_cut"]:
        W["cut"] = 0.0
    if not opts["heatmaps"]:
        W["heatmaps"] = 0.0
    if not compilations:
        W["clips"] = 0.0
    total_w = sum(W.values()) or 1.0
    bounds, acc = {}, 0.0
    for k in ("detect", "cut", "analyze", "merge", "heatmaps", "clips"):
        start = acc
        acc += 100.0 * W[k] / total_w
        bounds[k] = (start, acc)

    notes, warnings = [], []
    result = {"status": "success", "video_path": os.path.abspath(video_path), "run_id": run_id,
              "out_dir": out_dir, "options": {k: opts[k] for k in PIPELINE_DEFAULTS}}

    # a) félidők
    progress(bounds["detect"][0], "Félidők felismerése")
    halves, detection = resolve_halves(opts, funcs, _sub(progress, *bounds["detect"], "Félidők felismerése"), notes)
    result["halves"] = halves
    result["detection"] = {"confidence": (detection or {}).get("confidence"),
                           "duration_sec": (detection or {}).get("duration_sec"),
                           "manual": detection is None}
    _check(progress)

    result["kesz_reszek"] = ["felidok"]
    _publish(progress, result)

    # b) vágás (szünet nélküli meccsvideó) – a gyors (smart cut) vágás az elemzés ELŐTT fut,
    #    így a kliens percek alatt megkapja a vágott félidőket / meccsvideót (részeredmény),
    #    miközben a nehéz elemzés még órákig tarthat.
    cut_offsets = None
    result["cut"] = None
    if not opts["skip_cut"]:
        progress(bounds["cut"][0], "A meccs kivágása (szünet nélkül)")
        duration = (detection or {}).get("duration_sec")
        live = {"local_paths": {}}

        def on_part(key, path):
            # minden kész félidő / a meccsvideó azonnal megjelenik a job eredményében
            live["local_paths"][key] = path
            live[f"{key}_url"] = _url(path, out_dir)
            result["cut"] = dict(live, local_paths=dict(live["local_paths"]))
            result["kesz_reszek"] = ["felidok"] + [f"vagas_{k}" for k in live["local_paths"]]
            _publish(progress, result)
            _check(progress)
            if hasattr(progress, "check_pause"):
                progress.check_pause()  # szüneteltetés a következő fájl előtt

        cut_lo, cut_hi = bounds["cut"]
        step_labels = {"first_half": "1. félidő", "second_half": "2. félidő", "full_match": "összefűzés"}

        def on_step(key, index, total):
            # a vágás lépésenként halad (1. félidő → 2. félidő → összefűzés), hogy a kliens lássa
            progress(cut_lo + (cut_hi - cut_lo) * index / max(1, total),
                     f"A meccs kivágása (szünet nélkül): {step_labels.get(key, key)} ({index + 1}/{total})")

        kwargs = {"padding_sec": opts["padding_sec"], "duration_sec": duration}
        if _accepts(funcs["cut_match"], "on_part"):
            kwargs["on_part"] = on_part
        if _accepts(funcs["cut_match"], "on_step"):
            kwargs["on_step"] = on_step
        paths = dict(funcs["cut_match"](video_path, halves, out_dir, **kwargs))
        cut_modes = paths.pop("cut_modes", None)
        cut_offsets = _cut_offsets(halves, opts["padding_sec"], duration)
        result["cut"] = {
            "local_paths": paths,
            "first_half_url": _url(paths.get("first_half"), out_dir),
            "second_half_url": _url(paths.get("second_half"), out_dir),
            "full_match_url": _url(paths.get("full_match"), out_dir),
        }
        if cut_modes:
            result["cut"]["cut_modes"] = cut_modes
        result["kesz_reszek"] = ["felidok", "vagas"]
        _publish(progress, result)
        progress(bounds["cut"][1], "A meccs kivágva")
    _check(progress)

    # c) elemzés félidőnként (súly a félidők hossza szerint)
    analysis_config = build_analysis_config(opts)
    lens = {k: h["end_sec"] - h["start_sec"] for k, h in halves.items()}
    total_len = sum(lens.values()) or 1.0
    a0, a1 = bounds["analyze"]
    pos = a0
    analysis, half_data = {}, {}
    clips_root = funcs.get("clips_dir")
    clips_root = os.path.abspath(clips_root() if callable(clips_root) else (clips_root or out_dir))
    for key in ("first_half", "second_half"):
        if key not in halves:
            continue
        h = halves[key]
        span = (a1 - a0) * lens[key] / total_len
        cfg = dict(analysis_config)
        if key == "second_half":
            cfg["team1_attacks_right"] = not analysis_config["team1_attacks_right"]  # térfélcsere
            # Automatikus csapatoknál az 1. csapat = a klip elején bal oldalon álló csapat;
            # térfélcsere után ez a másik csapat lenne, ezért a sorrendet is megfordítjuk,
            # így a "team1" mindkét félidőben ugyanaz a csapat marad.
            cfg["swap_teams"] = not analysis_config.get("swap_teams", False)
            # Ha az 1. félidő mezszínei megvannak, a csapat-azonosság a mezszínből dől el (a swap_teams
            # helyett) – így a "team1" akkor is ugyanaz marad, ha a megadott 2. szakaszban nem volt csere.
            ref = (half_data.get("first_half") or {}).get("team_colors")
            if ref:
                cfg["reference_team_colors"] = ref
        # A kézi kalibráció csak abban a félidőben érvényes, amelyikben a képe készült;
        # a másik félidőben a pályavonalakból automatikusan kalibrálunk.
        calib_points = opts["calib_points"]
        if key == "second_half" and opts.get("calib2_points"):
            # külön 2. félidős kalibráció (pl. a szünetben odébb tett kamera)
            calib_points = opts["calib2_points"]
            cfg["calib_pitch_points"] = opts.get("calib2_pitch_points", "")
            cfg["calib_time_sec"] = opts.get("calib2_time_sec")
        ct = cfg.get("calib_time_sec")
        if calib_points and ct is not None and not (h["start_sec"] <= float(ct) <= h["end_sec"]):
            calib_points = ""
        elif calib_points and ct is None and key == "second_half" and not opts.get("calib2_points") \
                and cfg.get("dynamic_homography", True):
            calib_points = ""  # idő nélküli kalibráció = az 1. félidő eleje
        label = f"{HALF_LABELS[key]} elemzése"
        progress(pos, label)
        clip_url, local_path, events_path = funcs["analyze"](
            video_path, h["start_sec"], h["end_sec"], calib_points, cfg,
            detect_mode=opts["detect_mode"], progress=_sub(progress, pos, pos + span, label))
        pos += span
        data = _load_json(events_path) if events_path and os.path.exists(events_path) else {}
        half_data[key] = data
        analysis[key] = {"clip_url": clip_url, "local_path": local_path, "events_path": events_path,
                         "events_url": _url(events_path, clips_root), "review_url": data.get("review_url"),
                         "event_count": len(data.get("events") or []),
                         "timing": data.get("timing"),   # PROFILOZÁS: szakaszonkénti idő (félidőnként)
                         "team1_attacks_right": cfg["team1_attacks_right"]}
        result["analysis"] = analysis
        result["kesz_reszek"].append(f"elemzes_{key}")
        _publish(progress, result)
        _check(progress)
    result["analysis"] = analysis

    # d) összefésülés
    progress(bounds["merge"][0], "Események összefésülése")
    events = merge_half_events(half_data, halves, opts["expected_half_min"], cut_offsets)
    match_payload = {
        "events": events,
        "halves": halves,
        "match_stats": {k: (half_data.get(k) or {}).get("match_stats") for k in halves},
        "review_urls": {k: analysis[k].get("review_url") for k in analysis},
        "source": {"video_path": os.path.abspath(video_path),
                   "start_sec": halves["first_half"]["start_sec"],
                   "end_sec": max(h["end_sec"] for h in halves.values())},
        "full_match": (result["cut"] or {}).get("local_paths", {}).get("full_match") if result["cut"] else None,
        "trajectories": merge_trajectories(half_data),
        # a taktikai tábla visszajátszásához: mezszínek és a félidő hossza (a meccsórához)
        "team_colors": merge_team_colors(half_data),
        "expected_half_min": opts["expected_half_min"],
    }
    match_events_path = os.path.join(out_dir, f"match_events_{run_id}.json")
    with open(match_events_path, "w", encoding="utf-8") as f:
        json.dump(match_payload, f, ensure_ascii=False, indent=2)
    result["match_events_path"] = match_events_path
    result["match_events_url"] = _url(match_events_path, out_dir)
    result["event_count"] = len(events)
    result["event_counts"] = {}
    for ev in events:
        result["event_counts"][ev.get("type", "?")] = result["event_counts"].get(ev.get("type", "?"), 0) + 1
    result["kesz_reszek"].append("esemenyek")
    _publish(progress, result)
    progress(bounds["merge"][1], "Események összefésülve")
    _check(progress)

    # e) hőtérképek
    result["heatmaps"] = {}
    if opts["heatmaps"]:
        h0, h1 = bounds["heatmaps"]
        jobs = [(k, analysis[k]["events_path"], {}) for k in ("first_half", "second_half")
                if k in analysis and analysis[k]["events_path"]]
        if "second_half" in halves:
            # Összesített: a 2. félidő tükrözve (egy irányba támadva), csak csapat-hőtérkép
            jobs.append(("combined", match_events_path,
                         {"flip_at_sec": halves["second_half"]["start_sec"], "min_player_sec": float("inf")}))
        for i, (name, path, kw) in enumerate(jobs):
            progress(h0 + (h1 - h0) * i / max(1, len(jobs)), f"Hőtérkép: {name}")
            try:
                res = funcs["heatmaps"](path, out_dir, **kw)
            except JobCancelled:
                raise
            except Exception as ex:  # a hőtérkép hibája ne buktassa el a teljes elemzést
                res = {"status": "error", "message": str(ex)}
            if res.get("status") != "success":
                warnings.append(f"Hőtérkép ({name}): {res.get('message', 'hiba')}")
            result["heatmaps"][name] = res
            _check(progress)
        progress(h1, "Hőtérképek kész")

    # f) klip-összeállítások
    result["compilations"] = {}
    if compilations:
        tasks = []
        if "goalkeepers" in compilations:
            teams = sorted({ev.get("team") or "ismeretlen" for ev in events if ev.get("type") in GK_TIMELINE_TYPES},
                           key=lambda k: (k not in GK_TEAM_NAMES, k))
            if not teams:
                warnings.append("Kapus-összeállítás: nincs kapus esemény.")
            for team in teams:
                tasks.append((f"goalkeeper_{team}", GK_TEAM_NAMES.get(team, "Ismeretlen kapus"),
                              _clip_items(events, GK_TIMELINE_TYPES, team)))
        if "highlights" in compilations:
            items = _clip_items(events, HIGHLIGHT_TYPES)
            if items:
                tasks.append(("highlights", "Osszefoglalo", items))
            else:
                warnings.append("Összefoglaló: nincs gól vagy lövés esemény.")
        c0, c1 = bounds["clips"]
        for i, (name, title, items) in enumerate(tasks):
            lo = c0 + (c1 - c0) * i / len(tasks)
            progress(lo, f"Klip-összeállítás: {title}")
            try:
                res = funcs["export_clips"](video_path, items, out_dir, pre_sec=opts["clip_pre_sec"],
                                            post_sec=opts["clip_post_sec"], merge=True,
                                            accurate=opts["clip_accurate"], title=title,
                                            progress_cb=_clip_cb(progress, lo, c0 + (c1 - c0) * (i + 1) / len(tasks),
                                                                 f"Klip-összeállítás: {title}"))
            except JobCancelled:
                raise
            except Exception as ex:
                res = {"status": "error", "message": str(ex)}
            res = dict(res, title=title, item_count=len(items))
            if res.get("status") != "success":
                warnings.append(f"Klip-összeállítás ({title}): {res.get('message', 'hiba')}")
            result["compilations"][name] = res
            _check(progress)
        progress(c1, "Klip-összeállítások kész")

    result["notes"] = notes
    result["warnings"] = warnings
    result["elapsed_sec"] = round(time.monotonic() - t0, 1)
    result["kesz_reszek"].append("kesz")
    result.pop("reszleges", None)
    progress(100.0, "Kész")
    return result


def summarize(result):
    """Magyar nyelvű összefoglaló a parancssorhoz."""
    lines = ["", "=== A meccs elemzése elkészült ===", f"Videó: {result.get('video_path')}"]
    for key, h in (result.get("halves") or {}).items():
        lines.append(f"{HALF_LABELS.get(key, key)}: {h.get('start_hms', h['start_sec'])} – "
                     f"{h.get('end_hms', h['end_sec'])}")
    conf = (result.get("detection") or {}).get("confidence")
    if conf is not None:
        lines.append(f"Felismerés megbízhatósága: {conf:.2f}" + ("  (ellenőrizd a határokat!)" if conf < 0.5 else ""))
    cut = result.get("cut")
    if cut:
        lines.append(f"Szünet nélküli meccsvideó: {cut['local_paths'].get('full_match')}")
    for key, a in (result.get("analysis") or {}).items():
        lines.append(f"{HALF_LABELS.get(key, key)} – elemzett videó: {a.get('local_path')}")
        lines.append(f"    események ({a.get('event_count')} db): {a.get('events_path')}")
        if a.get("timing"):
            from stage_timer import format_timing
            lines.append(f"    időmérés: {format_timing(a['timing'])}")
    lines.append(f"Összesített események ({result.get('event_count', 0)} db): {result.get('match_events_path')}")
    if result.get("event_counts"):
        lines.append("    " + ", ".join(f"{k}: {v}" for k, v in sorted(result["event_counts"].items())))
    for name, hm in (result.get("heatmaps") or {}).items():
        if hm.get("status") == "success":
            out_dir = result.get("out_dir") or ""
            files = [os.path.join(out_dir, os.path.basename(u)) for u in (hm.get("team_heatmaps") or {}).values() if u]
            lines.append(f"Hőtérkép ({name}): {', '.join(files) or '-'}")
    for name, c in (result.get("compilations") or {}).items():
        if c.get("status") == "success":
            lines.append(f"{c.get('title', name)} ({len(c.get('clips') or [])} klip): {c.get('merged_local_path')}")
    for n in result.get("notes") or []:
        lines.append(f"Megjegyzés: {n}")
    for w in result.get("warnings") or []:
        lines.append(f"Figyelmeztetés: {w}")
    lines.append(f"Futásidő: {result.get('elapsed_sec', 0):.0f} mp")
    return "\n".join(lines)

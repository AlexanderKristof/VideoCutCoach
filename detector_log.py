"""
Az eseménydetektor bemenetének rögzítése és visszajátszása (detector log).

Az elemzés közben (main.py -> TacticalEngine.update_tracks) minden feldolgozott képkockáról egy sor kerül egy
gzip-elt JSON-sor fájlba (events_<id>.detlog.jsonl.gz a clips mappában):

    1. sor (fejléc):  {"v": 1, "kind": "detlog", "start_sec": ..., "end_sec": ..., "fps": ...,
                       "team1_attacks_right": true, "pitch": [105.0, 68.0], "video_path": "..."}
    képkocka-sorok:   [t, bx, by, ball_status, ball_conf, players]
        t            – idő az eredeti videóban (mp), mint az események "time" mezője
        bx, by       – a labda pálya-pozíciója (m) vagy null, ha nincs labda
        ball_status  – "d" (detektált), "p" (becsült / predikált) vagy null
        ball_conf    – a labda-detekció bizonyossága (0..1) vagy null
        players      – [[track_id, team_id, team_known(0/1), is_keeper(0/1), x, y], ...]
    egyéb sorok:      {"kind": "dir", "t": ..., "team1_attacks_right": ...}  (irányváltás)

Ez pontosan az, amit az EventDetector.update() kap, ezért a visszajátszás (iter_frames + replay) videó és GPU
nélkül ugyanazt az eseménylistát adja, mint az eredeti futás – így a detektor szabályai / küszöbei a valódi
meccseken hangolhatók (tools/replay_events.py, tools/hangolas.py).

A fájl mérete: kb. 100 bájt / képkocka (gzip) – egy 90 perces meccs néhány MB.
"""
import gzip
import json
import math
import os

FORMAT_VERSION = 1
SUFFIX = ".detlog.jsonl.gz"


def log_path_for(events_path):
    """Az events_<id>.json mellé tartozó napló útvonala."""
    base = events_path[:-5] if events_path.endswith(".json") else events_path
    return base + SUFFIX


def _r(v, nd=2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


class DetectorLogWriter:
    """Képkockánkénti detektor-bemenet írása. Hibánál (pl. betelt lemez) csendben leáll – az elemzés megy tovább."""

    def __init__(self, path, header=None):
        self.path = path
        self.frames = 0
        self.error = None
        self._f = None
        try:
            self._f = gzip.open(path, "wt", encoding="utf-8", compresslevel=6)
            head = {"v": FORMAT_VERSION, "kind": "detlog"}
            head.update(header or {})
            self._f.write(json.dumps(head, ensure_ascii=False) + "\n")
        except OSError as ex:
            self._fail(ex)

    @property
    def active(self):
        return self._f is not None

    def _fail(self, ex):
        self.error = str(ex)
        try:
            if self._f is not None:
                self._f.close()
        except OSError:
            pass
        self._f = None

    def frame(self, t, ball_xy, players, ball_status=None, ball_conf=None):
        if self._f is None:
            return
        ps = []
        for p in players or ():
            pc = p.get("pitch_coords")
            if pc is None:
                continue
            tid = p.get("track_id")
            ps.append([int(tid) if isinstance(tid, (int, float)) else tid,
                       int(p.get("team_id") or 0), 1 if p.get("team_known", True) else 0,
                       1 if p.get("is_keeper") else 0, _r(pc[0]), _r(pc[1])])
        bx = by = None
        if ball_xy is not None:
            bx, by = _r(ball_xy[0]), _r(ball_xy[1])
        row = [_r(t, 3), bx, by, ball_status, _r(ball_conf, 3), ps]
        try:
            self._f.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.frames += 1
        except (OSError, ValueError) as ex:
            self._fail(ex)

    def direction(self, t, team1_attacks_right):
        if self._f is None:
            return
        try:
            self._f.write(json.dumps({"kind": "dir", "t": _r(t, 3),
                                      "team1_attacks_right": bool(team1_attacks_right)}) + "\n")
        except (OSError, ValueError) as ex:
            self._fail(ex)

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except OSError as ex:
                self.error = str(ex)
            self._f = None


def read(path):
    """
    A napló beolvasása: (fejléc, sorok). A sorok: ("frame", t, ball | None, ball_status, ball_conf, players-dictek)
    vagy ("dir", t, team1_attacks_right). Csonka (pl. megszakított elemzésből maradt) fájlnál az olvasható
    részt adja vissza.
    """
    header, rows = {}, []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    break   # csonka utolsó sor
                if i == 0 and isinstance(obj, dict) and obj.get("kind") == "detlog":
                    header = obj
                    continue
                if isinstance(obj, dict):
                    if obj.get("kind") == "dir":
                        rows.append(("dir", float(obj["t"]), bool(obj["team1_attacks_right"])))
                    continue
                t, bx, by, bst, bconf, ps = obj
                ball = (float(bx), float(by)) if bx is not None and by is not None else None
                players = [{"track_id": p[0], "team_id": p[1], "team_known": bool(p[2]), "is_keeper": bool(p[3]),
                            "pitch_coords": (float(p[4]), float(p[5]))}
                           for p in ps if p[4] is not None and p[5] is not None]
                rows.append(("frame", float(t), ball, bst, bconf, players))
    except (OSError, EOFError, gzip.BadGzipFile):
        pass   # csonka gzip: ami eddig beolvasható volt
    return header, rows


def replay(path, detector_module=None, config=None, ball_filter=None):
    """
    A napló visszajátszása egy friss EventDetector-on. config: EventConfig (vagy None = alapértelmezett, a fejléc
    irányával). ball_filter: opcionális függvény (rows) -> rows a labda-sorok utólagos tisztítására.
    Visszatér: (események, statisztika, fejléc).
    """
    if detector_module is None:
        import events as detector_module
    header, rows = read(path)
    if ball_filter is not None:
        rows = ball_filter(rows)
    cfg = config if config is not None else detector_module.EventConfig()
    if config is None:
        cfg.team1_attacks_right = bool(header.get("team1_attacks_right", True))
    det = detector_module.EventDetector(cfg, *(header.get("pitch") or (105.0, 68.0)))
    last_t = None
    for row in rows:
        if row[0] == "dir":
            det.set_attack_direction(row[2])
            continue
        _, t, ball, _bst, _bconf, players = row
        det.update(t, ball, players)
        last_t = t
    end = header.get("end_sec")
    det.finalize(float(end) if end is not None else last_t)
    return sorted(det.events, key=lambda e: e["time"]), det.stats, header


def find_logs(match_or_events_data, base_dir=None):
    """
    Egy (match_)events JSON-hoz tartozó naplók: [(félidő-kulcs vagy None, útvonal)] – a "detector_logs"
    (meccs-összesítő) vagy "detector_log" (egy elemzés) mezőből; relatív útvonal a base_dir-hez képest.
    """
    out = []
    logs = match_or_events_data.get("detector_logs")
    if isinstance(logs, dict):
        for key in ("first_half", "second_half"):
            p = logs.get(key)
            if p:
                out.append((key, p))
    elif match_or_events_data.get("detector_log"):
        out.append((None, match_or_events_data["detector_log"]))
    res = []
    for key, p in out:
        if isinstance(p, dict):
            p = p.get("path") or p.get("url")
        if not p:
            continue
        cand = [p]
        if base_dir:
            cand.append(os.path.join(base_dir, os.path.basename(p)))
        for c in cand:
            if os.path.isfile(c):
                res.append((key, c))
                break
    return res

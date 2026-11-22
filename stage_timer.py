"""
Könnyű, szakaszonkénti időmérő az elemző ciklushoz (PROFILOZÁS).

    timer = StageTimer()
    with timer("decode"): ret, frame = cap.read()
    ...
    timer.frames += 1
    timer.summary()      -> az events JSON "timing" blokkja
    timer.table_text()   -> magyar táblázat a debug naplóba
    timer.one_line()     -> egysoros összefoglaló a képernyőre

time.perf_counter alapú; a "mérés" ~1 µs / szakasz, a ciklus költségéhez képest elhanyagolható.
"""
import time
from contextlib import contextmanager

STAGES = ("decode", "detect", "ball_pass", "pitch", "grass_quality", "filters", "team", "ball_tracker",
          "engine", "draw_review", "encode", "other", "pitch_overlap")
# Más szakasszal EGY IDŐBEN (külön szálon) futó szakaszok: a falióra-idő felosztásában nem számítanak
# (különben a szakaszok összege meghaladná a teljes időt), csak tájékoztató sorként jelennek meg
OVERLAPPED = {"pitch_overlap"}
STAGE_LABELS = {
    "decode": "dekódolás",
    "detect": "felismerés + követés",
    "ball_pass": "labda-kivágás / külön menet",
    "pitch": "pályakövetés",
    "grass_quality": "fűmodell + minőség",
    "filters": "szűrők (fű, pályahatár, álló alak)",
    "team": "csapatba sorolás",
    "ball_tracker": "labdakövető",
    "engine": "motor / események",
    "draw_review": "rajzolás + áttekintő",
    "encode": "kódolás / írás",
    "other": "egyéb",
    "pitch_overlap": "(pályakövetés + fűmaszk a háttérszálon, átfedve)",
}


class StageTimer:
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self.frames = 0
        self.t_start = time.perf_counter()
        self.t_end = None

    @contextmanager
    def __call__(self, stage):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - t0)

    def add(self, stage, seconds):
        self.totals[stage] = self.totals.get(stage, 0.0) + float(seconds)
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def stop(self):
        self.t_end = time.perf_counter()

    def wall(self):
        return (self.t_end or time.perf_counter()) - self.t_start

    def summary(self):
        wall = self.wall()
        measured = sum(v for k, v in self.totals.items() if k not in OVERLAPPED)
        totals = dict(self.totals)
        if wall > measured:
            totals["other"] = totals.get("other", 0.0) + (wall - measured)
        n = max(1, self.frames)
        order = [s for s in STAGES if s in totals] + [s for s in totals if s not in STAGES]
        stages = {s: {"label": STAGE_LABELS.get(s, s), "total_sec": round(totals[s], 3),
                      "ms_per_frame": round(1000.0 * totals[s] / n, 2),
                      "share_pct": round(100.0 * totals[s] / max(1e-9, wall), 1)} for s in order}
        return {"frames": self.frames, "total_sec": round(wall, 2),
                "fps": round(self.frames / max(1e-9, wall), 2),
                "ms_per_frame": round(1000.0 * wall / n, 1), "stages": stages}

    def table_text(self):
        s = self.summary()
        lines = [f"Időmérés: {s['frames']} képkocka, {s['total_sec']:.1f} mp, {s['fps']:.2f} kép/mp "
                 f"({s['ms_per_frame']:.1f} ms/képkocka)",
                 f"{'szakasz':<36}{'ms/képkocka':>12}{'arány':>8}"]
        for st in s["stages"].values():
            lines.append(f"{st['label']:<36}{st['ms_per_frame']:>12.2f}{st['share_pct']:>7.1f}%")
        return "\n".join(lines)

    def one_line(self):
        s = self.summary()
        top = sorted((v for k, v in s["stages"].items() if k not in OVERLAPPED), key=lambda v: -v["ms_per_frame"])[:5]
        return (f"Időmérés: {s['fps']:.2f} kép/mp ({s['ms_per_frame']:.0f} ms/képkocka) – "
                + ", ".join(f"{v['label']} {v['ms_per_frame']:.0f} ms ({v['share_pct']:.0f}%)" for v in top))


def format_timing(timing):
    """Egy events JSON "timing" blokkjából rövid szöveg (pl. az elemzes.py félidő-összegzéséhez)."""
    if not timing:
        return "nincs időmérés"
    top = sorted((v for k, v in (timing.get("stages") or {}).items() if k not in OVERLAPPED),
                 key=lambda v: -v.get("ms_per_frame", 0))[:4]
    return (f"{timing.get('fps', 0):.2f} kép/mp, {timing.get('ms_per_frame', 0):.0f} ms/képkocka; "
            + ", ".join(f"{v['label']} {v['ms_per_frame']:.0f} ms ({v['share_pct']:.0f}%)" for v in top))

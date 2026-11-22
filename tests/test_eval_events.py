"""
Regressziós teszt: az eseménydetektor (events.py) pontossága / felidézése zajosított szintetikus
ground truth-on (tools/eval_events.py + tools/synth_match.py --gt-only, renderelés nélkül).

Futtatás: python3 tests/test_eval_events.py   (pytest nem szükséges; ~20-40 mp)
numpy nélkül (a synth_match-hoz kell) a teszt kihagyva, 0-s kilépési kóddal ér véget.

A küszöbök a 2026-11 állapot mért értékei alatt vannak egy kis tartalékkal (lásd az alábbi
táblázatot a futás végén); ha egy módosítás ezek alá viszi, az valódi visszalépés.
"""
import os
import sys
import tempfile
import shutil
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

SEEDS = (6, 21, 42)
NOISE_SEEDS = (1, 2, 3)
DURATION = 120.0
TOL = 2.0

# (zajszint, típus) -> (min. pontosság, min. felidézés); "_összes" = minden értékelt típus együtt
THRESHOLDS = {
    ("gt", "_összes"): (0.95, 0.95),
    ("gt", "goal"): (1.0, 1.0),
    ("gt", "gk_goal_kick"): (0.9, 0.9),
    ("kozepes", "_összes"): (0.85, 0.60),
    ("kozepes", "shot"): (0.9, 0.55),
    ("kozepes", "goal"): (0.9, 0.75),
    ("kozepes", "gk_goal_kick"): (0.9, 0.6),
    # labdás mezőny-események (2026-11: gt P=1.00 R=0.90, kozepes P=0.99 R=0.78; a sikertelen passz és a
    # kezdőrúgás GT-je kevés / a szimulátorban a kezdőrúgó messze áll a labdától, ezért ott csak pontosság)
    ("gt", "pass"): (0.95, 0.8),
    ("kozepes", "pass"): (0.9, 0.65),
    ("gt", "kickoff"): (0.95, 0.5),
    ("kozepes", "pass_failed"): (0.8, 0.3),
}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("  (kihagyva: numpy nem érhető el – a synth_match.py-hoz kell)")
        return 0
    import eval_events as ev

    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="test_eval_events_")
    try:
        gts = [ev.generate_gt(s, DURATION, tmp) for s in SEEDS]
        res = ev.evaluate(gts, ["gt", "kozepes"], list(NOISE_SEEDS), TOL)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(ev.format_table(res, title=f"events.py – magok {SEEDS}, zaj-magok {NOISE_SEEDS}, tűrés ±{TOL} s"))
    failed = []
    for (lvl, typ), (pmin, rmin) in THRESHOLDS.items():
        v = ev.totals(res[lvl]) if typ == "_összes" else res[lvl][typ]
        p, r = ev.pr(v)
        ok = p is not None and r is not None and p >= pmin - 1e-9 and r >= rmin - 1e-9
        line = f"{'OK  ' if ok else 'HIBA'}  {lvl:8s} {typ:14s} P={p} (min {pmin})  R={r} (min {rmin})"
        print(line)
        if not ok:
            failed.append(line)
    # A gól-események új mezői (visszafelé kompatibilis bővítés)
    import events
    det_ok = True
    for gt in gts[:1]:
        frames = ev.degrade(gt, ev.NOISE_PRESETS["gt"], seed=0)
        evs = ev.run_detector(events, gt, frames)
        goals = [e for e in evs if e["type"] == "goal"]
        det_ok = bool(goals) and all({"time", "team", "goal_x", "desc", "start_sec", "end_sec", "confidence",
                                      "confirmed"} <= set(g) for g in goals)
    print(f"{'OK  ' if det_ok else 'HIBA'}  gól-esemény mezők (confidence, confirmed)")
    if not det_ok:
        failed.append("gól mezők")
    print(f"\nIdő: {time.time() - t0:.1f} s")
    if failed:
        print(f"{len(failed)} küszöb nem teljesült")
        return 1
    print("Minden küszöb teljesült")
    return 0


def test_eval_events_thresholds():   # pytest-kompatibilis belépési pont
    check(main() == 0, "a kiértékelési küszöbök nem teljesültek")


if __name__ == "__main__":
    sys.exit(main())

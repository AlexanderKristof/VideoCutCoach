"""
Valódi felvételes mérés: a teljes elemzés lefuttatása rövid valódi klipeken, és az eredmények
egy táblázatba összesítve (labda, pályakövetés, videóminőség, csapatok, sebesség, események).

Ground truth nincs – ez összehasonlító mérés: ugyanazon klipeken a változtatás előtt és után
futtatva látszik, mi javult. A klipek a valós felvételből vágott néhány mp-es részletek
(pl. a valos-minta ág minta/klip_*.mp4 fájljai).

Használat:
    python tools/eval_real.py minta/klip_*.mp4 --kimenet meres_uj
    python tools/eval_real.py minta/klip_*.mp4 --kimenet meres_uj --osszevet meres_regi/osszesites.json

A klipeket a pipeline két "félidőként" elemzi (első és második fele), hogy a teljes lánc
(vágás, elemzés, összefésülés, hőtérkép) is lefusson.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def clip_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except Exception:
        return None


def parse_log(log_path):
    """A motor naplójából a labda- és csapat-összefoglaló sorok."""
    res = {}
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return res
    m = re.search(r"Labdakövető: (\d+) képkocka, detektált (\d+) \(([\d.]+)%\), becsült (\d+) \(([\d.]+)%\)", txt)
    if m:
        res["labda_kepkocka"] = int(m.group(1))
        res["labda_detektalt_pct"] = float(m.group(3))
        res["labda_becsult_pct"] = float(m.group(5))
    m = re.search(r"Auto csapat: (.*)", txt)
    if m:
        res["csapat"] = m.group(1).strip()[:160]
    return res


def summarize_half(events_path):
    d = json.load(open(events_path, encoding="utf-8"))
    pt = (d.get("pitch_tracking") or {}).get("frames") or {}
    total = sum(v for k, v in pt.items() if k != "finomítás") or 1
    vq = d.get("video_quality") or {}
    tm = d.get("timing") or {}
    tracks = (d.get("trajectories") or {}).get("tracks") or {}
    out = {
        "palya_ok_pct": round(100.0 * pt.get("ok", 0) / total, 1),
        "palya_bizonytalan_pct": round(100.0 * pt.get("bizonytalan", 0) / total, 1),
        "palya_kalibr_kell_pct": round(100.0 * pt.get("kalibráció szükséges", 0) / total, 1),
        "minoseg": vq.get("grade"),
        "csapat_szinek": d.get("team_colors"),
        "esemenyek": len(d.get("events") or []),
        "palyak": len(tracks),
        "fps": tm.get("fps"),
    }
    log = os.path.join(os.path.dirname(events_path), "analysis_debug_" +
                       os.path.basename(events_path)[len("events_"):-len(".json")] + ".log")
    out.update(parse_log(log))
    return out


def run_clip(clip, out_dir, extra_args):
    dur = clip_duration(clip) or 10.0
    mid = dur / 2.0
    halves = f"0:00,{mid - 0.5:.1f},{mid:.1f},{dur - 0.1:.1f}"
    cmd = [sys.executable, os.path.join(ROOT, "elemzes.py"), clip, "--felidok", halves,
           "--kimenet", out_dir, "--osszeallitasok", ""] + list(extra_args)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0
    events = re.findall(r"események \(\d+ db\): (.+events_\w+\.json)", res.stdout)
    # csak a félidőnkénti events_<id>.json (a match_events_* összesítőt kihagyjuk)
    halves_sum = [summarize_half(p) for p in events
                  if os.path.basename(p).startswith("events_") and os.path.exists(p)]
    return {"klip": os.path.basename(clip), "hossz_mp": round(dur, 1), "futasido_mp": round(elapsed, 1),
            "kilepesi_kod": res.returncode, "felidok": halves_sum,
            "hiba": None if res.returncode == 0 else res.stderr[-800:]}


def merge_halves(halves):
    """Egy klip két felének átlaga (a táblázathoz)."""
    if not halves:
        return {}
    keys = ["palya_ok_pct", "palya_bizonytalan_pct", "palya_kalibr_kell_pct", "labda_detektalt_pct",
            "labda_becsult_pct", "fps"]
    out = {}
    for k in keys:
        vals = [h[k] for h in halves if isinstance(h.get(k), (int, float))]
        out[k] = round(sum(vals) / len(vals), 1) if vals else None
    out["esemenyek"] = sum(h.get("esemenyek", 0) for h in halves)
    out["minoseg"] = "/".join(str(h.get("minoseg")) for h in halves)
    return out


COLS = [("palya_ok_pct", "pálya ok %"), ("palya_kalibr_kell_pct", "kalibr. kell %"),
        ("labda_detektalt_pct", "labda %"), ("labda_becsult_pct", "becsült %"),
        ("minoseg", "minőség"), ("esemenyek", "esem."), ("fps", "kép/mp")]


def print_table(rows, prev=None):
    prev_by = {r["klip"]: merge_halves(r["felidok"]) for r in (prev or {}).get("klipek", [])}
    head = f"{'klip':<18}" + "".join(f"{name:>16}" for _, name in COLS)
    print(head)
    print("-" * len(head))
    for r in rows:
        m = merge_halves(r["felidok"])
        p = prev_by.get(r["klip"], {})
        cells = []
        for k, _ in COLS:
            v = m.get(k)
            if k in p and p[k] is not None and isinstance(v, (int, float)) and isinstance(p[k], (int, float)):
                cells.append(f"{p[k]}→{v}")
            else:
                cells.append("–" if v is None else str(v))
        print(f"{r['klip']:<18}" + "".join(f"{c:>16}" for c in cells))
        if r["hiba"]:
            print(f"   HIBA (kód {r['kilepesi_kod']}): {r['hiba'][-300:]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Valódi felvételes összehasonlító mérés rövid klipeken")
    ap.add_argument("klipek", nargs="+", help="klipek (pl. minta/klip_*.mp4)")
    ap.add_argument("--kimenet", required=True, help="kimeneti mappa (az elemzések és az osszesites.json)")
    ap.add_argument("--osszevet", help="egy korábbi osszesites.json – a táblázat előtte→utána értékeket mutat")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:  # a -- utáni kapcsolók változatlanul az elemzes.py-nak mennek
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    clips = []
    for pat in a.klipek:
        clips += sorted(glob.glob(pat)) or [pat]
    os.makedirs(a.kimenet, exist_ok=True)
    rows = []
    for c in clips:
        print(f"== {os.path.basename(c)} ...", flush=True)
        rows.append(run_clip(c, os.path.join(a.kimenet, os.path.splitext(os.path.basename(c))[0]), extra))
    summary = {"klipek": rows, "ido": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.kimenet, "osszesites.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    prev = json.load(open(a.osszevet, encoding="utf-8")) if a.osszevet else None
    print()
    print_table(rows, prev)
    print(f"\nRészletek: {os.path.join(a.kimenet, 'osszesites.json')}")
    return 0 if all(r["kilepesi_kod"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

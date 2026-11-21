#!/usr/bin/env python3
"""
A pályakövető (pitch_tracker.PitchTracker) mérése szintetikus Veo-szerű klipeken.

Mit csinál:
  * szintetikus klipet generál (tools/synth_match.py) vagy a meglévőt újrahasznosítja
    (a ground truth JSON képkockánként tartalmazza a valós kép -> pálya homográfiát);
  * a követőt két módban futtatja:
      kezi – valószerű kézi kalibráció: az első képkockán LÁTHATÓ pályapontok (felezővonal
             végei, kezdőkör metszéspontjai, 16-os sarkai ...) közül legfeljebb 6, 1,5 px
             kattintási zajjal;
      auto – kalibráció nélkül (automatikus kalibráció a pályavonalakból);
      rossz – szándékosan hibás kézi kalibráció (6 m eltolás): az önértékelés próbája
             (csak kérésre: --modes kezi,auto,rossz);
  * kiírja a klip tizedeire a pályahiba mediánját és 90. percentilisét (méter, a képen látható
    pályaterület rácspontjain), a státuszok megoszlását, a vonal-finomítás elfogadási arányát,
    az önértékelés megbízhatóságát (a > 5 m hibás képkockák mekkora részét jelzi nem 'ok'-nak)
    és a képkockánkénti időt.

A hiba a követő BELSŐ becslésére vonatkozik (akkor is, ha nem 'ok' státusszal nem adja vissza),
így látszik, hogy a jelzés mennyire fedi a valós hibát. A 'visszaadott' sor csak az 'ok'
képkockák hibája (amit a main.py ténylegesen felhasznál).

Példák:
    python3 tools/eval_pitch_tracker.py                       # 3 mag, alap + nehéz profil, 75 s
    python3 tools/eval_pitch_tracker.py --seeds 6 --profiles alap --modes kezi --duration 30
    python3 tools/eval_pitch_tracker.py --tracker-module /ut/a/regi/pitch_tracker.py   # régi változat
    python3 tools/eval_pitch_tracker.py --frames-dir /tmp/kepek    # annotált képkockák a klip végéről

Csak numpy + OpenCV + ffmpeg (a klip generálásához) kell.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

PITCH_L, PITCH_W = 105.0, 68.0

# Kattintható pályapontok (m): (név, x, y)
PITCH_LANDMARKS = [
    ("felezővonal felső vége", 52.5, 0.0), ("felezővonal alsó vége", 52.5, 68.0),
    ("kezdőkör felső metszéspont", 52.5, 34.0 - 9.15), ("kezdőkör alsó metszéspont", 52.5, 34.0 + 9.15),
    ("kezdőkör bal széle", 52.5 - 9.15, 34.0), ("kezdőkör jobb széle", 52.5 + 9.15, 34.0),
    ("bal 16-os felső sarka", 16.5, 13.84), ("bal 16-os alsó sarka", 16.5, 54.16),
    ("bal 16-os felső alapvonal-pont", 0.0, 13.84), ("bal 16-os alsó alapvonal-pont", 0.0, 54.16),
    ("jobb 16-os felső sarka", 88.5, 13.84), ("jobb 16-os alsó sarka", 88.5, 54.16),
    ("jobb 16-os felső alapvonal-pont", 105.0, 13.84), ("jobb 16-os alsó alapvonal-pont", 105.0, 54.16),
    ("bal felső szöglet", 0.0, 0.0), ("jobb felső szöglet", 105.0, 0.0),
    ("bal alsó szöglet", 0.0, 68.0), ("jobb alsó szöglet", 105.0, 68.0),
]

BAD_M = 5.0      # ennél nagyobb hiba: a követőnek jeleznie kell (nem 'ok')
GOOD_M = 2.0     # ennél kisebb hiba: jó becslés


def load_tracker_module(path):
    if not path:
        import pitch_tracker
        return pitch_tracker
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("pitch_tracker_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def manual_calibration(gt, max_points=6, sigma_px=1.5, margin_px=12, seed=0):
    """Az első képkockán látható pályapontokból kézi kalibráció (H kép -> pálya), kattintási zajjal."""
    w, h = gt["width"], gt["height"]
    H0 = np.array(gt["frames"][0]["H"], np.float64).reshape(3, 3)
    G0 = np.linalg.inv(H0)
    q = H0 @ np.array([w / 2.0, h * 0.9, 1.0])
    ref = G0 @ (q / q[2])
    rng = np.random.default_rng(seed)
    cand = []
    for name, x, y in PITCH_LANDMARKS:
        p = G0 @ np.array([x, y, 1.0])
        if np.sign(p[2]) != np.sign(ref[2]):
            continue
        u, v = p[0] / p[2], p[1] / p[2]
        if margin_px <= u <= w - margin_px and margin_px <= v <= h - margin_px:
            cand.append((name, x, y, u, v))
    if len(cand) < 4:
        return None, [c[0] for c in cand]
    if len(cand) > max_points:
        # a lehető legszétszórtabb pontok (mohó legtávolabbi-pont kiválasztás)
        pts = np.array([[c[3], c[4]] for c in cand])
        sel = [int(np.argmax(pts[:, 1]))]
        while len(sel) < max_points:
            d = np.min(np.linalg.norm(pts[:, None] - pts[sel][None], axis=2), axis=1)
            sel.append(int(np.argmax(d)))
        cand = [cand[i] for i in sorted(sel)]
    sig = sigma_px * w / 1280.0
    img = np.array([[u + rng.normal(0, sig), v + rng.normal(0, sig)] for _n, _x, _y, u, v in cand])
    pit = np.array([[x, y] for _n, x, y, _u, _v in cand])
    H, _ = cv2.findHomography(img, pit, 0)
    return H, [c[0] for c in cand]


_GRID = {}


def pitch_error(H_est, H_true, w, h):
    """Medián hiba (m) a képen látható pályaterület rácspontjain; nan, ha nincs látható pálya."""
    key = (w, h)
    if key not in _GRID:
        uu, vv = np.meshgrid(np.arange(20, w, 40), np.arange(20, h, 40))
        _GRID[key] = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)
    p = _GRID[key]

    def apply(H):
        X = H[:, :2] @ p.T + H[:, 2:3]
        return (X[:2] / X[2]).T, X[2]

    t, wt = apply(H_true)
    ref = (H_true @ np.array([w / 2, h - 1, 1.0]))[2]
    ok = (wt * np.sign(ref) > 0) & (t[:, 0] >= 0) & (t[:, 0] <= PITCH_L) & (t[:, 1] >= 0) & (t[:, 1] <= PITCH_W)
    if not ok.any():
        return float("nan")
    if H_est is None:
        return float("inf")
    e, we = apply(H_est)
    d = np.linalg.norm(e[ok] - t[ok], axis=1)
    d[~np.isfinite(d)] = 1e3
    return float(np.median(np.minimum(d, 1e3)))


def ensure_clip(clip_dir, seed, profile, duration, width, height):
    os.makedirs(clip_dir, exist_ok=True)
    video = os.path.join(clip_dir, f"{profile}_s{seed}_{int(duration)}s_{width}x{height}.mp4")
    gt_path = os.path.splitext(video)[0] + "_gt.json"
    if not (os.path.exists(video) and os.path.exists(gt_path)):
        import synth_match
        print(f"Klip generálása: {video}", flush=True)
        synth_match.generate(video, gt_path, duration=duration, width=width, height=height, fps=25, seed=seed,
                             audio=False, progress=False, camera_profile=profile)
    return video, gt_path


def run_one(job):
    """Egy (klip, mód) futtatás. Visszatér: eredmény-szótár (képkockánkénti adatokkal)."""
    video, gt_path, mode, module_path, frames_dir, threads = job
    if threads:
        cv2.setNumThreads(threads)
    pt = load_tracker_module(module_path)
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    w, h = gt["width"], gt["height"]
    calib_names = []
    H0 = None
    if mode in ("kezi", "rossz"):
        H0, calib_names = manual_calibration(gt, seed=gt.get("seed", 0))
        if mode == "rossz" and H0 is not None:
            # hibás kézi kalibráció (pl. rossz pontot kattintott): a pálya 6 m-rel eltolva
            H0 = np.array([[1, 0, 6.0], [0, 1, 0], [0, 0, 1]]) @ H0
    tracker = pt.PitchTracker((w, h), H0=H0, auto_init=True)
    cap = cv2.VideoCapture(video)
    n = len(gt["frames"])
    save_ids = set()
    if frames_dir:
        save_ids = {int(n * 0.85), int(n * 0.97)}
        os.makedirs(frames_dir, exist_ok=True)
    rows = []
    saved = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        fr = gt["frames"][i]
        boxes = [tuple(o[5:9]) for o in fr["objs"]]
        t0 = time.perf_counter()
        H_ret, q = tracker.update(frame, boxes)
        dt = time.perf_counter() - t0
        H_true = np.array(fr["H"], np.float64).reshape(3, 3)
        H_int = tracker.current_H()
        err = pitch_error(H_int, H_true, w, h)
        rows.append({"err": err, "status": q.get("status"), "dt": dt,
                     "returned": H_ret is not None,
                     "refine_tried": bool(q.get("refine_tried", q.get("n_visible") is not None)),
                     "refined": bool(q.get("refined")), "conf": q.get("conf"),
                     "q": {k: q.get(k) for k in ("err_m", "precision", "support", "innovation", "n_visible",
                                                 "motion_ok", "inliers", "lost", "camera_refined")
                           if q.get(k) is not None}})
        if q.get("relocalize"):
            rows[-1]["q"]["reloc"] = q["relocalize"]
        if i in save_ids:
            img = frame.copy()
            if H_true is not None:
                pt.draw_pitch_overlay(img, H_true, color=(0, 0, 255), alpha=0.6, thickness=1)
            if H_int is not None:
                pt.draw_pitch_overlay(img, H_int, color=(255, 255, 0), alpha=0.9, thickness=2)
            txt = f"#{i} {q.get('status')} conf={q.get('conf')} hiba={err:.2f} m (cian: becsles, piros: valos)"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (6, 8), (18 + tw, 18 + th), (0, 0, 0), -1)
            cv2.putText(img, txt, (12, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            base = os.path.splitext(os.path.basename(video))[0]
            path = os.path.join(frames_dir, f"{base}_{mode}_{i:05d}.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(path)
    cap.release()
    return {"video": os.path.basename(video), "mode": mode, "rows": rows, "calib_names": calib_names,
            "saved": saved}


def summarize(res):
    rows = res["rows"]
    n = len(rows)
    err = np.array([r["err"] for r in rows], np.float64)
    st = [r["status"] for r in rows]
    okm = np.array([s == "ok" for s in st])
    valid = np.isfinite(err) | np.isinf(err)      # nan: nincs látható pálya
    valid &= ~np.isnan(err)
    tenths = []
    for k in range(10):
        a, b = k * n // 10, (k + 1) * n // 10
        e = err[a:b][valid[a:b]]
        e = np.where(np.isinf(e), 1e3, e)
        tenths.append((float(np.median(e)) if len(e) else float("nan"),
                       float(np.percentile(e, 90)) if len(e) else float("nan")))
    e_all = np.where(np.isinf(err), 1e3, err)
    bad = valid & (e_all > BAD_M)
    good = valid & (e_all < GOOD_M)
    counts = {}
    for s in st:
        counts[s] = counts.get(s, 0) + 1
    tried = sum(r["refine_tried"] for r in rows)
    acc = sum(r["refined"] for r in rows)
    ret_ok = valid & okm
    dts = np.array([r["dt"] for r in rows])
    return {
        "tenths": tenths, "counts": counts, "n": n,
        "bad": int(bad.sum()), "bad_flagged": int((bad & ~okm).sum()),
        "good": int(good.sum()), "good_ok": int((good & okm).sum()),
        "ok_bad": int((okm & bad).sum()), "ok_n": int(okm.sum()),
        "ret_median": float(np.median(e_all[ret_ok])) if ret_ok.any() else float("nan"),
        "ret_p90": float(np.percentile(e_all[ret_ok], 90)) if ret_ok.any() else float("nan"),
        "refine_tried": tried, "refine_acc": acc,
        "ms_mean": float(dts.mean() * 1000), "ms_p95": float(np.percentile(dts, 95) * 1000),
        "ms_max": float(dts.max() * 1000),
    }


def fmt_tenths(t, idx):
    return " ".join(f"{v[idx]:6.1f}" if np.isfinite(v[idx]) else "     –" for v in t)


def print_summary(label, s):
    print(f"\n=== {label} ===")
    print(f"  tized:        " + " ".join(f"{k + 1:6d}" for k in range(10)))
    print(f"  medián (m):   " + fmt_tenths(s["tenths"], 0))
    print(f"  p90 (m):      " + fmt_tenths(s["tenths"], 1))
    cnt = ", ".join(f"{k}: {v} ({100.0 * v / s['n']:.0f}%)" for k, v in sorted(s["counts"].items()))
    print(f"  státusz:      {cnt}")
    flag = (100.0 * s["bad_flagged"] / s["bad"]) if s["bad"] else float("nan")
    okr = (100.0 * s["good_ok"] / s["good"]) if s["good"] else float("nan")
    print(f"  önértékelés:  > {BAD_M:.0f} m hibás képkocka {s['bad']}, ebből jelzett (nem 'ok') {s['bad_flagged']} "
          f"({flag:.0f}%); < {GOOD_M:.0f} m-es képkockák {okr:.0f}%-a 'ok'; "
          f"'ok' de > {BAD_M:.0f} m: {s['ok_bad']} / {s['ok_n']}")
    print(f"  visszaadott:  'ok' képkockák hibája medián {s['ret_median']:.2f} m, p90 {s['ret_p90']:.2f} m")
    rr = (100.0 * s["refine_acc"] / s["refine_tried"]) if s["refine_tried"] else float("nan")
    print(f"  finomítás:    {s['refine_tried']} próbálkozás, {s['refine_acc']} elfogadva ({rr:.0f}%)")
    print(f"  idő:          átlag {s['ms_mean']:.1f} ms/képkocka, p95 {s['ms_p95']:.1f} ms, max {s['ms_max']:.0f} ms")


def main(argv=None):
    ap = argparse.ArgumentParser(description="A pályakövető mérése szintetikus Veo-szerű klipeken.")
    ap.add_argument("--seeds", default="6,7,8", help="véletlen magok vesszővel (alap: 6,7,8)")
    ap.add_argument("--profiles", default="alap,nehez", help="kameraprofilok: alap, nehez (alap: mindkettő)")
    ap.add_argument("--modes", default="kezi,auto", help="módok: kezi, auto (alap: mindkettő)")
    ap.add_argument("--duration", type=float, default=75.0, help="klip hossza mp-ben (alap: 75)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--clip-dir", default=os.path.join(tempfile.gettempdir(), "pitch_eval_klipek"),
                    help="a generált klipek könyvtára (újrahasznosítja a meglévőket)")
    ap.add_argument("--tracker-module", default=None, help="más pitch_tracker.py változat mérése (útvonal)")
    ap.add_argument("--jobs", type=int, default=1, help="párhuzamos futások (az időmérést torzítja; alap: 1)")
    ap.add_argument("--frames-dir", default=None, help="annotált képkockák mentése a klip végéről")
    ap.add_argument("--json", default=None, help="összesítés JSON-ba")
    ap.add_argument("--dump", default=None, help="képkockánkénti napló (JSONL) könyvtára")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    jobs = []
    for prof in profiles:
        for seed in seeds:
            video, gt_path = ensure_clip(args.clip_dir, seed, prof, args.duration, args.width, args.height)
            for mode in modes:
                jobs.append((video, gt_path, mode, args.tracker_module, args.frames_dir,
                             1 if args.jobs > 1 else 0))
    t0 = time.time()
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(run_one, jobs, chunksize=1)
    else:
        results = [run_one(j) for j in jobs]
    out = {}
    agg = {}
    for (video, _g, mode, *_), res in zip(jobs, results):
        s = summarize(res)
        label = f"{os.path.basename(video)} – {mode}"
        if res["calib_names"]:
            label += f" (kézi pontok: {len(res['calib_names'])})"
        print_summary(label, s)
        for p in res["saved"]:
            print(f"  annotált képkocka: {p}")
        out[label] = s
        prof = os.path.basename(video).split("_")[0]
        a = agg.setdefault((prof, mode), {"bad": 0, "flag": 0, "worst": [], "ms": []})
        a["bad"] += s["bad"]
        a["flag"] += s["bad_flagged"]
        a["worst"].append(max(v[0] for v in s["tenths"] if np.isfinite(v[0])))
        a["ms"].append(s["ms_mean"])
    print("\n=== Összesítés ===")
    for (prof, mode), a in sorted(agg.items()):
        flag = 100.0 * a["flag"] / a["bad"] if a["bad"] else float("nan")
        print(f"  {prof:6s} {mode:5s}: legrosszabb tized-medián {max(a['worst']):.2f} m "
              f"(klipenként: {', '.join(f'{v:.2f}' for v in a['worst'])}); "
              f"> {BAD_M:.0f} m képkockák jelzése {flag:.0f}% ({a['flag']}/{a['bad']}); "
              f"idő átlag {np.mean(a['ms']):.1f} ms")
    print(f"Teljes futás: {time.time() - t0:.0f} s")
    if args.dump:
        # képkockánkénti napló (hibakereséshez): egy JSON sor / képkocka
        os.makedirs(args.dump, exist_ok=True)
        for res in results:
            path = os.path.join(args.dump, f"{os.path.splitext(res['video'])[0]}_{res['mode']}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i, r in enumerate(res["rows"]):
                    f.write(json.dumps({"i": i, **r}, ensure_ascii=False, default=float) + "\n")
            print(f"Képkockánkénti napló: {path}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

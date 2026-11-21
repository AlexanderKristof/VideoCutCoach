#!/usr/bin/env python3
"""
Változatos tanító képkockák kimentése saját (Veo) felvételekből a focis detektor tanításához.

Mit csinál:
  1. videónként egyenletesen mintavételez (alap: 2 mp-enként jelölt képkocka; a meccs eleje / vége
     levágható), és minden jelöltre jellemzőket számol: kicsinyített szürke kép (a tartalomhoz),
     HSV-hisztogram (a megvilágítás / színek), élesség (Laplace-szórás), átlagos fényerő, és a
     kamera elmozdulása az előző jelölthöz képest (pásztázás);
  2. a homályos (élesség < --min-sharpness) és a majdnem azonos képeket kiszűri (dHash
     Hamming-távolság < --dedupe-bits a már megtartottakhoz képest);
  3. a maradékból "legtávolabbi pont" mintavétellel választ --per-video darabot, így a különböző
     időpontok, kamera-irányok és fényviszonyok egyenletesen szerepelnek;
  4. a képeket JPEG-ként menti (<videonév>_<képkocka>.jpg) és egy frames.csv-t ír (fájl, videó,
     képkocka, idő, fényerő, élesség, mozgás, dHash).

Példa:
    python3 tools/export_training_frames.py D:\\Veo\\*.mp4 --out tanito_kepek --per-video 150
    python3 tools/export_training_frames.py meccs.mp4 --out tanito --every-sec 1 --skip-start 300 --skip-end 300

Ezután: előcímkézés (tools/prelabel.py), kézi javítás CVAT-ban / Label Studióban, tanítás –
lásd docs/MODELL_TANITAS.md. Csak olyan felvételt használj, amelyhez a hozzájárulás megvan
(docs/LICENC_ES_ARUSITAS.md).
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np


def dhash(gray, size=8):
    """Különbség-hash (64 bit): a kicsinyített kép szomszédos pixeleinek összehasonlítása."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def frame_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / max(1.0, hist.sum())
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    return {"thumb": thumb, "hist": hist, "sharpness": sharp, "brightness": bright, "dhash": dhash(gray)}


def camera_shift(prev_thumb, thumb):
    """A kamera durva elmozdulása két jelölt között (fázis-korreláció a kicsinyített képen, képpont/32)."""
    if prev_thumb is None:
        return 0.0
    (dx, dy), _ = cv2.phaseCorrelate(prev_thumb.astype(np.float64), thumb.astype(np.float64))
    return float(np.hypot(dx, dy))


def farthest_point_select(feats, k, seed=0):
    """Legtávolabbi pont mintavétel a jellemzővektorokon (változatosság maximalizálása)."""
    n = len(feats)
    if n <= k:
        return list(range(n))
    X = np.stack(feats)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    d = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return sorted(chosen)


def scan_video(path, every_sec=2.0, skip_start=0.0, skip_end=0.0, min_sharpness=40.0, dedupe_bits=6, log=print):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log(f"⚠️ A videó nem nyitható meg: {path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(every_sec * fps)))
    first = int(skip_start * fps)
    last = total - int(skip_end * fps) if total else None
    cands, kept_hashes, prev_thumb = [], [], None
    n_blur = n_dup = 0
    idx = first
    while last is None or idx < last:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        f = frame_features(frame)
        f["motion"] = camera_shift(prev_thumb, f["thumb"])
        prev_thumb = f["thumb"]
        if f["sharpness"] < min_sharpness:
            n_blur += 1
        elif any(hamming(f["dhash"], h) < dedupe_bits for h in kept_hashes[-200:]):
            n_dup += 1
        else:
            kept_hashes.append(f["dhash"])
            f.update({"frame": idx, "time": idx / fps})
            cands.append(f)
        idx += step
    cap.release()
    log(f"{os.path.basename(path)}: {len(cands)} jelölt (homályos: {n_blur}, majdnem azonos: {n_dup})")
    return cands


def export(videos, out_dir, per_video=150, every_sec=2.0, skip_start=0.0, skip_end=0.0, min_sharpness=40.0,
           dedupe_bits=6, jpeg_quality=95, seed=0, log=print):
    img_dir = os.path.join(out_dir, "kepek")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for vp in videos:
        cands = scan_video(vp, every_sec, skip_start, skip_end, min_sharpness, dedupe_bits, log)
        if not cands:
            continue
        bmax = max(1.0, max(c["brightness"] for c in cands))
        feats = [np.concatenate([c["thumb"].flatten() * 0.5, c["hist"] * 4.0,
                                 [c["brightness"] / bmax, min(1.0, c["motion"] / 4.0) * 0.5]]) for c in cands]
        sel = farthest_point_select(feats, per_video, seed)
        stem = os.path.splitext(os.path.basename(vp))[0].replace(" ", "_")
        cap = cv2.VideoCapture(vp)
        for i in sel:
            c = cands[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"{stem}_{c['frame']:07d}.jpg"
            cv2.imwrite(os.path.join(img_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            rows.append({"file": f"kepek/{name}", "video": os.path.abspath(vp), "frame": c["frame"],
                         "time_sec": round(c["time"], 2), "width": frame.shape[1], "height": frame.shape[0],
                         "brightness": round(c["brightness"], 1), "sharpness": round(c["sharpness"], 1),
                         "motion": round(c["motion"], 2), "dhash": f"{c['dhash']:016x}"})
        cap.release()
        log(f"  -> {len(sel)} kép mentve")
    csv_path = os.path.join(out_dir, "frames.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "video", "frame", "time_sec", "width", "height", "brightness",
                                          "sharpness", "motion", "dhash"])
        w.writeheader()
        w.writerows(rows)
    log(f"Kész: {len(rows)} kép, lista: {csv_path}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Változatos tanító képkockák kimentése videókból")
    ap.add_argument("videos", nargs="+", help="videófájlok vagy minták (pl. D:\\Veo\\*.mp4) vagy mappa")
    ap.add_argument("--out", required=True, help="kimeneti mappa (kepek/ + frames.csv)")
    ap.add_argument("--per-video", type=int, default=150, help="ennyi kép videónként (alap 150)")
    ap.add_argument("--every-sec", type=float, default=2.0, help="jelölt képkocka ennyi mp-enként (alap 2)")
    ap.add_argument("--skip-start", type=float, default=0.0, help="a videó elejéből ennyi mp kimarad")
    ap.add_argument("--skip-end", type=float, default=0.0, help="a videó végéből ennyi mp kimarad")
    ap.add_argument("--min-sharpness", type=float, default=40.0, help="ennél homályosabb kép kimarad (Laplace-szórás)")
    ap.add_argument("--dedupe-bits", type=int, default=6, help="dHash távolság: ennél hasonlóbb kép kimarad (0-64)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG minőség")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    videos = []
    for v in a.videos:
        if os.path.isdir(v):
            videos += sorted(glob.glob(os.path.join(v, "*.mp4")) + glob.glob(os.path.join(v, "*.mov")))
        else:
            videos += sorted(glob.glob(v)) or [v]
    if not videos:
        print("Nincs bemeneti videó.")
        return 1
    export(videos, a.out, a.per_video, a.every_sec, a.skip_start, a.skip_end, a.min_sharpness, a.dedupe_bits,
           a.quality, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

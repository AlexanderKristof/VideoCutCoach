#!/usr/bin/env python3
"""
Félautomata előcímkézés: a jelenlegi detektor (+ saját követő) javasolt dobozai, hogy a kézi címkézés
gyorsabb legyen. A kimenet CVAT-ba (COCO 1.0 / YOLO import) vagy Label Studióba (előcímkék,
"predictions") tölthető, ott csak javítani kell.

Módok:
  1. képek (tools/export_training_frames.py kimenete):
        python3 tools/prelabel.py --images tanito_kepek --out elocimke --format coco,labelstudio
  2. videó követéssel (a követés kiszűri az egyszeri téves dobozokat, és azonosítót ad – CVAT-ban
     sávként is javítható): minden N. képkockát menti a javasolt dobozokkal
        python3 tools/prelabel.py --video meccs.mp4 --every 25 --start 600 --duration 120 --out elocimke_video
  3. szintetikus meccs (tools/synth_match.py) ground truth-ja: hibátlan "előcímke" a teljes lánc
     (export -> CVAT / tanítás) kipróbálásához és szintetikus előtanításhoz
        python3 tools/prelabel.py --video synth.mp4 --synth-gt synth_gt.json --every 10 --out synth_cimke

Detektor: a model_config.json szerinti saját háttér (backend "onnx"), vagy --onnx <fájl>; ha a beállítás
"ultralytics", fejlesztői módban az ultralytics predict + saját követő fut (AGPL – csak belső használatra).
Osztályok (tools/dataset_formats.py): 0 ball, 1 goalkeeper, 2 player, 3 referee.
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_formats import CLASSES, write_coco, write_yolo, write_labelstudio  # noqa: E402

ROLE_TO_CLASS = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}


def make_backend(onnx_path=None, onnx_cfg=None, model_cfg_path=None):
    import model_config
    from detection.backend import create_backend
    cfg = model_config.load_model_config(model_cfg_path)
    if onnx_path:
        cfg["backend"], cfg["onnx_path"] = "onnx", onnx_path
    if onnx_cfg:
        cfg["onnx"] = dict(cfg.get("onnx") or {}, **onnx_cfg)
    if cfg["backend"] == "ultralytics":
        print("⚠️ ultralytics háttér: előcímkézéshez az ultralytics predict + saját követő fut (AGPL, csak belső használat)")
        cfg["backend"] = "ultralytics_predict"
    return create_backend(cfg, "cpu", base_dir=ROOT)


def _cls_of(role_classes):
    m = {}
    for role, ids in role_classes.items():
        for c in ids:
            m.setdefault(int(c), ROLE_TO_CLASS[role])
    return m


def boxes_from_detections(det, role_classes, conf=0.3, ball_conf=0.15):
    cmap = _cls_of(role_classes)
    out = []
    for i in range(len(det)):
        c = cmap.get(int(det.cls[i]))
        if c is None:
            continue
        thr = ball_conf if c == 0 else conf
        if det.conf[i] < thr:
            continue
        out.append({"cls": c, "xyxy": [float(v) for v in det.xyxy[i]], "score": float(det.conf[i])})
    return out


def tiled_detect(detector, img, tile=640, overlap=0.25):
    """
    Csempés felismerés eredeti felbontáson (a kicsi, távoli labdához és játékosokhoz): a teljes kép
    felismerése + átfedő tile x tile csempék, a dobozok visszavetítve és osztályonként NMS-sel összefésülve.
    """
    from detection.detector_base import Detections, nms
    h, w = img.shape[:2]
    full = detector.detect(img)
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile) + 1, step)) or [0]
    ys = list(range(0, max(1, h - tile) + 1, step)) or [0]
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    crops, offs = [], []
    for y in ys:
        for x in xs:
            crops.append(img[y:y + tile, x:x + tile])
            offs.append((x, y))
    xyxy, conf, cls = [full.xyxy], [full.conf], [full.cls]
    for d, (ox, oy) in zip(detector.detect_batch(crops), offs):
        if len(d) == 0:
            continue
        b = d.xyxy.copy()
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        # a csempe szélén levágott (félbe vágott) alakok dobozait kihagyjuk – azokat a szomszéd csempe adja
        ch, cw = tile if oy + tile <= h else h - oy, tile if ox + tile <= w else w - ox
        inner = ((d.xyxy[:, 0] > 2) | (ox == 0)) & ((d.xyxy[:, 1] > 2) | (oy == 0)) & \
                ((d.xyxy[:, 2] < cw - 2) | (ox + cw >= w)) & ((d.xyxy[:, 3] < ch - 2) | (oy + ch >= h))
        xyxy.append(b[inner]); conf.append(d.conf[inner]); cls.append(d.cls[inner])
    xyxy = np.concatenate(xyxy) if xyxy else np.zeros((0, 4), np.float32)
    conf = np.concatenate(conf); cls = np.concatenate(cls)
    keep = nms(xyxy, conf, 0.5, cls=cls)
    return Detections(xyxy[keep], conf[keep], cls[keep], full.names)


def fit_grass_model(image_paths, n=30):
    """Videónként tanult fűmodell a képek egy mintájából (a nézősor-szűréshez)."""
    from pitch_filters import GrassModel
    gm = GrassModel()
    step = max(1, len(image_paths) // n)
    for p in image_paths[::step]:
        img = cv2.imread(p)
        if img is not None:
            gm.observe(img)
    return gm


def drop_spectators(boxes, img, grass_model):
    """
    A kerítés / reklámtábla előtti nézősor kiszűrése: a személy talppontja a fő fűterület felső
    széléhez (a túlsó pályaszél menti sávhoz) túl közel van. A labda-dobozokat nem érinti.
    Visszatér: (megtartott, eldobott darabszám).
    """
    from pitch_filters import grass_top_profile
    m, sc = grass_model.mask(img)
    top = grass_top_profile(m)
    if top is None:
        return boxes, 0
    H = img.shape[0]
    keep, dropped = [], 0
    for b in boxes:
        if b["cls"] != 0:
            x1, y1, x2, y2 = b["xyxy"]
            cx = int(min(m.shape[1] - 1, max(0, (x1 + x2) / 2 * sc)))
            t = top[cx]
            if not np.isnan(t) and y2 * sc < t + max(0.5 * (y2 - y1) * sc, 0.012 * H * sc):
                dropped += 1
                continue
        keep.append(b)
    return keep, dropped


def prelabel_images(backend, roles, image_paths, images_root, conf, ball_conf, tile=0, spectators=False):
    items = []
    gm = fit_grass_model(image_paths) if spectators else None
    n_drop = 0
    for i, p in enumerate(image_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        det = tiled_detect(backend.detector, img, tile) if tile else backend.detector.detect(img)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(image_paths)} kép")
        boxes = boxes_from_detections(det, roles, conf, ball_conf)
        if gm is not None:
            boxes, d = drop_spectators(boxes, img, gm)
            n_drop += d
        items.append({"file": os.path.relpath(p, images_root), "width": img.shape[1], "height": img.shape[0],
                      "boxes": boxes})
    if gm is not None:
        print(f"Nézősor-szűrés: {n_drop} doboz eldobva")
    return items


def prelabel_video(backend, roles, video, out_img_dir, every=25, start=0.0, duration=None, min_track_len=3,
                   ball_conf=0.15):
    """Detektor + saját követő képkockánként; minden every. képkockán a legalább min_track_len hosszú követések."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    backend.reset(frame_rate=fps)
    cmap = _cls_of(roles)
    stem = os.path.splitext(os.path.basename(video))[0]
    os.makedirs(out_img_dir, exist_ok=True)
    seen = {}
    items = []
    k = 0
    n_max = int(duration * fps) if duration else None
    while n_max is None or k < n_max:
        ok, frame = cap.read()
        if not ok:
            break
        fidx = int(start * fps) + k
        res, raw_ball = backend.track(frame)
        boxes = []
        for b in res[0].boxes:
            tid = int(b.id[0]) if b.id is not None else None
            if tid is not None:
                seen[tid] = seen.get(tid, 0) + 1
            if tid is None or seen[tid] < min_track_len:
                continue
            c = cmap.get(int(b.cls[0]))
            if c is not None:
                boxes.append({"cls": c, "xyxy": [float(v) for v in b.xyxy[0]], "score": float(b.conf[0]), "track": tid})
        if raw_ball:
            bx, bc = max(raw_ball, key=lambda t: t[1])
            if bc >= ball_conf:
                boxes.append({"cls": 0, "xyxy": list(bx), "score": bc})
        if k % every == 0:
            name = f"{stem}_{fidx:07d}.jpg"
            cv2.imwrite(os.path.join(out_img_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            items.append({"file": name, "width": frame.shape[1], "height": frame.shape[0], "boxes": boxes})
        k += 1
    cap.release()
    return items


def labels_from_synth(gt_path, video, out_img_dir, every=10, min_h=6.0):
    """A synth_match.py ground truth-jából: játékos / kapus / bíró / látható labda dobozok (nézők nélkül)."""
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    roles = {p["id"]: p["role"] for p in gt["people"]}
    cap = cv2.VideoCapture(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    os.makedirs(out_img_dir, exist_ok=True)
    items = []
    for i, fr in enumerate(gt["frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        if i % every:
            continue
        boxes = []
        for o in fr["objs"]:
            role = roles.get(o[0])
            if role not in ROLE_TO_CLASS or (o[8] - o[6]) < min_h:
                continue
            boxes.append({"cls": ROLE_TO_CLASS[role], "xyxy": [float(v) for v in o[5:9]]})
        b = fr["ball"]
        if b.get("vis") and b.get("occ", 0) <= 0.5:
            boxes.append({"cls": 0, "xyxy": [float(v) for v in b["bbox"]]})
        name = f"{stem}_{i:07d}.jpg"
        cv2.imwrite(os.path.join(out_img_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        items.append({"file": name, "width": frame.shape[1], "height": frame.shape[0], "boxes": boxes})
    cap.release()
    return items


def write_outputs(items, out_dir, images_root, formats):
    for fmt in formats:
        if fmt == "coco":
            write_coco(items, os.path.join(out_dir, "coco"), images_root)
            print(f"COCO (CVAT: Upload annotations -> COCO 1.0): {os.path.join(out_dir, 'coco', 'annotations', 'train.json')}")
        elif fmt == "yolo":
            write_yolo(items, os.path.join(out_dir, "yolo"), images_root)
            print(f"YOLO txt: {os.path.join(out_dir, 'yolo')}")
        elif fmt == "labelstudio":
            p = write_labelstudio(items, os.path.join(out_dir, "labelstudio_tasks.json"))
            print(f"Label Studio import (előcímkékkel): {p}")
        else:
            print(f"⚠️ Ismeretlen formátum: {fmt}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Félautomata előcímkézés (detektor + saját követő / szintetikus GT)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="képmappa (vagy az export_training_frames.py kimeneti mappája)")
    src.add_argument("--video", help="videó (követéses előcímkézés vagy --synth-gt)")
    ap.add_argument("--synth-gt", help="a synth_match.py ground-truth JSON-ja (a --video szintetikus meccs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="coco,labelstudio", help="coco,yolo,labelstudio (vesszővel)")
    ap.add_argument("--onnx", help="ONNX modellfájl (felülírja a model_config.json-t)")
    ap.add_argument("--onnx-cfg", default="{}", help="további OnnxDetector beállítások JSON-ban")
    ap.add_argument("--model-config", help="model_config.json útvonala")
    ap.add_argument("--conf", type=float, default=0.3, help="személy-dobozok min. pontszáma (képes mód)")
    ap.add_argument("--ball-conf", type=float, default=0.15)
    ap.add_argument("--every", type=int, default=25, help="videó: minden N. képkocka kerül a kimenetbe")
    ap.add_argument("--start", type=float, default=0.0, help="videó: kezdés (mp)")
    ap.add_argument("--duration", type=float, default=None, help="videó: hossz (mp)")
    ap.add_argument("--min-track-len", type=int, default=3, help="videó: ennél rövidebb követés nem kerül be")
    ap.add_argument("--csempe", type=int, default=0, metavar="PX",
                    help="képes mód: csempés felismerés eredeti felbontáson (pl. 640) – a kicsi labdához; 0 = ki")
    ap.add_argument("--nezo-szures", action="store_true",
                    help="képes mód: a kerítés előtti nézősor kiszűrése (a fűterület felső széle alapján)")
    a = ap.parse_args(argv)
    formats = [f.strip() for f in a.format.split(",") if f.strip()]
    os.makedirs(a.out, exist_ok=True)
    if a.synth_gt:
        img_dir = os.path.join(a.out, "kepek")
        items = labels_from_synth(a.synth_gt, a.video, img_dir, max(1, a.every))
        write_outputs(items, a.out, img_dir, formats)
        print(f"Kész: {len(items)} kép (szintetikus GT)")
        return 0
    backend, roles = make_backend(a.onnx, json.loads(a.onnx_cfg), a.model_config)
    print(f"Detektor: {backend.describe()}")
    if a.images:
        root = a.images
        csvp = os.path.join(root, "frames.csv")
        if os.path.exists(csvp):
            with open(csvp, encoding="utf-8") as f:
                paths = [os.path.join(root, r["file"]) for r in csv.DictReader(f)]
        else:
            paths = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                           if p.lower().endswith((".jpg", ".jpeg", ".png")))
        items = prelabel_images(backend, roles, paths, root, a.conf, a.ball_conf, a.csempe, a.nezo_szures)
        write_outputs(items, a.out, root, formats)
    else:
        img_dir = os.path.join(a.out, "kepek")
        items = prelabel_video(backend, roles, a.video, img_dir, max(1, a.every), a.start, a.duration,
                               a.min_track_len, a.ball_conf)
        write_outputs(items, a.out, img_dir, formats)
    n = sum(len(it["boxes"]) for it in items)
    print(f"Kész: {len(items)} kép, {n} javasolt doboz")
    return 0


if __name__ == "__main__":
    sys.exit(main())

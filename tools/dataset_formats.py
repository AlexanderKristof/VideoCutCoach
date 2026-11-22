#!/usr/bin/env python3
"""
Tanító adatkészlet formátumok: YOLO txt, COCO json (CVAT-ba importálható, RF-DETR / D-FINE / YOLOX
tanításhoz), Label Studio előcímke-JSON – és átalakítás közöttük.

Belső ábrázolás (képenként egy dict):
    {"file": "kepek/video1_000123.jpg",     # a képek gyökeréhez relatív útvonal
     "width": 1920, "height": 1080,
     "boxes": [{"cls": 2, "xyxy": [x1, y1, x2, y2], "score": 0.87 (opcionális), "track": 5 (opcionális)}]}

Osztályok (a sorrend = osztály-sorszám; a COCO kimenetben a category_id ennél 1-gyel több – a CVAT így várja):
    0 ball (labda), 1 goalkeeper (kapus), 2 player (játékos), 3 referee (bíró)
Az angol nevek a model_config.build_role_class_map() alapértelmezett álneveivel egyeznek, így a
tanított modell beállítás nélkül is a helyes szerepkörökre képződik le (a magyar nevek is működnek).

Átalakítás (CLI):
    python3 tools/dataset_formats.py convert --from coco --input ann.json --images kepek/ --to yolo --out adat_yolo/ --val 0.15
    python3 tools/dataset_formats.py convert --from yolo --input adat_yolo/ --to coco --out adat_coco/ --layout rfdetr
"""
import argparse
import json
import os
import random
import shutil

CLASSES = ["ball", "goalkeeper", "player", "referee"]
CLASS_LABELS_HU = {"ball": "labda", "goalkeeper": "kapus", "player": "játékos", "referee": "bíró"}


# ====================================================================== közös
def split_items(items, val_frac=0.0, test_frac=0.0, seed=0, group_key=None):
    """
    Tanító / validáló / teszt felosztás. group_key(item) szerint csoportosít (pl. videónként), hogy
    ugyanannak a meccsnek a szomszédos képkockái ne kerüljenek a tanító ÉS a validáló halmazba is.
    """
    rng = random.Random(seed)
    if group_key is None:
        groups = {i: [it] for i, it in enumerate(items)}
    else:
        groups = {}
        for it in items:
            groups.setdefault(group_key(it), []).append(it)
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    n = len(items)
    out = {"train": [], "val": [], "test": []}
    for k in keys:
        if len(out["test"]) < test_frac * n:
            out["test"].extend(groups[k])
        elif len(out["val"]) < val_frac * n:
            out["val"].extend(groups[k])
        else:
            out["train"].extend(groups[k])
    if not out["train"]:   # kevés csoportnál se maradjon üresen a tanító halmaz
        donor = "val" if out["val"] else "test"
        out["train"], out[donor] = out[donor][: max(1, len(out[donor]) // 2)], out[donor][max(1, len(out[donor]) // 2):]
    return {k: v for k, v in out.items() if v or k == "train"}


def video_block_key(item, block_frames=750):
    """
    Csoportkulcs a felosztáshoz a "<videonév>_<képkocka>.jpg" fájlnévből: (videó, 30 mp-es blokk @ 25 FPS).
    Így egy meccs szomszédos képkockái egy halmazba kerülnek, de egyetlen videó is felosztható.
    """
    stem = os.path.splitext(os.path.basename(item["file"]))[0]
    video, _, num = stem.rpartition("_")
    if not video or not num.isdigit():
        return stem
    return f"{video}#{int(num) // block_frames}"


def _copy_image(src_root, rel, dst_dir, link=False):
    src = os.path.join(src_root, rel) if src_root else rel
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(rel))
    if os.path.abspath(src) == os.path.abspath(dst) or not os.path.exists(src):
        return os.path.basename(rel)
    if link:
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)
    return os.path.basename(rel)


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _clip_box(b, w, h):
    x1, y1, x2, y2 = [float(v) for v in b]
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    return x1, y1, x2, y2


# ====================================================================== YOLO txt
def write_yolo(items, out_dir, images_root=None, classes=CLASSES, splits=None, copy_images=True, link=False):
    """
    YOLO txt elrendezés: images/<split>/*.jpg, labels/<split>/*.txt ("osztály cx cy w h", 0..1 normalizált),
    data.yaml és classes.txt. splits: {"train": [...], "val": [...]} (alap: minden a train-be).
    """
    splits = splits or {"train": items}
    for split, its in splits.items():
        lab_dir = os.path.join(out_dir, "labels", split)
        img_dir = os.path.join(out_dir, "images", split)
        os.makedirs(lab_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        for it in its:
            name = _copy_image(images_root, it["file"], img_dir, link) if copy_images else os.path.basename(it["file"])
            w, h = float(it["width"]), float(it["height"])
            lines = []
            for b in it["boxes"]:
                x1, y1, x2, y2 = _clip_box(b["xyxy"], w, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"{int(b['cls'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            with open(os.path.join(lab_dir, os.path.splitext(name)[0] + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        for split in splits:
            f.write(f"{split}: images/{split}\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)))
    return out_dir


def read_yolo(root, classes=None):
    """A write_yolo elrendezésének beolvasása (a kép méretéhez OpenCV kell). Visszatér: (items, classes)."""
    import cv2
    if classes is None:
        p = os.path.join(root, "classes.txt")
        classes = _read_lines(p) if os.path.exists(p) else CLASSES
    items = []
    img_root = os.path.join(root, "images")
    for split in sorted(os.listdir(img_root)):
        for name in sorted(os.listdir(os.path.join(img_root, split))):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img = cv2.imread(os.path.join(img_root, split, name))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes = []
            lp = os.path.join(root, "labels", split, os.path.splitext(name)[0] + ".txt")
            if os.path.exists(lp):
                for line in _read_lines(lp):
                    p = line.split()
                    if len(p) < 5:
                        continue
                    c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                    boxes.append({"cls": c, "xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                                     (cx + bw / 2) * w, (cy + bh / 2) * h]})
            items.append({"file": os.path.join("images", split, name), "width": w, "height": h,
                          "boxes": boxes, "split": split})
    return items, classes


# ====================================================================== COCO json
def to_coco(items, classes=CLASSES, file_key=os.path.basename, id_start=1):
    """
    id_start: az első osztály category_id-ja. Alap 1 – a CVAT a 0-s category_id-t "címke nélkülinek"
    veszi ("annotation has no label"), és a COCO/Roboflow is 1-től számoz. A beolvasás (read_coco)
    mindkettőt kezeli (az id-k sorrendje szerint 0-tól folytonos osztály-sorszámra képez).
    """
    images, anns = [], []
    for i, it in enumerate(items):
        images.append({"id": i + 1, "file_name": file_key(it["file"]), "width": int(it["width"]),
                       "height": int(it["height"])})
        for b in it["boxes"]:
            x1, y1, x2, y2 = _clip_box(b["xyxy"], it["width"], it["height"])
            if x2 <= x1 or y2 <= y1:
                continue
            a = {"id": len(anns) + 1, "image_id": i + 1, "category_id": int(b["cls"]) + id_start,
                 "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                 "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0}
            if b.get("score") is not None:
                a["score"] = round(float(b["score"]), 4)
            if b.get("track") is not None:
                a["attributes"] = {"track_id": int(b["track"])}
            anns.append(a)
    cats = [{"id": i + id_start, "name": n, "supercategory": "foci"} for i, n in enumerate(classes)]
    return {"info": {"description": "Focielemző tanító adatkészlet"}, "licenses": [],
            "images": images, "annotations": anns, "categories": cats}


def write_coco(items, out_dir, images_root=None, classes=CLASSES, splits=None, layout="flat", link=False,
               id_start=1):
    """
    layout:
      "flat"   – out_dir/annotations/<split>.json + out_dir/images/<split>/ (CVAT "COCO 1.0", D-FINE, YOLOX)
      "rfdetr" – out_dir/{train,valid,test}/_annotations.coco.json a képek mellett (RF-DETR / Roboflow elrendezés)
    """
    splits = splits or {"train": items}
    for split, its in splits.items():
        if layout == "rfdetr":
            sdir = os.path.join(out_dir, {"val": "valid"}.get(split, split))
            img_dir, ann_path = sdir, os.path.join(sdir, "_annotations.coco.json")
        else:
            img_dir = os.path.join(out_dir, "images", split)
            ann_path = os.path.join(out_dir, "annotations", f"{split}.json")
        os.makedirs(os.path.dirname(ann_path), exist_ok=True)
        for it in its:
            _copy_image(images_root, it["file"], img_dir, link)
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(to_coco(its, classes, id_start=id_start), f, ensure_ascii=False)
    return out_dir


def read_coco(path, images_root=""):
    """COCO json -> (items, classes). A category_id-kat 0-tól folytonos osztály-sorszámra képezi le (névsorrend az id szerint)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    remap = {c["id"]: i for i, c in enumerate(cats)}
    classes = [c["name"] for c in cats]
    by_img = {}
    for a in data.get("annotations", []):
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append({"cls": remap[a["category_id"]], "xyxy": [x, y, x + w, y + h]})
    items = [{"file": os.path.join(images_root, im["file_name"]) if images_root else im["file_name"],
              "width": im["width"], "height": im["height"], "boxes": by_img.get(im["id"], [])}
             for im in data.get("images", [])]
    return items, classes


# ====================================================================== Label Studio
def write_labelstudio(items, out_path, image_url_prefix="/data/local-files/?d=", classes=CLASSES, model_version="elocimke"):
    """
    Label Studio import-JSON előcímkékkel ("predictions"). A címkézési felület (Labeling Interface):
        <View><Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
            <Label value="ball"/><Label value="goalkeeper"/><Label value="player"/><Label value="referee"/>
          </RectangleLabels></View>
    """
    tasks = []
    for it in items:
        w, h = float(it["width"]), float(it["height"])
        res = []
        for k, b in enumerate(it["boxes"]):
            x1, y1, x2, y2 = _clip_box(b["xyxy"], w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            res.append({"id": f"b{k}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                        "original_width": int(w), "original_height": int(h),
                        "value": {"x": 100 * x1 / w, "y": 100 * y1 / h, "width": 100 * (x2 - x1) / w,
                                  "height": 100 * (y2 - y1) / h, "rotation": 0,
                                  "rectanglelabels": [classes[int(b["cls"])]]},
                        **({"score": float(b["score"])} if b.get("score") is not None else {})})
        tasks.append({"data": {"image": image_url_prefix + it["file"].replace(os.sep, "/")},
                      "predictions": [{"model_version": model_version, "result": res}]})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=1)
    return out_path


def fix_coco_ids(path, out=None):
    """A 0-s category_id-val kezdődő COCO json átszámozása 1-től (CVAT "annotation has no label" hiba)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ids = sorted(c["id"] for c in data.get("categories", []))
    if not ids or ids[0] >= 1:
        print("Nincs mit javítani: a category_id-k már 1-től indulnak.")
        return 0
    shift = 1 - ids[0]
    for c in data["categories"]:
        c["id"] += shift
    for a in data.get("annotations", []):
        a["category_id"] += shift
    out = out or path
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Kész: {len(data.get('annotations', []))} doboz, a címkék 1-től számozva -> {out}")
    return 0


# ====================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Adatkészlet-formátumok átalakítása (YOLO txt <-> COCO json, Label Studio)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="átalakítás")
    c.add_argument("--from", dest="src", choices=("yolo", "coco"), required=True)
    c.add_argument("--input", required=True, help="YOLO: a gyökérmappa; COCO: az annotációs json")
    c.add_argument("--images", default="", help="COCO: a képek mappája")
    c.add_argument("--to", choices=("yolo", "coco", "labelstudio"), required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--layout", choices=("flat", "rfdetr"), default="flat")
    c.add_argument("--val", type=float, default=0.0, help="validáló arány (videónként csoportosítva)")
    c.add_argument("--test", type=float, default=0.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--coco-id-start", type=int, default=1,
                   help="COCO kimenet: az első osztály category_id-ja (alap 1; ha egy keretrendszer 0-tól várja: 0)")
    fx = sub.add_parser("cvat-javit", help="régi (0-tól számozott) COCO json javítása CVAT-importhoz (1-től számoz)")
    fx.add_argument("input", help="a COCO json (pl. elocimke/coco/annotations/train.json)")
    fx.add_argument("--out", help="kimenet (alap: felülírja a bemenetet)")
    a = ap.parse_args(argv)
    if a.cmd == "cvat-javit":
        return fix_coco_ids(a.input, a.out)
    if a.src == "yolo":
        items, classes = read_yolo(a.input)
        root = a.input
    else:
        items, classes = read_coco(a.input, a.images)
        root = None
    splits = split_items(items, a.val, a.test, a.seed, video_block_key) if (a.val or a.test) else None
    if a.to == "yolo":
        write_yolo(items, a.out, root, classes, splits)
    elif a.to == "coco":
        write_coco(items, a.out, root, classes, splits, layout=a.layout, id_start=a.coco_id_start)
    else:
        write_labelstudio(items, a.out, classes=classes)
    print(f"Kész: {len(items)} kép -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

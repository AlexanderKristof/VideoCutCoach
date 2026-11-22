"""
Saját focis detektor tanítása egy lépésben (RF-DETR, Apache-2.0) – a CVAT-ból exportált címkékből.

1) Adatkészlet (képek + CVAT "COCO 1.0" export → RF-DETR / Roboflow elrendezés, csempékkel):
    python tools\\tanitas.py adat --cimkek export\\annotations\\instances_default.json --kepek tanito\\kepek --ki adat_rfdetr

   Több CVAT-feladat (több meccs) együtt: a --cimkek és a --kepek többször is megadható, párban.

2) Tanítás + ONNX export + motor-beállítás (külön Python-környezetben, GPU-val):
    python tools\\tanitas.py tanit --adat adat_rfdetr --ki futasok\\elso --korok 40

   A végén: models\\focis_detektor.onnx és a model_config.json javaslata (a motor ezzel használja).

Miért csempék? A Veo-képen (1920×1080) a labda ~11 képpont. A modell bemenete ~900 képpont, ezért a
teljes kép kicsinyítve 5 képpont körüli labdát mutatna – ezt alig lehet megtanulni. A teljes kép mellé
eredeti felbontású 640×640-es kivágások is kerülnek (főleg a labda körül), így a modell a kicsi labdát
is megtanulja, és a motor labda-kivágásos menete (ball_crops.py, eredeti felbontás) is jól működik.

A felosztás (tanító / validáló) meccsenként, 30 mp-es blokkokban történik, a csempézés ELŐTT – így
ugyanannak a képnek a csempéi nem kerülnek mindkét halmazba.
"""
import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_formats as dfm  # noqa: E402

CLASSES = ["ball", "goalkeeper", "player", "referee"]


# ============================================================ adatkészlet
def _clip_box_to_tile(xyxy, tx, ty, ts, min_visible=0.5):
    """A doboz a csempe koordinátáiban, ha legalább min_visible része a csempén van; különben None."""
    x1, y1, x2, y2 = xyxy
    cx1, cy1 = max(x1, tx), max(y1, ty)
    cx2, cy2 = min(x2, tx + ts), min(y2, ty + ts)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    if (cx2 - cx1) * (cy2 - cy1) / area < min_visible:
        return None
    return [cx1 - tx, cy1 - ty, cx2 - tx, cy2 - ty]


def tile_positions(item, tile=640, per_image=2, rng=None, ball_cls=0):
    """
    Csempe-bal-felső sarkok egy képhez: minden labda körül egy (véletlen eltolással, hogy a labda ne
    mindig középen legyen), plusz per_image véletlen csempe a játékosok környékén (sok kép van labda nélkül).
    """
    rng = rng or random.Random(0)
    W, H = int(item["width"]), int(item["height"])
    if W < tile or H < tile:
        return []
    pos = []

    def clamp(x, y):
        return int(min(max(0, x), W - tile)), int(min(max(0, y), H - tile))

    for b in item["boxes"]:
        if b["cls"] != ball_cls:
            continue
        cx, cy = (b["xyxy"][0] + b["xyxy"][2]) / 2, (b["xyxy"][1] + b["xyxy"][3]) / 2
        pos.append(clamp(cx - tile / 2 + rng.uniform(-0.3, 0.3) * tile, cy - tile / 2 + rng.uniform(-0.3, 0.3) * tile))
    people = [b for b in item["boxes"] if b["cls"] != ball_cls]
    for _ in range(per_image):
        if people:
            b = rng.choice(people)
            cx, cy = (b["xyxy"][0] + b["xyxy"][2]) / 2, (b["xyxy"][1] + b["xyxy"][3]) / 2
            pos.append(clamp(cx - tile / 2 + rng.uniform(-0.4, 0.4) * tile, cy - tile / 2 + rng.uniform(-0.4, 0.4) * tile))
        else:
            pos.append(clamp(rng.uniform(0, W - tile), rng.uniform(0, H - tile)))
    out = []
    for p in pos:   # majdnem azonos csempék kiszűrése
        if all(abs(p[0] - q[0]) > tile * 0.25 or abs(p[1] - q[1]) > tile * 0.25 for q in out):
            out.append(p)
    return out


def make_tiles(item, image, tile=640, per_image=2, rng=None):
    """(csempe-item, csempe-kép) párok; a dobozok a csempe koordinátáiban."""
    res = []
    stem = os.path.splitext(os.path.basename(item["file"]))[0]
    for n, (tx, ty) in enumerate(tile_positions(item, tile, per_image, rng), 1):
        boxes = []
        for b in item["boxes"]:
            c = _clip_box_to_tile(b["xyxy"], tx, ty, tile)
            if c is not None:
                boxes.append({"cls": b["cls"], "xyxy": c})
        if not boxes:
            continue
        crop = image.crop((tx, ty, tx + tile, ty + tile))
        res.append(({"file": f"{stem}_cs{n}.jpg", "width": tile, "height": tile, "boxes": boxes}, crop))
    return res


def _check_classes(classes, path):
    missing = [c for c in CLASSES if c not in classes]
    extra = [c for c in classes if c not in CLASSES]
    if missing or extra:
        raise SystemExit(f"{path}: a címkék neve nem egyezik. Kell: {CLASSES}; a fájlban: {classes}")


def build_dataset(label_files, image_dirs, out_dir, val=0.15, tile=640, tiles_per_image=2, seed=0,
                  jpeg_quality=92, log=print):
    """A CVAT COCO export(ok)ból RF-DETR (Roboflow) elrendezésű adatkészlet: train/ valid/ test/ + _annotations.coco.json."""
    from PIL import Image   # a tanító környezetben (rfdetr) is megvan; OpenCV nem kell

    items = []
    for lf, imgdir in zip(label_files, image_dirs):
        its, classes = dfm.read_coco(lf, imgdir)
        _check_classes(classes, lf)
        order = [classes.index(c) for c in CLASSES]          # a fájl sorrendje -> a mi sorrendünk
        remap = {old: new for new, old in enumerate(order)}
        for it in its:
            it["boxes"] = [{"cls": remap[b["cls"]], "xyxy": b["xyxy"]} for b in it["boxes"]]
            it["src"] = os.path.join(imgdir, os.path.basename(it["file"]))
        missing = [it["src"] for it in its if not os.path.exists(it["src"])]
        if missing:
            raise SystemExit(f"{len(missing)} kép hiányzik a(z) {imgdir} mappából (pl. {missing[0]}). "
                             "A --kepek a CVAT-ba feltöltött képek mappája legyen.")
        items += its
        log(f"{lf}: {len(its)} kép")
    if not items:
        raise SystemExit("Nincs kép.")

    groups = {dfm.video_block_key(it) for it in items}
    # kevés 30 mp-es blokknál (pl. percenként kimentett képek) a blokkos felosztás nagyon egyenetlen lenne
    key = dfm.video_block_key if len(groups) >= 8 else None
    splits = dfm.split_items(items, val, 0.0, seed, key)
    if not splits.get("val"):
        # nagyon kevés képnél is legyen validáló halmaz (különben a tanítás nem tud mérni)
        tr = splits["train"]
        k = max(1, len(tr) // 7)
        splits = {"train": tr[k:], "val": tr[:k]}
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    rng = random.Random(seed)
    stats = {}
    for split, its in splits.items():
        folder = {"train": "train", "val": "valid"}.get(split, split)
        sdir = os.path.join(out_dir, folder)
        os.makedirs(sdir, exist_ok=True)
        out_items = []
        n_tiles = 0
        for it in its:
            try:
                img = Image.open(it["src"]).convert("RGB")
            except OSError:
                raise SystemExit(f"A kép nem olvasható: {it['src']}")
            name = os.path.basename(it["file"])
            shutil.copyfile(it["src"], os.path.join(sdir, name))
            out_items.append({"file": name, "width": it["width"], "height": it["height"], "boxes": it["boxes"]})
            if tile:
                for t_item, crop in make_tiles(it, img, tile, tiles_per_image, rng):
                    crop.save(os.path.join(sdir, t_item["file"]), quality=jpeg_quality)
                    out_items.append(t_item)
                    n_tiles += 1
        with open(os.path.join(sdir, "_annotations.coco.json"), "w", encoding="utf-8") as f:
            json.dump(roboflow_coco(out_items), f, ensure_ascii=False)
        stats[folder] = (len(its), n_tiles, sum(len(i["boxes"]) for i in out_items))
    # az RF-DETR a test/ mappát is várja: a validáló halmaz másolata (külön teszthez címkézz egy másik meccset)
    shutil.copytree(os.path.join(out_dir, "valid"), os.path.join(out_dir, "test"))
    for folder, (n, nt, nb) in stats.items():
        log(f"  {folder}: {n} kép + {nt} csempe, {nb} doboz")
    log(f"Kész: {out_dir}")
    return stats


def roboflow_coco(items):
    """COCO a Roboflow / RF-DETR szokása szerint: 0 = szülő-kategória ("foci"), 1..4 = ball, goalkeeper, player, referee."""
    data = dfm.to_coco(items, CLASSES, id_start=1)
    data["categories"] = [{"id": 0, "name": "foci", "supercategory": "none"}] + [
        {"id": i + 1, "name": n, "supercategory": "foci"} for i, n in enumerate(CLASSES)]
    return data


# ============================================================ tanítás
MODEL_SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase"}


def train(data_dir, out_dir, epochs=40, size="medium", resolution=None, batch=4, accum=4, lr=1e-4,
          models_dir="models", pretrain=None, log=print):
    try:
        import rfdetr
    except ImportError:
        raise SystemExit("Az rfdetr nincs telepítve ebben a Python-környezetben. Lásd docs/MODELL_TANITAS.md 5/a.")
    cls_name = MODEL_SIZES[size]
    Model = getattr(rfdetr, cls_name, None)
    if Model is None:
        raise SystemExit(f"Ez az rfdetr verzió nem ismeri: {cls_name}. Próbáld: --meret base")
    kwargs = {}
    if resolution:
        kwargs["resolution"] = int(resolution)
    if pretrain:
        # "nincs": előtanított súly nélkül (csak próbához – sokkal rosszabb); különben egy letöltött .pth útvonala
        kwargs["pretrain_weights"] = None if pretrain == "nincs" else pretrain
    model = Model(**kwargs)
    log(f"Tanítás: {cls_name}, {epochs} kör, adat: {data_dir} -> {out_dir}")
    model.train(dataset_dir=data_dir, epochs=epochs, batch_size=batch, grad_accum_steps=accum, lr=lr,
                output_dir=out_dir)
    best = None
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint.pth"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            best = p
            break
    if best is None:
        raise SystemExit(f"Nem található ellenőrzőpont a(z) {out_dir} mappában.")
    return export(best, out_dir, size, resolution, models_dir, log)


def export(checkpoint, out_dir, size="medium", resolution=None, models_dir="models", log=print):
    import rfdetr
    Model = getattr(rfdetr, MODEL_SIZES[size])
    kwargs = {"pretrain_weights": checkpoint}
    if resolution:
        kwargs["resolution"] = int(resolution)
    model = Model(**kwargs)
    onnx_dir = os.path.join(out_dir, "onnx")
    model.export(output_dir=onnx_dir)
    src = None
    for root, _, files in os.walk(onnx_dir):
        for f in files:
            if f.endswith(".onnx"):
                src = os.path.join(root, f)
    if src is None:
        raise SystemExit(f"Az ONNX export nem hozott létre .onnx fájlt ({onnx_dir}).")
    os.makedirs(models_dir, exist_ok=True)
    dst = os.path.join(models_dir, "focis_detektor.onnx")
    shutil.copyfile(src, dst)
    res = getattr(getattr(model, "model_config", None), "resolution", None) or resolution
    cfg = suggested_model_config(dst, res)
    cfg_path = os.path.join(out_dir, "model_config.javaslat.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log(f"ONNX: {dst}")
    log(f"Motor-beállítás javaslat: {cfg_path}  (másold a repó gyökerébe model_config.json néven)")
    return dst


def suggested_model_config(onnx_path, resolution=None):
    cfg = {
        "backend": "onnx",
        "onnx_path": onnx_path.replace("\\", "/"),
        # RF-DETR kimenet: 0 = szülő-kategória ("foci"), utána a mi osztályaink
        "onnx_class_names": ["-"] + CLASSES,
        "onnx": {"preset": "detr", "normalize": "imagenet"},
    }
    if resolution:
        cfg["input_size"] = [int(resolution), int(resolution)]
    return cfg


# ============================================================ parancssor
def main(argv=None):
    ap = argparse.ArgumentParser(description="Saját focis detektor tanítása (RF-DETR) a CVAT-címkékből")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("adat", help="adatkészlet a CVAT COCO exportból (csempékkel)")
    a1.add_argument("--cimkek", action="append", required=True, help="CVAT export: annotations/instances_default.json")
    a1.add_argument("--kepek", action="append", required=True, help="a feltöltött képek mappája (pl. tanito\\kepek)")
    a1.add_argument("--ki", required=True, help="kimeneti mappa (pl. adat_rfdetr)")
    a1.add_argument("--val", type=float, default=0.15, help="validáló arány (alap 0.15)")
    a1.add_argument("--csempe", type=int, default=640, help="csempeméret eredeti felbontásban (0 = nincs csempe)")
    a1.add_argument("--csempe-kepenkent", type=int, default=2, help="véletlen csempék száma képenként a labdásokon felül")
    a1.add_argument("--seed", type=int, default=0)

    a2 = sub.add_parser("tanit", help="tanítás + ONNX export (rfdetr kell hozzá)")
    a2.add_argument("--adat", required=True)
    a2.add_argument("--ki", required=True, help="a futás mappája (ellenőrzőpontok, napló)")
    a2.add_argument("--korok", type=int, default=40, help="tanítási körök (epoch) száma")
    a2.add_argument("--meret", choices=list(MODEL_SIZES), default="medium")
    a2.add_argument("--felbontas", type=int, default=None, help="bemeneti felbontás (56 többszöröse, pl. 896)")
    a2.add_argument("--batch", type=int, default=4)
    a2.add_argument("--accum", type=int, default=4, help="gradiens-gyűjtés (kis VRAM-nál nagyobb, kisebb batch-csel)")
    a2.add_argument("--lr", type=float, default=1e-4)
    a2.add_argument("--models", default="models")
    a2.add_argument("--elotanitott", default=None,
                    help="előtanított súly (.pth) útvonala, ha a letöltés nem megy; 'nincs' = nélküle (csak próba)")

    a3 = sub.add_parser("export", help="csak ONNX export egy meglévő ellenőrzőpontból")
    a3.add_argument("--ellenorzopont", required=True)
    a3.add_argument("--ki", required=True)
    a3.add_argument("--meret", choices=list(MODEL_SIZES), default="medium")
    a3.add_argument("--felbontas", type=int, default=None)
    a3.add_argument("--models", default="models")
    a = ap.parse_args(argv)

    if a.cmd == "adat":
        if len(a.cimkek) != len(a.kepek):
            raise SystemExit("A --cimkek és a --kepek párban adandó meg (ugyanannyiszor).")
        build_dataset(a.cimkek, a.kepek, a.ki, a.val, a.csempe, a.csempe_kepenkent, a.seed)
    elif a.cmd == "tanit":
        train(a.adat, a.ki, a.korok, a.meret, a.felbontas, a.batch, a.accum, a.lr, a.models, a.elotanitott)
    else:
        export(a.ellenorzopont, a.ki, a.meret, a.felbontas, a.models)
    return 0


if __name__ == "__main__":
    sys.exit(main())

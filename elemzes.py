"""
A teljes meccs elemzése parancssorból (szerver és MAUI kliens nélkül).

    python elemzes.py MECCS.mp4
    python elemzes.py MECCS.mp4 --felido-perc 40 --kalibracio "0.1,0.4;0.9,0.4;1.0,0.9;0.0,0.9" --kimenet eredmeny
    python elemzes.py MECCS.mp4 --nincs-vagas --gyors 3
    python elemzes.py MECCS.mp4 --hatterben            (laptopon: alacsony prioritás, kevés szál)
    python elemzes.py MECCS.mp4 --hatterben --szalak 2

Ugyanazt a folyamatot futtatja, mint a POST /jobs/analyze-match végpont (pipeline.py).
A --help torch nélkül is működik: a modell csak az elemzés indításakor töltődik be.
"""
import os
import sys
import time
import argparse
import contextlib


def build_parser():
    p = argparse.ArgumentParser(
        prog="elemzes.py",
        description="Teljes meccs elemzése egy lépésben: félidők felismerése, a meccs kivágása a szünet "
                    "nélkül, mindkét félidő elemzése, összesített események, hőtérképek és klip-összeállítások.",
    )
    p.add_argument("video", help="a teljes felvétel (pl. MECCS.mp4)")
    p.add_argument("--felido-perc", type=float, default=45.0, metavar="PERC",
                   help="várt félidőhossz percben (alap: 45; amatőr meccsen pl. 40 vagy 35)")
    p.add_argument("--kalibracio", default="", metavar='"x,y;x,y;x,y;x,y"',
                   help="kattintott pályapontok a képen, 0..1 relatív koordinátákkal (mint a kliensben); "
                        "nélküle a <videó>.kalibracio.json fájlt használja (tools/kalibracio.py), ha van, "
                        "különben a pályavonalakból próbál automatikusan kalibrálni")
    p.add_argument("--nincs-kalibracios-fajl", action="store_true",
                   help="a <videó>.kalibracio.json fájlt akkor se használja, ha létezik")
    p.add_argument("--kalibracio-pontok", default="", metavar='"x,y;..."',
                   help="a --kalibracio pontjainak pálya-koordinátái méterben (pl. a 16-os sarkai: "
                        '"88.5,13.84;105,13.84;105,54.16;88.5,54.16"); üresen a 4 pályasarok sorrendje')
    p.add_argument("--kalibracio-ido", type=float, default=None, metavar="MP",
                   help="melyik videó-időpont képén készült a kalibráció (mp, az eredeti videóban)")
    p.add_argument("--kimenet", default="clips", metavar="MAPPA",
                   help="kimeneti mappa (alap: clips – ezt szolgálja ki a szerver /clips alatt)")
    p.add_argument("--nincs-vagas", action="store_true",
                   help="ne készüljön szünet nélküli meccsvideó (az elemzés így is csak a félidőket nézi)")
    p.add_argument("--gyors", type=int, default=None, metavar="N",
                   help="gyors mód: felismerés csak minden N. képkockán (integrált GPU-hoz pl. 3)")
    p.add_argument("--felidok", default="", metavar="K1,V1,K2,V2",
                   help="kézi félidőhatárok (mp vagy pp:mm / ó:pp:mm), pl. 15:05,1:00:58,1:16:10,2:03:02 "
                        "– ha mind a négy megvan, nincs automatikus felismerés")
    p.add_argument("--csapat1-balra", action="store_true",
                   help="az 1. csapat az 1. félidőben balra támad (alap: jobbra)")
    p.add_argument("--osszeallitasok", default="goalkeepers,highlights", metavar="LISTA",
                   help="klip-összeállítások vesszővel: goalkeepers (kapusonként), highlights (gólok + lövések); "
                        "üres szöveg = nincs")
    p.add_argument("--letszam", action="store_true",
                   help="a félidő-felismerés YOLO létszámszámlálással (pontosabb, lassabb)")
    p.add_argument("--nincs-hoterkep", action="store_true", help="ne készüljenek hőtérképek")
    p.add_argument("--nincs-attekinto", action="store_true", help="ne készüljenek áttekintő lapok (review)")
    p.add_argument("--hatterben", action="store_true",
                   help="háttér-mód: alacsony prioritás (az ffmpeg is) és kevés szál, hogy közben a gép "
                        "használható maradjon (laptopon ajánlott); a szálszám alapból a magok fele, "
                        "akkumulátorról a negyede")
    p.add_argument("--szalak", type=int, default=None, metavar="N",
                   help="legfeljebb N szál a torch / OpenCV / ffmpeg számára (háttér-mód nélkül is használható)")
    p.add_argument("--kimenet-felbontas", type=float, default=1.0, metavar="ARÁNY",
                   help="az annotált elemzés-videó méretaránya (alap 1.0; pl. 0.5 = fél felbontás: 1080p helyett "
                        "960x540 – gyorsabb kódolás, kisebb fájl)")
    p.add_argument("--reszletes", action="store_true",
                   help="az elemző motor saját kiírásai is a képernyőre kerüljenek (alap: naplófájlba)")
    return p


def options_from_args(args):
    opts = {
        "expected_half_min": args.felido_perc,
        "calib_points": args.kalibracio,
        "calib_pitch_points": args.kalibracio_pontok,
        "calib_time_sec": args.kalibracio_ido,
        "skip_cut": args.nincs_vagas,
        "team1_attacks_right": not args.csapat1_balra,
        "compilations": args.osszeallitasok,
        "use_person_count": args.letszam,
        "heatmaps": not args.nincs_hoterkep,
        "review_export": not args.nincs_attekinto,
        "output_scale": args.kimenet_felbontas,
    }
    if args.felidok:
        parts = [x.strip() for x in args.felidok.split(",")]
        if len(parts) != 4:
            raise SystemExit("Hiba: a --felidok négy értéket vár (1. kezdet, 1. vége, 2. kezdet, 2. vége).")
        opts.update(zip(("first_start", "first_end", "second_start", "second_end"), parts))
    return opts


def apply_calibration_file(args, opts):
    """
    A tools/kalibracio.py által mentett <videó>.kalibracio.json használata, ha nincs --kalibracio
    kapcsoló (az 1. félidőé a calib_*, a 2. félidőé a calib2_* opciókba kerül).
    Visszatér: tájékoztató szöveg vagy None.
    """
    if args.kalibracio or getattr(args, "nincs_kalibracios_fajl", False):
        return None
    path = args.video + ".kalibracio.json"
    if not os.path.exists(path):
        return None
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import kalibracio
    data = kalibracio.load_calibration_json(path)
    calib = kalibracio.calibration_options(data)
    if not calib:
        return f"Figyelem: a kalibrációs fájl hibás vagy üres, nem használom: {path}"
    opts.update(calib)
    halves = [h for h, k in (("1.", "calib_points"), ("2.", "calib2_points")) if calib.get(k)]
    text = f"Kézi kalibráció a fájlból ({', '.join(halves)} félidő): {path}"
    size = (data.get("palya") or [105.0, 68.0])[:2]
    if (float(size[0]), float(size[1])) != (105.0, 68.0):
        text += (f"\nFigyelem: a fájl {size[0]:g} x {size[1]:g} m-es pályát ad meg, de az elemző motor egyelőre "
                 "105 x 68 m-es pályával számol (lásd README).")
    return text


class ProgressBar:
    """Egysoros haladásjelző a képernyőre (a kiírások naplóba terelése mellett is a valódi stdoutra ír)."""

    def __init__(self, stream, width=30):
        self.stream, self.width = stream, width
        self.t0 = time.monotonic()
        self._last = 0.0

    def __call__(self, pct, phase):
        now = time.monotonic()
        if now - self._last < 0.5 and pct < 100.0:
            return
        self._last = now
        filled = int(self.width * pct / 100.0)
        elapsed = now - self.t0
        eta = ""
        if 1.0 < pct < 100.0:
            rest = elapsed * (100.0 - pct) / pct
            eta = f" | hátra kb. {int(rest // 60)}:{int(rest % 60):02d}"
        line = f"\r[{'#' * filled}{'-' * (self.width - filled)}] {pct:5.1f}% {phase[:60]:<60}{eta}"
        self.stream.write(line)
        self.stream.flush()

    def done(self):
        self.stream.write("\n")
        self.stream.flush()


def apply_background(args):
    """
    Háttér-mód / szálkorlát beállítása a nehéz könyvtárak (torch, OpenCV) betöltése ELŐTT
    (az OMP_NUM_THREADS stb. csak így hat). Visszatér: tájékoztató szöveg vagy None.
    """
    if not args.hatterben and not args.szalak:
        return None
    import background
    threads = args.szalak or background.default_threads()
    background.set_default(background.BackgroundConfig(enabled=args.hatterben, threads=threads))
    background.limit_library_threads(threads)  # környezeti változók (OMP / MKL / OpenBLAS)
    text = f"Szálkorlát: {threads}"
    if args.hatterben:
        ok = background.lower_process_priority()
        text = ("Háttér-mód: alacsony prioritás" + ("" if ok else " (nem sikerült beállítani)") +
                f", {threads} szál")
        if background.on_battery():
            text += " (akkumulátoros üzem)"
    return text


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.video):
        print(f"Hiba: a videófájl nem található: {args.video}")
        return 2
    if args.gyors is not None:
        if args.gyors < 1:
            print("Hiba: a --gyors értéke legalább 1.")
            return 2
        # a model_config.py ezt a környezeti változót olvassa (a main.py importja előtt kell beállítani)
        os.environ["TACTICAL_DETECT_EVERY_N"] = str(args.gyors)
    if not (0.05 < args.kimenet_felbontas <= 1.0):
        print("Hiba: a --kimenet-felbontas 0.05 és 1.0 közötti arány (pl. 0.5).")
        return 2
    if args.szalak is not None and args.szalak < 1:
        print("Hiba: a --szalak értéke legalább 1.")
        return 2
    bg_info = apply_background(args)

    import pipeline
    from jobs import ProgressReporter, JobCancelled

    opts = options_from_args(args)
    calib_info = apply_calibration_file(args, opts)
    try:
        pipeline.coerce_options(opts)
        pipeline.parse_compilations(opts["compilations"])
    except ValueError as ex:
        print(f"Hiba: {ex}")
        return 2

    out_dir = os.path.abspath(args.kimenet)
    os.makedirs(out_dir, exist_ok=True)
    real_stdout = sys.stdout
    bar = ProgressBar(real_stdout)
    reporter = ProgressReporter(on_update=bar)
    log_path = os.path.join(out_dir, f"elemzes_{time.strftime('%Y%m%d_%H%M%S')}.log")
    print(f"Elemzés indul: {args.video}")
    if calib_info:
        print(calib_info)
    if bg_info:
        print(bg_info)
    print(f"Kimenet: {out_dir}" + ("" if args.reszletes else f"  (a motor kiírásai: {log_path})"))

    with open(log_path, "w", encoding="utf-8") as log:
        redirect = contextlib.nullcontext() if args.reszletes else contextlib.redirect_stdout(log)
        try:
            with redirect:
                import main as engine  # itt töltődik be a YOLO modell (torch)
                if args.hatterben or args.szalak:
                    import background
                    background.limit_library_threads(background.current().threads)  # torch / cv2 már betöltve
                # az elemzés (cut_and_analyze_sync) a main.CLIPS_DIR-be ír – ide irányítjuk
                engine.CLIPS_DIR = out_dir
                result = pipeline.run_match_pipeline(args.video, opts, progress=reporter,
                                                     funcs=pipeline.default_funcs(engine), out_dir=out_dir)
        except (KeyboardInterrupt, JobCancelled):
            bar.done()
            print("Megszakítva.")
            return 130
        except Exception as ex:
            bar.done()
            import traceback
            log.write("\n" + traceback.format_exc())
            print(f"Hiba: {type(ex).__name__}: {ex}\nRészletek: {log_path}")
            return 1
    bar.done()
    print(pipeline.summarize(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

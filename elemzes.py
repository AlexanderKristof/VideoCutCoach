"""
A teljes meccs elemzése parancssorból (szerver és MAUI kliens nélkül).

    python elemzes.py MECCS.mp4
    python elemzes.py MECCS.mp4 --felido-perc 40 --kalibracio "0.1,0.4;0.9,0.4;1.0,0.9;0.0,0.9" --kimenet eredmeny
    python elemzes.py MECCS.mp4 --nincs-vagas --gyors 3

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
                   help="4 pályapont a képen, 0..1 relatív koordinátákkal (mint a kliensben); "
                        "nélküle nincs pályakoordináta, hőtérkép és a legtöbb esemény")
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
    p.add_argument("--reszletes", action="store_true",
                   help="az elemző motor saját kiírásai is a képernyőre kerüljenek (alap: naplófájlba)")
    return p


def options_from_args(args):
    opts = {
        "expected_half_min": args.felido_perc,
        "calib_points": args.kalibracio,
        "skip_cut": args.nincs_vagas,
        "team1_attacks_right": not args.csapat1_balra,
        "compilations": args.osszeallitasok,
        "use_person_count": args.letszam,
        "heatmaps": not args.nincs_hoterkep,
        "review_export": not args.nincs_attekinto,
    }
    if args.felidok:
        parts = [x.strip() for x in args.felidok.split(",")]
        if len(parts) != 4:
            raise SystemExit("Hiba: a --felidok négy értéket vár (1. kezdet, 1. vége, 2. kezdet, 2. vége).")
        opts.update(zip(("first_start", "first_end", "second_start", "second_end"), parts))
    return opts


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

    import pipeline
    from jobs import ProgressReporter, JobCancelled

    opts = options_from_args(args)
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
    print(f"Kimenet: {out_dir}" + ("" if args.reszletes else f"  (a motor kiírásai: {log_path})"))

    with open(log_path, "w", encoding="utf-8") as log:
        redirect = contextlib.nullcontext() if args.reszletes else contextlib.redirect_stdout(log)
        try:
            with redirect:
                import main as engine  # itt töltődik be a YOLO modell (torch)
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

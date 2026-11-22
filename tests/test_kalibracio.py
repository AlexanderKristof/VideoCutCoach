"""
A kattintós kalibráló eszköz (tools/kalibracio.py) ablak nélküli részeinek tesztjei, és az
elemzes.py / pipeline kalibrációs fájl-kezelése. Futtatás: python3 tests/test_kalibracio.py
(torch nem kell; grafikus kijelző sem).
"""
import io
import os
import sys
import json
import tempfile
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cv2
import numpy as np

import kalibracio as kb
import pitch_tracker as pt


def _camera_H(w=1920, h=1080):
    """Valószerű Veo-nézet (a felezővonalnál álló kamera, enyhén balra néz): kép -> pálya H."""
    f = 0.5 * w / np.tan(np.radians(30))
    G = pt.camera_ground_homography(-12.0, 14.0, f, w, h, (52.5, 74.0, -7.0))
    H = np.linalg.inv(G)
    return H / H[2, 2], G


def _project(G, xy):
    p = G @ np.array([xy[0], xy[1], 1.0])
    return p[0] / p[2], p[1] / p[2]


def test_landmarks():
    lms = kb.landmarks()
    nums = [l[0] for l in lms]
    assert nums == list(range(1, len(lms) + 1))
    assert len(lms) == 33
    names = [l[1] for l in lms]
    assert len(set(names)) == len(names)
    xy = {(round(l[2], 2), round(l[3], 2)) for l in lms}
    assert len(xy) == len(lms), "két pont egybeesik"
    for must in ((0, 0), (105, 68), (52.5, 34), (52.5, 24.85), (52.5, 43.15), (11, 34), (94, 34),
                 (16.5, 13.84), (88.5, 54.16), (5.5, 24.84), (105, 43.16), (16.5, 26.69), (88.5, 41.31)):
        assert (round(must[0], 2), round(must[1], 2)) in xy, must
    # nem szabványos pálya: a szögletek a megadott méretben, a 16-os mérete fix
    small = {l[1]: (l[2], l[3]) for l in kb.landmarks(100, 64)}
    assert small["jobb alsó szöglet"] == (100.0, 64.0)
    assert abs(small["bal 16-os alsó sarka"][1] - (32 + 20.16)) < 1e-9
    assert small["kezdőpont"] == (50.0, 32.0)
    # legközelebbi pont
    assert kb.nearest_landmark(lms, 10.6, 34.3)[1] == "bal büntetőpont"
    assert kb.nearest_landmark(lms, 30.0, 34.0, max_dist=3.0) is None


def test_parse_time_and_size():
    assert kb.parse_time("10:00") == 600.0
    assert kb.parse_time("1:02:03") == 3723.0
    assert kb.parse_time("75.5") == 75.5
    assert kb.parse_time("0:07,5") == 7.5
    for bad in ("", "a:b", "1:2:3:4", "-5"):
        try:
            kb.parse_time(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert kb.parse_pitch_size("100x64") == (100.0, 64.0)
    assert kb.parse_pitch_size("64*100") == (100.0, 64.0)
    for bad in ("100", "10x5", "abc"):
        try:
            kb.parse_pitch_size(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert kb.format_time(600) == "10:00" and kb.format_time(3723) == "1:02:03"


def test_homography_from_pairs():
    H_true, G = _camera_H()
    names = ("kezdőpont", "kezdőkör × felezővonal (alsó)", "bal 16-os alsó sarka", "bal 16-os felső sarka",
             "felezővonal alsó vége", "kezdőkör bal széle")
    lm = {l[1]: (l[2], l[3]) for l in kb.landmarks()}
    pit = [lm[n] for n in names]
    rng = np.random.default_rng(1)
    img = [np.array(_project(G, p)) + rng.normal(0, 1.0, 2) for p in pit]
    H, errs = kb.homography_from_pairs(img, pit)
    assert H is not None and len(errs) == len(pit)
    assert max(e["m"] for e in errs) < 0.3 and np.median([e["px"] for e in errs]) < 2.0, errs
    # a pálya közepe környékén pontos (a kattintási zajhoz képest)
    for p in ((52.5, 34.0), (40.0, 50.0), (30.0, 30.0)):
        u, v = _project(G, p)
        q = H @ np.array([u, v, 1.0])
        assert np.hypot(q[0] / q[2] - p[0], q[1] / q[2] - p[1]) < 0.5
    # ugyanaz, mint a motor kalibráció-olvasója (parse_calib_points) a kiírt szövegből
    w, h = 1920, 1080
    a = kb.format_args(list(zip([tuple(x) for x in img], pit)), w, h, 600.0)
    H2 = pt.parse_calib_points(a["kalibracio"], w, h, a["kalibracio_pontok"])
    u, v = _project(G, (45.0, 40.0))
    q1, q2 = H @ np.array([u, v, 1.0]), H2 @ np.array([u, v, 1.0])
    assert np.hypot(*(q1[:2] / q1[2] - q2[:2] / q2[2])) < 0.05
    # elfajult: kevés pont, vagy mind egy egyenesen
    assert kb.homography_from_pairs(img[:3], pit[:3]) == (None, None)
    line_pts = [(52.5, 0.0), (52.5, 24.85), (52.5, 34.0), (52.5, 68.0)]
    assert kb.homography_from_pairs([_project(G, p) for p in line_pts], line_pts) == (None, None)
    # OpenCV nélküli tartalék (DLT) ugyanazt adja
    Hd = kb._dlt(np.array(img, float), np.array(pit, float))
    qd = Hd @ np.array([u, v, 1.0])
    assert np.hypot(*(q1[:2] / q1[2] - qd[:2] / qd[2])) < 0.05


def test_format_args():
    pairs = [((960.0, 540.0), (52.5, 34.0)), ((0.0, 1080.0), (0.0, 68.0)), ((1920.0, 0.0), (105.0, 0.0)),
             ((480.0, 270.0), (16.5, 13.84))]
    a = kb.format_args(pairs, 1920, 1080, 600.123)
    assert a["kalibracio"] == "0.5000,0.5000;0.0000,1.0000;1.0000,0.0000;0.2500,0.2500"
    assert a["kalibracio_pontok"] == "52.5,34;0,68;105,0;16.5,13.84"
    assert a["kalibracio_ido"] == 600.12
    assert a["args"] == ('--kalibracio "0.5000,0.5000;0.0000,1.0000;1.0000,0.0000;0.2500,0.2500" '
                         '--kalibracio-pontok "52.5,34;0,68;105,0;16.5,13.84" --kalibracio-ido 600.12')
    # az elemzes.py parancssora ezt elfogadja
    import elemzes
    import shlex
    args = elemzes.build_parser().parse_args(["x.mp4"] + shlex.split(a["args"]))
    opts = elemzes.options_from_args(args)
    assert opts["calib_points"] == a["kalibracio"] and opts["calib_pitch_points"] == a["kalibracio_pontok"]
    assert abs(opts["calib_time_sec"] - 600.12) < 1e-9


def test_json_roundtrip_and_halves():
    d = tempfile.mkdtemp()
    video = os.path.join(d, "meccs.mp4")
    path = kb.json_path_for(video)
    assert path == video + ".kalibracio.json"
    assert kb.load_calibration_json(path) is None
    p1 = [((100.0, 900.0), (0.0, 68.0)), ((1800.0, 900.0), (105.0, 68.0)), ((900.0, 300.0), (52.5, 0.0)),
          ((960.0, 600.0), (52.5, 34.0))]
    errs = [{"px": 0.5, "m": 0.05}] * 4
    e1 = kb.make_half_entry(p1, 1920, 1080, 600.0, errs, ["a", "b", "c", "d"])
    kb.save_calibration_json(path, 1, e1, (100.0, 64.0), video)
    e2 = kb.make_half_entry(p1[::-1], 1920, 1080, 3300.0)
    data = kb.save_calibration_json(path, 2, e2, (100.0, 64.0), video)
    back = kb.load_calibration_json(path)
    assert back == data
    assert back["palya"] == [100.0, 64.0] and back["video"] == "meccs.mp4"
    assert set(back["felidok"]) == {"1", "2"}
    assert back["felidok"]["1"]["pontok"][0]["nev"] == "a" and back["felidok"]["1"]["pontok"][0]["hiba_px"] == 0.5
    opts = kb.calibration_options(back)
    assert opts["calib_points"] == e1["kalibracio"] and opts["calib_time_sec"] == 600.0
    assert opts["calib2_points"] == e2["kalibracio"] and opts["calib2_time_sec"] == 3300.0
    # az 1. félidő újramentése nem törli a 2.-at
    kb.save_calibration_json(path, 1, e1, (105.0, 68.0), video)
    assert set(kb.load_calibration_json(path)["felidok"]) == {"1", "2"}
    # hibás fájl: None
    with open(path, "w", encoding="utf-8") as f:
        f.write("{nem json")
    assert kb.load_calibration_json(path) is None and kb.calibration_options(None) == {}


def test_elemzes_uses_calibration_file():
    import elemzes
    d = tempfile.mkdtemp()
    video = os.path.join(d, "meccs.mp4")
    open(video, "wb").close()
    args = elemzes.build_parser().parse_args([video])
    opts = elemzes.options_from_args(args)
    assert elemzes.apply_calibration_file(args, opts) is None and opts["calib_points"] == ""
    p = [((100.0, 900.0), (0.0, 68.0)), ((1800.0, 900.0), (105.0, 68.0)), ((900.0, 300.0), (52.5, 0.0)),
         ((960.0, 600.0), (52.5, 34.0))]
    kb.save_calibration_json(kb.json_path_for(video), 2, kb.make_half_entry(p, 1920, 1080, 3000.0), video=video)
    info = elemzes.apply_calibration_file(args, opts)
    assert "2." in info and opts["calib2_points"] and opts["calib2_time_sec"] == 3000.0 and opts["calib_points"] == ""
    # --kalibracio megadva: a fájl nem számít
    args2 = elemzes.build_parser().parse_args([video, "--kalibracio", "0,0;1,0;1,1;0,1"])
    opts2 = elemzes.options_from_args(args2)
    assert elemzes.apply_calibration_file(args2, opts2) is None and "calib2_points" not in opts2
    args3 = elemzes.build_parser().parse_args([video, "--nincs-kalibracios-fajl"])
    assert elemzes.apply_calibration_file(args3, elemzes.options_from_args(args3)) is None


def test_pipeline_second_half_calibration():
    """A pipeline a 2. félidőben a saját (calib2_*) kalibrációt adja tovább, az 1.-ben az elsőt."""
    import pipeline
    calls = []

    def analyze(video_path, start, end, calib, config, detect_mode="team", progress=None):
        calls.append((start, calib, config.get("calib_pitch_points"), config.get("calib_time_sec")))
        return None, None, None
    opts = {"calib_points": "A", "calib_pitch_points": "PA", "calib_time_sec": 600.0,
            "calib2_points": "B", "calib2_pitch_points": "PB", "calib2_time_sec": 3300.0,
            "first_start": "0", "first_end": "2700", "second_start": "3000", "second_end": "5700",
            "skip_cut": True, "heatmaps": False, "compilations": "", "review_export": False}
    funcs = {"analyze": analyze, "detect_halves": None, "cut_match": None}
    d = tempfile.mkdtemp()
    video = os.path.join(d, "v.mp4")
    open(video, "wb").close()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.run_match_pipeline(video, opts, progress=None, funcs=funcs, out_dir=d)
    except Exception:
        pass        # a további lépések (összefésülés, klipek) itt nem számítanak
    assert len(calls) >= 2, calls
    assert calls[0][1:] == ("A", "PA", 600.0), calls[0]
    assert calls[1][1:] == ("B", "PB", 3300.0), calls[1]


def test_preview_render_headless():
    """Az előnézet (videókép + cián vonalak + pálya-rajz) ablak nélkül is elkészül."""
    H, G = _camera_H()
    w, h = 1920, 1080
    frame = np.full((h, w, 3), (40, 110, 50), np.uint8)
    pt.draw_pitch_overlay(frame, H, color=(255, 255, 255), alpha=1.0, thickness=3)
    lm = {l[1]: l for l in kb.landmarks()}
    tool = kb.CalibrationTool(frame, 600.0)
    for name in ("kezdőpont", "kezdőkör × felezővonal (alsó)", "bal 16-os alsó sarka", "bal 16-os felső sarka",
                 "kezdőkör bal széle"):
        n, _, x, y = lm[name]
        u, v = _project(G, (x, y))
        # kattintás a nézetben (a nézet = a kép kicsinyítve), majd a rajzon
        vx, vy = tool.img_to_view(u, v)
        tool.click(vx, vy)
        px, py = tool.diag.to_px(x, y)
        tool.click(px + tool.view_w + 3, py)
    assert len(tool.pairs) == 5 and tool.lm_nums[0] == lm["kezdőpont"][0]
    Hs, errs = tool.solve()
    assert Hs is not None and max(e["px"] for e in errs) < 3.0, errs
    img = tool.render()
    assert img.shape[0] >= tool.view_h and img.shape[1] > tool.view_w + 300
    # a nagyítás a kurzor alatti pontot helyben tartja
    u0, v0 = tool.view_to_img(300, 200)
    tool.zoom_at(300, 200, 2.0)
    u1, v1 = tool.view_to_img(300, 200)
    assert abs(u0 - u1) < 1e-6 and abs(v0 - v1) < 1e-6
    tool.undo()
    assert len(tool.pairs) == 4
    tool.reset()
    assert tool.pairs == []


def test_no_display_message():
    """Kijelző nélkül érthető magyar hibaüzenet, nem összeomlás."""
    d = tempfile.mkdtemp()
    video = os.path.join(d, "v.mp4")
    wr = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    for _ in range(10):
        wr.write(np.full((180, 320, 3), 90, np.uint8))
    wr.release()
    old = {k: os.environ.pop(k, None) for k in ("DISPLAY", "WAYLAND_DISPLAY")}
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = kb.main([video, "--ido", "0:00"])
        if sys.platform.startswith("linux"):
            assert rc == 3 and "Hiba" in buf.getvalue(), buf.getvalue()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert kb.main([os.path.join(d, "nincs.mp4")]) == 2
        assert "nem található" in buf.getvalue()
        with contextlib.redirect_stdout(io.StringIO()):
            assert kb.main([video, "--ido", "x:y"]) == 2
            assert kb.main([video, "--elonezet", os.path.join(d, "e.jpg")]) == 2     # nincs mentett kalibráció
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Exception as ex:
            import traceback
            traceback.print_exc()
            failed += 1
            print(f"HIBA  {name}: {ex!r}")
    print(f"{len(tests) - failed}/{len(tests)} teszt sikeres")
    sys.exit(1 if failed else 0)

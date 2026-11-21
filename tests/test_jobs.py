"""
Háttérfeladatok (jobs.py) és a teljes meccs folyamatának (pipeline.py) tesztjei.
Futtatás: python3 tests/test_jobs.py

A main.py-t NEM importáljuk (torch / YOLO kellene hozzá): a folyamat nehéz függvényeit
(félidő-felismerés, vágás, elemzés, klipexport) hamis függvényekkel helyettesítjük.
"""
import os
import ast
import sys
import json
import time
import shutil
import tempfile
import threading
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jobs
from jobs import JobManager, ProgressReporter, JobCancelled
import pipeline


def _tmpdir():
    return tempfile.mkdtemp(prefix="jobs_test_")


def _wait_status(mgr, job_id, statuses, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = mgr.get(job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError(f"A job nem jutott {statuses} állapotba: {mgr.get(job_id)}")


# ------------------------------------------------------------------
# ProgressReporter
# ------------------------------------------------------------------

def test_reporter_monotonic_and_sub():
    seen = []
    r = ProgressReporter(on_update=lambda p, ph: seen.append((p, ph)))
    r.report(10, "a")
    r.report(5, "b")          # visszafelé nem megy
    assert r.pct == 10 and r.phase == "b"
    r.report(150)             # 100-ra vágva
    assert r.pct == 100
    r2 = ProgressReporter()
    s = r2.sub(20, 60, "Elemzés")
    s(50, "100/200 képkocka")
    assert abs(r2.pct - 40) < 1e-9 and r2.phase == "Elemzés: 100/200 képkocka"
    ss = s.sub(50, 100, "rész")
    ss(50)
    assert abs(r2.pct - 50) < 1e-9 and r2.phase == "Elemzés: rész", (r2.pct, r2.phase)


def test_reporter_cancel():
    r = ProgressReporter()
    r.report(1)
    r.cancel()
    try:
        r.report(2)
    except JobCancelled:
        pass
    else:
        raise AssertionError("JobCancelled várt")
    try:
        r.sub(0, 50).check_cancel()
    except JobCancelled:
        return
    raise AssertionError("A rész-jelzőnek is meg kell szakítania")


# ------------------------------------------------------------------
# JobManager
# ------------------------------------------------------------------

def test_job_progress_and_result():
    d = _tmpdir()
    try:
        mgr = JobManager(d)

        def work(progress, n):
            for i in range(n):
                progress((i + 1) * 100.0 / n, f"{i + 1}/{n}")
            return {"status": "success", "osszeg": n}

        jid = mgr.submit("teszt", work, 5, params={"n": 5})
        job = mgr.wait(jid, 5)
        assert job["status"] == "done", job
        assert job["progress"] == 100.0 and job["phase"] == "Kész"
        assert job["result"] == {"status": "success", "osszeg": 5}
        assert job["started_at"] and job["finished_at"] and job["elapsed_sec"] is not None
        with open(os.path.join(d, f"{jid}.json"), encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["status"] == "done" and saved["result"]["osszeg"] == 5 and saved["params"] == {"n": 5}
        assert [j["id"] for j in mgr.list()] == [jid] and "result" not in mgr.list()[0]
        mgr.shutdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_job_cancel_running_and_queued():
    d = _tmpdir()
    try:
        mgr = JobManager(d, max_workers=1)
        started = threading.Event()

        def long_work(progress):
            started.set()
            while True:
                progress(1, "dolgozik")
                time.sleep(0.005)

        def quick(progress):
            return {"status": "success"}

        j1 = mgr.submit("hosszu", long_work)
        j2 = mgr.submit("varakozo", quick)
        assert started.wait(2)
        assert mgr.get(j2)["status"] == "queued"
        # sorban álló job: azonnal megszakad
        assert mgr.cancel(j2)["status"] == "cancelled"
        # futó job: jelzés, majd a következő report()-nál megáll
        after = mgr.cancel(j1)
        assert after["cancel_requested"] is True
        job = _wait_status(mgr, j1, ("cancelled",))
        assert job["phase"] == "Megszakítva" and job["finished_at"]
        assert mgr.cancel("nincs-ilyen") is None
        # befejezett job megszakítása nem változtat rajta
        assert mgr.cancel(j1)["status"] == "cancelled"
        mgr.shutdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_job_error_capture():
    d = _tmpdir()
    try:
        mgr = JobManager(d)

        def broken(progress):
            progress(30, "félúton")
            raise RuntimeError("elromlott valami")

        def dict_error(progress):
            return {"status": "error", "message": "Az eseményfájl nem található."}

        jid = mgr.submit("hibas", broken)
        job = mgr.wait(jid, 5)
        assert job["status"] == "error" and "elromlott valami" in job["error"], job
        assert job["progress"] == 30
        with open(job["log_path"], encoding="utf-8") as f:
            log = f.read()
        assert "Traceback" in log and "RuntimeError" in log

        jid2 = mgr.submit("regi", dict_error)
        job2 = mgr.wait(jid2, 5)
        assert job2["status"] == "error" and job2["error"] == "Az eseményfájl nem található."
        mgr.shutdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_persistence_and_restart_marking():
    d = _tmpdir()
    try:
        mgr = JobManager(d, max_workers=1)
        release = threading.Event()
        running = threading.Event()

        def blocking(progress):
            running.set()
            release.wait(5)
            return {"status": "success"}

        done_id = mgr.submit("kesz", lambda p: {"status": "success", "x": 1})
        mgr.wait(done_id, 5)
        run_id = mgr.submit("fut", blocking)
        queued_id = mgr.submit("sorban", lambda p: {"status": "success"})
        assert running.wait(2)

        # „újraindítás”: új kezelő ugyanazzal a mappával, miközben a régi még fut
        mgr2 = JobManager(d)
        assert mgr2.get(done_id)["status"] == "done" and mgr2.get(done_id)["result"]["x"] == 1
        for jid in (run_id, queued_id):
            j = mgr2.get(jid)
            assert j["status"] == "megszakadt", j
            assert j["finished_at"] and "újraindult" in j["error"]
        with open(os.path.join(d, f"{run_id}.json"), encoding="utf-8") as f:
            assert json.load(f)["status"] == "megszakadt"
        release.set()
        mgr.shutdown()
        mgr2.shutdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------
# Pipeline hamis függvényekkel
# ------------------------------------------------------------------

H1 = {"start_sec": 100.0, "end_sec": 2800.0}
H2 = {"start_sec": 3700.0, "end_sec": 6500.0}


class Fakes:
    """A nehéz függvények hamis változatai; rögzítik a hívásokat."""

    def __init__(self, out_dir, second_half=True, heatmaps=None, cancel_after_analyze=None):
        self.out_dir = out_dir
        self.calls = []
        self.second_half = second_half
        self.heatmaps_fn = heatmaps
        self.cancel_after_analyze = cancel_after_analyze

    def funcs(self):
        return {
            "detect_halves": self.detect_halves, "cut_match": self.cut_match, "analyze": self.analyze,
            "heatmaps": self.heatmaps, "export_clips": self.export_clips,
            "parse_time": lambda v: float(v) if str(v).strip() else None,
            "clips_dir": self.out_dir,
        }

    def detect_halves(self, video_path, expected_half_min, use_person_count, progress_cb=None):
        self.calls.append(("detect", expected_half_min))
        progress_cb("hang", 100.0)
        for p in (20, 60, 99):
            progress_cb("kep", p)
        return {"first_half": dict(H1), "second_half": dict(H2) if self.second_half else None,
                "confidence": 0.8, "duration_sec": 7000.0, "notes": ["teszt megjegyzés"]}

    def cut_match(self, video_path, halves, out_dir, padding_sec=5.0, duration_sec=None):
        self.calls.append(("cut", padding_sec, duration_sec))
        return {"first_half": os.path.join(out_dir, "half1_x.mp4"),
                "second_half": os.path.join(out_dir, "half2_x.mp4") if "second_half" in halves else None,
                "full_match": os.path.join(out_dir, "match_x.mp4")}

    def analyze(self, video_path, start, end, calib, config, detect_mode="team", progress=None):
        n = len([c for c in self.calls if c[0] == "analyze"]) + 1
        self.calls.append(("analyze", start, end, calib, config["team1_attacks_right"], config["roi_top_pct"]))
        for i in range(0, 101, 10):
            progress(i, f"{i}/100 képkocka")
            if self.cancel_after_analyze is not None:
                self.cancel_after_analyze()
        events = [
            {"type": "shot", "time": start + 60.0, "team": "team1"},
            {"type": "goal", "time": start + 61.0, "team": "team1"},
            {"type": "gk_save", "time": start + 300.0, "team": "team2", "start_sec": start + 296.0,
             "end_sec": start + 303.0},
            {"type": "gk_goal_kick", "time": start + 400.0, "team": "team1"},
        ]
        pts = [[30.0, 20.0, start + i * 0.2] for i in range(200)]
        payload = {"events": events, "match_stats": {"half": n}, "review_url": f"/clips/review_{n}/index.html",
                   "source": {"video_path": video_path, "start_sec": start, "end_sec": end},
                   "trajectories": {"hz": 5.0, "tracks": {"3": {"team": 1, "keeper": False, "points": pts}},
                                    "ball_hz": 10.0, "ball": [[50.0, 30.0, start + i * 0.1] for i in range(20)]},
                   "team_colors": {"team1": [200, 30, 30], "team2": [240, 240, 240], "other": None}}
        path = os.path.join(self.out_dir, f"events_fake{n}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return f"/clips/raw_fake{n}.mp4", os.path.join(self.out_dir, f"raw_fake{n}.mp4"), path

    def heatmaps(self, events_path, out_dir, **kw):
        self.calls.append(("heatmap", os.path.basename(events_path), kw))
        if self.heatmaps_fn:
            return self.heatmaps_fn(events_path, out_dir, **kw)
        return {"status": "success", "team_heatmaps": {"team1": "/clips/h.png"}, "player_heatmaps": []}

    def export_clips(self, video_path, items, out_dir, pre_sec=5.0, post_sec=4.0, merge=True, accurate=True,
                     title=None, progress_cb=None):
        self.calls.append(("clips", title, items, pre_sec, accurate))
        for i in range(len(items)):
            if progress_cb:
                progress_cb(i + 1, len(items))
        return {"status": "success", "clips": [{"url": "/clips/c.mp4"}] * len(items),
                "merged_url": f"/clips/{title}.mp4", "merged_local_path": os.path.join(out_dir, f"{title}.mp4")}


def _video(d):
    p = os.path.join(d, "meccs.mp4")
    with open(p, "wb") as f:
        f.write(b"\0")
    return p


def test_pipeline_full_orchestration():
    d = _tmpdir()
    try:
        fakes = Fakes(d)
        seen = []
        reporter = ProgressReporter(on_update=lambda p, ph: seen.append((p, ph)))
        res = pipeline.run_match_pipeline(_video(d), {"calib_points": "0,0;1,0;1,1;0,1", "roi_top_pct": "0.3",
                                                      "expected_half_min": "40"},
                                          progress=reporter, funcs=fakes.funcs())
        # sorrend és paraméterek
        kinds = [c[0] for c in fakes.calls]
        assert kinds[:4] == ["detect", "cut", "analyze", "analyze"], kinds
        assert fakes.calls[0] == ("detect", 40.0)
        a1, a2 = fakes.calls[2], fakes.calls[3]
        assert a1[1:3] == (100.0, 2800.0) and a2[1:3] == (3700.0, 6500.0)
        assert a1[3] == "0,0;1,0;1,1;0,1" and a1[5] == 0.3
        assert a1[4] is True and a2[4] is False, "a 2. félidőben a támadási irány megfordul"

        # haladás: monoton, 100-ra ér, a fázisszöveg magyar
        pcts = [p for p, _ in seen]
        assert all(b >= a for a, b in zip(pcts, pcts[1:])), pcts
        assert pcts[-1] == 100.0 and seen[-1][1] == "Kész"
        assert any("1. félidő elemzése: 50/100 képkocka" == ph for _, ph in seen)
        assert any(ph.startswith("Félidők felismerése") for _, ph in seen)
        # a félidő-felismerés a súlyán belül marad (10%)
        det = [p for p, ph in seen if ph.startswith("Félidők felismerése")]
        assert max(det) <= 10.0 + 1e-6

        # eredmény alakja
        assert res["status"] == "success"
        assert set(res["analysis"]) == {"first_half", "second_half"}
        assert res["analysis"]["second_half"]["team1_attacks_right"] is False
        assert res["cut"]["full_match_url"] == "/clips/match_x.mp4"
        assert res["match_events_url"].startswith("/clips/match_events_")
        assert res["event_count"] == 8 and res["event_counts"]["gk_save"] == 2
        assert res["detection"]["confidence"] == 0.8 and "teszt megjegyzés" in res["notes"]
        assert set(res["heatmaps"]) == {"first_half", "second_half", "combined"}
        assert set(res["compilations"]) == {"goalkeeper_team1", "goalkeeper_team2", "highlights"}
        json.dumps(res)

        # összesített hőtérkép: a 2. félidő kezdeténél tükröz, csak csapat-hőtérkép
        comb = [c for c in fakes.calls if c[0] == "heatmap" and c[1].startswith("match_events_")][0]
        assert comb[2]["flip_at_sec"] == 3700.0 and comb[2]["min_player_sec"] == float("inf")

        # klipek: kapusonként a kapus események, összefoglaló = gól + lövés
        clips = {c[1]: c[2] for c in fakes.calls if c[0] == "clips"}
        assert len(clips["Vendég Kapus"]) == 2 and clips["Vendég Kapus"][0] == {"start_sec": 396.0, "end_sec": 403.0}
        assert len(clips["Hazai Kapus"]) == 2 and clips["Hazai Kapus"][0] == {"time": 500.0}
        assert len(clips["Osszefoglalo"]) == 4

        # összefésült események: félidő, eltolások
        with open(res["match_events_path"], encoding="utf-8") as f:
            merged = json.load(f)
        evs = merged["events"]
        assert [e["half"] for e in evs] == [1, 1, 1, 1, 2, 2, 2, 2]
        first_shot, second_shot = evs[0], evs[4]
        assert first_shot["time"] == 160.0 and first_shot["time_in_half"] == 60.0
        assert first_shot["time_in_match"] == 60.0 and first_shot["match_clock"] == "1:00"
        assert second_shot["time"] == 3760.0 and second_shot["time_in_half"] == 60.0
        assert second_shot["time_in_match"] == 2700.0 + 60.0      # az 1. félidő hossza + 60 mp
        assert second_shot["match_clock"] == "41:00"              # 40 perces félidő: a 2. 40:00-tól
        # a vágott meccsvideóban: 5 mp ráhagyás; a 2. félidő a 2710 mp-es első darab után
        assert first_shot["full_match_sec"] == 65.0
        assert second_shot["full_match_sec"] == 2710.0 + 5.0 + 60.0
        assert evs[0]["half_label"] == "1. félidő" and evs[4]["half_label"] == "2. félidő"
        assert merged["match_stats"] == {"first_half": {"half": 1}, "second_half": {"half": 2}}
        tracks = merged["trajectories"]["tracks"]
        assert set(tracks) == {"3", str(3 + pipeline.SECOND_HALF_TRACK_OFFSET)}
        assert tracks[str(3 + pipeline.SECOND_HALF_TRACK_OFFSET)]["original_track_id"] == 3
        # labda: a két félidő összefűzve, idő szerint; mezszínek és félidőhossz a táblának
        ball = merged["trajectories"]["ball"]
        assert len(ball) == 40 and ball[0][2] == 100.0 and ball[20][2] == 3700.0
        assert all(ball[i][2] <= ball[i + 1][2] for i in range(len(ball) - 1))
        assert merged["team_colors"]["team1"] == [200, 30, 30]
        assert merged["expected_half_min"] == 40.0
        summary = pipeline.summarize(res)
        assert "A meccs elemzése elkészült" in summary and "Összesített események (8 db)" in summary
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_pipeline_skip_cut_manual_halves_no_clips():
    d = _tmpdir()
    try:
        fakes = Fakes(d)
        seen = []
        res = pipeline.run_match_pipeline(
            _video(d), {"skip_cut": "true", "first_start": "100", "first_end": "2800", "second_start": "3700",
                        "second_end": "6500", "compilations": "", "heatmaps": "false",
                        "team1_attacks_right": "false"},
            progress=lambda p, ph: seen.append((p, ph)), funcs=fakes.funcs())
        kinds = [c[0] for c in fakes.calls]
        assert kinds == ["analyze", "analyze"], kinds     # se felismerés, se vágás, se hőtérkép, se klip
        assert fakes.calls[0][4] is False and fakes.calls[1][4] is True
        assert res["cut"] is None and res["detection"]["manual"] is True
        assert res["heatmaps"] == {} and res["compilations"] == {}
        assert "full_match_sec" not in json.load(open(res["match_events_path"], encoding="utf-8"))["events"][0]
        pcts = [p for p, _ in seen if p is not None]
        assert all(b >= a - 1e-9 for a, b in zip(pcts, pcts[1:])) and pcts[-1] == 100.0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_pipeline_single_half_and_bad_options():
    d = _tmpdir()
    try:
        fakes = Fakes(d, second_half=False)
        res = pipeline.run_match_pipeline(_video(d), {"compilations": "highlights"}, funcs=fakes.funcs())
        assert set(res["analysis"]) == {"first_half"}
        assert "combined" not in res["heatmaps"] and set(res["compilations"]) == {"highlights"}
        for bad in ({"compilations": "kapus"}, {"expected_half_min": "sok"}, {"skip_cut": "talan"}):
            try:
                pipeline.run_match_pipeline(_video(d), bad, funcs=fakes.funcs())
            except ValueError:
                continue
            raise AssertionError(f"ValueError várt: {bad}")
        try:
            pipeline.run_match_pipeline(os.path.join(d, "nincs.mp4"), {}, funcs=fakes.funcs())
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("FileNotFoundError várt")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_pipeline_cancel_as_job():
    d = _tmpdir()
    try:
        mgr = JobManager(os.path.join(d, "jobs"))
        holder = {}
        fakes = Fakes(d, cancel_after_analyze=lambda: mgr.cancel(holder["id"]))
        holder["id"] = None
        gate = threading.Event()

        def work(progress):
            gate.wait(2)
            return pipeline.run_match_pipeline(_video(d), {}, progress=progress, funcs=fakes.funcs())

        holder["id"] = mgr.submit("analyze-match", work)
        gate.set()
        job = mgr.wait(holder["id"], 5)
        assert job["status"] == "cancelled", job
        assert [c[0] for c in fakes.calls].count("analyze") == 1
        assert not [c for c in fakes.calls if c[0] in ("heatmap", "clips")]
        mgr.shutdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_pipeline_with_real_heatmap():
    """Az összesített hőtérkép a valódi heatmap modullal (OpenCV kell hozzá)."""
    try:
        import heatmap
    except ImportError as ex:
        print(f"   (kihagyva: {ex})")
        return
    d = _tmpdir()
    try:
        fakes = Fakes(d, heatmaps=heatmap.generate_heatmaps_from_file)
        res = pipeline.run_match_pipeline(_video(d), {"compilations": "", "skip_cut": True}, funcs=fakes.funcs())
        comb = res["heatmaps"]["combined"]
        assert comb["status"] == "success", comb
        assert comb["team_heatmaps"]["team1"] and comb["player_heatmaps"] == []
        assert len(comb["player_stats"]) == 2
        assert res["heatmaps"]["first_half"]["player_heatmaps"], "félidőnként játékos-hőtérkép is készül"
        assert os.path.exists(os.path.join(d, os.path.basename(comb["team_heatmaps"]["team1"])))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------
# Egyezés a main.py-val (AST alapján – a main.py-t nem importáljuk)
# ------------------------------------------------------------------

def _main_ast():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as f:
        return ast.parse(f.read())


def test_defaults_match_cut_raw_endpoint():
    tree = _main_ast()
    fn = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "cut_raw_endpoint"][0]
    args = fn.args.args
    defaults = fn.args.defaults
    found = {}
    for a, dflt in zip(args[len(args) - len(defaults):], defaults):
        # Form(<alap>)
        found[a.arg] = ast.literal_eval(dflt.args[0]) if dflt.args else None
    for key in ("video_path", "start_sec", "end_sec", "dev_mode"):
        found.pop(key, None)
    assert found == pipeline.ANALYSIS_DEFAULTS, {k: (found.get(k), pipeline.ANALYSIS_DEFAULTS.get(k))
                                                 for k in set(found) | set(pipeline.ANALYSIS_DEFAULTS)
                                                 if found.get(k) != pipeline.ANALYSIS_DEFAULTS.get(k)}
    cfg = pipeline.build_analysis_config({})
    assert cfg["grass_lower"] == [30, 40, 40] and cfg["team1_upper"] == [179, 255, 255]
    assert cfg["team1_attacks_right"] is True and cfg["roi_top_pct"] == 0.28


def test_gk_constants_match_main():
    tree = _main_ast()
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and \
                n.targets[0].id in ("GK_TIMELINE_TYPES", "GK_TEAM_NAMES"):
            consts[n.targets[0].id] = ast.literal_eval(n.value)
    assert consts["GK_TIMELINE_TYPES"] == pipeline.GK_TIMELINE_TYPES
    assert consts["GK_TEAM_NAMES"] == pipeline.GK_TEAM_NAMES


def test_cli_help_without_torch():
    code = ("import sys, runpy; sys.argv=['elemzes.py','--help']\n"
            "try:\n    runpy.run_path('elemzes.py', run_name='__main__')\n"
            "except SystemExit: pass\n"
            "assert 'torch' not in sys.modules and 'main' not in sys.modules and 'cv2' not in sys.modules\n")
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "--felido-perc" in res.stdout and "félidő" in res.stdout


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Exception as ex:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"HIBA  {name}: {ex!r}")
    print(f"{len(tests) - failed}/{len(tests)} teszt sikeres")
    sys.exit(1 if failed else 0)

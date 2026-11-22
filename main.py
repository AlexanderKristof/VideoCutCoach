import contextlib
import cv2
import os
import uuid
import json
import re
import asyncio
import subprocess
import time
import math
import threading
import numpy as np
import traceback
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Form, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import app_paths  # adat- és erőforrás-útvonalak (fejlesztői módban a régi relatív utak)
import background  # háttér-mód és megszakítható ffmpeg-futtatás
import model_manager  # súlyfájlok keresése / letöltése a models mappába

from database import get_db, engine
from models import Base, User, Match
from tactical_engine import TacticalEngine
from events import PositioningMonitor  # kapus helyezkedési hiba (tartós eltérés)
from pitch_filters import (build_grass_mask, foot_on_grass, CameraMotion,
                           StationaryFilter, extract_dominant_color,
                           PitchBoundary, PitchTrackSanity, KeeperGate, pitch_h_trusted)  # PÁLYAHATÁR
from pitch_filters import GrassModel, GrassMaskCache, grass_mode_from_config  # ADAPTÍV FŰ
from video_quality import VideoQualityMonitor, annotate_event_reliability  # VIDEÓMINŐSÉG
from model_config import (load_model_config, resolve_weights, build_role_class_map,
                          describe_role_map, should_detect, reset_tracker, ROLES, auto_flag)
from review_export import ReviewExporter
from match_segmenter import (detect_match_halves, cut_match_halves,
                             parse_time_value, format_hms)
from auth import hash_password, verify_password, create_access_token, get_current_user
from heatmap import TrackMetaCollector, compact_trajectories, generate_heatmaps_from_file
from clip_export import export_clips, DEFAULT_PRE_SEC, DEFAULT_POST_SEC
from team_classifier import TeamClassifier, auto_teams_enabled
from pitch_tracker import PitchTracker, parse_calib_points, draw_pitch_overlay as render_pitch_overlay
from ball_tracker import BallTracker, draw_ball_state  # labdakövetés (Kálmán + jelölt-pontozás)
from ball_crops import BallCropSearch  # LABDA-KIVÁGÁS: natív felbontású labda-felismerés
from stage_timer import StageTimer  # PROFILOZÁS: szakaszonkénti időmérés
from video_writer import AnnotatedVideoWriter, FramePrefetcher, choose_h264_encoder, encoder_args as video_writer_encoder_args  # KIMENETI KÓDOLÁS: hardveres H.264, író szál
from pitch_filters import BoundaryCrowdFilter  # NÉZŐ A TÚLSÓ PÁLYASZÉLEN (kalibráció nélkül)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Tactical Analyzer Pro Engine")
CLIPS_DIR = app_paths.clips_dir()  # fejlesztői módban "clips" (mint eddig), alkalmazás-módban az adatmappában
os.makedirs(CLIPS_DIR, exist_ok=True)
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
executor = ThreadPoolExecutor(max_workers=4)


# --- LUSTA MODELLBETÖLTÉS ---
# A torch / ultralytics importja és a YOLO modell betöltése (esetleg a súly letöltése) lassú: a szerver
# indulását nem várakoztatjuk vele (a GET /health azonnal válaszol). A modell az első elemzéskor
# (ensure_model) vagy a POST /models/load hívásra töltődik be, szálbiztosan, egyszer.
def YOLO(*args, **kwargs):
    from ultralytics import YOLO as _YOLO  # lusta import (torch-ot is betölt)
    if app_paths.app_mode():
        try:  # telepített alkalmazás: ne küldjön használati statisztikát az ultralytics
            from ultralytics.utils import SETTINGS
            if SETTINGS.get("sync"):
                SETTINGS.update({"sync": False})
        except Exception:
            pass
    return _YOLO(*args, **kwargs)


device = None          # 'cuda' / 'cpu' – detect_device() tölti ki
model = None           # a megosztott YOLO modell – ensure_model() tölti be
model_weights = None   # a betöltött súlyfájl útvonala
model_cfg = load_model_config()
role_classes = {r: set() for r in ROLES}
model_state = {"loading": False, "error": None, "loaded_at": None}
# --- DETEKTOR-HÁTTÉR: "ultralytics" (model.track + beépített ByteTrack, fejlesztéshez) vagy saját
# háttér ("onnx" / "ultralytics_predict": detector.detect() + detection/tracker.py követő). A saját
# háttérnél nincs ultralytics / torch import, és a labda nyers jelöltjei közvetlenül jönnek (nem kell
# a lenti callback-trükk). Az ensure_model() hozza létre. Lásd docs/MODELL_BEALLITAS.md 8. pont.
DETECTOR_BACKEND = model_cfg.get("backend", "ultralytics")
OWN_BACKEND = None     # OwnTrackerBackend, ha nem az ultralytics model.track() fut
model_names = {}       # a betöltött modell osztálynevei (mindkét háttérnél)
# --- /DETEKTOR-HÁTTÉR ---
_model_init_lock = threading.Lock()
DETECT_EVERY_N = model_cfg["detect_every_n"]
BALL_CONF = model_cfg["ball_conf"]
IMGSZ = int(model_cfg.get("imgsz") or 640)   # SEBESSÉG: a fő felismerés bemeneti mérete


def use_half():
    """SEBESSÉG: félpontosság (FP16) – csak CUDA-n (a "half" beállítás: auto / true / false)."""
    return device == "cuda" and auto_flag(model_cfg.get("half", "auto"), device)
# A megosztott modellt (és a benne tárolt ByteTrack állapotot) egyszerre csak egy elemzés
# használhatja. Kompromisszum: a párhuzamos elemzések sorban várnak egymásra, cserébe nem kell
# elemzésenként külön modellt betölteni (a yolov8x több száz MB GPU memória és lassú betöltés),
# és a követő azonosítói nem keverednek két klip között.
model_lock = threading.Lock()
if DETECT_EVERY_N > 1:
    print(f"   Gyors mód: felismerés minden {DETECT_EVERY_N}. képkockán")


def detect_device():
    """'cuda', ha van használható GPU, különben 'cpu' (az első híváskor importálja a torch-ot)."""
    global device
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:  # DETEKTOR-HÁTTÉR: ONNX háttérnél a torch nem szükséges
            device = 'cpu'
    return device

# --- NYERS LABDA-DETEKTÁLÁSOK (a követő előtti állapot, második futtatás nélkül) ---
# A model.track() eredményéből a ByteTrack kihagyja a meg nem erősített dobozokat – a gyors, ritkán
# detektált labdáét is (a szintetikus mérésen a labda-detektálások ~70%-át). Ultralytics-feltételezés
# (8.x, ultralytics/engine/model.py + trackers/track.py, lásd docs/MODELL_BEALLITAS.md):
#   * a követést a register_tracker() által az ELSŐ track() hívásnál regisztrált
#     'on_predict_postprocess_end' callback végzi: ez írja felül a predictor.results-ot a követett
#     dobozokkal;
#   * a callbackek a model.callbacks listáiban, regisztrálási sorrendben futnak, és a predictor
#     ugyanazt a dict-et kapja (_callbacks=self.callbacks).
# Ezért a saját callbackünket az első track() ELŐTT (itt, a modell betöltése után) regisztráljuk:
# így még a nyers predictor.results-ot látja, és kimásolja belőle a labda-osztályú dobozokat.
# Ha az API nincs meg (vagy a callback nem fut le), a követett dobozokra esünk vissza.
_raw_ball_buf = {"boxes": None, "after_tracker": False}


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _stash_raw_ball_boxes(predictor):
    """on_predict_postprocess_end callback: a nyers labda-dobozok másolata [(xyxy, conf), ...]."""
    try:
        out = []
        for r in getattr(predictor, "results", None) or []:
            b = getattr(r, "boxes", None)
            if b is None or len(b) == 0:
                continue
            if getattr(b, "id", None) is not None:
                _raw_ball_buf["after_tracker"] = True   # a követő már lefutott előttünk -> nem nyers
            xyxy, conf, cls = _to_numpy(b.xyxy), _to_numpy(b.conf), _to_numpy(b.cls)
            for k in range(len(cls)):
                if int(cls[k]) in role_classes["ball"]:
                    out.append((tuple(float(v) for v in xyxy[k][:4]), float(conf[k])))
        _raw_ball_buf["boxes"] = out
    except Exception as ex:  # a követést semmiképp ne törjük el
        print(f"⚠️ Nyers labda-callback hiba: {ex}")
        _raw_ball_buf["boxes"] = None


RAW_BALL_HOOK = False  # az ensure_model() állítja be (a callback a modell betöltésekor regisztrálódik)
BALL_EXTRA_PASS = bool(model_cfg.get("ball_extra_pass", False))
_ball_pass_model = None


def _extra_ball_pass(frame, conf):
    """
    Tartalék (ball_extra_pass): külön labda-felismerés, ha a nyers callback nem érhető el. KÜLÖN
    modellpéldánnyal, mert ugyanazon a modellen a predict() a track() után a követő callbackjét
    is lefuttatná (ugyanaz a predictor), és összezavarná a játékos-azonosítókat. Drága: második
    futtatás + második modell a memóriában – ezért alapból KI.
    """
    global _ball_pass_model
    if _ball_pass_model is None:
        _ball_pass_model = YOLO(model_weights)
    res = _ball_pass_model.predict(frame, classes=sorted(role_classes["ball"]), conf=conf,
                                   device=device, verbose=False, imgsz=IMGSZ, half=use_half())
    out = []
    for r in res:
        for b in r.boxes:
            if int(b.cls[0]) in role_classes["ball"]:
                out.append((tuple(float(v) for v in b.xyxy[0][:4]), float(b.conf[0])))
    return out


def _ball_crop_detect(crops, conf, crop_size):
    """
    LABDA-KIVÁGÁS: labda-felismerés natív felbontású kivágásokon, egy (batch) hívásban.
    Visszatér: kivágásonként [(xyxy a kivágásban, conf), ...].
    * saját háttér (onnx / ultralytics_predict): a detektor detect_batch()-e – követés nélkül;
    * ultralytics model.track(): KÜLÖN modellpéldány (mint a _extra_ball_pass), mert ugyanazon a
      modellen a predict() a követő callbackjét is futtatná (összezavarná a ByteTrack-et).
    """
    ball_ids = role_classes["ball"]
    if not ball_ids or not crops:
        return [[] for _ in crops]
    if OWN_BACKEND is not None:
        det = getattr(OWN_BACKEND, "detector", None)
        if det is None or not hasattr(det, "detect_batch"):
            return [[] for _ in crops]   # TODO: a háttér nem támogat tetszőleges képet
        out = []
        for d in det.detect_batch(crops):
            boxes = []
            for k in range(len(d)):
                if int(d.cls[k]) in ball_ids and float(d.conf[k]) >= conf:
                    boxes.append((tuple(float(v) for v in d.xyxy[k][:4]), float(d.conf[k])))
            out.append(boxes)
        return out
    global _ball_pass_model
    if _ball_pass_model is None:
        _ball_pass_model = YOLO(model_weights)
    res = _ball_pass_model.predict(list(crops), classes=sorted(ball_ids), conf=conf, device=device,
                                   verbose=False, imgsz=int(crop_size), half=use_half())
    out = []
    for r in res:
        boxes = []
        b = getattr(r, "boxes", None)
        if b is not None and len(b):
            xyxy, cf, cls = _to_numpy(b.xyxy), _to_numpy(b.conf), _to_numpy(b.cls)
            for k in range(len(cls)):
                if int(cls[k]) in ball_ids:
                    boxes.append((tuple(float(v) for v in xyxy[k][:4]), float(cf[k])))
        out.append(boxes)
    return out
# --- /NYERS LABDA-DETEKTÁLÁSOK ---


def ensure_model(progress=None):
    """
    A megosztott YOLO modell betöltése az első használatkor (szálbiztos, egyszer fut le).
    A súlyfájlt a model_manager keresi meg (models mappa ...); alkalmazás-módban a hiányzó hivatalos
    súlyt előbb letölti a models mappába (progress: jobs.ProgressReporter-szerű haladásjelző).
    """
    global model, model_weights, role_classes, RAW_BALL_HOOK, OWN_BACKEND, model_names, device
    if model is not None:
        return model
    with _model_init_lock:
        if model is not None:
            return model
        model_state["loading"] = True
        model_state["error"] = None
        try:
            if DETECTOR_BACKEND != "ultralytics":
                # --- DETEKTOR-HÁTTÉR: saját detektor + követő (ultralytics / torch import nélkül) ---
                from detection.backend import create_backend
                cfg = dict(model_cfg)
                if cfg.get("onnx_path"):   # a models mappában / az erőforrások között is keressük
                    cfg["onnx_path"] = model_manager.find_weights(cfg["onnx_path"]) or cfg["onnx_path"]
                if progress is not None:
                    progress(None, f"Detektor betöltése: {os.path.basename(str(cfg.get('onnx_path') or DETECTOR_BACKEND))}")
                dev = detect_device() if DETECTOR_BACKEND == "ultralytics_predict" else None
                backend, roles = create_backend(cfg, dev or "cpu", build_role_class_map,
                                                base_dir=os.path.dirname(os.path.abspath(__file__)))
                if not roles["player"] and not roles["goalkeeper"]:
                    print("⚠️ A modellben nincs játékos osztály (player / person) – állítsd be a class_names kulcsot!")
                if dev is None:   # ONNX: a "device" a végrehajtót jelzi (cuda / dml / cpu)
                    provs = getattr(backend.detector, "providers", []) or []
                    device = ("cuda" if "CUDAExecutionProvider" in provs else
                              "dml" if "DmlExecutionProvider" in provs else "cpu")
                role_classes = roles
                OWN_BACKEND = backend
                model_names = backend.names
                model_weights = backend.describe()
                RAW_BALL_HOOK = False
                model = backend    # a /health "model_loaded" mezője ezt nézi; a track() a saját háttéren fut
                model_state["loaded_at"] = datetime.now().isoformat(timespec="seconds")
                print(f"🔥 Tactical AI Motor aktív: {device.upper()}, detektor: {model_weights}")
                print(f"   Osztályok: {describe_role_map(role_classes, model_names)}")
                return model
                # --- /DETEKTOR-HÁTTÉR ---
            dev = detect_device()
            spec = resolve_weights(model_cfg, dev)
            weights = model_manager.resolve_for_load(spec, progress=progress)
            if model_cfg.get("weights") and not os.path.exists(weights):
                print(f"⚠️ A beállított súlyfájl nem található: {weights} (lásd docs/MODELL_BEALLITAS.md)")
            if progress is not None:
                progress(None, f"Modell betöltése: {os.path.basename(str(weights))}")
            m = YOLO(weights)
            # Szerepkör -> osztály id-k a modell osztálynevei alapján (COCO: person / sports ball,
            # focis modell: player / goalkeeper / referee / ball)
            roles = build_role_class_map(m.names, model_cfg.get("class_names"))
            if not roles["player"] and not roles["goalkeeper"]:
                print("⚠️ A modellben nincs játékos osztály (player / person) – állítsd be a class_names kulcsot!")
            role_classes = roles
            # A nyers labda-callbacket az első track() ELŐTT kell regisztrálni (lásd fent)
            hook = False
            if roles["ball"] and callable(getattr(m, "add_callback", None)):
                try:
                    m.add_callback("on_predict_postprocess_end", _stash_raw_ball_boxes)
                    hook = True
                except Exception as ex:
                    print(f"⚠️ A nyers labda-callback nem regisztrálható: {ex} – a követett labda-dobozokat használjuk")
            RAW_BALL_HOOK = hook
            model_weights = weights
            model_names = m.names
            model = m
            model_state["loaded_at"] = datetime.now().isoformat(timespec="seconds")
            print(f"🔥 Tactical AI Motor aktív: {dev.upper()}, modell: {model_weights}")
            print(f"   Osztályok: {describe_role_map(role_classes, model.names)}")
        except BaseException as ex:
            model_state["error"] = f"{type(ex).__name__}: {ex}"
            raise
        finally:
            model_state["loading"] = False
    return model
# --- /LUSTA MODELLBETÖLTÉS ---

def check_color_match(hsv_frame, x1, y1, x2, y2, lower, upper):
    if np.array_equal(lower, np.array([0, 0, 0])) and np.array_equal(upper, np.array([179, 255, 255])):
        return False
    torso_bottom = int(y1 + (y2 - y1) / 2.5)
    patch = hsv_frame[int(y1):torso_bottom, int(x1):int(x2)]
    if patch.size == 0:
        return False
    mask = cv2.inRange(patch, lower, upper)
    return (cv2.countNonZero(mask) / (patch.shape[0] * patch.shape[1])) > 0.15

def is_point_in_pitch_trapezoid(foot_x, foot_y, width, height, roi_top_pct, roi_bottom_pct):
    """
    A kamera ferde látószögét lekövető trapéz geometriai szűrő.
    A túlsó oldalvonal dőlését interpolálja X mentén.
    """
    # A túlsó oldalvonal Y pozíciója bal oldalon mélyebben fut be a képbe
    y_top_left = height * (roi_top_pct + 0.05)   # kb. 480-500px
    y_top_right = height * (roi_top_pct - 0.02)  # kb. 400-420px
    
    # X koordináta alapján kiszámoljuk az adott pontban a túlsó vonal magasságát
    ratio_x = foot_x / float(width)
    dynamic_top_y = y_top_left + ratio_x * (y_top_right - y_top_left)
    
    y_bottom = height * (1.0 - roi_bottom_pct)
    
    # Alsó bal sarok kispad kizárási zóna (Y > 920 és X < 450)
    if foot_y > height * 0.85 and foot_x < width * 0.25:
        return False, "Kispad zóna kizárva (bal alsó sarok)"

    if foot_y < dynamic_top_y:
        return False, f"Túlsó oldalvonalon kívül (Y={foot_y} < DinamikusTop={dynamic_top_y:.1f})"
    if foot_y > y_bottom:
        return False, f"Alsó ROI sávon kívül (Y={foot_y} > {y_bottom:.1f})"
        
    return True, "Pályán belül"

def create_pitch_tracker(video_file, width, height, fps, start_sec, calib_points_str, H_static, config):
    """
    Pályakövető létrehozása a klip elemzéséhez. Visszatér: (tracker vagy None, leírás a naplóba).
    - dynamic_homography = False: a régi viselkedés (egyetlen, statikus H).
    - calib_pitch_points: a kattintott pontok pálya-koordinátái ("x,y;..." méterben) – így
      fél pályás képen is lehet kalibrálni (pl. a 16-os sarkai); üresen a régi négysarkos sorrend.
    - calib_time_sec: melyik videó-időpont képén kattintott a felhasználó. Ha a klip
      közepén, a mozgást a klip elejéig visszaszámoljuk (előfuttatás, csak követés).
    - Kalibráció nélkül a pályavonalakból próbál automatikusan kalibrálni.
    """
    if not config.get("dynamic_homography", True):
        return None, "KI (statikus homográfia)" + ("" if H_static is not None else ", nincs kalibráció")
    H0 = H_static
    if config.get("calib_pitch_points") and calib_points_str:
        try:
            H0 = parse_calib_points(calib_points_str, width, height, config["calib_pitch_points"])
        except Exception as ex:
            print(f"Homográfia hiba (pálya-pontok): {ex}")
    # SEBESSÉG: adaptív vonal-finomítás (stabil követésnél ritkább; 3 = a régi, rögzített ütem)
    refine_max = config.get("pitch_refine_max_every", model_cfg.get("pitch_refine_max_every", 3))
    tracker = PitchTracker((width, height), np.array(config["grass_lower"]), np.array(config["grass_upper"]),
                           refine_max_every=refine_max)
    if H0 is None:
        return tracker, "BE, kézi kalibráció nincs -> automatikus kalibráció a pályavonalakból"
    msg = "BE, kézi kalibráció a klip elején"
    calib_t = config.get("calib_time_sec")
    k = int(round((calib_t - start_sec) * fps)) if calib_t is not None else 0
    if k > 20 * fps:
        # Messze (> 20 mp) a klip elejétől: a visszaszámolás sodródna és lassú lenne – a kalibrációból
        # csak a (rögzített) kamerahelyet vesszük át, az állást a követő keresi meg (pitch_tracker.py)
        tracker.set_calibration(H0, as_prior=True)
        ok = tracker.manual_prior is not None
        return tracker, (f"BE, kézi kalibráció {calib_t:.2f} mp-nél: " +
                         ("kamerahely átvéve, az állást a követő keresi" if ok else
                          "nem írható le rögzített kamerával -> automatikus kalibráció"))
    if k > 0:
        cap2 = cv2.VideoCapture(video_file)
        read_count = [0]

        def frames_until_calib():
            for _ in range(k + 1):
                ok, fr = cap2.read()
                if not ok:
                    return
                read_count[0] += 1
                yield fr
        try:
            H_back, fails = tracker.backpropagate(frames_until_calib(), H0)
        finally:
            cap2.release()
        if read_count[0] == k + 1:
            H0 = H_back
            msg = f"BE, kézi kalibráció {calib_t:.2f} mp-nél, visszaszámolva {k} képkockán (hibás lépés: {fails})"
        else:
            msg = f"BE, a kalibráció ideje ({calib_t:.2f} mp) a klipen kívül esik – a klip elejére vonatkoztatva"
    tracker.set_calibration(H0)
    return tracker, msg


def cut_and_analyze_sync(video_path, start_sec, end_sec, dev_mode, detect_mode, calib_points_str, config, progress=None):
    ensure_model()  # lusta betöltés (első elemzéskor)
    t_setup0 = time.perf_counter()  # PROFILOZÁS: előkészítés (vágás, fűmodell, pályakövető) ideje
    clip_id = str(uuid.uuid4())[:8]
    output_filename = f"raw_{clip_id}.mp4"
    output_path = os.path.abspath(os.path.join(CLIPS_DIR, output_filename))
    events_path = os.path.abspath(os.path.join(CLIPS_DIR, f"events_{clip_id}.json"))
    debug_log_path = os.path.abspath(os.path.join(CLIPS_DIR, f"analysis_debug_{clip_id}.log"))
    temp_cut = os.path.abspath(os.path.join(CLIPS_DIR, f"temp_cut_{clip_id}.mp4"))

    # background.run: háttér-mód prioritás, és a job megszakítása ezt a másolást is leállítja
    background.run([
        "ffmpeg.exe", "-y", "-ss", str(start_sec), "-i", video_path,
        "-t", str(end_sec - start_sec), "-c", "copy", temp_cut
    ], capture_output=True)

    if not dev_mode:
        os.rename(temp_cut, output_path)
        return f"/clips/{output_filename}", output_path, events_path

    cap = cv2.VideoCapture(temp_cut)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    engine_instance = TacticalEngine(team1_attacks_right=config.get("team1_attacks_right", True))

    H = None
    if calib_points_str and config.get("calib_pitch_points"):
        # tetszőleges látható pályapontok (pl. 16-os sarkai) – a statikus H is ezekből (nem sarok-sorrendből)
        try:
            H = parse_calib_points(calib_points_str, width, height, config["calib_pitch_points"])
        except Exception as ex:
            print(f"Homográfia hiba (pálya-pontok): {ex}")
    elif calib_points_str:
        try:
            pts = []
            for p in calib_points_str.split(';'):
                if p:
                    rx, ry = map(float, p.split(','))
                    pts.append([rx * width, ry * height])
            if len(pts) == 4:
                H = engine_instance.compute_homography(pts)
        except Exception as ex:
            print(f"Homográfia hiba: {ex}")

    # --- ADAPTÍV FŰ: videónként tanult fűmodell (pitch_filters.GrassModel) a fix HSV tartomány helyett ---
    # "auto" (alap): ha a kliens nem állított be saját fű-HSV tartományt; a kézi tartomány mindig nyer.
    grass_mode = grass_mode_from_config(config)
    grass_model = GrassModel.learn_from_video(temp_cut) if grass_mode == "auto" else None
    pt_config = config
    if grass_model is not None and grass_model.ready:
        # a pályakövető (pitch_tracker.py) csak HSV tartományt fogad: az alap tartomány és a tanult fű
        # HSV-burkának uniója (sosem szűkebb a réginél; sötét / árnyékos / fakó fűnél tágabb)
        # TODO(pitch_tracker): fűmaszk-függvény / GrassModel átadása, ha a PitchTracker fogad ilyet
        g_lo, g_hi = grass_model.hsv_bounds(union_default=True)
        pt_config = dict(config, grass_lower=[int(v) for v in g_lo], grass_upper=[int(v) for v in g_hi])
    grass_update_every = max(1, int(round(2.0 * fps)))   # a fűmodell lassú frissítése ~2 mp-enként
    # --- /ADAPTÍV FŰ ---

    # --- Pályakövetés: képkockánkénti homográfia mozgó (Veo) kamerához (pitch_tracker.py) ---
    pitch_tracker, pitch_setup_msg = create_pitch_tracker(temp_cut, width, height, fps, start_sec,
                                                          calib_points_str, H, pt_config)
    pitch_stats = {"ok": 0, "bizonytalan": 0, "elveszett": 0, "kalibráció szükséges": 0, "finomítás": 0}
    pitch_boxes = []   # az előző felismerés dobozai (a követés ezeket kitakarja)
    H_cur = H
    # --- /Pályakövetés ---

    colors = {k: (np.array(config[f"{k}_lower"]), np.array(config[f"{k}_upper"])) 
              for k in ["grass", "ref", "team1", "team2", "keeper1", "keeper2"]}

    # Megemelt küszöbértékek a nézők végleges kizárásához
    roi_top_pct = max(config["roi_top_pct"], 0.28) # Felső horizont lejjebb hozva
    # Min. játékosmagasság a képmagasság arányában (nincs fix px alsó korlát: a régi 50 px 720p-n a túlsó
    # oldal játékosainak ~harmadát eldobta). A többi szűrő a dobozmérethez viszonyított, kis dobozra is működik.
    # A 8 px csak technikai alsó korlát (a mezszín-leíró ez alatt nem ad eredményt) – 480p felett nem számít.
    min_h = max(height * config["min_player_height_pct"], 8.0)

    grass_lower, grass_upper = colors["grass"]
    camera_motion = CameraMotion()
    stationary_filter = StationaryFilter(fps, config["stationary_window_sec"], config["stationary_max_move"])

    # --- PÁLYAHATÁR / KÖVETÉS-ÉPSÉG / KAPUS-MEGERŐSÍTÉS (pitch_filters.py) ---
    # Csak megbízható képkockánkénti H mellett szűr (pitch_h_trusted); pitch_margin_m <= 0: kikapcsolva.
    pitch_margin_m = float(config.get("pitch_margin_m", 2.5))
    pitch_boundary = PitchBoundary((width, height), margin_m=pitch_margin_m)
    track_sanity = PitchTrackSanity(fps) if pitch_margin_m > 0 else None
    keeper_gate = KeeperGate(fps)
    pitch_q = None               # a pályakövető minősége (statikus H-nál None)
    referee_votes = {}           # követés -> [bíró osztályú, összes] detektálás (bíró játékosnak tévesztve)
    ball_last_frame = None       # LABDAKÖVETÉS: a legutóbbi követő-frissítés képkockája (gyors mód: dt_frames)
    # --- /PÁLYAHATÁR ---

    # --- AUTO CSAPAT (team_classifier.py): csak ha nincs kézi csapat-HSV tartomány ---
    team_clf = None
    if auto_teams_enabled(config):
        team_clf = TeamClassifier(grass_lower=grass_lower, grass_upper=grass_upper,
                                  swap_teams=config.get("swap_teams", False),
                                  reference_colors=config.get("reference_team_colors"),
                                  grass_model=grass_model)  # ADAPTÍV FŰ: fű-kiszűrés + megvilágítás-korrekció
    # --- AUTO CSAPAT VÉGE ---

    # --- KIMENETI KÓDOLÁS: a fast_cut által felderített H.264 kódoló (NVENC / QSV / AMF / libx264 veryfast /
    # libopenh264), külön író szálon, opcionálisan kicsinyítve (output_scale, pl. 0.5 -> 960x540) ---
    output_scale = float(config.get("output_scale", 1.0) or 1.0)
    writer = AnnotatedVideoWriter(output_path, width, height, fps, output_scale=output_scale)
    # --- /KIMENETI KÓDOLÁS ---

    # --- LABDA-KIVÁGÁS: natív felbontású labda-felismerés a becsült labdahely körül / csempéken ---
    crop_setting = config.get("ball_crop_pass")
    if crop_setting is None:
        crop_setting = model_cfg.get("ball_crop_pass", "auto")
    ball_crops = None
    if role_classes["ball"] and auto_flag(crop_setting, device):
        ball_crops = BallCropSearch((width, height), crop=model_cfg.get("ball_crop_size", 640),
                                    tile_every=model_cfg.get("ball_tile_every", 2), fps=fps)
    ball_crop_conf = float(model_cfg.get("ball_crop_conf", 0.10))
    prev_full_hit = False
    # --- /LABDA-KIVÁGÁS ---

    # NÉZŐ A TÚLSÓ PÁLYASZÉLEN: kalibráció nélkül (nem megbízható H) a kerítés előtt álló nézők kiszűrése
    crowd_filter = BoundaryCrowdFilter(fps) if config.get("crowd_filter", True) else None
    # SEBESSÉG: HSV-kép csak akkor kell, ha van kézi színtartomány (fű / kapus / csapat)
    need_hsv = (grass_model is None
                or any(not np.array_equal(colors[k][0], np.array([0, 0, 0])) or
                       not np.array_equal(colors[k][1], np.array([179, 255, 255]))
                       for k in (["keeper1", "keeper2"] + ([] if team_clf is not None else ["team1", "team2"]))))
    OTHER_COLOR = (160, 160, 160)   # ismeretlen / "egyéb" mez: semleges szürke (a hiba így látszik)
    timer = StageTimer()            # PROFILOZÁS

    # --- SEBESSÉG: fűmaszk-gyorsítótár + párhuzamos pályakövetés (model_config: grass_mask_every, parallel_pitch) ---
    grass_mask_every = int(config.get("grass_mask_every", model_cfg.get("grass_mask_every", 1)) or 1)
    grass_cache = GrassMaskCache(grass_model.mask, every=grass_mask_every) if grass_model is not None else None

    def _pitch_step(frame_, boxes_, want_mask_):
        """Kameramozgás + pályakövetés (+ a fűmaszk) egy képkockán. Csak a saját objektumait írja
        (camera_motion, pitch_tracker, grass_cache), így külön szálon is futhat a felismerés alatt."""
        t0 = time.perf_counter()
        camera_motion.update(frame_)
        H_, q_, F_ = None, None, None
        if pitch_tracker is not None:
            H_, q_ = pitch_tracker.update(frame_, boxes_)
            F_ = pitch_tracker.frame_motion() if hasattr(pitch_tracker, "frame_motion") else None
        t1 = time.perf_counter()
        m_ = grass_cache.get(frame_, camera_motion.offset) if want_mask_ else None
        return {"H": H_, "q": q_, "F": F_, "mask": m_, "t_pitch": t1 - t0, "t_mask": time.perf_counter() - t1}

    parallel_pitch = auto_flag(config.get("parallel_pitch", model_cfg.get("parallel_pitch", "auto")), device)
    pitch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="palyakoveto") if parallel_pitch else None
    # --- /SEBESSÉG ---
    setup_sec = time.perf_counter() - t_setup0

    frames_done = 0
    timeline_events = []
    gk_pos_monitor = PositioningMonitor()   # kapus helyezkedési hiba: csak tartós, értelmezhető eltérés

    # Áttekintő lapok (esemény-lapok + percenkénti áttekintés) – kevés képből ellenőrizhető az elemzés
    review_dir = os.path.abspath(os.path.join(CLIPS_DIR, f"review_{clip_id}"))
    review = ReviewExporter(review_dir, fps) if config.get("review_export", True) else None
    fed_engine_events = 0
    fed_timeline_events = 0

    def feed_review_events():
        nonlocal fed_engine_events, fed_timeline_events
        engine_events = engine_instance.event_detector.events
        for ev in engine_events[fed_engine_events:]:
            if ev.get("category") in REVIEW_SKIP_CATEGORIES:
                continue   # a sok passz / labdavezetés ne gyártson egy-egy ellenőrző lapot
            review.add_event(ev)
        fed_engine_events = len(engine_events)
        for ev in timeline_events[fed_timeline_events:]:
            review.add_event(ev)
        fed_timeline_events = len(timeline_events)

    track_meta = TrackMetaCollector()  # követésenkénti csapat / kapus (a mentett pályákhoz)
    quality = VideoQualityMonitor(fps, grass_model=grass_model)  # VIDEÓMINŐSÉG: 0,5 mp-enkénti mérés
    grass_mask, grass_scale = None, 0.5
    # Gyors mód: a kihagyott képkockákon az utolsó felismerés eredményét rajzoljuk ki újra,
    # így a kimeneti videóban minden képkocka megmarad
    last_valid_players = []
    last_referee_boxes = []
    last_other_boxes = []
    last_ball_pixels = []
    team_cols = None

    # --- LABDAKÖVETÉS (ball_tracker.py) ---
    # config["ball_tracker"] = False esetén a régi működés (az utolsó küszöb feletti labda-detektálás)
    ball_tracker = BallTracker(fps, frame_size=(width, height),
                               min_conf=min(BALL_CONF, 0.15)) if config.get("ball_tracker", True) else None
    ball_cand_conf = ball_tracker.min_conf if ball_tracker is not None else BALL_CONF
    ball_cam_ref = None      # a CameraMotion összesített eltolása a legutóbbi követő-frissítéskor
    # Kamera-kompenzáció a pályakövető képkockák közti mozgás-homográfiájával (előző kép -> aktuális kép),
    # a legutóbbi labda-frissítés óta összeszorozva; tartalék: inv(H_t)·H_ref, végül a CameraMotion eltolás
    ball_motion_H = None     # összesített képtér-mozgás a legutóbbi labda-frissítés óta (None = megszakadt)
    ball_H_ref = None        # a kép -> pálya H a legutóbbi labda-frissítéskor
    ball_comp_stats = {"homográfia": 0, "H különbség": 0, "eltolás": 0}
    last_ball_state = None
    # NYERS LABDA: honnan jöttek a labda-jelöltek (képkockánként)
    ball_extra_pass = bool(config.get("ball_extra_pass", BALL_EXTRA_PASS))
    ball_stats = {"nyers": 0, "külön menet": 0, "követett": 0}
    # --- /LABDAKÖVETÉS ---

    # A megosztott modellt / követőt az egész klip idejére lefoglaljuk (lásd model_lock),
    # és töröljük az előző elemzésből megmaradt ByteTrack állapotot
    # cleanup: a háttérszálak (képkocka-olvasó, párhuzamos pályakövető) leállítása hiba / megszakítás esetén is
    with model_lock, open(debug_log_path, "w", encoding="utf-8") as debug_file, contextlib.ExitStack() as cleanup:
        if pitch_pool is not None:
            cleanup.callback(pitch_pool.shutdown, wait=True)
        # DETEKTOR-HÁTTÉR: a saját követő nullázása (az FPS a követés-puffer méretéhez kell)
        trackers_reset = OWN_BACKEND.reset(frame_rate=fps) if OWN_BACKEND is not None else reset_tracker(model)
        debug_file.write(f"=== TACTICAL ANALYZER RUN LOG: {clip_id} ===\n")
        debug_file.write(f"Videó felbontás: {width}x{height} @ {fps} FPS, Összes képkocka: {total_frames}\n")
        debug_file.write(f"Dinamikus Felső ROI Százalék: {roi_top_pct:.2f}, Min Magasság: {min_h:.1f}px\n")
        debug_file.write(f"Fix trapéz szűrő: {'BE' if config['use_static_roi'] else 'KI'}, "
                         f"Fű arány min.: {config['grass_min_ratio']:.2f}, "
                         f"Álló alak ablak: {config['stationary_window_sec']:.1f}s\n")
        debug_file.write(f"Modell: {OWN_BACKEND.describe() if OWN_BACKEND is not None else model_weights}, "
                         f"osztályok: {describe_role_map(role_classes, model_names)}\n")
        debug_file.write(f"Felismerés minden {DETECT_EVERY_N}. képkockán, törölt követők: {trackers_reset}\n")
        debug_file.write(f"Pályakövetés: {pitch_setup_msg}\n")
        debug_file.write(f"Nyers labda-callback: {'saját háttér (közvetlen)' if OWN_BACKEND is not None else ('BE' if RAW_BALL_HOOK else 'nincs (a követett dobozok)')}, "
                         f"külön labda-menet: {'BE' if ball_extra_pass else 'KI'}\n")
        debug_file.write("Fűmaszk: " + (grass_model.describe() if grass_model is not None
                                         else f"kézi HSV {list(grass_lower)}–{list(grass_upper)}") + "\n")  # ADAPTÍV FŰ
        debug_file.write(f"Felismerés: imgsz={IMGSZ}, félpontosság (FP16): {'BE' if use_half() else 'KI'}; "
                         f"kimeneti videó: {writer.describe()}; labda-kivágás: "
                         + (f"BE ({ball_crops.crop}px, {len(ball_crops.tiles)} csempe)" if ball_crops is not None else "KI")
                         + f"; előkészítés: {setup_sec:.1f} mp\n")
        debug_file.write(f"Gyorsítás: pályakövetés {'külön szálon (párhuzamos)' if pitch_pool is not None else 'soros'}"
                         + (f", vonal-finomítás legritkábban {pitch_tracker.refine_max_every} képkockánként"
                            if getattr(pitch_tracker, "refine_max_every", None) else "")
                         + f", fűmaszk {grass_mask_every} képkockánként\n\n")

        # BEMENETI DEKÓDOLÁS: előolvasás külön szálon (a dekódolás átfed a feldolgozással)
        frame_reader = FramePrefetcher(cap) if config.get("prefetch_frames", True) else None
        if frame_reader is not None:
            cleanup.callback(frame_reader.close)
        timer.t_start = time.perf_counter()
        while frame_reader is not None or cap.isOpened():
            with timer("decode"):
                ret, frame = frame_reader.read() if frame_reader is not None else cap.read()
            if not ret:
                break

            current_time_sec = start_sec + (frames_done / fps)
            cam_prev = camera_motion.offset.copy()   # VIDEÓMINŐSÉG: a kamera elfordulása ezen a képkockán
            q_mask = grass_mask                      # VIDEÓMINŐSÉG: az előző felismerés fűmaszkja (mint eddig)
            detected = should_detect(frames_done, DETECT_EVERY_N)
            gk_real_coords = None
            # --- ADAPTÍV FŰ: a fűmodell lassú frissítése (a maszk előtt, a nyers képkockán) ---
            with timer("grass_quality"):
                if grass_model is not None and frames_done > 0 and frames_done % grass_update_every == 0:
                    grass_model.observe(frame)
            # --- Pályakövetés + kameramozgás (+ fűmaszk): soros módban itt, párhuzamos módban külön szálon,
            # a felismeréssel egy időben (PÁRHUZAMOS PÁLYAKÖVETÉS) ---
            want_mask = detected and grass_model is not None
            if pitch_pool is not None:
                pitch_future = pitch_pool.submit(_pitch_step, frame, pitch_boxes, want_mask)
            else:
                pitch_res = _pitch_step(frame, pitch_boxes, want_mask)
                timer.add("pitch", pitch_res["t_pitch"])
                timer.add("grass_quality", pitch_res["t_mask"])
            if frames_done % 250 == 0:
                stationary_filter.prune(frames_done)
                keeper_gate.prune(frames_done)  # PÁLYAHATÁR
                if track_sanity is not None:
                    track_sanity.prune(frames_done)

            own_det = None
            crop_pre = None
            if detected and pitch_pool is not None:
                # PÁRHUZAMOS PÁLYAKÖVETÉS: a felismerés (GPU) a pályakövetéssel egy időben; a saját követő
                # társítása (kamera-kompenzációval) csak a pályakövetés mozgásbecslése után
                t_det = time.perf_counter()
                if OWN_BACKEND is not None:
                    own_det = OWN_BACKEND.detector.detect(frame)
                else:
                    _raw_ball_buf["boxes"] = None
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device, verbose=False,
                                          imgsz=IMGSZ, half=use_half())
                timer.add("detect", time.perf_counter() - t_det)
                if ball_crops is not None:
                    # LABDA-KIVÁGÁS (GPU) is a pályakövetés alatt: csak az előző képkocka labdaállapota kell hozzá
                    t_bp = time.perf_counter()
                    windows = ball_crops.plan(frames_done, last_ball_state, prev_full_hit)
                    crop_pre = ball_crops.run(
                        frame, windows,
                        lambda cs: _ball_crop_detect(cs, min(ball_crop_conf, ball_cand_conf), ball_crops.crop)) if windows else []
                    timer.add("ball_pass", time.perf_counter() - t_bp)
            if pitch_pool is not None:
                t_wait = time.perf_counter()
                pitch_res = pitch_future.result()
                timer.add("pitch", time.perf_counter() - t_wait)       # csak a NEM átfedett (várakozási) idő
                timer.add("pitch_overlap", pitch_res["t_pitch"] + pitch_res["t_mask"])   # a háttérszál ideje

            # --- Pályakövetés eredménye: az aktuális képkocka homográfiája (a rajzolás előtt) ---
            t_pitch = time.perf_counter()
            if pitch_tracker is not None:
                H_cur, pitch_q = pitch_res["H"], pitch_res["q"]
                pitch_stats[pitch_q["status"]] = pitch_stats.get(pitch_q["status"], 0) + 1
                pitch_stats["finomítás"] += int(pitch_q.get("refined", False))
                if frames_done % 25 == 0:
                    debug_file.write(f"F#{frames_done} PÁLYA: {pitch_q['status']}, bizalom={pitch_q.get('conf')}, "
                                     f"inlier={pitch_q.get('inliers', '-')}, elveszett={pitch_q.get('lost')}, "
                                     f"forrás={pitch_q.get('source')}\n")
                # LABDAKÖVETÉS: a képtér-mozgás (előző -> aktuális kép) összeszorzása a labda-frissítések között
                F_img = pitch_res["F"]
                if ball_motion_H is not None:
                    ball_motion_H = None if F_img is None else F_img @ ball_motion_H
                if OWN_BACKEND is not None:   # DETEKTOR-HÁTTÉR: kamera-kompenzáció a saját követőnek
                    OWN_BACKEND.note_motion(F_img)
            timer.add("pitch", time.perf_counter() - t_pitch)
            # --- /Pályakövetés ---
            # --- VIDEÓMINŐSÉG (a nyers képkockán, a rajzolás előtt; a kamera elfordulása a kameramozgásból) ---
            with timer("grass_quality"):
                if quality.due(frames_done):
                    pan_px = float(np.hypot(*(camera_motion.offset - cam_prev))) * 1280.0 / max(1, width)
                    q_sample = quality.update(frame, current_time_sec, grass_mask=q_mask, pan_px=pan_px)
                    debug_file.write(f"F#{frames_done} {VideoQualityMonitor.describe_sample(q_sample)}\n")
            # --- /VIDEÓMINŐSÉG ---

            if detected:
                t_gq = time.perf_counter()
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if need_hsv else None  # SEBESSÉG: csak ha kell
                if grass_model is not None:  # ADAPTÍV FŰ (tanulás előtt a modell az alap HSV-re esik vissza)
                    grass_mask, grass_scale = pitch_res["mask"]
                else:
                    grass_mask, grass_scale = build_grass_mask(hsv_frame, grass_lower, grass_upper)
                if crowd_filter is not None:
                    crowd_filter.set_frame(frames_done, grass_mask, grass_scale, height)
                timer.add("grass_quality", time.perf_counter() - t_gq)
                # --- DETEKTOR-HÁTTÉR: saját detektor + követő, vagy az ultralytics model.track() ---
                t_det = time.perf_counter()
                own_raw_ball = None
                if OWN_BACKEND is not None:
                    # a labda nyers jelöltjei külön (párhuzamos módban a már kész felismerésből)
                    results, own_raw_ball = OWN_BACKEND.track(frame, detections=own_det)
                elif pitch_pool is None:
                    _raw_ball_buf["boxes"] = None   # NYERS LABDA: a callback tölti ki a track() alatt
                    # SEBESSÉG: imgsz a beállításból, CUDA-n félpontosság (FP16)
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device, verbose=False,
                                          imgsz=IMGSZ, half=use_half())
                timer.add("detect", time.perf_counter() - t_det)
                # --- /DETEKTOR-HÁTTÉR ---
                pitch_boxes = [tuple(map(int, b.xyxy[0])) for r in results for b in r.boxes]  # pályakövetéshez
                # PÁLYAHATÁR: a képkocka H-ja csak akkor szűr, ha megbízható (a pályakövető None-t is adhat)
                h_trusted = pitch_h_trusted(H_cur, pitch_q if pitch_tracker is not None else None)
                pitch_boundary.set_frame(H_cur, h_trusted)
                ball_person_boxes = [tuple(float(v) for v in b.xyxy[0][:4]) for r in results for b in r.boxes
                                     if int(b.cls[0]) not in role_classes["ball"]]  # LABDA: fej-prior

                raw_detections = []
                referee_boxes = []
                other_boxes = []      # ismeretlen / "egyéb" mez (nem bíró): szürke keret
                ball_pixels = []
                ball_pixel = None
                ball_candidates = []  # LABDAKÖVETÉS: az összes jelölt a követőnek

                # --- NYERS LABDA: a követő előtti labda-dobozok; tartalék: külön menet / a követett dobozok ---
                ball_src = own_raw_ball if own_raw_ball is not None else (
                    _raw_ball_buf["boxes"] if RAW_BALL_HOOK else None)   # DETEKTOR-HÁTTÉR: saját háttér = nyers
                t_bp = time.perf_counter()
                if ball_src is not None:
                    ball_stats["nyers"] += 1
                elif ball_extra_pass:
                    ball_src = _extra_ball_pass(frame, ball_cand_conf)
                    ball_stats["külön menet"] += 1
                else:
                    ball_src = [(tuple(float(v) for v in b.xyxy[0][:4]), float(b.conf[0]))
                                for r in results for b in r.boxes if int(b.cls[0]) in role_classes["ball"]]
                    ball_stats["követett"] += 1
                # LABDA-KIVÁGÁS: natív felbontású kivágás(ok) a becsült labdahely körül / csempén; a dobozok
                # a teljes képre vetítve, a teljes képes jelöltekkel egyesítve
                if ball_crops is not None:
                    if crop_pre is not None:       # párhuzamos mód: már lefutott a pályakövetés alatt
                        crop_boxes = crop_pre
                    else:
                        windows = ball_crops.plan(frames_done, last_ball_state, prev_full_hit)
                        crop_boxes = ball_crops.run(
                            frame, windows,
                            lambda cs: _ball_crop_detect(cs, min(ball_crop_conf, ball_cand_conf), ball_crops.crop)) if windows else []
                    prev_full_hit = any(c > BALL_CONF for _b, c in ball_src)
                    ball_src = ball_crops.merge(ball_src, crop_boxes, count_conf=BALL_CONF)
                timer.add("ball_pass", time.perf_counter() - t_bp)
                for bxyxy, bconf in ball_src:
                    if bconf <= ball_cand_conf:
                        continue
                    bx1, by1, bx2, by2 = map(int, bxyxy)
                    ball_candidates.append({"xyxy": (bx1, by1, bx2, by2), "conf": bconf})
                    if bconf > BALL_CONF:
                        ball_pixel = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                        ball_pixels.append(ball_pixel)
                # --- /NYERS LABDA ---

                t_filters = time.perf_counter()
                team_sec = 0.0
                team_cols = team_clf.team_colors() if team_clf is not None else None
                for r in results:

                    # Bírók: nem kerülnek a játékosok közé, csak semleges színnel rajzoljuk ki őket
                    for box in [b for b in r.boxes if int(b.cls[0]) in role_classes["referee"]]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if box.id is not None:  # PÁLYAHATÁR: követésenkénti bíró-szavazat
                            rv = referee_votes.setdefault(int(box.id[0].item()), [0, 0])
                            rv[0] += 1
                            rv[1] += 1
                        on_grass, _ = foot_on_grass(grass_mask, grass_scale, x1, x2, y2, config["grass_min_ratio"])
                        if on_grass:
                            referee_boxes.append((x1, y1, x2, y2))

                    if r.boxes.id is not None:
                        for box in [b for b in r.boxes
                                    if int(b.cls[0]) in role_classes["player"] or int(b.cls[0]) in role_classes["goalkeeper"]]:
                            # A focis modell külön kapus osztályt ad – ez a színegyezés mellett is kapusnak jelöl
                            model_keeper = int(box.cls[0]) in role_classes["goalkeeper"]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            tid = int(box.id[0].item()) if box.id is not None else None
                            foot_x = max(0, min(int((x1 + x2) / 2), width - 1))
                            foot_y = max(0, min(int(y2), height - 1))
                            box_w, box_h = x2 - x1, y2 - y1

                            # 1. Méretellenőrzés
                            if box_h < min_h:
                                debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> Túl kicsi méret ({box_h}px < {min_h:.1f}px)\n")
                                continue

                            # 2. Dobozarány szűrés
                            if box_w > box_h * 0.82:
                                debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> Nem álló alak (Arány={box_w/box_h:.2f})\n")
                                continue

                            # 3. Fix trapéz pályahatár (csak álló kamerához – a forgó Veo képnél kikapcsolva)
                            if config["use_static_roi"]:
                                in_pitch, reason = is_point_in_pitch_trapezoid(foot_x, foot_y, width, height, roi_top_pct, config["roi_bottom_pct"])
                                if not in_pitch:
                                    debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> {reason}\n")
                                    continue

                            # 4. Fűszűrő: a lábnak füvön kell állnia (lelátó, kispad, reklámtábla mögött nincs fű)
                            on_grass, grass_ratio = foot_on_grass(grass_mask, grass_scale, x1, x2, y2, config["grass_min_ratio"])
                            if not on_grass:
                                debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> Nem füvön áll (fű arány={grass_ratio:.2f})\n")
                                continue

                            # --- NÉZŐ A TÚLSÓ PÁLYASZÉLEN (csak megbízható H nélkül; a pályahatár-szűrő helyett) ---
                            if crowd_filter is not None and not h_trusted and not model_keeper and \
                                    not (team_clf is not None and team_clf.is_keeper_track(tid, current_time_sec)):
                                csx, csy = camera_motion.stabilize(foot_x, foot_y)
                                if crowd_filter.update(frames_done, foot_x, foot_y, csx, csy, box_h):
                                    debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> Néző a túlsó pályaszélen "
                                                     f"(a kerítés előtti sávban áll)\n")
                                    continue

                            # --- PÁLYAHATÁR: 4b. a (megbízható) képkocka-H szerinti pályahatár + margó ---
                            in_bounds, bound_reason, pitch_raw = pitch_boundary.check(foot_x, foot_y, box_h)
                            if not in_bounds:
                                debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> {bound_reason}\n")
                                continue
                            # bíró, akit a modell néha játékosnak lát: a követés többségi osztálya dönt
                            if tid is not None:
                                rv = referee_votes.setdefault(tid, [0, 0])
                                rv[1] += 1
                                if rv[1] >= 5 and rv[0] >= 0.5 * rv[1]:
                                    referee_boxes.append((x1, y1, x2, y2))
                                    continue
                            # --- /PÁLYAHATÁR ---

                            is_k1 = check_color_match(hsv_frame, x1, y1, x2, y2, colors["keeper1"][0], colors["keeper1"][1])
                            is_k2 = check_color_match(hsv_frame, x1, y1, x2, y2, colors["keeper2"][0], colors["keeper2"][1])
                            # PÁLYAHATÁR: kapus-megerősítés – a csak színre egyező, a 16-osokon kívül élő alak
                            # (pl. narancs mezes csere a sárga kapus tartományában) nem kapus
                            is_keeper, keeper_confirmed, colour_ok = keeper_gate.update(
                                tid, frames_done, model_keeper, is_k1 or is_k2, pitch_raw)
                            if (is_k1 or is_k2) and not colour_ok:
                                if frames_done % 25 == 0:
                                    debug_file.write(f"F#{frames_done} ID#{tid}: kapus-szín, de a 16-osokon kívül él -> nem kapus\n")
                                is_k1 = is_k2 = False
                            if team_clf is not None and team_clf.is_keeper_track(tid, current_time_sec):
                                keeper_confirmed = True  # AUTO CSAPAT: pozíció alapú kapus

                            # --- PÁLYAHATÁR: 4c. követés-épség pályatérben (kint él / sosem mozdul) ---
                            if track_sanity is not None and h_trusted:
                                suppressed, why = track_sanity.update(tid, frames_done, pitch_raw,
                                                                      exempt_still=keeper_confirmed)
                                if suppressed:
                                    debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> {why}\n")
                                    continue
                            # --- /PÁLYAHATÁR ---

                            # 5. Álló alak szűrő (néző, edző) – csak a MEGERŐSÍTETT kapus kivétel (modell / pozíció),
                            # a csak színre egyező nem
                            stab_x, stab_y = camera_motion.stabilize(foot_x, foot_y)
                            is_static = stationary_filter.update(tid, frames_done, stab_x, stab_y, box_h)
                            if is_static and not keeper_confirmed:
                                debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> Mozdulatlan alak ({config['stationary_window_sec']:.1f}s)\n")
                                continue

                            t_team = time.perf_counter()
                            crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                            dominant_color = extract_dominant_color(crop, grass_lower, grass_upper, grass_model=grass_model)
                            team_known = True

                            # --- AUTO CSAPAT: mezszín-klaszterezés + követésenkénti szavazás ---
                            if team_clf is not None:
                                pitch_xy = engine_instance.transform_to_pitch(foot_x, foot_y, H_cur) if H_cur is not None else None
                                # ADAPTÍV FŰ: árnyékban álló játékos mezszíne a napos fűhöz igazítva
                                illum_gain = (grass_model.illumination_gain(frame, (x1, y1, x2, y2))[0]
                                              if grass_model is not None else None)
                                auto = team_clf.observe(tid, crop, current_time_sec, pitch_xy=pitch_xy,
                                                        model_keeper=model_keeper, illum_gain=illum_gain)
                                if (auto["is_referee"] or auto["role"] == "other") and not is_keeper:
                                    # bíró: sárga keret; egyéb mez (pl. kapus pályakoordináta nélkül): szürke keret
                                    (referee_boxes if auto["is_referee"] else other_boxes).append((x1, y1, x2, y2))
                                    team_sec += time.perf_counter() - t_team
                                    continue
                                is_keeper = is_keeper or auto["is_keeper"]
                                if is_k1 or is_k2:
                                    team_id = 1 if is_k1 else 2
                                else:
                                    # ismeretlen (bemelegítés eleje / kapus csapata nem dönthető el): szín szerinti
                                    # tipp a motornak, de a rajzon szürke (így a sikertelen besorolás látszik)
                                    team_known = auto["team_id"] is not None
                                    team_id = (auto["team_id"] or auto["color_team"]
                                               or (1 if dominant_color[0] > dominant_color[2] else 2))
                            # --- AUTO CSAPAT VÉGE ---
                            # Kézi színek: ELŐBB mindkét kapus-tartomány (különben a 2. kapust, akinek a mezét
                            # az 1. csapat tartománya is lefedi, az 1. csapatba tennénk), csak utána a csapatok
                            elif is_k1 or is_k2:
                                team_id = 1 if is_k1 else 2
                            elif check_color_match(hsv_frame, x1, y1, x2, y2, colors["team1"][0], colors["team1"][1]):
                                team_id = 1
                            elif check_color_match(hsv_frame, x1, y1, x2, y2, colors["team2"][0], colors["team2"][1]):
                                team_id = 2
                            else:
                                team_id = 1 if dominant_color[0] > dominant_color[2] else 2

                            team_sec += time.perf_counter() - t_team
                            raw_detections.append({
                                'track_id': tid,
                                'team_id': team_id,
                                'team_known': team_known,
                                'screen_foot': (foot_x, foot_y),
                                'is_keeper': is_keeper,
                                'bbox': (x1, y1, x2, y2)
                            })

                timer.add("team", team_sec)
                timer.add("filters", time.perf_counter() - t_filters - team_sec)
                # --- LABDAKÖVETÉS: a képkocka labdája a követő kimenete (detektált vagy rövid ideig becsült) ---
                t_bt = time.perf_counter()
                if ball_tracker is not None:
                    cam_now = camera_motion.offset.copy()
                    cam_shift = None if ball_cam_ref is None else cam_now - ball_cam_ref
                    ball_cam_ref = cam_now
                    # erősorrend: pályakövető mozgás-homográfia > inv(H_t)·H_ref (ha a mozgásbecslés kimaradt) > eltolás
                    cam_H = None
                    if ball_motion_H is not None:
                        cam_H = ball_motion_H
                        ball_comp_stats["homográfia"] += 1
                    elif H_cur is not None and ball_H_ref is not None:
                        try:
                            cam_H = np.linalg.inv(H_cur) @ ball_H_ref
                            cam_H = cam_H / cam_H[2, 2]
                            ball_comp_stats["H különbség"] += 1
                        except (np.linalg.LinAlgError, ZeroDivisionError):
                            cam_H = None
                    if cam_H is None and cam_shift is not None:
                        ball_comp_stats["eltolás"] += 1
                    # PÁLYAHATÁR: dt_frames = a valódi képkocka-különbség (gyors mód), pálya/méret/fej priorok
                    ball_dt = 1 if ball_last_frame is None else max(1, frames_done - ball_last_frame)
                    ball_last_frame = frames_done
                    last_ball_state = ball_tracker.update(ball_candidates, camera_shift=cam_shift, camera_H=cam_H,
                                                          grass_mask=grass_mask, grass_scale=grass_scale,
                                                          frame_idx=frames_done, dt_frames=ball_dt,
                                                          pitch_H=pitch_boundary.geom,
                                                          person_boxes=ball_person_boxes)
                    ball_motion_H = np.eye(3) if pitch_tracker is not None else None
                    ball_H_ref = None if H_cur is None else np.array(H_cur, np.float64)
                    ball_pixel = last_ball_state.pixel
                timer.add("ball_tracker", time.perf_counter() - t_bt)
                # --- /LABDAKÖVETÉS ---

                t_eng = time.perf_counter()
                if H_cur is not None:
                    valid_players, real_ball = engine_instance.update_tracks(frames_done, current_time_sec, raw_detections, ball_pixel, H_cur)
                else:
                    valid_players = raw_detections
                    real_ball = None

                last_valid_players, last_referee_boxes, last_ball_pixels = valid_players, referee_boxes, ball_pixels
                last_other_boxes = other_boxes
                track_meta.add(valid_players)
                timer.add("engine", time.perf_counter() - t_eng)
            else:
                # Kihagyott képkocka (gyors mód): az előző felismerés újrarajzolása,
                # statisztika és esemény nem keletkezik belőle (különben duplán számolnánk)
                valid_players, referee_boxes, ball_pixels = last_valid_players, last_referee_boxes, last_ball_pixels
                other_boxes = last_other_boxes
                real_ball = None

            t_draw = time.perf_counter()
            # a tanult mezszínek (RGB -> BGR) a csapatok keretéhez; nélkülük a régi piros / kék
            team_bgr = {1: (0, 0, 255), 2: (255, 0, 0)}
            if team_cols:
                for k_t, key in ((1, "team1"), (2, "team2")):
                    if team_cols.get(key):
                        r_, g_, b_ = team_cols[key]
                        team_bgr[k_t] = (int(b_), int(g_), int(r_))

            if H_cur is not None and config.get("draw_pitch_overlay", True):
                render_pitch_overlay(frame, H_cur)  # halvány pályavonalak: látszik, mennyire pontos a kalibráció

            if ball_tracker is not None:
                draw_ball_state(frame, last_ball_state)  # LABDAKÖVETÉS: teli kör = detektált, üres = becsült
            else:
                for bp in ball_pixels:
                    cv2.circle(frame, bp, 5, (0, 255, 255), -1)

            for x1, y1, x2, y2 in referee_boxes:
                # Bíró: sárga keret fekete szegéllyel
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for x1, y1, x2, y2 in other_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), OTHER_COLOR, 1)

            for p in valid_players:
                x1, y1, x2, y2 = p['bbox']
                color = team_bgr.get(p['team_id'], OTHER_COLOR) if p.get('team_known', True) else OTHER_COLOR
                if p['is_keeper']:
                    color = (0, 165, 255)
                    gk_real_coords = p.get('pitch_coords')
                # fekete szegély: a fehér / világos mezszínű keret is látszik a füvön
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"#{p['track_id']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if gk_real_coords and real_ball:
                gk_eval = engine_instance.analyze_goalkeeper(gk_real_coords, real_ball)
                # csak tartós, nagy eltérés, ha a labda a kapu közelében van (events.PositioningMonitor)
                # A "helyezkedési hiba" véleményes ítélet (az edző dönti el), ezért alapból nem adjuk ki
                # eseményként; a "gk_positioning_errors": true analysis-beállítással visszakapcsolható.
                if config.get("gk_positioning_errors", False):
                    pos_ev = gk_pos_monitor.update(current_time_sec, gk_eval,
                                                   engine_instance.keeper_team_near(gk_real_coords))
                    if pos_ev is not None:
                        timeline_events.append(pos_ev)

            if review is not None:
                review.add_frame(frames_done, current_time_sec, frame,
                                 valid_players if H_cur is not None else None, real_ball)
                feed_review_events()
            timer.add("draw_review", time.perf_counter() - t_draw)

            with timer("encode"):
                writer.write(frame)   # KIMENETI KÓDOLÁS: író szál (a frame-et ezután nem módosítjuk)
            frames_done += 1
            timer.frames = frames_done
            if frames_done % 25 == 0:
                el = max(1e-6, time.perf_counter() - timer.t_start)
                print(f"Feldolgozva: {frames_done}/{total_frames} képkocka ({(frames_done/max(1, total_frames))*100:.1f}%), "
                      f"{frames_done / el:.2f} kép/mp")
                if progress: progress(100.0 * frames_done / max(1, total_frames), f"{frames_done}/{total_frames} képkocka")  # háttérfeladat haladás (+ megszakítás)

        if pitch_pool is not None:
            pitch_pool.shutdown(wait=True)
        if frame_reader is not None:
            frame_reader.close()
        t_close = time.perf_counter()
        writer_error = writer.close()   # KIMENETI KÓDOLÁS: a sor kiürítése, az ffmpeg lezárása
        timer.add("encode", time.perf_counter() - t_close)
        timer.stop()
        timing = timer.summary()   # PROFILOZÁS
        timing["setup_sec"] = round(setup_sec, 2)
        timing["detect_frames"] = sum(1 for i in range(frames_done) if should_detect(i, DETECT_EVERY_N))
        timing["imgsz"], timing["half"] = IMGSZ, use_half()
        timing["device"] = device
        timing["output_video"] = writer.describe()
        if writer_error:
            debug_file.write(f"\n⚠️ A kimeneti videó írása hibával zárult: {writer_error}\n")
        debug_file.write("\n" + timer.table_text() + f"\nElőkészítés (vágás, fűmodell, pályakövető): {setup_sec:.1f} mp; "
                         f"kimeneti videó: {writer.describe()}\n")
        print(timer.one_line())
        if ball_crops is not None:   # LABDA-KIVÁGÁS: statisztika
            debug_file.write(ball_crops.describe() + "\n")
        if crowd_filter is not None:
            debug_file.write(f"Néző-szűrő (túlsó pályaszél): {crowd_filter.stats}\n")
        quality_summary = quality.summary()  # VIDEÓMINŐSÉG: összesítés a naplóba
        debug_file.write("\n" + quality_summary.get("summary_text", "") + "\n")
        if grass_model is not None:
            debug_file.write(grass_model.describe() + "\n")
        if team_clf is not None:  # AUTO CSAPAT: a tanult klaszterek a naplóba (és a képernyőre)
            debug_file.write("\n" + team_clf.describe() + "\n")
            print(team_clf.describe())
        # PÁLYAHATÁR: összesítés a naplóba
        debug_file.write(f"\nPályahatár (margó {pitch_margin_m:.1f} m) kizárások: {pitch_boundary.stats}"
                         + (f", követés-épség: {track_sanity.stats}" if track_sanity is not None else "") + "\n")

        if ball_tracker is not None:  # LABDAKÖVETÉS: összesítés a naplóba
            debug_file.write("\n" + ball_tracker.summary() + "\n")
            debug_file.write(f"Labda kamera-kompenzáció: {ball_comp_stats}\n")
        # NYERS LABDA: a labda-jelöltek forrása (nyers = a követő előtti callback-másolat)
        debug_file.write(f"Labda-jelöltek forrása (képkocka): {ball_stats}"
                         + (" – FIGYELEM: a callback a követő UTÁN futott, a dobozok nem nyersek"
                            if _raw_ball_buf.get("after_tracker") else "") + "\n")

    cap.release()
    if os.path.exists(temp_cut):
        os.remove(temp_cut)

    # Lövés/gól/kapus események lezárása és összefésülése a helyezkedési hibákkal (idő szerint rendezve)
    engine_instance.finalize(start_sec + frames_done / fps)
    if review is not None:
        feed_review_events()
        review.close(video_quality=quality_summary)
    timeline_events = engine_instance.merge_timeline(timeline_events)
    annotate_event_reliability(timeline_events, quality)  # VIDEÓMINŐSÉG: esemény-megbízhatóság

    analysis_payload = {
        "events": timeline_events,
        "review_url": f"/clips/review_{clip_id}/index.html" if review is not None else None,
        "match_stats": engine_instance.stats,
        "pitch_tracking": {"setup": pitch_setup_msg, "frames": pitch_stats},
        "source": {"video_path": video_path, "start_sec": start_sec, "end_sec": end_sec},
        # Tömörített pályák (5 Hz, 0.1 m) a hőtérképhez / statisztikához, a labda útja 10 Hz-en – lásd heatmap.py
        "trajectories": compact_trajectories(engine_instance.player_trajectories, track_meta.summary(),
                                             ball=engine_instance.ball_trajectory),
        # AUTO CSAPAT: tanult mezszínek (RGB) a kliensnek / taktikai táblának; kézi módban null
        "team_colors": team_clf.team_colors() if team_clf is not None else None,
        "team_mode": "auto" if team_clf is not None else "manual",
        # VIDEÓMINŐSÉG: osztályzat, az egyes képhibák idő-aránya, a legrosszabb szakaszok; a fűmaszk módja
        "video_quality": quality_summary,
        "grass_model": grass_model.describe() if grass_model is not None else "kézi HSV tartomány",
        # PROFILOZÁS: szakaszonkénti idő (ms/képkocka, arány), tényleges kép/mp, kódoló
        "timing": timing,
        "ball_crop": ({"stats": ball_crops.stats, "crop_size": ball_crops.crop, "tiles": len(ball_crops.tiles)}
                      if ball_crops is not None else None),
        "team_model": team_clf.describe() if team_clf is not None else None,
    }
    with open(events_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_payload, f, ensure_ascii=False, indent=4)

    return f"/clips/{output_filename}", output_path, events_path

@app.post("/cut-raw")
async def cut_raw_endpoint(
    video_path: str = Form(...),
    start_sec: float = Form(0.0),
    end_sec: float = Form(30.0),
    dev_mode: bool = Form(True),
    detect_mode: str = Form("team"),
    calib_points: str = Form(""),
    roi_top_pct: float = Form(0.28),
    roi_bottom_pct: float = Form(0.10),
    min_player_height_pct: float = Form(0.022),
    use_static_roi: bool = Form(False),
    grass_min_ratio: float = Form(0.3),
    stationary_window_sec: float = Form(4.0),
    stationary_max_move: float = Form(0.5),
    grass_hsv_lower: str = Form("30,40,40"), grass_hsv_upper: str = Form("85,255,255"),
    ref_hsv_lower: str = Form("0,0,0"), ref_hsv_upper: str = Form("179,255,255"),
    team1_hsv_lower: str = Form("0,0,0"), team1_hsv_upper: str = Form("179,255,255"),
    team2_hsv_lower: str = Form("0,0,0"), team2_hsv_upper: str = Form("179,255,255"),
    keeper1_hsv_lower: str = Form("0,0,0"), keeper1_hsv_upper: str = Form("179,255,255"),
    keeper2_hsv_lower: str = Form("0,0,0"), keeper2_hsv_upper: str = Form("179,255,255"),
    team1_attacks_right: bool = Form(True),
    review_export: bool = Form(True),
    auto_teams: bool = Form(True),
    swap_teams: bool = Form(False),
    dynamic_homography: bool = Form(True),
    draw_pitch_overlay: bool = Form(True),
    calib_time_sec: Optional[float] = Form(None),
    calib_pitch_points: str = Form(""),
    pitch_margin_m: float = Form(2.5),
    grass_model: str = Form("auto"),
    output_scale: float = Form(1.0)
):
    config = {
        "team1_attacks_right": team1_attacks_right,
        "auto_teams": auto_teams, "swap_teams": swap_teams,
        "review_export": review_export,
        # Pályakövetés (lásd pitch_tracker.py, docs/FEJLESZTESI_TERV.md)
        "dynamic_homography": dynamic_homography, "draw_pitch_overlay": draw_pitch_overlay,
        "calib_time_sec": calib_time_sec, "calib_pitch_points": calib_pitch_points,
        "pitch_margin_m": pitch_margin_m,  # PÁLYAHATÁR: a pályán kívüli tűrés (m), 0 = ki
        "grass_model": grass_model,  # ADAPTÍV FŰ: "auto" (tanult fűmodell) / "hsv" (fix tartomány)
        "output_scale": output_scale,  # KIMENETI KÓDOLÁS: az annotált videó méretaránya (pl. 0.5 = 960x540)
        "roi_top_pct": roi_top_pct, "roi_bottom_pct": roi_bottom_pct, "min_player_height_pct": min_player_height_pct,
        "use_static_roi": use_static_roi, "grass_min_ratio": grass_min_ratio,
        "stationary_window_sec": stationary_window_sec, "stationary_max_move": stationary_max_move,
        "grass_lower": [int(x) for x in grass_hsv_lower.split(',')], "grass_upper": [int(x) for x in grass_hsv_upper.split(',')],
        "ref_lower": [int(x) for x in ref_hsv_lower.split(',')], "ref_upper": [int(x) for x in ref_hsv_upper.split(',')],
        "team1_lower": [int(x) for x in team1_hsv_lower.split(',')], "team1_upper": [int(x) for x in team1_hsv_upper.split(',')],
        "team2_lower": [int(x) for x in team2_hsv_lower.split(',')], "team2_upper": [int(x) for x in team2_hsv_upper.split(',')],
        "keeper1_lower": [int(x) for x in keeper1_hsv_lower.split(',')], "keeper1_upper": [int(x) for x in keeper1_hsv_upper.split(',')],
        "keeper2_lower": [int(x) for x in keeper2_hsv_lower.split(',')], "keeper2_upper": [int(x) for x in keeper2_hsv_upper.split(',')]
    }
    loop = asyncio.get_running_loop()
    url, local, events_path = await loop.run_in_executor(
        executor, cut_and_analyze_sync, video_path, start_sec, end_sec, dev_mode, detect_mode, calib_points, config
    )
    return {"status": "success", "clip_url": url, "local_path": local, "events_path": events_path}

@app.post("/analyze-click-ai")
async def analyze_click_ai_endpoint(
    image_url: str = Form(...),
    ui_x: float = Form(...),
    ui_y: float = Form(...),
    ui_w: float = Form(...),
    ui_h: float = Form(...),
    existing_lower: str = Form(""),
    existing_upper: str = Form("")
):
    filepath = os.path.join(CLIPS_DIR, image_url.split('/')[-1])
    img = cv2.imread(filepath)
    if img is None:
        return {"status": "error"}

    real_h, real_w = img.shape[:2]
    real_x, real_y = int((ui_x / ui_w) * real_w), int((ui_y / ui_h) * real_h)

    x1, y1 = max(0, real_x - 25), max(0, real_y - 25)
    x2, y2 = min(real_w, real_x + 25), min(real_h, real_y + 25)
    crop = img[y1:y2, x1:x2]

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_m, s_m, v_m = cv2.mean(hsv_crop)[:3]

    cur_low = np.array([max(0, int(h_m - 12)), max(20, int(s_m - 50)), max(20, int(v_m - 50))])
    cur_upp = np.array([min(179, int(h_m + 12)), min(255, int(s_m + 50)), min(255, int(v_m + 50))])

    if existing_lower and existing_upper:
        try:
            ex_low = np.array([int(x) for x in existing_lower.split(',')])
            ex_upp = np.array([int(x) for x in existing_upper.split(',')])
            cur_low = np.minimum(cur_low, ex_low)
            cur_upp = np.maximum(cur_upp, ex_upp)
        except Exception:
            pass

    box_rect = {
        "x": (x1 / real_w) * ui_w,
        "y": (y1 / real_h) * ui_h,
        "w": ((x2 - x1) / real_w) * ui_w,
        "h": ((y2 - y1) / real_h) * ui_h
    }

    return {
        "status": "success",
        "lower": f"{cur_low[0]},{cur_low[1]},{cur_low[2]}",
        "upper": f"{cur_upp[0]},{cur_upp[1]},{cur_upp[2]}",
        "ui_box": box_rect
    }

@app.post("/get-frame-time")
async def get_frame_time_endpoint(video_path: str = Form(...), time_sec: float = Form(0.0)):
    if not os.path.exists(video_path):
        return {"status": "error"}
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * (cap.get(cv2.CAP_PROP_FPS) or 25.0)))
    ret, frame = cap.read()
    cap.release()
    if ret:
        filename = f"calib_{uuid.uuid4().hex[:6]}.jpg"
        cv2.imwrite(os.path.join(CLIPS_DIR, filename), frame)
        return {"status": "success", "frame_url": f"/clips/{filename}"}
    return {"status": "error"}

# Az ellenőrző (review) lapok ezekhez a kalapokhoz nem készülnek (túl sok esemény, kis egyenkénti érték)
REVIEW_SKIP_CATEGORIES = ("passz", "labdavezetes")

GK_TIMELINE_TYPES = ("gk_save", "gk_claim_cross", "gk_goal_kick", "gk_long_ball", "gk_short_pass",
                     "gk_conceded", "gk_sweeper", "gk_positioning_error")
GK_TEAM_NAMES = {"team1": "Hazai Kapus", "team2": "Vendég Kapus"}

def _find_events_file(video_path, start_sec, end_sec):
    """A /cut-raw által készített legfrissebb, ugyanarra a videóra (és átfedő időszakra) vonatkozó events JSON."""
    if not video_path:
        return None
    try:
        candidates = [os.path.join(CLIPS_DIR, f) for f in os.listdir(CLIPS_DIR)
                      if f.startswith("events_") and f.endswith(".json")]
    except OSError:
        return None
    target = os.path.normcase(os.path.abspath(video_path))
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = json.load(f).get("source") or {}
        except (OSError, ValueError):
            continue
        if not src.get("video_path"):
            continue
        if os.path.normcase(os.path.abspath(src["video_path"])) != target:
            continue
        if start_sec is not None and end_sec is not None:
            if src.get("end_sec", end_sec) < start_sec or src.get("start_sec", start_sec) > end_sec:
                continue
        return path
    return None

@app.post("/detect-goalkeepers")
async def detect_goalkeepers_endpoint(
    video_path: str = Form(""),
    start_sec: Optional[float] = Form(None),
    end_sec: Optional[float] = Form(None),
    events_path: str = Form("")
):
    """A kapus eseményeket egy korábbi /cut-raw elemzés events JSON-jából adja vissza kapusonként (csapatonként) csoportosítva."""
    path = events_path if events_path and os.path.exists(events_path) else _find_events_file(video_path, start_sec, end_sec)
    if not path:
        return {"status": "success", "goalkeepers": [],
                "message": "Nem található elemzés ehhez a videóhoz. Előbb futtasd az elemzést (/cut-raw, fejlesztői módban)."}

    try:
        with open(path, "r", encoding="utf-8") as f:
            events = json.load(f).get("events", [])
    except (OSError, ValueError) as ex:
        return {"status": "success", "goalkeepers": [], "message": f"Az eseményfájl nem olvasható: {ex}"}

    groups = {}
    for ev in events:
        if ev.get("type") not in GK_TIMELINE_TYPES:
            continue
        t = float(ev.get("time", 0.0))
        if start_sec is not None and t < start_sec:
            continue
        if end_sec is not None and t > end_sec:
            continue
        item = {
            "start_sec": ev.get("start_sec", round(max(0.0, t - 4.0), 2)),
            "end_sec": ev.get("end_sec", round(t + 3.0, 2)),
            "type": ev.get("type"),
            "time": t,
            "desc": ev.get("desc", "")
        }
        if ev.get("keeper_track_id") is not None:
            item["keeper_track_id"] = ev["keeper_track_id"]
        groups.setdefault(ev.get("team") or "ismeretlen", []).append(item)

    goalkeepers = [
        {
            "name": GK_TEAM_NAMES.get(team, "Ismeretlen kapus"),
            "team": team,
            "events": sorted(groups[team], key=lambda e: e["time"])
        }
        for team in sorted(groups.keys(), key=lambda k: (k not in GK_TEAM_NAMES, k))
    ]

    result = {"status": "success", "goalkeepers": goalkeepers, "events_path": os.path.abspath(path)}
    if not goalkeepers:
        result["message"] = "Az elemzésben nem található kapus esemény."
    return result

@app.post("/burn-drawings")
async def burn_drawings_endpoint(
    clip_path: str = Form(...),
    freeze_timestamp_sec: float = Form(0.0),
    freeze_duration_sec: float = Form(3.0),
    drawings_json: str = Form(...)
):
    output_filename = f"burned_{uuid.uuid4().hex[:6]}.mp4"
    output_path = os.path.abspath(os.path.join(CLIPS_DIR, output_filename))

    def process_burn():
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        drawings = json.loads(drawings_json)
        freeze_frame_idx = int(freeze_timestamp_sec * fps)
        freeze_len_frames = int(freeze_duration_sec * fps)

        process = subprocess.Popen([
            "ffmpeg.exe", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-", *video_writer_encoder_args(choose_h264_encoder("ffmpeg.exe"), width, height), output_path
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # LGPL-barát kódolóválasztás

        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if f_idx == freeze_frame_idx:
                annotated = frame.copy()
                for d in drawings:
                    x1, y1 = int(d['x1'] * width), int(d['y1'] * height)
                    x2, y2 = int(d['x2'] * width), int(d['y2'] * height)
                    tool = d['tool']

                    if tool == 'line':
                        cv2.line(annotated, (x1, y1), (x2, y2), (7, 193, 255), 3)
                    elif tool == 'circle':
                        r = int(math.hypot(x2 - x1, y2 - y1))
                        cv2.circle(annotated, (x1, y1), r, (99, 30, 233), 3)
                    elif tool == 'arrow':
                        cv2.arrowedLine(annotated, (x1, y1), (x2, y2), (80, 175, 76), 3, tipLength=0.08)

                for _ in range(freeze_len_frames):
                    process.stdin.write(annotated.tobytes())

            process.stdin.write(frame.tobytes())
            f_idx += 1

        cap.release()
        process.stdin.close()
        process.wait()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, process_burn)

    return {"status": "success", "clip_url": f"/clips/{output_filename}", "local_path": output_path}

# ============================================================
# A 90 PERC AUTOMATIKUS KIVÁGÁSA — félidők felismerése
# (sípszó + képi aktivitás, lásd match_segmenter.py) és vágás
# ============================================================

_segmenter_person_model = None
_segmenter_person_lock = threading.Lock()

def _segmenter_person_counter(frame):
    """
    Létszám a mintaképkockán a félidő-felismeréshez. Külön (nano) YOLO példány,
    hogy ne zavarja a /cut-raw követését (a model.track állapota nem szálbiztos).
    """
    global _segmenter_person_model
    if OWN_BACKEND is not None:   # DETEKTOR-HÁTTÉR: a saját detektor (állapotmentes, követés nélkül)
        return OWN_BACKEND.count_role(frame, ("player", "goalkeeper"), conf=0.3)
    with _segmenter_person_lock:
        if _segmenter_person_model is None:
            _segmenter_person_model = YOLO(model_manager.resolve_for_load('yolov8n.pt'))
        results = _segmenter_person_model.predict(frame, classes=[0], conf=0.3, device=detect_device(), verbose=False)
    return int(len(results[0].boxes)) if results else 0

def _detect_halves_sync(video_path, expected_half_min, use_person_count):
    return detect_match_halves(
        video_path,
        expected_half_min=expected_half_min,
        person_counter=_segmenter_person_counter if use_person_count else None,
    )

@app.post("/detect-halves")
async def detect_halves_endpoint(
    video_path: str = Form(...),
    expected_half_min: float = Form(45.0),
    use_person_count: bool = Form(False)
):
    if not os.path.exists(video_path):
        return {"status": "error", "message": "A videófájl nem található."}
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _detect_halves_sync, video_path, expected_half_min, use_person_count)
    except Exception as ex:
        traceback.print_exc()
        return {"status": "error", "message": f"Félidő-felismerési hiba: {ex}"}
    return {"status": "success", **result}

@app.post("/cut-match")
async def cut_match_endpoint(
    video_path: str = Form(...),
    expected_half_min: float = Form(45.0),
    first_start: str = Form(""),
    first_end: str = Form(""),
    second_start: str = Form(""),
    second_end: str = Form(""),
    padding_sec: float = Form(5.0),
    use_person_count: bool = Form(False)
):
    if not os.path.exists(video_path):
        return {"status": "error", "message": "A videófájl nem található."}
    overrides = [parse_time_value(v) for v in (first_start, first_end, second_start, second_end)]

    def process_cut():
        notes = []
        detection = None
        if any(v is None for v in overrides):
            detection = _detect_halves_sync(video_path, expected_half_min, use_person_count)
            notes += detection.get("notes", [])
        else:
            notes.append("Kézzel megadott határok – automatikus felismerés nélkül.")
        fh = dict((detection or {}).get("first_half") or {})
        sh = dict((detection or {}).get("second_half") or {})
        for half, keys, vals in ((fh, ("start_sec", "end_sec"), overrides[:2]),
                                 (sh, ("start_sec", "end_sec"), overrides[2:])):
            for k, v in zip(keys, vals):
                if v is not None:
                    half[k] = v
        halves = {}
        for key, half in (("first_half", fh), ("second_half", sh)):
            if "start_sec" in half and "end_sec" in half and half["end_sec"] > half["start_sec"]:
                halves[key] = {"start_sec": round(float(half["start_sec"]), 2),
                               "end_sec": round(float(half["end_sec"]), 2),
                               "start_hms": format_hms(half["start_sec"]),
                               "end_hms": format_hms(half["end_sec"])}
            elif half:
                notes.append(f"{'Az 1.' if key == 'first_half' else 'A 2.'} félidő határai hibásak, kihagyva.")
        if not halves:
            raise RuntimeError("Nincs érvényes félidő a vágáshoz.")
        duration = detection.get("duration_sec") if detection else None
        paths = cut_match_halves(video_path, halves, CLIPS_DIR, padding_sec=padding_sec, duration_sec=duration)
        return halves, paths, notes, (detection or {}).get("confidence")

    loop = asyncio.get_running_loop()
    try:
        halves, paths, notes, confidence = await loop.run_in_executor(executor, process_cut)
    except Exception as ex:
        traceback.print_exc()
        return {"status": "error", "message": f"Meccs vágási hiba: {ex}"}
    paths = dict(paths)
    cut_modes = paths.pop("cut_modes", None)  # képkocka-pontos smart cut: copy / smart / full

    def to_url(p):
        return f"/clips/{os.path.basename(p)}" if p else None

    return {
        "status": "success",
        "halves": halves,
        "confidence": confidence,
        "notes": notes,
        "first_half_url": to_url(paths["first_half"]),
        "second_half_url": to_url(paths["second_half"]),
        "full_match_url": to_url(paths["full_match"]),
        "local_paths": paths,
        "cut_modes": cut_modes,
    }


# ============================================================
# HŐTÉRKÉP ÉS KLIPEXPORT
# ============================================================

@app.post("/heatmap")
async def heatmap_endpoint(
    events_path: str = Form(...),
    team: str = Form("all"),
    player_id: Optional[int] = Form(None),
    flip_second_half_at_sec: Optional[float] = Form(None)
):
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            executor, lambda: generate_heatmaps_from_file(
                events_path, CLIPS_DIR, team=team, player_id=player_id, flip_at_sec=flip_second_half_at_sec))
    except Exception as ex:
        traceback.print_exc()
        return {"status": "error", "message": f"Hőtérkép hiba: {ex}"}

@app.post("/export-clips")
async def export_clips_endpoint(
    video_path: str = Form(...),
    events_json: str = Form(...),
    pre_sec: float = Form(DEFAULT_PRE_SEC),
    post_sec: float = Form(DEFAULT_POST_SEC),
    merge: bool = Form(True),
    accurate: bool = Form(True),
    title: Optional[str] = Form(None)
):
    try:
        items = json.loads(events_json) if events_json else []
    except ValueError:
        return {"status": "error", "message": "Az events_json nem érvényes JSON."}
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            executor, lambda: export_clips(video_path, items, CLIPS_DIR, pre_sec=pre_sec, post_sec=post_sec,
                                           merge=merge, accurate=accurate, title=title))
    except Exception as ex:
        traceback.print_exc()
        return {"status": "error", "message": f"Klipexport hiba: {ex}"}

# ============================================================
# TAKTIKAI TÁBLA — böngészős felület (felállás, rajzolás, felülnézeti visszajátszás)
# Megnyitás: http://localhost:8000/tabla  vagy  /tabla?events=/clips/events_<id>.json
# ============================================================

TACTICS_BOARD_PATH = app_paths.resource_path("static", "taktikai_tabla.html")  # csomagolva: sys._MEIPASS

@app.get("/tabla", response_class=HTMLResponse)
async def tactics_board_page():
    with open(TACTICS_BOARD_PATH, encoding="utf-8") as f:
        body = f.read()
    return HTMLResponse(
        "<!doctype html><html lang=\"hu\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
        "<style>[hidden]{display:none!important}body{margin:0}</style></head><body>"
        + body + "</body></html>"
    )


# Mentett táblák a szerveren (clips/boards/<id>.json) – bármelyik eszközről elérhetők. Lásd boards.py
from fastapi import Request as _BoardRequest
from fastapi.responses import JSONResponse as _BoardJSON
from boards import BoardStore, BoardError, parse_request_body

board_store = BoardStore(app_paths.boards_dir())  # fejlesztői módban clips/boards


def _board_error(ex):
    return _BoardJSON(status_code=ex.status, content={"status": "error", "message": ex.message})


@app.get("/boards")
async def list_boards_endpoint(kind: Optional[str] = None):
    """Mentett táblák listája: [{id, name, kind, updated_at, ...}], a legfrissebb elöl."""
    return {"status": "success", "boards": board_store.list(kind=kind or None)}


@app.get("/boards/{board_id}")
async def get_board_endpoint(board_id: str):
    try:
        return {"status": "success", "board": board_store.get(board_id)}
    except BoardError as ex:
        return _board_error(ex)


@app.post("/boards")
async def save_board_endpoint(request: _BoardRequest):
    """Törzs (JSON): {"name", "data", "id"?: felülíráshoz, "kind"?: "board" | "replay"}"""
    try:
        name, data, board_id, kind = parse_request_body(await request.body())
        return {"status": "success", "board": board_store.save(name, data, board_id=board_id, kind=kind)}
    except BoardError as ex:
        return _board_error(ex)


@app.delete("/boards/{board_id}")
async def delete_board_endpoint(board_id: str):
    try:
        board_store.delete(board_id)
        return {"status": "success"}
    except BoardError as ex:
        return _board_error(ex)


# Videó elemzés (a tábla 3. módja): a szerveren lévő videók listája és a rajzos export. Lásd video_annotate.py
import video_annotate
from boards import slugify as _board_slugify


@app.get("/videos")
async def list_videos_endpoint():
    """{videos: [{name, url, size, mtime, kind, fps?, duration_sec?, events_url?}], events: [{name, url, ...}]}"""
    loop = asyncio.get_running_loop()
    listing = await loop.run_in_executor(None, video_annotate.list_videos, CLIPS_DIR)
    return {"status": "success", **listing}


def _job_burn_stops(progress, video_path, stops, opts):
    name = _board_slugify(opts.get("title") or os.path.splitext(os.path.basename(video_path))[0])
    out_name = f"rajzos_{name}_{uuid.uuid4().hex[:6]}.mp4"
    out_path = os.path.abspath(os.path.join(CLIPS_DIR, out_name))
    try:
        res = video_annotate.burn_stops(
            video_path, stops, out_path, default_hold=opts["default_hold_sec"],
            start_sec=opts.get("start_sec"), end_sec=opts.get("end_sec"),
            progress_cb=lambda done, total, label: progress(100.0 * done / max(1, total), label))
    except (ValueError, RuntimeError) as ex:
        return {"status": "error", "message": str(ex)}
    return {**res, "url": f"/clips/{out_name}", "name": out_name}


@app.post("/burn-stops")
async def burn_stops_endpoint(request: _BoardRequest):
    """
    Rajzos videóexport háttérfeladatként. Törzs (JSON):
      {"video_url": "/clips/x.mp4", "stops": [{t, items, hold_sec?, note?}], "title"?,
       "default_hold_sec"?: 3, "start_sec"?, "end_sec"?: csak a részlet}
    Válasz: {"status": "success", "job_id", "status_url"}; a kész job eredménye: {url, duration_sec, ...}.
    """
    try:
        body = json.loads((await request.body()).decode("utf-8") or "null")
    except (UnicodeDecodeError, ValueError):
        return _form_error("A kérés törzse nem érvényes JSON.")
    if not isinstance(body, dict):
        return _form_error("A kérés törzse JSON objektum legyen.")
    try:
        video_path = video_annotate.resolve_clip_url(body.get("video_url"), CLIPS_DIR)
        num = lambda k: (None if body.get(k) in (None, "") else float(body.get(k)))
        opts = {"title": str(body.get("title") or "")[:80], "default_hold_sec": num("default_hold_sec") or 3.0,
                "start_sec": num("start_sec"), "end_sec": num("end_sec")}
        stops = body.get("stops") or []
        info = video_annotate.probe_video(video_path) or {}
        ranged = opts["start_sec"] is not None or opts["end_sec"] is not None
        video_annotate.normalize_stops(stops, info.get("fps") or 25.0, info.get("frames"),
                                       allow_empty=ranged)  # gyors ellenőrzés a job indítása előtt
    except (ValueError, TypeError) as ex:
        return _form_error(str(ex))
    params = {"video_url": body.get("video_url"), "stops": len(stops), **opts}
    job_id = job_manager.submit("burn-stops", _job_burn_stops, video_path, stops, opts, params=params)
    return {"status": "success", "job_id": job_id, "status_url": f"/jobs/{job_id}"}


# ============================================================
# HÁTTÉRFELADATOK (jobok) — a hosszú munkák nem blokkolják a kérést.
# A kliens job azonosítót kap, és a GET /jobs/{id} végponton 1–2 mp-enként
# lekérdezi a haladást. Lásd jobs.py, pipeline.py, docs/HATTERFELADATOK.md.
# A fenti szinkron végpontok változatlanul működnek.
# ============================================================
import sys
from fastapi import Request
from fastapi.responses import JSONResponse
import pipeline
from jobs import JobManager


# ============================================================
# EDZŐI VISSZAJELZÉS az automatikus eseményekre (✓ valódi / ✗ téves / + hiányzó) – feedback.py
#   GET  /feedback?events=/clips/match_events_x.json
#   POST /feedback  {"events": ..., "event": {...}, "verdict": "ok"|"hamis"|""}
#   POST /feedback  {"events": ..., "missing": {"type", "time", "team", "note"}}
#   POST /feedback  {"events": ..., "remove_missing": "<id>"}
# ============================================================
import feedback as _feedback


@app.get("/feedback")
async def get_feedback(events: str):
    try:
        return {"status": "success", "feedback": _feedback.load(CLIPS_DIR, events)}
    except ValueError as ex:
        return JSONResponse({"status": "error", "message": str(ex)}, status_code=400)


@app.post("/feedback")
async def post_feedback(request: Request):
    try:
        body = await request.json()
        events = body.get("events")
        if body.get("event") is not None:
            data = _feedback.set_verdict(CLIPS_DIR, events, body["event"], body.get("verdict") or None)
            return {"status": "success", "feedback": data}
        if body.get("missing") is not None:
            m = body["missing"]
            item = _feedback.add_missing(CLIPS_DIR, events, m.get("type"), float(m.get("time", 0.0)),
                                         m.get("team"), m.get("note", ""))
            return {"status": "success", "item": item}
        if body.get("remove_missing"):
            return {"status": "success", "feedback": _feedback.remove_missing(CLIPS_DIR, events, body["remove_missing"])}
        raise ValueError("Hiányzik: event + verdict, missing vagy remove_missing.")
    except (ValueError, TypeError) as ex:
        return JSONResponse({"status": "error", "message": str(ex)}, status_code=400)


# KAPUS MODUL: a választott csapat kapusát érintő jelenetek egy meccs-eredményből (keeper_focus.py)
#   GET /keeper-focus?events=/clips/match_events_x.json&team=team1&moves=1&min_move=2
import keeper_focus as _keeper_focus


@app.get("/keeper-focus")
async def get_keeper_focus(events: str, team: str = "team1", moves: int = 1, min_move: float = 2.0):
    try:
        name = os.path.basename(str(events or "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name):
            raise ValueError("Érvénytelen eseményfájl-név.")
        path = os.path.join(CLIPS_DIR, name)
        if not os.path.isfile(path):
            return JSONResponse({"status": "error", "message": "Nincs ilyen eredményfájl."}, status_code=404)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        res = _keeper_focus.build(data, team, min_move_m=max(0.5, float(min_move)), include_moves=bool(moves))
        return {"status": "success", **res}
    except (ValueError, TypeError) as ex:
        return JSONResponse({"status": "error", "message": str(ex)}, status_code=400)


job_manager = JobManager(app_paths.jobs_dir(), max_workers=2)  # fejlesztői módban clips/jobs


def _job_funcs():
    # A már betöltött main modult adjuk át (python main.py esetén __main__), hogy ne töltődjön be újra
    return pipeline.default_funcs(sys.modules[__name__])


async def _form_dict(request: Request):
    form = await request.form()
    return {k: v for k, v in form.items() if isinstance(v, str)}


def _form_error(message, code=400):
    return JSONResponse(status_code=code, content={"status": "error", "message": message})


def _background_opts(raw):
    """
    Háttér-mód a form mezőkből: background (vagy hatter) = true → alacsony prioritás + kevés szál,
    threads (vagy szalak) = N → szálkorlát (háttér-módban alapból a magok fele, akkun a negyede).
    Hibás értéknél ValueError.
    """
    bg_raw = raw.get("background", raw.get("hatter", ""))
    bg = pipeline._to_bool(bg_raw) if str(bg_raw).strip() else False
    th_raw = str(raw.get("threads", raw.get("szalak", "")) or "").strip()
    threads = None
    if th_raw:
        try:
            threads = int(th_raw)
        except ValueError:
            raise ValueError(f"Érvénytelen szálszám: {th_raw!r}")
        if threads < 1:
            raise ValueError("A szálszám legalább 1.")
    return {"background": bg, "threads": threads}


def _run_match_job(progress, video_path, opts):
    return pipeline.run_match_pipeline(video_path, opts, progress=progress, funcs=_job_funcs())


@app.post("/jobs/analyze-match")
async def analyze_match_job_endpoint(request: Request):
    """A teljes meccs elemzése egy háttérfeladatként (félidők → vágás → elemzés → hőtérkép → klipek)."""
    raw = await _form_dict(request)
    video_path = raw.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        return _form_error("A videófájl nem található.")
    try:
        opts = pipeline.coerce_options(raw)
        pipeline.parse_compilations(opts["compilations"])
        pipeline.build_analysis_config(opts)  # a HSV mezők ellenőrzése még a job indítása előtt
        bg = _background_opts(raw)
    except ValueError as ex:
        return _form_error(str(ex))
    params = {"video_path": video_path, **{k: opts[k] for k in pipeline.PIPELINE_DEFAULTS},
              "calib_points": opts["calib_points"], "team1_attacks_right": opts["team1_attacks_right"]}
    job_id = job_manager.submit("analyze-match", _run_match_job, video_path, opts, params=params, **bg)
    return {"status": "success", "job_id": job_id, "status_url": f"/jobs/{job_id}"}


def _job_cut_raw(progress, raw):
    url, local, events_path = cut_and_analyze_sync(
        raw.get("video_path", ""), float(raw.get("start_sec", 0.0)), float(raw.get("end_sec", 30.0)),
        pipeline._to_bool(raw.get("dev_mode", True)), raw.get("detect_mode", "team"),
        raw.get("calib_points", ""), pipeline.build_analysis_config(raw), progress=progress)
    return {"status": "success", "clip_url": url, "local_path": local, "events_path": events_path}


def _job_detect_halves(progress, raw):
    result = detect_match_halves(
        raw["video_path"], expected_half_min=float(raw.get("expected_half_min", 45.0)),
        person_counter=_segmenter_person_counter if pipeline._to_bool(raw.get("use_person_count", False)) else None,
        progress_cb=pipeline._segmenter_cb(progress))
    return {"status": "success", **result}


def _job_cut_match(progress, raw):
    opts = pipeline.coerce_options(raw)
    notes = []
    halves, detection = pipeline.resolve_halves({**opts, "video_path": raw["video_path"]}, _job_funcs(),
                                                progress.sub(0, 90, "Félidők felismerése"), notes)
    progress(90.0, "Vágás")
    duration = (detection or {}).get("duration_sec")
    to_url = lambda p: f"/clips/{os.path.basename(p)}" if p else None
    done = {}

    def on_part(key, path):
        # a kész félidő már a job futása közben elérhető (részeredmény)
        done[f"{key}_url"] = to_url(path)
        progress.publish({"status": "success", "reszleges": True, "halves": halves, **done,
                          "kesz_reszek": [k[:-4] for k in done]})
        progress.check_cancel()
        progress.check_pause()  # szüneteltetés / megszakítás a következő fájl előtt

    paths = dict(cut_match_halves(raw["video_path"], halves, CLIPS_DIR, padding_sec=opts["padding_sec"],
                                  duration_sec=duration, on_part=on_part))
    cut_modes = paths.pop("cut_modes", None)
    return {"status": "success", "halves": halves, "confidence": (detection or {}).get("confidence"),
            "notes": notes, "first_half_url": to_url(paths["first_half"]),
            "second_half_url": to_url(paths["second_half"]), "full_match_url": to_url(paths["full_match"]),
            "local_paths": paths, "cut_modes": cut_modes}


def _job_heatmap(progress, raw):
    player_id = raw.get("player_id")
    flip = raw.get("flip_second_half_at_sec")
    return generate_heatmaps_from_file(
        raw.get("events_path", ""), CLIPS_DIR, team=raw.get("team", "all"),
        player_id=int(player_id) if player_id not in (None, "") else None,
        flip_at_sec=float(flip) if flip not in (None, "") else None)


def _job_export_clips(progress, raw):
    try:
        items = json.loads(raw.get("events_json") or "[]")
    except ValueError:
        return {"status": "error", "message": "Az events_json nem érvényes JSON."}
    return export_clips(
        raw.get("video_path", ""), items, CLIPS_DIR,
        pre_sec=float(raw.get("pre_sec", DEFAULT_PRE_SEC)), post_sec=float(raw.get("post_sec", DEFAULT_POST_SEC)),
        merge=pipeline._to_bool(raw.get("merge", True)), accurate=pipeline._to_bool(raw.get("accurate", True)),
        title=raw.get("title") or None,
        progress_cb=lambda done, total: progress(100.0 * done / max(1, total), f"{done}/{total} klip"))


JOB_KINDS = {
    "cut-raw": _job_cut_raw,
    "detect-halves": _job_detect_halves,
    "cut-match": _job_cut_match,
    "heatmap": _job_heatmap,
    "export-clips": _job_export_clips,
}


@app.post("/jobs/run")
async def run_job_endpoint(request: Request):
    """A régi végpontok háttérfeladatos változata: kind=cut-raw|detect-halves|cut-match|heatmap|export-clips + ugyanazok a form mezők."""
    raw = await _form_dict(request)
    kind = raw.get("kind", "")
    if kind not in JOB_KINDS:
        return _form_error(f"Ismeretlen feladattípus: {kind!r} (lehetséges: {', '.join(JOB_KINDS)})")
    if kind in ("cut-raw", "detect-halves", "cut-match", "export-clips") and not os.path.exists(raw.get("video_path", "")):
        return _form_error("A videófájl nem található.")
    try:
        if kind == "cut-raw":
            pipeline.build_analysis_config(raw)
        elif kind == "cut-match":
            pipeline.coerce_options(raw)
        bg = _background_opts(raw)
    except ValueError as ex:
        return _form_error(str(ex))
    params = {k: v for k, v in raw.items() if k != "events_json"}
    job_id = job_manager.submit(kind, JOB_KINDS[kind], raw, params=params, **bg)
    return {"status": "success", "job_id": job_id, "status_url": f"/jobs/{job_id}"}


@app.get("/jobs")
async def list_jobs_endpoint(limit: int = 50):
    return {"status": "success", "jobs": job_manager.list(limit=limit)}


@app.get("/jobs/{job_id}")
async def get_job_endpoint(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        return _form_error("Nincs ilyen feladat.", 404)
    return {"status": "success", "job": job}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str):
    job = job_manager.cancel(job_id)
    if job is None:
        return _form_error("Nincs ilyen feladat.", 404)
    return {"status": "success", "job": job}


@app.post("/jobs/{job_id}/pause")
async def pause_job_endpoint(job_id: str):
    """Szüneteltetés (kooperatív: a feladat a következő haladásjelzésnél áll meg, állapot: paused)."""
    job = job_manager.pause(job_id)
    if job is None:
        return _form_error("Nincs ilyen feladat.", 404)
    return {"status": "success", "job": job}


@app.post("/jobs/{job_id}/resume")
async def resume_job_endpoint(job_id: str):
    """A szüneteltetett feladat folytatása."""
    job = job_manager.resume(job_id)
    if job is None:
        return _form_error("Nincs ilyen feladat.", 404)
    return {"status": "success", "job": job}


# ============================================================
# MOTOR-VEZÉRLÉS az asztali alkalmazásnak: GET /health, GET /version, POST /shutdown (csak helyi
# gépről), GET /models/status, POST /models/download, POST /models/load. Lásd engine_control.py,
# packaging/README_CSOMAGOLAS.md
# ============================================================
import engine_control
engine_control.install_endpoints(app, sys.modules[__name__])


# ============================================================
# AUTH VÉGPONTOK — eddig hiányoztak, pedig az AuthService.cs
# már hívta őket (/auth/register, /auth/login)
# ============================================================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@app.post("/auth/register", response_model=TokenResponse)
async def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ez az email cím már regisztrálva van.")

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.id})
    return TokenResponse(access_token=token)

@app.post("/auth/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Az AuthService.cs LoginAsync "username" mezőben küldi az emailt (OAuth2 form-konvenció)
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Hibás email cím vagy jelszó.")

    token = create_access_token({"sub": user.id})
    return TokenResponse(access_token=token)


# ============================================================
# MECCS MENTÉS / LEKÉRÉS — eddig hiányoztak, pedig a
# MatchService.cs már hívta őket (/matches/save, /matches/{id})
# ============================================================

class MatchSavePayload(BaseModel):
    match_id: str
    video_url: str = ""
    frames: List[dict] = []
    title: Optional[str] = None

@app.post("/matches/save")
async def save_match(
    payload: MatchSavePayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    frames_json = json.dumps(payload.frames, ensure_ascii=False)
    match = db.query(Match).filter(Match.id == payload.match_id, Match.user_id == current_user.id).first()

    if match:
        match.video_url = payload.video_url
        match.analysis_data = frames_json
        match.updated_at = datetime.utcnow()
        if payload.title:
            match.title = payload.title
    else:
        match = Match(
            id=payload.match_id,
            title=payload.title or "Névtelen meccs",
            user_id=current_user.id,
            video_url=payload.video_url,
            analysis_data=frames_json,
        )
        db.add(match)

    db.commit()
    return {"status": "success", "match_id": match.id}

@app.get("/matches/{match_id}")
async def get_match(
    match_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    match = db.query(Match).filter(Match.id == match_id, Match.user_id == current_user.id).first()
    if not match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="A meccs nem található.")

    return {
        "match_id": match.id,
        "video_url": match.video_url or "",
        "frames": json.loads(match.analysis_data) if match.analysis_data else []
    }


if __name__ == "__main__":
    # Fejlesztői indítás (mint eddig: minden hálózati címen, 8000-es port). Az asztali alkalmazás
    # az engine_launcher.py-t használja (csak 127.0.0.1, szabad port, READY sor, naplófájl).
    engine_control.serve(app, host="0.0.0.0", port=8000)
import cv2
import os
import uuid
import json
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
import torch
from ultralytics import YOLO

from database import get_db, engine
from models import Base, User, Match
from tactical_engine import TacticalEngine
from pitch_filters import (build_grass_mask, foot_on_grass, CameraMotion,
                           StationaryFilter, extract_dominant_color)
from model_config import (load_model_config, resolve_weights, build_role_class_map,
                          describe_role_map, should_detect, reset_tracker)
from review_export import ReviewExporter
from match_segmenter import (detect_match_halves, cut_match_halves,
                             parse_time_value, format_hms)
from auth import hash_password, verify_password, create_access_token, get_current_user
from heatmap import TrackMetaCollector, compact_trajectories, generate_heatmaps_from_file
from clip_export import export_clips, DEFAULT_PRE_SEC, DEFAULT_POST_SEC
from team_classifier import TeamClassifier, auto_teams_enabled
from pitch_tracker import PitchTracker, parse_calib_points, draw_pitch_overlay as render_pitch_overlay

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Tactical Analyzer Pro Engine")
CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
executor = ThreadPoolExecutor(max_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_cfg = load_model_config()
model_weights = resolve_weights(model_cfg, device)
if model_cfg.get("weights") and not os.path.exists(model_weights):
    print(f"⚠️ A beállított súlyfájl nem található: {model_weights} (lásd docs/MODELL_BEALLITAS.md)")
model = YOLO(model_weights)
# Szerepkör -> osztály id-k a modell osztálynevei alapján (COCO: person / sports ball,
# focis modell: player / goalkeeper / referee / ball)
role_classes = build_role_class_map(model.names, model_cfg.get("class_names"))
if not role_classes["player"] and not role_classes["goalkeeper"]:
    print("⚠️ A modellben nincs játékos osztály (player / person) – állítsd be a class_names kulcsot!")
DETECT_EVERY_N = model_cfg["detect_every_n"]
BALL_CONF = model_cfg["ball_conf"]
# A megosztott modellt (és a benne tárolt ByteTrack állapotot) egyszerre csak egy elemzés
# használhatja. Kompromisszum: a párhuzamos elemzések sorban várnak egymásra, cserébe nem kell
# elemzésenként külön modellt betölteni (a yolov8x több száz MB GPU memória és lassú betöltés),
# és a követő azonosítói nem keverednek két klip között.
model_lock = threading.Lock()
print(f"🔥 Tactical AI Motor aktív: {device.upper()}, modell: {model_weights}")
print(f"   Osztályok: {describe_role_map(role_classes, model.names)}")
if DETECT_EVERY_N > 1:
    print(f"   Gyors mód: felismerés minden {DETECT_EVERY_N}. képkockán")

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
    tracker = PitchTracker((width, height), np.array(config["grass_lower"]), np.array(config["grass_upper"]))
    if H0 is None:
        return tracker, "BE, kézi kalibráció nincs -> automatikus kalibráció a pályavonalakból"
    msg = "BE, kézi kalibráció a klip elején"
    calib_t = config.get("calib_time_sec")
    k = int(round((calib_t - start_sec) * fps)) if calib_t is not None else 0
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
    clip_id = str(uuid.uuid4())[:8]
    output_filename = f"raw_{clip_id}.mp4"
    output_path = os.path.abspath(os.path.join(CLIPS_DIR, output_filename))
    events_path = os.path.abspath(os.path.join(CLIPS_DIR, f"events_{clip_id}.json"))
    debug_log_path = os.path.abspath(os.path.join(CLIPS_DIR, f"analysis_debug_{clip_id}.log"))
    temp_cut = os.path.abspath(os.path.join(CLIPS_DIR, f"temp_cut_{clip_id}.mp4"))

    subprocess.run([
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
    if calib_points_str:
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

    # --- Pályakövetés: képkockánkénti homográfia mozgó (Veo) kamerához (pitch_tracker.py) ---
    pitch_tracker, pitch_setup_msg = create_pitch_tracker(temp_cut, width, height, fps, start_sec,
                                                          calib_points_str, H, config)
    pitch_stats = {"ok": 0, "elveszett": 0, "kalibráció szükséges": 0, "finomítás": 0}
    pitch_boxes = []   # az előző felismerés dobozai (a követés ezeket kitakarja)
    H_cur = H
    # --- /Pályakövetés ---

    colors = {k: (np.array(config[f"{k}_lower"]), np.array(config[f"{k}_upper"])) 
              for k in ["grass", "ref", "team1", "team2", "keeper1", "keeper2"]}

    # Megemelt küszöbértékek a nézők végleges kizárásához
    roi_top_pct = max(config["roi_top_pct"], 0.28) # Felső horizont lejjebb hozva
    min_h = max(height * config["min_player_height_pct"], 50.0) # Min. 50px magas játékos

    grass_lower, grass_upper = colors["grass"]
    camera_motion = CameraMotion()
    stationary_filter = StationaryFilter(fps, config["stationary_window_sec"], config["stationary_max_move"])

    # --- AUTO CSAPAT (team_classifier.py): csak ha nincs kézi csapat-HSV tartomány ---
    team_clf = None
    if auto_teams_enabled(config):
        team_clf = TeamClassifier(grass_lower=grass_lower, grass_upper=grass_upper,
                                  swap_teams=config.get("swap_teams", False))
    # --- AUTO CSAPAT VÉGE ---

    process = subprocess.Popen([
        "ffmpeg.exe", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", output_path
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames_done = 0
    timeline_events = []

    # Áttekintő lapok (esemény-lapok + percenkénti áttekintés) – kevés képből ellenőrizhető az elemzés
    review_dir = os.path.abspath(os.path.join(CLIPS_DIR, f"review_{clip_id}"))
    review = ReviewExporter(review_dir, fps) if config.get("review_export", True) else None
    fed_engine_events = 0
    fed_timeline_events = 0

    def feed_review_events():
        nonlocal fed_engine_events, fed_timeline_events
        engine_events = engine_instance.event_detector.events
        for ev in engine_events[fed_engine_events:]:
            review.add_event(ev)
        fed_engine_events = len(engine_events)
        for ev in timeline_events[fed_timeline_events:]:
            review.add_event(ev)
        fed_timeline_events = len(timeline_events)

    track_meta = TrackMetaCollector()  # követésenkénti csapat / kapus (a mentett pályákhoz)
    # Gyors mód: a kihagyott képkockákon az utolsó felismerés eredményét rajzoljuk ki újra,
    # így a kimeneti videóban minden képkocka megmarad
    last_valid_players = []
    last_referee_boxes = []
    last_ball_pixels = []

    # A megosztott modellt / követőt az egész klip idejére lefoglaljuk (lásd model_lock),
    # és töröljük az előző elemzésből megmaradt ByteTrack állapotot
    with model_lock, open(debug_log_path, "w", encoding="utf-8") as debug_file:
        trackers_reset = reset_tracker(model)
        debug_file.write(f"=== TACTICAL ANALYZER RUN LOG: {clip_id} ===\n")
        debug_file.write(f"Videó felbontás: {width}x{height} @ {fps} FPS, Összes képkocka: {total_frames}\n")
        debug_file.write(f"Dinamikus Felső ROI Százalék: {roi_top_pct:.2f}, Min Magasság: {min_h:.1f}px\n")
        debug_file.write(f"Fix trapéz szűrő: {'BE' if config['use_static_roi'] else 'KI'}, "
                         f"Fű arány min.: {config['grass_min_ratio']:.2f}, "
                         f"Álló alak ablak: {config['stationary_window_sec']:.1f}s\n")
        debug_file.write(f"Modell: {model_weights}, osztályok: {describe_role_map(role_classes, model.names)}\n")
        debug_file.write(f"Felismerés minden {DETECT_EVERY_N}. képkockán, törölt követők: {trackers_reset}\n")
        debug_file.write(f"Pályakövetés: {pitch_setup_msg}\n\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_sec = start_sec + (frames_done / fps)
            camera_motion.update(frame)
            if frames_done % 250 == 0:
                stationary_filter.prune(frames_done)

            # --- Pályakövetés: az aktuális képkocka homográfiája (a rajzolás előtt) ---
            if pitch_tracker is not None:
                H_cur, pitch_q = pitch_tracker.update(frame, pitch_boxes)
                pitch_stats[pitch_q["status"]] = pitch_stats.get(pitch_q["status"], 0) + 1
                pitch_stats["finomítás"] += int(pitch_q.get("refined", False))
                if frames_done % 25 == 0:
                    debug_file.write(f"F#{frames_done} PÁLYA: {pitch_q['status']}, bizalom={pitch_q.get('conf')}, "
                                     f"inlier={pitch_q.get('inliers', '-')}, elveszett={pitch_q.get('lost')}, "
                                     f"forrás={pitch_q.get('source')}\n")
            # --- /Pályakövetés ---

            detected = should_detect(frames_done, DETECT_EVERY_N)
            gk_real_coords = None

            if detected:
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                grass_mask, grass_scale = build_grass_mask(hsv_frame, grass_lower, grass_upper)
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device, verbose=False)
                pitch_boxes = [tuple(map(int, b.xyxy[0])) for r in results for b in r.boxes]  # pályakövetéshez

                raw_detections = []
                referee_boxes = []
                ball_pixels = []
                ball_pixel = None

                for r in results:
                    for box in [b for b in r.boxes if int(b.cls[0]) in role_classes["ball"] and float(b.conf[0]) > BALL_CONF]:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        ball_pixel = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                        ball_pixels.append(ball_pixel)

                    # Bírók: nem kerülnek a játékosok közé, csak semleges színnel rajzoljuk ki őket
                    for box in [b for b in r.boxes if int(b.cls[0]) in role_classes["referee"]]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
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

                            is_k1 = check_color_match(hsv_frame, x1, y1, x2, y2, colors["keeper1"][0], colors["keeper1"][1])
                            is_k2 = check_color_match(hsv_frame, x1, y1, x2, y2, colors["keeper2"][0], colors["keeper2"][1])
                            is_keeper = is_k1 or is_k2 or model_keeper

                            # 5. Álló alak szűrő (néző, edző) – a kapus kivétel, ő sokáig állhat egy helyben
                            stab_x, stab_y = camera_motion.stabilize(foot_x, foot_y)
                            is_static = stationary_filter.update(tid, frames_done, stab_x, stab_y, box_h)
                            if is_static and not is_keeper and not (
                                    team_clf is not None and team_clf.is_keeper_track(tid, current_time_sec)):  # AUTO CSAPAT: kapus kivétel
                                debug_file.write(f"F#{frames_done} ID#{tid}: ELDOBVA -> Mozdulatlan alak ({config['stationary_window_sec']:.1f}s)\n")
                                continue

                            crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                            dominant_color = extract_dominant_color(crop, grass_lower, grass_upper)

                            # --- AUTO CSAPAT: mezszín-klaszterezés + követésenkénti szavazás ---
                            if team_clf is not None:
                                pitch_xy = engine_instance.transform_to_pitch(foot_x, foot_y, H_cur) if H_cur is not None else None
                                auto = team_clf.observe(tid, crop, current_time_sec, pitch_xy=pitch_xy,
                                                        model_keeper=model_keeper)
                                if (auto["is_referee"] or auto["role"] == "other") and not is_keeper:
                                    # bíró / egyéb mez (pl. kapus pályakoordináta nélkül): semleges keret
                                    referee_boxes.append((x1, y1, x2, y2))
                                    continue
                                is_keeper = is_keeper or auto["is_keeper"]
                                if is_k1 or is_k2:
                                    team_id = 1 if is_k1 else 2
                                else:
                                    # ismeretlen (bemelegítés eleje / kapus csapata nem dönthető el): szín szerinti tipp
                                    team_id = (auto["team_id"] or auto["color_team"]
                                               or (1 if dominant_color[0] > dominant_color[2] else 2))
                            # --- AUTO CSAPAT VÉGE ---
                            elif is_k1 or check_color_match(hsv_frame, x1, y1, x2, y2, colors["team1"][0], colors["team1"][1]):
                                team_id = 1
                            elif is_k2 or check_color_match(hsv_frame, x1, y1, x2, y2, colors["team2"][0], colors["team2"][1]):
                                team_id = 2
                            else:
                                team_id = 1 if dominant_color[0] > dominant_color[2] else 2

                            raw_detections.append({
                                'track_id': tid,
                                'team_id': team_id,
                                'screen_foot': (foot_x, foot_y),
                                'is_keeper': is_keeper,
                                'bbox': (x1, y1, x2, y2)
                            })

                if H_cur is not None:
                    valid_players, real_ball = engine_instance.update_tracks(frames_done, current_time_sec, raw_detections, ball_pixel, H_cur)
                else:
                    valid_players = raw_detections
                    real_ball = None

                last_valid_players, last_referee_boxes, last_ball_pixels = valid_players, referee_boxes, ball_pixels
                track_meta.add(valid_players)
            else:
                # Kihagyott képkocka (gyors mód): az előző felismerés újrarajzolása,
                # statisztika és esemény nem keletkezik belőle (különben duplán számolnánk)
                valid_players, referee_boxes, ball_pixels = last_valid_players, last_referee_boxes, last_ball_pixels
                real_ball = None

            if H_cur is not None and config.get("draw_pitch_overlay", True):
                render_pitch_overlay(frame, H_cur)  # halvány pályavonalak: látszik, mennyire pontos a kalibráció

            for bp in ball_pixels:
                cv2.circle(frame, bp, 5, (0, 255, 255), -1)

            for x1, y1, x2, y2 in referee_boxes:
                # Bíró: sárga keret fekete szegéllyel
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            for p in valid_players:
                x1, y1, x2, y2 = p['bbox']
                color = (0, 0, 255) if p['team_id'] == 1 else (255, 0, 0)
                if p['is_keeper']:
                    color = (0, 165, 255)
                    gk_real_coords = p.get('pitch_coords')

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"#{p['track_id']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if gk_real_coords and real_ball:
                gk_eval = engine_instance.analyze_goalkeeper(gk_real_coords, real_ball)
                if gk_eval and abs(gk_eval["bisector_offset_m"]) > 1.2:
                    if not timeline_events or (current_time_sec - timeline_events[-1]["time"]) > 3.0:
                        timeline_events.append({
                            "type": "gk_positioning_error",
                            "time": round(current_time_sec, 2),
                            "team": engine_instance.keeper_team_near(gk_real_coords),
                            "start_sec": round(max(0.0, current_time_sec - 4.0), 2),
                            "end_sec": round(current_time_sec + 3.0, 2),
                            "desc": f"Kapus helyezkedési hiba: {gk_eval['bisector_offset_m']} m eltolódás"
                        })

            if review is not None:
                review.add_frame(frames_done, current_time_sec, frame,
                                 valid_players if H_cur is not None else None, real_ball)
                feed_review_events()

            process.stdin.write(frame.tobytes())
            frames_done += 1
            if frames_done % 25 == 0:
                print(f"Feldolgozva: {frames_done}/{total_frames} képkocka ({(frames_done/total_frames)*100:.1f}%)")
                if progress: progress(100.0 * frames_done / max(1, total_frames), f"{frames_done}/{total_frames} képkocka")  # háttérfeladat haladás (+ megszakítás)

        if team_clf is not None:  # AUTO CSAPAT: a tanult klaszterek a naplóba
            debug_file.write("\n" + team_clf.describe() + "\n")

    cap.release()
    process.stdin.close()
    process.wait()
    if os.path.exists(temp_cut):
        os.remove(temp_cut)

    # Lövés/gól/kapus események lezárása és összefésülése a helyezkedési hibákkal (idő szerint rendezve)
    engine_instance.finalize(start_sec + frames_done / fps)
    if review is not None:
        feed_review_events()
        review.close()
    timeline_events = engine_instance.merge_timeline(timeline_events)

    analysis_payload = {
        "events": timeline_events,
        "review_url": f"/clips/review_{clip_id}/index.html" if review is not None else None,
        "match_stats": engine_instance.stats,
        "pitch_tracking": {"setup": pitch_setup_msg, "frames": pitch_stats},
        "source": {"video_path": video_path, "start_sec": start_sec, "end_sec": end_sec},
        # Tömörített pályák (5 Hz, 0.1 m) a hőtérképhez / statisztikához – lásd heatmap.py
        "trajectories": compact_trajectories(engine_instance.player_trajectories, track_meta.summary()),
        # AUTO CSAPAT: tanult mezszínek (RGB) a kliensnek / taktikai táblának; kézi módban null
        "team_colors": team_clf.team_colors() if team_clf is not None else None,
        "team_mode": "auto" if team_clf is not None else "manual"
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
    min_player_height_pct: float = Form(0.048),
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
    calib_pitch_points: str = Form("")
):
    config = {
        "team1_attacks_right": team1_attacks_right,
        "auto_teams": auto_teams, "swap_teams": swap_teams,
        "review_export": review_export,
        # Pályakövetés (lásd pitch_tracker.py, docs/FEJLESZTESI_TERV.md)
        "dynamic_homography": dynamic_homography, "draw_pitch_overlay": draw_pitch_overlay,
        "calib_time_sec": calib_time_sec, "calib_pitch_points": calib_pitch_points,
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
            "-i", "-", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", output_path
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
    with _segmenter_person_lock:
        if _segmenter_person_model is None:
            _segmenter_person_model = YOLO('yolov8n.pt')
        results = _segmenter_person_model.predict(frame, classes=[0], conf=0.3, device=device, verbose=False)
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

TACTICS_BOARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "taktikai_tabla.html")

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

job_manager = JobManager(os.path.join(CLIPS_DIR, "jobs"), max_workers=2)


def _job_funcs():
    # A már betöltött main modult adjuk át (python main.py esetén __main__), hogy ne töltődjön be újra
    return pipeline.default_funcs(sys.modules[__name__])


async def _form_dict(request: Request):
    form = await request.form()
    return {k: v for k, v in form.items() if isinstance(v, str)}


def _form_error(message, code=400):
    return JSONResponse(status_code=code, content={"status": "error", "message": message})


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
    except ValueError as ex:
        return _form_error(str(ex))
    params = {"video_path": video_path, **{k: opts[k] for k in pipeline.PIPELINE_DEFAULTS},
              "calib_points": opts["calib_points"], "team1_attacks_right": opts["team1_attacks_right"]}
    job_id = job_manager.submit("analyze-match", _run_match_job, video_path, opts, params=params)
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
    paths = cut_match_halves(raw["video_path"], halves, CLIPS_DIR, padding_sec=opts["padding_sec"], duration_sec=duration)
    to_url = lambda p: f"/clips/{os.path.basename(p)}" if p else None
    return {"status": "success", "halves": halves, "confidence": (detection or {}).get("confidence"),
            "notes": notes, "first_half_url": to_url(paths["first_half"]),
            "second_half_url": to_url(paths["second_half"]), "full_match_url": to_url(paths["full_match"]),
            "local_paths": paths}


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
    except ValueError as ex:
        return _form_error(str(ex))
    params = {k: v for k, v in raw.items() if k != "events_json"}
    job_id = job_manager.submit(kind, JOB_KINDS[kind], raw, params=params)
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
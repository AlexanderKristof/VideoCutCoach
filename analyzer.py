import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

model = YOLO("yolov8n.pt")

def extract_dominant_color(image_crop):
    """Mezszín kinyerése a játékos felsőtestéből K-Means klaszterezéssel."""
    if image_crop.size == 0:
        return [0, 0, 0]
    
    # Vágjuk le a fej- és lábrészt, csak a mezre (felsőtestre) fókuszálunk
    h, w, _ = image_crop.shape
    torso = image_crop[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
    
    if torso.size == 0:
        return [0, 0, 0]

    # BGR -> RGB átalakítás és pixel tömb formázás
    torso_rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
    pixels = torso_rgb.reshape(-1, 3)

    if len(pixels) < 5:
        return [0, 0, 0]

    kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
    kmeans.fit(pixels)
    
    # A legdominánsabb szín kiválasztása
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    
    return [int(c) for c in dominant_color]

def apply_homography(points: list, h_matrix: np.ndarray) -> list:
    if not points or h_matrix is None:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, h_matrix)
    return transformed.reshape(-1, 2).tolist()

def analyze_image_with_tactics(image_bytes: bytes, src_points: list = None):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    detections = []
    foot_positions = []
    player_colors = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            if label in ["person", "sports ball"]:
                xmin, ymin, xmax, ymax = [int(c) for c in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                
                foot_x = (xmin + xmax) / 2.0
                foot_y = float(ymax)
                
                det_data = {
                    "class": label,
                    "confidence": round(confidence, 2),
                    "bbox": [xmin, ymin, xmax, ymax],
                    "screen_pos": [round(foot_x, 1), round(foot_y, 1)]
                }

                # Csapatbesoroláshoz mezszín kinyerése a személyeknél
                if label == "person":
                    crop = img[ymin:ymax, xmin:xmax]
                    color = extract_dominant_color(crop)
                    det_data["team_color_rgb"] = color
                    player_colors.append(color)

                detections.append(det_data)
                foot_positions.append([foot_x, foot_y])

    # Csapatok csoportosítása mezszín alapján (2 csapat + bíró)
    if len(player_colors) >= 3:
        try:
            team_kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
            team_labels = team_kmeans.fit_predict(player_colors)
            
            player_idx = 0
            for det in detections:
                if det["class"] == "person":
                    det["team_id"] = int(team_labels[player_idx])
                    player_idx += 1
        except Exception:
            pass

    # Homográfiai transzformáció 2D pálya koordinátákra
    if src_points and len(src_points) == 4:
        dst_points = np.array([
            [0, 0],
            [1050, 0],
            [1050, 680],
            [0, 680]
        ], dtype=np.float32)
        
        src_np = np.array(src_points, dtype=np.float32)
        H, _ = cv2.findHomography(src_np, dst_points)
        
        if H is not None:
            raw_mapped = apply_homography(foot_positions, H)
            for i, det in enumerate(detections):
                det["tactical_pos"] = [round(raw_mapped[i][0], 1), round(raw_mapped[i][1], 1)]

    return {
        "total_objects": len(detections),
        "detections": detections
    }
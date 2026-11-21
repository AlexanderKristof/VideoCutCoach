"""
Cserélhető felismerő (detektor) és saját többobjektumos követő.

Modulok:
  detector_base       – a Detector interfész és a Detections adatszerkezet (csak numpy)
  onnx_detector       – ONNX Runtime alapú detektor (MIT licenc), YOLO/YOLOX és DETR-kimenetekkel
  ultralytics_detector – fejlesztői háttér: ultralytics YOLO predict (AGPL-3.0, csak ha kiválasztják!)
  tracker             – saját ByteTrack-szerű követő (Kálmán-szűrő + kétlépcsős IoU párosítás)
  backend             – detektor + követő összekötése a main.py-hoz (ultralytics-szerű eredmény-adapter)

Ez a csomag szándékosan NEM importál ultralytics-et / torch-ot: az ultralytics_detector modult
csak akkor töltjük be, ha a beállításban backend = "ultralytics".
"""

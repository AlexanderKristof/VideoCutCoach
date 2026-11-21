# Harmadik féltől származó komponensek (THIRD PARTY NOTICES)

> VÁZLAT – kiadás előtt minden sort a ténylegesen csomagolt verzióval és a teljes licencszöveggel
> kell kitölteni (lásd docs/LICENC_ES_ARUSITAS.md, 3. pont). A licencszövegek a `licenses/` mappába
> kerüljenek (`licenses/<komponens>.txt`).

| Komponens | Verzió | Licenc | Forrás | Licencszöveg |
|---|---|---|---|---|
| ONNX Runtime | TODO | MIT | https://github.com/microsoft/onnxruntime | licenses/onnxruntime.txt |
| Detektor-modell: TODO (pl. YOLOX-S finomhangolva) | TODO | Apache-2.0 | https://github.com/Megvii-BaseDetection/YOLOX | licenses/yolox.txt |
| OpenCV (opencv-python) | TODO | Apache-2.0 | https://github.com/opencv/opencv-python | licenses/opencv.txt |
| FFmpeg (LGPL build) | TODO | LGPL-2.1-or-later | https://ffmpeg.org (a pontos forrás-csomag linkje: TODO) | licenses/ffmpeg-lgpl.txt |
| OpenH264 (futásidőben letöltve, nem csomagolva) | 2.5.0 | BSD-2-Clause + Cisco bináris licenc | https://www.openh264.org | licenses/openh264.txt |
| NumPy | TODO | BSD-3-Clause | https://numpy.org | licenses/numpy.txt |
| SciPy (opcionális) | TODO | BSD-3-Clause | https://scipy.org | licenses/scipy.txt |
| FastAPI | TODO | MIT | https://github.com/fastapi/fastapi | licenses/fastapi.txt |
| Starlette | TODO | BSD-3-Clause | https://github.com/encode/starlette | licenses/starlette.txt |
| Uvicorn | TODO | BSD-3-Clause | https://github.com/encode/uvicorn | licenses/uvicorn.txt |
| Pydantic | TODO | MIT | https://github.com/pydantic/pydantic | licenses/pydantic.txt |
| SQLAlchemy | TODO | MIT | https://www.sqlalchemy.org | licenses/sqlalchemy.txt |
| python-jose | TODO | MIT | https://github.com/mpdavis/python-jose | licenses/python-jose.txt |
| passlib | TODO | BSD-3-Clause | https://foss.heptapod.net/python-libs/passlib | licenses/passlib.txt |
| bcrypt | TODO | Apache-2.0 | https://github.com/pyca/bcrypt | licenses/bcrypt.txt |
| python-multipart | TODO | Apache-2.0 | https://github.com/Kludex/python-multipart | licenses/python-multipart.txt |

Saját kód: a `detection/tracker.py` követő a ByteTrack-cikk (Zhang et al., ECCV 2022) algoritmusa alapján
készült saját megvalósítás; harmadik féltől származó kódot nem tartalmaz.

Tanító adatok forrásai és licencei: TODO (adatkészletenként: név, licenc, forrásmegjelölés, hozzájárulás).

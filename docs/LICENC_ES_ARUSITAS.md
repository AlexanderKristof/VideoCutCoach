# Licencek és értékesítés (zárt forrású asztali alkalmazás)

> Tájékoztató összefoglaló a fejlesztéshez – **nem jogi tanácsadás**. Kereskedelmi kiadás előtt a
> licencfeltételeket (és a felhasznált adatok jogi helyzetét) szakjogásszal is nézesd át.

## 1. Komponensek

| Komponens | Licenc | Mire használjuk | Kereskedelmi, zárt forrású terjesztés |
|---|---|---|---|
| **ultralytics** (YOLOv8/11, `model.track`, ByteTrack-integráció) | **AGPL-3.0** (vagy fizetős Ultralytics Enterprise licenc) | csak fejlesztői háttér: `backend: "ultralytics"` / `"ultralytics_predict"` | **NEM** kerülhet a termékbe (az AGPL a teljes alkalmazás forrásának kiadását követelné, hálózati használatnál is). Az Ultralytics által tanított súlyok (`yolov8*.pt`, `yolo11*.pt`) és az azokból finomhangolt modellek is AGPL alá esnek. |
| `analyzer.py` (régi modul) | – (ultralytics-et importál) | régi kísérleti kód | a kiadott csomagból **ki kell hagyni** |
| **ONNX Runtime** (`onnxruntime`, `-gpu`, `-directml`) | MIT | a termék detektor-futtatója (`backend: "onnx"`) | igen; a licencszöveget mellékelni kell |
| **YOLOX** kód + előtanított súlyok (Megvii) | Apache-2.0 (a `LICENSE` fájl ellenőrizve: Apache License 2.0) | ellenőrzött, letölthető ONNX modell (COCO); finomhangolási keretrendszer | igen; Apache-2.0 szöveg + NOTICE (ha van) mellékelése |
| RF-DETR (Roboflow) | Apache-2.0 (kód és a nano/small/medium/base súlyok) | javasolt finomhangolási keretrendszer | igen; **minden használt ellenőrzőpont licencét külön nézd meg** (az újabb, nagyobb modellek egy része más licenc alatt jelenhet meg) |
| D-FINE, RT-DETR (eredeti repó) | Apache-2.0 | alternatív finomhangolási keretrendszer | igen (az Objects365 / COCO előtanítás adat-feltételeit lásd lent) |
| Saját követő (`detection/tracker.py`) | saját kód | játékoskövetés | igen. A ByteTrack-algoritmus a publikált cikk alapján, nulláról megírva (a ByteTrack referencia-implementáció MIT licencű; GPL/AGPL kód nincs átvéve) |
| Saját detektor-réteg (`detection/`), labdakövető, motor | saját kód | – | igen |
| OpenCV (`opencv-python`) | Apache-2.0 (4.5-től) | képfeldolgozás | igen; a pip-kerék tartalmaz egy FFmpeg-buildet is (LGPL) – lásd FFmpeg |
| NumPy | BSD-3-Clause | – | igen |
| SciPy (opcionális, a követő gyorsabb párosításához) | BSD-3-Clause | – | igen |
| FastAPI, Starlette, Uvicorn, Pydantic | MIT / BSD-3-Clause | helyi motor-API | igen |
| SQLAlchemy | MIT | adatbázis | igen |
| python-jose, passlib, bcrypt | MIT / BSD / Apache-2.0 | hitelesítés | igen |
| PyTorch | BSD-3-Clause | **csak ha** az ultralytics háttér vagy a tanítás kell; az ONNX háttérhez nem szükséges | igen, de a CUDA-s kerék NVIDIA-licencű komponenseket is tartalmaz – a termékbe ne kerüljön, ha nem kell |
| scikit-learn | BSD-3-Clause | csak a régi `analyzer.py` | – |
| **FFmpeg** (`ffmpeg.exe`) | LGPL-2.1+ **vagy** GPL (a build beállításaitól függ) | videóvágás / -kódolás | csak **LGPL build** (`--enable-gpl` és `--enable-nonfree` NÉLKÜL); külön futtatható fájlként hívva, cserélhetően, a licenc + a forrás elérhetőségének megadásával. Az x264 / x265 GPL → GPL buildet ne terjesszünk. |
| **OpenH264** (`openh264-2.5.0-win64.dll`) | BSD-2-Clause forrás; a Cisco bináris H.264 szabadalmi licencét csak a **Cisco által terjesztett** bináris kapja | H.264 kódolás az LGPL FFmpeg mellett | a DLL-t **ne csomagoljuk a telepítőbe**: első használatkor a felhasználó gépére a Cisco szerveréről kell letölteni (a felhasználó által kikapcsolhatóan), és a Cisco előírt szövegét meg kell jeleníteni. Ha mi fordítjuk vagy mi terjesztjük a DLL-t, a szabadalmi licenc nem vonatkozik ránk. |

## 2. Adatok és modellek

| Adat | Feltétel | Következmény |
|---|---|---|
| **COCO** (a YOLOX / D-FINE / RF-DETR előtanítás) | a képek Flickr-licencűek, az annotációk CC BY 4.0 | a modellsúlyok licence a kiadótól jön (YOLOX: Apache-2.0); a gyakorlatban elfogadott kereskedelmi alap, de jogi kockázatként jegyezd fel |
| Objects365 (D-FINE előtanítás) | kutatási célú adatfeltételek | ha D-FINE Obj365-ös súlyt használsz, a feltételeket nézesd át; alternatíva: COCO-előtanított ellenőrzőpont |
| **SoccerNet** | kutatási / nem kereskedelmi (NDA alapú hozzáférés) | **termékmodell tanítására nem használható** (értékelésre se a kiadott termékhez) |
| **Roboflow Universe** adatkészletek | adatkészletenként eltérő (CC BY 4.0, CC BY-NC, MIT, „privát” ...) | egyenként ellenőrizni; NC (nem kereskedelmi) licenc kizárva; CC BY esetén forrásmegjelölés a THIRD_PARTY_NOTICES-ben. A Roboflow-n tanított modell letöltésének feltételei is a Roboflow felhasználási feltételeitől függnek. |
| **Saját Veo felvételek** | a felvett személyek (játékosok, kiskorúaknál a szülők) és a klub hozzájárulása, GDPR | tanító adatként csak írásos hozzájárulással / megfelelő jogalappal; a Veo felhasználási feltételei a letöltött videó felhasználására is vonatkozhatnak; a tanító képeket ne terjeszd a termékkel |
| Szintetikus képek (`tools/synth_match.py`) | saját | szabadon használható (előcímkézés, tesztelés) |

## 3. Ellenőrzőlista kereskedelmi kiadás előtt

1. [ ] A `model_config.json` alapértéke `"backend": "onnx"`, és a termék-modell egy **Apache-2.0 / MIT**
       licencű keretrendszerrel (YOLOX, RF-DETR, D-FINE), **engedélyezett adatokon** tanított ONNX fájl.
2. [ ] A telepítő **nem tartalmazza** az `ultralytics` csomagot, a `torch`-ot (ha nem kell), az `analyzer.py`-t,
       sem `*.pt` YOLOv8/11 súlyt. Ellenőrzés: `pip list` a csomagolt környezetben; a `main.py` ONNX módban nem
       importálja az ultralytics-et (lusta import).
3. [ ] FFmpeg: LGPL build, a `ffmpeg -version` kimenetében nincs `--enable-gpl` / `--enable-nonfree`; a licenc és
       a forrás-hivatkozás mellékelve.
4. [ ] OpenH264: a DLL nincs a telepítőben; futásidejű letöltés a Cisco-tól, a Cisco-szöveg megjelenítése, kikapcsolható.
5. [ ] `THIRD_PARTY_NOTICES.md` kitöltve (minden csomag + verzió + licencszöveg), a telepítő mellékeli.
6. [ ] A tanító adatok listája (forrás, licenc, hozzájárulás) dokumentálva és archiválva; SoccerNet és NC-licencű
       adat nincs benne.
7. [ ] A modellfájl (`.onnx`) metaadata / kísérő `MODEL_CARD` tartalmazza: keretrendszer, előtanító súly és licence,
       tanító adatok, dátum.
8. [ ] Ha a GPU-s futtatóhoz `onnxruntime-gpu` (CUDA) kell: a CUDA / cuDNN DLL-ek NVIDIA-licence engedi az
       újraterjesztést (a megengedett fájlok listája szerint), különben a felhasználó telepíti; Windows-on a
       `onnxruntime-directml` (MIT, a DirectML a Windows része / MIT-es NuGet) egyszerűbb.
9. [ ] Jogi átnézés (licencek + adatvédelem + a Veo felhasználási feltételei).

# Felismerő modell beállítása

Alapból (beállítás nélkül) a motor az általános COCO YOLO modellt használja:
`yolov8x.pt` NVIDIA kártyán (CUDA), `yolov8n.pt` processzoron. Ez a „személy” és a „sportlabda”
osztályt ismeri, kapust és bírót nem különböztet meg.

Pontosabb eredményt ad egy **focira tanított modell** (pl. a Roboflow „football-players-detection”
modelljei), amelynek osztályai: `player`, `goalkeeper`, `referee`, `ball`.

## 1. A modellfájl helye

1. Töltsd le / tanítsd be a modellt, és mentsd el `.pt` fájlként, pl. `models/football-players-detection.pt`
   (a `main.py` melletti `models` mappába).
2. A `.pt` fájlokat a `.gitignore` kizárja – **modellfájlt ne tegyél fel a repóba**.

## 2. Beállítás

Két lehetőség van (a környezeti változó erősebb, mint a fájl).

### a) `model_config.json` a `main.py` mellett

Másold le a `model_config.example.json` fájlt `model_config.json` néven, és írd át:

```json
{
    "weights": "models/football-players-detection.pt",
    "class_names": {
        "player": ["player"],
        "goalkeeper": ["goalkeeper"],
        "referee": ["referee"],
        "ball": ["ball"]
    },
    "ball_conf": 0.35,
    "detect_every_n": 1
}
```

| Kulcs | Jelentés | Alapérték |
|---|---|---|
| `weights` | a súlyfájl útvonala (mindkét eszközre) | – |
| `weights_cuda` / `weights_cpu` | külön súlyfájl NVIDIA kártyához / processzorhoz | `yolov8x.pt` / `yolov8n.pt` |
| `class_names` | szerepkör → a modell osztálynevei (lásd lent) | automatikus |
| `ball_conf` | a labda minimális megbízhatósága | `0.35` |
| `detect_every_n` | gyors mód: csak minden N. képkockán fut felismerés | `1` |
| `ball_extra_pass` | külön labda-felismerő menet, ha a nyers labda-callback nem érhető el (lásd 6. pont) | `false` |

A fájl máshol is lehet: a `TACTICAL_MODEL_CONFIG` környezeti változóban add meg az útvonalát.

### b) Környezeti változók

```bat
set TACTICAL_MODEL_WEIGHTS=models\football-players-detection.pt
set TACTICAL_DETECT_EVERY_N=3
python main.py
```

## 3. Osztálynevek

A motor **név alapján** keresi meg az osztályokat a modellben (`model.names`), nem fix sorszám alapján.
Felismert nevek (kis-/nagybetű, `_` és `-` mindegy):

| Szerepkör | Nevek |
|---|---|
| játékos | `player`, `person` |
| kapus | `goalkeeper`, `goalie`, `keeper` |
| bíró | `referee`, `ref` |
| labda | `ball`, `sports ball`, `football`, `soccer ball` |

Ha a modelled más neveket használ, a `class_names` kulcsban add meg őket (a megadott szerepkörnél
csak ezek számítanak). Indításkor a konzolon látszik, melyik osztály melyik szerepkörhöz került, pl.:

```
Osztályok: player=[2:player], goalkeeper=[1:goalkeeper], referee=[3:referee], ball=[0:ball]
```

Mit változtat a focis modell:
- **Kapus osztály:** a kapusnak felismert alak kapusként (narancs keret) jelenik meg akkor is,
  ha a kapus mezszíne nincs beállítva; az álló alak szűrő őt sem dobja el.
- **Bíró osztály:** a bíró nem kerül a játékosok közé (statisztika, csapat), csak sárga-fekete
  kerettel rajzoljuk ki.

## 4. Gyors mód (gyenge laptopra)

`detect_every_n` (vagy `TACTICAL_DETECT_EVERY_N`) = N esetén a felismerés és követés csak minden
N. képkockán fut, a köztes képkockákon az előző eredményt rajzoljuk ki újra. A kimeneti videó
így is minden képkockát tartalmaz, de a keretek legfeljebb N−1 képkockát „késhetnek”.

- Javasolt érték integrált videókártyán: 2–3 (nagyobb értéknél a követés is pontatlanabb lesz).
- A statisztikák és események csak a ténylegesen elemzett képkockákból számolódnak.

## 5. Párhuzamos elemzések

A modell egy példányban van betöltve, ezért egyszerre csak egy klip elemzése használja: ha a kliens
egyszerre több elemzést indít, azok sorban futnak le. Minden elemzés elején a követő (ByteTrack)
állapota törlődik, így a játékosazonosítók nem keverednek két klip között.

## 6. Labda-detektálások a követő előtt (ultralytics callback)

A `model.track()` (ByteTrack) a kimenetből **elhagyja a meg nem erősített dobozokat**. A labdát a
modell ritkán és szaggatottan látja, ezért a követő a labda-detektálások nagy részét eldobja
(a szintetikus mérésen ~70%-át). Második futtatás nélkül így nyerjük vissza őket:

**Ultralytics-feltételezés (8.x – `ultralytics/engine/model.py`, `ultralytics/trackers/track.py`):**

1. A követést a `register_tracker()` végzi, amit a `Model.track()` az **első** hívásnál futtat
   (`if not hasattr(self.predictor, "trackers")`): ez a `model.callbacks["on_predict_postprocess_end"]`
   lista **végére** fűzi a követő callbackjét, amely a `predictor.results[i]` elemeit a követett
   dobozokra cseréli (`predictor.results[i] = result[idx]`).
2. A callbackek regisztrálási sorrendben futnak, és a predictor ugyanazt a `callbacks` dict-et
   kapja, mint a modell (`_callbacks=self.callbacks`).

Ezért a `main.py` a modell betöltése után, **az első `track()` előtt** saját
`on_predict_postprocess_end` callbacket regisztrál (`model.add_callback(...)`): ez még a nyers
`predictor.results`-ot látja, és képkockánként kimásolja a labda-osztályú dobozokat
(`xyxy` / `conf` / `cls`, numpy másolatként). Ezek mennek a labdakövetőbe; a játékosok és bírók
továbbra is a követett (azonosítós) dobozokból jönnek. Ugyanazt a képet nem futtatjuk kétszer.

Védőkorlátok:
- ha a modellnek nincs `add_callback` metódusa, vagy a callback az adott képkockán nem futott le,
  a követett labda-dobozokat használjuk (a régi működés);
- ha a callback már azonosítós (követett) dobozokat lát, vagyis egy másik ultralytics-verzióban a
  követő előtte futott, a napló (`analysis_debug_*.log`) figyelmeztet;
- `ball_extra_pass: true`: ha nincs nyers callback, képkockánként külön labda-menet fut
  (`predict(classes=<labda>, conf=alacsony)`) egy **második modellpéldánnyal**, mert ugyanazon a
  modellen a `predict()` a `track()` után a követő callbackjét is lefuttatná, és összekeverné a
  játékos-azonosítókat. Ez lassú (kétszeres felismerés) és több GPU-memóriát kér – csak végső
  esetre.

Ultralytics-frissítés után érdemes megnézni a naplóban a „Labda-jelöltek forrása” sort: ha a
`nyers` érték 0, a callback-mechanizmus megváltozott. A `tools/fake_detector.py` ál-YOLO ugyanezt
a mechanizmust utánozza (a követő az első `track()`-nél regisztrált callback), így a
`tests/test_e2e_synth.py` a valódi kódutat méri.

### 6b. Kis labda: natív felbontású labda-kivágás, sebesség

A Veo 1920x1080-as képén a labda ~6–12 px; a felismerő a teljes képet `imgsz`-re (640) kicsinyíti,
így a labda 2–4 px lesz, és a COCO modell alig látja (valódi felvételen ~4% labda-találat). A
**labda-kivágás** (`ball_crops.py`) a labdakövető becsült helye körül egy natív felbontású
(`ball_crop_size`, alap 640x640) kivágáson külön labda-felismerést futtat, és a dobozokat a teljes
képre vetítve jelöltként adja a követőnek; elveszett labdánál a pálya csempéin (a kép alsó ~70%-a)
körbeforgó keresés fut (`ball_tile_every` képkockánként egy csempe; a legutóbbi hely / a képközép
közelében kezdve). Az ultralytics háttérnél külön modellpéldánnyal (a `model.track()` követője nem
zavarodik össze), a saját (ONNX) háttérnél a detektor `detect_batch()`-ével.

| kulcs | alap | jelentés |
|---|---|---|
| `imgsz` | 640 | a fő (játékos) felismerés bemeneti mérete |
| `half` | `"auto"` | FP16 a GPU-n (`auto` = CUDA-n be) |
| `ball_crop_pass` | `"auto"` | labda-kivágás (`auto` = CUDA-n be, CPU-n ki; `TACTICAL_BALL_CROP_PASS`) |
| `ball_crop_size` | 640 | a kivágás mérete (px, natív felbontáson) |
| `ball_tile_every` | 2 | elveszett labdánál hány képkockánként fusson egy csempe |
| `ball_crop_conf` | 0.10 | a kivágásos menet minimális pontszáma |

Az elemzés végén a napló (`analysis_debug_*.log`) és az events JSON `timing` blokkja szakaszonként
mutatja az időt (ms/képkocka, arány), a tényleges kép/mp-t és a kimeneti videó kódolóját
(NVENC / QSV / AMF / libx264 veryfast / libopenh264 – a `fast_cut` felderítése szerint). Az annotált
videó kicsinyíthető: `output_scale` (pl. 0.5), parancssorban `--kimenet-felbontas 0.5`.

### 6c. A CPU-szakaszok gyorsítása (pályakövetés, fűmaszk)

A GPU-s felismerés mellett a képkockánkénti idő nagy része CPU-munka (pályakövetés, fűmodell,
rajzolás). Ezek beállításai (a környezeti változó felülírja a JSON-t):

| kulcs | alap | jelentés |
|---|---|---|
| `pitch_refine_max_every` | 5 | adaptív vonal-finomítás: stabil, a vonalakkal rendre megerősített követésnél a drága vonalkeresés + illesztés legfeljebb ennyi képkockánként fut (közte a kameramodell mozgásbecslése adja az állást); gyanús jelnél (rossz illesztés, nagy eltérés, gyors pásztázás / zoom, sikertelen mozgásbecslés) azonnal vissza a 3 képkockás ütemre. `3` = a régi, rögzített ütemezés (`TACTICAL_PITCH_REFINE_MAX_EVERY`) |
| `grass_mask_every` | 2 | a fűmaszk újraszámolása ennyi képkockánként; közte az előző maszk a kameramozgással eltolva. `1` = minden képkockán (`TACTICAL_GRASS_MASK_EVERY`) |
| `parallel_pitch` | `"auto"` | a pályakövetés (+ kameramozgás, fűmaszk) külön szálon, a felismeréssel egy időben; az eredmény ugyanaz, csak az átfedés gyorsít. `auto` = CUDA-n be (CPU-s felismerésnél a két szál csak egymást lassítaná; `TACTICAL_PARALLEL_PITCH`) |

A legpontosabb (régi) működés: `pitch_refine_max_every: 3`, `grass_mask_every: 1`. Párhuzamos módban a
napló időmérése a pályakövetésnél csak a várakozást mutatja; a háttérszál ideje külön sorban
(„átfedve”) jelenik meg.


## 7. Betöltés és a súlyfájl helye (lusta betöltés, models mappa)

A modell **nem** a szerver indulásakor töltődik be, hanem az első elemzéskor (`main.ensure_model()`,
szálbiztos, egyszer), vagy előre a `POST /models/load` háttérfeladattal. Így a `GET /health`
azonnal válaszol, és a torch / ultralytics importja sem lassítja az indulást. A nyers
labda-callback (6. pont) a betöltéskor, az első `track()` előtt regisztrálódik.

A súlyfájlt a `model_manager.py` keresi, ebben a sorrendben: a megadott út (a munkakönyvtárhoz
képest), a `models/` mappa (fájlnévvel is), az adatmappa, a csomagolt erőforrások, a program
`models/` mappája. Ha nincs meg:
- fejlesztői módban (`python main.py`) a régi viselkedés: a név változatlanul megy az ultralytics-nek,
  amely a hivatalos súlyt a munkakönyvtárba tölti le;
- alkalmazás-módban (telepített program / `TACTICAL_DATA_DIR`) a hivatalos súlyt (`yolov8*.pt`,
  `yolo11*.pt`) a motor maga tölti le az adatmappa `models/` mappájába (`TACTICAL_AUTO_DOWNLOAD=0`
  kikapcsolja; `TACTICAL_MODEL_BASE_URL` belső tükröt ad meg). A letöltés haladása:
  `POST /models/download` → `GET /jobs/{id}`; állapot: `GET /models/status`.

Alkalmazás-módban a `model_config.json` helye is az adatmappa (`%LOCALAPPDATA%\TacAnaly`).

## 8. Detektor-háttér: ONNX Runtime + saját követő (a kereskedelmi út)

Az ultralytics AGPL-3.0 licencű, ezért zárt forrású, eladott termékbe nem kerülhet
(lásd `docs/LICENC_ES_ARUSITAS.md`). A `backend` kulcs választja ki a felismerő láncot:

| `backend` | Felismerés | Követés | Mire |
|---|---|---|---|
| `"ultralytics"` (alap) | `model.track()` | az ultralytics ByteTrack-je | fejlesztés, összevetés |
| `"onnx"` | ONNX Runtime (`detection/onnx_detector.py`, MIT) | saját követő (`detection/tracker.py`) | **termék** |
| `"ultralytics_predict"` | ultralytics `predict()` | saját követő | fejlesztés: a saját követő valódi YOLO-dobozokon |

Az `"onnx"` háttért is az `ensure_model()` hozza létre (lusta betöltés, 7. pont). Ennél a háttérnél a `main.py` nem importálja sem az ultralytics-et, sem a torch-ot. A labda
minden nyers jelöltje közvetlenül a labdakövetőbe megy (a 6. pont callback-trükkje itt nem kell).
Kész minta: `model_config.onnx.example.json`.

### Beállítás

```json
{
    "backend": "onnx",
    "onnx_path": "models/focis_detektor.onnx",
    "onnx_class_names": ["ball", "goalkeeper", "player", "referee"],
    "role_conf": {"ball": 0.05, "player": 0.1, "goalkeeper": 0.1, "referee": 0.1},
    "onnx": {"preset": "auto", "iou": 0.5, "threads": 0},
    "providers": "auto",
    "tracker": {"track_buffer": 30, "high_thresh": 0.25}
}
```

| Kulcs | Jelentés | Alapérték |
|---|---|---|
| `backend` | lásd fent; környezeti változó: `TACTICAL_DETECTOR_BACKEND` | `ultralytics` |
| `onnx_path` | az `.onnx` fájl; a 7. pont szerinti helyeken keresve (`models/` mappa, adatmappa, erőforrások), végül a `main.py` mappájához képest; környezeti változó: `TACTICAL_ONNX_PATH` | – |
| `input_size` | a modell bemenete `[szélesség, magasság]` (csak dinamikus bemenetű modellnél kell) | a modellből |
| `onnx_class_names` | a modell osztálynevei, ha a modell metaadata nem tartalmazza (lista, `{id: név}` vagy `"coco80"`) | metaadat / COCO |
| `class_names` | szerepkör → osztálynevek (mint eddig, 3. pont) | automatikus |
| `role_conf` | szerepkörönkénti minimális pontszám a detektorban (a labda alacsonyabb; a személyeknél a követő 0.1-től a gyenge dobozokat is használja) | lásd fent |
| `onnx` | további detektor-beállítások: `preset` (`auto` / `yolo` / `yolox` / `detr`), `iou`, `max_det`, `normalize` (`"imagenet"` RF-DETR-hez), `box_format`, `threads` (CPU-szálak, 0 = automatikus) | – |
| `providers` | `"auto"` = CUDA > DirectML > CPU, vagy lista | `auto` |
| `tracker` | a saját követő paraméterei (`detection/tracker.py`, `ByteTracker`) | ultralytics-szerű |

GPU-gyorsítás: NVIDIA kártyán `pip install onnxruntime-gpu` (CUDA), bármely Windows GPU-n (Intel / AMD
integrált is) `pip install onnxruntime-directml`. Egyszerre csak az egyik onnxruntime csomag legyen
telepítve. A naplóban (`Modell: ONNX (..., végrehajtó: ...)`) látszik, melyik végrehajtó fut.

### Támogatott modell-kimenetek

- **YOLO-szerű** (egy kimenet, NMS-t mi végzünk, osztály-tudatosan): `[1, 4+C, N]` és a **YOLOX**
  `[1, N, 5+C]` (a hivatalos YOLOX ONNX fájlok dekódolatlan rács-kimenetét is kezeli).
- **DETR-szerű** (NMS nélkül): RT-DETR / D-FINE export (`labels`, `boxes`, `scores` + `orig_target_sizes`
  bemenet), RF-DETR export (`dets` + `labels` logitok; ehhez `"onnx": {"normalize": "imagenet"}`).

Ellenőrzés egy képen (kimenetek, felismert konvenció, osztályok, detektálások, idő):
```bat
python -m detection.onnx_detector models\focis_detektor.onnx kep.jpg
```

### Kipróbált, kereskedelmileg használható modell

A YOLOX (Megvii, **Apache-2.0**, kód és súlyok) hivatalos COCO ONNX fájljai működnek a háttérrel
(`person` → játékos, `sports ball` → labda; kapust / bírót nem különböztet meg). Letöltés: a YOLOX
projekt GitHub-oldalának „Releases” részéről (0.1.1rc0: `yolox_nano.onnx`, `yolox_tiny.onnx`,
`yolox_s.onnx`, `yolox_m.onnx`). Ez csak átmeneti megoldás: a kis, gyors labdát a COCO modell alig
látja – a termékhez focira finomhangolt modell kell (`docs/MODELL_TANITAS.md`).

### A saját követő

ByteTrack-szerű, saját kód: Kálmán-szűrő (középpont, oldalarány, magasság + sebességek), kétlépcsős
párosítás (erős, majd gyenge detektálások; Magyar-módszer), megerősítés a második találatnál,
elveszett követés visszatérése `track_buffer` képkockán belül. A kamera mozgását a pályakövető
képkockák közti homográfiájával (`PitchTracker.frame_motion()`) kompenzálja – a forgó Veo képnél
ez csökkenti az azonosító-váltásokat. A szintetikus mérés (`tests/test_e2e_synth.py`,
`oracle_owntracker` mód) azonos detektálásokon hasonlítja össze az ultralytics-szerű követővel.


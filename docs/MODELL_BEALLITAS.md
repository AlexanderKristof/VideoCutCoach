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


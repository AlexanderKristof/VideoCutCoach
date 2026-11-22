# Saját focis detektor tanítása (kereskedelmileg használható úton)

Cél: egy **Apache-2.0** licencű keretrendszerrel, **saját (engedélyezett) felvételeken** finomhangolt
detektor ONNX formátumban, amelyet a motor a `backend: "onnx"` háttérrel futtat
(`docs/MODELL_BEALLITAS.md` 8. pont). Osztályok:

| Sorszám | Név a modellben | Magyarul |
|---|---|---|
| 0 | `ball` | labda |
| 1 | `goalkeeper` | kapus |
| 2 | `player` | játékos |
| 3 | `referee` | bíró (játékvezető, partjelző) |

A motor a neveket automatikusan a szerepkörökhöz rendeli (angol és magyar név is működik:
`labda`, `kapus`, `jatekos` / `játékos`, `biro` / `bíró`).

## 1. Képkockák kimentése

```bat
python tools\export_training_frames.py D:\Veo\*.mp4 --out tanito --per-video 150 --skip-start 120 --skip-end 120
```

- 2 mp-enként jelölt képkocka, a homályos (`--min-sharpness`) és a majdnem azonos (dHash,
  `--dedupe-bits`) képek kimaradnak, a maradékból a legváltozatosabb `--per-video` darab marad
  (különböző időpontok, kamera-irányok, fényviszonyok).
- Kimenet: `tanito/kepek/<videó>_<képkocka>.jpg` + `tanito/frames.csv` (videó, képkocka, idő,
  fényerő, élesség, kamera-mozgás).
- Javasolt mennyiség az első modellhez: 1500–3000 kép, legalább 10–15 különböző meccsről
  (eltérő pálya, mez, napszak, villanyfény, eső). A labdás képek aránya legyen magas.
- Csak olyan felvétel, amelyhez a hozzájárulás megvan (`docs/LICENC_ES_ARUSITAS.md`).

## 2. Előcímkézés (a kézi munka gyorsítása)

```bat
REM a jelenlegi detektorral (ONNX, pl. YOLOX COCO, vagy egy korábbi saját modell)
python tools\prelabel.py --images tanito --onnx models\yolox_s.onnx --csempe 640 --nezo-szures --out elocimke --format coco,labelstudio

REM videóból, követéssel: az egyszeri téves dobozok kiesnek, a dobozok azonosítót kapnak
python tools\prelabel.py --video meccs.mp4 --every 25 --start 600 --duration 300 --out elocimke_video

REM szintetikus meccs ground truth-ja (hibátlan címkék a lánc kipróbálásához)
python tools\synth_match.py --out synth.mp4 --duration 60 --seed 7
python tools\prelabel.py --video synth.mp4 --synth-gt synth_gt.json --every 10 --out synth_cimke --format coco,yolo
```

Importálás:
- **CVAT**: új feladat a `kepek` mappával, a címkék: `ball, goalkeeper, player, referee`;
  *Actions → Upload annotations → COCO 1.0* → `elocimke/coco/annotations/train.json`.
- **Label Studio**: a címkéző felület a `tools/dataset_formats.py` `write_labelstudio` leírása szerint;
  *Import* → `elocimke/labelstudio_tasks.json` (az előcímkék „predictions”-ként jelennek meg).

- `--csempe 640`: csempés felismerés eredeti felbontáson – a kicsi, távoli labda és játékos így is meglesz
  (lassabb, de az előcímkézés egyszeri munka).
- `--nezo-szures`: a túlsó kerítés előtti nézősor automatikus kiszűrése (a fűterület felső széle alapján);
  a közeli oldalvonalnál / kispadnál állókat kézzel kell törölni.

Az ONNX COCO-modell `person` dobozai mind `player` címkét kapnak: a kapusokat és a bírókat kézzel kell
átcímkézni, a pályán kívüli embereket (nézők, cserék, edzők) törölni.

## 3. Címkézési útmutató

- **Kit címkézünk:** csak a pályán lévő résztvevőket: játékos, kapus, bíró (a partjelző is `referee`).
  Nézőt, cserét, edzőt, labdaszedőt **nem** (ha kint melegítő cserét is rendszeresen „játékosnak”
  látna a modell, a motor pályahatár-szűrője úgyis kiszűri – a címke legyen következetes).
- **Doboz:** szorosan az alak köré, fejtől a talpig (a láb a dobozban legyen – a motor a doboz alsó
  szélét veszi a talppontnak). Árnyék nincs a dobozban.
- **Részben takart alak:** ha legalább kb. 30%-a látszik, címkézd, és a dobozt a **látható** részre
  húzd, kivéve, ha a talp látszik – akkor a talpig. Teljesen takart alakot ne találgass.
- **Képszélen levágott alak:** címkézd a képen belüli részt.
- **Csoport / tömeg (szöglet, sorfal):** minden alakot külön dobozzal, akkor is, ha átfednek.
  Ha egy alak nem különíthető el, inkább hagyd ki, mint hogy két embert egy dobozba tegyél.
- **Labda:** minden látható labdát, a nagyon kicsit is (2–3 px): szoros, kb. négyzetes doboz.
  Elmosódott (gyorsan repülő) labdánál a doboz a teljes elmosódott csíkot fogja át. A lábnál, kézben
  lévő labdát is. A pályán kívüli tartaléklabdát **ne**. Ha nem biztos, hogy labda, hagyd ki.
- **Kapus:** a mezszín alapján (a kapusmez eltér); a mezőnyjátékos nem kapus akkor sem, ha a kapuban áll.
- **Bíró:** a játékvezető és a partjelzők; a negyedik játékvezető (a pálya mellett) nem.
- **Következetesség:** inkább kevesebb, de egyformán címkézett kép, mint sok, vegyes szabályú.

## 4. Adatkészlet-formátum és felosztás

A CVAT / Label Studio exportját (COCO 1.0) alakítsd a keretrendszer formátumára; a felosztás
meccsenként, 30 mp-es blokkokban történik (a szomszédos képkockák egy halmazba kerülnek):

```bat
REM RF-DETR (Roboflow-elrendezés: train/ valid/ test/ + _annotations.coco.json)
python tools\dataset_formats.py convert --from coco --input export\annotations.json --images export\images ^
    --to coco --layout rfdetr --out adat_rfdetr --val 0.15 --test 0.05

REM D-FINE / YOLOX (annotations/train.json + images/train/ ...)
python tools\dataset_formats.py convert --from coco --input export\annotations.json --images export\images ^
    --to coco --out adat_coco --val 0.15

REM YOLO txt (images/ + labels/ + data.yaml) – ha egy YOLO-formátumú eszköz kell
python tools\dataset_formats.py convert --from coco --input export\annotations.json --images export\images ^
    --to yolo --out adat_yolo --val 0.15
```

A COCO `category_id` 1-től számoz (1 = ball ... 4 = referee; a CVAT a 0-t nem fogadja el). Ha egy keretrendszer 0-tól várja, add meg: `--coco-id-start 0`.

## 5. Tanítás – javasolt keretrendszerek (Apache-2.0)

### Gyors út: `tools/tanitas.py` (RF-DETR, egy-egy paranccsal)

```bat
REM egyszer: tanító környezet (GPU-s PyTorch + rfdetr[train])
powershell -ExecutionPolicy Bypass -File tools\tanitas_telepites.ps1

REM adatkészlet a CVAT exportból (Actions -> Export task dataset -> COCO 1.0, kicsomagolva) + a képek
venv-tanitas\Scripts\python tools\tanitas.py adat --cimkek export\annotations\instances_default.json --kepek tanito\kepek --ki adat_rfdetr

REM tanítás + ONNX export (-> models\focis_detektor.onnx és futasok\elso\model_config.javaslat.json)
venv-tanitas\Scripts\python tools\tanitas.py tanit --adat adat_rfdetr --ki futasok\elso --korok 40
```

- Több meccs (több CVAT-feladat): a `--cimkek` / `--kepek` pár többször is megadható.
- Az `adat` lépés a teljes képek mellé eredeti felbontású 640×640-es csempéket is tesz (főleg a labda
  körül), mert a kicsinyített teljes képen a ~11 px-es labda alig látszana.
- Kevés VRAM-nál: `--batch 2 --accum 8`; ha a modell-letöltés nem megy: `--elotanitott <letöltött .pth>`.
- A végén a `model_config.javaslat.json`-t másold a repó gyökerébe `model_config.json` néven, és a motor
  környezetébe tedd fel az `onnxruntime-gpu` csomagot (NVIDIA) – lásd 6. pont.

A tanításhoz külön Python-környezet kell PyTorch-csal (GPU-val; a termékbe a torch nem kerül).
Bármelyik keretrendszer használható – mindhárom ONNX-et ad, amelyet a motor kezel.

### a) RF-DETR (Roboflow, Apache-2.0) – egyszerű, jó pontosság

```bat
python -m venv venv-tanitas && venv-tanitas\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install rfdetr
```

```python
from rfdetr import RFDETRBase            # vagy RFDETRSmall / RFDETRMedium – a licencet ellenőrizd!
model = RFDETRBase()
model.train(dataset_dir="adat_rfdetr", epochs=60, batch_size=4, grad_accum_steps=4, lr=1e-4,
            output_dir="futasok/rfdetr", resolution=896)   # nagyobb felbontás a kis labdához (56 többszöröse)
model = RFDETRBase(pretrain_weights="futasok/rfdetr/checkpoint_best_total.pth")
model.export(output_dir="futasok/rfdetr/onnx")          # -> inference_model.onnx
```

Motor-beállítás: `"onnx": {"preset": "detr", "normalize": "imagenet"}`, `"input_size": [896, 896]`.
Az RF-DETR logit-kimenetének osztály-sorrendjét ellenőrizd egy képen
(`python -m detection.onnx_detector inference_model.onnx kep.jpg`), és ha kell, add meg az
`onnx_class_names` kulcsban (pl. ha a 0. index háttér / szülő-kategória: `["-", "ball", "goalkeeper", "player", "referee"]`).

### b) D-FINE (Apache-2.0) – nagyon jó pontosság / sebesség arány

A D-FINE projekt (forráskód a GitHubon, „Peterande/D-FINE”) leírása szerint:
1. `configs/dataset/custom_detection.yml`: `num_classes: 4`, `remap_mscoco_category: False`, a
   `adat_coco/images/train` + `adat_coco/annotations/train.json` (és val) útvonalak.
2. Tanítás COCO-előtanított súlyról (az Objects365-ös súlyok adatfeltételeit lásd a licenc-doksiban):
   `python train.py -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml --use-amp --seed=0 -t dfine_s_coco.pth`
3. Export: `python tools/deployment/export_onnx.py --check -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml -r output/.../best_stg2.pth`
   (kimenet: `labels`, `boxes`, `scores`, bemenet: `images` + `orig_target_sizes`).

Motor-beállítás: `"onnx": {"preset": "detr"}`, `"onnx_class_names": ["ball", "goalkeeper", "player", "referee"]`.

### c) YOLOX (Megvii, Apache-2.0) – a már kipróbált út, egyszerű ONNX

1. A YOLOX forráskódjában `exps/example/custom/yolox_s.py` mintájára saját exp-fájl:
   `self.num_classes = 4`, `self.data_dir = "adat_coco"`, `self.train_ann = "train.json"`,
   `self.val_ann = "val.json"`, és kis labdához nagyobb bemenet: `self.input_size = self.test_size = (960, 960)`.
   (A YOLOX COCO-betöltője a `train2017` / `val2017` mappanevet várja – a `name` paraméterrel vagy a mappák
   átnevezésével igazítsd.)
2. Tanítás: `python tools/train.py -f exps/sajat/focis_yolox_s.py -d 1 -b 16 --fp16 -o -c yolox_s.pth`
3. Export: `python tools/export_onnx.py --output-name focis_yolox_s.onnx -f exps/sajat/focis_yolox_s.py -c YOLOX_outputs/focis_yolox_s/best_ckpt.pth`

Motor-beállítás: a háttér felismeri (`yolox`); `"onnx_class_names": ["ball", "goalkeeper", "player", "referee"]`
(a YOLOX ONNX nem tárolja a neveket).

## 6. Beépítés és ellenőrzés

1. A `.onnx` fájl a `models/` mappába (a `.gitignore` kizárja – **modellfájlt ne tegyél a repóba**).
2. `model_config.json` (minta: `model_config.onnx.example.json`): `"backend": "onnx"`, `"onnx_path"`,
   `"onnx_class_names"`, szükség esetén `"onnx"` / `"input_size"`.
3. Gyors ellenőrzés egy képen: `python -m detection.onnx_detector models\focis_detektor.onnx kep.jpg`.
4. Szintetikus végponttól végpontig mérés a valódi modellel (a saját követővel):
   ```bat
   set E2E_MODES=onnx_owntracker
   set E2E_ONNX_PATH=models\focis_detektor.onnx
   python tests\test_e2e_synth.py
   ```
   (a szintetikus figurák csak durva ellenőrzésre jók – a valódi mérce egy kézzel címkézett, a
   tanításból kihagyott saját meccsrészlet: személy-felidézés / pontosság, labda-felidézés.)
5. Elemzés egy valódi klipen, a naplóban (`analysis_debug_*.log`) a `Modell:` sor mutatja a hátteret és
   a végrehajtót (CPU / CUDA / DirectML).
6. Modellkártya: keretrendszer + előtanító súly (licenc), tanító meccsek listája, dátum, mért pontosság
   – a kereskedelmi kiadás ellenőrzőlistájához.

## 7. Jelenetek címkézése laptopon (eseményfelismerés javítása)

A képcímkézés (fenti pontok) a felismerő "szemét" tanítja; az eseményeket (passz, beadás, lövés, szöglet …)
szabályok döntik el. Ezek javításához az edző jelöli, mi történt valójában – ehhez nem kell videókártya
és Visual Studio sem, csak egy böngésző:

```bat
REM az elemző gépen: rövid részletek minden eseményhez + 20 véletlen, nem jelölt jelenet
python tools\cimkezo_csomag.py keszit --esemenyek clips\match_events_ab12.json --ki cimkezo_meccs1 --veletlen 20
```

A `cimkezo_meccs1` mappát (pendrive / Google Drive) a laptopra másolva az `index.html` böngészőben megnyílik:
minden jelenetnél a program tippje; ✓ Jó / ✗ Nem jó → mi volt valójában (típus, csapat, a pillanat, megjegyzés).
A végén „Mentés fájlba” → `cimkek_<meccs>.json`, amit az elemző gépen beolvasunk:

```bat
python tools\cimkezo_csomag.py beolvas --cimkek cimkek_match_events_ab12.json
python tools\eval_feedback.py clips\feedback_match_events_ab12.json clips\match_events_ab12.json
```

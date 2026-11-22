# Focielemző (TacAnaly)

Veo-felvételek (labdakövető, forgó kamera) automatikus elemzése: a 90 perc kivágása,
játékos- és labdakövetés, csapatok felismerése mezszín alapján, lövések, gólok és
kapus események, hőtérkép, jelenetek kivágása, taktikai tábla és felülnézeti visszajátszás.

- **Elemző motor:** Python (FastAPI szerver, `main.py`) – ezt hívja a MAUI program.
- **Felület:** .NET MAUI (külön projekt), valamint a böngészős taktikai tábla (`/tabla`).

## Telepítés (Windows, egyszer)

1. **Python 3.10–3.12** és **ffmpeg**: az `ffmpeg.exe` legyen a PATH-on, vagy add meg a
   helyét a `FFMPEG_PATH` környezeti változóban.
2. A program mappájában:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **PyTorch** – NVIDIA kártyához (pl. RTX 3060) a CUDA-s változat kell. A pontos parancsot a
   https://pytorch.org/get-started/locally/ oldal adja meg (Windows, Pip, CUDA kiválasztásával).
   Integrált videókártyás laptopon elég a sima `pip install torch`.
4. A többi csomag:
   ```powershell
   pip install -r requirements.txt
   ```
5. Focira tanított modell (ajánlott): lásd `docs/MODELL_BEALLITAS.md`. Enélkül az általános
   `yolov8x.pt` / `yolov8n.pt` modellt használja (az első elemzéskor tölti be / tölti le – a szerver
   indulását ez nem lassítja; a `models\` mappába tett súlyfájlt is megtalálja).

## Indítás

**Szerver (a MAUI programhoz és a taktikai táblához):**
```powershell
python main.py
```
- Taktikai tábla: http://localhost:8000/tabla
- Egy elemzett meccs visszajátszása: `http://localhost:8000/tabla?events=/clips/events_<azonosító>.json`
- Rajzolás a meccsvideóra („Videó elemzés” mód), sarokban a felülnézettel:
  `http://localhost:8000/tabla?video=/clips/raw_<azonosító>.mp4&events=/clips/events_<azonosító>.json`
  - A rajzok (nyíl, futás, vonal, karika, zóna, toll, felirat, kiemelés) a videó egy képkockájához
    tapadnak („rajz-megálló”): lejátszáskor ott megáll. Billentyűk: szóköz, ← →, vessző / pont (képkocka).
  - „Videó exportálása rajzokkal” / „Részlet mentése”: MP4, a megállóknál kimerevítve a rajzzal
    (`POST /burn-stops` háttérfeladat, lásd `video_annotate.py`). A szerver videói: `GET /videos`.
  - Időillesztés a felülnézethez: `raw_<id>.mp4` → elemzési idő = `source.start_sec` + videóidő;
    `match_<id>.mp4` (vágott meccs) → félidőnként az események `full_match_sec` mezőjéből; eredeti videó → azonos.
  - Ékezetes feliratokhoz az exportban a Pillow ajánlott (`pip install pillow`); nélküle az OpenCV rajzol, ékezet nélkül.

**Teljes meccs elemzése parancssorból (MAUI nélkül):**
```powershell
python elemzes.py MECCS.mp4 --felido-perc 45 --kimenet eredmeny
```
Hasznos kapcsolók (`python elemzes.py --help`):
- `--gyors 3` – integrált videókártyán: felismerés csak minden 3. képkockán
- `--felidok 15:05,1:00:58,1:16:10,2:03:02` – kézi félidőhatárok, ha az automatikus téved
- `--kalibracio "x,y;x,y;x,y;x,y"` és `--kalibracio-pontok "..."` – kézi pályakalibráció
  (nélküle a `MECCS.mp4.kalibracio.json` fájlt használja, ha van – lásd lent –, különben a
  pályavonalakból kalibrál)
- `--csapat1-balra` – ha a hazai csapat az 1. félidőben balra támad
- `--hatterben` – laptopon: alacsony prioritás és kevés szál, hogy közben dolgozni lehessen
  (`--szalak N` a szálszámot is megadja)

### Kalibráció a kattintós eszközzel

Ha a kimeneti `raw_<id>.mp4` videón nem látszanak (vagy elcsúszva látszanak) a halvány cián
pályavonalak, az automatikus kalibráció nem boldogult a felvétellel (sötét / esti kép, kopott
vonalak, a kerítés mögötti másik pálya, a megszokottól eltérő kamerahely). Ilyenkor egyszer, pár perc
alatt kézzel kalibrálhatsz – a program utána ezt használja (a Veo kamerája egy félidőn belül nem
mozdul el, így egyetlen képkocka elég):

1. Indítsd el az eszközt egy olyan időpontnál, ahol sok vonal látszik (pl. a kezdőkör és a 16-os):
   ```powershell
   python tools\kalibracio.py MECCS.mp4 --ido 10:00
   ```
   Nem szabványos (pl. 100 x 64 m-es) pályán add meg a méretet is: `--palya 100x64`.
2. Bal oldalt a videó képe, jobb oldalt a pálya rajza számozott pontokkal (szögletek, a 16-os és az
   5-ös sarkai, a felezővonal végei, a kezdőpont, a kezdőkör és a felezővonal metszéspontjai,
   a büntetőpontok, a 16-os ív és a 16-os vonal metszéspontjai).
   - `a` / `d`: ±1 mp, `A` / `D` (vagy `j` / `l`): ±10 mp – keress olyan képet, ahol sok vonal látszik
     (időváltáskor a már felvett pontok törlődnek);
   - kattints egy pontra a **videón** (görgővel nagyíthatsz, a kurzor melletti nagyító segít a
     pontos kattintásban), majd a hozzá tartozó pontra a **rajzon** (a legközelebbi jelölt pontra ugrik);
   - ismételd legalább **4 pontpárra** – ajánlott **5–8**, a képen minél szétszórtabban (ne mind egy
     vonalon legyen!);
   - 4 pártól a program cián színnel rárajzolja a videóra a kiszámolt pályavonalakat, és minden pontnál
     kiírja a hibát. Ha a cián vonalak a fehér vonalakon futnak, jó a kalibráció; ha egy pontnál nagy
     a hiba, valószínűleg rossz pontot választottál a rajzon: `Backspace` / `u` visszavonja, `r`
     mindent újrakezd;
   - `Enter`: kész. (`Esc` / `q`: kilépés mentés nélkül, `h`: súgó, `f`: nagyítás vissza.)
3. Az eszköz kiírja a bemásolható kapcsolókat
   (`--kalibracio "..." --kalibracio-pontok "..." --kalibracio-ido ...`), és elmenti őket a
   `MECCS.mp4.kalibracio.json` fájlba. Az `elemzes.py` ezt **magától** használja, ha nem adsz meg
   `--kalibracio` kapcsolót (ezt induláskor ki is írja; kikapcsolás: `--nincs-kalibracios-fajl`).
4. Ha a szünetben odébb tették a kamerát, a 2. félidőt is kalibráld egy 2. félidei időpontnál:
   ```powershell
   python tools\kalibracio.py MECCS.mp4 --ido 55:00 --felido 2
   ```
   Ugyanabba a fájlba kerül; mindkét félidő a saját kalibrációját kapja. (Ha csak az 1. félidőt
   kalibrálod, a 2.-ban a program automatikusan kalibrál.)
5. Ellenőrzés ablak nélkül: `python tools\kalibracio.py MECCS.mp4 --elonezet ellenorzes.jpg` – a
   mentett kalibráció vonalait a kalibráció képére rajzolja.

Az eszköz grafikus kijelzőt és az `opencv-python` csomagot igényli (a `opencv-python-headless`
nem nyit ablakot – erről érthető hibaüzenetet ad). A kalibráció időpontja bárhol lehet a félidőn belül:
ha messze esik a félidő elejétől, a program a kalibrációból a kamera helyét veszi át, a kamera
irányát pedig a pályavonalakból keresi meg; ha a követés közben elveszik, szintén ehhez a kamerahelyhez
igazodva keres újra.

**Pályaméret:** az elemző motor (pályakövető, események, hőtérkép, labdakövetés, áttekintő lapok)
jelenleg a 105 x 68 m-es pályával számol. A `--palya` értéket az eszköz elmenti és a rajzot ehhez
igazítja, az `elemzes.py` pedig figyelmeztet, ha eltér – a teljes támogatáshoz a motor több
moduljában kell a méretet paraméterezni (lásd `docs/FEJLESZTESI_TERV.md`).

**Gyors vágás:** a félidők, a meccsvideó és a klipek képkocka-pontosan, mégis percek alatt készülnek
(csak a vágási pontoknál lévő rövid rész kódolódik újra, hardveres kódolóval, ha van). A szerveres
háttérfeladatnál (`/jobs/analyze-match`) a vágott meccsvideó már az elemzés közben elérhető, a feladat
szüneteltethető (`/jobs/{id}/pause`, `/resume`). Részletek: `docs/HATTERFELADATOK.md`.

**Asztali alkalmazásként (MAUI-ból indítva, Python nélkül):** a motor PyInstallerrel önálló
programmá csomagolható (`packaging\build_engine.ps1`); a MAUI a `tacanaly_engine.exe --port 0
--parent-pid <pid>` paranccsal indítja, a `TACANALY_READY port=...` sorból olvassa a portot, és
`/health`, `/shutdown`, `/models/*` végpontokkal kezeli. Ilyenkor az adatok a
`%LOCALAPPDATA%\TacAnaly` mappába kerülnek. Részletek: `packaging/README_CSOMAGOLAS.md`.
Fejlesztéshez ugyanez: `python engine_launcher.py --port 0 --data-dir adat --console`.

## Mit kapsz eredményül (a kimeneti mappában)

- `match_<id>.mp4` – a meccs a szünet nélkül; `half1_` / `half2_` – a félidők
- `raw_<id>.mp4` – az elemzett félidő kirajzolt játékosokkal, labdával és halvány pályavonalakkal
  (ebből látszik, jó-e a kalibráció)
- `events_<id>.json`, `match_events_<id>.json` – események, statisztika, játékos- és labdapályák
- `review_<id>/index.html` – **áttekintő lapok**: minden eseményről egy rácsos kép, percenként
  áttekintés – a teljes meccs pár képen átnézhető
- hőtérkép-képek (csapat és játékos), kapus- és összefoglaló videó-összeállítások

## Dokumentáció

- `docs/FEJLESZTESI_TERV.md` – a teljes felépítés, minden végpont mezőkkel
- `docs/HATTERFELADATOK.md` – háttérfeladatok (haladásjelzés, szüneteltetés, háttér-mód), gyors vágás, parancssor
- `docs/MODELL_BEALLITAS.md` – modell, gyors mód, osztálynevek, detektor-háttér (ONNX / ultralytics)
- `docs/MODELL_TANITAS.md` – saját focis detektor tanítása (képkocka-export, címkézés, ONNX export)
- `docs/LICENC_ES_ARUSITAS.md` – licencek és a kereskedelmi kiadás ellenőrzőlistája (`THIRD_PARTY_NOTICES.md`)
- `packaging/README_CSOMAGOLAS.md` – telepítő, a motor indítása / leállítása a MAUI-ból, licencek

## Tesztek

Videókártya és YOLO nélkül is futnak (szintetikus adatokon):
```powershell
python tests\test_events.py
python tests\test_pitch_tracker.py
python tests\test_fast_cut.py      (ffmpeg + ffprobe kell; a vágás pontossága és sebessége)
```
`tools/synth_match.py` valósághű, forgó kamerás tesztmeccset készít ismert eseményekkel;
`tests/test_e2e_synth.py` ezen méri a teljes lánc pontosságát.

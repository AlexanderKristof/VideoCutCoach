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

**Teljes meccs elemzése parancssorból (MAUI nélkül):**
```powershell
python elemzes.py MECCS.mp4 --felido-perc 45 --kimenet eredmeny
```
Hasznos kapcsolók (`python elemzes.py --help`):
- `--gyors 3` – integrált videókártyán: felismerés csak minden 3. képkockán
- `--felidok 15:05,1:00:58,1:16:10,2:03:02` – kézi félidőhatárok, ha az automatikus téved
- `--kalibracio "x,y;x,y;x,y;x,y"` és `--kalibracio-pontok "..."` – kézi pályakalibráció
  (nélküle a program a pályavonalakból kalibrál)
- `--csapat1-balra` – ha a hazai csapat az 1. félidőben balra támad

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
- `docs/HATTERFELADATOK.md` – háttérfeladatok (haladásjelzés a MAUI-ban), parancssor
- `docs/MODELL_BEALLITAS.md` – modell, gyors mód, osztálynevek
- `packaging/README_CSOMAGOLAS.md` – telepítő, a motor indítása / leállítása a MAUI-ból, licencek

## Tesztek

Videókártya és YOLO nélkül is futnak (szintetikus adatokon):
```powershell
python tests\test_events.py
python tests\test_pitch_tracker.py
```
`tools/synth_match.py` valósághű, forgó kamerás tesztmeccset készít ismert eseményekkel;
`tests/test_e2e_synth.py` ezen méri a teljes lánc pontosságát.

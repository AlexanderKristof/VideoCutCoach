# Focielemző – fejlesztési terv

## Áttekintés

- **Felület:** .NET MAUI (C#, XAML) – csak Windows asztali gépen fut.
- **Elemző motor:** Python (felismerés, követés, eseménydetektálás, vágás).
- **Videóforrás:** főleg Veo felvétel – labdakövető, forgó kamera, egyszerre egy térfelet mutat.
- **Hardver:** NVIDIA RTX 3060 (pontos mód) és integrált videókártyás laptop (gyors mód).

### A két rész kapcsolata

A MAUI program helyi folyamatként vagy helyi szerverként (pl. FastAPI, `localhost`) indítja a Python motort.
- Az elemzés hosszú (percek–órák), ezért a Python folyamatosan jelenti a haladást (pl. JSON sorok: `{"fazis": "kovetes", "szazalek": 42}`).
- Az eredmény egy meccs-JSON: események időbélyeggel, játékospozíciók a 2D pályán, statisztikák.
- A klipek vágását a Python végzi ffmpeg-gel, a MAUI csak listáz, előnézetet mutat és exportál.

---

## Alapötlet: virtuális pálya („tiszta, nyers színű pálya”)

A videó minden képkockájára ráillesztünk egy **ismert méretű pálya-modellt** (vonalak, 16-os, ötös, középkör, kapuk).

1. **Pályapontok felismerése** képkockánként (vonalkereszteződések, 16-os sarkai, középkör) – kulcspont-modell.
2. **Homográfia:** ebből kiszámoljuk, hogy a kép melyik pontja a valódi pálya melyik pontja (méterben).
   - Mivel a Veo kamera forog, ezt **minden képkockára** (vagy néhány képkockánként + simítás) újra kell számolni.
3. **Ami a pálya-poligonon kívül esik, azt eldobjuk** → nézők, edző, kispad, melegítők kiesnek.
4. A játékosokat és a labdát **tiszta színekkel kirajzoljuk egy 2D (felülnézeti) pályára**:
   - A csapat = egy szín, B csapat = másik szín, kapusok külön szín, bíró sárga/fekete, labda fehér.
   - Ez a „radar / minitérkép” nézet – ebből készül a hőtérkép, a pozíciók, a passzhálózat.
5. **3D csak a labdához kell:** a földön gurul vagy a levegőben repül (ívelés vs. lapos passz).
   - Egy kamerából a magasság csak becsülhető: a labda pályájából (parabola) és látszó méretéből.

**Fontos:** a felismerő modell továbbra is az eredeti videóképen dolgozik (azon tanították), a „tiszta színes pálya” az eredmény megjelenítése és a szűrés eszköze.
Kiegészítésként a színek egyszerűsítése (pl. a mezszínek klaszterezése) segíthet a csapatok szétválasztásában.

---

## Fázisok

### 1. Felismerés tisztítása (ez a mostani probléma)
- Focira tanított modell: `játékos / kapus / bíró / labda` osztályok (beállítás: [MODELL_BEALLITAS.md](MODELL_BEALLITAS.md)).
- **Pályamaszk:** csak az a téglalap marad, amelynek **alsó közepe (a láb)** a pályán belül van.
  - Első lépés: zöld szín (HSV) alapú maszk.
  - Később: a virtuális pálya poligonja (lásd fent).
- **Követés** (ByteTrack / BoT-SORT): stabil azonosítók, a sokáig mozdulatlan szélső alakok kiszűrése.
- **Csapatba sorolás mezszín alapján** (KMeans a felsőtest színeire) – ami nem illik egyik csapatba sem: „egyéb” (edző, mellényes csere).

### 2. Labdakövetés
- Kis, gyors objektum → külön labda-modell vagy nagyobb felbontású kivágás a labda várható helye körül.
- Hiányzó képkockák pótlása interpolációval / Kalman-szűrővel.
- A Veo kamera iránya maga is jelzi, merre van a labda – ez segíthet.

### 3. Virtuális pálya, hőtérkép
- Pályapont-modell + homográfia képkockánként, simítással.
- 2D felülnézet, játékos- és csapat-hőtérkép, megtett táv, pozíciók.

### 4. Események
**Mezőny:** gól, lövés (kapura / mellé), szöglet, szabadrúgás, bedobás, szabálytalanság, beadás, egy kiválasztott játékos összes jelenete.

**Kapus fül (külön szabályok):**

| Esemény | Felismerés alapja |
|---|---|
| Védés | lövés a kapu felé → labda a kapus közelében → nem gól |
| Beadás elfogása | labda oldalról a 16-osba → kapus birtokolja |
| Kifutás / 1v1 | kapus gyorsan kimozdul a kapuból egy támadó felé |
| Kirúgás | labda áll az ötösben → hosszú repülés |
| Ívelés | kapusnál a labda → hosszú, magas repülés |
| Rövid passz / kidobás | kapus után rövid, lapos labdaút csapattárshoz |
| Kapott gól | labda a kapuba megy |

### 5. Klipek és felület
- Eseménylista szűrőkkel (típus, csapat, játékos, félidő).
- Kijelölt események klipjei: esemény előtt X mp, utána Y mp (állítható), összefűzés ffmpeg-gel.
- Statisztika oldal: lövések száma, hőtérkép, 2D minitérkép visszajátszás.
- Kapus fül saját eseménylistával és klipexporttal.

### 6. A 90 perc automatikus kivágása
- Kezdőrúgás: mindenki a saját térfelén, labda a középpontban.
- Sípszó felismerése a hangsávból.
- Félidő: a játékosok elhagyják a pályát / tömörülnek a szélen.
- Kimenet: 1. félidő + 2. félidő, a holtidő nélkül.

### 7. Saját tanítóadat
- Pár száz képkocka kijelölése saját Veo-felvételekből (pl. CVAT vagy Roboflow).
- A modell finomhangolása → nagy pontosságjavulás a saját pályákon, fényviszonyokon.

---

## Teljesítmény – két üzemmód

| | Pontos mód (RTX 3060) | Gyors mód (integrált GPU-s laptop) |
|---|---|---|
| Modell | közepes/nagy YOLO | nano/small YOLO |
| Futtatás | CUDA | ONNX + DirectML vagy OpenVINO |
| Elemzett képkockák | sűrűn (pl. minden 2.) | ritkábban (minden 3–5.), köztük követés |
| Várható idő | perctől néhány tízpercig | akár órák – háttérben, haladásjelzővel |

---

## Következő lépés

Amikor a kód felkerül a repóba: az 1. fázissal kezdünk (nézők, edzők, melegítők kiszűrése).

---

## Hőtérkép és klipexport (API a MAUI klienshez)

### Mentett pályák az események JSON-jában
A `/cut-raw` (dev_mode, kalibrációval) által írt `events_<id>.json` új mezője a `trajectories`:

```json
"trajectories": {
  "hz": 5.0,
  "tracks": {
    "12": {"team": 1, "keeper": false, "points": [[x_m, y_m, t_sec], ...]}
  }
}
```
- 5 Hz-re ritkítva, a koordináták 0.1 m-re kerekítve (105 x 68 m-es pálya, `t_sec` a teljes videóhoz mért idő).
- `team`: a követés leggyakoribb csapata (1/2), `keeper`: volt-e valaha kapusnak jelölve.
- Kalibráció nélkül a `tracks` üres (nincs pályakoordináta).

### `POST /heatmap` (form)
| Mező | Kötelező | Leírás |
|---|---|---|
| `events_path` | igen | a `/cut-raw` által visszaadott `events_path` |
| `team` | nem | `1`, `2` vagy `all` (alapértelmezett) |
| `player_id` | nem | csak ennek a követésnek (track_id) a hőtérképe |
| `flip_second_half_at_sec` | nem | ettől az időponttól az x tengelyt tükrözzük, így a két félidő egy irányba támadva összeadódik |

Válasz:
```json
{"status": "success",
 "team_heatmaps": {"team1": "/clips/heatmap_..._team1.png", "team2": "/clips/heatmap_..._team2.png"},
 "player_heatmaps": [{"track_id": 12, "team": 1, "url": "/clips/heatmap_..._p12.png", "distance_m": 812.4, "time_sec": 305.2}],
 "player_stats": [{"track_id": 12, "team": 1, "keeper": false, "distance_m": 812.4,
                   "avg_x": 31.2, "avg_y": 22.8, "time_sec": 305.2, "samples": 1526}]}
```
- Idővel súlyozott, 1 m-es rács, Gauss-simítás (2 m), 0..1-re normálva; a > 1 mp-es kimaradások nem számítanak teljes hosszal.
- Játékos-hőtérkép csak a legalább 20 mp-ig követett azonosítókhoz készül (a `player_stats` mindenkit tartalmaz).
- Megtett táv: 0.2 mp-es lépésekkel, a > 10 m/s-os ugrásokat (téves vetítés, azonosító-csere) kihagyva.
- A kép: tiszta zöld pálya fehér vonalakkal, rajta átlátszó színskála (kék → piros); ahol nincs adat, a pálya látszik.
- Hiba esetén: `{"status": "error", "message": "..."}`.

### `POST /export-clips` (form)
| Mező | Alapértelmezett | Leírás |
|---|---|---|
| `video_path` | – | az EREDETI videó útvonala |
| `events_json` | – | JSON lista: `[{"start_sec": 10, "end_sec": 18}, {"time": 42.5}, ...]` |
| `pre_sec` / `post_sec` | 5 / 4 | a csak `time`-mal megadott elemeknél az esemény előtti / utáni rész |
| `merge` | true | a klipek összefűzése egy videóvá |
| `accurate` | true | true: újrakódolás (képkocka-pontos, lassabb); false: stream copy (gyors, de csak kulcskockánál vág) |
| `title` | – | a fájlnevekbe kerül (ékezet nélkül) |

- Az átfedő vagy 1 mp-nél közelebbi ablakokat előbb összevonjuk, így ugyanaz a jelenet nem ismétlődik.
- Válasz:
```json
{"status": "success",
 "clips": [{"start_sec": 5.0, "end_sec": 17.0, "url": "/clips/Kapus_vedesek_ab12cd_01.mp4", "local_path": "C:\\...\\clips\\..."}],
 "merged_url": "/clips/Kapus_vedesek_ab12cd_osszefuzott.mp4", "merged_local_path": "C:\\..."}
```
- Egyetlen klipnél a `merged_url` maga a klip; `merge=false` esetén `null`.
- Az ffmpeg útvonala az `FFMPEG_PATH` környezeti változóval állítható (alapértelmezett: `ffmpeg.exe`).
- Hiba esetén (üres lista, hibás időablak, hiányzó videó / ffmpeg): `{"status": "error", "message": "..."}`.

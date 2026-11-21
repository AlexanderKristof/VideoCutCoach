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
- **Megvalósítva:** háttérfeladatok haladásjelzéssel (`POST /jobs/analyze-match`, `GET /jobs/{id}`) és a teljes
  meccs elemzése egy lépésben, parancssorból is (`python elemzes.py MECCS.mp4`) – lásd [HATTERFELADATOK.md](HATTERFELADATOK.md).

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

#### Megvalósítva (`match_segmenter.py`, végpontok a `main.py`-ban)

**Működés:** a hangsávból sípszavakat keres (tartós, keskeny sávú hang kb. 2,5–4,5 kHz között, ≥ 0,3 mp; az 1 mp-en belül egymást követő sípolások egy csoport, pl. a lefújás 3 sípszava),
a képből másodpercenként egy kicsinyített képkockán méri a fű arányát, a füvön látható mozgást (a kamera pásztázását kompenzálva) és magát a pásztázást.
Ebből „játék folyik” pontszám lesz, amit simít, majd két hihető hosszúságú (alapból a várt félidőhossz 44–125%-a), legalább 3 perc szünettel elválasztott szakaszt választ;
a határokat a ±90 mp-en belüli erős sípszóhoz igazítja. Két óra videón a futásidő főleg a dekódolás (néhány perc); a hang elemzése gyors.
Az ffmpeg neve az `FFMPEG_PATH` környezeti változóval felülírható (alap: `ffmpeg.exe`).

**`POST /detect-halves`** (form mezők)

| Mező | Típus | Alap | Jelentés |
|---|---|---|---|
| `video_path` | szöveg | – | a teljes felvétel helyi elérési útja |
| `expected_half_min` | szám | `45` | várt félidőhossz percben (amatőr: 40 vagy 35) |
| `use_person_count` | bool | `false` | YOLO létszámszámlálás is (pontosabb, lassabb) |

Válasz:
```json
{
  "status": "success",
  "first_half":  {"start_sec": 905.0, "end_sec": 3658.0, "duration_sec": 2753.0, "start_hms": "15:05", "end_hms": "1:00:58",
                  "start_snapped_to_whistle": true, "end_snapped_to_whistle": true},
  "second_half": {"start_sec": 4570.0, "end_sec": 7382.0, "...": "..."},
  "confidence": 0.82,
  "whistles": [{"time_sec": 905.0, "end_sec": 905.6, "duration_sec": 0.6, "strength": 0.5, "freq_hz": 3200, "blasts": 1}],
  "notes": ["Az 1. félidő kezdete sípszóhoz igazítva (15:05)."],
  "duration_sec": 8000.0,
  "play_timeline": [[0.0, 0.12], [5.0, 0.13]]
}
```
- `second_half` lehet `null`: túl rövid videó, vagy nem található két félidő – ilyenkor a `first_half` a teljes videó, a `notes` megmondja, miért.
- `confidence` < 0,5: a kliens kérje meg a felhasználót a határok ellenőrzésére (a `play_timeline` grafikonon megjeleníthető, a `whistles` jelölőként az idősávon).
- Hiba esetén: `{"status": "error", "message": "..."}`.

**`POST /cut-match`** (form mezők)

| Mező | Típus | Alap | Jelentés |
|---|---|---|---|
| `video_path` | szöveg | – | a teljes felvétel |
| `expected_half_min` | szám | `45` | mint fent (csak ha felismerés is fut) |
| `first_start`, `first_end`, `second_start`, `second_end` | szöveg | üres | kézi felülírás: mp (`754.5`) vagy `pp:mm` / `ó:pp:mm`; üres = automatikus |
| `padding_sec` | szám | `5` | ennyi mp ráhagyás a félidők elé és mögé |
| `use_person_count` | bool | `false` | mint fent |

Ha mind a négy kézi érték meg van adva, nincs felismerés (gyors). Részleges megadásnál a felismert értékeket írja felül.
A vágás újrakódolás nélküli (`-c copy`), ezért a kezdet a legközelebbi kulcskockára esik – ezt a ráhagyás fedi.

Válasz:
```json
{
  "status": "success",
  "halves": {"first_half": {"start_sec": 905.0, "end_sec": 3658.0, "start_hms": "15:05", "end_hms": "1:00:58"},
             "second_half": {"start_sec": 4570.0, "end_sec": 7382.0, "start_hms": "1:16:10", "end_hms": "2:03:02"}},
  "confidence": 0.82,
  "notes": ["..."],
  "first_half_url": "/clips/half1_ab12cd34.mp4",
  "second_half_url": "/clips/half2_ab12cd34.mp4",
  "full_match_url": "/clips/match_ab12cd34.mp4",
  "local_paths": {"first_half": "C:\\...\\clips\\half1_ab12cd34.mp4", "second_half": "...", "full_match": "..."}
}
```
A `full_match_url` a két félidő összefűzve, a szünet nélkül. Javasolt kliens-folyamat: `/detect-halves` → a felhasználó ellenőrzi / módosítja a határokat → `/cut-match` mind a négy kézi értékkel.

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

## Automatikus csapatba sorolás (team_classifier.py)

A kézi HSV-tartományok (az edzőnek a mezre kell kattintania) helyett a motor magától tanulja meg a két mezszínt.

### Mikor melyik mód?
- **Kézi mód** marad, ha a kliens a `/cut-raw` hívásban `team1_hsv_*` / `team2_hsv_*` tartományt ad (nem a teljes `0,0,0`–`179,255,255` tartományt) – a kézi beállítás mindig nyer.
- **Automatikus mód**: `auto_teams=true` (alapértelmezett) és nincs csapat-tartomány megadva.
- A kapus-tartományok (`keeper1/2_hsv_*`) és a modell kapus osztálya automatikus módban is érvényesek.
- Új form mezők: `auto_teams: bool = true`, `swap_teams: bool = false` (a két csapat felcserélése, ha a sorrend nem a kívánt).

### Működés
1. **Mezjellemző** detektálásonként (kb. 0.12 ms): felsőtest (a doboz 20–55%-a magasságban, középső 60% szélesség) → 12x16-os folt; a fűszínű pixelek kimaradnak (ha szinte minden „fű”, akkor zöld mezről van szó, és mindent megtartunk); a világosság szélső 10–10%-a (árnyék, csillanás) kimarad; Lab medián (L kisebb súllyal) + 8 rekeszes, telítettséggel súlyozott színezet-hisztogram + 1 színtelen rekesz. Nincs detektálásonkénti KMeans.
2. **Bemelegítés**: az első 12 mp-ben (vagy 1500 mintáig) követésenként 0.2 mp-enként gyűjtünk mintát. `cv2.kmeans` k=7-tel, majd:
   - „Csapatrész” az a klaszter, amely sok különböző követésből áll (effektív követésszám ≥ 3.5, méret ≥ 4%). A bíró és a kapus klasztere jellemzően 1–2 követésből áll, ezért „egyéb” marad.
   - A csapatrészeket (pl. napos és árnyékos fehér mez) a középpontok távolsága szerint két csoportba vonjuk össze → a két csapat.
   - A csapathoz nagyon közeli „egyéb” klaszter a csapathoz kerül.
   - A végleges illesztésig 150 mintánként ideiglenes illesztés készül, így az elejétől van (ideiglenes) besorolás; az illesztés előtti legelső képkockákon a régi tartalék szabály (piros > kék) dönt.
3. **Online besorolás**: legközelebbi klaszter (a klaszter szórásával normált távolság); ami minden klasztertől messze van (z > 4), az „egyéb” szavazat (új, eddig nem látott mez). Követésenként exponenciálisan felejtő szavazás (felezési idő 1 mp) hiszterézissel (váltás csak 1.3-szoros előnynél): 20% hibás kivágásnál is stabil.
4. **Szerepek**:
   - „Egyéb” mezű követés **kapus**, ha a pályakoordinátái legalább 60%-ban a kapu előtti 20 m-es sávban vannak, vagy a modell kapus osztályt ad; a csapata az, amelyik azt a kaput védi (a csapatok átlagos x-e alapján).
   - Különben **bíró** (pályakoordinátával) vagy **egyéb** (kalibráció nélkül). Mindkettő semleges kerettel rajzolódik, és nem kerül a játékosok közé.
   - Az automatikusan kapusnak minősült követést az álló alak szűrő nem dobja el.
5. **Csapatsorrend** (determinisztikus): az **1. csapat** az, amelyiknek a játékosai a bemelegítés alatt átlagosan a **bal térfélen** (kisebb x) voltak; ha nincs elég pályakoordináta (nincs kalibráció), a nagyobb klaszter. `swap_teams=true` megcseréli. Automatikus módban tehát a klip elején az 1. csapat védi a bal (x=0) kaput – ehhez illik a `team1_attacks_right=true`.

### Kimenet
Az `events_<id>.json` új mezői:
```json
"team_colors": {"team1": [r, g, b], "team2": [r, g, b], "other": [r, g, b] | null},
"team_mode": "auto"
```
Kézi módban `"team_colors": null, "team_mode": "manual"`. A debug naplóba a tanult klaszterek (szerep, szín, mintaszám, szétválás) is bekerülnek.

### Tesztek (`python3 tests/test_team_classifier.py`)
Szintetikus kivágások (fű háttér, 20–120 px, árnyék, csillanás, elmosódás, zaj, takarás), 20 mezőnyjátékos + 2 kapus (zöld, narancs) + bíró (sárga/fekete):
- piros–fehér és kék–égszínkék: ~100% csapat-pontosság; a bíró 100%-ban, a kapusok ~97%-ban nem kerülnek mezőnyjátékosként csapatba;
- fekete–nagyon sötétkék (nehéz pár): ~81% (a világosság-ingadozás itt nagyobb, mint a két mez különbsége);
- 20% hibás kivágás: egyedi besorolás ~80%, szavazással 100%;
- költség: ~0.14 ms / detektálás; a (ritka) illesztés 1600 mintán ~160 ms.

### Korlátok
- A tanult színek a klip elejéhez igazodnak; nagy fényviszony-változásnál (este, reflektor) új klipnél újra tanul, de egy klipen belül nincs újratanulás.
- Ha a bíró / kapus mez a csapatéhoz hasonló (pl. fekete bíró – fekete mez), nem választható szét.
- Kalibráció nélkül a kapus nem ismerhető fel pozíció alapján (csak a modell kapus osztályával vagy kézi színnel).
- Az illesztés előtt kiadott ideiglenes címkéket nem írjuk vissza; a pályák csapata (`trajectories`) a leggyakoribb címke, így ez alig számít.
- Ha a bemelegítés alatt csak az egyik csapat látszik (a Veo kamera egy térfélen marad), a két „csapat” rossz lehet – ilyenkor kézi tartomány vagy hosszabb klip ajánlott.

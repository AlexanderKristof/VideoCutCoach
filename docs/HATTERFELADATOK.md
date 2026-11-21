# Háttérfeladatok és a teljes meccs elemzése egy lépésben

A hosszú munkák (egy teljes meccs elemzése akár órákig tart) **háttérfeladatként (job)** futnak:
a kérés azonnal visszatér egy azonosítóval, a kliens pedig 1–2 másodpercenként lekérdezi a haladást.
A régi, szinkron végpontok (`/cut-raw`, `/detect-halves`, `/cut-match`, `/heatmap`, `/export-clips`, …)
**változatlanul működnek** – a MAUI kliens a meglévő hívásait nem kell, hogy átírja.

Fájlok:
- `jobs.py` – a jobkezelő (szálkészlet, állapot, haladás, megszakítás, mentés `clips/jobs/<id>.json`-ba).
- `pipeline.py` – a teljes meccs folyamata (HTTP nélkül is hívható, a tesztek hamis függvényekkel futtatják).
- `elemzes.py` – parancssori indítás (szerver és kliens nélkül).
- `fast_cut.py` – gyors, képkocka-pontos vágás ("smart cut") és összefűzés.
- `background.py` – háttér-mód: alacsony prioritás, szálkorlát (laptopon a gép közben használható marad).

## A teljes folyamat lépései

| Lépés | Mit csinál | Haladás súlya |
|---|---|---|
| Félidők felismerése | `match_segmenter.detect_match_halves` (vagy kézi határok) | 10% |
| Vágás | szünet nélküli meccsvideó (`cut_match_halves`, képkocka-pontos smart cut), `skip_cut=true` esetén kimarad; a kész fájlok azonnal megjelennek a job részeredményében | 5% |
| Elemzés | mindkét félidő `cut_and_analyze_sync`-kel; a 2. félidőben a `team1_attacks_right` megfordul (térfélcsere) | 68% (a félidők hossza szerint elosztva) |
| Összefésülés | a két félidő eseményei egy meccs-JSON-ba (`match_events_<id>.json`) | 1% |
| Hőtérképek | félidőnként + összesítve (a 2. félidő tükrözve, csak csapat-hőtérkép) | 4% |
| Klip-összeállítások | kapusonként egy videó + összefoglaló (gólok, lövések) | 12% |

A kikapcsolt lépések súlyát a többi kapja, így a haladás mindig 0-tól 100-ig megy, és sosem csökken.

A vágás az elemzés **előtt** fut, és percek alatt kész (lásd lent: smart cut) – a `match_<id>.mp4` és a
`half1_` / `half2_` fájlok így már akkor lejátszhatók, amikor a több órás elemzés még csak elindult
(részeredmény: `result.reszleges = true`, `result.kesz_reszek`).

Az elemzés mindig az **eredeti** videón fut (a félidők határaival), ezért az események `time`, `start_sec`,
`end_sec` mezői az eredeti videó idejét mutatják, és a klipek is az eredeti videóból készülnek.

## HTTP végpontok

### `POST /jobs/analyze-match` (form)

| Mező | Alap | Leírás |
|---|---|---|
| `video_path` | – | a teljes felvétel helyi útvonala |
| `expected_half_min` | `45` | várt félidőhossz percben |
| `use_person_count` | `false` | YOLO létszámszámlálás a félidő-felismeréshez |
| `first_start`, `first_end`, `second_start`, `second_end` | üres | kézi félidőhatárok (mp vagy `pp:mm` / `ó:pp:mm`); ha mind a négy megvan, nincs felismerés |
| `skip_cut` | `false` | ne készüljön szünet nélküli meccsvideó |
| `padding_sec` | `5` | ráhagyás a vágott félidők elején / végén |
| `heatmaps` | `true` | hőtérképek készítése |
| `compilations` | `goalkeepers,highlights` | klip-összeállítások vesszővel: `goalkeepers` (kapusonként), `highlights` (gólok + lövések); üres = nincs |
| `clip_pre_sec` / `clip_post_sec` | `5` / `4` | a csak időponttal megadott eseményeknél a klip eleje / vége |
| `clip_accurate` | `true` | képkocka-pontos klipek (smart cut: csak a klip eleje / vége kódolódik újra) |
| `background` (vagy `hatter`) | `false` | háttér-mód: alacsony prioritás, kevés szál (lásd lent) |
| `threads` (vagy `szalak`) | üres | szálkorlát (háttér-módban alapból a magok fele, akkumulátorról a negyede) |
| a `/cut-raw` összes hangolási mezője | ugyanaz, mint ott | `calib_points`, `roi_top_pct`, `min_player_height_pct`, `use_static_roi`, `grass_min_ratio`, `stationary_window_sec`, `stationary_max_move`, `pitch_margin_m`, `*_hsv_lower` / `*_hsv_upper`, `team1_attacks_right` (az 1. félidőre), `review_export`, `detect_mode` |

Az ismert mezők listája a `pipeline.py` `ANALYSIS_DEFAULTS` / `PIPELINE_DEFAULTS` szótára (új mezőt ott kell felvenni;
a teszt ellenőrzi, hogy a `/cut-raw` alapértékeivel egyezik). Ismeretlen mezőket a szerver figyelmen kívül hagy.

Válasz (azonnal): `{"status": "success", "job_id": "3f2a9c1b7d4e", "status_url": "/jobs/3f2a9c1b7d4e"}`
Hibás mező / hiányzó videó: HTTP 400, `{"status": "error", "message": "..."}`.

### `GET /jobs/{id}`

```json
{"status": "success",
 "job": {"id": "3f2a9c1b7d4e", "kind": "analyze-match", "status": "running",
         "progress": 42.7, "phase": "1. félidő elemzése: 31250/67500 képkocka",
         "created_at": "2026-11-21T14:02:11", "started_at": "2026-11-21T14:02:11", "finished_at": null,
         "elapsed_sec": null, "cancel_requested": false, "paused": false, "background": true, "threads": 2,
         "params": {"video_path": "...", "...": "..."},
         "result": null, "error": null, "log_path": null}}
```
- `status`: `queued` (vár), `running`, `paused` (szüneteltetve), `done`, `error`, `cancelled`,
  `megszakadt` (a szerver újraindult futás közben).
- `paused`: szüneteltetést kértek (a `status` akkor lesz `paused`, amikor a feladat ténylegesen megállt).
- `background` / `threads`: háttér-mód és szálkorlát.
- `result` futás közben is lehet: **részeredmény** (`"reszleges": true`, `"kesz_reszek": [...]`), lásd lent.
- `progress`: 0–100, sosem csökken; `phase`: magyar szöveg a kijelzéshez.
- `error`: rövid hibaüzenet; a teljes traceback a `log_path` fájlban (`clips/jobs/<id>.log`).
- Ismeretlen azonosító: HTTP 404.

A kész `analyze-match` job `result` mezője:
```json
{"status": "success", "video_path": "C:\\...\\meccs.mp4", "run_id": "ab12cd34", "out_dir": "C:\\...\\clips",
 "halves": {"first_half": {"start_sec": 905.0, "end_sec": 3658.0, "start_hms": "15:05", "end_hms": "1:00:58"},
            "second_half": {"start_sec": 4570.0, "end_sec": 7382.0, "...": "..."}},
 "detection": {"confidence": 0.82, "duration_sec": 8000.0, "manual": false},
 "cut": {"full_match_url": "/clips/match_x.mp4", "first_half_url": "...", "second_half_url": "...", "local_paths": {"...": "..."}},
 "analysis": {"first_half":  {"clip_url": "/clips/raw_1.mp4", "local_path": "...", "events_path": "...", "events_url": "/clips/events_1.json",
                              "review_url": "/clips/review_1/index.html", "event_count": 57, "team1_attacks_right": true},
              "second_half": {"...": "...", "team1_attacks_right": false}},
 "match_events_path": "C:\\...\\clips\\match_events_ab12cd34.json", "match_events_url": "/clips/match_events_ab12cd34.json",
 "event_count": 118, "event_counts": {"shot": 14, "goal": 3, "gk_save": 6},
 "heatmaps": {"first_half": {"status": "success", "team_heatmaps": {"...": "..."}, "player_heatmaps": [], "player_stats": []},
              "second_half": {"...": "..."}, "combined": {"...": "..."}},
 "compilations": {"goalkeeper_team1": {"status": "success", "title": "Hazai Kapus", "item_count": 9, "merged_url": "/clips/...mp4", "clips": []},
                  "goalkeeper_team2": {"...": "..."}, "highlights": {"title": "Osszefoglalo", "...": "..."}},
 "notes": ["Az 1. félidő kezdete sípszóhoz igazítva (15:05)."],
 "warnings": ["Hőtérkép (first_half): Az eseményfájlban nincsenek pályakoordináták (kalibráció nélkül futott az elemzés?)."],
 "elapsed_sec": 5321.4}
```
A hőtérkép vagy klip-összeállítás hibája nem buktatja el a jobot: a `warnings` listába kerül.

**Az összesített meccs-JSON** (`match_events_<id>.json`) eseményei a félidők eseményei, kiegészítve:
- `half` (1 / 2), `half_label` („1. félidő”),
- `time_in_half` – a félidő kezdetétől eltelt mp,
- `time_in_match` – a szünet nélküli játékidő (a 2. félidőben: az 1. félidő hossza + `time_in_half`),
- `match_clock` – hivatalos óra `pp:mm` (a 2. félidő `expected_half_min`-től indul, pl. 45:00),
- `full_match_sec` – az időpont a vágott, szünet nélküli meccsvideóban (csak ha volt vágás).

Ezen kívül: `halves`, `match_stats` félidőnként, `review_urls`, `trajectories` (a 2. félidő követési
azonosítói +100000-rel eltolva, `half` és `original_track_id` mezővel). A `/heatmap` és a `/tabla` is megnyitja.

### `GET /jobs?limit=50`
A jobok listája (legújabb elöl), `result` nélkül: `{"status": "success", "jobs": [...]}`.

### `POST /jobs/{id}/cancel`
Megszakítás kérése. A sorban álló job azonnal `cancelled` lesz; a futó job a következő haladásjelzésnél
(az elemzésben 25 képkockánként) áll le, addig `cancel_requested: true` és `phase: "Megszakítás kérve…"`.
Válasz: `{"status": "success", "job": {...}}`.

### `POST /jobs/{id}/pause` és `POST /jobs/{id}/resume`
Szüneteltetés / folytatás. A szüneteltetés kooperatív: a feladat a következő haladásjelzésnél áll meg
(az elemzésben 25 képkockánként; a vágásnál a következő fájl előtt – az éppen futó ffmpeg lépés még
befejeződik), addig `paused: true` és `phase: "Szüneteltetés kérve…"`, utána `status: "paused"`,
`phase: "Szüneteltetve"`. A szüneteltetett job is megszakítható (`/cancel`). A sorban álló job is
szüneteltethető: amikor sorra kerül, el sem indul, amíg nem folytatják.
Válasz: `{"status": "success", "job": {...}}`; ismeretlen azonosító: HTTP 404.

### `POST /jobs/run` – a régi végpontok háttérben
`kind` = `cut-raw` | `detect-halves` | `cut-match` | `heatmap` | `export-clips`, mellette ugyanazok a form mezők,
mint a szinkron végpontnál. A job `result` mezője pontosan az, amit a szinkron végpont visszaadna.
Ha a régi végpont `{"status": "error"}`-t adna, a job állapota `error`, és a `result` is megvan.
A `background` / `threads` mező itt is használható (pl. `kind=export-clips&background=true`).

## Részeredmény (a vágott meccs már az elemzés közben)

Az `analyze-match` job `result` mezője futás közben is frissül (`"reszleges": true`):
- a félidők felismerése után: `halves`, `"kesz_reszek": ["felidok"]`;
- minden kész vágott fájl után: `cut.first_half_url` / `cut.second_half_url` / `cut.full_match_url`
  (`"kesz_reszek": ["felidok", "vagas_first_half", ...]`), a vágás végén `"vagas"`;
- félidőnként az elemzés után: `analysis.<félidő>` (`"elemzes_first_half"`, `"elemzes_second_half"`),
  az összefésülés után `"esemenyek"`;
- a végén a teljes eredmény (`"reszleges"` nélkül, a `kesz_reszek` utolsó eleme `"kesz"`).

A kliens tehát a `cut.full_match_url`-t már pár perc után lejátszhatja. A `cut-match` job (`/jobs/run`)
hasonlóan a kész félidő URL-jét már a teljes meccsvideó elkészülte előtt közzéteszi.

## Gyors, képkocka-pontos vágás (smart cut, `fast_cut.py`)

A félidők, a teljes meccs és a klipek vágása **képkocka-pontos**, mégis majdnem olyan gyors, mint a
sima stream copy:
1. ffprobe-bal csak a vágási pontok környékén (néhány mp-es ablakban) olvassuk ki a kulcskockákat;
2. ha a kezdőpont kulcskocka, nincs újrakódolás; különben csak a kezdőponttól a következő kulcskockáig
   tartó rövid részt (általában < 2 mp) kódoljuk újra, a forrással egyező paraméterekkel (kodek, profil,
   szint, pix_fmt, felbontás, időalap, B-kocka késleltetés); a végén ugyanígy az utolsó kulcskockától;
3. a középső (hosszú) rész újrakódolás nélkül, közvetlenül a forrásból másolódik (concat demuxer);
4. a hang a teljes szakaszra a forrásból jön, mintapontosan igazítva (AAC: másolva);
5. ellenőrzés: képkockaszám, A/V kezdet, időbélyegek és dekódolás a csatlakozásoknál. Ha bármi nem
   stimmel, vagy a forrás nem alkalmas (változó képkockasebesség, MPEG-TS, 4:4:4 / 10 bites H.264,
   váltott soros kép, …), **teljes újrakódolásra** vált – hibás fájl nem kerül a kimenetre.

Az újrakódolt részekhez hardveres kódolót használ, ha van (egyszer, induláskor felderítve, egy apró
próbakódolással): NVIDIA `h264_nvenc` / `hevc_nvenc` → Intel `h264_qsv` → AMD `h264_amf` → szoftveres
`libx264 -preset veryfast`. Ha a hardveres kódoló hibázik vagy nem egyező paraméterű képet ad,
automatikusan a szoftveresre vált.

Környezeti változók:
| Változó | Jelentés |
|---|---|
| `FAST_CUT_ENCODER` | `auto` (alap), `cpu` (csak szoftveres), vagy egy kódolónév, pl. `h264_qsv` |
| `FAST_CUT_MODE` | `full`: mindig teljes újrakódolás (ha egy lejátszó gondot okozna a vegyes részekkel) |
| `FFPROBE_PATH` | az ffprobe útvonala (alap: az ffmpeg mellől, `ffprobe.exe`) |

Mért idők (90 mp-es 720p szintetikus forrás, 80,4 mp-es szakasz, 4 magos gép, libx264, `tests/test_fast_cut.py`):
stream copy 0,2–0,3 mp (de pontatlan: a kulcskockánál kezd), smart cut 2–3,5 mp, teljes újrakódolás 17–30 mp.
Egy 45 perces félidőnél a smart cut ideje lényegében a fájl másolásáé (a fej/farok újrakódolása 1–2 mp),
a teljes újrakódolás ezzel szemben a félidő hosszának többszöröse is lehet egy laptopon.

A klip-összeállítások (sok klip → egy fájl) minden klipet smart cut-tal vágnak, majd egyszer fűzik össze;
azonos paraméterű részeknél újrakódolás nélkül (ellenőrzéssel), eltérőknél újrakódolva.

## Háttér-mód (laptopon: a gép közben használható marad)

`background=true` (HTTP) vagy `--hatterben` (parancssor):
- a feladat alacsony prioritású szálon fut (Linux: a szál nice értéke 10; Windows: BELOW_NORMAL szálprioritás),
  az általa indított ffmpeg-folyamatok is alacsony prioritással indulnak (Windows: `BELOW_NORMAL_PRIORITY_CLASS`,
  máshol `nice -n 10`) és `-threads N`-nel;
- a torch / OpenCV / OpenMP szálszáma `N`-re korlátozódik (a szerverben ez a háttérjob idejére az egész
  folyamatra érvényes, a végén visszaáll);
- `N` alapból a processzormagok fele, akkumulátoros üzemben (ha a `psutil` telepítve van) a negyede;
  `threads=N` / `--szalak N` felülírja;
- a parancssori `--hatterben` az egész folyamat prioritását csökkenti (Windows-on a gyermekfolyamatok is örökölik).

A HTTP kérések kiszolgálása normál prioritáson marad, így a kliens a háttérjob alatt is gyorsan válaszol.

## Használat a MAUI kliensből

```csharp
// 1) Indítás
var form = new MultipartFormDataContent {
    { new StringContent(videoPath), "video_path" },
    { new StringContent("45"), "expected_half_min" },
    { new StringContent(calibPoints), "calib_points" },
    { new StringContent("true"), "team1_attacks_right" },
};
var start = await http.PostAsync("http://localhost:8000/jobs/analyze-match", form);
var jobId = (await start.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("job_id").GetString();

// 2) Lekérdezés 1–2 mp-enként (pl. PeriodicTimer), a haladássáv és a fázisszöveg frissítése
using var timer = new PeriodicTimer(TimeSpan.FromSeconds(1.5));
while (await timer.WaitForNextTickAsync(ct)) {
    var job = (await http.GetFromJsonAsync<JsonElement>($"http://localhost:8000/jobs/{jobId}")).GetProperty("job");
    ProgressBar.Progress = job.GetProperty("progress").GetDouble() / 100.0;
    PhaseLabel.Text = job.GetProperty("phase").GetString();
    var status = job.GetProperty("status").GetString();
    if (status is "done" or "error" or "cancelled" or "megszakadt") { /* eredmény / hiba megjelenítése */ break; }
}

// 3) Megszakítás gomb
await http.PostAsync($"http://localhost:8000/jobs/{jobId}/cancel", null);
```
- Az URL-ek (`/clips/...`) a szerverről tölthetők le; a `local_path` / `*_local_path` mezők a helyi fájlok.
- Ha a kliens újraindul, a `GET /jobs` listából visszakereshető a futó / kész job.
- A szerver újraindítása után a félbemaradt jobok `megszakadt` állapotúak (újra kell indítani őket); a kész jobok eredménye megmarad.

## Parancssorból (szerver nélkül)

```
python elemzes.py MECCS.mp4
python elemzes.py MECCS.mp4 --felido-perc 40 --kalibracio "0.12,0.41;0.88,0.40;1.0,0.93;0.0,0.95" --kimenet eredmeny
python elemzes.py MECCS.mp4 --nincs-vagas --gyors 3 --osszeallitasok goalkeepers
python elemzes.py MECCS.mp4 --felidok 15:05,1:00:58,1:16:10,2:03:02
python elemzes.py --help
```

| Kapcsoló | Jelentés |
|---|---|
| `--felido-perc N` | várt félidőhossz percben (alap: 45) |
| `--kalibracio "x,y;…"` | 4 pályapont relatív (0..1) koordinátákkal, mint a kliensben |
| `--kimenet MAPPA` | kimeneti mappa (alap: `clips`) |
| `--nincs-vagas` | ne készüljön szünet nélküli meccsvideó |
| `--gyors N` | felismerés csak minden N. képkockán (a `TACTICAL_DETECT_EVERY_N` környezeti változót állítja) |
| `--felidok K1,V1,K2,V2` | kézi félidőhatárok |
| `--csapat1-balra` | az 1. csapat az 1. félidőben balra támad |
| `--osszeallitasok LISTA` | `goalkeepers,highlights` (alap), üres szöveg = nincs |
| `--letszam` | YOLO létszámszámlálás a félidő-felismeréshez |
| `--nincs-hoterkep`, `--nincs-attekinto` | hőtérkép / áttekintő lapok kikapcsolása |
| `--reszletes` | a motor saját kiírásai is a képernyőre kerülnek (alap: `<kimenet>/elemzes_<idő>.log`) |
| `--hatterben` | háttér-mód: alacsony prioritás, kevés szál (laptopon ajánlott) |
| `--szalak N` | legfeljebb N szál (torch / OpenCV / ffmpeg) |

A képernyőn egysoros haladásjelző fut (`[#####-----]  42.7% 1. félidő elemzése: … | hátra kb. 31:20`),
a végén magyar összefoglaló a félidőkkel, az elkészült fájlok útvonalaival, megjegyzésekkel és figyelmeztetésekkel.
Ctrl+C megszakítja (kilépési kód 130). Hiba esetén a traceback a naplófájlba kerül (kilépési kód 1).

## Korlátok

- A jobok a szerver folyamatán belül futnak (2 párhuzamos szál); az elemzések a közös YOLO modell miatt
  amúgy is egymás után futnak (`model_lock`). Újraindítás után a félbemaradt job nem folytatódik.
- A megszakítás az elemzés közben a következő 25 képkockás jelzésnél történik; a félbehagyott kimeneti
  videó (`raw_<id>.mp4`) és az ideiglenes vágás (`temp_cut_<id>.mp4`) a `clips` mappában maradhat.
- Mindkét félidő ugyanazt a kalibrációt (`calib_points`) használja.
- Az összesített hőtérképen játékosonkénti kép nem készül (a követési azonosítók félidőnként újraindulnak).
- Az elemzés (`cut_and_analyze_sync`) a `ffmpeg.exe` nevet használja (Windows); máshol PATH-on lévő `ffmpeg.exe` kell.
- Háttér-módban az elemzés saját ffmpeg-hívásai (`cut_and_analyze_sync`) Windows-on nem kapnak külön alacsony
  prioritást (Linuxon a szál prioritását öröklik); a parancssori `--hatterben` mindenre érvényes.
- A szüneteltetés nem állítja meg az éppen futó ffmpeg lépést (pl. egy félidő másolását), csak utána áll meg.
- Smart cut: a vegyes (újrakódolt fej + másolt törzs) fájlban a H.264 paraméterkészletek a fájl közben
  változnak (szabványos, az ffmpeg, a böngészők és a Windows lejátszó kezeli); ha egy lejátszó mégis gondot
  okozna, `FAST_CUT_MODE=full`. HEVC forrásnál a B-kocka késleltetést nem mindig lehet egyeztetni – ilyenkor
  teljes újrakódolás (H.264) lesz.

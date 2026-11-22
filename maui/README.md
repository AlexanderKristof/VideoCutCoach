# TacAnaly – asztali program (.NET MAUI)

A program a `CLOUDtacticAnalyzer` mappában van, megnyitás: `CLOUDtacticAnalyzer.slnx` (Visual Studio 2022/2026).
Fordítás és indítás: a felső sávban **Windows Machine** legyen kiválasztva, majd F5.

## Indulás: az elemző motor

A program induláskor maga keresi meg vagy indítja el a Python motort (`Service/EngineHost.cs`):

1. Ha már fut egy motor a `http://127.0.0.1:8000` címen (pl. kézzel indított `python main.py`), azt használja.
2. Telepített programnál: az `engine\tacanaly_engine.exe` (a PyInstaller build, lásd `packaging/`).
3. Fejlesztői gépen: a Python forrás. A program a saját mappájából felfelé haladva megkeresi a repó
   gyökerét (ahol a `main.py` és az `engine_launcher.py` van), és elindítja:
   `python engine_launcher.py --port 0 --parent-pid <program>`. Python: a repó `.venv` / `venv`
   mappája, ha van, különben a `python`, végül a `py -3` parancs.

A 2–3. esetben a motor szabad portot választ, és a programmal együtt leáll (bezáráskor `/shutdown`,
összeomláskor a `--parent-pid` miatt magától).

Ha a motor nem indul, az indító képernyő kiírja az okát, és ott megadható:
- a motor mappája (a repó gyökere, pl. `E:\ASZTALRA\Edzesterv\TacAnalyCLOUD`),
- a Python program (pl. `C:\Python312\python.exe`), ha nem a PATH-on lévőt kell használni,
- vagy egy már futó motor címe (pl. telefonról: `http://192.168.1.20:8000` – ehhez a motort a gépen
  `--lan` kapcsolóval kell indítani).

Ugyanezek környezeti változóval is: `TACANALY_ENGINE_DIR`, `TACANALY_PYTHON_EXE`, `TACANALY_ENGINE_URL`.

## Fülek

| Fül | Mit tud | Motor végpontok |
|---|---|---|
| **Meccs elemzés** | teljes meccs egy gombnyomásra a háttérben (félidők felismerése, vágás, elemzés, hőtérkép, kapus- és összefoglaló videók); haladás, szünet / folytatás / megszakítás; eseménylista szűrőkkel (esemény, csapat, félidő); kattintásra lejátszás az eseménynél; a kijelölt jelenetek kivágása egy videóba; hőtérképek; megnyitás a taktikai táblán / rajzolás a videón | `/jobs/analyze-match`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/pause|resume|cancel`, `/jobs/run` (`export-clips`), `/clips/...` |
| **Taktikai tábla** | felállás, nyilak, körök, felülnézeti visszajátszás, rajzolás a videóra, PNG/WebP és videó mentés (a motor `/tabla` oldala, WebView-ban) | `/tabla`, `/boards` |
| **Kézi vágás** | a régi folyamat: képkocka, kézi színjelölés, 4 sarokpontos kalibráció, egy szakasz vágása és elemzése | `/get-frame-time`, `/analyze-click-ai`, `/cut-raw` |
| **Kapus Modul** | szakasz betöltése, AI kapuskeresés, rajzok beégetése | `/detect-goalkeepers`, `/burn-drawings` |

Az automatikusan felismert kapus események (védés, beadás leszedése, kirúgás, hosszú labda, rövid passz,
kapott gól, kifutás) a **Meccs elemzés** fül eseménylistájában vannak, a szűrővel külön is listázhatók,
és egy gombbal egy videóba vághatók.

A Windows WebView a WebView2 (Edge) futtatókörnyezetet használja – Windows 11-en megvan.

## Ellenőrzés Visual Studio nélkül

`ellenorzes/ellenoriz.sh` – a kódot (a XAML-lel együtt, a `Platforms` mappa nélkül) MAUI workload nélkül
lefordítja sima `net10.0` könyvtárként. Gyors szintaxis- és típusellenőrzésre jó (pl. Linuxon, CI-ban);
a valódi Windows-fordítást nem helyettesíti.

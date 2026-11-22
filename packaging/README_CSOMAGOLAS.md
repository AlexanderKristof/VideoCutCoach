# Csomagolás: asztali alkalmazás (MAUI + elemző motor egy telepítőben)

Cél: a felhasználó egyetlen telepítőt futtat; a gépén nem kell Python. A telepítő tartalma:

```
{app}\                         – a MAUI program (önálló, unpackaged vagy MSIX)
{app}\engine\                  – az elemző motor (PyInstaller one-folder build)
    tacanaly_engine.exe
    _internal\                 – Python, torch, OpenCV, static\ (taktikai tábla), model_config.example.json
    ffmpeg\ffmpeg.exe          – LGPL ffmpeg build (lásd Licencek)
    models\                    – (opcionális) előre mellékelt .pt súlyok
```

Az írható adatok felhasználónként a `%LOCALAPPDATA%\TacAnaly` mappába kerülnek
(`clips\`, `jobs\`, `boards\`, `models\`, `logs\engine.log`, `tactical_app.db`, `secret.key`,
`model_config.json`, `engine.json`). Felülírható: `--data-dir` vagy `TACTICAL_DATA_DIR`.
Fejlesztői módban (`python main.py`, `python elemzes.py`) minden a régi helyén marad (`clips\` stb. a munkakönyvtárban).

## 1. A motor buildje (Windows)

Előfeltétel: Python 3.11 (py launcher), internet a csomagokhoz.

```powershell
# CPU-s build (bármely gépen fut, kisebb: ~1 GB)
powershell -ExecutionPolicy Bypass -File packaging\build_engine.ps1 -Torch cpu -FfmpegExe C:\tools\ffmpeg\bin\ffmpeg.exe
# NVIDIA GPU-s build (CUDA 12.4; több GB)
powershell -ExecutionPolicy Bypass -File packaging\build_engine.ps1 -Torch cu124 -FfmpegExe C:\tools\ffmpeg\bin\ffmpeg.exe -Clean
```

A szkript venv-et hoz létre (`packaging\.venv-<torch>`), telepíti a torch-ot a választott
változatban, a `requirements.txt`-t + `pyinstaller psutil lap opencv-python-headless`, lefuttatja a
`packaging\tacanaly_engine.spec`-et, bemásolja az ffmpeg-et, végül füsttesztet futtat (READY sor,
`/health`, `/shutdown`). Kimenet: `packaging\dist\tacanaly_engine\`.

Megjegyzések:
- **One-folder** build: a torch DLL-jei miatt a one-file mód minden indításkor több száz MB-ot
  csomagolna ki a %TEMP%-be (lassú indulás).
- A CUDA-s build csak NVIDIA kártyán gyorsít, de CPU-n is fut. Két telepítő (CPU / GPU) helyett
  adható egy GPU-s is – csak nagyobb.
- A modell súlyai nincsenek a buildben: első elemzéskor (vagy a `POST /models/download` hívásra)
  töltődnek le a `%LOCALAPPDATA%\TacAnaly\models` mappába. Offline telepítéshez tedd a `.pt`
  fájlokat az `{app}\engine\models\` mappába (a motor ott is keresi). Focis modellhez a
  `model_config.json` a `%LOCALAPPDATA%\TacAnaly` mappába (vagy a buildbe, a repó gyökeréből) kerül.
- Próbabuild torch / ultralytics nélkül (pl. Linuxon, a csomag szerkezetének ellenőrzésére):
  `TACANALY_SPEC_NO_ML=1 pyinstaller packaging/tacanaly_engine.spec`.

### Árusítható kiadási build (`-Release`)

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_engine.ps1 -Release -FfmpegExe C:\tools\ffmpeg-lgpl\bin\ffmpeg.exe
```

- **Nincs benne** torch, ultralytics (AGPL-3.0) és scikit-learn – a build ellenőrzi, hogy a kiadási
  környezetben ezek nincsenek telepítve, különben leáll.
- A felismerés **ONNX Runtime + DirectML** (bármely Windowsos videókártya, integrált is; CPU-tartalékkal),
  a saját követővel (`detection/`). A gyári `model_config.json` az ONNX hátteret állítja be
  (`model_config.onnx.example.json` alapján) – ha a repó gyökerében van saját `model_config.json`, az kerül be.
- A **focis ONNX modellt** (`models/focis_detektor.onnx`, a saját tanításból – `docs/MODELL_TANITAS.md`)
  a telepítő tegye az adatmappa `models\` mappájába (vagy az `engine\models\` mappába).
- **ffmpeg:** LGPL build kell. Ebben nincs libx264 – a program a videókártya kódolóját (NVENC / Quick Sync /
  AMF) vagy a libopenh264-et használja (`fast_cut.py`, `video_writer.py`).
- A kiadás előtti ellenőrzőlista: `docs/LICENC_ES_ARUSITAS.md`.

## 2. A motor indítása és leállítása a MAUI programból

Parancssor: `tacanaly_engine.exe --port 0 --parent-pid <MAUI PID>`

| kapcsoló | jelentés |
|---|---|
| `--port 0` | szabad portot választ (alap); fix porthoz pl. `--port 8000` |
| `--host 127.0.0.1` | alap: csak a helyi gép érheti el |
| `--lan` | 0.0.0.0 – a telefon is elérheti a helyi hálózaton (lásd Biztonság) |
| `--data-dir PATH` | adatmappa (alap: `%LOCALAPPDATA%\TacAnaly`) |
| `--parent-pid PID` | ha a MAUI folyamat megszűnik (összeomlás is), a motor magától leáll |
| `--control-token T` | a `/shutdown` csak `X-TacAnaly-Token: T` fejléccel (vagy `TACTICAL_CONTROL_TOKEN` env) |
| `--preload none/device/model` | háttérben: semmi / torch + GPU felismerés (alap) / a modell betöltése is |
| `--console` | a napló a stderr-re is |

Protokoll:
1. Indítsd átirányított stdout-tal, ablak nélkül.
2. Várd az **egyetlen** gépi sort: `TACANALY_READY port=<p> pid=<pid>` (hiba esetén
   `TACANALY_ERROR <üzenet>` és nem nulla kilépési kód: 3 = a motor nem tölthető be, 4 = foglalt port).
   Ugyanez a `%LOCALAPPDATA%\TacAnaly\engine.json` fájlba is kerül. Utána a stdout-ra nem ír
   semmit (minden más a `logs\engine.log` forgó naplóba megy), tehát nem kell tovább olvasni.
3. `GET /health` – mindig gyors, nem tölt be modellt:
   `{"status":"ok","version","model_loaded","model_loading","model_error","model_weights","device","ffmpeg","data_dir","mode","jobs_active","shutting_down","pid"}`.
   A `device` `null`, amíg a torch be nem töltődött (a `--preload device` pár mp alatt kitölti).
4. Modellek: `GET /models/status` (megvannak-e a súlyok), `POST /models/download` (`{"name":"yolov8x.pt"}`
   vagy üresen: az aktuális eszköz súlya) és `POST /models/load` (bemelegítés) – mindkettő
   `{"status":"success","job_id"}`-t ad, a haladás a szokásos `GET /jobs/{id}` végponton
   (`job.progress`, `job.phase`), mint a többi háttérfeladatnál (docs/HATTERFELADATOK.md).
5. Kilépéskor `POST /shutdown` (csak 127.0.0.1-ről): a futó jobokat megszakítja, legfeljebb
   `grace_sec` (alap 10) mp-et vár, a maradékot „megszakadt” állapotba teszi, majd leáll.
   Ha 15–20 mp alatt nem lép ki: `Kill(entireProcessTree: true)`.

`GET /version`: `{"name","version","api_version","frozen","python"}` – a MAUI ellenőrizheti, hogy
kompatibilis motort indított-e (`api_version`).

### C# minta (MAUI, Windows)

```csharp
using System.Diagnostics;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json.Serialization;

public sealed record EngineHealth(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("version")] string Version,
    [property: JsonPropertyName("model_loaded")] bool ModelLoaded,
    [property: JsonPropertyName("device")] string? Device,
    [property: JsonPropertyName("ffmpeg")] string? Ffmpeg,
    [property: JsonPropertyName("data_dir")] string DataDir);

public sealed class EngineHost : IAsyncDisposable
{
    private readonly string _token = Guid.NewGuid().ToString("N");
    private HttpClient? _http;
    private Process? _proc;

    public Uri? BaseUri { get; private set; }
    public HttpClient Http => _http ?? throw new InvalidOperationException("A motor nem fut.");

    public async Task StartAsync(CancellationToken ct = default)
    {
        var exe = Path.Combine(AppContext.BaseDirectory, "engine", "tacanaly_engine.exe");
        var psi = new ProcessStartInfo(exe)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            WorkingDirectory = Path.GetDirectoryName(exe)!,
        };
        psi.ArgumentList.Add("--port"); psi.ArgumentList.Add("0");
        psi.ArgumentList.Add("--parent-pid"); psi.ArgumentList.Add(Environment.ProcessId.ToString());
        psi.Environment["TACTICAL_CONTROL_TOKEN"] = _token;

        var ready = new TaskCompletionSource<int>(TaskCreationOptions.RunContinuationsAsynchronously);
        _proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _proc.OutputDataReceived += (_, e) =>
        {
            var line = e.Data;
            if (line is null) return;
            if (line.StartsWith("TACANALY_READY "))
            {
                foreach (var part in line.Split(' '))
                    if (part.StartsWith("port=") && int.TryParse(part[5..], out var p))
                        ready.TrySetResult(p);
            }
            else if (line.StartsWith("TACANALY_ERROR "))
                ready.TrySetException(new InvalidOperationException(line[15..]));
        };
        _proc.ErrorDataReceived += (_, _) => { };   // csak ürítjük (a napló: logs\engine.log)
        _proc.Exited += (_, _) => ready.TrySetException(
            new InvalidOperationException($"A motor kilépett (kód: {_proc.ExitCode})."));
        _proc.Start();
        _proc.BeginOutputReadLine();
        _proc.BeginErrorReadLine();

        using var startTimeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        startTimeout.CancelAfter(TimeSpan.FromSeconds(90));   // első indulás (víruskereső!) lassú lehet
        var port = await ready.Task.WaitAsync(startTimeout.Token);

        BaseUri = new Uri($"http://127.0.0.1:{port}/");
        _http = new HttpClient { BaseAddress = BaseUri, Timeout = TimeSpan.FromSeconds(30) };

        for (var i = 0; i < 50; i++)                        // /health: legfeljebb ~10 mp
        {
            try
            {
                var h = await _http.GetFromJsonAsync<EngineHealth>("health", ct);
                if (h?.Status == "ok") return;
            }
            catch (HttpRequestException) { }
            await Task.Delay(200, ct);
        }
        throw new TimeoutException("A motor nem válaszol a /health kérésre.");
    }

    public async ValueTask DisposeAsync()
    {
        if (_proc is null) return;
        if (!_proc.HasExited && _http is not null)
        {
            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Post, "shutdown")
                {
                    Content = JsonContent.Create(new { grace_sec = 5 })
                };
                req.Headers.Add("X-TacAnaly-Token", _token);
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3));
                await _http.SendAsync(req, cts.Token);
            }
            catch { /* már nem válaszol – jön a kill */ }

            using var wait = new CancellationTokenSource(TimeSpan.FromSeconds(15));
            try { await _proc.WaitForExitAsync(wait.Token); }
            catch (OperationCanceledException) { _proc.Kill(entireProcessTree: true); }
        }
        _proc.Dispose();
        _http?.Dispose();
    }
}
```

Bekötés: az `EngineHost`-ot singletonként regisztráld (`builder.Services.AddSingleton<EngineHost>()`),
indítsd az első ablak megjelenésekor, és a Windows életciklus-eseményben állítsd le:

```csharp
builder.ConfigureLifecycleEvents(events =>
{
#if WINDOWS
    events.AddWindows(w => w.OnClosed((window, args) =>
    {
        var host = IPlatformApplication.Current!.Services.GetRequiredService<EngineHost>();
        // Task.Run: a UI szálon blokkoló Wait() ne akadjon össze az await-ekkel (deadlock)
        Task.Run(() => host.DisposeAsync().AsTask()).Wait(TimeSpan.FromSeconds(20));
    }));
#endif
});
```

A meglévő kliens `BaseAddress`-ét (eddig `http://localhost:8000`) cseréld az `EngineHost.BaseUri`-ra.
Ha a MAUI folyamat összeomlik, a `--parent-pid` miatt a motor magától leáll. (Még szigorúbb: a
motort egy `KILL_ON_JOB_CLOSE` Windows Job Objectbe tenni – ekkor az OS zárja be.)

### A taktikai tábla a MAUI-ban

A tábla (`static/taktikai_tabla.html`) a motorról töltődik be, és relatív címekkel hívja
(`/boards`, `/clips/...`) – ezért **ne** helyi fájlként (HybridWebView `Resources/Raw`) ágyazd be,
hanem a motor URL-jével:

```xml
<WebView x:Name="Board" />
```
```csharp
Board.Source = new Uri(engine.BaseUri!, "tabla").ToString();
// egy elemzett meccs visszajátszása:
Board.Source = new Uri(engine.BaseUri!, $"tabla?events=/clips/events_{clipId}.json").ToString();
```

Windows-on a WebView a WebView2 (Edge) futtatókörnyezetet használja: Windows 11-en megvan,
Windows 10-re a telepítő tegye fel az Evergreen bootstrapperrel.

## 3. Telepítő

**Inno Setup** (unpackaged MAUI: `dotnet publish -f net8.0-windows10.0.19041.0 -c Release
-p:WindowsPackageType=None -p:SelfContained=true -p:WindowsAppSDKSelfContained=true`):

```ini
[Files]
Source: "maui\publish\*";                       DestDir: "{app}";        Flags: recursesubdirs ignoreversion
Source: "packaging\dist\tacanaly_engine\*";     DestDir: "{app}\engine"; Flags: recursesubdirs ignoreversion
Source: "third_party\ffmpeg\ffmpeg.exe";        DestDir: "{app}\engine\ffmpeg"
Source: "third_party\ffmpeg\LICENSE.txt";       DestDir: "{app}\engine\ffmpeg"
Source: "third_party\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; Check: NeedsWebView2
```
Az adatmappát (`%LOCALAPPDATA%\TacAnaly`) az eltávolító alapból NE törölje (a felhasználó elemzései).

**MSIX**: a motor mappáját a csomag tartalmaként add hozzá (pl. `engine\**`, `CopyToOutputDirectory`).
A gyermekfolyamat indításához `runFullTrust` képesség kell. MSIX alatt a `%LOCALAPPDATA%` a csomag
saját (virtualizált) mappájába íródik – ez a motornak jó, de a felhasználó a fájlokat a
`%LOCALAPPDATA%\Packages\<csomag>\LocalCache\Local\TacAnaly` alatt találja (vagy adj meg
`--data-dir`-t pl. a Dokumentumok alá).

Mindkét esetben: a csomag mérete miatt (CPU ~1 GB, CUDA több GB) érdemes tömöríteni (Inno: `lzma2/ultra64`);
aláírás nélkül a SmartScreen és egyes víruskeresők gyanúsnak jelölhetik a PyInstalleres exe-t –
kód-aláíró tanúsítvánnyal írd alá a `tacanaly_engine.exe`-t és a telepítőt is.

## 4. Biztonság

- Alapból csak a `127.0.0.1`-en figyel: a hálózatról nem érhető el, és a Windows tűzfal sem kérdez.
- `--lan` (0.0.0.0): a telefon a helyi hálózaton megnyithatja a táblát (`http://<pc-ip>:<port>/tabla`),
  de ekkor **a hálózat bármely gépe** hívhatja az API-t (videóút megadásával fájlokat olvastathat,
  elemzést indíthat, a `/clips` tartalmát letöltheti). Csak megbízható (otthoni) hálózaton,
  fix porttal és tűzfalszabállyal használd; nyilvános Wi-Fi-n soha. A `/shutdown` LAN módban is csak
  helyi gépről hívható.
- A JWT aláírókulcs telepítésenként véletlen (`secret.key`, első induláskor generálva, POSIX-on 0600;
  Windows-on a `%LOCALAPPDATA%` felhasználónkénti). Felülírható: `TACTICAL_SECRET_KEY`.
- A `--control-token` / `TACTICAL_CONTROL_TOKEN` megakadályozza, hogy egy másik helyi program leállítsa a motort.

## 5. Licencek (tájékoztató, nem jogi tanács)

- **Ultralytics (YOLOv8 / YOLO11, kód és a hivatalos súlyok): AGPL-3.0.** Ha a programot terjeszted,
  és az ultralytics benne van, az AGPL szerint a teljes (vele egy művet alkotó) program forráskódját
  AGPL alatt elérhetővé kell tenned. Az, hogy a MAUI felület külön folyamatként, HTTP-n beszél a
  motorral, nem biztos menekülőút – zárt forrású terjesztéshez a biztos út az **Ultralytics Enterprise
  License**, vagy a felismerő cseréje megengedő licencűre, pl. **RT-DETR / RF-DETR / D-FINE
  (Apache-2.0)**, **YOLOX (Apache-2.0)**; a ByteTrack eredeti megvalósítása MIT. Az Ultralytics
  álláspontja szerint az ultralytics-szel tanított saját modellekre is az AGPL vonatkozik.
- **FFmpeg**: alapból LGPL-2.1+, de a `--enable-gpl` buildek (libx264, libx265 – pl. a gyan.dev
  „essentials/full” csomagjai) GPL-esek. **Fontos:** a motor jelenleg `-c:v libx264`-gyel kódol
  (main.py, clip_export.py), ez csak GPL buildben van. Lehetőségek: (a) GPL ffmpeg.exe külön
  programként mellékelve – ekkor az ffmpeg-re a GPL kötelezettségei vonatkoznak (licencszöveg +
  a pontos forráskód elérhetősége), a saját program jellemzően „aggregátum” marad; (b) LGPL build
  (pl. BtbN `*-lgpl` csomagok) és a kódoló cseréje `libopenh264`-re vagy hardveres kódolóra
  (`h264_nvenc`, `h264_qsv`, `h264_amf`) – ehhez a kódolót konfigurálhatóvá kell tenni (még nincs meg).
  Mindkét esetben: mellékeld a licencet és a forrás elérhetőségét, az exe legyen cserélhető.
- **OpenH264** (`openh264-*.dll`, BSD): a Cisco által terjesztett binárisra vonatkozik a Cisco
  szabadalmi licence – ha a DLL-t a saját telepítődbe teszed, ez a fedezet nem érvényes.
- **PyTorch: BSD-3-Clause**; a CUDA-s kerekekben lévő NVIDIA könyvtárak (cuDNN, cuBLAS ...) az NVIDIA
  licence szerint az alkalmazással együtt terjeszthetők.
- **OpenCV (4.5+): Apache-2.0**; FastAPI, Starlette, uvicorn, SQLAlchemy, python-jose: MIT/BSD;
  passlib: BSD; bcrypt: Apache-2.0; NumPy: BSD; Python: PSF. A harmadik féltől származó
  licencszövegeket a telepítő mellékelje (pl. `{app}\licenses\`).

using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace TacticalAnalyzer.Services;

/// <summary>
/// Az elemző motor (Python) elérése, és ha kell, elindítása.
///
/// Sorrend induláskor (StartAsync):
///   1. Ha már fut egy motor a megszokott címen (pl. kézzel indított "python main.py" a 8000-es porton),
///      azt használjuk – így a fejlesztés közben semmi nem változik.
///   2. Telepített program: az app mellett lévő engine\tacanaly_engine.exe (PyInstaller build).
///   3. Fejlesztői gép: a Python forráskód (engine_launcher.py) – a mappát a program felfelé haladva
///      maga keresi meg (a MAUI projekt a repó "maui" mappájában van), vagy a TACANALY_ENGINE_DIR
///      környezeti változó / a beállításokban megadott "engine_dir" mondja meg.
/// A 2–3. esetben a motor szabad portot választ (--port 0), és a programmal együtt áll le
/// (--parent-pid, kilépéskor /shutdown). Részletek: packaging/README_CSOMAGOLAS.md.
/// </summary>
public sealed class EngineHost
{
    public const string ReadyPrefix = "TACANALY_READY";
    public const string ErrorPrefix = "TACANALY_ERROR";

#if ANDROID
    // Android emulátorból a fejlesztő gép: 10.0.2.2. Valódi telefonon a beállításokban megadható
    // ("engine_url", pl. http://192.168.1.20:8000 – a motort ekkor --lan kapcsolóval kell indítani).
    public const string DefaultUrl = "http://10.0.2.2:8000";
#else
    public const string DefaultUrl = "http://127.0.0.1:8000";
#endif

    private readonly string _token = Guid.NewGuid().ToString("N");
    private readonly HttpClient _probe = new() { Timeout = TimeSpan.FromSeconds(2) };
    private Process? _proc;
    private Task<string>? _startTask;

    /// <summary>A motor címe, perjel nélkül (pl. "http://127.0.0.1:53817").</summary>
    public string BaseUrl { get; private set; } = LoadConfiguredUrl();

    /// <summary>Rövid magyar leírás: honnan jön a motor (kijelzéshez).</summary>
    public string Description { get; private set; } = "";

    /// <summary>Igaz, ha a motort mi indítottuk (kilépéskor mi is állítjuk le).</summary>
    public bool OwnsProcess => _proc is { HasExited: false };

    public bool IsReady { get; private set; }

    /// <summary>Teljes URL egy motor-relatív útvonalhoz ("/clips/x.mp4" → "http://.../clips/x.mp4").</summary>
    public string Url(string? relative)
    {
        if (string.IsNullOrEmpty(relative)) return BaseUrl;
        if (relative.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            relative.StartsWith("https://", StringComparison.OrdinalIgnoreCase)) return relative;
        return BaseUrl + (relative.StartsWith('/') ? relative : "/" + relative);
    }

    /// <summary>A beállítások (engine_url stb.) újraolvasása egy sikertelen indítás után.</summary>
    public void Reconfigure()
    {
        lock (this)
        {
            if (_startTask is { IsCompletedSuccessfully: true }) return;
            _startTask = null;
            BaseUrl = LoadConfiguredUrl();
        }
    }

    /// <summary>A motor elérése/indítása. Többszöri hívásnál ugyanazt a feladatot adja vissza.</summary>
    public Task<string> StartAsync(IProgress<string>? status = null, CancellationToken ct = default)
    {
        lock (this)
        {
            if (_startTask is null || _startTask.IsFaulted || _startTask.IsCanceled)
                _startTask = StartCoreAsync(status, ct);
            return _startTask;
        }
    }

    private async Task<string> StartCoreAsync(IProgress<string>? status, CancellationToken ct)
    {
        status?.Report("Motor keresése…");
        if (await PingAsync(BaseUrl, ct))
        {
            IsReady = true;
            Description = $"már futó motor ({BaseUrl})";
            return Description;
        }

        if (!CanStartProcess)
            throw new InvalidOperationException(
                $"Az elemző motor nem érhető el: {BaseUrl}. Indítsd el a gépen (python main.py), " +
                "és ellenőrizd a címet a beállításokban.");

        var errors = new List<string>();
        foreach (var candidate in FindLaunchCandidates())
        {
            ct.ThrowIfCancellationRequested();
            status?.Report($"Motor indítása: {candidate.Label}…");
            try
            {
                var port = await LaunchAsync(candidate, ct);
                BaseUrl = $"http://127.0.0.1:{port}";
                status?.Report("Kapcsolódás a motorhoz…");
                for (var i = 0; i < 50 && !await PingAsync(BaseUrl, ct); i++)
                    await Task.Delay(200, ct);
                IsReady = true;
                Description = $"{candidate.Label} (port {port})";
                return Description;
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex)
            {
                errors.Add($"{candidate.Label}: {ex.Message}");
                KillProcess();
            }
        }

        var detail = errors.Count == 0
            ? "Nem található sem telepített motor (engine\\tacanaly_engine.exe), sem a Python forráskód (engine_launcher.py)."
            : string.Join(Environment.NewLine, errors);
        throw new InvalidOperationException("Az elemző motor nem indult el." + Environment.NewLine + detail);
    }

    private static bool CanStartProcess =>
        OperatingSystem.IsWindows() || OperatingSystem.IsMacCatalyst() || OperatingSystem.IsLinux() || OperatingSystem.IsMacOS();

    private async Task<bool> PingAsync(string baseUrl, CancellationToken ct)
    {
        try
        {
            using var resp = await _probe.GetAsync(baseUrl + "/health", ct);
            if (!resp.IsSuccessStatusCode) return false;
            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct));
            return doc.RootElement.TryGetProperty("status", out var s) && s.GetString() == "ok";
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
        catch { return false; }
    }

    // ------------------------------------------------------------------ indítási jelöltek

    public sealed record LaunchCandidate(string Label, string FileName, IReadOnlyList<string> Args, string WorkingDirectory);

    /// <summary>A lehetséges motorindítások sorrendben (telepített exe, majd Python forrás).</summary>
    public static IEnumerable<LaunchCandidate> FindLaunchCandidates()
    {
        var exeName = OperatingSystem.IsWindows() ? "tacanaly_engine.exe" : "tacanaly_engine";
        var exe = Path.Combine(AppContext.BaseDirectory, "engine", exeName);
        if (File.Exists(exe))
            yield return new LaunchCandidate("telepített motor", exe, Array.Empty<string>(), Path.GetDirectoryName(exe)!);

        var src = FindEngineSourceDir();
        if (src is null) yield break;
        var launcher = Path.Combine(src, "engine_launcher.py");
        foreach (var py in PythonCandidates(src))
            yield return new LaunchCandidate($"Python forrás ({py.Label})", py.Exe, py.PrefixArgs.Append(launcher).ToList(), src);
    }

    /// <summary>A Python forrás mappája (ahol az engine_launcher.py van), vagy null.</summary>
    public static string? FindEngineSourceDir()
    {
        foreach (var configured in new[] { Environment.GetEnvironmentVariable("TACANALY_ENGINE_DIR"), Settings.Get("engine_dir") })
        {
            if (!string.IsNullOrWhiteSpace(configured) && File.Exists(Path.Combine(configured, "engine_launcher.py")))
                return Path.GetFullPath(configured);
        }
        // A futó program mappájából felfelé (…\TacAnalyCLOUD\maui\CLOUDtacticAnalyzer\bin\Debug\…)
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; dir is not null && i < 10; i++, dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "engine_launcher.py")) &&
                File.Exists(Path.Combine(dir.FullName, "main.py")))
                return dir.FullName;
        }
        return null;
    }

    private sealed record PythonExe(string Label, string Exe, IReadOnlyList<string> PrefixArgs);

    private static IEnumerable<PythonExe> PythonCandidates(string srcDir)
    {
        var configured = Settings.Get("python_exe");
        if (!string.IsNullOrWhiteSpace(configured))
            yield return new PythonExe(configured, configured, Array.Empty<string>());
        foreach (var venv in new[] { ".venv", "venv", "env" })
        {
            foreach (var rel in new[] { Path.Combine("Scripts", "python.exe"), Path.Combine("bin", "python") })
            {
                var p = Path.Combine(srcDir, venv, rel);
                if (File.Exists(p)) yield return new PythonExe($"{venv}", p, Array.Empty<string>());
            }
        }
        if (OperatingSystem.IsWindows())
        {
            yield return new PythonExe("python", "python", Array.Empty<string>());
            yield return new PythonExe("py -3", "py", new[] { "-3" });
        }
        else
        {
            yield return new PythonExe("python3", "python3", Array.Empty<string>());
        }
    }

    // ------------------------------------------------------------------ folyamat

    private async Task<int> LaunchAsync(LaunchCandidate c, CancellationToken ct)
    {
        var psi = new ProcessStartInfo(c.FileName)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            WorkingDirectory = c.WorkingDirectory,
        };
        foreach (var a in c.Args) psi.ArgumentList.Add(a);
        psi.ArgumentList.Add("--port"); psi.ArgumentList.Add("0");
        psi.ArgumentList.Add("--parent-pid"); psi.ArgumentList.Add(Environment.ProcessId.ToString());
        psi.Environment["TACTICAL_CONTROL_TOKEN"] = _token;
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        psi.Environment["PYTHONUTF8"] = "1";

        var ready = new TaskCompletionSource<int>(TaskCreationOptions.RunContinuationsAsynchronously);
        var stderrTail = new StringBuilder();
        var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        proc.OutputDataReceived += (_, e) =>
        {
            if (e.Data is null) return;
            if (TryParseReady(e.Data, out var port)) ready.TrySetResult(port);
            else if (e.Data.StartsWith(ErrorPrefix, StringComparison.Ordinal))
                ready.TrySetException(new InvalidOperationException(e.Data[ErrorPrefix.Length..].Trim()));
        };
        proc.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is null) return;
            lock (stderrTail)
            {
                stderrTail.AppendLine(e.Data);
                if (stderrTail.Length > 4000) stderrTail.Remove(0, stderrTail.Length - 4000);
            }
        };
        proc.Exited += (_, _) =>
        {
            string tail;
            lock (stderrTail) tail = LastLines(stderrTail.ToString(), 3);
            ready.TrySetException(new InvalidOperationException(
                $"kilépett (kód: {SafeExitCode(proc)})" + (tail.Length > 0 ? $": {tail}" : "")));
        };

        proc.Start();   // ha a fájl nincs meg (pl. nincs "python" a PATH-on): Win32Exception → következő jelölt
        _proc = proc;
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(120));   // első indulás (víruskereső, Python importok) lassú lehet
        try
        {
            return await ready.Task.WaitAsync(timeout.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            throw new TimeoutException("2 perc alatt sem jelezte, hogy elindult.");
        }
    }

    /// <summary>"TACANALY_READY port=53817 pid=1234" → 53817.</summary>
    public static bool TryParseReady(string line, out int port)
    {
        port = 0;
        if (!line.StartsWith(ReadyPrefix + " ", StringComparison.Ordinal)) return false;
        foreach (var part in line.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            if (part.StartsWith("port=", StringComparison.Ordinal) && int.TryParse(part[5..], out port) && port > 0)
                return true;
        return false;
    }

    private static int SafeExitCode(Process p)
    {
        try { return p.ExitCode; } catch { return -1; }
    }

    private static string LastLines(string text, int n)
    {
        var lines = text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return string.Join(" | ", lines.Skip(Math.Max(0, lines.Length - n)));
    }

    /// <summary>Kilépéskor: ha mi indítottuk, kecses leállítás (/shutdown), végső esetben kill.</summary>
    public async Task StopAsync()
    {
        var proc = _proc;
        if (proc is null) return;
        try
        {
            if (!proc.HasExited)
            {
                try
                {
                    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
                    using var req = new HttpRequestMessage(HttpMethod.Post, BaseUrl + "/shutdown")
                    {
                        Content = new StringContent("{\"grace_sec\": 5}", Encoding.UTF8, "application/json")
                    };
                    req.Headers.Add("X-TacAnaly-Token", _token);
                    await http.SendAsync(req);
                }
                catch { /* már nem válaszol – jön a kill */ }

                using var wait = new CancellationTokenSource(TimeSpan.FromSeconds(15));
                try { await proc.WaitForExitAsync(wait.Token); }
                catch (OperationCanceledException) { KillProcess(); }
            }
        }
        finally
        {
            proc.Dispose();
            _proc = null;
            IsReady = false;
        }
    }

    private void KillProcess()
    {
        try { if (_proc is { HasExited: false }) _proc.Kill(entireProcessTree: true); } catch { }
    }

    private static string LoadConfiguredUrl()
    {
        var url = Settings.Get("engine_url");
        return string.IsNullOrWhiteSpace(url) ? DefaultUrl : url.TrimEnd('/');
    }
}

/// <summary>Az egész alkalmazásban közös motor-példány.</summary>
public static class Engine
{
    public static EngineHost Host { get; } = new();

    /// <summary>A motor címe (perjel nélkül). Induláskor a motor indítása után véglegesedik.</summary>
    public static string BaseUrl => Host.BaseUrl;

    public static string Url(string? relative) => Host.Url(relative);
}

/// <summary>
/// Egyszerű beállítás-tár. MAUI alatt a Preferences-t használja; ha az nem érhető el
/// (pl. egységtesztben), a környezeti változót (TACANALY_&lt;KULCS&gt;).
/// </summary>
public static class Settings
{
    public static string? Get(string key)
    {
        try
        {
            var v = Microsoft.Maui.Storage.Preferences.Default.Get(key, "");
            if (!string.IsNullOrEmpty(v)) return v;
        }
        catch { }
        return Environment.GetEnvironmentVariable("TACANALY_" + key.ToUpperInvariant());
    }

    public static void Set(string key, string? value)
    {
        try
        {
            if (string.IsNullOrEmpty(value)) Microsoft.Maui.Storage.Preferences.Default.Remove(key);
            else Microsoft.Maui.Storage.Preferences.Default.Set(key, value);
        }
        catch { }
    }
}

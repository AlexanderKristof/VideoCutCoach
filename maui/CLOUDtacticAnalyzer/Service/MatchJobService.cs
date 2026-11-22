using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TacticalAnalyzer.Services;

// ============================================================================
// A teljes meccs elemzése háttérfeladatként (POST /jobs/analyze-match), a haladás lekérdezése,
// szüneteltetés / folytatás / megszakítás, az eredmény (események, hőtérképek, klipek) beolvasása,
// és a kiválasztott események kivágása (export-clips). API leírás: docs/HATTERFELADATOK.md
// ============================================================================

public sealed class JobInfo
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("kind")] public string Kind { get; set; } = "";
    [JsonPropertyName("status")] public string Status { get; set; } = "";
    [JsonPropertyName("progress")] public double Progress { get; set; }
    [JsonPropertyName("phase")] public string? Phase { get; set; }
    [JsonPropertyName("paused")] public bool Paused { get; set; }
    [JsonPropertyName("cancel_requested")] public bool CancelRequested { get; set; }
    [JsonPropertyName("background")] public bool Background { get; set; }
    [JsonPropertyName("created_at")] public string? CreatedAt { get; set; }
    [JsonPropertyName("finished_at")] public string? FinishedAt { get; set; }
    [JsonPropertyName("elapsed_sec")] public double? ElapsedSec { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
    [JsonPropertyName("params")] public JsonElement? Params { get; set; }
    [JsonPropertyName("result")] public JsonElement? Result { get; set; }

    public bool IsFinished => Status is "done" or "error" or "cancelled" or "megszakadt";
    public bool IsRunning => Status is "running" or "queued";

    public string? VideoPath =>
        Params is { ValueKind: JsonValueKind.Object } p && p.TryGetProperty("video_path", out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString() : null;

    public string StatusLabel => Status switch
    {
        "queued" => "Sorban áll",
        "running" => Paused ? "Szüneteltetés kérve…" : "Fut",
        "paused" => "Szüneteltetve",
        "done" => "Kész",
        "error" => "Hiba",
        "cancelled" => "Megszakítva",
        "megszakadt" => "Megszakadt (a motor újraindult)",
        _ => Status,
    };
}

/// <summary>Egy esemény a meccs-JSON-ból (match_events_*.json / events_*.json).</summary>
public sealed class MatchEvent
{
    [JsonPropertyName("type")] public string Type { get; set; } = "";
    [JsonPropertyName("time")] public double Time { get; set; }
    [JsonPropertyName("start_sec")] public double? StartSec { get; set; }
    [JsonPropertyName("end_sec")] public double? EndSec { get; set; }
    [JsonPropertyName("team")] public string? Team { get; set; }
    [JsonPropertyName("desc")] public string? Desc { get; set; }
    [JsonPropertyName("reliability")] public double? Reliability { get; set; }
    [JsonPropertyName("half")] public int? Half { get; set; }
    [JsonPropertyName("half_label")] public string? HalfLabel { get; set; }
    [JsonPropertyName("match_clock")] public string? MatchClock { get; set; }
    [JsonPropertyName("full_match_sec")] public double? FullMatchSec { get; set; }

    public string Label => EventLabels.Label(Type);
    public string Icon => EventLabels.Icon(Type);
    public bool IsGoalkeeper => Type.StartsWith("gk_", StringComparison.Ordinal);
    public string TeamLabel => Team switch { "team1" => "Hazai", "team2" => "Vendég", _ => "" };
    public string ClockLabel => !string.IsNullOrEmpty(MatchClock) ? MatchClock! : FormatTime(Time);
    public string Title => string.IsNullOrEmpty(TeamLabel) ? $"{Icon} {Label}" : $"{Icon} {Label} – {TeamLabel}";
    public string Subtitle => $"{ClockLabel}{(string.IsNullOrEmpty(HalfLabel) ? "" : $" ({HalfLabel})")}  {Desc}";

    /// <summary>Klip-időablak az eredeti videóban (ha az eseménynek nincs saját ablaka: idő −5 / +4 mp).</summary>
    public (double Start, double End) ClipWindow(double pre = 5, double post = 4)
    {
        var s = StartSec ?? Time - pre;
        var e = EndSec ?? Time + post;
        return (Math.Max(0, s), Math.Max(s + 0.5, e));
    }

    public static string FormatTime(double sec)
    {
        if (sec < 0) sec = 0;
        var t = TimeSpan.FromSeconds(sec);
        return t.TotalHours >= 1 ? $"{(int)t.TotalHours}:{t.Minutes:00}:{t.Seconds:00}" : $"{t.Minutes}:{t.Seconds:00}";
    }
}

public static class EventLabels
{
    private static readonly Dictionary<string, (string Label, string Icon)> Map = new()
    {
        ["goal"] = ("Gól", "⚽"),
        ["shot"] = ("Lövés", "🎯"),
        ["pass"] = ("Passz", "➡"),
        ["tackle"] = ("Szerelés", "🦵"),
        ["gk_save"] = ("Kapus védés", "🧤"),
        ["gk_claim_cross"] = ("Beadás leszedése", "🙌"),
        ["gk_goal_kick"] = ("Kirúgás", "🦶"),
        ["gk_long_ball"] = ("Kapus hosszú labda", "📏"),
        ["gk_short_pass"] = ("Kapus rövid passz", "↪"),
        ["gk_conceded"] = ("Kapott gól", "❌"),
        ["gk_sweeper"] = ("Kapus kifutás (söprögetés)", "🏃"),
        ["gk_positioning_error"] = ("Kapus helyezkedési hiba", "⚠"),
    };

    public static string Label(string type) => Map.TryGetValue(type, out var v) ? v.Label : type;
    public static string Icon(string type) => Map.TryGetValue(type, out var v) ? v.Icon : "•";
}

public sealed class ExportedClip
{
    [JsonPropertyName("start_sec")] public double StartSec { get; set; }
    [JsonPropertyName("end_sec")] public double EndSec { get; set; }
    [JsonPropertyName("url")] public string? Url { get; set; }
    [JsonPropertyName("local_path")] public string? LocalPath { get; set; }
}

public sealed class ExportResult
{
    [JsonPropertyName("status")] public string? Status { get; set; }
    [JsonPropertyName("message")] public string? Message { get; set; }
    [JsonPropertyName("clips")] public List<ExportedClip> Clips { get; set; } = new();
    [JsonPropertyName("merged_url")] public string? MergedUrl { get; set; }
    [JsonPropertyName("merged_local_path")] public string? MergedLocalPath { get; set; }
}

/// <summary>Egy kész (vagy részben kész) analyze-match eredmény a kijelzéshez szükséges mezőkkel.</summary>
public sealed class MatchResultView
{
    public string? RunId { get; init; }
    public string? VideoPath { get; init; }
    public string? MatchEventsUrl { get; init; }
    public string? MatchEventsPath { get; init; }
    public string? FullMatchUrl { get; init; }
    public string? FullMatchLocalPath { get; init; }
    public bool Partial { get; init; }
    public Dictionary<string, int> EventCounts { get; init; } = new();
    /// <summary>Hőtérkép URL-ek: kulcs pl. "Összesítve – Hazai", érték: "/clips/heatmap_...png".</summary>
    public List<(string Title, string Url)> Heatmaps { get; init; } = new();
    /// <summary>Klip-összeállítások: cím + összefűzött videó URL.</summary>
    public List<(string Title, string Url, int Count)> Compilations { get; init; } = new();
    public List<string> Notes { get; init; } = new();
    public List<string> Warnings { get; init; } = new();
    public List<(string Title, string Url)> ReviewPages { get; init; } = new();

    public static MatchResultView? From(JsonElement? result)
    {
        if (result is not { ValueKind: JsonValueKind.Object } r) return null;
        var heat = new List<(string, string)>();
        if (r.TryGetProperty("heatmaps", out var hm) && hm.ValueKind == JsonValueKind.Object)
        {
            foreach (var (key, title) in new[] { ("combined", "Összesítve"), ("first_half", "1. félidő"), ("second_half", "2. félidő") })
            {
                if (!hm.TryGetProperty(key, out var h) || h.ValueKind != JsonValueKind.Object) continue;
                if (!h.TryGetProperty("team_heatmaps", out var th) || th.ValueKind != JsonValueKind.Object) continue;
                foreach (var team in th.EnumerateObject())
                {
                    if (team.Value.ValueKind != JsonValueKind.String) continue;
                    var teamLabel = team.Name == "team1" ? "Hazai" : team.Name == "team2" ? "Vendég" : team.Name;
                    heat.Add(($"{title} – {teamLabel}", team.Value.GetString()!));
                }
            }
        }
        var comps = new List<(string, string, int)>();
        if (r.TryGetProperty("compilations", out var cp) && cp.ValueKind == JsonValueKind.Object)
        {
            foreach (var c in cp.EnumerateObject())
            {
                var v = c.Value;
                if (v.ValueKind != JsonValueKind.Object) continue;
                var url = Str(v, "merged_url");
                if (string.IsNullOrEmpty(url)) continue;
                var count = v.TryGetProperty("item_count", out var ic) && ic.TryGetInt32(out var n) ? n : 0;
                comps.Add((Str(v, "title") ?? c.Name, url!, count));
            }
        }
        var reviews = new List<(string, string)>();
        if (r.TryGetProperty("analysis", out var an) && an.ValueKind == JsonValueKind.Object)
        {
            foreach (var (key, title) in new[] { ("first_half", "1. félidő"), ("second_half", "2. félidő") })
                if (an.TryGetProperty(key, out var a) && Str(a, "review_url") is { } ru) reviews.Add((title, ru));
        }
        var counts = new Dictionary<string, int>();
        if (r.TryGetProperty("event_counts", out var ec) && ec.ValueKind == JsonValueKind.Object)
            foreach (var p in ec.EnumerateObject())
                if (p.Value.TryGetInt32(out var n)) counts[p.Name] = n;

        string? fullUrl = null, fullLocal = null;
        if (r.TryGetProperty("cut", out var cut) && cut.ValueKind == JsonValueKind.Object)
        {
            fullUrl = Str(cut, "full_match_url");
            if (cut.TryGetProperty("local_paths", out var lp) && lp.ValueKind == JsonValueKind.Object)
                fullLocal = Str(lp, "full_match");
        }
        return new MatchResultView
        {
            RunId = Str(r, "run_id"),
            VideoPath = Str(r, "video_path"),
            MatchEventsUrl = Str(r, "match_events_url"),
            MatchEventsPath = Str(r, "match_events_path"),
            FullMatchUrl = fullUrl,
            FullMatchLocalPath = fullLocal,
            Partial = r.TryGetProperty("reszleges", out var rp) && rp.ValueKind == JsonValueKind.True,
            EventCounts = counts,
            Heatmaps = heat,
            Compilations = comps,
            Notes = StrList(r, "notes"),
            Warnings = StrList(r, "warnings"),
            ReviewPages = reviews,
        };
    }

    private static string? Str(JsonElement e, string name) =>
        e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static List<string> StrList(JsonElement e, string name)
    {
        var list = new List<string>();
        if (e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Array)
            foreach (var x in v.EnumerateArray())
                if (x.ValueKind == JsonValueKind.String) list.Add(x.GetString()!);
        return list;
    }
}

/// <summary>Az elemzés beállításai (a /jobs/analyze-match form mezői).</summary>
public sealed class MatchAnalysisOptions
{
    public double ExpectedHalfMin { get; set; } = 45;
    /// <summary>Kézi félidőhatárok ("pp:mm" vagy mp); ha mind a négy meg van adva, nincs automatikus felismerés.</summary>
    public string FirstStart { get; set; } = "";
    public string FirstEnd { get; set; } = "";
    public string SecondStart { get; set; } = "";
    public string SecondEnd { get; set; } = "";
    public bool Background { get; set; } = true;
    public bool Heatmaps { get; set; } = true;
    public bool GoalkeeperCompilations { get; set; } = true;
    public bool Highlights { get; set; } = true;
    public bool SkipCut { get; set; }
    public bool Team1AttacksRight { get; set; } = true;

    public IEnumerable<KeyValuePair<string, string>> ToForm(string videoPath)
    {
        yield return new("video_path", videoPath);
        yield return new("expected_half_min", ExpectedHalfMin.ToString(CultureInfo.InvariantCulture));
        if (!string.IsNullOrWhiteSpace(FirstStart)) yield return new("first_start", FirstStart.Trim());
        if (!string.IsNullOrWhiteSpace(FirstEnd)) yield return new("first_end", FirstEnd.Trim());
        if (!string.IsNullOrWhiteSpace(SecondStart)) yield return new("second_start", SecondStart.Trim());
        if (!string.IsNullOrWhiteSpace(SecondEnd)) yield return new("second_end", SecondEnd.Trim());
        yield return new("background", Background ? "true" : "false");
        yield return new("heatmaps", Heatmaps ? "true" : "false");
        yield return new("skip_cut", SkipCut ? "true" : "false");
        yield return new("team1_attacks_right", Team1AttacksRight ? "true" : "false");
        var comps = new List<string>();
        if (GoalkeeperCompilations) comps.Add("goalkeepers");
        if (Highlights) comps.Add("highlights");
        yield return new("compilations", string.Join(",", comps));
    }
}

public sealed class EngineApiException : Exception
{
    public EngineApiException(string message) : base(message) { }
}

public sealed class MatchJobService
{
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };
    private readonly HttpClient _http;
    private readonly Func<string> _baseUrl;

    public MatchJobService() : this(new HttpClient { Timeout = TimeSpan.FromMinutes(2) }, () => Engine.BaseUrl) { }

    public MatchJobService(HttpClient http, Func<string> baseUrl)
    {
        _http = http;
        _baseUrl = baseUrl;
    }

    private string U(string path) => _baseUrl() + path;

    /// <summary>Motor-relatív ("/clips/…") vagy teljes URL → teljes URL.</summary>
    public string FullUrl(string relativeOrAbsolute) =>
        relativeOrAbsolute.StartsWith("http", StringComparison.OrdinalIgnoreCase)
            ? relativeOrAbsolute
            : _baseUrl() + (relativeOrAbsolute.StartsWith('/') ? relativeOrAbsolute : "/" + relativeOrAbsolute);

    public async Task<string> StartMatchAnalysisAsync(string videoPath, MatchAnalysisOptions options, CancellationToken ct = default)
    {
        using var content = new FormUrlEncodedContent(options.ToForm(videoPath));
        using var resp = await _http.PostAsync(U("/jobs/analyze-match"), content, ct);
        var root = await ReadJsonAsync(resp, ct);
        return root.GetProperty("job_id").GetString() ?? throw new EngineApiException("A motor nem adott feladat-azonosítót.");
    }

    public async Task<JobInfo> GetJobAsync(string jobId, CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync(U($"/jobs/{Uri.EscapeDataString(jobId)}"), ct);
        var root = await ReadJsonAsync(resp, ct);
        return root.GetProperty("job").Deserialize<JobInfo>(JsonOptions)!;
    }

    public async Task<List<JobInfo>> ListJobsAsync(int limit = 50, CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync(U($"/jobs?limit={limit}"), ct);
        var root = await ReadJsonAsync(resp, ct);
        return root.GetProperty("jobs").Deserialize<List<JobInfo>>(JsonOptions) ?? new();
    }

    public Task<JobInfo> PauseAsync(string jobId, CancellationToken ct = default) => JobActionAsync(jobId, "pause", ct);
    public Task<JobInfo> ResumeAsync(string jobId, CancellationToken ct = default) => JobActionAsync(jobId, "resume", ct);
    public Task<JobInfo> CancelAsync(string jobId, CancellationToken ct = default) => JobActionAsync(jobId, "cancel", ct);

    private async Task<JobInfo> JobActionAsync(string jobId, string action, CancellationToken ct)
    {
        using var resp = await _http.PostAsync(U($"/jobs/{Uri.EscapeDataString(jobId)}/{action}"), null, ct);
        var root = await ReadJsonAsync(resp, ct);
        return root.GetProperty("job").Deserialize<JobInfo>(JsonOptions)!;
    }

    /// <summary>A meccs-JSON eseményei (URL: "/clips/match_events_….json").</summary>
    public async Task<List<MatchEvent>> LoadEventsAsync(string eventsUrl, CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync(FullUrl(eventsUrl), ct);
        if (!resp.IsSuccessStatusCode) throw new EngineApiException($"Az eseményfájl nem tölthető le ({(int)resp.StatusCode}).");
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        if (!doc.RootElement.TryGetProperty("events", out var evs) || evs.ValueKind != JsonValueKind.Array) return new();
        var list = evs.Deserialize<List<MatchEvent>>(JsonOptions) ?? new();
        list.Sort((a, b) => a.Time.CompareTo(b.Time));
        return list;
    }

    /// <summary>A kiválasztott események kivágása háttérfeladatként (kind=export-clips); visszatér: job id.</summary>
    public async Task<string> StartExportClipsAsync(string videoPath, IEnumerable<MatchEvent> events, string title,
        bool merge = true, CancellationToken ct = default)
    {
        var items = events.Select(e =>
        {
            var (s, en) = e.ClipWindow();
            return new Dictionary<string, double> { ["start_sec"] = Math.Round(s, 3), ["end_sec"] = Math.Round(en, 3) };
        }).ToList();
        if (items.Count == 0) throw new EngineApiException("Nincs kiválasztott esemény.");
        var form = new Dictionary<string, string>
        {
            ["kind"] = "export-clips",
            ["video_path"] = videoPath,
            ["events_json"] = JsonSerializer.Serialize(items),
            ["merge"] = merge ? "true" : "false",
            ["accurate"] = "true",
            ["title"] = title,
            ["background"] = "true",
        };
        using var content = new FormUrlEncodedContent(form);
        using var resp = await _http.PostAsync(U("/jobs/run"), content, ct);
        var root = await ReadJsonAsync(resp, ct);
        return root.GetProperty("job_id").GetString()!;
    }

    public static ExportResult? ParseExportResult(JobInfo job) =>
        job.Result is { ValueKind: JsonValueKind.Object } r ? r.Deserialize<ExportResult>(JsonOptions) : null;

    /// <summary>Megvárja a feladat végét, közben a haladást jelenti.</summary>
    public async Task<JobInfo> WaitAsync(string jobId, IProgress<JobInfo>? progress = null, TimeSpan? interval = null,
        CancellationToken ct = default)
    {
        while (true)
        {
            var job = await GetJobAsync(jobId, ct);
            progress?.Report(job);
            if (job.IsFinished) return job;
            await Task.Delay(interval ?? TimeSpan.FromSeconds(1.5), ct);
        }
    }

    private static async Task<JsonElement> ReadJsonAsync(HttpResponseMessage resp, CancellationToken ct)
    {
        var text = await resp.Content.ReadAsStringAsync(ct);
        JsonElement root;
        try
        {
            using var doc = JsonDocument.Parse(text);
            root = doc.RootElement.Clone();
        }
        catch (JsonException)
        {
            throw new EngineApiException($"A motor válasza nem értelmezhető (HTTP {(int)resp.StatusCode}).");
        }
        var status = root.TryGetProperty("status", out var s) ? s.GetString() : null;
        if (!resp.IsSuccessStatusCode || status == "error")
        {
            var msg = root.TryGetProperty("message", out var m) ? m.GetString()
                    : root.TryGetProperty("detail", out var d) ? d.ToString() : null;
            throw new EngineApiException(msg ?? $"A motor hibát jelzett (HTTP {(int)resp.StatusCode}).");
        }
        return root;
    }
}

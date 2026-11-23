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
    [JsonPropertyName("started_at")] public string? StartedAt { get; set; }
    [JsonPropertyName("finished_at")] public string? FinishedAt { get; set; }
    [JsonPropertyName("elapsed_sec")] public double? ElapsedSec { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
    [JsonPropertyName("params")] public JsonElement? Params { get; set; }
    [JsonPropertyName("result")] public JsonElement? Result { get; set; }

    public bool IsFinished => Status is "done" or "error" or "cancelled" or "megszakadt";

    /// <summary>Az indítás óta eltelt idő (a motor ugyanezen a gépen fut, helyi idővel).</summary>
    public TimeSpan? Elapsed
    {
        get
        {
            if (ElapsedSec is { } done) return TimeSpan.FromSeconds(done);
            if (!DateTime.TryParse(StartedAt, out var start)) return null;
            var d = DateTime.Now - start;
            return d < TimeSpan.Zero ? TimeSpan.Zero : d;
        }
    }

    /// <summary>Durva becslés a hátralévő időre (csak ha már elég adat van), különben null.</summary>
    public TimeSpan? Remaining
    {
        get
        {
            if (IsFinished || Status == "paused" || Elapsed is not { } el) return null;
            if (Progress < 16 || el.TotalSeconds < 120) return null;   // a félidő-felismerés és a vágás még nem jellemző
            return TimeSpan.FromSeconds(el.TotalSeconds * (100 - Progress) / Progress);
        }
    }

    /// <summary>"10.4% • 4:12 telt el • kb. 1 ó 20 perc van hátra"</summary>
    public string ProgressText
    {
        get
        {
            var parts = new List<string>();
            if (!IsFinished) parts.Add($"{Progress.ToString("0.0", CultureInfo.InvariantCulture)}%");
            if (Elapsed is { } el) parts.Add($"{FormatDuration(el)} {(IsFinished ? "volt" : "telt el")}");
            if (Remaining is { } rem) parts.Add($"kb. {FormatDuration(rem)} van hátra");
            return string.Join(" • ", parts);
        }
    }

    public static string FormatDuration(TimeSpan t) =>
        t.TotalHours >= 1 ? $"{(int)t.TotalHours} ó {t.Minutes} perc"
        : t.TotalMinutes >= 1 ? $"{(int)t.TotalMinutes} perc {t.Seconds:00} mp"
        : $"{t.Seconds} mp";
    public bool IsRunning => Status is "running" or "queued";

    public string? VideoPath =>
        Params is { ValueKind: JsonValueKind.Object } p && p.TryGetProperty("video_path", out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString() : null;

    public string StatusLabel => Status switch
    {
        "queued" => "Sorban áll",
        "running" => CancelRequested ? "Megszakítás folyamatban…" : Paused ? "Szüneteltetés kérve…" : "Fut",
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
    [JsonPropertyName("category")] public string? CategoryRaw { get; set; }
    [JsonPropertyName("experimental")] public bool? ExperimentalRaw { get; set; }
    /// <summary>Passznál / szabadrúgásnál: "sajat_terfel" vagy "ellenfel_terfel".</summary>
    [JsonPropertyName("zone")] public string? Zone { get; set; }
    /// <summary>Kapus Modul csoport (keeper_focus.py): kapus / lovesek / beadasok / rogzitett / zona / mozgas.</summary>
    [JsonPropertyName("group")] public string? Group { get; set; }

    /// <summary>A kalap (events.py EVENT_CATEGORIES); régi eredményfájlnál a típusból számolva.</summary>
    public string Category => !string.IsNullOrEmpty(CategoryRaw) ? CategoryRaw! : EventCategories.Of(Type);
    public bool Experimental => ExperimentalRaw ?? EventCategories.IsExperimental(Type);

    public string Label => EventLabels.Label(Type);
    /// <summary>A visszajelzés kulcsa – egyezik a feedback.py event_key()-ével ("shot@634.20").</summary>
    public string FeedbackKey => $"{Type}@{Time.ToString("0.00", CultureInfo.InvariantCulture)}";
    public string Icon => EventLabels.Icon(Type);
    public bool IsGoalkeeper => Type.StartsWith("gk_", StringComparison.Ordinal);
    public string TeamLabel => TeamNames.Of(Team);
    /// <summary>Az időpont az eredeti videóban (ezt látod a lejátszón is).</summary>
    public string VideoTimeLabel => FormatTime(Time);
    /// <summary>Hivatalos meccsidő: az 1. félidő 0:00-tól, a 2. félidő 45:00-tól indul (a beállított félidőhossztól).</summary>
    public string ClockLabel => !string.IsNullOrEmpty(MatchClock) ? MatchClock! : FormatTime(Time);
    public string Title => string.IsNullOrEmpty(TeamLabel) ? $"{Icon} {Label}" : $"{Icon} {Label} – {TeamLabel}";
    public string Subtitle =>
        $"🎬 {VideoTimeLabel} a videóban" +
        (string.IsNullOrEmpty(MatchClock) ? "" : $"  •  ⏱ {MatchClock} meccsidő{(string.IsNullOrEmpty(HalfLabel) ? "" : $" ({HalfLabel})")}") +
        $"  •  {EventCategories.Label(Category)}{(Experimental ? " (kísérleti)" : "")}  •  {Desc}";

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

/// <summary>
/// A két csapat megjelenített neve. A motor a csapatokat mezszín szerint két csoportra osztja: a "team1" az a
/// csapat, amelyik az 1. félidőben a bal oldali kaput védi (jobbra támad), a "team2" a másik – hogy melyik a
/// hazai, azt nem tudja. Ezért alapból "1. csapat" / "2. csapat", és az edző nevezheti át (pl. "Mi", "Ellenfél").
/// </summary>
public static class TeamNames
{
    public static string Team1 { get; set; } = "1. csapat";
    public static string Team2 { get; set; } = "2. csapat";
    public static string Of(string? team) => team switch { "team1" => Team1, "team2" => Team2, _ => "" };
}

public static class EventLabels
{
    private static readonly Dictionary<string, (string Label, string Icon)> Map = new()
    {
        ["goal"] = ("Gól", "⚽"),
        ["shot"] = ("Lövés", "🎯"),
        ["pass"] = ("Passz", "➡"),
        ["pass_failed"] = ("Sikertelen passz", "↯"),
        ["cross"] = ("Beadás", "↗"),
        ["throw_in"] = ("Bedobás", "🙆"),
        ["corner"] = ("Szöglet", "🚩"),
        ["free_kick"] = ("Szabadrúgás", "🎯"),
        ["kickoff"] = ("Kezdőrúgás", "⏺"),
        ["tackle"] = ("Szerelés", "🦵"),
        ["interception"] = ("Labdaszerzés (elfogott passz)", "✋"),
        ["dribble"] = ("Labdavezetés", "🏃"),
        ["gk_save"] = ("Kapus védés", "🧤"),
        ["gk_claim_cross"] = ("Beadás leszedése", "🙌"),
        ["gk_goal_kick"] = ("Kirúgás", "🦶"),
        ["gk_long_ball"] = ("Kapus hosszú labda", "📏"),
        ["gk_short_pass"] = ("Kapus rövid passz", "↪"),
        ["gk_conceded"] = ("Kapott gól", "❌"),
        ["gk_sweeper"] = ("Kapus kifutás (söprögetés)", "🏃"),
        ["gk_positioning_error"] = ("Kapus helyezkedési hiba", "⚠"),
        ["gk_move"] = ("Kapus mozgás", "↔"),
        ["attack_in_zone"] = ("Támadás a saját zónában", "🛡"),
    };

    public static string Label(string type) => Map.TryGetValue(type, out var v) ? v.Label : type;

    /// <summary>A kézzel felvehető eseménytípusok (kód, magyar név) – a "+ Hiányzó esemény" választólistája.</summary>
    public static IReadOnlyList<(string Type, string Label)> All =>
        Map.Where(kv => kv.Key is not ("gk_positioning_error" or "gk_move" or "attack_in_zone")).Select(kv => (kv.Key, $"{kv.Value.Icon} {kv.Value.Label}")).ToList();
    public static string Icon(string type) => Map.TryGetValue(type, out var v) ? v.Icon : "•";
}

public sealed class KeeperFocus
{
    public List<(string Key, string Label, int Count)> Groups { get; } = new();
    public List<MatchEvent> Items { get; set; } = new();
}

/// <summary>Esemény-kalapok (egyezik az events.py EVENT_CATEGORIES / CATEGORY_LABELS értékeivel).</summary>
public static class EventCategories
{
    public static readonly IReadOnlyList<(string Key, string Label)> All = new List<(string, string)>
    {
        ("rogzitett", "🚩 Rögzített helyzetek"),
        ("befejezes", "🎯 Befejezések"),
        ("beadas", "↗ Beadások"),
        ("passz", "➡ Passzok"),
        ("labdaszerzes", "✋ Labdaszerzések"),
        ("labdavezetes", "🏃 Labdavezetés"),
        ("kapus", "🧤 Kapus események"),
        ("egyeb", "• Egyéb"),
    };

    private static readonly Dictionary<string, string> Map = new()
    {
        ["corner"] = "rogzitett", ["free_kick"] = "rogzitett", ["throw_in"] = "rogzitett",
        ["kickoff"] = "rogzitett", ["gk_goal_kick"] = "rogzitett",
        ["shot"] = "befejezes", ["goal"] = "befejezes",
        ["cross"] = "beadas",
        ["pass"] = "passz", ["pass_failed"] = "passz",
        ["tackle"] = "labdaszerzes", ["interception"] = "labdaszerzes",
        ["dribble"] = "labdavezetes",
    };

    /// <summary>A valós felvételen még bizonytalan típusok (events.py EXPERIMENTAL_TYPES).</summary>
    private static readonly HashSet<string> ExperimentalTypes = new() { "pass_failed", "tackle", "interception", "cross", "dribble" };

    public static string Of(string type) =>
        Map.TryGetValue(type, out var c) ? c : type.StartsWith("gk_", StringComparison.Ordinal) ? "kapus" : "egyeb";

    public static bool IsExperimental(string type) => ExperimentalTypes.Contains(type);

    public static string Label(string key) => All.FirstOrDefault(c => c.Key == key).Label ?? key;
}

/// <summary>Edzői visszajelzés egy meccsre (feedback.py): ítéletek + kézzel felvett hiányzó események.</summary>
public sealed class FeedbackData
{
    [JsonPropertyName("items")] public List<FeedbackVerdict> Items { get; set; } = new();
    [JsonPropertyName("missing")] public List<FeedbackMissing> Missing { get; set; } = new();
}

public sealed class FeedbackVerdict
{
    [JsonPropertyName("key")] public string Key { get; set; } = "";
    [JsonPropertyName("verdict")] public string? Verdict { get; set; }
    [JsonPropertyName("fontos")] public bool? Fontos { get; set; }
}

public sealed class FeedbackMissing
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("type")] public string Type { get; set; } = "";
    [JsonPropertyName("time")] public double Time { get; set; }
    [JsonPropertyName("team")] public string? Team { get; set; }
    [JsonPropertyName("note")] public string? Note { get; set; }
    [JsonPropertyName("fontos")] public bool? Fontos { get; set; }
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
                    var teamLabel = TeamNames.Of(team.Name) is { Length: > 0 } tn ? tn : team.Name;
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

    /// <summary>A legutóbb betöltött meccs-JSON mezszínei (RGB): "team1", "team2", "other".</summary>
    public Dictionary<string, (double R, double G, double B)> LastTeamColors { get; private set; } = new();

    /// <summary>A meccs-JSON eseményei (URL: "/clips/match_events_….json").</summary>
    public async Task<List<MatchEvent>> LoadEventsAsync(string eventsUrl, CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync(FullUrl(eventsUrl), ct);
        if (!resp.IsSuccessStatusCode) throw new EngineApiException($"Az eseményfájl nem tölthető le ({(int)resp.StatusCode}).");
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        LastTeamColors = new();
        if (doc.RootElement.TryGetProperty("team_colors", out var tc) && tc.ValueKind == JsonValueKind.Object)
        {
            foreach (var p in tc.EnumerateObject())
            {
                if (p.Value.ValueKind != JsonValueKind.Array || p.Value.GetArrayLength() < 3) continue;
                var c = p.Value.EnumerateArray().Select(x => x.TryGetDouble(out var d) ? d : 0).ToArray();
                LastTeamColors[p.Name] = (c[0], c[1], c[2]);
            }
        }
        if (!doc.RootElement.TryGetProperty("events", out var evs) || evs.ValueKind != JsonValueKind.Array) return new();
        var list = evs.Deserialize<List<MatchEvent>>(JsonOptions) ?? new();
        list.Sort((a, b) => a.Time.CompareTo(b.Time));
        return list;
    }

    /// <summary>Kapus Modul: a csapat kapusát érintő jelenetek egy meccs-eredményből (GET /keeper-focus).</summary>
    public async Task<KeeperFocus> GetKeeperFocusAsync(string eventsUrl, string team, double minMoveM = 2.0,
        bool moves = true, CancellationToken ct = default)
    {
        var url = U($"/keeper-focus?events={Uri.EscapeDataString(eventsUrl)}&team={team}&moves={(moves ? 1 : 0)}" +
                    $"&min_move={minMoveM.ToString(CultureInfo.InvariantCulture)}");
        using var resp = await _http.GetAsync(url, ct);
        var body = await resp.Content.ReadAsStringAsync(ct);
        using var doc = JsonDocument.Parse(body);
        var root = doc.RootElement;
        if (!resp.IsSuccessStatusCode || root.TryGetProperty("status", out var st) && st.GetString() != "success")
        {
            var msg = root.TryGetProperty("message", out var m) ? m.GetString() : null;
            if ((int)resp.StatusCode == 404 && msg is null)
                msg = "A motor még nem ismeri a Kapus Modult – frissítsd (git pull) és indítsd újra.";
            throw new EngineApiException(msg ?? $"A Kapus Modul nem tölthető be ({(int)resp.StatusCode}).");
        }
        var res = new KeeperFocus();
        if (root.TryGetProperty("groups", out var gs) && gs.ValueKind == JsonValueKind.Array)
            foreach (var g in gs.EnumerateArray())
                res.Groups.Add((g.GetProperty("key").GetString() ?? "", g.GetProperty("label").GetString() ?? "",
                    g.TryGetProperty("count", out var c) ? c.GetInt32() : 0));
        if (root.TryGetProperty("items", out var its) && its.ValueKind == JsonValueKind.Array)
            res.Items = its.Deserialize<List<MatchEvent>>(JsonOptions) ?? new();
        return res;
    }

    /// <summary>A motor állapota (GET /health) a kijelzéshez: melyik modell, melyik eszközön.</summary>
    public async Task<string> GetEngineSummaryAsync(CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync(U("/health"), ct);
        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct));
        var r = doc.RootElement;
        string? S(string n) => r.TryGetProperty(n, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        var loaded = r.TryGetProperty("model_loaded", out var ml) && ml.ValueKind == JsonValueKind.True;
        var device = (S("device") ?? "…").ToUpperInvariant();
        if (!loaded)
            return $"🧠 Beállított modell: {S("model_configured") ?? "?"} (az első elemzéskor töltődik be) • eszköz: {device}";
        var weights = S("model_weights") ?? "?";
        // ONNX: "végrehajtó: CUDAExecutionProvider, CPUExecutionProvider" – az első a ténylegesen használt
        // (a CPU csak tartalék); figyelmeztetés csak akkor, ha a CUDA nincs a listában, pedig van videókártya.
        var warn = weights.StartsWith("ONNX") && device == "CUDA" && !weights.Contains("CUDAExecutionProvider")
                   && !weights.Contains("DmlExecutionProvider")
            ? "  ⚠ a modell processzoron fut (az onnxruntime-gpu nem működik)" : "";
        return $"🧠 Modell: {weights} • eszköz: {device}{warn}";
    }

    public async Task<FeedbackData> GetFeedbackAsync(string eventsUrl, CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync(U($"/feedback?events={Uri.EscapeDataString(eventsUrl)}"), ct);
        var root = await ReadJsonAsync(resp, ct);
        return root.GetProperty("feedback").Deserialize<FeedbackData>(JsonOptions) ?? new FeedbackData();
    }

    /// <summary>Egy automatikus esemény ítélete: "ok" (valódi), "hamis" (téves), vagy null (törlés).</summary>
    public async Task SetVerdictAsync(string eventsUrl, MatchEvent ev, string? verdict, CancellationToken ct = default)
    {
        var body = JsonSerializer.Serialize(new
        {
            events = eventsUrl,
            @event = new { type = ev.Type, time = ev.Time, team = ev.Team },
            verdict = verdict ?? "",
        });
        using var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        using var resp = await _http.PostAsync(U("/feedback"), content, ct);
        await ReadJsonAsync(resp, ct);
    }

    /// <summary>"Fontos" (⭐) jelölés – az automatikus eseményre (ev) vagy a kézzel felvettre (manualId).</summary>
    public async Task SetImportantAsync(string eventsUrl, MatchEvent ev, string? manualId, bool fontos,
        CancellationToken ct = default)
    {
        object target = manualId is not null
            ? new { id = manualId }
            : new { type = ev.Type, time = ev.Time, team = ev.Team };
        var body = JsonSerializer.Serialize(new { events = eventsUrl, important = new { @event = target, fontos } });
        using var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        using var resp = await _http.PostAsync(U("/feedback"), content, ct);
        await ReadJsonAsync(resp, ct);
    }

    /// <summary>Háttérfeladat indítása a /jobs/run végponton (kind + form mezők) – a job azonosítóját adja.</summary>
    public async Task<string> StartJobAsync(string kind, IDictionary<string, string> fields, CancellationToken ct = default)
    {
        var form = new Dictionary<string, string>(fields) { ["kind"] = kind };
        using var content = new FormUrlEncodedContent(form);
        using var resp = await _http.PostAsync(U("/jobs/run"), content, ct);
        var root = await ReadJsonAsync(resp, ct);
        return root.GetProperty("job_id").GetString()!;
    }

    /// <summary>Címkéző csomag (rövid részletek + offline weboldal) egy meccs eseményeiből.</summary>
    public Task<string> StartLabelPackageAsync(string eventsUrl, int nRandom = 20, int nDoubt = 20, CancellationToken ct = default) =>
        StartJobAsync("label-package", new Dictionary<string, string>
        {
            ["events_url"] = eventsUrl, ["n_random"] = nRandom.ToString(CultureInfo.InvariantCulture),
            ["n_doubt"] = nDoubt.ToString(CultureInfo.InvariantCulture), ["background"] = "true",
        }, ct);

    /// <summary>A laptopon mentett cimkek_*.json beolvasása (tune: utána tanulás is).</summary>
    public Task<string> StartImportLabelsAsync(string labelsPath, bool tune, CancellationToken ct = default) =>
        StartJobAsync("import-labels", new Dictionary<string, string>
        {
            ["labels_path"] = labelsPath, ["tune"] = tune ? "true" : "false", ["background"] = "true",
        }, ct);

    /// <summary>Tanulás az összes eddigi jelölésből (az esemény-küszöbök hangolása).</summary>
    public Task<string> StartTuneAsync(CancellationToken ct = default) =>
        StartJobAsync("tune-events", new Dictionary<string, string> { ["background"] = "true" }, ct);

    public async Task<FeedbackMissing> AddMissingAsync(string eventsUrl, string type, double timeSec, string? team,
        CancellationToken ct = default)
    {
        var body = JsonSerializer.Serialize(new { events = eventsUrl, missing = new { type, time = timeSec, team } });
        using var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        using var resp = await _http.PostAsync(U("/feedback"), content, ct);
        var root = await ReadJsonAsync(resp, ct);
        return root.GetProperty("item").Deserialize<FeedbackMissing>(JsonOptions)!;
    }

    public async Task RemoveMissingAsync(string eventsUrl, string id, CancellationToken ct = default)
    {
        var body = JsonSerializer.Serialize(new { events = eventsUrl, remove_missing = id });
        using var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        using var resp = await _http.PostAsync(U("/feedback"), content, ct);
        await ReadJsonAsync(resp, ct);
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

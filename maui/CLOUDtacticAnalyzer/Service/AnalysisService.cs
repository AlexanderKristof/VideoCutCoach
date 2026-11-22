using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace TacticalAnalyzer.Services;

public class ClipResult
{
    [JsonPropertyName("clip_url")] public string? ClipUrl { get; set; }
    [JsonPropertyName("local_path")] public string? LocalPath { get; set; }
    [JsonPropertyName("events_path")] public string? EventsPath { get; set; }
}

public class TimelineEvent
{
    [JsonPropertyName("type")] public string Type { get; set; } = "";
    [JsonPropertyName("time")] public double Time { get; set; }
    [JsonPropertyName("desc")] public string Desc { get; set; } = "";
}

public class AiClickResult
{
    public string Lower { get; set; } = "";
    public string Upper { get; set; } = "";
    public double BoxX { get; set; }
    public double BoxY { get; set; }
    public double BoxW { get; set; }
    public double BoxH { get; set; }
}

public class AnalysisService
{
    private readonly HttpClient _httpClient;
#if ANDROID
    public const string BaseUrl = "http://10.0.2.2:8000";
#else
    public const string BaseUrl = "http://localhost:8000";
#endif
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public AnalysisService() { _httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(15) }; }

    public async Task<string?> GetFrameAtTimeAsync(string filePath, double timeSec)
    {
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent(filePath), "video_path");
        content.Add(new StringContent(timeSec.ToString(CultureInfo.InvariantCulture)), "time_sec");

        var response = await _httpClient.PostAsync($"{BaseUrl}/get-frame-time", content);
        if (response.IsSuccessStatusCode)
        {
            var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            if (doc.RootElement.TryGetProperty("frame_url", out var url)) return $"{BaseUrl}{url.GetString()}";
        }
        return null;
    }

    public async Task<AiClickResult?> AnalyzeClickWithAiAsync(string imageUrl, double x, double y, double w, double h, string existingLower = "", string existingUpper = "")
    {
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent(imageUrl), "image_url");
        content.Add(new StringContent(x.ToString(CultureInfo.InvariantCulture)), "ui_x");
        content.Add(new StringContent(y.ToString(CultureInfo.InvariantCulture)), "ui_y");
        content.Add(new StringContent(w.ToString(CultureInfo.InvariantCulture)), "ui_w");
        content.Add(new StringContent(h.ToString(CultureInfo.InvariantCulture)), "ui_h");
        content.Add(new StringContent(existingLower), "existing_lower");
        content.Add(new StringContent(existingUpper), "existing_upper");

        var response = await _httpClient.PostAsync($"{BaseUrl}/analyze-click-ai", content);
        if (response.IsSuccessStatusCode)
        {
            var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            if (doc.RootElement.TryGetProperty("status", out var status) && status.GetString() == "success")
            {
                var box = doc.RootElement.GetProperty("ui_box");
                return new AiClickResult
                {
                    Lower = doc.RootElement.GetProperty("lower").GetString() ?? "",
                    Upper = doc.RootElement.GetProperty("upper").GetString() ?? "",
                    BoxX = box.GetProperty("x").GetDouble(),
                    BoxY = box.GetProperty("y").GetDouble(),
                    BoxW = box.GetProperty("w").GetDouble(),
                    BoxH = box.GetProperty("h").GetDouble()
                };
            }
        }
        return null;
    }

    public async Task<ClipResult?> CutRawVideoAsync(
        string filePath, double startSec, double endSec, bool devMode = false, string detectMode = "team",
        double roiTopPct = 0.25, double roiBottomPct = 0.10, double minPlayerHeightPct = 0.04,
        string grassHsvLower = "30,40,40", string grassHsvUpper = "85,255,255",
        string refHsvLower = "0,0,0", string refHsvUpper = "179,255,255",
        string team1Lower = "0,0,0", string team1Upper = "179,255,255",
        string team2Lower = "0,0,0", string team2Upper = "179,255,255",
        string keeper1Lower = "0,0,0", string keeper1Upper = "179,255,255",
        string keeper2Lower = "0,0,0", string keeper2Upper = "179,255,255",
        string calibPoints = "")
    {
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent(filePath), "video_path");
        content.Add(new StringContent(startSec.ToString(CultureInfo.InvariantCulture)), "start_sec");
        content.Add(new StringContent(endSec.ToString(CultureInfo.InvariantCulture)), "end_sec");
        content.Add(new StringContent(devMode.ToString().ToLower()), "dev_mode");
        content.Add(new StringContent(detectMode), "detect_mode");
        content.Add(new StringContent(calibPoints), "calib_points");

        content.Add(new StringContent(grassHsvLower), "grass_hsv_lower"); content.Add(new StringContent(grassHsvUpper), "grass_hsv_upper");
        content.Add(new StringContent(refHsvLower), "ref_hsv_lower"); content.Add(new StringContent(refHsvUpper), "ref_hsv_upper");
        content.Add(new StringContent(team1Lower), "team1_hsv_lower"); content.Add(new StringContent(team1Upper), "team1_hsv_upper");
        content.Add(new StringContent(team2Lower), "team2_hsv_lower"); content.Add(new StringContent(team2Upper), "team2_hsv_upper");
        content.Add(new StringContent(keeper1Lower), "keeper1_hsv_lower"); content.Add(new StringContent(keeper1Upper), "keeper1_hsv_upper");
        content.Add(new StringContent(keeper2Lower), "keeper2_hsv_lower"); content.Add(new StringContent(keeper2Upper), "keeper2_hsv_upper");

        content.Add(new StringContent(roiTopPct.ToString(CultureInfo.InvariantCulture)), "roi_top_pct");
        content.Add(new StringContent(roiBottomPct.ToString(CultureInfo.InvariantCulture)), "roi_bottom_pct");
        content.Add(new StringContent(minPlayerHeightPct.ToString(CultureInfo.InvariantCulture)), "min_player_height_pct");

        var response = await _httpClient.PostAsync($"{BaseUrl}/cut-raw", content);
        if (response.IsSuccessStatusCode) return JsonSerializer.Deserialize<ClipResult>(await response.Content.ReadAsStringAsync(), JsonOptions);
        throw new Exception($"HTTP Hiba: {response.StatusCode}");
    }

    public async Task<string?> DetectGoalkeepersAsync(string filePath, double startSec, double endSec, System.Threading.CancellationToken cancellationToken = default)
    {
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent(filePath), "video_path");
        content.Add(new StringContent(startSec.ToString(CultureInfo.InvariantCulture)), "start_sec");
        content.Add(new StringContent(endSec.ToString(CultureInfo.InvariantCulture)), "end_sec");
        var response = await _httpClient.PostAsync($"{BaseUrl}/detect-goalkeepers", content, cancellationToken);
        if (response.IsSuccessStatusCode) return await response.Content.ReadAsStringAsync();
        return null;
    }

    public async Task<ClipResult?> BurnDrawingsAsync(string clipPath, double freezeTimestampSec, double freezeDurationSec, string drawingsJson)
    {
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent(clipPath), "clip_path");
        content.Add(new StringContent(freezeTimestampSec.ToString(CultureInfo.InvariantCulture)), "freeze_timestamp_sec");
        content.Add(new StringContent(freezeDurationSec.ToString(CultureInfo.InvariantCulture)), "freeze_duration_sec");
        content.Add(new StringContent(drawingsJson), "drawings_json");
        var response = await _httpClient.PostAsync($"{BaseUrl}/burn-drawings", content);
        if (response.IsSuccessStatusCode) return JsonSerializer.Deserialize<ClipResult>(await response.Content.ReadAsStringAsync(), JsonOptions);
        throw new Exception($"HTTP Hiba: {response.StatusCode}");
    }
}
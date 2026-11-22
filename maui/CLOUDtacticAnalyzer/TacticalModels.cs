using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TacticalAnalyzer.Services;

#region Kapus Elemzés & Metrikák

public class GoalkeeperMetricsDto
{
    [JsonPropertyName("reaction_speed_mps")]
    public double ReactionSpeedMps { get; set; }

    [JsonPropertyName("positioning_score_percent")]
    public double PositioningScorePercent { get; set; }

    [JsonPropertyName("successful_distributions")]
    public int SuccessfulDistributions { get; set; }

    [JsonPropertyName("total_distributions")]
    public int TotalDistributions { get; set; }

    [JsonPropertyName("total_saves")]
    public int TotalSaves { get; set; }
}

public class GoalkeeperClipDto
{
    [JsonPropertyName("clip_id")]
    public string ClipId { get; set; } = string.Empty;

    [JsonPropertyName("tag")]
    public string Tag { get; set; } = string.Empty;

    [JsonPropertyName("start_time")]
    public double StartTime { get; set; }

    [JsonPropertyName("end_time")]
    public double EndTime { get; set; }

    [JsonPropertyName("clip_url")]
    public string ClipUrl { get; set; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("dist_from_goal_m")]
    public double DistFromGoalM { get; set; }

    [JsonPropertyName("bisector_offset_m")]
    public double BisectorOffsetM { get; set; }

    [JsonPropertyName("ball_speed_kmh")]
    public double BallSpeedKmh { get; set; }
}

public class GoalkeeperAnalysisResponse
{
    [JsonPropertyName("total_frames")]
    public int TotalFrames { get; set; }

    [JsonPropertyName("goalkeeper_track_ids")]
    public List<int> GoalkeeperTrackIds { get; set; } = new();

    [JsonPropertyName("metrics")]
    public GoalkeeperMetricsDto? Metrics { get; set; }

    [JsonPropertyName("clips")]
    public List<GoalkeeperClipDto> Clips { get; set; } = new();
}

#endregion

#region Taktikai / AI Detektálás és Követés DTO-k

public class DetectionItem
{
    [JsonPropertyName("track_id")]
    public int? TrackId { get; set; }

    [JsonPropertyName("class")]
    public string Class { get; set; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("team_id")]
    public int TeamId { get; set; }

    [JsonPropertyName("team_color_rgb")]
    public List<int>? TeamColorRgb { get; set; }

    [JsonPropertyName("screen_pos")]
    public List<double>? ScreenPos { get; set; }

    [JsonPropertyName("tactical_pos")]
    public List<double>? TacticalPos { get; set; }
}

public class AnalysisData
{
    [JsonPropertyName("total_objects")]
    public int TotalObjects { get; set; }

    [JsonPropertyName("detections")]
    public List<DetectionItem> Detections { get; set; } = new();
}

public class ImageAnalysisResponse
{
    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("analysis")]
    public AnalysisData Analysis { get; set; } = new();
}

public class VideoFrameResult
{
    [JsonPropertyName("frame_index")]
    public int FrameIndex { get; set; }

    [JsonPropertyName("timestamp")]
    public double Timestamp { get; set; }

    [JsonPropertyName("detections")]
    public List<DetectionItem> Detections { get; set; } = new();
}

public class VideoAnalysisResponse
{
    [JsonPropertyName("total_frames")]
    public int TotalFrames { get; set; }

    [JsonPropertyName("fps")]
    public double Fps { get; set; }

    [JsonPropertyName("frames")]
    public List<VideoFrameResult> Frames { get; set; } = new();
}

#endregion

#region Meccselemzés és Rajzolt Taktikai DTO-k

public class Point2D
{
    [JsonPropertyName("x")]
    public float X { get; set; }

    [JsonPropertyName("y")]
    public float Y { get; set; }

    public Point2D() { }

    public Point2D(float x, float y)
    {
        X = x;
        Y = y;
    }
}

public class DrawingElementDto
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = Guid.NewGuid().ToString();

    [JsonPropertyName("type")]
    public string Type { get; set; } = "ARROW";

    [JsonPropertyName("points")]
    public List<Point2D> Points { get; set; } = new();

    [JsonPropertyName("color")]
    public string Color { get; set; } = "#FFFF00";
}

public class PlayerDto
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("number")]
    public string Number { get; set; } = string.Empty;

    [JsonPropertyName("team")]
    public string Team { get; set; } = "HOME";

    [JsonPropertyName("position")]
    public Point2D Position { get; set; } = new();
}

public class TacticalFrameDto
{
    [JsonPropertyName("frame_id")]
    public string FrameId { get; set; } = Guid.NewGuid().ToString();

    [JsonPropertyName("timestamp_seconds")]
    public double TimestampSeconds { get; set; }

    [JsonPropertyName("players")]
    public List<PlayerDto> Players { get; set; } = new();

    [JsonPropertyName("drawings")]
    public List<DrawingElementDto> Drawings { get; set; } = new();
}

public class MatchAnalysisDto
{
    [JsonPropertyName("match_id")]
    public string MatchId { get; set; } = string.Empty;

    [JsonPropertyName("video_url")]
    public string VideoUrl { get; set; } = string.Empty;

    [JsonPropertyName("frames")]
    public List<TacticalFrameDto> Frames { get; set; } = new();
}

#endregion
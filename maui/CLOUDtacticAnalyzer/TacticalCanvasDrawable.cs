using Microsoft.Maui.Graphics;
using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

public class TacticalCanvasDrawable : IDrawable
{
    public List<DetectionItem>? Detections { get; set; }
    private readonly Dictionary<int, List<PointF>> _trajectories = new();

    public void ClearTrajectories()
    {
        _trajectories.Clear();
    }

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        // 1. Zöld gyep háttér
        canvas.FillColor = Color.FromArgb("#1B5E20");
        canvas.FillRectangle(dirtyRect);

        // 2. Fehér pályavonalak
        canvas.StrokeColor = Colors.White;
        canvas.StrokeSize = 2;

        float margin = 12;
        float width = dirtyRect.Width - (margin * 2);
        float height = dirtyRect.Height - (margin * 2);

        // Pálya kerete
        canvas.DrawRectangle(margin, margin, width, height);

        // Felezővonal és Felezőkör
        float midX = margin + (width / 2);
        float midY = margin + (height / 2);
        canvas.DrawLine(midX, margin, midX, margin + height);
        canvas.DrawCircle(midX, midY, 26);

        // Büntetőterületek (16-osok)
        float penaltyWidth = width * 0.16f;
        float penaltyHeight = height * 0.55f;
        float penaltyY = margin + (height - penaltyHeight) / 2;

        canvas.DrawRectangle(margin, penaltyY, penaltyWidth, penaltyHeight);
        canvas.DrawRectangle(margin + width - penaltyWidth, penaltyY, penaltyWidth, penaltyHeight);

        // Kapuk
        float goalWidth = width * 0.04f;
        float goalHeight = height * 0.22f;
        float goalY = margin + (height - goalHeight) / 2;
        canvas.DrawRectangle(margin - goalWidth, goalY, goalWidth, goalHeight);
        canvas.DrawRectangle(margin + width, goalY, goalWidth, goalHeight);

        if (Detections == null) return;

        // 3. Játékosok és Labda rajzolása
        foreach (var det in Detections)
        {
            float normX, normY;

            if (det.TacticalPos != null && det.TacticalPos.Count == 2)
            {
                normX = Math.Clamp((float)(det.TacticalPos[0] / 1050.0), 0.02f, 0.98f);
                normY = Math.Clamp((float)(det.TacticalPos[1] / 680.0), 0.02f, 0.98f);
            }
            else if (det.ScreenPos != null && det.ScreenPos.Count == 2)
            {
                normX = Math.Clamp((float)(det.ScreenPos[0] / 1920.0), 0.02f, 0.98f);
                normY = Math.Clamp((float)(det.ScreenPos[1] / 1080.0), 0.02f, 0.98f);
            }
            else
            {
                continue;
            }

            float px = margin + (normX * width);
            float py = margin + (normY * height);

            // Nyomvonal rajzolása ugrásszűréssel
            if (det.TrackId.HasValue)
            {
                int tid = det.TrackId.Value;
                if (!_trajectories.ContainsKey(tid))
                    _trajectories[tid] = new List<PointF>();

                var pts = _trajectories[tid];

                // Ha az új pont túl messze van az előzőtől (kamera pan / tracker hiba), reseteljük a nyomvonalat
                if (pts.Count > 0)
                {
                    var lastPt = pts[^1];
                    float dist = MathF.Sqrt(MathF.Pow(px - lastPt.X, 2) + MathF.Pow(py - lastPt.Y, 2));
                    if (dist > 45.0f)
                    {
                        pts.Clear();
                    }
                }

                pts.Add(new PointF(px, py));
                if (pts.Count > 8)
                    pts.RemoveAt(0);

                if (pts.Count > 1)
                {
                    canvas.StrokeColor = det.TeamId == 1
                        ? Color.FromRgba(255, 90, 90, 160)
                        : Color.FromRgba(90, 170, 255, 160);
                    canvas.StrokeSize = 2.0f;
                    for (int i = 1; i < pts.Count; i++)
                    {
                        canvas.DrawLine(pts[i - 1].X, pts[i - 1].Y, pts[i].X, pts[i].Y);
                    }
                }
            }

            // Labda
            if (det.Class == "sports ball")
            {
                canvas.FillColor = Colors.Yellow;
                canvas.FillCircle(px, py, 4.5f);
                canvas.StrokeColor = Colors.Black;
                canvas.StrokeSize = 1;
                canvas.DrawCircle(px, py, 4.5f);
            }
            else // Játékosok
            {
                canvas.FillColor = det.TeamId == 1 ? Color.FromArgb("#D32F2F") : Color.FromArgb("#1976D2");
                canvas.FillCircle(px, py, 6.0f);

                canvas.StrokeColor = Colors.White;
                canvas.StrokeSize = 1.2f;
                canvas.DrawCircle(px, py, 6.0f);

                if (det.TrackId.HasValue)
                {
                    canvas.FontColor = Colors.White;
                    canvas.FontSize = 8;
                    canvas.DrawString($"#{det.TrackId.Value}", px - 12, py - 7, 24, 10, HorizontalAlignment.Center, VerticalAlignment.Center);
                }
            }
        }
    }
}
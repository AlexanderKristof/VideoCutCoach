using Microsoft.Maui.Graphics;
using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

public class PitchDrawable : IDrawable
{
    public TacticalFrameDto? CurrentFrame { get; set; }

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        canvas.FillColor = Color.FromArgb("#2E7D32");
        canvas.FillRectangle(dirtyRect);

        canvas.StrokeColor = Colors.White;
        canvas.StrokeSize = 2;

        float margin = 10;
        float width = dirtyRect.Width - (margin * 2);
        float height = dirtyRect.Height - (margin * 2);

        // Külső pályaív
        canvas.DrawRectangle(margin, margin, width, height);

        // Felezővonal & Középkör
        float midX = dirtyRect.Width / 2;
        canvas.DrawLine(midX, margin, midX, margin + height);
        canvas.DrawCircle(midX, dirtyRect.Height / 2, 30);

        if (CurrentFrame == null) return;

        // Rajzolt elemek (nyilak, sávok) kirajzolása
        if (CurrentFrame.Drawings != null)
        {
            foreach (var drawing in CurrentFrame.Drawings)
            {
                if (drawing.Points == null || drawing.Points.Count < 2) continue;

                canvas.StrokeColor = Color.FromArgb(drawing.Color ?? "#FFFF00");
                canvas.StrokeSize = 3;

                for (int i = 1; i < drawing.Points.Count; i++)
                {
                    float x1 = margin + (drawing.Points[i - 1].X * width);
                    float y1 = margin + (drawing.Points[i - 1].Y * height);
                    float x2 = margin + (drawing.Points[i].X * width);
                    float y2 = margin + (drawing.Points[i].Y * height);

                    canvas.DrawLine(x1, y1, x2, y2);
                }
            }
        }

        // Játékosok és labda kirajzolása
        if (CurrentFrame.Players != null)
        {
            foreach (var player in CurrentFrame.Players)
            {
                if (player.Position == null) continue;

                float px = margin + (player.Position.X * width);
                float py = margin + (player.Position.Y * height);

                if (player.Team == "BALL")
                {
                    canvas.FillColor = Colors.Yellow;
                    canvas.FillCircle(px, py, 5);
                    canvas.StrokeColor = Colors.Black;
                    canvas.StrokeSize = 1;
                    canvas.DrawCircle(px, py, 5);
                }
                else
                {
                    canvas.FillColor = player.Team == "HOME" ? Colors.Red : Colors.Blue;
                    canvas.FillCircle(px, py, 8);

                    canvas.StrokeColor = Colors.Black;
                    canvas.StrokeSize = 1;
                    canvas.DrawCircle(px, py, 8);

                    if (!string.IsNullOrEmpty(player.Number))
                    {
                        canvas.FontColor = Colors.White;
                        canvas.FontSize = 10;
                        canvas.DrawString(player.Number, px - 6, py + 4, HorizontalAlignment.Center);
                    }
                }
            }
        }
    }
}
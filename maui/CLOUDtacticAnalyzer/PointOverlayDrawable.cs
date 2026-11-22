using Microsoft.Maui.Graphics;

namespace TacticalAnalyzer;

public class PointOverlayDrawable : IDrawable
{
    public List<PointF> Points { get; } = new();

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        if (Points.Count == 0) return;

        canvas.StrokeSize = 2;
        canvas.StrokeColor = Colors.Cyan;
        canvas.FillColor = Colors.OrangeRed;

        for (int i = 0; i < Points.Count; i++)
        {
            var p = Points[i];
            canvas.FillCircle(p.X, p.Y, 7);
            canvas.DrawCircle(p.X, p.Y, 7);

            canvas.FontColor = Colors.White;
            canvas.FontSize = 14;
            canvas.DrawString((i + 1).ToString(), p.X + 10, p.Y - 10, HorizontalAlignment.Left);
        }

        if (Points.Count > 1)
        {
            for (int i = 0; i < Points.Count - 1; i++)
            {
                canvas.DrawLine(Points[i], Points[i + 1]);
            }

            if (Points.Count == 4)
            {
                canvas.DrawLine(Points[3], Points[0]);
            }
        }
    }
}
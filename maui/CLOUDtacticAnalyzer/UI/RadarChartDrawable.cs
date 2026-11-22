using Microsoft.Maui.Graphics;
using System;
using System.Collections.Generic;

namespace TacticalAnalyzer.UI
{
    public class RadarChartDrawable : IDrawable
    {
        public List<string> Categories { get; set; } = new()
        { "Gyorsaság", "Lövések", "Passzok", "Védekezés", "Állóképesség", "Cselezés" };

        public List<double> PlayerStats { get; set; } = new()
        { 85, 60, 92, 45, 78, 88 };

        public void Draw(ICanvas canvas, RectF dirtyRect)
        {
            if (Categories.Count != PlayerStats.Count || Categories.Count < 3) return;

            float centerX = dirtyRect.Center.X;
            float centerY = dirtyRect.Center.Y;
            float maxRadius = Math.Min(dirtyRect.Width, dirtyRect.Height) / 2.5f;
            int numPoints = Categories.Count;

            // 1. Háttér háló (Pókháló)
            canvas.StrokeColor = Color.FromArgb("#333333");
            canvas.StrokeSize = 1;

            int webLevels = 5;
            for (int level = 1; level <= webLevels; level++)
            {
                float radius = maxRadius * (level / (float)webLevels);
                PathF webPath = new PathF();

                for (int i = 0; i < numPoints; i++)
                {
                    float angle = (float)(i * 2 * Math.PI / numPoints - Math.PI / 2);
                    float x = centerX + radius * (float)Math.Cos(angle);
                    float y = centerY + radius * (float)Math.Sin(angle);

                    if (i == 0) webPath.MoveTo(x, y);
                    else webPath.LineTo(x, y);

                    if (level == webLevels)
                        canvas.DrawLine(centerX, centerY, x, y);
                }
                webPath.Close();
                canvas.DrawPath(webPath);
            }

            // 2. Játékos adatainak rajzolása
            PathF dataPath = new PathF();
            for (int i = 0; i < numPoints; i++)
            {
                float statRatio = (float)(PlayerStats[i] / 100.0);
                float radius = maxRadius * statRatio;

                float angle = (float)(i * 2 * Math.PI / numPoints - Math.PI / 2);
                float x = centerX + radius * (float)Math.Cos(angle);
                float y = centerY + radius * (float)Math.Sin(angle);

                if (i == 0) dataPath.MoveTo(x, y);
                else dataPath.LineTo(x, y);

                canvas.FillColor = Color.FromArgb("#FF3B30");
                canvas.FillCircle(x, y, 4);
            }
            dataPath.Close();

            canvas.FillColor = Color.FromArgb("#80FF3B30"); // 50% átlátszó
            canvas.FillPath(dataPath);

            canvas.StrokeColor = Color.FromArgb("#FF3B30");
            canvas.StrokeSize = 2;
            canvas.DrawPath(dataPath);

            // 3. Feliratok
            canvas.FontColor = Colors.White;
            canvas.FontSize = 12;
            for (int i = 0; i < numPoints; i++)
            {
                float angle = (float)(i * 2 * Math.PI / numPoints - Math.PI / 2);
                float labelRadius = maxRadius + 20;
                float x = centerX + labelRadius * (float)Math.Cos(angle);
                float y = centerY + labelRadius * (float)Math.Sin(angle);

                HorizontalAlignment align = HorizontalAlignment.Center;
                if (Math.Cos(angle) > 0.1) align = HorizontalAlignment.Left;
                else if (Math.Cos(angle) < -0.1) align = HorizontalAlignment.Right;

                canvas.DrawString(Categories[i], x - 50, y - 10, 100, 20, align, VerticalAlignment.Center);
            }
        }
    }
}
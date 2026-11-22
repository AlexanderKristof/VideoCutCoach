using Microsoft.Maui.Graphics;
using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

public class PitchTouchHandler
{
    private readonly PitchDrawable _drawable;
    private readonly GraphicsView _graphicsView;
    private DrawingElementDto? _currentDrawing;

    public PitchTouchHandler(PitchDrawable drawable, GraphicsView graphicsView)
    {
        _drawable = drawable;
        _graphicsView = graphicsView;
    }

    public void OnTouchStart(PointF touchPoint, RectF bounds)
    {
        if (_drawable.CurrentFrame == null)
        {
            _drawable.CurrentFrame = new TacticalFrameDto();
        }

        float normX = Math.Clamp((touchPoint.X - 10) / (bounds.Width - 20), 0f, 1f);
        float normY = Math.Clamp((touchPoint.Y - 10) / (bounds.Height - 20), 0f, 1f);

        _currentDrawing = new DrawingElementDto
        {
            Type = "ARROW",
            Color = "#FFFF00",
            Points = new List<Point2D> { new Point2D(normX, normY) }
        };

        _drawable.CurrentFrame.Drawings.Add(_currentDrawing);
        _graphicsView.Invalidate();
    }

    public void OnTouchMove(PointF touchPoint, RectF bounds)
    {
        if (_currentDrawing == null) return;

        float normX = Math.Clamp((touchPoint.X - 10) / (bounds.Width - 20), 0f, 1f);
        float normY = Math.Clamp((touchPoint.Y - 10) / (bounds.Height - 20), 0f, 1f);

        _currentDrawing.Points.Add(new Point2D(normX, normY));
        _graphicsView.Invalidate();
    }

    public void OnTouchEnd()
    {
        _currentDrawing = null;
        _graphicsView.Invalidate();
    }

    public void ClearDrawings()
    {
        if (_drawable.CurrentFrame?.Drawings != null)
        {
            _drawable.CurrentFrame.Drawings.Clear();
            _graphicsView.Invalidate();
        }
    }
}
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Maui.Views;
using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Storage;
using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

public enum DrawingTool { Line, Circle, Arrow }

public class CoachDrawingItem
{
    public DrawingTool Tool { get; set; }
    public PointF StartPoint { get; set; }
    public PointF EndPoint { get; set; }
}

public class InteractiveCoachDrawable : IDrawable
{
    public List<CoachDrawingItem> Items { get; } = new();
    public CoachDrawingItem? CurrentItem { get; set; }

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        foreach (var item in Items) DrawItem(canvas, item);
        if (CurrentItem != null) DrawItem(canvas, CurrentItem);
    }

    private void DrawItem(ICanvas canvas, CoachDrawingItem item)
    {
        switch (item.Tool)
        {
            case DrawingTool.Line:
                canvas.StrokeColor = Color.FromArgb("#FFC107"); canvas.StrokeSize = 3;
                canvas.DrawLine(item.StartPoint.X, item.StartPoint.Y, item.EndPoint.X, item.EndPoint.Y); break;
            case DrawingTool.Circle:
                canvas.StrokeColor = Color.FromArgb("#E91E63"); canvas.StrokeSize = 3;
                float r = MathF.Sqrt(MathF.Pow(item.EndPoint.X - item.StartPoint.X, 2) + MathF.Pow(item.EndPoint.Y - item.StartPoint.Y, 2));
                canvas.DrawCircle(item.StartPoint.X, item.StartPoint.Y, r); break;
            case DrawingTool.Arrow:
                canvas.StrokeColor = Color.FromArgb("#4CAF50"); canvas.StrokeSize = 3;
                canvas.DrawLine(item.StartPoint.X, item.StartPoint.Y, item.EndPoint.X, item.EndPoint.Y);
                canvas.FillColor = Color.FromArgb("#4CAF50"); canvas.FillCircle(item.EndPoint.X, item.EndPoint.Y, 6); break;
        }
    }
}

public partial class GoalkeeperPage : ContentPage
{
    private readonly AnalysisService _analysisService;
    private readonly InteractiveCoachDrawable _drawable;
    private FileResult? _selectedVideoFile;
    private bool _isDrawingMode = false;
    private DrawingTool _currentTool = DrawingTool.Line;

    private CancellationTokenSource? _aiCts;
    private string? _currentClipLocalPath;

    public GoalkeeperPage()
    {
        InitializeComponent();
        _analysisService = new AnalysisService();
        _drawable = new InteractiveCoachDrawable();
        DrawingCanvasView.Drawable = _drawable;
    }

    private static double ParseTime(string input)
    {
        if (string.IsNullOrWhiteSpace(input)) return 0.0;
        input = input.Trim().Replace(',', '.');
        if (input.Contains(':'))
        {
            var parts = input.Split(':');
            if (parts.Length == 2 && double.TryParse(parts[0], NumberStyles.Any, CultureInfo.InvariantCulture, out double m) && double.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out double s))
                return (m * 60.0) + s;
        }
        else if (double.TryParse(input, NumberStyles.Any, CultureInfo.InvariantCulture, out double rawSec)) return rawSec;
        return 0.0;
    }

    private static string SecondsToMinuteString(double totalSeconds)
    {
        if (totalSeconds < 0) totalSeconds = 0;
        int minutes = (int)(totalSeconds / 60);
        int seconds = (int)(totalSeconds % 60);
        return $"{minutes}:{seconds:D2}";
    }

    private void SafeDisplayAlert(string title, string message)
    {
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            await DisplayAlertAsync(title, message, "Rendben");
        });
    }

    private void SetLoading(bool isLoading, string status)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            GkLoadingIndicator.IsRunning = isLoading;
            GkLoadingIndicator.IsVisible = isLoading;
            if (!string.IsNullOrEmpty(status)) StatusLabel.Text = status;
        });
    }

    public void OnPickVideoClicked(object? sender, EventArgs e)
    {
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            try
            {
                var result = await FilePicker.Default.PickAsync(new PickOptions { PickerTitle = "Válassz videót", FileTypes = FilePickerFileType.Videos });
                if (result != null && !string.IsNullOrEmpty(result.FullPath))
                {
                    _selectedVideoFile = result;
                    SelectedVideoLabel.Text = $"Kiválasztva: {result.FileName}";
                    ManualTimeContainer.IsVisible = true;
                }
            }
            catch (Exception ex) { SafeDisplayAlert("Hiba", ex.Message); }
        });
    }

    private void PlayVideoSafely(string videoUrl)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            ClipVideoPlayer.Stop();
            ClipVideoPlayer.Source = null;
            ClipVideoPlayer.Source = MediaSource.FromUri(videoUrl);
            ShowEditor();
            ClipVideoPlayer.Play();
        });
    }

    public async void OnLoadClipClicked(object? sender, EventArgs e)
    {
        if (_selectedVideoFile == null) { SafeDisplayAlert("Figyelem", "Válassz videót!"); return; }
        try
        {
            double s = ParseTime(StartTimeEntry.Text);
            double en = ParseTime(EndTimeEntry.Text);
            if (en <= s) en = s + 120.0;

            SetLoading(true, $"Kézi vágás folyamatban (Fejlesztői mód: {(DevModeSwitch.IsToggled ? "BE" : "KI")})...");
            var res = await _analysisService.CutRawVideoAsync(_selectedVideoFile.FullPath, s, en, DevModeSwitch.IsToggled, "gk");
            if (res != null && !string.IsNullOrEmpty(res.ClipUrl))
            {
                _currentClipLocalPath = res.LocalPath;
                string fullUrl = $"{AnalysisService.BaseUrl}{res.ClipUrl}";
                PlayVideoSafely(fullUrl);
                SetLoading(false, "Kész! " + fullUrl);
            }
        }
        catch (Exception ex)
        {
            SetLoading(false, "Hiba történt!");
            SafeDisplayAlert("Hiba a szerverről", ex.Message);
        }
    }

    public async void OnAiAnalyzeClicked(object? sender, EventArgs e)
    {
        if (_selectedVideoFile == null) { SafeDisplayAlert("Figyelem", "Válassz videót!"); return; }
        _aiCts = new CancellationTokenSource();

        try
        {
            double userStart = ParseTime(StartTimeEntry.Text);
            double userEnd = ParseTime(EndTimeEntry.Text);
            if (userEnd <= userStart) userEnd = userStart + 120.0;

            SetLoading(true, $"Elemzés a megadott sávban ({StartTimeEntry.Text} - {EndTimeEntry.Text})...");

            MainThread.BeginInvokeOnMainThread(() =>
            {
                AiAnalyzeButton.IsVisible = false;
                CancelAiButton.IsVisible = true;
                ManualTimeContainer.IsVisible = true;
            });

            string? json = await _analysisService.DetectGoalkeepersAsync(_selectedVideoFile.FullPath, userStart, userEnd, _aiCts.Token);
            if (!string.IsNullOrEmpty(json))
            {
                using var doc = JsonDocument.Parse(json);
                var gkArray = doc.RootElement.GetProperty("goalkeepers");
                List<string> gkNames = new(); List<Tuple<double, double>> timeWindows = new();

                foreach (var gk in gkArray.EnumerateArray())
                {
                    gkNames.Add(gk.GetProperty("name").GetString() ?? "Kapus");
                    var evArr = gk.GetProperty("events");
                    if (evArr.GetArrayLength() > 0) timeWindows.Add(new Tuple<double, double>(evArr[0].GetProperty("start_sec").GetDouble(), evArr[0].GetProperty("end_sec").GetDouble()));
                    else timeWindows.Add(new Tuple<double, double>(userStart, userEnd));
                }

                string sel = "";
                await MainThread.InvokeOnMainThreadAsync(async () =>
                {
                    sel = await DisplayActionSheetAsync("Válaszd ki a kapust:", "Mégse", null, gkNames.ToArray());
                });

                if (!string.IsNullOrEmpty(sel) && sel != "Mégse")
                {
                    int idx = gkNames.IndexOf(sel);
                    double s = timeWindows[idx].Item1; double en = timeWindows[idx].Item2;

                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        StartTimeEntry.Text = SecondsToMinuteString(s);
                        EndTimeEntry.Text = SecondsToMinuteString(en);
                    });

                    SetLoading(true, $"AI Jelenet vágása (Fejlesztői mód: {(DevModeSwitch.IsToggled ? "BE" : "KI")})...");
                    var res = await _analysisService.CutRawVideoAsync(_selectedVideoFile.FullPath, s, en, DevModeSwitch.IsToggled, "gk");

                    if (res != null && !string.IsNullOrEmpty(res.ClipUrl))
                    {
                        _currentClipLocalPath = res.LocalPath;
                        string fullUrl = $"{AnalysisService.BaseUrl}{res.ClipUrl}";
                        PlayVideoSafely(fullUrl);
                        SetLoading(false, $"✅ Jelenet aktív: {sel}");
                    }
                }
                else
                {
                    SetLoading(false, "Vágás megszakítva.");
                }
            }
        }
        catch (TaskCanceledException) { SetLoading(false, "Vágás megszakítva."); }
        catch (Exception ex)
        {
            SetLoading(false, "Hiba történt!");
            SafeDisplayAlert("Hiba a szerverről", ex.Message);
        }
        finally
        {
            MainThread.BeginInvokeOnMainThread(() =>
            {
                AiAnalyzeButton.IsVisible = true;
                CancelAiButton.IsVisible = false;
            });
        }
    }

    public void OnCancelAiClicked(object? sender, EventArgs e) { _aiCts?.Cancel(); }

    private void ShowEditor()
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            PlayerBorder.IsVisible = true;
            ControlsBorder.IsVisible = true;
            IntegrateBorder.IsVisible = true;
        });
    }

    private void OnToggleDrawingClicked(object? sender, EventArgs e)
    {
        _isDrawingMode = !_isDrawingMode;
        DrawingCanvasView.InputTransparent = !_isDrawingMode;
        ToggleDrawingButton.Text = _isDrawingMode ? "Rajzolás: BEKAPCSOLVA" : "KIKAPCSOLVA";
        ToggleDrawingButton.BackgroundColor = _isDrawingMode ? Color.FromArgb("#4CAF50") : Color.FromArgb("#424242");
        BtnLine.IsEnabled = BtnCircle.IsEnabled = BtnArrow.IsEnabled = _isDrawingMode;
        if (_isDrawingMode) ClipVideoPlayer.Pause();
    }

    public async void OnIntegrateAndExportClicked(object? sender, EventArgs e)
    {
        if (string.IsNullOrEmpty(_currentClipLocalPath)) return;
        try
        {
            SetLoading(true, "Beégetés...");
            double freezeTimestamp = ClipVideoPlayer.Position.TotalSeconds;
            float w = (float)DrawingCanvasView.Width; float h = (float)DrawingCanvasView.Height;
            if (w <= 0) { w = 800; h = 450; }

            var exp = _drawable.Items.Select(it => new { tool = it.Tool.ToString().ToLower(), x1 = it.StartPoint.X / w, y1 = it.StartPoint.Y / h, x2 = it.EndPoint.X / w, y2 = it.EndPoint.Y / h }).ToList();

            var res = await _analysisService.BurnDrawingsAsync(_currentClipLocalPath, freezeTimestamp, 3.0, JsonSerializer.Serialize(exp));
            if (res != null && !string.IsNullOrEmpty(res.ClipUrl))
            {
                string fullUrl = $"{AnalysisService.BaseUrl}{res.ClipUrl}";
                PlayVideoSafely(fullUrl);

                MainThread.BeginInvokeOnMainThread(() =>
                {
                    _drawable.Items.Clear();
                    DrawingCanvasView.Invalidate();
                });
                SetLoading(false, "Beégetés sikeres!");
                SafeDisplayAlert("Kész", "Videó sikeresen integrálva!");
            }
        }
        catch (Exception ex)
        {
            SetLoading(false, "Hiba a beégetésnél!");
            SafeDisplayAlert("Hiba a szerverről", ex.Message);
        }
    }

    #region Rajzolás
    private void OnDrawingStart(object? sender, TouchEventArgs e) { if (!_isDrawingMode || e.Touches == null || !e.Touches.Any()) return; var pt = new PointF((float)e.Touches.First().X, (float)e.Touches.First().Y); _drawable.CurrentItem = new CoachDrawingItem { Tool = _currentTool, StartPoint = pt, EndPoint = pt }; DrawingCanvasView.Invalidate(); }
    private void OnDrawingDrag(object? sender, TouchEventArgs e) { if (!_isDrawingMode || e.Touches == null || !e.Touches.Any() || _drawable.CurrentItem == null) return; _drawable.CurrentItem.EndPoint = new PointF((float)e.Touches.First().X, (float)e.Touches.First().Y); DrawingCanvasView.Invalidate(); }
    private void OnDrawingEnd(object? sender, TouchEventArgs e) { if (!_isDrawingMode || _drawable.CurrentItem == null) return; _drawable.Items.Add(_drawable.CurrentItem); _drawable.CurrentItem = null; DrawingCanvasView.Invalidate(); }
    private void OnToolLineClicked(object? sender, EventArgs e) { _currentTool = DrawingTool.Line; HighlightActiveTool(BtnLine); }
    private void OnToolCircleClicked(object? sender, EventArgs e) { _currentTool = DrawingTool.Circle; HighlightActiveTool(BtnCircle); }
    private void OnToolArrowClicked(object? sender, EventArgs e) { _currentTool = DrawingTool.Arrow; HighlightActiveTool(BtnArrow); }
    private void HighlightActiveTool(Button active) { BtnLine.BackgroundColor = BtnCircle.BackgroundColor = BtnArrow.BackgroundColor = Color.FromArgb("#333333"); active.BackgroundColor = Color.FromArgb("#FFC107"); }
    private void OnClearDrawingsClicked(object? sender, EventArgs e) { _drawable.Items.Clear(); _drawable.CurrentItem = null; DrawingCanvasView.Invalidate(); }
    #endregion
}
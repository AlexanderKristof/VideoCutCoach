using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Controls.Shapes;
using Microsoft.Maui.Layouts;
using Microsoft.Maui.Storage;
using TacticalAnalyzer.Services;
using System.Linq;

namespace TacticalAnalyzer;

public partial class MainPage : ContentPage
{
    private readonly AnalysisService _analysisService;
    private readonly MatchService _matchService = new();
    private string? _loadedVideoPath;
    private string? _currentFrameUrl;
    private string _currentCalibTarget = "";
    private ClipResult? _lastClipResult;

    private Dictionary<string, int> _currentCounts = new() {
        { "grass", 0 }, { "ref", 0 }, { "team1", 0 }, { "team2", 0 }, { "keeper1", 0 }, { "keeper2", 0 }, { "pitch", 0 }
    };

    private Dictionary<string, List<View>> _placedMarkers = new() {
        { "grass", new() }, { "ref", new() }, { "team1", new() }, { "team2", new() }, { "keeper1", new() }, { "keeper2", new() }
    };

    // Pálya-sarok kalibráció (homográfiához) — normalizált (0..1) képkoordináták, óramutató járása szerint
    private readonly List<PointF> _pitchCorners = new();
    private readonly PointOverlayDrawable _pitchCornerDrawable = new();

    private string _grassLower = "", _grassUpper = "";
    private string _refLower = "", _refUpper = "";
    private string _team1Lower = "", _team1Upper = "";
    private string _team2Lower = "", _team2Upper = "";
    private string _keeper1Lower = "", _keeper1Upper = "";
    private string _keeper2Lower = "", _keeper2Upper = "";

    public MainPage()
    {
        InitializeComponent();
        _analysisService = new AnalysisService();
        PitchCornerCanvas.Drawable = _pitchCornerDrawable;
    }

    private async void OnLoadVideoClicked(object? sender, EventArgs e)
    {
        var result = await FilePicker.Default.PickAsync(new PickOptions { PickerTitle = "Válassz videót", FileTypes = FilePickerFileType.Videos });
        if (result != null) { _loadedVideoPath = result.FullPath; SelectedFileLabel.Text = result.FileName; }
    }

    private async void OnLoadFrameClicked(object? sender, EventArgs e)
    {
        if (string.IsNullOrEmpty(_loadedVideoPath)) return;
        WizardInstructionLabel.Text = "⏳ Képkocka betöltése...";
        _currentFrameUrl = await _analysisService.GetFrameAtTimeAsync(_loadedVideoPath, ParseTimeToSeconds(CalibTimeEntry.Text ?? "0"));

        if (!string.IsNullOrEmpty(_currentFrameUrl))
        {
            DynamicCalibImage.Source = ImageSource.FromUri(new Uri(_currentFrameUrl));

            var boxes = ImageOverlayContainer.Children.Where(c => c is Rectangle).ToList();
            foreach (var b in boxes) ImageOverlayContainer.Children.Remove(b);

            foreach (var key in _currentCounts.Keys.ToList()) _currentCounts[key] = 0;
            foreach (var key in _placedMarkers.Keys) _placedMarkers[key].Clear();
            _pitchCorners.Clear();
            _pitchCornerDrawable.Points.Clear();
            PitchCornerCanvas.Invalidate();
            _currentCalibTarget = "";
            BtnNextStep.IsVisible = false;

            HorizonMaskView.IsVisible = true;
            BenchMaskView.IsVisible = true;
            UpdateMaskOverlays();

            WizardInstructionLabel.Text = "📷 Képkocka betöltve! Kezdd a 'Fű (Pálya)' gomb megnyomásával.";
        }
    }

    private void OnSlidersChanged(object? sender, ValueChangedEventArgs e)
    {
        UpdateMaskOverlays();
    }

    private void UpdateMaskOverlays()
    {
        if (HorizonMaskView != null && BenchMaskView != null)
        {
            AbsoluteLayout.SetLayoutBounds(HorizonMaskView, new Rect(0, 0, 1, HorizonSlider.Value));
            AbsoluteLayout.SetLayoutBounds(BenchMaskView, new Rect(0, 1, 1, BottomSlider.Value));
        }
    }

    private void OnSelectCalibModeClicked(object? sender, EventArgs e)
    {
        if (sender == BtnCalibGrass) _currentCalibTarget = "grass";
        else if (sender == BtnCalibRef) _currentCalibTarget = "ref";
        else if (sender == BtnCalibTeam1) _currentCalibTarget = "team1";
        else if (sender == BtnCalibTeam2) _currentCalibTarget = "team2";
        else if (sender == BtnCalibKeeper1) _currentCalibTarget = "keeper1";
        else if (sender == BtnCalibKeeper2) _currentCalibTarget = "keeper2";
        else if (sender == BtnCalibPitch) _currentCalibTarget = "pitch";

        UpdateWizardLabel();
    }

    private void UpdateWizardLabel()
    {
        if (string.IsNullOrEmpty(_currentCalibTarget)) return;

        string name = _currentCalibTarget switch
        {
            "grass" => "Fű (Pálya)",
            "ref" => "Bíró",
            "team1" => "Csapat 1",
            "team2" => "Csapat 2",
            "keeper1" => "Kapus 1",
            "keeper2" => "Kapus 2",
            "pitch" => "Pálya sarkai",
            _ => ""
        };

        string hint = _currentCalibTarget == "pitch"
            ? " (max 4 pont, óramutató járása szerint: bal-felső, jobb-felső, jobb-alsó, bal-alsó)"
            : "";
        WizardInstructionLabel.Text = $"📍 {name}: Kattints a célpont(ok)ra!{hint} (Eddig rögzítve: {_currentCounts[_currentCalibTarget]} db)";
        ActiveCalibModeLabel.Text = $"Aktuális Mód: {name}";
        BtnNextStep.IsVisible = _currentCounts[_currentCalibTarget] > 0;
    }

    private void OnNextStepClicked(object? sender, EventArgs e)
    {
        if (!string.IsNullOrEmpty(_currentCalibTarget))
        {
            string name = _currentCalibTarget switch
            {
                "grass" => "Fű",
                "ref" => "Bíró",
                "team1" => "Csapat 1",
                "team2" => "Csapat 2",
                "keeper1" => "Kapus 1",
                "keeper2" => "Kapus 2",
                "pitch" => "Pálya sarkai",
                _ => ""
            };

            WizardInstructionLabel.Text = $"✅ {name} lezárva. Válassz másik kategóriát!";
            BtnNextStep.IsVisible = false;
            _currentCalibTarget = "";
            ActiveCalibModeLabel.Text = "Aktuális Mód: Nincs";
        }
    }

    private void OnUndoMarkerClicked(object? sender, EventArgs e)
    {
        if (string.IsNullOrEmpty(_currentCalibTarget) || _currentCounts[_currentCalibTarget] <= 0) return;

        if (_currentCalibTarget == "pitch")
        {
            if (_pitchCorners.Count > 0) _pitchCorners.RemoveAt(_pitchCorners.Count - 1);
            if (_pitchCornerDrawable.Points.Count > 0) _pitchCornerDrawable.Points.RemoveAt(_pitchCornerDrawable.Points.Count - 1);
            PitchCornerCanvas.Invalidate();

            _currentCounts["pitch"]--;
            UpdateWizardLabel();
            StatusLabel.Text = "Utolsó pálya-sarok törölve.";
            return;
        }

        var list = _placedMarkers[_currentCalibTarget];
        var lastView = list.Last();
        ImageOverlayContainer.Children.Remove(lastView);
        list.Remove(lastView);

        _currentCounts[_currentCalibTarget]--;
        UpdateWizardLabel();
        StatusLabel.Text = "Utolsó jelölés törölve.";
    }

    private async void OnCalibrationImageTapped(object? sender, TappedEventArgs e)
    {
        if (string.IsNullOrEmpty(_currentFrameUrl) || string.IsNullOrEmpty(_currentCalibTarget)) return;
        var pos = e.GetPosition(ImageOverlayContainer);
        if (!pos.HasValue) return;

        try
        {
            // A pálya-sarok kalibráció geometriai pont, nincs hozzá szín-AI hívás
            if (_currentCalibTarget == "pitch")
            {
                if (_currentCounts["pitch"] >= 4)
                {
                    StatusLabel.Text = "Már megvan mind a 4 sarokpont. Az 'Utolsó Jelölés Törlése' gombbal tudsz újrajelölni.";
                    return;
                }

                double normX = pos.Value.X / ImageOverlayContainer.Width;
                double normY = pos.Value.Y / ImageOverlayContainer.Height;
                _pitchCorners.Add(new PointF((float)normX, (float)normY));

                _pitchCornerDrawable.Points.Add(new PointF((float)pos.Value.X, (float)pos.Value.Y));
                PitchCornerCanvas.Invalidate();

                _currentCounts["pitch"]++;
                UpdateWizardLabel();
                StatusLabel.Text = _currentCounts["pitch"] == 4
                    ? "✅ Mind a 4 pálya-sarok kijelölve!"
                    : $"Pálya-sarok #{_currentCounts["pitch"]} rögzítve. Jelöld a következőt.";
                return;
            }

            Color mc = _currentCalibTarget switch
            {
                "grass" => Colors.Green,
                "ref" => Colors.Yellow,
                "team1" => Colors.Blue,
                "team2" => Colors.Red,
                "keeper1" => Colors.Orange,
                "keeper2" => Colors.Purple,
                _ => Colors.White
            };

            StatusLabel.Text = "AI elemzi a színt...";
            var result = await _analysisService.AnalyzeClickWithAiAsync(_currentFrameUrl, pos.Value.X, pos.Value.Y, ImageOverlayContainer.Width, ImageOverlayContainer.Height);

            if (result != null)
            {
                if (_currentCalibTarget == "grass") { _grassLower = result.Lower; _grassUpper = result.Upper; }
                else if (_currentCalibTarget == "ref") { _refLower = result.Lower; _refUpper = result.Upper; }
                else if (_currentCalibTarget == "team1") { _team1Lower = result.Lower; _team1Upper = result.Upper; }
                else if (_currentCalibTarget == "team2") { _team2Lower = result.Lower; _team2Upper = result.Upper; }
                else if (_currentCalibTarget == "keeper1") { _keeper1Lower = result.Lower; _keeper1Upper = result.Upper; }
                else if (_currentCalibTarget == "keeper2") { _keeper2Lower = result.Lower; _keeper2Upper = result.Upper; }

                _currentCounts[_currentCalibTarget]++;

                var rect = new Rectangle { Fill = Colors.Transparent, Stroke = mc, StrokeThickness = 3 };
                AbsoluteLayout.SetLayoutBounds(rect, new Rect(result.BoxX, result.BoxY, result.BoxW, result.BoxH));
                ImageOverlayContainer.Children.Add(rect);
                _placedMarkers[_currentCalibTarget].Add(rect);

                UpdateWizardLabel();
                StatusLabel.Text = "Sikeres kalibráció!";
            }
        }
        catch (Exception ex) { System.Diagnostics.Debug.WriteLine($"[HIBA]: {ex.Message}"); }
    }

    private string BuildCalibPointsString()
    {
        // A szerver (main.py) csak akkor számol homográfiát, ha pontosan 4 pontot kap:
        // "rx,ry;rx,ry;rx,ry;rx,ry" formátumban, 0..1 közötti relatív koordinátákkal.
        if (_pitchCorners.Count != 4) return "";
        return string.Join(";", _pitchCorners.Select(p =>
            $"{p.X.ToString(CultureInfo.InvariantCulture)},{p.Y.ToString(CultureInfo.InvariantCulture)}"));
    }

    private async void OnCutSceneClicked(object? sender, EventArgs e)
    {
        if (string.IsNullOrEmpty(_loadedVideoPath)) return;

        if (_pitchCorners.Count > 0 && _pitchCorners.Count < 4)
        {
            bool proceed = await DisplayAlert(
                "Hiányos pálya-kalibráció",
                $"Csak {_pitchCorners.Count}/4 pálya-sarok van kijelölve, ezért a méteres statisztikák (kapus-eltolódás, lövés-sebesség) most kimaradnak. Folytatod kalibráció nélkül?",
                "Igen, folytatás", "Mégse");
            if (!proceed) return;
        }

        ProgressLayout.IsVisible = true;
        AnalysisProgressBar.Progress = 0.05;
        ProgressTextLabel.Text = "Feldolgozás indítása...";
        StatusLabel.Text = "AI elemzés fut...";

        bool isRunning = true;
        var progressTask = Task.Run(async () => {
            double p = 0.05;
            while (isRunning && p < 0.95)
            {
                p += 0.02;
                MainThread.BeginInvokeOnMainThread(() => {
                    AnalysisProgressBar.Progress = p;
                    ProgressTextLabel.Text = $"Feldolgozás alatt: {(int)(p * 100)}%";
                });
                await Task.Delay(500);
            }
        });

        try
        {
            var result = await _analysisService.CutRawVideoAsync(
                _loadedVideoPath, ParseTimeToSeconds(StartTimeEntry.Text ?? "0"), ParseTimeToSeconds(EndTimeEntry.Text ?? "10"),
                devMode: AiSwitch.IsToggled,
                roiTopPct: HorizonSlider.Value,
                roiBottomPct: BottomSlider.Value,
                grassHsvLower: string.IsNullOrEmpty(_grassLower) ? "30,40,40" : _grassLower,
                grassHsvUpper: string.IsNullOrEmpty(_grassUpper) ? "85,255,255" : _grassUpper,
                refHsvLower: string.IsNullOrEmpty(_refLower) ? "0,0,0" : _refLower,
                refHsvUpper: string.IsNullOrEmpty(_refUpper) ? "179,255,255" : _refUpper,
                team1Lower: string.IsNullOrEmpty(_team1Lower) ? "0,0,0" : _team1Lower,
                team1Upper: string.IsNullOrEmpty(_team1Upper) ? "179,255,255" : _team1Upper,
                team2Lower: string.IsNullOrEmpty(_team2Lower) ? "0,0,0" : _team2Lower,
                team2Upper: string.IsNullOrEmpty(_team2Upper) ? "179,255,255" : _team2Upper,
                keeper1Lower: string.IsNullOrEmpty(_keeper1Lower) ? "0,0,0" : _keeper1Lower,
                keeper1Upper: string.IsNullOrEmpty(_keeper1Upper) ? "179,255,255" : _keeper1Upper,
                keeper2Lower: string.IsNullOrEmpty(_keeper2Lower) ? "0,0,0" : _keeper2Lower,
                keeper2Upper: string.IsNullOrEmpty(_keeper2Upper) ? "179,255,255" : _keeper2Upper,
                calibPoints: BuildCalibPointsString()
            );

            isRunning = false;
            AnalysisProgressBar.Progress = 1.0;
            ProgressTextLabel.Text = "Kész! Videó megnyitása...";

            if (result != null && !string.IsNullOrEmpty(result.LocalPath))
            {
                _lastClipResult = result;
                BtnSaveMatch.IsVisible = true;
                SaveMatchStatusLabel.IsVisible = false;
                await Launcher.Default.OpenAsync(new OpenFileRequest("Elemzés", new ReadOnlyFile(result.LocalPath)));
            }
        }
        catch (Exception ex)
        {
            isRunning = false;
            System.Diagnostics.Debug.WriteLine($"[HIBA]: {ex.Message}");
            ProgressTextLabel.Text = "Hiba történt a feldolgozás során!";
        }
    }

    private async void OnSaveMatchClicked(object? sender, EventArgs e)
    {
        if (_lastClipResult == null) return;

        if (!SessionService.IsLoggedIn)
        {
            SaveMatchStatusLabel.IsVisible = true;
            SaveMatchStatusLabel.Text = "⚠ Nincs bejelentkezve — a mentéshez jelentkezz be újra.";
            return;
        }

        BtnSaveMatch.IsEnabled = false;
        SaveMatchStatusLabel.IsVisible = true;
        SaveMatchStatusLabel.Text = "Mentés folyamatban...";

        try
        {
            var dto = new MatchAnalysisDto
            {
                MatchId = Guid.NewGuid().ToString(),
                VideoUrl = _lastClipResult.ClipUrl ?? "",
                // Megjegyzés: a képkockánkénti Players/Drawings adatok (frames) még nem
                // épülnek fel az events_path JSON-ból — ez egy következő lépés.
                // Egyelőre a klip elérhetőségét mentjük el, hogy legyen mihez visszatérni.
                Frames = new List<TacticalFrameDto>()
            };

            bool ok = await _matchService.SaveMatchAnalysisAsync(dto);
            SaveMatchStatusLabel.Text = ok
                ? "✅ Meccs elmentve a szerverre."
                : "❌ Sikertelen mentés (nézd meg a szerver logot).";
        }
        catch (Exception ex)
        {
            SaveMatchStatusLabel.Text = $"❌ Hiba: {ex.Message}";
        }
        finally
        {
            BtnSaveMatch.IsEnabled = true;
        }
    }

    private double ParseTimeToSeconds(string timeStr)
    {
        if (timeStr.Contains(':'))
        {
            var p = timeStr.Split(':');
            if (p.Length == 2 && int.TryParse(p[0], out int m) && int.TryParse(p[1], out int s)) return (m * 60) + s;
        }
        return double.TryParse(timeStr, out double t) ? t : 0;
    }
}
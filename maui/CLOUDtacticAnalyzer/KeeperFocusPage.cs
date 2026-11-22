using System.Collections.ObjectModel;
using CommunityToolkit.Maui.Views;
using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

/// <summary>
/// Kapus Modul: egy kész meccs-elemzésből a választott csapat kapusát érintő jelenetek egy helyen
/// (a motor /keeper-focus végpontja, keeper_focus.py): a kapus saját eseményei, az ellenfél lövései a kapura,
/// beadásai, szögletei, veszélyes szabadrúgásai, a támadások a saját zónában és a kapus 2 m-nél nagyobb mozgásai.
/// A régi kézi kapus-eszközök (vágás, rajzolás) a "Kézi kapus-eszközök" gombbal érhetők el.
/// </summary>
public class KeeperFocusPage : ContentPage
{
    private static readonly Color Bg = Color.FromArgb("#121212");
    private static readonly Color Panel = Color.FromArgb("#1E1E1E");
    private static readonly Color Muted = Color.FromArgb("#B0B0B0");

    private readonly MatchJobService _api = new();
    private List<JobInfo> _matchJobs = new();
    private MatchResultView? _result;
    private KeeperFocus? _focus;
    private readonly ObservableCollection<MatchEvent> _visible = new();
    private string? _playerSource;
    private string? _exportJobId;
    private IDispatcherTimer? _exportTimer;

    private readonly Picker _matchPicker = new() { Title = "Kész meccs-elemzés", TextColor = Colors.White, WidthRequest = 360 };
    private readonly Picker _teamPicker = new() { Title = "Kapus", TextColor = Colors.White, WidthRequest = 200 };
    private readonly Picker _groupPicker = new() { Title = "Csoport", TextColor = Colors.White, WidthRequest = 300 };
    private readonly Picker _halfPicker = new() { Title = "Félidő", TextColor = Colors.White, WidthRequest = 130 };
    private readonly Entry _minMove = new() { Text = "2", WidthRequest = 60, TextColor = Colors.White, BackgroundColor = Panel, Keyboard = Keyboard.Numeric };
    private readonly CheckBox _showMoves = new() { IsChecked = true };
    private readonly Label _status = new() { TextColor = Muted };
    private readonly Label _count = new() { TextColor = Muted };
    private readonly CollectionView _list = new() { SelectionMode = SelectionMode.Single, HeightRequest = 380 };
    private readonly MediaElement _player = new() { ShouldAutoPlay = false, ShouldShowPlaybackControls = true, Aspect = Aspect.AspectFit, HeightRequest = 360, BackgroundColor = Colors.Black };
    private readonly Button _exportButton = new() { Text = "✂ A látható jelenetek kivágása egy videóba", BackgroundColor = Color.FromArgb("#00897B"), IsEnabled = false };
    private readonly Label _exportStatus = new() { TextColor = Muted };

    private List<(string? Key, string Label)> _groupOptions = new() { (null, "Minden csoport") };

    public KeeperFocusPage()
    {
        Title = "Kapus Modul";
        BackgroundColor = Bg;

        _teamPicker.ItemsSource = new List<string> { $"{TeamNames.Team1} kapusa", $"{TeamNames.Team2} kapusa" };
        _teamPicker.SelectedIndex = 0;
        _halfPicker.ItemsSource = new List<string> { "Teljes meccs", "1. félidő", "2. félidő" };
        _halfPicker.SelectedIndex = 0;
        _groupPicker.ItemsSource = _groupOptions.Select(g => g.Label).ToList();
        _groupPicker.SelectedIndex = 0;

        _matchPicker.SelectedIndexChanged += async (_, _) => await LoadFocusAsync();
        _teamPicker.SelectedIndexChanged += async (_, _) => await LoadFocusAsync();
        _showMoves.CheckedChanged += async (_, _) => await LoadFocusAsync();
        _minMove.Completed += async (_, _) => await LoadFocusAsync();
        _groupPicker.SelectedIndexChanged += (_, _) => ApplyFilter();
        _halfPicker.SelectedIndexChanged += (_, _) => ApplyFilter();

        _list.ItemTemplate = new DataTemplate(() =>
        {
            var title = new Label { TextColor = Colors.White, FontAttributes = FontAttributes.Bold };
            title.SetBinding(Label.TextProperty, nameof(MatchEvent.Title));
            var sub = new Label { TextColor = Muted, FontSize = 12 };
            sub.SetBinding(Label.TextProperty, nameof(MatchEvent.Subtitle));
            return new VerticalStackLayout { Padding = new Thickness(8, 6), Spacing = 2, Children = { title, sub } };
        });
        _list.ItemsSource = _visible;
        _list.SelectionChanged += async (_, e) =>
        {
            if (e.CurrentSelection.FirstOrDefault() is MatchEvent ev) await PlayAsync(ev);
        };
        _exportButton.Clicked += async (_, _) => await ExportAsync();

        Content = BuildLayout();
    }

    private static View Row(params View[] views)
    {
        var row = new HorizontalStackLayout { Spacing = 8 };
        foreach (var v in views) row.Children.Add(v);
        return row;
    }

    private static Label Text(string t, double size = 14, bool bold = false) =>
        new() { Text = t, FontSize = size, TextColor = Colors.White, FontAttributes = bold ? FontAttributes.Bold : FontAttributes.None, VerticalOptions = LayoutOptions.Center };

    private View BuildLayout()
    {
        var refresh = new Button { Text = "🔄 Frissítés", BackgroundColor = Color.FromArgb("#455A64") };
        refresh.Clicked += async (_, _) => await LoadMatchesAsync();
        var manual = new Button { Text = "🛠 Kézi kapus-eszközök (vágás, rajzolás)", BackgroundColor = Color.FromArgb("#5D4037") };
        manual.Clicked += async (_, _) => await Navigation.PushAsync(new GoalkeeperPage());

        var left = new VerticalStackLayout
        {
            Spacing = 10,
            Children =
            {
                Text("🧤 Kapus Modul", 22, true),
                new Label
                {
                    Text = "Egy kész meccs-elemzésből összegyűjti, ami a kapust érinti: a kapus saját eseményeit, az ellenfél " +
                           "lövéseit, beadásait, szögleteit, veszélyes szabadrúgásait, a támadásokat a kapu előtti zónában, " +
                           "és a kapus mozgásait (alapból a 2 m-nél nagyobbakat). Kattints egy sorra a visszanézéshez.",
                    TextColor = Muted, FontSize = 12,
                },
                Row(_matchPicker, refresh),
                Row(_teamPicker, Text("Mozgás legalább"), _minMove, Text("m"), _showMoves, Text("Kapus-mozgások")),
                Row(_groupPicker, _halfPicker),
                _status,
                _count,
                _list,
                manual,
            },
        };
        var right = new VerticalStackLayout { Spacing = 10, Children = { _player, _exportButton, _exportStatus } };

        var grid = new Grid
        {
            Padding = 16, ColumnSpacing = 16,
            ColumnDefinitions = { new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Star) },
        };
        grid.Add(left, 0, 0);
        grid.Add(right, 1, 0);
        return new ScrollView { Content = grid };
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        var sel = Math.Max(0, _teamPicker.SelectedIndex);
        _teamPicker.ItemsSource = new List<string> { $"{TeamNames.Team1} kapusa", $"{TeamNames.Team2} kapusa" };
        _teamPicker.SelectedIndex = sel;
        if (_matchJobs.Count == 0) await LoadMatchesAsync();
    }

    private async Task LoadMatchesAsync()
    {
        try
        {
            _status.Text = "Elemzések betöltése…";
            var jobs = await _api.ListJobsAsync(100);
            _matchJobs = jobs.Where(j => j.Kind == "analyze-match" && j.Status == "done"
                                         && MatchResultView.From(j.Result)?.MatchEventsUrl is not null).ToList();
            _matchPicker.ItemsSource = _matchJobs
                .Select(j => $"{Path.GetFileName(j.VideoPath ?? j.Id)}  ({j.FinishedAt?.Replace('T', ' ')[..Math.Min(16, j.FinishedAt?.Length ?? 0)]})")
                .ToList();
            _status.Text = _matchJobs.Count == 0
                ? "Még nincs kész meccs-elemzés. Indíts egyet a „Meccs elemzés” fülön."
                : "Válassz egy elemzést.";
            if (_matchJobs.Count > 0) _matchPicker.SelectedIndex = 0;
        }
        catch (Exception ex)
        {
            _status.Text = $"A motor nem érhető el: {ex.Message}";
        }
    }

    private async Task LoadFocusAsync()
    {
        if (_matchPicker.SelectedIndex < 0 || _matchPicker.SelectedIndex >= _matchJobs.Count) return;
        _result = MatchResultView.From(_matchJobs[_matchPicker.SelectedIndex].Result);
        if (_result?.MatchEventsUrl is null) return;
        var team = _teamPicker.SelectedIndex == 1 ? "team2" : "team1";
        var minMove = double.TryParse(_minMove.Text?.Replace(',', '.'), System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out var mm) ? Math.Max(0.5, mm) : 2.0;
        try
        {
            _status.Text = "Betöltés…";
            _focus = await _api.GetKeeperFocusAsync(_result.MatchEventsUrl, team, minMove, _showMoves.IsChecked);
            var prev = _groupOptions.ElementAtOrDefault(Math.Max(0, _groupPicker.SelectedIndex)).Key;
            _groupOptions = new() { (null, $"Minden csoport ({_focus.Items.Count})") };
            foreach (var (key, label, count) in _focus.Groups)
                if (key != "mozgas" || _showMoves.IsChecked) _groupOptions.Add((key, $"{label} ({count})"));
            _groupPicker.ItemsSource = _groupOptions.Select(g => g.Label).ToList();
            _groupPicker.SelectedIndex = Math.Max(0, _groupOptions.FindIndex(g => g.Key == prev));
            _status.Text = $"{TeamNames.Of(team)} kapusa – {_focus.Items.Count} jelenet.";
            ApplyFilter();
        }
        catch (Exception ex)
        {
            _status.Text = $"Hiba: {ex.Message}";
        }
    }

    private void ApplyFilter()
    {
        _visible.Clear();
        if (_focus is null) return;
        var group = _groupOptions.ElementAtOrDefault(Math.Max(0, _groupPicker.SelectedIndex)).Key;
        var half = _halfPicker.SelectedIndex switch { 1 => 1, 2 => 2, _ => 0 };
        foreach (var ev in _focus.Items)
        {
            if (group is not null && ev.Group != group) continue;
            if (half != 0 && ev.Half != half) continue;
            _visible.Add(ev);
        }
        _count.Text = $"{_visible.Count} jelenet látszik.";
        _exportButton.IsEnabled = _visible.Count > 0 && _exportJobId is null;
    }

    private async Task PlayAsync(MatchEvent ev)
    {
        var video = _result?.VideoPath;
        if (string.IsNullOrEmpty(video)) return;
        var (start, _) = ev.ClipWindow();
        try
        {
            if (_playerSource != video)
            {
                _player.Source = File.Exists(video) ? MediaSource.FromFile(video) : MediaSource.FromUri(Engine.Url(video));
                _playerSource = video;
                await Task.Delay(400);   // a forrás betöltése, különben a SeekTo elveszhet
            }
            await _player.SeekTo(TimeSpan.FromSeconds(start));
            _player.Play();
        }
        catch (Exception ex)
        {
            _exportStatus.Text = $"Lejátszási hiba: {ex.Message}";
        }
    }

    private async Task ExportAsync()
    {
        var video = _result?.VideoPath;
        if (string.IsNullOrEmpty(video) || _visible.Count == 0) return;
        try
        {
            _exportButton.IsEnabled = false;
            _exportStatus.Text = $"Kivágás indul: {_visible.Count} jelenet…";
            _exportJobId = await _api.StartExportClipsAsync(video, _visible.ToList(), "Kapus");
            _exportTimer ??= Dispatcher.CreateTimer();
            _exportTimer.Interval = TimeSpan.FromSeconds(2);
            _exportTimer.Tick -= OnExportTick;
            _exportTimer.Tick += OnExportTick;
            _exportTimer.Start();
        }
        catch (Exception ex)
        {
            _exportStatus.Text = $"Hiba: {ex.Message}";
            _exportButton.IsEnabled = true;
        }
    }

    private async void OnExportTick(object? sender, EventArgs e)
    {
        if (_exportJobId is null) { _exportTimer?.Stop(); return; }
        try
        {
            var job = await _api.GetJobAsync(_exportJobId);
            if (!job.IsFinished)
            {
                _exportStatus.Text = $"Kivágás: {job.Progress:0}% – {job.Phase}";
                return;
            }
            _exportTimer?.Stop();
            _exportJobId = null;
            _exportButton.IsEnabled = _visible.Count > 0;
            var res = MatchJobService.ParseExportResult(job);
            if (job.Status != "done" || res?.Status != "success" || string.IsNullOrEmpty(res.MergedUrl))
            {
                _exportStatus.Text = $"A kivágás nem sikerült: {res?.Message ?? job.Error ?? job.Status}";
                return;
            }
            _exportStatus.Text = "Kész – lejátszás a fenti lejátszóban.";
            _player.Source = MediaSource.FromUri(Engine.Url(res.MergedUrl));
            _playerSource = res.MergedUrl;
            _player.Play();
        }
        catch (Exception ex)
        {
            _exportStatus.Text = $"Hiba: {ex.Message}";
        }
    }
}

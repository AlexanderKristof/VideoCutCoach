using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using CommunityToolkit.Maui.Views;
using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

/// <summary>Egy esemény a listában (kijelölhető).</summary>
public sealed class EventItem : INotifyPropertyChanged
{
    private bool _selected;
    public EventItem(MatchEvent ev) => Event = ev;
    public MatchEvent Event { get; }
    public string Title => Event.Title;
    public string Subtitle => Event.Subtitle;

    public bool Selected
    {
        get => _selected;
        set { if (_selected == value) return; _selected = value; OnChanged(); }
    }

    public event PropertyChangedEventHandler? PropertyChanged;
    private void OnChanged([CallerMemberName] string? name = null) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

/// <summary>Egy elemzés a listában (helyben frissül, így a lista nem villog).</summary>
public sealed class JobItem : INotifyPropertyChanged
{
    public JobItem(JobInfo job) => Job = job;
    public JobInfo Job { get; private set; }
    public string Id => Job.Id;
    public string Title => Path.GetFileName(Job.VideoPath ?? "") is { Length: > 0 } n ? n : Job.Id;
    public string Subtitle => $"{Job.StatusLabel} – {Job.ProgressText}";
    public string Created => DateTime.TryParse(Job.CreatedAt, out var t) ? t.ToString("yyyy.MM.dd HH:mm") : "";
    public double Progress => Job.Progress / 100.0;
    public bool ShowProgress => !Job.IsFinished;

    public void Update(JobInfo job)
    {
        Job = job;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(null));   // minden kötött mező frissül
    }

    public event PropertyChangedEventHandler? PropertyChanged;
}

/// <summary>
/// "Meccs elemzés" fül: a teljes meccs elemzése egy gombnyomásra a háttérben (POST /jobs/analyze-match),
/// a haladás követése (szüneteltetés / folytatás / megszakítás), majd az eredmény: eseménylista szűrőkkel,
/// lejátszás az eseménynél, a kijelölt jelenetek kivágása egy videóba, hőtérképek, kapus- és
/// összefoglaló-videók, és megnyitás a taktikai táblán.
/// </summary>
public class MatchAnalysisPage : ContentPage
{
    private static readonly Color Bg = Color.FromArgb("#121212");
    private static readonly Color Panel = Color.FromArgb("#1E1E1E");
    private static readonly Color Muted = Color.FromArgb("#B0B0B0");

    private readonly MatchJobService _api = new();
    private readonly ObservableCollection<JobItem> _jobs = new();
    private readonly ObservableCollection<EventItem> _visibleEvents = new();
    private List<EventItem> _allEvents = new();
    private IDispatcherTimer? _timer;
    private bool _polling;

    private string? _videoPath;
    private JobInfo? _currentJob;
    private MatchResultView? _result;
    private string? _loadedEventsUrl;
    private string? _exportJobId;

    // --- bal oldal: új elemzés + elemzések listája
    private readonly Label _videoLabel = new() { Text = "Nincs videó kiválasztva", TextColor = Muted, LineBreakMode = LineBreakMode.MiddleTruncation };
    private readonly Entry _halfMin = new() { Text = "45", Keyboard = Keyboard.Numeric, WidthRequest = 60, TextColor = Colors.White, BackgroundColor = Panel };
    private readonly Entry _firstStart = TimeEntry("1. kezdete"), _firstEnd = TimeEntry("1. vége"), _secondStart = TimeEntry("2. kezdete"), _secondEnd = TimeEntry("2. vége");
    private readonly CheckBox _background = new() { IsChecked = true }, _heatmaps = new() { IsChecked = true },
        _gkComp = new() { IsChecked = true }, _highlights = new() { IsChecked = true }, _team1Right = new() { IsChecked = true };
    private readonly Button _startButton = new() { Text = "▶ Elemzés indítása", BackgroundColor = Color.FromArgb("#E91E63"), FontAttributes = FontAttributes.Bold };
    private readonly CollectionView _jobList = new() { SelectionMode = SelectionMode.Single, HeightRequest = 260 };

    // --- jobb oldal: kiválasztott elemzés
    private readonly Label _jobTitle = new() { Text = "Válassz ki egy elemzést, vagy indíts újat.", FontSize = 18, FontAttributes = FontAttributes.Bold, TextColor = Colors.White };
    private readonly Label _jobPhase = new() { TextColor = Muted };
    private readonly ProgressBar _jobProgress = new() { ProgressColor = Color.FromArgb("#E91E63"), IsVisible = false };
    private readonly Button _pauseButton = SmallButton("⏸ Szünet"), _cancelButton = SmallButton("✖ Megszakítás");
    private readonly Label _summary = new() { TextColor = Colors.White };
    private readonly Label _warnings = new() { TextColor = Color.FromArgb("#FFB74D"), IsVisible = false };
    private readonly Picker _typeFilter = new() { Title = "Esemény", TextColor = Colors.White, WidthRequest = 220 };
    private readonly Picker _teamFilter = new() { Title = "Csapat", TextColor = Colors.White, WidthRequest = 130 };
    private readonly Picker _halfFilter = new() { Title = "Félidő", TextColor = Colors.White, WidthRequest = 130 };
    private readonly CollectionView _eventList = new() { SelectionMode = SelectionMode.Single, HeightRequest = 330 };
    private readonly Label _eventCount = new() { TextColor = Muted, VerticalOptions = LayoutOptions.Center };
    private readonly Button _exportButton = new() { Text = "✂ Kijelöltek kivágása egy videóba", BackgroundColor = Color.FromArgb("#00897B") };
    private readonly Label _exportStatus = new() { TextColor = Muted };
    private readonly MediaElement _player = new() { ShouldAutoPlay = false, ShouldShowPlaybackControls = true, Aspect = Aspect.AspectFit, HeightRequest = 360, BackgroundColor = Colors.Black };
    private readonly HorizontalStackLayout _videoButtons = new() { Spacing = 8 };
    private readonly HorizontalStackLayout _heatmapStrip = new() { Spacing = 10 };
    private readonly VerticalStackLayout _resultBox = new() { Spacing = 10, IsVisible = false };
    private string? _playerSource;

    private static readonly (string Key, string Label, Func<MatchEvent, bool> Match)[] TypeFilters =
    {
        ("all", "Minden esemény", _ => true),
        ("goals_shots", "Gólok és lövések", e => e.Type is "goal" or "shot"),
        ("gk", "Kapus események (mind)", e => e.IsGoalkeeper),
        ("gk_save", "Kapus védés", e => e.Type == "gk_save"),
        ("gk_claim_cross", "Beadás leszedése", e => e.Type == "gk_claim_cross"),
        ("gk_goal_kick", "Kirúgás", e => e.Type == "gk_goal_kick"),
        ("gk_long_ball", "Kapus hosszú labda", e => e.Type == "gk_long_ball"),
        ("gk_short_pass", "Kapus rövid passz", e => e.Type == "gk_short_pass"),
        ("gk_conceded", "Kapott gól", e => e.Type == "gk_conceded"),
        ("gk_sweeper", "Kapus kifutás", e => e.Type == "gk_sweeper"),
        ("other", "Egyéb", e => !e.IsGoalkeeper && e.Type is not ("goal" or "shot")),
    };

    public MatchAnalysisPage()
    {
        Title = "Meccs elemzés";
        BackgroundColor = Bg;

        _typeFilter.ItemsSource = TypeFilters.Select(f => f.Label).ToList();
        _typeFilter.SelectedIndex = 0;
        _teamFilter.ItemsSource = new List<string> { "Mindkét csapat", "Hazai", "Vendég" };
        _teamFilter.SelectedIndex = 0;
        _halfFilter.ItemsSource = new List<string> { "Teljes meccs", "1. félidő", "2. félidő" };
        _halfFilter.SelectedIndex = 0;
        _typeFilter.SelectedIndexChanged += (_, _) => ApplyFilter();
        _teamFilter.SelectedIndexChanged += (_, _) => ApplyFilter();
        _halfFilter.SelectedIndexChanged += (_, _) => ApplyFilter();

        _jobList.ItemsSource = _jobs;
        _jobList.ItemTemplate = new DataTemplate(BuildJobCell);
        _jobList.SelectionChanged += async (_, e) =>
        {
            if (e.CurrentSelection.FirstOrDefault() is JobItem item && item.Job.Id != _currentJob?.Id)
                await SelectJobAsync(item.Job);
        };

        _eventList.ItemsSource = _visibleEvents;
        _eventList.ItemTemplate = new DataTemplate(BuildEventCell);
        _eventList.SelectionChanged += async (_, e) =>
        {
            if (e.CurrentSelection.FirstOrDefault() is EventItem item) await PlayEventAsync(item.Event);
        };

        _startButton.Clicked += async (_, _) => await StartAnalysisAsync();
        _pauseButton.Clicked += async (_, _) => await PauseResumeAsync();
        _cancelButton.Clicked += async (_, _) => await CancelAsync();
        _exportButton.Clicked += async (_, _) => await ExportSelectedAsync();

        Content = BuildLayout();
    }

    // ================================================================ felület

    private static Entry TimeEntry(string placeholder) =>
        new() { Placeholder = placeholder, WidthRequest = 95, TextColor = Colors.White, PlaceholderColor = Colors.Gray, BackgroundColor = Panel };

    private static Button SmallButton(string text) =>
        new() { Text = text, BackgroundColor = Color.FromArgb("#455A64"), Padding = new Thickness(10, 4), IsVisible = false };

    private static View Row(params View[] views)
    {
        var row = new HorizontalStackLayout { Spacing = 8 };
        foreach (var v in views) row.Children.Add(v);
        return row;
    }

    private static View Check(CheckBox box, string text) =>
        Row(box, new Label { Text = text, TextColor = Colors.White, VerticalOptions = LayoutOptions.Center });

    private static Label Header(string text) =>
        new() { Text = text, FontSize = 16, FontAttributes = FontAttributes.Bold, TextColor = Colors.White, Margin = new Thickness(0, 8, 0, 0) };

    private View BuildLayout()
    {
        var pick = new Button { Text = "📁 Meccsvideó kiválasztása", BackgroundColor = Color.FromArgb("#2196F3") };
        pick.Clicked += async (_, _) => await PickVideoAsync();
        var refresh = new Button { Text = "↻", BackgroundColor = Color.FromArgb("#455A64"), WidthRequest = 44 };
        refresh.Clicked += async (_, _) => await RefreshJobsAsync();

        var left = new VerticalStackLayout
        {
            Spacing = 8,
            Padding = 14,
            Children =
            {
                Header("Új elemzés"),
                pick,
                _videoLabel,
                Row(new Label { Text = "Félidő hossza (perc):", TextColor = Colors.White, VerticalOptions = LayoutOptions.Center }, _halfMin),
                new Label { Text = "Félidők (nem kötelező – üresen automatikus felismerés; pp:mm vagy ó:pp:mm):", TextColor = Muted, FontSize = 12 },
                Row(_firstStart, _firstEnd),
                Row(_secondStart, _secondEnd),
                Check(_team1Right, "A hazai csapat az 1. félidőben jobbra támad"),
                Check(_background, "Háttér-mód (a gép közben használható marad)"),
                Check(_heatmaps, "Hőtérképek"),
                Check(_gkComp, "Kapus-videók (kapusonként)"),
                Check(_highlights, "Összefoglaló videó (gólok, lövések)"),
                _startButton,
                Row(Header("Elemzések"), refresh),
                _jobList,
            }
        };

        var selectAll = new Button { Text = "Mind kijelöl", BackgroundColor = Color.FromArgb("#455A64") };
        selectAll.Clicked += (_, _) => { foreach (var e in _visibleEvents) e.Selected = true; UpdateEventCount(); };
        var selectNone = new Button { Text = "Kijelölés törlése", BackgroundColor = Color.FromArgb("#455A64") };
        selectNone.Clicked += (_, _) => { foreach (var e in _allEvents) e.Selected = false; UpdateEventCount(); };
        var board = new Button { Text = "🗺 Megnyitás a taktikai táblán", BackgroundColor = Color.FromArgb("#5E35B1") };
        board.Clicked += async (_, _) => await OpenBoardAsync(withVideo: false);
        var drawOnVideo = new Button { Text = "✏ Rajzolás a videón", BackgroundColor = Color.FromArgb("#5E35B1") };
        drawOnVideo.Clicked += async (_, _) => await OpenBoardAsync(withVideo: true);
        var review = new Button { Text = "🔍 Áttekintő oldal", BackgroundColor = Color.FromArgb("#455A64") };
        review.Clicked += async (_, _) =>
        {
            if (_result?.ReviewPages.FirstOrDefault() is { Url: { Length: > 0 } u })
                await Launcher.Default.OpenAsync(new Uri(Engine.Url(u)));
        };

        _resultBox.Children.Add(_summary);
        _resultBox.Children.Add(_warnings);
        _resultBox.Children.Add(Row(board, drawOnVideo, review));
        _resultBox.Children.Add(_player);
        _resultBox.Children.Add(new ScrollView { Orientation = ScrollOrientation.Horizontal, Content = _videoButtons });
        _resultBox.Children.Add(Header("Események"));
        _resultBox.Children.Add(Row(_typeFilter, _teamFilter, _halfFilter));
        _resultBox.Children.Add(Row(selectAll, selectNone, _eventCount));
        _resultBox.Children.Add(_eventList);
        _resultBox.Children.Add(_exportButton);
        _resultBox.Children.Add(_exportStatus);
        _resultBox.Children.Add(Header("Hőtérképek"));
        _resultBox.Children.Add(new ScrollView { Orientation = ScrollOrientation.Horizontal, Content = _heatmapStrip });

        var right = new VerticalStackLayout
        {
            Spacing = 8,
            Padding = 14,
            Children = { _jobTitle, _jobPhase, _jobProgress, Row(_pauseButton, _cancelButton), _resultBox }
        };

        var grid = new Grid
        {
            ColumnDefinitions = { new ColumnDefinition(new GridLength(380)), new ColumnDefinition(GridLength.Star) },
        };
        grid.Add(new ScrollView { Content = left, BackgroundColor = Color.FromArgb("#181818") }, 0, 0);
        grid.Add(new ScrollView { Content = right }, 1, 0);
        return grid;
    }

    private static View BuildJobCell()
    {
        var title = new Label { TextColor = Colors.White, FontAttributes = FontAttributes.Bold, LineBreakMode = LineBreakMode.MiddleTruncation };
        title.SetBinding(Label.TextProperty, nameof(JobItem.Title));
        var sub = new Label { TextColor = Muted, FontSize = 12 };
        sub.SetBinding(Label.TextProperty, nameof(JobItem.Subtitle));
        var created = new Label { TextColor = Color.FromArgb("#808080"), FontSize = 11 };
        created.SetBinding(Label.TextProperty, nameof(JobItem.Created));
        var bar = new ProgressBar { ProgressColor = Color.FromArgb("#E91E63") };
        bar.SetBinding(ProgressBar.ProgressProperty, nameof(JobItem.Progress));
        bar.SetBinding(IsVisibleProperty, nameof(JobItem.ShowProgress));
        return new Border
        {
            Stroke = Color.FromArgb("#333333"),
            BackgroundColor = Panel,
            Padding = 8,
            Margin = new Thickness(0, 3),
            Content = new VerticalStackLayout { Spacing = 2, Children = { title, sub, bar, created } }
        };
    }

    private View BuildEventCell()
    {
        var box = new CheckBox { VerticalOptions = LayoutOptions.Center };
        box.SetBinding(CheckBox.IsCheckedProperty, nameof(EventItem.Selected), BindingMode.TwoWay);
        box.CheckedChanged += (_, _) => UpdateEventCount();
        var title = new Label { TextColor = Colors.White, FontAttributes = FontAttributes.Bold };
        title.SetBinding(Label.TextProperty, nameof(EventItem.Title));
        var sub = new Label { TextColor = Muted, FontSize = 12, LineBreakMode = LineBreakMode.TailTruncation };
        sub.SetBinding(Label.TextProperty, nameof(EventItem.Subtitle));
        var grid = new Grid
        {
            ColumnDefinitions = { new ColumnDefinition(GridLength.Auto), new ColumnDefinition(GridLength.Star) },
            Padding = new Thickness(4, 3),
            BackgroundColor = Panel,
            Margin = new Thickness(0, 2),
        };
        grid.Add(box, 0, 0);
        grid.Add(new VerticalStackLayout { Spacing = 1, Children = { title, sub } }, 1, 0);
        return grid;
    }

    // ================================================================ életciklus

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        _timer ??= CreateTimer();
        _timer.Start();
        await RefreshJobsAsync();
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        _timer?.Stop();
    }

    private IDispatcherTimer CreateTimer()
    {
        var t = Dispatcher.CreateTimer();
        t.Interval = TimeSpan.FromSeconds(2);
        t.Tick += async (_, _) => await PollAsync();
        return t;
    }

    private int _tick;

    private async Task PollAsync()
    {
        if (_polling) return;
        _polling = true;
        try
        {
            if (_currentJob is { IsFinished: false })
                await SelectJobAsync(await _api.GetJobAsync(_currentJob.Id), fromPoll: true);
            if (_exportJobId is not null)
                await PollExportAsync();
            if (++_tick % 5 == 0 || _jobs.Any(j => !j.Job.IsFinished))
                await RefreshJobsAsync();
        }
        catch (Exception ex)
        {
            _jobPhase.Text = $"⚠ A motor nem válaszol: {ex.Message}";
        }
        finally
        {
            _polling = false;
        }
    }

    // ================================================================ műveletek

    private async Task PickVideoAsync()
    {
        try
        {
            var result = await FilePicker.Default.PickAsync(new PickOptions { PickerTitle = "Válaszd ki a meccsvideót", FileTypes = FilePickerFileType.Videos });
            if (result is null) return;
            _videoPath = result.FullPath;
            _videoLabel.Text = result.FileName;
            _videoLabel.TextColor = Colors.White;
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync("Hiba", ex.Message, "OK");
        }
    }

    private async Task StartAnalysisAsync()
    {
        if (string.IsNullOrEmpty(_videoPath))
        {
            await DisplayAlertAsync("Nincs videó", "Előbb válaszd ki a meccsvideót.", "OK");
            return;
        }
        var manual = new[] { _firstStart.Text, _firstEnd.Text, _secondStart.Text, _secondEnd.Text }.Count(s => !string.IsNullOrWhiteSpace(s));
        if (manual is > 0 and < 4)
        {
            await DisplayAlertAsync("Félidők", "A félidőket vagy mind a négy mezővel add meg, vagy hagyd mindet üresen (automatikus felismerés).", "OK");
            return;
        }
        if (!double.TryParse(_halfMin.Text?.Replace(',', '.'), System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var halfMin) || halfMin < 5 || halfMin > 60)
        {
            await DisplayAlertAsync("Félidő hossza", "A félidő hossza 5 és 60 perc között legyen.", "OK");
            return;
        }
        var opts = new MatchAnalysisOptions
        {
            ExpectedHalfMin = halfMin,
            FirstStart = _firstStart.Text ?? "",
            FirstEnd = _firstEnd.Text ?? "",
            SecondStart = _secondStart.Text ?? "",
            SecondEnd = _secondEnd.Text ?? "",
            Background = _background.IsChecked,
            Heatmaps = _heatmaps.IsChecked,
            GoalkeeperCompilations = _gkComp.IsChecked,
            Highlights = _highlights.IsChecked,
            Team1AttacksRight = _team1Right.IsChecked,
        };
        _startButton.IsEnabled = false;
        try
        {
            var id = await _api.StartMatchAnalysisAsync(_videoPath, opts);
            await SelectJobAsync(await _api.GetJobAsync(id));
            await RefreshJobsAsync();
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync("Nem indult el az elemzés", ex.Message, "OK");
        }
        finally
        {
            _startButton.IsEnabled = true;
        }
    }

    private async Task RefreshJobsAsync()
    {
        try
        {
            var jobs = (await _api.ListJobsAsync(100)).Where(j => j.Kind == "analyze-match").ToList();
            var sameOrder = jobs.Count == _jobs.Count && jobs.Select(j => j.Id).SequenceEqual(_jobs.Select(x => x.Id));
            if (sameOrder)
            {
                // helyben frissítés: nem épül újra a lista (nem villog, a kijelölés megmarad)
                for (var i = 0; i < jobs.Count; i++) _jobs[i].Update(jobs[i]);
                return;
            }
            var selectedId = _currentJob?.Id;
            _jobs.Clear();
            foreach (var j in jobs) _jobs.Add(new JobItem(j));
            if (selectedId is not null)
                _jobList.SelectedItem = _jobs.FirstOrDefault(x => x.Id == selectedId);
        }
        catch (Exception ex)
        {
            _jobPhase.Text = $"⚠ Az elemzések listája nem tölthető be: {ex.Message}";
        }
    }

    private async Task SelectJobAsync(JobInfo job, bool fromPoll = false)
    {
        if (!fromPoll && job.Result is null)
        {
            try { job = await _api.GetJobAsync(job.Id); } catch { /* a listás adat is jó */ }
        }
        _currentJob = job;
        _jobTitle.Text = Path.GetFileName(job.VideoPath ?? job.Id);
        _jobPhase.Text = job.Status == "error"
            ? $"Hiba: {job.Error}"
            : $"{job.StatusLabel} – {job.Phase}{Environment.NewLine}{job.ProgressText}";
        _jobProgress.IsVisible = !job.IsFinished;
        _jobProgress.Progress = job.Progress / 100.0;
        _pauseButton.IsVisible = _cancelButton.IsVisible = !job.IsFinished;
        _pauseButton.Text = job.Status == "paused" || job.Paused ? "▶ Folytatás" : "⏸ Szünet";
        _cancelButton.IsEnabled = _pauseButton.IsEnabled = !job.CancelRequested;
        _cancelButton.Text = job.CancelRequested ? "Megszakítás folyamatban…" : "✖ Megszakítás";

        var result = MatchResultView.From(job.Result);
        _result = result;
        _resultBox.IsVisible = result is not null;
        if (result is null) return;

        ShowSummary(job, result);
        ShowVideos(result);
        ShowHeatmaps(result);
        if (!string.IsNullOrEmpty(result.MatchEventsUrl) && result.MatchEventsUrl != _loadedEventsUrl)
            await LoadEventsAsync(result.MatchEventsUrl);
    }

    private void ShowSummary(JobInfo job, MatchResultView r)
    {
        var counts = r.EventCounts.Count == 0
            ? "még nincs esemény"
            : string.Join(", ", r.EventCounts.OrderByDescending(kv => kv.Value).Select(kv => $"{EventLabels.Label(kv.Key)}: {kv.Value}"));
        var time = job.ElapsedSec is { } s ? $"  •  elemzési idő: {MatchEvent.FormatTime(s)}" : "";
        _summary.Text = (r.Partial ? "Részeredmény (az elemzés még fut). " : "") + counts + time;
        var msgs = r.Warnings.Concat(r.Notes).ToList();
        _warnings.Text = string.Join(Environment.NewLine, msgs.Select(m => "• " + m));
        _warnings.IsVisible = msgs.Count > 0;
    }

    private void ShowVideos(MatchResultView r)
    {
        _videoButtons.Children.Clear();
        void Add(string text, string? url)
        {
            if (string.IsNullOrEmpty(url)) return;
            var b = new Button { Text = text, BackgroundColor = Color.FromArgb("#37474F") };
            b.Clicked += (_, _) => PlayUrl(url);
            _videoButtons.Children.Add(b);
        }
        Add("🎬 Szünet nélküli meccs", r.FullMatchUrl);
        foreach (var (title, url, count) in r.Compilations)
            Add($"🎞 {title} ({count})", url);
    }

    private void ShowHeatmaps(MatchResultView r)
    {
        _heatmapStrip.Children.Clear();
        foreach (var (title, url) in r.Heatmaps)
        {
            var img = new Image { Source = ImageSource.FromUri(new Uri(Engine.Url(url))), WidthRequest = 320, HeightRequest = 210, Aspect = Aspect.AspectFit };
            var tap = new TapGestureRecognizer();
            tap.Tapped += async (_, _) => await Launcher.Default.OpenAsync(new Uri(Engine.Url(url)));
            img.GestureRecognizers.Add(tap);
            _heatmapStrip.Children.Add(new VerticalStackLayout
            {
                Spacing = 4,
                Children = { img, new Label { Text = title, TextColor = Muted, HorizontalOptions = LayoutOptions.Center } }
            });
        }
        if (r.Heatmaps.Count == 0)
            _heatmapStrip.Children.Add(new Label { Text = "Nincs hőtérkép (lásd a figyelmeztetéseket).", TextColor = Muted });
    }

    private async Task LoadEventsAsync(string url)
    {
        try
        {
            var events = await _api.LoadEventsAsync(url);
            _loadedEventsUrl = url;
            _allEvents = events.Select(e => new EventItem(e)).ToList();
            ApplyFilter();
        }
        catch (Exception ex)
        {
            _eventCount.Text = $"Az események nem tölthetők be: {ex.Message}";
        }
    }

    private void ApplyFilter()
    {
        var type = TypeFilters[Math.Max(0, _typeFilter.SelectedIndex)].Match;
        var team = _teamFilter.SelectedIndex switch { 1 => "team1", 2 => "team2", _ => null };
        var half = _halfFilter.SelectedIndex switch { 1 => 1, 2 => 2, _ => 0 };
        _visibleEvents.Clear();
        foreach (var item in _allEvents)
        {
            var e = item.Event;
            if (!type(e)) continue;
            if (team is not null && e.Team != team) continue;
            if (half != 0 && e.Half != half) continue;
            _visibleEvents.Add(item);
        }
        UpdateEventCount();
    }

    private void UpdateEventCount()
    {
        var selected = _allEvents.Count(e => e.Selected);
        _eventCount.Text = $"{_visibleEvents.Count} / {_allEvents.Count} esemény látszik, {selected} kijelölve";
        _exportButton.IsEnabled = selected > 0;
    }

    private async Task PlayEventAsync(MatchEvent ev)
    {
        var video = _result?.VideoPath ?? _currentJob?.VideoPath;
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

    private void PlayUrl(string url)
    {
        _player.Source = MediaSource.FromUri(Engine.Url(url));
        _playerSource = url;
        _player.Play();
    }

    private async Task PauseResumeAsync()
    {
        if (_currentJob is null) return;
        try
        {
            var job = _currentJob.Status == "paused" || _currentJob.Paused
                ? await _api.ResumeAsync(_currentJob.Id)
                : await _api.PauseAsync(_currentJob.Id);
            await SelectJobAsync(job, fromPoll: true);
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync("Hiba", ex.Message, "OK");
        }
    }

    private async Task CancelAsync()
    {
        if (_currentJob is null) return;
        if (!await DisplayAlertAsync("Megszakítás", "Biztosan leállítod az elemzést? A már elkészült részek megmaradnak.", "Igen", "Nem"))
            return;
        try
        {
            await SelectJobAsync(await _api.CancelAsync(_currentJob.Id), fromPoll: true);
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync("Hiba", ex.Message, "OK");
        }
    }

    private async Task ExportSelectedAsync()
    {
        var video = _result?.VideoPath ?? _currentJob?.VideoPath;
        var selected = _allEvents.Where(e => e.Selected).Select(e => e.Event).ToList();
        if (string.IsNullOrEmpty(video) || selected.Count == 0) return;
        var title = selected.Select(e => e.Type).Distinct().Count() == 1 ? selected[0].Label : "Valogatas";
        try
        {
            _exportButton.IsEnabled = false;
            _exportStatus.Text = $"Kivágás indul: {selected.Count} jelenet…";
            _exportJobId = await _api.StartExportClipsAsync(video, selected, title);
        }
        catch (Exception ex)
        {
            _exportStatus.Text = $"Hiba: {ex.Message}";
            _exportButton.IsEnabled = true;
        }
    }

    private async Task PollExportAsync()
    {
        var job = await _api.GetJobAsync(_exportJobId!);
        if (!job.IsFinished)
        {
            _exportStatus.Text = $"Kivágás: {job.Progress:0}% – {job.Phase}";
            return;
        }
        _exportJobId = null;
        _exportButton.IsEnabled = true;
        var res = MatchJobService.ParseExportResult(job);
        if (job.Status != "done" || res?.Status != "success" || string.IsNullOrEmpty(res.MergedUrl))
        {
            _exportStatus.Text = $"A kivágás nem sikerült: {res?.Message ?? job.Error ?? job.Status}";
            return;
        }
        _exportStatus.Text = $"Kész: {res.Clips.Count} jelenet → {res.MergedLocalPath}";
        PlayUrl(res.MergedUrl);
        var open = new Button { Text = "📂 Mappa megnyitása", BackgroundColor = Color.FromArgb("#37474F") };
        var path = res.MergedLocalPath;
        open.Clicked += async (_, _) =>
        {
            if (!string.IsNullOrEmpty(path) && File.Exists(path))
                await Launcher.Default.OpenAsync(new OpenFileRequest("Videó", new ReadOnlyFile(path)));
        };
        _videoButtons.Children.Add(open);
    }

    private async Task OpenBoardAsync(bool withVideo)
    {
        if (_result?.MatchEventsUrl is not { Length: > 0 } events) return;
        var query = $"events={Uri.EscapeDataString(events)}";
        if (withVideo)
        {
            if (_result.FullMatchUrl is not { Length: > 0 } video)
            {
                await DisplayAlertAsync("Nincs videó", "Ehhez az elemzéshez nem készült szünet nélküli meccsvideó.", "OK");
                return;
            }
            // A szünet nélküli videóhoz a meccs-JSON full_match_sec ideje tartozik – a tábla ezt használja.
            query = $"video={Uri.EscapeDataString(video)}&{query}";
        }
        await Shell.Current.GoToAsync($"//TacticsBoardPage?{query}");
    }
}

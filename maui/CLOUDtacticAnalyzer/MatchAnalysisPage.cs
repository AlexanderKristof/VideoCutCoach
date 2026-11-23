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
    private string? _verdict;
    private bool _fontos;
    public EventItem(MatchEvent ev, string? manualId = null)
    {
        Event = ev;
        ManualId = manualId;
    }
    public MatchEvent Event { get; }
    /// <summary>Kézzel felvett (az automatikából kimaradt) esemény azonosítója, különben null.</summary>
    public string? ManualId { get; }
    public bool IsManual => ManualId is not null;
    public string Title => IsManual ? $"➕ {Event.Label}{(string.IsNullOrEmpty(Event.TeamLabel) ? "" : " – " + Event.TeamLabel)} (kézzel felvett)" : Event.Title;
    public string Subtitle => Score is { } sc ? $"⭐ ajánlott: {sc * 100:0}%  •  {Event.Subtitle}" : Event.Subtitle;

    /// <summary>Az edző ⭐ jelöléseiből tanult ajánlottság (0..1), vagy null, ha még nincs modell.</summary>
    public double? Score
    {
        get => _score;
        set { _score = value; OnChanged(); OnChanged(nameof(Subtitle)); }
    }
    private double? _score;

    public bool Selected
    {
        get => _selected;
        set { if (_selected == value) return; _selected = value; OnChanged(); }
    }

    /// <summary>Edzői ítélet: "ok" (valódi), "hamis" (téves) vagy null (még nem nézte meg).</summary>
    public string? Verdict
    {
        get => _verdict;
        set
        {
            if (_verdict == value) return;
            _verdict = value;
            OnChanged();
            OnChanged(nameof(OkColor));
            OnChanged(nameof(WrongColor));
            OnChanged(nameof(RowColor));
        }
    }

    /// <summary>Az edző "fontos" (⭐) jelölése: ezt a jelenetet látni akarja a válogatásban.</summary>
    public bool Fontos
    {
        get => _fontos;
        set
        {
            if (_fontos == value) return;
            _fontos = value;
            OnChanged();
            OnChanged(nameof(StarText));
            OnChanged(nameof(StarColor));
        }
    }

    public string StarText => Fontos ? "⭐" : "☆";
    public Color StarColor => Fontos ? Color.FromArgb("#8D6E00") : Color.FromArgb("#37474F");

    public Color OkColor => Verdict == "ok" ? Color.FromArgb("#2E7D32") : Color.FromArgb("#37474F");
    public Color WrongColor => Verdict == "hamis" ? Color.FromArgb("#C62828") : Color.FromArgb("#37474F");
    public Color RowColor => IsManual ? Color.FromArgb("#1B2A20") : Verdict == "hamis" ? Color.FromArgb("#2A1B1B")
        : Verdict == "ok" ? Color.FromArgb("#1B2A1E") : Color.FromArgb("#1E1E1E");

    /// <summary>Minden kötött mező frissítése (pl. a csapatnevek átnevezése után).</summary>
    public void Refresh() => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(null));

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
        _gkComp = new() { IsChecked = true }, _highlights = new() { IsChecked = true },
        _makeFullMatch = new() { IsChecked = true };
    private readonly Button _startButton = new() { Text = "▶ Elemzés indítása", BackgroundColor = Color.FromArgb("#E91E63"), FontAttributes = FontAttributes.Bold };
    private readonly CollectionView _jobList = new() { SelectionMode = SelectionMode.Single, HeightRequest = 260 };

    // --- jobb oldal: kiválasztott elemzés
    private readonly Label _jobTitle = new() { Text = "Válassz ki egy elemzést, vagy indíts újat.", FontSize = 18, FontAttributes = FontAttributes.Bold, TextColor = Colors.White };
    private readonly Label _jobPhase = new() { TextColor = Muted };
    private readonly BoxView _swatch1 = new() { WidthRequest = 22, HeightRequest = 22, CornerRadius = 4, Color = Colors.Gray, VerticalOptions = LayoutOptions.Center };
    private readonly BoxView _swatch2 = new() { WidthRequest = 22, HeightRequest = 22, CornerRadius = 4, Color = Colors.Gray, VerticalOptions = LayoutOptions.Center };
    private readonly Entry _name1 = new() { WidthRequest = 170, TextColor = Colors.White, BackgroundColor = Panel, Placeholder = "1. csapat neve" };
    private readonly Entry _name2 = new() { WidthRequest = 170, TextColor = Colors.White, BackgroundColor = Panel, Placeholder = "2. csapat neve" };
    private readonly Label _engineInfo = new() { TextColor = Color.FromArgb("#90CAF9"), FontSize = 12 };
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
    private readonly Button _labelPackageButton = new() { Text = "📦 Címkéző csomag (laptopra)", BackgroundColor = Color.FromArgb("#5E35B1") };
    private readonly Button _importLabelsButton = new() { Text = "📥 Címkék beolvasása", BackgroundColor = Color.FromArgb("#3949AB") };
    private readonly Button _tuneButton = new() { Text = "🧠 Tanulás a jelöléseimből", BackgroundColor = Color.FromArgb("#00838F") };
    private readonly Button _openFolderButton = new() { Text = "📂 Mappa megnyitása", BackgroundColor = Color.FromArgb("#455A64"), IsVisible = false };
    private readonly Label _learnStatus = new() { TextColor = Muted, VerticalOptions = LayoutOptions.Center, MaxLines = 4, WidthRequest = 520 };
    private string? _learnJobId;
    private string? _learnFolder;
    private readonly MediaElement _player = new() { ShouldAutoPlay = false, ShouldShowPlaybackControls = true, Aspect = Aspect.AspectFit, HeightRequest = 360, BackgroundColor = Colors.Black };
    private readonly HorizontalStackLayout _videoButtons = new() { Spacing = 8 };
    private readonly HorizontalStackLayout _heatmapStrip = new() { Spacing = 10 };
    private readonly VerticalStackLayout _resultBox = new() { Spacing = 10, IsVisible = false };
    private string? _playerSource;

    /// <summary>Kalap-szűrő: (kulcs vagy null = mind, felirat) – a betöltött események darabszámával.</summary>
    private List<(string? Key, string Label)> _categoryOptions = new() { (null, "Minden kalap") };
    /// <summary>Típus-szűrő a kiválasztott kalapon belül (passznál saját / ellenfél térfél külön is).</summary>
    private List<(string Label, Func<MatchEvent, bool> Match)> _typeOptions = new() { ("Minden típus", _ => true) };
    private readonly Picker _categoryFilter = new() { Title = "Kalap", TextColor = Colors.White, WidthRequest = 230 };
    private readonly CheckBox _showExperimental = new() { IsChecked = false };
    private readonly CheckBox _recommended = new() { IsChecked = false };
    private readonly Label _recommendedInfo = new() { TextColor = Muted, FontSize = 12, VerticalOptions = LayoutOptions.Center };

    public MatchAnalysisPage()
    {
        Title = "Meccs elemzés";
        BackgroundColor = Bg;

        RebuildCategoryOptions();
        _categoryFilter.SelectedIndexChanged += (_, _) => { RebuildTypeOptions(); ApplyFilter(); };
        _showExperimental.CheckedChanged += (_, _) => { RebuildCategoryOptions(); ApplyFilter(); };
        _recommended.CheckedChanged += (_, _) => ApplyFilter();
        _teamFilter.ItemsSource = new List<string> { "Mindkét csapat", TeamNames.Team1, TeamNames.Team2 };
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
        _labelPackageButton.Clicked += async (_, _) => await StartLearnJobAsync("package");
        _importLabelsButton.Clicked += async (_, _) => await StartLearnJobAsync("import");
        _tuneButton.Clicked += async (_, _) => await StartLearnJobAsync("tune");
        _openFolderButton.Clicked += (_, _) => OpenFolder(_learnFolder);

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
                Check(_background, "Háttér-mód (a gép közben használható marad)"),
                Check(_makeFullMatch, "Szünet nélküli meccsvideó készítése"),
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
        _name1.Completed += (_, _) => ApplyTeamNames();
        _name1.Unfocused += (_, _) => ApplyTeamNames();
        _name2.Completed += (_, _) => ApplyTeamNames();
        _name2.Unfocused += (_, _) => ApplyTeamNames();
        _resultBox.Children.Add(Row(
            new Label { Text = "Csapatok (mezszín szerint):", TextColor = Colors.White, VerticalOptions = LayoutOptions.Center },
            _swatch1, _name1, _swatch2, _name2));
        _resultBox.Children.Add(new Label
        {
            Text = "A program a két csapatot mezszín alapján választja szét; az 1. csapat az, amelyik az 1. félidőben a bal oldali " +
                   "kaput védi. Nevezd el őket (pl. Mi / Ellenfél) – az eseménylista és a hőtérképek ezt a nevet mutatják.",
            TextColor = Muted, FontSize = 12,
        });
        _resultBox.Children.Add(_warnings);
        _resultBox.Children.Add(Row(board, drawOnVideo, review));
        _resultBox.Children.Add(_player);
        var addMissing = new Button { Text = "➕ Hiányzó esemény ide (a videó mostani pillanatára)", BackgroundColor = Color.FromArgb("#2E7D32") };
        addMissing.Clicked += async (_, _) => await AddMissingHereAsync();
        _resultBox.Children.Add(addMissing);
        _resultBox.Children.Add(new Label
        {
            Text = "Ellenőrzés: a ✓ / ✗ gombokkal jelöld, melyik esemény valódi és melyik téves; ami kimaradt, azt a videón " +
                   "odatekerve a ➕ gombbal vedd fel. A jelölések mentődnek, és ezekből mérjük, javul-e a felismerés.",
            TextColor = Muted, FontSize = 12,
        });
        _resultBox.Children.Add(new ScrollView { Orientation = ScrollOrientation.Horizontal, Content = _videoButtons });
        _resultBox.Children.Add(Header("Események"));
        _resultBox.Children.Add(Row(_categoryFilter, _typeFilter, _teamFilter, _halfFilter));
        _resultBox.Children.Add(Check(_showExperimental,
            "Kísérleti események is (sikertelen passz, szerelés, labdaszerzés, beadás, labdavezetés – még bizonytalan felismerés)"));
        _resultBox.Children.Add(Row(_recommended, new Label
        {
            Text = "Csak az ajánlottak, a legfontosabb elöl (a ⭐ jelöléseidből tanulva)", TextColor = Colors.White,
            VerticalOptions = LayoutOptions.Center,
        }, _recommendedInfo));
        _resultBox.Children.Add(Row(selectAll, selectNone, _eventCount));
        _resultBox.Children.Add(_eventList);
        _resultBox.Children.Add(_exportButton);
        _resultBox.Children.Add(_exportStatus);
        _resultBox.Children.Add(Header("Tanulás a jelöléseidből"));
        _resultBox.Children.Add(new Label
        {
            Text = "📦 A címkéző csomag rövid részleteket és egy weboldalt készít, amivel laptopon (internet és telepítés " +
                   "nélkül) végignézheted a jeleneteket – a program tippjeit, amit majdnem jelzett, és véletlen részleteket. " +
                   "📥 A mentett cimkek_*.json fájlt itt olvasd vissza. 🧠 A tanulás az összes eddigi jelölésből " +
                   "(✓ / ✗ / ➕ / címkéző) beállítja a felismerést; a következő elemzés már így fut.",
            TextColor = Muted, FontSize = 12,
        });
        _resultBox.Children.Add(Row(_labelPackageButton, _importLabelsButton, _tuneButton));
        _resultBox.Children.Add(Row(_learnStatus, _openFolderButton));
        _resultBox.Children.Add(Header("Hőtérképek"));
        _resultBox.Children.Add(new ScrollView { Orientation = ScrollOrientation.Horizontal, Content = _heatmapStrip });

        var right = new VerticalStackLayout
        {
            Spacing = 8,
            Padding = 14,
            Children = { _engineInfo, _jobTitle, _jobPhase, _jobProgress, Row(_pauseButton, _cancelButton), _resultBox }
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

        // Edzői ellenőrzés: ✓ valódi / ✗ téves (újra megnyomva visszavonja). Kézzel felvett eseménynél a ✗ törli.
        var ok = new Button { Text = "✓", WidthRequest = 40, HeightRequest = 34, Padding = 0, FontSize = 16, TextColor = Colors.White };
        ok.SetBinding(BackgroundColorProperty, nameof(EventItem.OkColor));
        ok.SetBinding(IsVisibleProperty, nameof(EventItem.IsManual), converter: new InvertBool());
        ok.Clicked += async (s, _) => { if ((s as BindableObject)?.BindingContext is EventItem it) await SetVerdictAsync(it, "ok"); };
        var wrong = new Button { Text = "✗", WidthRequest = 40, HeightRequest = 34, Padding = 0, FontSize = 16, TextColor = Colors.White };
        wrong.SetBinding(BackgroundColorProperty, nameof(EventItem.WrongColor));
        wrong.Clicked += async (s, _) => { if ((s as BindableObject)?.BindingContext is EventItem it) await SetVerdictAsync(it, "hamis"); };
        var star = new Button { WidthRequest = 40, HeightRequest = 34, Padding = 0, FontSize = 16, TextColor = Colors.White };
        star.SetBinding(Button.TextProperty, nameof(EventItem.StarText));
        star.SetBinding(BackgroundColorProperty, nameof(EventItem.StarColor));
        star.Clicked += async (s, _) => { if ((s as BindableObject)?.BindingContext is EventItem it) await ToggleImportantAsync(it); };
        ToolTipProperties.SetText(star, "Fontos – ezt a jelenetet látni akarom a válogatásban");
        ToolTipProperties.SetText(ok, "Valódi esemény (helyes)");
        ToolTipProperties.SetText(wrong, "Téves esemény (nem történt meg / nem ez volt)");

        var grid = new Grid
        {
            ColumnDefinitions = { new ColumnDefinition(GridLength.Auto), new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Auto) },
            Padding = new Thickness(4, 3),
            Margin = new Thickness(0, 2),
        };
        grid.SetBinding(BackgroundColorProperty, nameof(EventItem.RowColor));
        grid.Add(box, 0, 0);
        grid.Add(new VerticalStackLayout { Spacing = 1, Children = { title, sub } }, 1, 0);
        grid.Add(new HorizontalStackLayout { Spacing = 4, VerticalOptions = LayoutOptions.Center, Children = { star, ok, wrong } }, 2, 0);
        return grid;
    }

    private sealed class InvertBool : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo culture) => value is not true;
        public object? ConvertBack(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo culture) => value is not true;
    }

    private async Task SetVerdictAsync(EventItem item, string verdict)
    {
        if (_loadedEventsUrl is null) return;
        try
        {
            if (item.IsManual)
            {
                if (!await DisplayAlertAsync("Törlés", "Törlöd ezt a kézzel felvett eseményt?", "Igen", "Nem")) return;
                await _api.RemoveMissingAsync(_loadedEventsUrl, item.ManualId!);
                _allEvents.Remove(item);
                ApplyFilter();
                return;
            }
            var newVerdict = item.Verdict == verdict ? null : verdict;   // újra megnyomva: visszavonás
            await _api.SetVerdictAsync(_loadedEventsUrl, item.Event, newVerdict);
            item.Verdict = newVerdict;
            UpdateEventCount();
        }
        catch (Exception ex)
        {
            _eventCount.Text = $"A jelölés nem menthető: {ex.Message}";
        }
    }

    /// <summary>Az ajánlottság betöltése (a ⭐ jelölésekből tanult modell); hiba / kevés jelölés esetén üres.</summary>
    private async Task LoadImportanceAsync(string url)
    {
        try
        {
            var (scores, message) = await _api.GetImportanceAsync(url);
            foreach (var it in _allEvents)
                it.Score = scores is not null && scores.TryGetValue(it.Event.FeedbackKey, out var sc) ? sc : null;
            _recommended.IsEnabled = scores is not null;
            if (scores is null) _recommended.IsChecked = false;
            _recommendedInfo.Text = scores is null ? (message ?? "") : "";
        }
        catch
        {
            _recommended.IsEnabled = false;
        }
    }

    private async Task ToggleImportantAsync(EventItem item)
    {
        if (_loadedEventsUrl is null) return;
        try
        {
            await _api.SetImportantAsync(_loadedEventsUrl, item.Event, item.ManualId, !item.Fontos);
            item.Fontos = !item.Fontos;
            UpdateEventCount();
            await LoadImportanceAsync(_loadedEventsUrl);   // a modell minden jelöléssel tanul
        }
        catch (Exception ex)
        {
            _eventCount.Text = $"A jelölés nem menthető: {ex.Message}";
        }
    }

    private async Task AddMissingHereAsync()
    {
        if (_loadedEventsUrl is null) return;
        var video = _result?.VideoPath ?? _currentJob?.VideoPath;
        if (string.IsNullOrEmpty(video) || _playerSource != video)
        {
            await DisplayAlertAsync("Hiányzó esemény",
                "Előbb kattints egy eseményre a listában (így az eredeti meccsvideó töltődik be), majd tekerj oda, " +
                "ahol a kimaradt esemény történt, és nyomd meg újra ezt a gombot.", "OK");
            return;
        }
        _player.Pause();
        var t = _player.Position.TotalSeconds;
        var types = EventLabels.All;
        var choice = await DisplayActionSheetAsync($"Mi történt itt? ({MatchEvent.FormatTime(t)})", "Mégse", null,
            types.Select(x => x.Label).ToArray());
        var picked = types.FirstOrDefault(x => x.Label == choice);
        if (string.IsNullOrEmpty(picked.Type)) return;
        var teamChoice = await DisplayActionSheetAsync("Melyik csapat?", "Mégse", null, TeamNames.Team1, TeamNames.Team2, "Nem tudom");
        if (teamChoice is null || teamChoice == "Mégse") return;
        var team = teamChoice == TeamNames.Team1 ? "team1" : teamChoice == TeamNames.Team2 ? "team2" : null;
        try
        {
            var item = await _api.AddMissingAsync(_loadedEventsUrl, picked.Type, t, team);
            var ev = new MatchEvent { Type = item.Type, Time = item.Time, Team = item.Team, Desc = "az automatikából kimaradt" };
            _allEvents.Add(new EventItem(ev, item.Id));
            _allEvents.Sort((a, b) => a.Event.Time.CompareTo(b.Event.Time));
            ApplyFilter();
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync("Hiba", ex.Message, "OK");
        }
    }

    // ================================================================ életciklus

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        _timer ??= CreateTimer();
        _timer.Start();
        await RefreshEngineInfoAsync();
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
            if (_learnJobId is not null)
                await PollLearnAsync();
            if (++_tick % 5 == 0 || _jobs.Any(j => !j.Job.IsFinished))
                await RefreshJobsAsync();
            if (_tick % 5 == 0)
                await RefreshEngineInfoAsync();
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
            Team1AttacksRight = true,   // automatikus csapatok: a "team1" mindig az 1. félidőben balról jobbra támadó
            SkipCut = !_makeFullMatch.IsChecked,
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

    private async Task RefreshEngineInfoAsync()
    {
        try { _engineInfo.Text = await _api.GetEngineSummaryAsync(); }
        catch (Exception ex) { _engineInfo.Text = $"🧠 A motor állapota nem kérdezhető le: {ex.Message}"; }
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
            try
            {
                var fb = await _api.GetFeedbackAsync(url);
                var byKey = fb.Items.GroupBy(i => i.Key).ToDictionary(g => g.Key, g => g.Last());
                foreach (var it in _allEvents)
                    if (byKey.TryGetValue(it.Event.FeedbackKey, out var v))
                    {
                        it.Verdict = string.IsNullOrEmpty(v.Verdict) ? null : v.Verdict;
                        it.Fontos = v.Fontos == true;
                    }
                foreach (var m in fb.Missing)
                    _allEvents.Add(new EventItem(new MatchEvent { Type = m.Type, Time = m.Time, Team = m.Team,
                        Desc = string.IsNullOrEmpty(m.Note) ? "az automatikából kimaradt" : m.Note }, m.Id) { Fontos = m.Fontos == true });
                _allEvents.Sort((a, b) => a.Event.Time.CompareTo(b.Event.Time));
            }
            catch { /* régebbi motor: nincs /feedback – a lista ítéletek nélkül is használható */ }
            LoadTeamNames();
            await LoadImportanceAsync(url);
            RebuildCategoryOptions();
            ApplyFilter();
        }
        catch (Exception ex)
        {
            _eventCount.Text = $"Az események nem tölthetők be: {ex.Message}";
        }
    }

    private string TeamNameKey(int n) => $"csapatnev_{_result?.RunId ?? _currentJob?.Id}_{n}";

    /// <summary>A mezszínek és a mentett csapatnevek betöltése az eseménylista betöltése után.</summary>
    private void LoadTeamNames()
    {
        static Color ToColor((double R, double G, double B) c) =>
            Color.FromRgb((int)Math.Clamp(c.R, 0, 255), (int)Math.Clamp(c.G, 0, 255), (int)Math.Clamp(c.B, 0, 255));
        var tc = _api.LastTeamColors;
        _swatch1.Color = tc.TryGetValue("team1", out var c1) ? ToColor(c1) : Colors.Gray;
        _swatch2.Color = tc.TryGetValue("team2", out var c2) ? ToColor(c2) : Colors.Gray;
        _name1.Text = Settings.Get(TeamNameKey(1)) ?? "";
        _name2.Text = Settings.Get(TeamNameKey(2)) ?? "";
        ApplyTeamNames(save: false);
    }

    private void ApplyTeamNames(bool save = true)
    {
        var n1 = string.IsNullOrWhiteSpace(_name1.Text) ? "1. csapat" : _name1.Text.Trim();
        var n2 = string.IsNullOrWhiteSpace(_name2.Text) ? "2. csapat" : _name2.Text.Trim();
        if (save)
        {
            Settings.Set(TeamNameKey(1), _name1.Text?.Trim());
            Settings.Set(TeamNameKey(2), _name2.Text?.Trim());
        }
        if (n1 == TeamNames.Team1 && n2 == TeamNames.Team2 && save) return;
        TeamNames.Team1 = n1;
        TeamNames.Team2 = n2;
        var sel = Math.Max(0, _teamFilter.SelectedIndex);
        _teamFilter.ItemsSource = new List<string> { "Mindkét csapat", n1, n2 };
        _teamFilter.SelectedIndex = sel;
        foreach (var e in _allEvents) e.Refresh();
        if (_result is not null) ShowHeatmaps(_result);
    }

    private bool PassesExperimental(EventItem item) =>
        _showExperimental.IsChecked || !item.Event.Experimental || item.IsManual || item.Verdict is not null;

    /// <summary>A kalap-választó újraépítése a betöltött események darabszámával (az üres kalapok is látszanak).</summary>
    private void RebuildCategoryOptions()
    {
        var prev = _categoryOptions.ElementAtOrDefault(Math.Max(0, _categoryFilter.SelectedIndex)).Key;
        var shown = _allEvents.Where(PassesExperimental).ToList();
        _categoryOptions = new() { (null, $"Minden kalap ({shown.Count})") };
        foreach (var (key, label) in EventCategories.All)
        {
            var n = shown.Count(e => e.Event.Category == key);
            if (n > 0 || key != "egyeb") _categoryOptions.Add((key, $"{label} ({n})"));
        }
        _categoryFilter.ItemsSource = _categoryOptions.Select(c => c.Label).ToList();
        _categoryFilter.SelectedIndex = Math.Max(0, _categoryOptions.FindIndex(c => c.Key == prev));
        RebuildTypeOptions();
    }

    private void RebuildTypeOptions()
    {
        var cat = _categoryOptions.ElementAtOrDefault(Math.Max(0, _categoryFilter.SelectedIndex)).Key;
        var prev = _typeOptions.ElementAtOrDefault(Math.Max(0, _typeFilter.SelectedIndex)).Label;
        _typeOptions = new() { ("Minden típus", _ => true) };
        var types = _allEvents.Where(PassesExperimental).Select(e => e.Event)
            .Where(e => cat is null || e.Category == cat).Select(e => e.Type).Distinct()
            .OrderBy(t => EventLabels.Label(t)).ToList();
        foreach (var t in types)
        {
            var tt = t;
            _typeOptions.Add(($"{EventLabels.Icon(tt)} {EventLabels.Label(tt)}", e => e.Type == tt));
            if (tt is "pass" or "pass_failed")
            {
                _typeOptions.Add(($"   {EventLabels.Label(tt)} – saját térfélen", e => e.Type == tt && e.Zone == "sajat_terfel"));
                _typeOptions.Add(($"   {EventLabels.Label(tt)} – ellenfél térfelén", e => e.Type == tt && e.Zone == "ellenfel_terfel"));
            }
        }
        _typeFilter.ItemsSource = _typeOptions.Select(o => o.Label).ToList();
        _typeFilter.SelectedIndex = Math.Max(0, _typeOptions.FindIndex(o => o.Label == prev));
    }

    private void ApplyFilter()
    {
        var cat = _categoryOptions.ElementAtOrDefault(Math.Max(0, _categoryFilter.SelectedIndex)).Key;
        var type = _typeOptions.ElementAtOrDefault(Math.Max(0, _typeFilter.SelectedIndex)).Match ?? (_ => true);
        var team = _teamFilter.SelectedIndex switch { 1 => "team1", 2 => "team2", _ => null };
        var half = _halfFilter.SelectedIndex switch { 1 => 1, 2 => 2, _ => 0 };
        _visibleEvents.Clear();
        foreach (var item in _allEvents)
        {
            var e = item.Event;
            if (!PassesExperimental(item)) continue;
            if (cat is not null && e.Category != cat) continue;
            if (!type(e)) continue;
            if (team is not null && e.Team != team) continue;
            if (half != 0 && e.Half != half) continue;
            if (_recommended.IsChecked && (item.Score ?? 0) < 0.5) continue;
            _visibleEvents.Add(item);
        }
        if (_recommended.IsChecked)
        {
            var sorted = _visibleEvents.OrderByDescending(i => i.Score ?? 0).ToList();
            _visibleEvents.Clear();
            foreach (var i in sorted) _visibleEvents.Add(i);
        }
        UpdateEventCount();
    }

    private void UpdateEventCount()
    {
        var selected = _allEvents.Count(e => e.Selected);
        var okN = _allEvents.Count(e => e.Verdict == "ok");
        var badN = _allEvents.Count(e => e.Verdict == "hamis");
        var manN = _allEvents.Count(e => e.IsManual);
        var starN = _allEvents.Count(e => e.Fontos);
        _eventCount.Text = $"{_visibleEvents.Count} / {_allEvents.Count} esemény látszik, {selected} kijelölve" +
            (okN + badN + manN > 0 ? $"  •  ellenőrizve: {okN} ✓, {badN} ✗, {manN} kézzel felvett" : "") +
            (starN > 0 ? $"  •  {starN} ⭐" : "");
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

    // ================================================================ tanulás (címkéző csomag, beolvasás, hangolás)

    private void SetLearnBusy(bool busy)
    {
        _labelPackageButton.IsEnabled = _importLabelsButton.IsEnabled = _tuneButton.IsEnabled = !busy;
    }

    private async Task StartLearnJobAsync(string what)
    {
        try
        {
            _openFolderButton.IsVisible = false;
            switch (what)
            {
                case "package":
                    if (_loadedEventsUrl is null) return;
                    SetLearnBusy(true);
                    _learnStatus.Text = "Címkéző csomag készül…";
                    _learnJobId = await _api.StartLabelPackageAsync(_loadedEventsUrl);
                    break;
                case "import":
                    var pick = await FilePicker.Default.PickAsync(new PickOptions
                    {
                        PickerTitle = "A laptopon mentett cimkek_*.json",
                        FileTypes = new FilePickerFileType(new Dictionary<DevicePlatform, IEnumerable<string>>
                        {
                            [DevicePlatform.WinUI] = new[] { ".json" },
                            [DevicePlatform.MacCatalyst] = new[] { "public.json" },
                        }),
                    });
                    if (pick is null) return;
                    var tune = await DisplayAlertAsync("Címkék beolvasása",
                        "A beolvasás után tanuljon is a program az összes eddigi jelölésből? (Ez néhány perc lehet.)",
                        "Igen", "Csak beolvasás");
                    SetLearnBusy(true);
                    _learnStatus.Text = "Címkék beolvasása…";
                    _learnJobId = await _api.StartImportLabelsAsync(pick.FullPath, tune);
                    break;
                default:
                    SetLearnBusy(true);
                    _learnStatus.Text = "Tanulás az eddigi jelölésekből…";
                    _learnJobId = await _api.StartTuneAsync();
                    break;
            }
        }
        catch (Exception ex)
        {
            _learnStatus.Text = $"Hiba: {ex.Message}";
            SetLearnBusy(false);
        }
    }

    private async Task PollLearnAsync()
    {
        var job = await _api.GetJobAsync(_learnJobId!);
        if (!job.IsFinished)
        {
            _learnStatus.Text = $"{job.Progress:0}% – {job.Phase}";
            return;
        }
        _learnJobId = null;
        SetLearnBusy(false);
        if (job.Status != "done" || job.Result is not { ValueKind: System.Text.Json.JsonValueKind.Object } r)
        {
            _learnStatus.Text = $"Nem sikerült: {job.Error ?? job.Status}";
            return;
        }
        _learnStatus.Text = DescribeLearnResult(job.Kind, r);
        if (job.Kind == "label-package" && r.TryGetProperty("folder", out var f) && f.GetString() is { Length: > 0 } folder)
        {
            _learnFolder = folder;
            _openFolderButton.IsVisible = true;
        }
        if (job.Kind == "import-labels" && _loadedEventsUrl is not null)
            await LoadEventsAsync(_loadedEventsUrl);   // a beolvasott ítéletek a listában is látszanak
    }

    private static string Str(System.Text.Json.JsonElement r, string name) =>
        r.TryGetProperty(name, out var v) && v.ValueKind == System.Text.Json.JsonValueKind.String ? v.GetString() ?? "" : "";

    /// <summary>A tanulás-feladatok eredménye egy-két mondatban.</summary>
    internal static string DescribeLearnResult(string kind, System.Text.Json.JsonElement r)
    {
        if (Str(r, "status") == "error") return "Hiba: " + Str(r, "message");
        switch (kind)
        {
            case "label-package":
                var n = r.TryGetProperty("clips", out var c) && c.TryGetInt32(out var ci) ? ci : 0;
                return $"Kész: {n} jelenet. Másold a mappát a laptopra (pendrive / OneDrive), és nyisd meg benne az " +
                       $"index.html-t. A végén a „Mentés fájlba” gombbal mentett fájlt itt olvasd vissza. Mappa: {Str(r, "folder")}";
            case "import-labels":
                var text = Str(r, "message");
                if (r.TryGetProperty("tuning", out var t) && t.ValueKind == System.Text.Json.JsonValueKind.Object)
                    text += "  " + DescribeLearnResult("tune-events", t);
                return text;
            default:
                var st = Str(r, "status");
                if (st != "success") return Str(r, "message");
                var changed = r.TryGetProperty("overrides", out var o) && o.ValueKind == System.Text.Json.JsonValueKind.Object
                    ? o.EnumerateObject().Count() : 0;
                return $"Tanulás kész – {Str(r, "message")}. " +
                       (changed > 0 ? $"{changed} beállítás változott; a következő elemzés már ezekkel fut."
                                    : "Az eddigi beállítások maradnak (ezek a legjobbak a jelöléseidre).");
        }
    }

    private static void OpenFolder(string? folder)
    {
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder)) return;
        try
        {
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo { FileName = folder, UseShellExecute = true });
        }
        catch
        {
            // ha a rendszer nem tud mappát nyitni, az útvonal a szövegben is ott van
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

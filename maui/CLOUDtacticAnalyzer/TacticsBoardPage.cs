using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

/// <summary>
/// A taktikai tábla (felállás, rajzolás, felülnézeti visszajátszás, rajzolás a videóra, kép/videó mentés).
/// A tábla a motor weboldala (/tabla) – egy WebView-ban jelenik meg, így a böngészős változattal
/// mindig azonos. Egy elemzett meccs a "?events=/clips/match_events_….json" paraméterrel nyílik meg.
/// Megnyitás máshonnan: Shell.Current.GoToAsync($"//TacticsBoardPage?events={Uri.EscapeDataString(url)}"),
/// rajzolás a videón: …?video=/clips/match_….mp4&amp;events=… (a tábla videós módja nyílik meg).
/// </summary>
public class TacticsBoardPage : ContentPage, IQueryAttributable
{
    private readonly WebView _web = new();
    private readonly Label _info = new() { TextColor = Colors.LightGray, VerticalOptions = LayoutOptions.Center, LineBreakMode = LineBreakMode.TailTruncation };
    private string? _eventsUrl;
    private string? _videoUrl;
    private string? _loadedUrl;

    public TacticsBoardPage()
    {
        Title = "Taktikai tábla";
        BackgroundColor = Color.FromArgb("#121212");

        var empty = new Button { Text = "Üres tábla", BackgroundColor = Color.FromArgb("#455A64") };
        empty.Clicked += (_, _) => { _eventsUrl = null; _videoUrl = null; Load(force: true); };
        var reload = new Button { Text = "↻ Újratöltés", BackgroundColor = Color.FromArgb("#455A64") };
        reload.Clicked += (_, _) => Load(force: true);
        var browser = new Button { Text = "Megnyitás böngészőben", BackgroundColor = Color.FromArgb("#455A64") };
        browser.Clicked += async (_, _) => await Launcher.Default.OpenAsync(new Uri(CurrentUrl()));

        var bar = new HorizontalStackLayout { Spacing = 8, Padding = new Thickness(8, 6), Children = { empty, reload, browser, _info } };
        var grid = new Grid { RowDefinitions = { new RowDefinition(GridLength.Auto), new RowDefinition(GridLength.Star) } };
        grid.Add(bar, 0, 0);
        grid.Add(_web, 0, 1);
        Content = grid;
    }

    public void ApplyQueryAttributes(IDictionary<string, object> query)
    {
        _eventsUrl = Arg(query, "events");
        _videoUrl = Arg(query, "video");
        Load(force: false);
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        Load(force: false);
    }

    private static string? Arg(IDictionary<string, object> query, string key) =>
        query.TryGetValue(key, out var v) && v is string s && !string.IsNullOrWhiteSpace(s) ? Uri.UnescapeDataString(s) : null;

    private string CurrentUrl()
    {
        var args = new List<string>();
        if (!string.IsNullOrEmpty(_videoUrl)) args.Add("video=" + Uri.EscapeDataString(_videoUrl));
        if (!string.IsNullOrEmpty(_eventsUrl)) args.Add("events=" + Uri.EscapeDataString(_eventsUrl));
        return Engine.Url("/tabla" + (args.Count > 0 ? "?" + string.Join("&", args) : ""));
    }

    private void Load(bool force)
    {
        var url = CurrentUrl();
        if (!force && url == _loadedUrl) return;
        _loadedUrl = url;
        _info.Text = !string.IsNullOrEmpty(_videoUrl) ? $"Videó: {Path.GetFileName(_videoUrl)}"
                   : !string.IsNullOrEmpty(_eventsUrl) ? $"Meccs: {Path.GetFileName(_eventsUrl)}" : "Üres tábla";
        _web.Source = new UrlWebViewSource { Url = url };
    }
}

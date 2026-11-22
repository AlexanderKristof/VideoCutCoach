using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

/// <summary>
/// Indító képernyő: megkeresi / elindítja az elemző motort (EngineHost), és csak utána jön a bejelentkezés.
/// Hiba esetén megmutatja az okot, és megadható a motor mappája (a Python forrás, ahol a main.py van)
/// vagy a motor címe (pl. telefonról: http://192.168.1.20:8000).
/// </summary>
public class StartupPage : ContentPage
{
    private readonly Label _status = new() { TextColor = Colors.LightGray, HorizontalTextAlignment = TextAlignment.Center };
    private readonly ActivityIndicator _spinner = new() { IsRunning = true, Color = Colors.White };
    private readonly Label _error = new() { TextColor = Color.FromArgb("#FF8A80"), IsVisible = false };
    private readonly VerticalStackLayout _settings = new() { Spacing = 8, IsVisible = false };
    private readonly Entry _engineDir = new() { Placeholder = @"pl. E:\ASZTALRA\Edzesterv\TacAnalyCLOUD", TextColor = Colors.White, BackgroundColor = Color.FromArgb("#1E1E1E") };
    private readonly Entry _pythonExe = new() { Placeholder = "python.exe (üresen: automatikus)", TextColor = Colors.White, BackgroundColor = Color.FromArgb("#1E1E1E") };
    private readonly Entry _engineUrl = new() { Placeholder = EngineHost.DefaultUrl, TextColor = Colors.White, BackgroundColor = Color.FromArgb("#1E1E1E") };
    private bool _started;

    public StartupPage()
    {
        Title = "TacAnaly";
        BackgroundColor = Color.FromArgb("#121212");

        var retry = new Button { Text = "Újrapróbálás", BackgroundColor = Color.FromArgb("#2196F3") };
        retry.Clicked += async (_, _) => await SaveAndRetryAsync();

        _settings.Children.Add(new Label { Text = "A motor (Python) mappája – ahol a main.py és az engine_launcher.py van:", TextColor = Colors.White });
        _settings.Children.Add(_engineDir);
        _settings.Children.Add(new Label { Text = "Python program (ha nem a PATH-on lévőt kell használni):", TextColor = Colors.White });
        _settings.Children.Add(_pythonExe);
        _settings.Children.Add(new Label { Text = "Vagy egy már futó motor címe:", TextColor = Colors.White });
        _settings.Children.Add(_engineUrl);
        _settings.Children.Add(retry);

        Content = new ScrollView
        {
            Content = new VerticalStackLayout
            {
                Padding = 30,
                Spacing = 14,
                VerticalOptions = LayoutOptions.Center,
                MaximumWidthRequest = 720,
                Children =
                {
                    new Label { Text = "⚽ TacAnaly", FontSize = 28, FontAttributes = FontAttributes.Bold, TextColor = Colors.White, HorizontalOptions = LayoutOptions.Center },
                    _spinner,
                    _status,
                    _error,
                    _settings,
                }
            }
        };
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        if (_started) return;
        _started = true;
        await StartEngineAsync();
    }

    private async Task StartEngineAsync()
    {
        _spinner.IsRunning = _spinner.IsVisible = true;
        _error.IsVisible = false;
        _settings.IsVisible = false;
        try
        {
            var desc = await Engine.Host.StartAsync(new Progress<string>(s => _status.Text = s));
            _status.Text = $"Motor: {desc}";
            if (Application.Current?.Windows.Count > 0)
                Application.Current.Windows[0].Page = new NavigationPage(new LoginPage());
        }
        catch (Exception ex)
        {
            _spinner.IsRunning = _spinner.IsVisible = false;
            _status.Text = "Az elemző motor nem indult el.";
            _error.Text = ex.Message;
            _error.IsVisible = true;
            _engineDir.Text = Settings.Get("engine_dir") ?? EngineHost.FindEngineSourceDir() ?? "";
            _pythonExe.Text = Settings.Get("python_exe") ?? "";
            _engineUrl.Text = Settings.Get("engine_url") ?? "";
            _settings.IsVisible = true;
        }
    }

    private async Task SaveAndRetryAsync()
    {
        Settings.Set("engine_dir", _engineDir.Text?.Trim());
        Settings.Set("python_exe", _pythonExe.Text?.Trim());
        Settings.Set("engine_url", _engineUrl.Text?.Trim().TrimEnd('/'));
        Engine.Host.Reconfigure();
        await StartEngineAsync();
    }
}

namespace TacticalAnalyzer;

public partial class App : Application
{
    public App()
    {
        InitializeComponent();
    }

    protected override Window CreateWindow(IActivationState? activationState)
    {
        // Először az indító képernyő: megkeresi / elindítja az elemző motort, utána jön a
        // bejelentkezés (a LoginPage maga cseréli AppShell-re, ha sikeres vagy már van token).
        var window = new Window(new StartupPage()) { Title = "TacAnaly" };
        window.Destroying += (_, _) =>
        {
            // Ha a motort mi indítottuk, kecsesen leállítjuk (a futó elemzés "megszakadt" lesz).
            // Task.Run: a UI szálon blokkoló várakozás ne akadjon össze az await-ekkel.
            try { Task.Run(() => Services.Engine.Host.StopAsync()).Wait(TimeSpan.FromSeconds(20)); }
            catch { /* kilépéskor nem baj */ }
        };
        return window;
    }
}

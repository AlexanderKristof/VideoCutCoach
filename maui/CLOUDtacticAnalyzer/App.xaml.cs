namespace TacticalAnalyzer;

public partial class App : Application
{
    public App()
    {
        InitializeComponent();

        // TÖRÖLD vagy kommentezd ki ezt a sort:
        // MainPage = new AppShell(); 
    }

    // ADD HOZZÁ ezt a metódust az osztályon belülre:
    protected override Window CreateWindow(IActivationState? activationState)
    {
        // Előbb a bejelentkezés — a LoginPage saját maga cseréli le AppShell-re,
        // ha sikeres a login/regisztráció, vagy ha már van elmentett token.
        return new Window(new NavigationPage(new LoginPage()));
    }
}
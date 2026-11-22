using System;
using System.Threading.Tasks;
using Microsoft.Maui.Controls;
using TacticalAnalyzer.Services;

namespace TacticalAnalyzer;

public partial class LoginPage : ContentPage
{
    private readonly AuthService _authService = new();

    public LoginPage()
    {
        InitializeComponent();
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();

        // Ha már van elmentett, korábbi bejelentkezésből származó token, ne kérjünk
        // újra jelszót - lépjünk egyenesen az appba.
        var existingToken = await _authService.GetTokenAsync();
        if (!string.IsNullOrEmpty(existingToken))
        {
            SessionService.AuthToken = existingToken;
            GoToMainApp();
        }
    }

    private void SetBusy(bool busy)
    {
        LoadingIndicator.IsRunning = busy;
        LoadingIndicator.IsVisible = busy;
        LoginButton.IsEnabled = !busy;
        RegisterButton.IsEnabled = !busy;
    }

    private void ShowError(string message)
    {
        ErrorLabel.Text = message;
        ErrorLabel.IsVisible = true;
    }

    private async void OnLoginClicked(object? sender, EventArgs e) => await DoAuthAsync(isRegister: false);

    private async void OnRegisterClicked(object? sender, EventArgs e) => await DoAuthAsync(isRegister: true);

    private async Task DoAuthAsync(bool isRegister)
    {
        ErrorLabel.IsVisible = false;
        var email = EmailEntry.Text?.Trim() ?? "";
        var password = PasswordEntry.Text ?? "";

        if (string.IsNullOrEmpty(email) || string.IsNullOrEmpty(password))
        {
            ShowError("Add meg az email címet és a jelszót!");
            return;
        }

        SetBusy(true);
        try
        {
            bool success = isRegister
                ? await _authService.RegisterAsync(email, password)
                : await _authService.LoginAsync(email, password);

            if (!success)
            {
                ShowError(isRegister
                    ? "Sikertelen regisztráció (talán már foglalt ez az email cím?)."
                    : "Hibás email cím vagy jelszó.");
                return;
            }

            GoToMainApp();
        }
        catch (Exception ex)
        {
            ShowError($"Hiba történt: {ex.Message}");
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void GoToMainApp()
    {
        if (Application.Current?.Windows.Count > 0)
        {
            Application.Current.Windows[0].Page = new AppShell();
        }
    }
}

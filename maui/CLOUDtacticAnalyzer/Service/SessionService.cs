namespace TacticalAnalyzer.Services;

/// <summary>
/// Egyszerű, app-szintű munkamenet-állapot. Bejelentkezés után ide kerül a JWT token,
/// hogy bármelyik service (pl. MatchService) automatikusan használni tudja anélkül,
/// hogy minden oldalon kézzel át kellene adogatni.
/// </summary>
public static class SessionService
{
    public static string? AuthToken { get; set; }

    public static bool IsLoggedIn => !string.IsNullOrEmpty(AuthToken);

    public static void Clear()
    {
        AuthToken = null;
    }
}

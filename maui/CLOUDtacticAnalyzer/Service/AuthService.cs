using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace TacticalAnalyzer.Services;

public class AuthService
{
    private readonly HttpClient _httpClient;
    private const string TokenKey = "auth_token";

    private static string BaseUrl => Engine.BaseUrl;

    /// <summary>Az utolsó sikertelen kérés oka a motor válaszából (magyar szöveg), vagy null.</summary>
    public string? LastError { get; private set; }

    public AuthService()
    {
        _httpClient = new HttpClient();
    }

    public async Task<bool> RegisterAsync(string email, string password)
    {
        LastError = null;
        var response = await _httpClient.PostAsJsonAsync($"{BaseUrl}/auth/register", new { email, password });
        if (!response.IsSuccessStatusCode) { LastError = await ErrorTextAsync(response); return false; }

        var result = await response.Content.ReadFromJsonAsync<AuthResponse>();
        if (result?.AccessToken == null) return false;

        await SecureStorage.Default.SetAsync(TokenKey, result.AccessToken);
        SessionService.AuthToken = result.AccessToken;
        return true;
    }

    public async Task<bool> LoginAsync(string email, string password)
    {
        var formData = new FormUrlEncodedContent(new[]
        {
            new KeyValuePair<string, string>("username", email),
            new KeyValuePair<string, string>("password", password)
        });

        LastError = null;
        var response = await _httpClient.PostAsync($"{BaseUrl}/auth/login", formData);
        if (!response.IsSuccessStatusCode) { LastError = await ErrorTextAsync(response); return false; }

        var result = await response.Content.ReadFromJsonAsync<AuthResponse>();
        if (result?.AccessToken == null) return false;

        await SecureStorage.Default.SetAsync(TokenKey, result.AccessToken);
        SessionService.AuthToken = result.AccessToken;
        return true;
    }

    /// <summary>A FastAPI hibaválasz "detail" mezője (szöveg, vagy a mezőhibák listája) olvasható formában.</summary>
    private static async Task<string> ErrorTextAsync(HttpResponseMessage response)
    {
        var code = (int)response.StatusCode;
        try
        {
            var text = await response.Content.ReadAsStringAsync();
            using var doc = System.Text.Json.JsonDocument.Parse(text);
            if (doc.RootElement.TryGetProperty("detail", out var d))
            {
                if (d.ValueKind == System.Text.Json.JsonValueKind.String) return d.GetString()!;
                if (d.ValueKind == System.Text.Json.JsonValueKind.Array)
                {
                    var msgs = d.EnumerateArray()
                        .Select(x => x.TryGetProperty("msg", out var m) ? m.GetString() : null)
                        .Where(m => !string.IsNullOrEmpty(m));
                    return $"Hibás adat: {string.Join("; ", msgs)}";
                }
            }
        }
        catch { /* nem JSON (pl. 500-as hibaoldal) */ }
        return code >= 500
            ? $"A motor belső hibát jelzett (HTTP {code}). Részletek a motor naplójában."
            : $"A motor elutasította a kérést (HTTP {code}).";
    }

    public async Task<string?> GetTokenAsync()
    {
        return await SecureStorage.Default.GetAsync(TokenKey);
    }

    public void Logout()
    {
        SecureStorage.Default.Remove(TokenKey);
        SessionService.Clear();
    }
}

public class AuthResponse
{
    [JsonPropertyName("access_token")]
    public string AccessToken { get; set; } = string.Empty;

    [JsonPropertyName("token_type")]
    public string TokenType { get; set; } = string.Empty;
}
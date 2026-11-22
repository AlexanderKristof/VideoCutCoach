using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace TacticalAnalyzer.Services;

public class AuthService
{
    private readonly HttpClient _httpClient;
    private const string TokenKey = "auth_token";

    private static string BaseUrl => Engine.BaseUrl;

    public AuthService()
    {
        _httpClient = new HttpClient();
    }

    public async Task<bool> RegisterAsync(string email, string password)
    {
        var response = await _httpClient.PostAsJsonAsync($"{BaseUrl}/auth/register", new { email, password });
        if (!response.IsSuccessStatusCode) return false;

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

        var response = await _httpClient.PostAsync($"{BaseUrl}/auth/login", formData);
        if (!response.IsSuccessStatusCode) return false;

        var result = await response.Content.ReadFromJsonAsync<AuthResponse>();
        if (result?.AccessToken == null) return false;

        await SecureStorage.Default.SetAsync(TokenKey, result.AccessToken);
        SessionService.AuthToken = result.AccessToken;
        return true;
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
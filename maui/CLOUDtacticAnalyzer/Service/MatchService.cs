using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text.Json;

namespace TacticalAnalyzer.Services;

public class MatchService
{
    private readonly HttpClient _httpClient;

#if ANDROID
    private const string BaseUrl = "http://10.0.2.2:8000";
#else
    private const string BaseUrl = "http://localhost:8000";
#endif

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public MatchService()
    {
        _httpClient = new HttpClient();
    }

    public void SetAuthToken(string token)
    {
        _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
    }

    /// <summary>
    /// Ha még nem állította be senki kézzel a tokent ezen a példányon, felveszi a
    /// bejelentkezéskor a SessionService-ben eltárolt tokent. Így nem kell minden
    /// oldalon manuálisan SetAuthToken()-t hívni a MatchService használata előtt.
    /// </summary>
    private void EnsureAuthHeader()
    {
        if (_httpClient.DefaultRequestHeaders.Authorization == null && !string.IsNullOrEmpty(SessionService.AuthToken))
        {
            SetAuthToken(SessionService.AuthToken);
        }
    }

    public async Task<MatchAnalysisDto?> GetMatchAnalysisAsync(string matchId)
    {
        EnsureAuthHeader();
        try
        {
            var response = await _httpClient.GetAsync($"{BaseUrl}/matches/{matchId}");

            if (response.IsSuccessStatusCode)
            {
                string json = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<MatchAnalysisDto>(json, JsonOptions);
            }
            else
            {
                Debug.WriteLine($"Szerver hiba (Match get): {response.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Meccs lekérési kivétel: {ex.Message}");
        }

        return null;
    }

    public async Task<bool> SaveMatchAnalysisAsync(MatchAnalysisDto matchData)
    {
        EnsureAuthHeader();
        try
        {
            string jsonPayload = JsonSerializer.Serialize(matchData, JsonOptions);
            var content = new StringContent(jsonPayload, System.Text.Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync($"{BaseUrl}/matches/save", content);

            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Meccs mentési kivétel: {ex.Message}");
            return false;
        }
    }
}
using System.Diagnostics;

namespace TacticalAnalyzer.Services;

public class VideoTrimmerService
{
    public async Task<double> GetVideoDurationSecondsAsync(string inputFilePath)
    {
#if WINDOWS
        try
        {
            if (string.IsNullOrEmpty(inputFilePath) || !File.Exists(inputFilePath))
                return 0;

            var storageFile = await Windows.Storage.StorageFile.GetFileFromPathAsync(inputFilePath);
            var properties = await storageFile.Properties.GetVideoPropertiesAsync();
            return properties.Duration.TotalSeconds;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Időtartam lekérési hiba: {ex.Message}");
            return 0;
        }
#else
        await Task.CompletedTask;
        return 0;
#endif
    }

    public async Task<string> TrimVideoAsync(string inputFilePath, double startSec, double endSec)
    {
#if WINDOWS
        try
        {
            var storageFile = await Windows.Storage.StorageFile.GetFileFromPathAsync(inputFilePath);
            var clip = await Windows.Media.Editing.MediaClip.CreateFromFileAsync(storageFile);

            clip.TrimTimeFromStart = TimeSpan.FromSeconds(startSec);
            double totalDuration = clip.OriginalDuration.TotalSeconds;
            double trimEnd = Math.Max(0, totalDuration - endSec);
            clip.TrimTimeFromEnd = TimeSpan.FromSeconds(trimEnd);

            var composition = new Windows.Media.Editing.MediaComposition();
            composition.Clips.Add(clip);

            string outputFileName = $"trimmed_{Guid.NewGuid():N}.mp4";
            var tempFolder = Windows.Storage.ApplicationData.Current.TemporaryFolder;
            var destinationFile = await tempFolder.CreateFileAsync(outputFileName, Windows.Storage.CreationCollisionOption.GenerateUniqueName);

            var action = composition.RenderToFileAsync(destinationFile, Windows.Media.Editing.MediaTrimmingPreference.Fast);
            await action.AsTask();

            return destinationFile.Path;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Kliensoldali vágási hiba: {ex.Message}");
            return inputFilePath;
        }
#else
        await Task.CompletedTask;
        return inputFilePath;
#endif
    }
}
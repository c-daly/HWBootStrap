using System.Text.Json;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// Writes the whole snapshot to the log, once a minute, as one line.
    ///
    /// It exists because the deployment this ships to has a log and nothing else. A metrics endpoint only
    /// helps somebody who is already scraping it, and the numbers that matter after an incident are the
    /// ones from before anybody went looking. One structured line per minute is cheap enough to leave on
    /// everywhere and complete enough to reconstruct what the host was doing.
    ///
    /// The period runs off the registered TimeProvider, so a test drives it rather than waits for it.
    /// </summary>
    public sealed class MetricsLogService(
        MatchMetrics metrics, TimeProvider time, ILogger<MetricsLogService> logger) : BackgroundService
    {
        /// <summary>Often enough to see a spike, rarely enough that the line is not the log.</summary>
        public static readonly TimeSpan Period = TimeSpan.FromSeconds(60);

        static readonly JsonSerializerOptions Shape = new(JsonSerializerDefaults.Web);

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            // PeriodicTimer rather than a delay loop so a slow serialisation does not push every later tick
            // back, and so the period is the clock the rest of the host shares.
            using var timer = new PeriodicTimer(Period, time);

            try
            {
                while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
                {
                    logger.LogInformation(
                        "Metrics {Snapshot}", JsonSerializer.Serialize(metrics.Snapshot(), Shape));
                }
            }
            catch (OperationCanceledException)
            {
                // The host is going away. The last snapshot is of no interest to anybody.
            }
        }
    }
}

using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// The retention decision, applied on a cadence.
    ///
    /// It implements docs/operations/match-data-retention.md and nothing else: expire the matches that never
    /// started, abandon the ones that went quiet, purge long-expired credentials, and hard-delete terminal
    /// matches past the keep window. The rules themselves live in the store, as four indexed statements in
    /// one transaction; this class decides only WHEN they run and what happens to the sockets afterwards.
    ///
    /// Two things it deliberately does not do. It never logs an identifier - row counts only, because a
    /// sweep touches every match on the host and a log line that named them would be a slow leak of who was
    /// playing what. And it never touches an active match with recent activity: a game in progress is not
    /// disturbed by retention however long it has been running, as long as players keep playing it.
    ///
    /// The cadence runs on the injected TimeProvider, so a test drives it rather than waiting an hour.
    /// </summary>
    public sealed class MatchRetentionService(
        IMatchStore store,
        DurableMatchCoordinator coordinator,
        IOptions<MatchHostingOptions> options,
        TimeProvider time,
        ILogger<MatchRetentionService> logger) : BackgroundService
    {
        /// <summary>The close a socket gets when its match was abandoned under it. 1001 is going away, which
        /// is what actually happened: the server is not coming back to this game.</summary>
        public const int AbandonedCloseStatus = 1001;

        public const string AbandonedCloseReason = "abandoned";

        int _sweeps;

        /// <summary>Sweeps that have finished, whatever they found. What proves the cadence is running.</summary>
        internal int Sweeps => Volatile.Read(ref _sweeps);

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            TimeSpan period = TimeSpan.FromMinutes(options.Value.RetentionSweepMinutes);

            // A PeriodicTimer rather than a delay loop, so a slow sweep does not push every later one back,
            // and through the TimeProvider so a test can drive a ninety-day window without waiting for it.
            using var timer = new PeriodicTimer(period, time);

            try
            {
                while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
                {
                    await SweepOnceAsync(stoppingToken).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                // Shutdown.
            }
        }

        /// <summary>
        /// One pass of the policy, plus the eviction that has to follow it.
        ///
        /// A failure is logged and swallowed. Retention is housekeeping: a database that refused this sweep
        /// is a database that will be asked again in an hour, and a background service that died on it would
        /// take the whole policy with it silently, which is the one outcome worse than a late sweep.
        /// </summary>
        internal async Task<RetentionResult> SweepOnceAsync(CancellationToken ct)
        {
            RetentionPolicy policy = options.Value.RetentionPolicy;

            RetentionResult result;
            try
            {
                result = await store.ApplyRetentionAsync(policy, time.GetUtcNow(), ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                logger.LogError(failure, "The retention sweep failed and will be retried on the next cadence");
                return RetentionResult.Nothing;
            }

            Interlocked.Increment(ref _sweeps);

            // After the rows, never before them. A socket closed for a match the sweep then failed to
            // abandon would have dropped a player out of a game that is still being played.
            if (result.AbandonedIds.Count > 0)
            {
                await coordinator
                    .EvictAsync(result.AbandonedIds, AbandonedCloseStatus, AbandonedCloseReason)
                    .ConfigureAwait(false);
            }

            // Counts only. Never a match id, a Steam id, a command wire or credential material.
            if (!result.IsEmpty)
            {
                logger.LogInformation(
                    "Retention swept {Expired} expired, {Abandoned} abandoned, {Credentials} credentials deleted, "
                    + "{Matches} matches deleted",
                    result.Expired, result.Abandoned, result.CredentialsDeleted, result.MatchesDeleted);
            }

            return result;
        }
    }
}

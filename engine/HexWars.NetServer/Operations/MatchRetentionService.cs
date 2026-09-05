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
        IMatchEvictor evictor,
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

        /// <summary>How long one eviction is given before the sweep moves on to the next match.</summary>
        internal static readonly TimeSpan EvictionTimeout = TimeSpan.FromSeconds(5);

        /// <summary>Matches waiting to be evicted at once, at most.</summary>
        internal const int MaxPendingEvictions = 256;

        readonly object _pendingGate = new();
        readonly HashSet<Guid> _pending = new();

        /// <summary>Ids an earlier pass could not close, waiting for the next one.</summary>
        internal int PendingEvictions
        {
            get { lock (_pendingGate) return _pending.Count; }
        }

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
            //
            // Per id, and each one on its own. An eviction that throws or never returns must not take the
            // retention loop with it: the rows are already correct, the sockets are what is left, and a
            // socket that could not be closed is worth retrying rather than worth dying over.
            await EvictAsync(Due(result.AbandonedIds), ct).ConfigureAwait(false);

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

        /// <summary>
        /// The ids this pass should try to close: what it just abandoned, plus whatever an earlier pass
        /// could not get to. Retrying is the point of the pending set - an abandoned match whose socket was
        /// never closed is a player still sitting in a game the database says is over.
        /// </summary>
        IReadOnlyList<Guid> Due(IReadOnlyList<Guid> abandoned)
        {
            lock (_pendingGate)
            {
                if (_pending.Count == 0) return abandoned;

                // A snapshot, not a hand-off. An id leaves the pending set when it has actually been
                // evicted; clearing it here would lose every id this pass does not reach, and a pass can
                // stop at any point because shutdown cancelled it.
                var due = new List<Guid>(_pending);

                foreach (Guid id in abandoned)
                {
                    if (!due.Contains(id)) due.Add(id);
                }

                return due;
            }
        }

        /// <summary>Closes the sockets of each match in turn, and remembers the ones it could not.</summary>
        async Task EvictAsync(IReadOnlyList<Guid> matchIds, CancellationToken ct)
        {
            var single = new Guid[1];

            foreach (Guid matchId in matchIds)
            {
                ct.ThrowIfCancellationRequested();
                single[0] = matchId;

                try
                {
                    IReadOnlyList<Guid> left = await evictor
                        .EvictAsync(single, AbandonedCloseStatus, AbandonedCloseReason, ct)
                        .WaitAsync(EvictionTimeout, time, ct)
                        .ConfigureAwait(false);

                    if (left.Count > 0) Remember(matchId);
                    else Forget(matchId);
                }
                catch (OperationCanceledException)
                {
                    // Shutdown. The id stays pending so the next process, or the next pass, tries again.
                    Remember(matchId);
                    throw;
                }
                catch (TimeoutException)
                {
                    logger.LogWarning(
                        "An eviction did not finish within {Seconds}s; it will be tried again next sweep",
                        (int)EvictionTimeout.TotalSeconds);
                    Remember(matchId);
                }
                catch (Exception failure)
                {
                    // One match that would not close is not a reason to stop closing the others, and it is
                    // certainly not a reason to end the loop that applies the whole retention policy.
                    logger.LogWarning(failure, "An abandoned match could not be evicted; retrying next sweep");
                    Remember(matchId);
                }
            }
        }

        /// <summary>Drops an id that has been dealt with, so a retry does not become a permanent one.</summary>
        void Forget(Guid matchId)
        {
            lock (_pendingGate) _pending.Remove(matchId);
        }

        /// <summary>Keeps an id for the next pass, up to a ceiling. The set is bounded because it is fed by
        /// whatever the host is failing to do, and an unbounded record of failures is its own outage.</summary>
        void Remember(Guid matchId)
        {
            lock (_pendingGate)
            {
                if (_pending.Count >= MaxPendingEvictions)
                {
                    logger.LogWarning(
                        "More than {Ceiling} matches are waiting to be evicted; dropping the newest",
                        MaxPendingEvictions);
                    return;
                }

                _pending.Add(matchId);
            }
        }
    }
}

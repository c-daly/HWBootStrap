namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// Verifies every open match while the host is starting, and keeps trying until it manages to.
    ///
    /// It runs after the migration service because hosted services start in registration order and a
    /// journal checked against a schema that has not been brought up to date yet would be refused for a
    /// reason that has nothing to do with the match.
    ///
    /// It never throws, which is the whole point of it being separate from readiness. A database that is
    /// down at boot is the most likely reason this pass fails, and a hosted service that threw would take
    /// the process with it - into a restart loop, against the same database, holding no traffic and telling
    /// nobody why.
    ///
    /// And it retries, which is the rest of the point. A pass that ran once and failed left this host
    /// permanently unready over a database outage that ended minutes later: the only way back was a deploy
    /// or a kill, so an outage in one dependency became an outage that needed a human. The first attempt is
    /// awaited, so a healthy host has verified everything before it serves anything; a host that could not
    /// goes on trying in the background and starts serving the moment the database returns.
    ///
    /// With no database configured at all - the legacy deployment - there is nothing to verify and it says
    /// so immediately, so readiness does not wait forever for a pass that will never run.
    /// </summary>
    public sealed class RecoveryStartupService(
        RecoveryState state,
        MatchRecoveryService? recovery,
        TimeProvider time,
        ILogger<RecoveryStartupService> logger) : IHostedService, IDisposable
    {
        /// <summary>
        /// How long the first retries wait, before it settles into <see cref="RetryInterval"/>.
        ///
        /// Short at first because most failures at boot are a database that is a few seconds behind the
        /// host, and longer afterwards because one that is still down after a minute and a half is down for
        /// a reason that will not be fixed by asking again quickly.
        /// </summary>
        internal static readonly IReadOnlyList<TimeSpan> RetryBackoff = new[]
        {
            TimeSpan.FromSeconds(30),
            TimeSpan.FromSeconds(60),
        };

        /// <summary>The steady cadence every attempt after the backoff uses.</summary>
        internal static readonly TimeSpan RetryInterval = TimeSpan.FromSeconds(120);

        readonly CancellationTokenSource _stopping = new();

        Task? _retries;

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            if (recovery is null)
            {
                state.RecordReport(new RecoveryReport(
                    0,
                    0,
                    Array.Empty<(Guid MatchId, MatchRecoveryFailure Failure, string Detail)>(),
                    time.GetUtcNow()));
                return;
            }

            // Awaited, as it always was: on a healthy host the pass is finished before the first request
            // arrives, which is what lets readiness answer honestly from the moment it can answer at all.
            if (await TryVerifyAsync(cancellationToken).ConfigureAwait(false)) return;

            // Only a host that could not verify goes on trying, and it does that behind startup rather than
            // inside it. Blocking here would hold the whole host down for as long as the database is,
            // which is the crash loop this service exists to avoid.
            _retries = Task.Run(() => RetryUntilItSucceedsAsync(_stopping.Token), CancellationToken.None);
        }

        public async Task StopAsync(CancellationToken cancellationToken)
        {
            await _stopping.CancelAsync().ConfigureAwait(false);

            if (_retries is null) return;

            try
            {
                await _retries.WaitAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Shutdown ran out of patience. The loop is cancelled either way.
            }
        }

        public void Dispose() => _stopping.Dispose();

        /// <summary>One attempt. True when the host now knows what it is hosting.</summary>
        async Task<bool> TryVerifyAsync(CancellationToken ct)
        {
            try
            {
                RecoveryReport report = await recovery!.VerifyOpenMatchesAsync(ct).ConfigureAwait(false);

                state.RecordReport(report);

                if (report.Failed.Count > 0)
                    logger.LogWarning(
                        "Startup recovery finished with {Refused} match(es) this build will not host",
                        report.Failed.Count);

                return true;
            }
            catch (Exception failure)
            {
                // Every exception, cancellation included: a pass abandoned because the host is shutting
                // down has still not verified anything, and readiness must say so rather than inherit an
                // all-clear from a run that did not happen.
                state.RecordFailure(failure);
                logger.LogError(failure,
                    "Startup recovery could not run; this host will report unready and try again");

                return false;
            }
        }

        async Task RetryUntilItSucceedsAsync(CancellationToken stopping)
        {
            var attempt = 0;

            while (!stopping.IsCancellationRequested)
            {
                TimeSpan wait = attempt < RetryBackoff.Count ? RetryBackoff[attempt] : RetryInterval;
                attempt++;

                try
                {
                    await Task.Delay(wait, time, stopping).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    return;
                }

                if (await TryVerifyAsync(stopping).ConfigureAwait(false)) return;
            }
        }
    }
}

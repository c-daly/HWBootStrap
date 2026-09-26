using HexWars.NetServer.Operations;

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

        /// <summary>Guards the one state transition this service has: cancelled, then disposed. Stopping and
        /// disposing arrive from different parts of the host and in either order.</summary>
        readonly object _gate = new();

        bool _cancelled;
        bool _disposed;

        /// <summary>Written under the gate by StartAsync, read by StopAsync from whichever thread the host
        /// stops on.</summary>
        volatile Task? _retries;

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
            // The token is taken under the gate, and the loop is not started at all if this service has
            // already been torn down. Reading Token on a disposed source throws, and a host CAN be disposed
            // while its own start is still finishing.
            CancellationToken stopping;
            lock (_gate)
            {
                if (_disposed) return;

                stopping = _stopping.Token;
                _retries = Task.Run(() => RetryUntilItSucceedsAsync(stopping), CancellationToken.None);
            }
        }

        /// <summary>
        /// Stops the retry loop and waits for it to unwind.
        ///
        /// It does NOT assume it runs before <see cref="Dispose"/>. Nothing in the hosted-service contract
        /// promises that: the container disposes singletons when the provider is disposed, and disposing an
        /// IHost disposes the provider WITHOUT stopping anything - which is exactly what the synchronous
        /// dispose of a test host, or any `using var host = builder.Build()`, does. A service that cancelled
        /// an already-disposed source there threw ObjectDisposedException out of the host teardown and took
        /// whatever was tearing it down with it.
        /// </summary>
        public async Task StopAsync(CancellationToken cancellationToken)
        {
            CancelOnce();

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

        /// <summary>
        /// Releases the cancellation source, cancelling it first.
        ///
        /// Cancelling here rather than only in <see cref="StopAsync"/> is the other half of the same
        /// problem. A host that is disposed without being stopped used to dispose this source while the
        /// retry loop was still awaiting a delay on its token: the registration goes with the source, so the
        /// delay never completes and the loop is stranded for the life of the process, holding the host it
        /// belonged to alive behind it.
        /// </summary>
        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;

                if (!_cancelled)
                {
                    _cancelled = true;
                    _stopping.Cancel();
                }

                _stopping.Dispose();
            }
        }

        /// <summary>
        /// Cancels the loop exactly once, whichever way this service is being torn down.
        ///
        /// The cancel happens under the gate rather than outside it, because the whole point is that
        /// cancelling and disposing cannot interleave. Its callbacks are the token registrations of a delay
        /// and a wait, neither of which re-enters this service, so there is nothing here to deadlock on.
        /// </summary>
        void CancelOnce()
        {
            lock (_gate)
            {
                if (_cancelled || _disposed) return;

                _cancelled = true;
                _stopping.Cancel();
            }
        }

        /// <summary>The retry loop, for a test that needs to see it finish. Null until one is started.</summary>
        internal Task? RetryLoop => _retries;

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
                logger.LogRedacted(LogLevel.Error, failure,
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

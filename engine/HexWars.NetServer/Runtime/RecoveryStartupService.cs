namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// Verifies every open match once, while the host is starting, and records what it found.
    ///
    /// It runs after the migration service because hosted services start in registration order and a
    /// journal checked against a schema that has not been brought up to date yet would be refused for a
    /// reason that has nothing to do with the match.
    ///
    /// It never throws, which is the whole point of it being separate from readiness. A database that is
    /// down at boot is the most likely reason this pass fails, and a hosted service that threw would take
    /// the process with it - into a restart loop, against the same database, holding no traffic and telling
    /// nobody why. Recording the failure and coming up unready instead leaves a process that answers
    /// /health/live, names the problem on /health/ready, and starts serving the moment the database returns.
    ///
    /// With no database configured at all - the legacy deployment - there is nothing to verify and it says
    /// so immediately, so readiness does not wait forever for a pass that will never run.
    /// </summary>
    public sealed class RecoveryStartupService(
        RecoveryState state,
        MatchRecoveryService? recovery,
        ILogger<RecoveryStartupService> logger) : IHostedService
    {
        public async Task StartAsync(CancellationToken cancellationToken)
        {
            if (recovery is null)
            {
                state.RecordReport(new RecoveryReport(
                    0,
                    Array.Empty<(Guid MatchId, MatchRecoveryFailure Failure, string Detail)>(),
                    DateTimeOffset.UtcNow));
                return;
            }

            try
            {
                RecoveryReport report =
                    await recovery.VerifyOpenMatchesAsync(cancellationToken).ConfigureAwait(false);

                state.RecordReport(report);

                if (report.Failed.Count > 0)
                    logger.LogWarning(
                        "Startup recovery finished with {Refused} match(es) this build will not host",
                        report.Failed.Count);
            }
            catch (Exception failure)
            {
                // Every exception, cancellation included: a pass abandoned because the host is shutting
                // down has still not verified anything, and readiness must say so rather than inherit an
                // all-clear from a run that did not happen.
                state.RecordFailure(failure);
                logger.LogError(failure,
                    "Startup recovery could not run; this host will report unready until it can");
            }
        }

        public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    }
}

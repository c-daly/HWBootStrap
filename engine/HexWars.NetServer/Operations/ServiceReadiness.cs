using System.Security.Cryptography;
using System.Text;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Npgsql;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// Whether this process still wants traffic.
    ///
    /// One bit, and it only ever moves one way. A host that has been told to stop will be gone in seconds,
    /// and anything that let it come back would mean a load balancer sending a player into a process that
    /// is already closing their socket. Separate from the health checks because several things read it -
    /// readiness, the allocation endpoints, the socket route - and none of them should have to know how a
    /// health check is registered.
    /// </summary>
    public sealed class ServiceReadiness
    {
        int _shuttingDown;

        /// <summary>True once this host has begun going away.</summary>
        public bool ShuttingDown => Volatile.Read(ref _shuttingDown) != 0;

        public void BeginShutdown() => Interlocked.Exchange(ref _shuttingDown, 1);
    }

    /// <summary>The routes, tags and check names the operations surface is made of, in one place so a probe
    /// configured in a deployment file and the code it hits cannot drift apart.</summary>
    public static class HealthEndpoints
    {
        /// <summary>Answered 200 for as long as the process is running, whatever state it is in.</summary>
        public const string LiveRoute = "/health/live";

        /// <summary>The name this server has always answered liveness on. Kept as an alias because a
        /// deployment that still points at it is not wrong, only old.</summary>
        public const string LegacyLiveRoute = "/healthz";

        /// <summary>Answered 200 only when this host can actually serve a match.</summary>
        public const string ReadyRoute = "/health/ready";

        public const string MetricsRoute = "/api/v1/metrics";

        public const string MetricsTokenHeader = "X-Metrics-Token";

        /// <summary>Every check readiness consults carries this tag; nothing else does.</summary>
        public const string ReadyTag = "ready";

        public const string DatabaseCheck = "database";
        public const string SchemaCheck = "schema";
        public const string RecoveryCheck = "recovery";
        public const string ShutdownCheck = "shutdown";

        /// <summary>
        /// Whether an offered metrics token is the configured one, in time that does not depend on how much
        /// of it is right. Both sides are hashed first so the comparison is over two equal-length digests:
        /// FixedTimeEquals over the raw strings would refuse a wrong length immediately and hand a caller
        /// the length of the secret for free.
        /// </summary>
        public static bool TokenMatches(string expected, string? offered)
        {
            if (string.IsNullOrEmpty(offered)) return false;

            Span<byte> expectedDigest = stackalloc byte[32];
            Span<byte> offeredDigest = stackalloc byte[32];

            SHA256.HashData(Encoding.UTF8.GetBytes(expected), expectedDigest);
            SHA256.HashData(Encoding.UTF8.GetBytes(offered), offeredDigest);

            return CryptographicOperations.FixedTimeEquals(expectedDigest, offeredDigest);
        }

        /// <summary>
        /// The readiness body: the overall verdict, then every check by name with what it said.
        ///
        /// The per-check lines are the reason this is not the default writer. A 503 with no body tells an
        /// operator that something is wrong and nothing else, and the three reasons this server can be
        /// unready - the database, the schema, the recovery pass - want three different responses.
        /// </summary>
        public static Task WriteAsync(HttpContext context, HealthReport report)
        {
            ArgumentNullException.ThrowIfNull(context);
            ArgumentNullException.ThrowIfNull(report);

            return context.Response.WriteAsJsonAsync(new
            {
                status = report.Status.ToString(),
                checks = report.Entries.Select(entry => new
                {
                    name = entry.Key,
                    status = entry.Value.Status.ToString(),
                    description = entry.Value.Description,
                }).ToArray(),
            });
        }
    }

    /// <summary>
    /// Can this host reach its database right now.
    ///
    /// The data source arrives behind a delegate rather than as itself. Readiness is asked again on every
    /// probe precisely because the answer changes while the process runs, and a check that captured one
    /// connection pool at startup could not be pointed anywhere else - which is exactly what a test proving
    /// the check notices an outage has to do.
    /// </summary>
    public sealed class DatabaseHealthCheck(Func<NpgsqlDataSource> source) : IHealthCheck
    {
        /// <summary>A probe is not a query. Past this the answer an operator needs is already no.</summary>
        public static readonly TimeSpan Deadline = TimeSpan.FromSeconds(2);

        public async Task<HealthCheckResult> CheckHealthAsync(
            HealthCheckContext context, CancellationToken cancellationToken = default)
        {
            using CancellationTokenSource deadline =
                CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            deadline.CancelAfter(Deadline);

            try
            {
                await using NpgsqlConnection connection =
                    await source().OpenConnectionAsync(deadline.Token).ConfigureAwait(false);
                await using NpgsqlCommand probe = connection.CreateCommand();
                probe.CommandText = "SELECT 1";
                await probe.ExecuteScalarAsync(deadline.Token).ConfigureAwait(false);

                return HealthCheckResult.Healthy();
            }
            catch (Exception failure)
            {
                // A fixed description: the exception carries the address it could not reach, and this
                // string is served to whoever can call the probe.
                return HealthCheckResult.Unhealthy("the database did not answer", failure);
            }
        }
    }

    /// <summary>
    /// Is the schema this build expects actually applied.
    ///
    /// It is a separate question from whether the database answers. A rolling deploy can put a new binary
    /// in front of an old schema for a moment, and a host that took traffic there would accept players into
    /// matches it cannot journal - which is the one failure this whole design exists to prevent.
    /// </summary>
    public sealed class SchemaHealthCheck(MigrationRunner runner) : IHealthCheck
    {
        /// <summary>Longer than the database probe: reading the ledger takes the advisory lock, so a peer
        /// that is mid-migration is a wait rather than a failure. Past this, unready is the honest answer.</summary>
        public static readonly TimeSpan Deadline = TimeSpan.FromSeconds(5);

        public async Task<HealthCheckResult> CheckHealthAsync(
            HealthCheckContext context, CancellationToken cancellationToken = default)
        {
            using CancellationTokenSource deadline =
                CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            deadline.CancelAfter(Deadline);

            try
            {
                IReadOnlyList<string> pending = await runner.PendingAsync(deadline.Token).ConfigureAwait(false);

                return pending.Count == 0
                    ? HealthCheckResult.Healthy()
                    : HealthCheckResult.Unhealthy(
                        "the schema is behind this build: " + string.Join(", ", pending) + " not applied");
            }
            catch (Exception failure)
            {
                return HealthCheckResult.Unhealthy("the schema could not be read", failure);
            }
        }
    }

    /// <summary>
    /// What the startup recovery pass concluded.
    ///
    /// Three answers, and the middle one is the reason this check exists rather than a boolean. A pass that
    /// has not run has verified nothing and the host must hold traffic. A pass that refused a match found
    /// something a human has to look at - but every OTHER match on this host is fine, and reporting unready
    /// over it would take a working deployment offline because one journal is broken.
    /// </summary>
    public sealed class RecoveryHealthCheck(RecoveryState state) : IHealthCheck
    {
        /// <summary>Enough refused matches for an operator to see the shape of the problem, and few enough
        /// that a probe response stays a probe response.</summary>
        public const int NamedFailures = 20;

        public Task<HealthCheckResult> CheckHealthAsync(
            HealthCheckContext context, CancellationToken cancellationToken = default)
        {
            if (state.Error is Exception error)
                return Task.FromResult(HealthCheckResult.Unhealthy(
                    "the startup recovery pass could not run", error));

            RecoveryReport? report = state.Report;
            if (!state.Completed || report is null)
                return Task.FromResult(HealthCheckResult.Unhealthy(
                    "the startup recovery pass has not finished"));

            if (report.Failed.Count == 0) return Task.FromResult(HealthCheckResult.Healthy());

            string named = string.Join(", ", report.Failed
                .Take(NamedFailures)
                .Select(refusal => refusal.MatchId.ToString() + " (" + refusal.Failure + ")"));

            if (report.Failed.Count > NamedFailures)
                named += " and " + (report.Failed.Count - NamedFailures).ToString() + " more";

            return Task.FromResult(HealthCheckResult.Degraded(
                report.Failed.Count.ToString() + " match(es) this build will not host: " + named));
        }
    }

    /// <summary>Says no from the moment this host is told to stop, which is what takes it out of rotation
    /// before its sockets are closed rather than after.</summary>
    public sealed class ShutdownHealthCheck(ServiceReadiness readiness) : IHealthCheck
    {
        public Task<HealthCheckResult> CheckHealthAsync(
            HealthCheckContext context, CancellationToken cancellationToken = default) =>
            Task.FromResult(readiness.ShuttingDown
                ? HealthCheckResult.Unhealthy("this host is shutting down")
                : HealthCheckResult.Healthy());
    }
}

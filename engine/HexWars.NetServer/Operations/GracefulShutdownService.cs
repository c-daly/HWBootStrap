using HexWars.NetServer.Hosting;
using HexWars.NetServer.Runtime;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// What this process does between being told to stop and stopping.
    ///
    /// The order is the whole design. Readiness goes false FIRST, so the platform stops sending new players
    /// here while the sockets that are already here are still being served. Then the in-flight commits are
    /// drained, because a command that was journalled but never broadcast is the one failure the durable
    /// design exists to prevent. Only then are the seats told, and only then are their sockets closed with
    /// a status that says restart rather than error - a client that reads 1012 reconnects, and a client
    /// that reads a torn connection shows a player a failure that never happened.
    ///
    /// A legacy deployment has no coordinator and no socket registry. There it flips readiness and stops,
    /// which is all there is to do.
    /// </summary>
    public sealed class GracefulShutdownService : IHostedService, IDisposable
    {
        /// <summary>The frame every seated client is sent before its socket goes. Not a protocol message
        /// the client has to understand: a client that does not know it shows nothing and reconnects on the
        /// close, which is the same outcome one paragraph later.</summary>
        public const string RestartNotice = "SERVER RESTART";

        /// <summary>1012 Service Restart. The one close code that means come back, rather than go away.</summary>
        public const int RestartCloseStatus = 1012;

        public const string RestartCloseReason = "service restart";

        /// <summary>How long in-flight commits are given to finish. Comfortably inside the host shutdown
        /// timeout, so the drain finishing late still leaves room to say goodbye.</summary>
        public static readonly TimeSpan DrainWindow = TimeSpan.FromSeconds(10);

        /// <summary>What the host waits for before it stops waiting. Long enough for the drain, the notice
        /// and every close handshake; short enough that a platform that kills at 30 seconds does not.</summary>
        public static readonly TimeSpan ShutdownTimeout = TimeSpan.FromSeconds(25);

        readonly ServiceReadiness _readiness;
        readonly IHostApplicationLifetime _lifetime;
        readonly DurableMatchCoordinator? _coordinator;
        readonly V2ConnectionRegistry? _registry;
        readonly TimeProvider _time;
        readonly ILogger<GracefulShutdownService> _logger;
        readonly object _gate = new();

        CancellationTokenRegistration _stopping;
        Task? _running;

        public GracefulShutdownService(
            ServiceReadiness readiness,
            IHostApplicationLifetime lifetime,
            DurableMatchCoordinator? coordinator,
            V2ConnectionRegistry? registry,
            TimeProvider time,
            ILogger<GracefulShutdownService> logger)
        {
            _readiness = readiness;
            _lifetime = lifetime;
            _coordinator = coordinator;
            _registry = registry;
            _time = time;
            _logger = logger;
        }

        public Task StartAsync(CancellationToken cancellationToken)
        {
            // ApplicationStopping, not StopAsync: it fires before any hosted service is stopped and before
            // the server stops accepting, which is the only moment where readiness can go false while the
            // sockets are all still there to be told.
            _stopping = _lifetime.ApplicationStopping.Register(() => Begin());
            return Task.CompletedTask;
        }

        public async Task StopAsync(CancellationToken cancellationToken)
        {
            _stopping.Dispose();

            // Begin rather than a read of the field: a host stopped some way that did not raise
            // ApplicationStopping still owes its players a goodbye.
            await Begin().ConfigureAwait(false);
        }

        public void Dispose() => _stopping.Dispose();

        /// <summary>Starts the sequence, once, and hands back the same task to everyone who asks.</summary>
        Task Begin()
        {
            lock (_gate) return _running ??= RunAsync();
        }

        async Task RunAsync()
        {
            // Everything up to the first await runs on the caller thread, which is what makes a readiness
            // probe issued straight after StopApplication see the answer this method decided.
            _readiness.BeginShutdown();
            _logger.LogInformation("Shutdown: readiness false");

            long started = _time.GetTimestamp();
            int matches = _coordinator?.LiveMatchCount ?? 0;

            if (_coordinator is not null)
            {
                try
                {
                    await _coordinator.DrainAsync(DrainWindow).ConfigureAwait(false);
                }
                catch (Exception failure)
                {
                    _logger.LogWarning(failure, "Shutdown: the in-flight commits did not drain cleanly");
                }

                try
                {
                    await _coordinator.BroadcastAllAsync(RestartNotice).ConfigureAwait(false);
                }
                catch (Exception failure)
                {
                    _logger.LogWarning(failure, "Shutdown: the restart notice could not be broadcast");
                }
            }

            var closed = 0;

            if (_registry is not null)
            {
                // Every socket at once. Each close waits on a peer that may never answer, and doing them in
                // turn would spend that window per connection instead of once.
                IReadOnlyCollection<V2Connection> open = _registry.Snapshot();
                closed = open.Count;

                try
                {
                    await Task.WhenAll(open.Select(
                        connection => connection.CloseAsync(RestartCloseStatus, RestartCloseReason)))
                        .ConfigureAwait(false);
                }
                catch (Exception failure)
                {
                    _logger.LogWarning(failure, "Shutdown: not every socket closed politely");
                }
            }

            _logger.LogInformation(
                "Shutdown: {Matches} live match(es), {Sockets} socket(s) closed, took {Elapsed}",
                matches, closed, _time.GetElapsedTime(started));
        }
    }
}

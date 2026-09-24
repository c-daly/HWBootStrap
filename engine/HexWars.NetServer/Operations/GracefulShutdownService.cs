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

        /// <summary>How long in-flight commits are given to finish, at most.</summary>
        public static readonly TimeSpan DrainWindow = TimeSpan.FromSeconds(10);

        /// <summary>How long the notice and the closes together are given, at most.</summary>
        public static readonly TimeSpan NotifyWindow = TimeSpan.FromSeconds(5);

        /// <summary>
        /// The whole budget, drain and goodbye together.
        ///
        /// Five seconds inside the host timeout on purpose. Every step below is bounded against what is
        /// left of THIS, not against its own window, because the failure that matters is the one where a
        /// step overruns: a shutdown killed partway through sends 1012 to nobody, and every client it was
        /// serving sees a torn connection instead of an instruction to come back.
        /// </summary>
        public static readonly TimeSpan ShutdownBudget = TimeSpan.FromSeconds(20);

        /// <summary>What the closes get when the budget is already spent. The closes are attempted
        /// whatever happened before them, and V2Connection bounds each one at three seconds and then
        /// aborts, so the worst case stays inside the host timeout.</summary>
        public static readonly TimeSpan CloseFloor = TimeSpan.FromSeconds(3);

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
        int _quiesced;
        int _saidGoodbye;

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
            // ApplicationStopping fires before any hosted service is stopped and before the server stops
            // accepting, which is the only moment where readiness can go false and admission can close
            // while every socket is still here to be told. It does nothing that waits: a token callback
            // runs on the thread that stopped the host, and awaiting there would hold up the stop itself.
            _stopping = _lifetime.ApplicationStopping.Register(Quiesce);
            return Task.CompletedTask;
        }

        /// <summary>
        /// Everything that has to happen at the instant of stopping, and nothing that waits.
        ///
        /// Order matters and is the whole point: readiness first so the platform stops routing here, then
        /// the coordinator so no further command is committed, then the registry so no further socket is
        /// admitted. All three are set before StopAsync takes its snapshot, so nothing can be added to the
        /// set of things that need closing after that set is read.
        /// </summary>
        void Quiesce()
        {
            if (Interlocked.Exchange(ref _quiesced, 1) != 0) return;

            _readiness.BeginShutdown();
            _coordinator?.BeginShutdown();
            _registry?.BeginShutdown();

            _logger.LogInformation("Shutdown: readiness false");
        }

        /// <summary>
        /// The goodbye, inside one budget the host token can cut short.
        ///
        /// Run here rather than from the ApplicationStopping callback because this is the only place with
        /// a cancellation token the host actually enforces. The callback cannot be awaited by anything, so
        /// work started there runs unobserved and unbounded, which is how a wedged store used to carry
        /// shutdown past the platform kill deadline.
        /// </summary>
        public async Task StopAsync(CancellationToken cancellationToken)
        {
            _stopping.Dispose();

            // A host stopped some way that did not raise ApplicationStopping still owes its players this.
            Quiesce();

            // Once, however many times the host asks. A second pass would spend the budget again and send
            // a second SERVER RESTART to clients that are already closing, and the interesting case - a
            // host disposed after it was stopped - takes this path twice by construction.
            if (Interlocked.Exchange(ref _saidGoodbye, 1) != 0) return;

            long started = _time.GetTimestamp();
            int matches = _coordinator?.LiveMatchCount ?? 0;
            string skipped = "0";

            if (_coordinator is not null)
            {
                DurableMatchCoordinator.DrainSummary drain = default;

                bool finished = await StepAsync(
                    Remaining(started, DrainWindow),
                    async () => drain = await _coordinator.DrainAsync(Remaining(started, DrainWindow))
                        .ConfigureAwait(false),
                    "draining in-flight commits",
                    cancellationToken).ConfigureAwait(false);

                skipped = finished ? drain.Skipped.ToString() : "unknown (drain interrupted)";

                await StepAsync(
                    Remaining(started, NotifyWindow),
                    () => _coordinator.BroadcastAllAsync(RestartNotice),
                    "broadcasting the restart notice",
                    cancellationToken).ConfigureAwait(false);
            }

            var closed = 0;

            if (_registry is not null)
            {
                // Attempted whatever happened above, and given a floor of its own. A drain that overran is
                // a reason to hurry, not a reason to leave every client reading a torn socket.
                IReadOnlyCollection<V2Connection> open = _registry.Snapshot();
                closed = open.Count;

                TimeSpan left = Remaining(started, ShutdownBudget);
                if (left < CloseFloor) left = CloseFloor;

                await StepAsync(
                    left,
                    () => Task.WhenAll(open.Select(
                        connection => connection.CloseAsync(RestartCloseStatus, RestartCloseReason))),
                    "closing the sockets",
                    CancellationToken.None).ConfigureAwait(false);
            }

            _logger.LogInformation(
                "Shutdown: {Matches} live match(es), {Sockets} socket(s) closed, {Skipped} not drained, "
                + "took {DrainMs} ms",
                matches, closed, skipped, (long)_time.GetElapsedTime(started).TotalMilliseconds);
        }

        public void Dispose() => _stopping.Dispose();

        /// <summary>What is left of the budget, never more than this step is allowed and never below
        /// zero.</summary>
        TimeSpan Remaining(long started, TimeSpan cap)
        {
            TimeSpan left = ShutdownBudget - _time.GetElapsedTime(started);
            if (left < TimeSpan.Zero) left = TimeSpan.Zero;

            return left < cap ? left : cap;
        }

        /// <summary>
        /// One step, bounded. A step that overruns is logged and abandoned; the next one still runs.
        ///
        /// Abandoned rather than cancelled: WaitAsync stops waiting, it does not stop the work, and there
        /// is nothing useful to do about a store call that will never return. The process is about to end.
        /// </summary>
        async Task<bool> StepAsync(TimeSpan limit, Func<Task> step, string what, CancellationToken cancellation)
        {
            if (limit <= TimeSpan.Zero)
            {
                _logger.LogWarning("Shutdown: no budget left for {Step}", what);
                return false;
            }

            try
            {
                await step().WaitAsync(limit, cancellation).ConfigureAwait(false);
                return true;
            }
            catch (TimeoutException)
            {
                _logger.LogWarning("Shutdown: {Step} did not finish within {Limit}", what, limit);
            }
            catch (OperationCanceledException)
            {
                _logger.LogWarning("Shutdown: {Step} was cut short by the host deadline", what);
            }
            catch (Exception failure)
            {
                _logger.LogRedactedWarning(failure, "Shutdown: {Step} failed", what);
            }

            return false;
        }
    }
}

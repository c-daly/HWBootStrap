using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Runtime;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// The only thing that happens on an idle match: a PING on every seated socket, a close for every
    /// socket that has gone quiet, and a sweep of the matches nobody is watching.
    ///
    /// Both halves are needed and neither replaces the other. Without the PING an idle socket is
    /// indistinguishable from a dead one and an intermediary will eventually drop it silently; without the
    /// silence check a client that lost power holds its seat until the process restarts, because TCP has
    /// nothing to say about a peer that stopped listening.
    ///
    /// Unauthenticated sockets are left alone on purpose. They have their own, much shorter deadline in the
    /// handshake, and pinging one would extend the window in which a socket that has proved nothing gets to
    /// hold a connection slot.
    /// </summary>
    internal sealed class ConnectionHeartbeatService(
        V2ConnectionRegistry registry,
        DurableMatchCoordinator coordinator,
        IMatchCredentialService credentials,
        IOptions<MatchHostingOptions> options,
        TimeProvider time,
        ILogger<ConnectionHeartbeatService> logger) : BackgroundService
    {
        internal const string PingFrame = "PING";
        internal const int StaleCloseStatus = 1001;
        internal const string StaleReason = "stale";

        /// <summary>The close a socket gets when the credential behind it stopped working.</summary>
        internal const int CredentialCloseStatus = 1008;

        internal const string ExpiredReason = "credential expired";

        internal const string RevokedReason = "credential revoked";

        /// <summary>Credential re-checks in flight at once, across every socket on this host.</summary>
        internal const int MaxConcurrentRechecks = 8;

        /// <summary>How long one re-check is given before it is abandoned until the next cadence.</summary>
        internal static readonly TimeSpan RecheckTimeout = TimeSpan.FromSeconds(5);

        /// <summary>How often a cadence that could not start is allowed to say so.</summary>
        static readonly TimeSpan OverrunLogInterval = TimeSpan.FromMinutes(1);

        long _candidates;
        long _scanned;
        long _cadences;

        Task? _inflight;
        CancellationTokenSource? _inflightCancellation;
        DateTimeOffset _lastOverrunLog = DateTimeOffset.MinValue;

        /// <summary>Connections that have been offered to the batch selection. A socket that is not due is
        /// never counted, so this is what proves the selection walks the due ones and not the host.</summary>
        internal long RecheckCandidatesConsidered => Interlocked.Read(ref _candidates);

        /// <summary>Every connection the selection has looked at, due or not.</summary>
        internal long RecheckConnectionsScanned => Interlocked.Read(ref _scanned);

        /// <summary>Re-check passes that have found something to do.</summary>
        internal long RecheckCadences => Interlocked.Read(ref _cadences);

        /// <summary>Re-check passes running right now: one, or none. Never two.</summary>
        internal int RecheckBatchesInFlight =>
            Volatile.Read(ref _inflight) is { IsCompleted: false } ? 1 : 0;

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            TimeSpan interval = TimeSpan.FromSeconds(options.Value.HeartbeatIntervalSeconds);
            TimeSpan silence = TimeSpan.FromSeconds(options.Value.StaleConnectionSeconds);

            // Through the TimeProvider, so a test can drive the cadence rather than wait it out, and a
            // PeriodicTimer rather than a delay loop so a slow pass does not push every later tick back.
            using var timer = new PeriodicTimer(interval, time);

            try
            {
                while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
                {
                    DateTimeOffset now = time.GetUtcNow();

                    TimeSpan recheck = TimeSpan.FromSeconds(options.Value.CredentialRecheckSeconds);

                    foreach (V2Connection connection in registry.Snapshot())
                    {
                        if (!connection.IsAuthenticated) continue;

                        // Arithmetic, so it happens on every tick and costs nothing.
                        if (connection.CredentialExpiresAt is DateTimeOffset expiresAt && expiresAt <= now)
                        {
                            logger.LogInformation("Closing a v2 socket whose credential has expired");
                            _ = connection.CloseAsync(CredentialCloseStatus, ExpiredReason);
                            continue;
                        }

                        if (now - connection.LastInbound >= silence)
                        {
                            logger.LogInformation(
                                "Closing a v2 socket that has said nothing for {Seconds}s",
                                (int)silence.TotalSeconds);

                            // Not awaited: a close waits on a socket, and one unresponsive client must not
                            // delay the ping every other player in every other match is waiting for.
                            _ = connection.CloseAsync(StaleCloseStatus, StaleReason);
                            continue;
                        }

                        connection.TryEnqueue(PingFrame);
                    }

                    // After the pings, and never in the way of them. A re-check is a database round trip,
                    // and doing one inline would put every socket on this host behind the slowest of them:
                    // one match querying a wedged connection would hold up the ping that keeps every other
                    // match alive.
                    StartDueRechecks(now, recheck, interval, stoppingToken);

                    await coordinator.SweepAsync(now).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                // Shutdown. Sockets are closed by the host, not here.
            }
            catch (Exception failure)
            {
                // A heartbeat that died silently would leave every socket on this host looking healthy
                // forever, which is worse than the failure itself.
                logger.LogError(failure, "The connection heartbeat stopped");
            }
        }

        /// <summary>Whether this socket is due to have its credential asked about again.</summary>
        static bool IsRecheckDue(V2Connection connection, DateTimeOffset now, TimeSpan recheck) =>
            connection.CredentialHash is not null
            && connection.MatchId is not null
            && now - connection.LastCredentialCheck >= recheck;

        /// <summary>
        /// Starts a pass over the sockets whose credentials are due to be asked about again.
        ///
        /// One pass at a time, ever. The pass is detached so it cannot delay a ping, and a detached thing
        /// started once per tick is a leak waiting for a check that ignores its cancellation: the token
        /// only ASKS, so passes would pile up eight workers at a time and never come down. A tick that
        /// finds the previous pass still running cancels it and stands down instead.
        /// </summary>
        void StartDueRechecks(DateTimeOffset now, TimeSpan recheck, TimeSpan cadence, CancellationToken stopping)
        {
            if (_inflight is { IsCompleted: false })
            {
                _inflightCancellation?.Cancel();
                NoteOverrun(now);
                return;
            }

            _inflightCancellation?.Dispose();
            _inflightCancellation = null;
            Volatile.Write(ref _inflight, null);

            V2Connection[] batch = SelectOldestDue(now, recheck, options.Value.MaxRechecksPerCadence);
            if (batch.Length == 0) return;

            Interlocked.Increment(ref _cadences);

            var cancellation = CancellationTokenSource.CreateLinkedTokenSource(stopping);
            cancellation.CancelAfter(cadence);
            _inflightCancellation = cancellation;
            Volatile.Write(ref _inflight, DrainAsync(batch, now, cancellation.Token));
        }

        /// <summary>Says a pass overran, at most once a minute. A host that is behind is behind on every
        /// tick, and a line per tick would bury the fact rather than report it.</summary>
        void NoteOverrun(DateTimeOffset now)
        {
            if (now - _lastOverrunLog < OverrunLogInterval) return;

            _lastOverrunLog = now;
            logger.LogWarning(
                "A credential re-check pass was still running when the next was due; it has been cancelled "
                + "and this tick starts none. Sockets keep their place in the queue.");
        }

        /// <summary>
        /// The oldest-checked due sockets, up to <paramref name="budget"/>, in oldest-first order.
        ///
        /// Oldest FIRST and not merely the oldest SET. The batch is handed to a worker pool that takes
        /// items in order, so a batch in registry order starts whichever sockets happen to sit near the
        /// front of the dictionary - and when checks take longer than a cadence, those same few are all
        /// that ever start while an hour-old socket behind them is cancelled every time.
        ///
        /// A bounded heap rather than a sort: the host may have thousands of due sockets and this runs on
        /// the heartbeat loop. The heap keeps the newest of the chosen at its top, so the common case for
        /// a candidate is one comparison against it.
        /// </summary>
        V2Connection[] SelectOldestDue(DateTimeOffset now, TimeSpan recheck, int budget)
        {
            // Inverted, so the heap surfaces the NEWEST of the chosen - which is the one to evict.
            var chosen = new PriorityQueue<V2Connection, DateTimeOffset>(
                Comparer<DateTimeOffset>.Create((left, right) => right.CompareTo(left)));

            foreach (V2Connection connection in registry.Snapshot())
            {
                Interlocked.Increment(ref _scanned);

                if (!connection.IsAuthenticated || connection.IsClosed) continue;
                if (!IsRecheckDue(connection, now, recheck)) continue;

                Interlocked.Increment(ref _candidates);

                DateTimeOffset checkedAt = connection.LastCredentialCheck;

                if (chosen.Count < budget)
                {
                    chosen.Enqueue(connection, checkedAt);
                    continue;
                }

                if (checkedAt < chosen.Peek().LastCredentialCheck)
                    chosen.EnqueueDequeue(connection, checkedAt);
            }

            // Dequeue hands back newest first, so it is filled from the end: batch[0] is the oldest thing
            // on the host, and the pool starts there.
            var batch = new V2Connection[chosen.Count];
            for (int i = batch.Length - 1; i >= 0; i--) batch[i] = chosen.Dequeue();

            return batch;
        }

        /// <summary>
        /// Runs one batch of re-checks, eight at a time, until it is done or the cadence runs out.
        ///
        /// The deadline is the next heartbeat: a pass still going when the following one is due has fallen
        /// behind, and carrying on would stack passes on top of each other. What it did not reach is simply
        /// not stamped, so the next pass starts with exactly those sockets.
        /// </summary>
        async Task DrainAsync(V2Connection[] batch, DateTimeOffset now, CancellationToken deadline)
        {
            var pool = new ParallelOptions
            {
                MaxDegreeOfParallelism = MaxConcurrentRechecks,
                CancellationToken = deadline,
            };

            try
            {
                await Parallel.ForEachAsync(batch, pool, async (connection, ct) =>
                {
                    // Stamped as the check starts, not when the batch was chosen: a socket the deadline
                    // never reached keeps its place at the front of the queue.
                    connection.LastCredentialCheck = now;
                    await RecheckAsync(connection, now, ct).ConfigureAwait(false);
                }).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Expected on a busy host and on shutdown. The pass that could not finish leaves the rest
                // of its batch unstamped, which is what puts them first next time.
            }
            catch (Exception failure)
            {
                logger.LogWarning(failure, "A credential re-check pass stopped early");
            }
        }

        /// <summary>
        /// Asks the store whether a live socket still has a credential, and closes it when it does not.
        ///
        /// Deadlined on purpose; the concurrency is bounded by the pool that calls this. Anything that
        /// does not answer inside the deadline leaves the socket exactly as it was: an unanswered question
        /// is not evidence against the player, and closing every live socket over a database having a bad
        /// minute would be a self-inflicted outage.
        /// </summary>
        async Task RecheckAsync(V2Connection connection, DateTimeOffset now, CancellationToken batch)
        {
            try
            {
                if (connection.CredentialHash is not byte[] hash) return;
                if (connection.MatchId is not Guid matchId) return;

                using var deadline = CancellationTokenSource.CreateLinkedTokenSource(batch);
                deadline.CancelAfter(RecheckTimeout);

                // The token is passed AND the await is bounded. A store client that ignores cancellation
                // would otherwise hold this worker for as long as it liked, and the deadline would be a
                // request rather than a deadline. WaitAsync abandons it instead: the worker moves on, and
                // the orphaned call is bounded by eight per pass and one pass at a time.
                bool valid = await credentials
                    .IsStillValidAsync(hash, matchId, now, deadline.Token)
                    .WaitAsync(deadline.Token)
                    .ConfigureAwait(false);

                if (valid) return;

                logger.LogInformation("Closing a v2 socket whose credential is no longer valid");
                _ = connection.CloseAsync(CredentialCloseStatus, RevokedReason);
            }
            catch (OperationCanceledException)
            {
                if (!batch.IsCancellationRequested)
                    logger.LogWarning(
                        "A credential re-check did not answer inside {Seconds}s; leaving the socket open",
                        (int)RecheckTimeout.TotalSeconds);
            }
            catch (Exception failure)
            {
                logger.LogWarning(failure, "A live credential could not be re-checked");
            }
        }
    }
}

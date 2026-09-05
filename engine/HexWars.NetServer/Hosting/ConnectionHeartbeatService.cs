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

        /// <summary>
        /// Store calls that may be OUTSTANDING at once, whatever this host has given up waiting for.
        ///
        /// Abandoning an await frees the worker; it does not free the store. A client that treats
        /// cancellation as a suggestion therefore leaves a call running after its worker has moved on, and
        /// without a ceiling those orphans grow by up to eight a pass forever - invisible to any count of
        /// passes or workers. Twice the concurrency, so one slow generation of checks can overlap the next
        /// without stalling it, and no more than that.
        /// </summary>
        internal const int MaxOutstandingChecks = 2 * MaxConcurrentRechecks;

        /// <summary>How long one re-check is given before it is abandoned until the next cadence.</summary>
        internal static readonly TimeSpan RecheckTimeout = TimeSpan.FromSeconds(5);

        readonly SemaphoreSlim _outstanding = new(MaxOutstandingChecks, MaxOutstandingChecks);

        /// <summary>How often a cadence that could not start is allowed to say so.</summary>
        static readonly TimeSpan OverrunLogInterval = TimeSpan.FromMinutes(1);

        long _candidates;
        long _scanned;
        long _cadences;

        Task? _inflight;
        CancellationTokenSource? _inflightCancellation;
        DateTimeOffset _lastOverrunLog = DateTimeOffset.MinValue;
        DateTimeOffset _lastStarvedLog = DateTimeOffset.MinValue;

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

        /// <summary>Store calls this host is still waiting on, including the ones it has stopped awaiting.
        /// Never above <see cref="MaxOutstandingChecks"/>.</summary>
        internal int OutstandingChecks => MaxOutstandingChecks - _outstanding.CurrentCount;

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

        /// <summary>Says re-checks have stopped because every outstanding slot is held by a call that never
        /// came back. At most once a minute: a store in this state is in it for a while.</summary>
        void NoteStarved(DateTimeOffset now)
        {
            if (now - _lastStarvedLog < OverrunLogInterval) return;

            _lastStarvedLog = now;
            logger.LogWarning(
                "The credential store is not answering; live re-checks are paused until it does. "
                + "Sockets stay open.");
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
                    // The ceiling is taken BEFORE the stamp. When every slot is held by a call this host
                    // has given up on, re-checks simply pause - fail open, sockets stay where they are -
                    // and the sockets that could not be reached are left unstamped and still due.
                    try
                    {
                        await _outstanding.WaitAsync(ct).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        if (_outstanding.CurrentCount == 0) NoteStarved(now);
                        throw;
                    }

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
            // One outstanding slot is held on entry. Every path below either hands it to a store call or
            // gives it back here; it is never simply dropped.
            if (connection.CredentialHash is not byte[] hash || connection.MatchId is not Guid matchId)
            {
                _outstanding.Release();
                return;
            }

            var deadline = CancellationTokenSource.CreateLinkedTokenSource(batch);
            deadline.CancelAfter(RecheckTimeout);

            Task<bool> call;
            try
            {
                call = credentials.IsStillValidAsync(hash, matchId, now, deadline.Token);
            }
            catch (Exception failure)
            {
                // A client that threw before it managed to return a task.
                deadline.Dispose();
                _outstanding.Release();
                logger.LogWarning(failure, "A live credential could not be re-checked");
                return;
            }

            // The slot follows the CALL and not this method, and the continuation is attached before the
            // await so abandoning the await cannot release it early. Giving the slot back when this method
            // returns would be counting the workers again, which is the thing that does not bound anything:
            // the store is still busy whether or not anybody is still listening to it.
            _ = call.ContinueWith(
                finished =>
                {
                    _ = finished.Exception;   // observed, so an abandoned failure is not an unhandled one
                    _outstanding.Release();
                },
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);

            try
            {
                // The token is passed AND the await is bounded. A store client that ignores the token
                // would otherwise hold this worker for as long as it liked, and the deadline would be a
                // request rather than a deadline. WaitAsync abandons it instead: the worker moves on, and
                // the call it left behind still owns a slot until it finishes.
                bool valid = await call.WaitAsync(deadline.Token).ConfigureAwait(false);

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
            finally
            {
                // Only once the call is really over. The token belongs to it as well as to the await, and
                // a call this host abandoned is still holding one - disposing the source underneath it
                // would turn a slow store into an ObjectDisposedException. An abandoned source clears
                // itself when its own timer fires, which is at most one re-check timeout away.
                if (call.IsCompleted) deadline.Dispose();
            }
        }
    }
}

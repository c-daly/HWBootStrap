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

        long _candidates;
        long _cadences;

        /// <summary>Connections that have been offered to the batch selection. A socket that is not due is
        /// never counted, so this is what proves the selection walks the due ones and not the host.</summary>
        internal long RecheckCandidatesConsidered => Interlocked.Read(ref _candidates);

        /// <summary>Re-check passes that have found something to do.</summary>
        internal long RecheckCadences => Interlocked.Read(ref _cadences);

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
        /// Detached, and it drains. The previous shape took a slot per socket and stopped at the first
        /// refusal, so exactly eight genuinely asynchronous checks ran per heartbeat however many were due
        /// and however quickly they finished - a slot freed a millisecond later sat idle until the next
        /// tick. With a hundred live sockets the hundredth was re-checked minutes after it was promised,
        /// which makes the documented interval a fiction rather than a cadence.
        ///
        /// Eight workers pull from the batch until it is empty or the next heartbeat is due. Whatever is
        /// left keeps its old check time, so it is at the front of the queue next cadence.
        /// </summary>
        void StartDueRechecks(DateTimeOffset now, TimeSpan recheck, TimeSpan cadence, CancellationToken stopping)
        {
            V2Connection[] batch = SelectOldestDue(now, recheck, options.Value.MaxRechecksPerCadence);
            if (batch.Length == 0) return;

            Interlocked.Increment(ref _cadences);
            _ = DrainAsync(batch, now, cadence, stopping);
        }

        /// <summary>
        /// The oldest-checked due sockets, up to <paramref name="budget"/>, without sorting the rest.
        ///
        /// Oldest first is what makes it fair: a socket that did not fit in one cadence keeps the oldest
        /// check time on the host and goes to the front of the next one. Sorting every due connection to
        /// learn that is work proportional to the whole host, repeated every heartbeat, to choose a few
        /// hundred - so the candidates are walked once against a buffer of the best so far instead.
        /// </summary>
        V2Connection[] SelectOldestDue(DateTimeOffset now, TimeSpan recheck, int budget)
        {
            var chosen = new List<V2Connection>(Math.Min(budget, 64));
            var newest = -1;

            foreach (V2Connection connection in registry.Snapshot())
            {
                if (!connection.IsAuthenticated || connection.IsClosed) continue;
                if (!IsRecheckDue(connection, now, recheck)) continue;

                Interlocked.Increment(ref _candidates);

                if (chosen.Count < budget)
                {
                    chosen.Add(connection);
                    if (newest < 0
                        || connection.LastCredentialCheck > chosen[newest].LastCredentialCheck)
                        newest = chosen.Count - 1;

                    continue;
                }

                // The common case once the buffer is full: one comparison against the newest thing in it.
                if (connection.LastCredentialCheck >= chosen[newest].LastCredentialCheck) continue;

                chosen[newest] = connection;
                newest = 0;
                for (var i = 1; i < chosen.Count; i++)
                    if (chosen[i].LastCredentialCheck > chosen[newest].LastCredentialCheck) newest = i;
            }

            return chosen.ToArray();
        }

        /// <summary>
        /// Runs one batch of re-checks, eight at a time, until it is done or the cadence runs out.
        ///
        /// The deadline is the next heartbeat: a pass that is still going when the following one is due has
        /// fallen behind, and carrying on would stack passes on top of each other. What it did not reach is
        /// simply not stamped, so the next pass starts with exactly those sockets.
        /// </summary>
        async Task DrainAsync(
            V2Connection[] batch, DateTimeOffset now, TimeSpan cadence, CancellationToken stopping)
        {
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(stopping);
            deadline.CancelAfter(cadence);

            var pool = new ParallelOptions
            {
                MaxDegreeOfParallelism = MaxConcurrentRechecks,
                CancellationToken = deadline.Token,
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
                if (!stopping.IsCancellationRequested)
                    logger.LogWarning(
                        "A credential re-check pass ran out of cadence with work left; it resumes next tick");
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
        async Task RecheckAsync(V2Connection connection, DateTimeOffset now, CancellationToken stopping)
        {
            try
            {
                if (connection.CredentialHash is not byte[] hash) return;
                if (connection.MatchId is not Guid matchId) return;

                using var deadline = CancellationTokenSource.CreateLinkedTokenSource(stopping);
                deadline.CancelAfter(RecheckTimeout);

                bool valid = await credentials
                    .IsStillValidAsync(hash, matchId, now, deadline.Token).ConfigureAwait(false);

                if (valid) return;

                logger.LogInformation("Closing a v2 socket whose credential is no longer valid");
                _ = connection.CloseAsync(CredentialCloseStatus, RevokedReason);
            }
            catch (OperationCanceledException)
            {
                if (!stopping.IsCancellationRequested)
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

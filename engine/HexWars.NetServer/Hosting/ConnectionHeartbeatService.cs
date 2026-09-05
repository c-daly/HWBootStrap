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
        /// Re-checks this loop will even consider starting in one pass.
        ///
        /// A cap on the work, not only on the concurrency. A host that has just come up finds every socket
        /// due at once, and building a task per socket - each with its own linked token source and timer -
        /// is thousands of allocations before a single query has run, on a tick that has a ping to send.
        /// Four times the concurrency is enough to keep the slots busy and nothing like enough to notice.
        /// </summary>
        internal const int MaxRechecksPerCadence = MaxConcurrentRechecks * 4;

        /// <summary>How long one re-check is given before it is abandoned until the next cadence.</summary>
        internal static readonly TimeSpan RecheckTimeout = TimeSpan.FromSeconds(5);

        readonly SemaphoreSlim _recheckSlots = new(MaxConcurrentRechecks, MaxConcurrentRechecks);

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
                    StartDueRechecks(now, recheck, stoppingToken);

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
        /// Starts as many due re-checks as there are free slots, oldest first.
        ///
        /// Oldest first is what makes it fair. A socket that did not get a slot is not stamped, so it keeps
        /// the oldest check time on the host and is at the front of the queue next cadence - without that,
        /// a busy host could starve the same sockets forever while re-checking the same lucky ones.
        ///
        /// The slot is taken BEFORE any work is created. Creating a task that will immediately queue for a
        /// slot is the amplification this cap exists to prevent: the work would be allocated whether or not
        /// there was anywhere to run it.
        /// </summary>
        void StartDueRechecks(DateTimeOffset now, TimeSpan recheck, CancellationToken stopping)
        {
            List<V2Connection> due = registry.Snapshot()
                .Where(c => c.IsAuthenticated && !c.IsClosed && IsRecheckDue(c, now, recheck))
                .OrderBy(c => c.LastCredentialCheck)
                .Take(MaxRechecksPerCadence)
                .ToList();

            foreach (V2Connection connection in due)
            {
                if (!_recheckSlots.Wait(0)) break;

                // Stamped only now, when the check is certain to run. A socket that was skipped keeps its
                // place at the front of the queue.
                connection.LastCredentialCheck = now;
                _ = RecheckAsync(connection, now, stopping);
            }
        }

        /// <summary>
        /// Asks the store whether a live socket still has a credential, and closes it when it does not.
        ///
        /// Bounded and deadlined on purpose. Revocation is a question only the store can answer, so it
        /// costs a round trip per socket, and a host with a thousand of them must not turn one heartbeat
        /// into a thousand simultaneous queries. Anything that does not answer inside the deadline leaves
        /// the socket exactly as it was: an unanswered question is not evidence against the player, and
        /// closing every live socket over a database having a bad minute would be a self-inflicted outage.
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
            finally
            {
                _recheckSlots.Release();
            }
        }
    }
}

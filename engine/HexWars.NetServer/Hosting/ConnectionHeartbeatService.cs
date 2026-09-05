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

                        if (await IsCredentialDeadAsync(connection, now, recheck).ConfigureAwait(false))
                            continue;

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

        /// <summary>
        /// Closes a socket whose credential has stopped being one, and says whether it did.
        ///
        /// Expiry is arithmetic and is checked every tick; revocation is a question only the store can
        /// answer, so it is asked on a slower cadence. Without either, a credential that was withdrawn -
        /// by the same player reconnecting elsewhere, or simply by time - leaves a socket that was
        /// legitimate when it opened and holds a seat for as long as the client cares to keep it.
        /// </summary>
        async Task<bool> IsCredentialDeadAsync(
            V2Connection connection, DateTimeOffset now, TimeSpan recheck)
        {
            if (connection.CredentialExpiresAt is DateTimeOffset expiresAt && expiresAt <= now)
            {
                logger.LogInformation("Closing a v2 socket whose credential has expired");
                _ = connection.CloseAsync(CredentialCloseStatus, ExpiredReason);
                return true;
            }

            if (connection.CredentialHash is not byte[] hash) return false;
            if (connection.MatchId is not Guid matchId) return false;
            if (now - connection.LastCredentialCheck < recheck) return false;

            connection.LastCredentialCheck = now;

            bool valid;
            try
            {
                valid = await credentials.IsStillValidAsync(hash, matchId, now).ConfigureAwait(false);
            }
            catch (Exception failure)
            {
                // A store that will not answer is not evidence against the player. Closing every live
                // socket over one failed query would turn a database blip into a mass disconnect.
                logger.LogWarning(failure, "A live credential could not be re-checked");
                return false;
            }

            if (valid) return false;

            logger.LogInformation("Closing a v2 socket whose credential is no longer valid");
            _ = connection.CloseAsync(CredentialCloseStatus, RevokedReason);
            return true;
        }
    }
}

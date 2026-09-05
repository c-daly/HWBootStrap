using System.Net.WebSockets;
using System.Text;
using HexWars.Engine;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Endpoints;
using HexWars.NetServer.Runtime;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// The protocol-v2 socket: <c>/ws/v2</c>.
    ///
    /// Everything about a match lives in <see cref="DurableMatchCoordinator"/>; this file is the edge that
    /// turns a socket into calls on it. The one rule it owns by itself is the handshake: a v2 socket is
    /// worth nothing until its first frame proves which seat of which match it is, so the first frame MUST
    /// be AUTH, it has a deadline, and an unauthenticated socket never reaches the coordinator at all.
    ///
    /// The AUTH frame carries a bearer credential, so it is never logged - not the frame, not the payload,
    /// not at Trace. What a log gets is that a socket authenticated and the first eight characters of the
    /// match id it authenticated into.
    /// </summary>
    internal static class ProtocolV2WebSocketServer
    {
        /// <summary>The same cap the legacy socket uses. A frame this big is not a command; the largest
        /// thing a v2 client ever sends is a barracks catalog, which is a few hundred bytes.</summary>
        internal const int MaxIncomingBytes = 64 * 1024;

        internal const string AuthMessage = "AUTH";
        internal const string PongMessage = "PONG";
        internal const string AuthFailPrefix = "AUTH FAIL ";

        internal const string TextFramesOnly = "text frames only";

        internal const int CloseNormal = 1000;
        internal const int CloseStale = 1001;
        internal const int ClosePolicy = 1008;
        internal const int CloseTooBig = 1009;

        /// <summary>The close a socket gets for sending data this protocol has no reading for.</summary>
        internal const int CloseUnsupportedData = 1003;

        /// <summary>Handshakes this process will validate at once, across every socket.</summary>
        internal const int MaxConcurrentValidations = 64;

        /// <summary>How long an AUTH frame waits for a validation slot before it is told unavailable.</summary>
        internal static readonly TimeSpan ValidationQueueWindow = TimeSpan.FromSeconds(2);

        /// <summary>
        /// The validation slots, shared by every v2 socket on this host.
        ///
        /// An AUTH frame is at least one database round trip made on behalf of somebody who has proved
        /// nothing, and a socket is cheap to open. Without a ceiling a burst of handshakes points the whole
        /// connection pool at the credential table and starves the matches already being played, which is a
        /// far worse outcome than telling a few newcomers to try again in a moment.
        /// </summary>
        static readonly SemaphoreSlim ValidationSlots =
            new(MaxConcurrentValidations, MaxConcurrentValidations);

        const string LoggerCategory = "HexWars.NetServer.Hosting.ProtocolV2WebSocketServer";

        /// <summary>How much of an inbound payload a debug line is allowed to carry.</summary>
        const int LoggedPayloadLength = 60;

        /// <summary>Enough of a match id to follow one game through a log, and not enough to be a handle on
        /// anything else.</summary>
        const int LoggedMatchIdLength = 8;

        enum FrameKind
        {
            Text,
            Closed,
            TooBig,

            /// <summary>A binary frame. Every message in this protocol is UTF-8 text, so bytes arriving as
            /// binary are not a message that failed to parse - they are a client speaking something else.</summary>
            Unsupported,
        }

        readonly record struct Frame(FrameKind Kind, string Text);

        /// <summary>Accept a socket, make it prove which seat it is, then pump frames until it closes.</summary>
        public static async Task Handle(HttpContext context)
        {
            if (!context.WebSockets.IsWebSocketRequest)
            {
                context.Response.StatusCode = StatusCodes.Status400BadRequest;
                return;
            }

            MatchHostingOptions options =
                context.RequestServices.GetRequiredService<IOptions<MatchHostingOptions>>().Value;

            // Before Accept, which is what makes it a defence: a cross-site page that has been upgraded is
            // already talking to this server, and refusing it afterwards would refuse it too late.
            if (!OriginPolicy.IsAllowed(context, options.AllowedWebOrigins))
            {
                context.Response.StatusCode = StatusCodes.Status403Forbidden;
                return;
            }

            var registry = context.RequestServices.GetRequiredService<V2ConnectionRegistry>();
            var coordinator = context.RequestServices.GetRequiredService<DurableMatchCoordinator>();
            var time = context.RequestServices.GetRequiredService<TimeProvider>();
            var environment = context.RequestServices.GetRequiredService<IWebHostEnvironment>();
            ILogger logger = context.RequestServices
                .GetRequiredService<ILoggerFactory>()
                .CreateLogger(LoggerCategory);

            // Before the upgrade, because the very next frame on this socket is a bearer credential. In
            // production TLS is terminated at the platform proxy and the scheme is forwarded, so a request
            // that still says http after the forwarded-headers middleware has run either arrived over
            // plaintext or came round the side of that proxy. Neither is a socket to accept a credential on.
            if (environment.IsProduction() && !context.Request.IsHttps)
            {
                logger.LogWarning("Refused a plaintext v2 socket: this host only accepts https in production");
                context.Response.StatusCode = StatusCodes.Status400BadRequest;
                return;
            }

            string remoteIp =
                context.Connection.RemoteIpAddress?.ToString() ?? SteamMatchEndpoints.UnknownCaller;

            // Also before the upgrade, and as a reservation rather than a count. An accepted socket costs a
            // receive buffer, a writer task and a registry entry before it has proved anything - and a
            // ceiling checked against the sockets that already exist is no ceiling at all when a hundred
            // upgrades arrive together and every one of them reads the same low number.
            if (!registry.TryReserve(remoteIp))
            {
                logger.LogWarning(
                    "Refused a v2 socket: this address already holds {Cap} of them", options.MaxSocketsPerIp);
                context.Response.StatusCode = StatusCodes.Status429TooManyRequests;
                return;
            }

            WebSocket socket;
            try
            {
                socket = await context.WebSockets.AcceptWebSocketAsync().ConfigureAwait(false);
            }
            catch
            {
                // The reservation never became a connection, so nothing else will ever give it back.
                registry.Release(remoteIp);
                throw;
            }

            var connection = new V2Connection(
                Guid.NewGuid().ToString("N"),
                socket,
                remoteIp,
                options.OutboundQueueCapacity,
                options.OutboundQueueBytes,
                time.GetUtcNow());

            // Registered before the handshake, because the coordinator answers AUTH by sending SEAT through
            // the sink: a connection the sink cannot find would authenticate into silence. Add takes over
            // the reservation above; Remove is what hands it back.
            registry.Add(connection);

            try
            {
                if (!await AuthenticateAsync(context, connection, coordinator, options, time, logger)
                        .ConfigureAwait(false))
                    return;

                await PumpAsync(context, connection, coordinator, time, logger).ConfigureAwait(false);
            }
            finally
            {
                // Said first, and said however this socket ended. It is what lets a close decided
                // elsewhere - the heartbeat, an eviction - stop waiting for a receive loop that has already
                // stopped, and it always runs, so the registry entry always goes.
                connection.ReceiveLoopEnded();

                await coordinator.DisconnectAsync(connection.Id).ConfigureAwait(false);
                registry.Remove(connection.Id);
                await connection.CloseAsync(CloseNormal, "bye").ConfigureAwait(false);
            }
        }

        /// <summary>The handshake. True when the socket now holds a seat; false when it has been closed.</summary>
        static async Task<bool> AuthenticateAsync(
            HttpContext context,
            V2Connection connection,
            DurableMatchCoordinator coordinator,
            MatchHostingOptions options,
            TimeProvider time,
            ILogger logger)
        {
            CancellationToken aborted = context.RequestAborted;
            var throttle = context.RequestServices.GetRequiredService<AuthFailureThrottle>();

            using var deadline = new CancellationTokenSource(
                TimeSpan.FromSeconds(options.AuthFrameTimeoutSeconds), time);
            using CancellationTokenSource window =
                CancellationTokenSource.CreateLinkedTokenSource(aborted, deadline.Token);

            Frame first = await ReceiveAsync(connection.WebSocket, window.Token).ConfigureAwait(false);

            if (first.Kind == FrameKind.TooBig)
            {
                await connection.CloseFromReceiveLoopAsync(CloseTooBig, "message too large")
                    .ConfigureAwait(false);
                return false;
            }

            if (first.Kind == FrameKind.Unsupported)
            {
                // Closed without a word. The bytes are not dispatched and not decoded, so a binary frame
                // that happens to spell a valid AUTH authenticates nothing.
                logger.LogDebug("Closed a v2 socket that opened with a binary frame");
                await connection.CloseFromReceiveLoopAsync(CloseUnsupportedData, TextFramesOnly)
                    .ConfigureAwait(false);
                return false;
            }

            if (first.Kind == FrameKind.Closed)
            {
                // A socket that never spoke costs a connection slot and holds no seat. Closing it is the
                // whole point of the deadline; a client that hung up on its own needs nothing said to it.
                if (deadline.IsCancellationRequested && !aborted.IsCancellationRequested)
                {
                    logger.LogDebug("Closed a v2 socket that never sent AUTH");
                    await connection.CloseFromReceiveLoopAsync(ClosePolicy, "auth timeout")
                        .ConfigureAwait(false);
                }

                return false;
            }

            NetMessage opening = NetProtocol.Parse(first.Text);

            if (!string.Equals(opening.Type, AuthMessage, StringComparison.Ordinal))
            {
                // A byte count and nothing else. The type looked safe to log because a well-behaved client
                // puts a keyword there - but this branch is the one a MISBEHAVING client reaches, and a
                // client that opened with its bare credential would have that credential written to the log
                // by the code that refused it. Truncating does not help: a prefix of a secret is a secret.
                logger.LogDebug(
                    "A v2 socket opened with something other than AUTH, {Bytes} bytes", first.Text.Length);
                throttle.RecordFailure(connection.RemoteIp);
                await RefuseAsync(connection, DurableMatchCoordinator.AuthFailInvalid).ConfigureAwait(false);
                return false;
            }

            string[] tokens = opening.Payload.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length != 2)
            {
                logger.LogDebug("A v2 socket sent an AUTH frame that is not a match id and a credential");
                throttle.RecordFailure(connection.RemoteIp);
                await RefuseAsync(connection, DurableMatchCoordinator.AuthFailInvalid).ConfigureAwait(false);
                return false;
            }

            // Before the credential service, which is a database round trip made for a caller who has
            // proved nothing. Guessing 256 bits is not the threat; a client wedged in a retry loop on a
            // credential that will never work is, and every attempt costs this server a query.
            if (throttle.IsThrottled(connection.RemoteIp))
            {
                logger.LogWarning("Refused an AUTH frame from a caller that has spent its failures");
                await RefuseAsync(connection, DurableMatchCoordinator.AuthFailInvalid).ConfigureAwait(false);
                return false;
            }

            bool admitted;
            try
            {
                admitted = await ValidationSlots
                    .WaitAsync(ValidationQueueWindow, aborted).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                return false;
            }

            if (!admitted)
            {
                logger.LogWarning(
                    "Refused an AUTH frame: {Slots} handshakes are already being validated",
                    MaxConcurrentValidations);
                await RefuseAsync(connection, DurableMatchCoordinator.AuthFailUnavailable).ConfigureAwait(false);
                return false;
            }

            DurableMatchCoordinator.AuthOutcome outcome;
            try
            {
                outcome = await coordinator
                    .AuthenticateAsync(connection.Id, tokens[0], tokens[1], aborted)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                return false;
            }
            catch (Exception failure)
            {
                // The coordinator turns the failures it expects into a fail code of its own, so anything
                // that reaches here is a bug. The player still gets an answer they can retry on, and it is
                // not counted against them: this one is ours.
                logger.LogError(failure, "A v2 handshake failed unexpectedly");
                await RefuseAsync(connection, DurableMatchCoordinator.AuthFailUnavailable).ConfigureAwait(false);
                return false;
            }
            finally
            {
                ValidationSlots.Release();
            }

            if (!outcome.Ok)
            {
                string code = outcome.FailCode ?? DurableMatchCoordinator.AuthFailInvalid;

                // Only a refusal counts. Unavailable means this host could not answer, and counting it
                // would lock the players of a match out during exactly the outage they are retrying
                // through.
                if (string.Equals(code, DurableMatchCoordinator.AuthFailInvalid, StringComparison.Ordinal))
                    throttle.RecordFailure(connection.RemoteIp);

                await RefuseAsync(connection, code).ConfigureAwait(false);
                return false;
            }

            connection.MatchId = Guid.Parse(tokens[0]);
            connection.LastInbound = time.GetUtcNow();

            // Kept for the life of the socket, because the handshake is a moment and a match is an hour: a
            // credential that expires or is revoked mid-game has to end this connection then, not at
            // whatever reconnect the client happens to make next.
            connection.CredentialHash = outcome.CredentialHash;
            connection.CredentialExpiresAt = outcome.CredentialExpiresAt;
            connection.LastCredentialCheck = time.GetUtcNow();

            logger.LogInformation(
                "A v2 socket took seat {Seat} of match {MatchId}",
                outcome.Seat,
                Truncate(tokens[0], LoggedMatchIdLength));

            return true;
        }

        /// <summary>Frames from a seated socket, until it closes or breaks a rule.</summary>
        static async Task PumpAsync(
            HttpContext context,
            V2Connection connection,
            DurableMatchCoordinator coordinator,
            TimeProvider time,
            ILogger logger)
        {
            CancellationToken aborted = context.RequestAborted;

            while (!aborted.IsCancellationRequested && !connection.IsClosed)
            {
                Frame frame = await ReceiveAsync(connection.WebSocket, aborted).ConfigureAwait(false);

                if (frame.Kind == FrameKind.TooBig)
                {
                    await connection.CloseFromReceiveLoopAsync(CloseTooBig, "message too large")
                        .ConfigureAwait(false);
                    return;
                }

                if (frame.Kind == FrameKind.Unsupported)
                {
                    logger.LogDebug("Closed a seated v2 socket that sent a binary frame");
                    await connection.CloseFromReceiveLoopAsync(CloseUnsupportedData, TextFramesOnly)
                        .ConfigureAwait(false);
                    return;
                }

                if (frame.Kind == FrameKind.Closed) return;

                // Any inbound frame is liveness, PONG included and PONG especially: the heartbeat asks
                // whether the client is there, not what it has to say.
                connection.LastInbound = time.GetUtcNow();

                if (frame.Text.Length == 0) continue;
                if (string.Equals(frame.Text, PongMessage, StringComparison.Ordinal)) continue;

                NetMessage inbound = NetProtocol.Parse(frame.Text);
                if (inbound.Type is "CMD" or "CATALOG")
                {
                    // The type and a byte count, never the payload. A client that puts its credential in
                    // the wrong frame would otherwise have it written to the log by the code that read it,
                    // and no amount of truncation makes that safe - the credential is 43 characters and
                    // any prefix of it is still a prefix of a secret.
                    logger.LogDebug(
                        "RECV {Type} {Bytes} bytes", inbound.Type, inbound.Payload.Length);
                }

                await coordinator.ReceiveAsync(connection.Id, frame.Text, aborted).ConfigureAwait(false);
            }
        }

        /// <summary>Names the refusal, then closes. The frame goes out first because the close drains the
        /// outbound queue: a client told only by a close status cannot tell invalid from unavailable.</summary>
        static async Task RefuseAsync(V2Connection connection, string failCode)
        {
            connection.TryEnqueue(AuthFailPrefix + failCode);
            await connection.CloseFromReceiveLoopAsync(ClosePolicy, "auth failed").ConfigureAwait(false);
        }

        /// <summary>One whole message, or why there is not one. The size cap is enforced as the message
        /// arrives rather than after it, so a hostile client cannot make this process buffer megabytes
        /// before it is refused.</summary>
        static async Task<Frame> ReceiveAsync(WebSocket socket, CancellationToken ct)
        {
            var buffer = new byte[8192];
            using var message = new MemoryStream();
            WebSocketReceiveResult result;

            do
            {
                try
                {
                    result = await socket
                        .ReceiveAsync(new ArraySegment<byte>(buffer), ct)
                        .ConfigureAwait(false);
                }
                catch
                {
                    return new Frame(FrameKind.Closed, string.Empty);
                }

                if (result.MessageType == WebSocketMessageType.Close)
                    return new Frame(FrameKind.Closed, string.Empty);

                // On every fragment, not only the first. A message can change type part-way through only
                // because a client is doing something wrong, and reading the rest of it to find out what
                // would mean decoding bytes this protocol has no meaning for - as text, at that.
                if (result.MessageType == WebSocketMessageType.Binary)
                    return new Frame(FrameKind.Unsupported, string.Empty);

                message.Write(buffer, 0, result.Count);
                if (message.Length > MaxIncomingBytes) return new Frame(FrameKind.TooBig, string.Empty);
            }
            while (!result.EndOfMessage);

            return new Frame(FrameKind.Text, Encoding.UTF8.GetString(message.ToArray()));
        }

        static string Truncate(string value, int length) =>
            value.Length <= length ? value : value[..length];
    }
}

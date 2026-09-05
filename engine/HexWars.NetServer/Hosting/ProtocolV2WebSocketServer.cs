using System.Net.WebSockets;
using System.Text;
using HexWars.Engine;
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

        internal const int CloseNormal = 1000;
        internal const int CloseStale = 1001;
        internal const int ClosePolicy = 1008;
        internal const int CloseTooBig = 1009;

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
            ILogger logger = context.RequestServices
                .GetRequiredService<ILoggerFactory>()
                .CreateLogger(LoggerCategory);

            WebSocket socket = await context.WebSockets.AcceptWebSocketAsync().ConfigureAwait(false);

            var connection = new V2Connection(
                Guid.NewGuid().ToString("N"),
                socket,
                context.Connection.RemoteIpAddress?.ToString() ?? SteamMatchEndpoints.UnknownCaller,
                options.OutboundQueueCapacity,
                time.GetUtcNow());

            // Registered before the handshake, because the coordinator answers AUTH by sending SEAT through
            // the sink: a connection the sink cannot find would authenticate into silence.
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

            using var deadline = new CancellationTokenSource(
                TimeSpan.FromSeconds(options.AuthFrameTimeoutSeconds), time);
            using CancellationTokenSource window =
                CancellationTokenSource.CreateLinkedTokenSource(aborted, deadline.Token);

            Frame first = await ReceiveAsync(connection.WebSocket, window.Token).ConfigureAwait(false);

            if (first.Kind == FrameKind.TooBig)
            {
                await connection.CloseAsync(CloseTooBig, "message too large").ConfigureAwait(false);
                return false;
            }

            if (first.Kind == FrameKind.Closed)
            {
                // A socket that never spoke costs a connection slot and holds no seat. Closing it is the
                // whole point of the deadline; a client that hung up on its own needs nothing said to it.
                if (deadline.IsCancellationRequested && !aborted.IsCancellationRequested)
                {
                    logger.LogDebug("Closed a v2 socket that never sent AUTH");
                    await connection.CloseAsync(ClosePolicy, "auth timeout").ConfigureAwait(false);
                }

                return false;
            }

            NetMessage opening = NetProtocol.Parse(first.Text);

            if (!string.Equals(opening.Type, AuthMessage, StringComparison.Ordinal))
            {
                // The TYPE only, truncated. A client that sent its credential in the wrong frame would
                // otherwise have it written to the log by the code that refused it.
                logger.LogDebug(
                    "A v2 socket opened with {Type} rather than AUTH", Truncate(opening.Type, LoggedMatchIdLength));
                await RefuseAsync(connection, DurableMatchCoordinator.AuthFailInvalid).ConfigureAwait(false);
                return false;
            }

            string[] tokens = opening.Payload.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length != 2)
            {
                logger.LogDebug("A v2 socket sent an AUTH frame that is not a match id and a credential");
                await RefuseAsync(connection, DurableMatchCoordinator.AuthFailInvalid).ConfigureAwait(false);
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
                // that reaches here is a bug. The player still gets an answer they can retry on.
                logger.LogError(failure, "A v2 handshake failed unexpectedly");
                await RefuseAsync(connection, DurableMatchCoordinator.AuthFailUnavailable).ConfigureAwait(false);
                return false;
            }

            if (!outcome.Ok)
            {
                await RefuseAsync(connection, outcome.FailCode ?? DurableMatchCoordinator.AuthFailInvalid)
                    .ConfigureAwait(false);
                return false;
            }

            connection.MatchId = Guid.Parse(tokens[0]);
            connection.LastInbound = time.GetUtcNow();

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
                    await connection.CloseAsync(CloseTooBig, "message too large").ConfigureAwait(false);
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
                    logger.LogDebug(
                        "RECV {Type} {Payload}", inbound.Type, Truncate(inbound.Payload, LoggedPayloadLength));
                }

                await coordinator.ReceiveAsync(connection.Id, frame.Text, aborted).ConfigureAwait(false);
            }
        }

        /// <summary>Names the refusal, then closes. The frame goes out first because the close drains the
        /// outbound queue: a client told only by a close status cannot tell invalid from unavailable.</summary>
        static async Task RefuseAsync(V2Connection connection, string failCode)
        {
            connection.TryEnqueue(AuthFailPrefix + failCode);
            await connection.CloseAsync(ClosePolicy, "auth failed").ConfigureAwait(false);
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

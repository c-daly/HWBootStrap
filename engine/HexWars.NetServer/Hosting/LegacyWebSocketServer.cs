using System.Collections.Concurrent;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Channels;
using HexWars.Engine;
using HexWars.NetServer.Configuration;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// Thin WebSocket adapter over <see cref="MatchHub"/>: turns sockets into Connect/Receive calls and
    /// routes the resulting <see cref="Outbound"/> messages back to the right connections. All game logic
    /// lives in the (unit-tested) engine; this file is just plumbing. This is the v1 lobby that the WebGL
    /// build talks to (room codes in the query string); it is mapped only when LOBBY_PROVIDER contains
    /// Legacy, and its behaviour is deliberately frozen.
    /// </summary>
    internal static class LegacyWebSocketServer
    {
        static readonly ConcurrentDictionary<string, Conn> Conns = new();
        static readonly MatchHub Hub = new(
            setup => GameFactory.Build(setup),
            newCatalogGame: (setup, p0, p1) => GameFactory.Build(setup, p0, p1));
        static readonly object HubLock = new();
        const int MaxIncomingBytes = 64 * 1024;

        const string LoggerCategory = "HexWars.NetServer.Hosting.LegacyWebSocketServer";

        /// <summary>How much of a payload a debug line is allowed to carry. The v1 lobby has no secrets in
        /// its message set, but a room code and a command wire are still traffic, and a log that reproduced
        /// every frame in full would be a transcript of every game played on this host.</summary>
        const int LoggedPayloadLength = 60;

        /// <summary>Accept a socket, seat it in the room from ?room=, then pump messages until it closes.</summary>
        internal static async Task Handle(HttpContext ctx)
        {
            if (!ctx.WebSockets.IsWebSocketRequest) { ctx.Response.StatusCode = 400; return; }
            var hosting = ctx.RequestServices.GetRequiredService<IOptions<MatchHostingOptions>>().Value;
            if (!OriginPolicy.IsAllowed(ctx, hosting.AllowedWebOrigins)) { ctx.Response.StatusCode = 403; return; }
            string room = NormalizeRoom(ctx.Request.Query["room"].ToString());
            bool isPrivate = ctx.Request.Query["private"].ToString() == "1";
            var setup = ParseSetup(ctx.Request.Query["setup"].ToString());
            bool joinOnly = ctx.Request.Query["join"].ToString() == "1";
            string? token = ctx.Request.Query["token"].ToString();
            if (string.IsNullOrWhiteSpace(token)) token = null; // absent/garbled -> fresh identity (today's behavior)

            // Through ILogger rather than the console, so these lines carry a level and a category
            // and can be turned off in a deployment. At Debug, because a line per frame per connection is a
            // debugging aid rather than something a production log should carry.
            ILogger logger = ctx.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger(LoggerCategory);

            var socket = await ctx.WebSockets.AcceptWebSocketAsync();
            var conn = new Conn(Guid.NewGuid().ToString("N"), socket);
            Conns[conn.Id] = conn;
            logger.LogDebug(
                "v1 CONNECT room {Room} connection {Connection} setup {Mode} {Width}x{Height} "
                + "points {Points} seed {Seed}, {Total} live",
                room, conn.Id[..8], setup.Mode, setup.Width, setup.Height,
                setup.StartingPoints, setup.Seed, Conns.Count);
            try
            {
                Locked(logger, () => Hub.Connect(room, conn.Id, setup, isPrivate, joinOnly, token));
                while (socket.State == WebSocketState.Open)
                {
                    string? text = await Receive(socket);
                    if (text is null) break;          // closed / errored / over the size cap
                    if (text.Length == 0) continue;
                    logger.LogDebug(
                        "v1 RECV room {Room} connection {Connection}: {Payload}",
                        room, conn.Id[..8], Truncated(text));
                    Locked(logger, () => Hub.Receive(room, conn.Id, text));
                }
            }
            finally
            {
                logger.LogDebug(
                    "v1 DISCONNECT room {Room} connection {Connection}", room, conn.Id[..8]);
                // Only drops this connection's membership/token mapping — a Started room's seat itself is
                // HELD (see MatchHub) for HoldWindowTicks so the same token can reclaim it on reconnect.
                Locked(logger, () => Hub.Disconnect(room, conn.Id));
                Conns.TryRemove(conn.Id, out _);
                conn.Close();
                if (socket.State == WebSocketState.Open)
                    try { await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", default); } catch { }
            }
        }

        // MatchHub isn't thread-safe; serialize all access. The outbound Enqueue below is synchronous
        // (a Channel Write never blocks/awaits), so it runs INSIDE the lock along with the hub call —
        // this is what closes the APPLY-ordering race: two concurrent Handle() calls can no longer
        // interleave their actual sends, because "compute the outbound messages" and "hand them to each
        // connection's own ordered queue" are now one atomic, lock-serialized step (audit N2).
        static void Locked(ILogger logger, Func<IReadOnlyList<Outbound>> f)
        {
            lock (HubLock)
            {
                var outs = f();
                foreach (var o in outs)
                {
                    logger.LogDebug(
                        "v1 SEND connection {Connection}: {Payload}",
                        o.ConnectionId[..8], Truncated(o.Message));
                    if (Conns.TryGetValue(o.ConnectionId, out var c)) c.Enqueue(o.Message);
                }
            }
        }

        /// <summary>As much of a payload as a debug line may carry, with an ellipsis when there was more.
        /// The same rule the v2 handler follows, so the two logs read alike.</summary>
        static string Truncated(string payload) =>
            payload.Length > LoggedPayloadLength ? payload[..LoggedPayloadLength] + "\u2026" : payload;

        static async Task<string?> Receive(WebSocket socket)
        {
            var buf = new byte[8192];
            using var ms = new MemoryStream();
            WebSocketReceiveResult res;
            do
            {
                try { res = await socket.ReceiveAsync(new ArraySegment<byte>(buf), CancellationToken.None); }
                catch { return null; }
                if (res.MessageType == WebSocketMessageType.Close) return null;
                ms.Write(buf, 0, res.Count);
                if (ms.Length > MaxIncomingBytes)
                {
                    try { await socket.CloseAsync(WebSocketCloseStatus.MessageTooBig, "message too large", CancellationToken.None); } catch { }
                    return null;
                }
            } while (!res.EndOfMessage);
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        /// <summary>Parse the host's lobby picks from the connect query (?setup=...); default if absent/bad.</summary>
        static GameSetup ParseSetup(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw)) return GameSetup.Default;
            try { return GameSetup.Parse(raw); } catch { return GameSetup.Default; }
        }

        /// <summary>Uppercase alphanumerics only, capped at 16 — so "kq7kp", " KQ7KP " and a pasted
        /// URL fragment all land in the same room, and a hostile room string can't be huge.</summary>
        internal static string NormalizeRoom(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw)) return "DEFAULT";
            var sb = new StringBuilder();
            foreach (char ch in raw.Trim().ToUpperInvariant())
            {
                if ((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) sb.Append(ch);
                if (sb.Length == 16) break;
            }
            return sb.Length == 0 ? "DEFAULT" : sb.ToString();
        }

        /// <summary>Lobby snapshot for the /games browser and the selftest: a locked read of the hub.</summary>
        internal static IReadOnlyList<OpenGame> OpenGamesSnapshot() { lock (HubLock) return Hub.OpenGames(); }
    }

    /// <summary>One live connection: its id + socket, with an ordered per-connection outbound queue (a
    /// single writer task drains it) instead of a semaphore around SendAsync — Enqueue is synchronous
    /// and never blocks the hub-lock caller (see LegacyWebSocketServer.Locked).</summary>
    sealed class Conn
    {
        public readonly string Id;
        public readonly WebSocket Socket;
        readonly Channel<string> _outbox = Channel.CreateUnbounded<string>();
        readonly Task _writer;

        public Conn(string id, WebSocket socket)
        {
            Id = id;
            Socket = socket;
            _writer = Task.Run(PumpAsync);
        }

        /// <summary>Enqueue a message for this connection's single writer task. Never blocks or throws —
        /// an unbounded channel Write always succeeds synchronously.</summary>
        public void Enqueue(string msg) => _outbox.Writer.TryWrite(msg);

        async Task PumpAsync()
        {
            await foreach (var msg in _outbox.Reader.ReadAllAsync())
            {
                try { await Socket.SendAsync(Encoding.UTF8.GetBytes(msg), WebSocketMessageType.Text, true, CancellationToken.None); }
                catch { /* dead socket; Handle's own receive loop notices and cleans up */ }
            }
        }

        /// <summary>Stop accepting new messages and let the writer task drain what is already queued.</summary>
        public void Close() => _outbox.Writer.TryComplete();
    }
}

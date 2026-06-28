using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using HexWars.Engine;

namespace HexWars.NetServer
{
    /// <summary>
    /// Thin WebSocket adapter over <see cref="MatchHub"/>: turns sockets into Connect/Receive calls and
    /// routes the resulting <see cref="Outbound"/> messages back to the right connections. All game logic
    /// lives in the (unit-tested) engine; this file is just plumbing. Cloud-ready: binds 0.0.0.0:$PORT
    /// when a host injects PORT, and serves the WebGL client from wwwroot when present (single origin).
    /// Run `HexWars.NetServer selftest` to drive two in-process clients through a move and assert.
    /// </summary>
    public static class Program
    {
        static readonly ConcurrentDictionary<string, Conn> Conns = new();
        static readonly MatchHub Hub = new(NewGame);
        static readonly object HubLock = new();

        public static async Task<int> Main(string[] args)
        {
            if (args.Length > 0 && args[0] == "selftest") return await SelfTest.Run();

            var builder = WebApplication.CreateBuilder(args);
            var port = Environment.GetEnvironmentVariable("PORT");
            if (!string.IsNullOrWhiteSpace(port)) builder.WebHost.UseUrls($"http://0.0.0.0:{port}");

            var app = builder.Build();
            app.UseWebSockets();
            app.UseDefaultFiles();   // serve the WebGL client from wwwroot/ when a deploy copies it in
            app.UseStaticFiles();
            app.MapGet("/healthz", () => "ok");
            app.Map("/ws", Handle);
            await app.RunAsync();
            return 0;
        }

        /// <summary>Accept a socket, seat it in the room from ?room=, then pump messages until it closes.</summary>
        internal static async Task Handle(HttpContext ctx)
        {
            if (!ctx.WebSockets.IsWebSocketRequest) { ctx.Response.StatusCode = 400; return; }
            string room = ctx.Request.Query["room"].ToString();
            if (string.IsNullOrWhiteSpace(room)) room = "default";

            var socket = await ctx.WebSockets.AcceptWebSocketAsync();
            var conn = new Conn(Guid.NewGuid().ToString("N"), socket);
            Conns[conn.Id] = conn;
            try
            {
                await Dispatch(Locked(() => Hub.Connect(room, conn.Id)));
                while (socket.State == WebSocketState.Open)
                {
                    string? text = await Receive(socket);
                    if (text is null) break;          // closed / errored
                    if (text.Length == 0) continue;
                    await Dispatch(Locked(() => Hub.Receive(room, conn.Id, text)));
                }
            }
            finally
            {
                Conns.TryRemove(conn.Id, out _);
                if (socket.State == WebSocketState.Open)
                    try { await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", default); } catch { }
            }
        }

        // MatchHub isn't thread-safe; serialize all access. Calls are synchronous and fast (no awaits).
        static IReadOnlyList<Outbound> Locked(Func<IReadOnlyList<Outbound>> f) { lock (HubLock) return f(); }

        static async Task Dispatch(IReadOnlyList<Outbound> outs)
        {
            foreach (var o in outs)
                if (Conns.TryGetValue(o.ConnectionId, out var c))
                    try { await c.Send(o.Message); } catch { /* drop a dead socket; cleanup happens on its own loop */ }
        }

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
            } while (!res.EndOfMessage);
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        /// <summary>The v0 slice game: a generated board with a few distinct units per side, regular annihilation rules.</summary>
        static GameState NewGame()
        {
            var board = new RandomBoardGenerator(BoardGenConfig.Default()).Generate(7);
            int nextId = 1;
            var p0 = Seed(board, PlayerId.Player0, ref nextId);
            var p1 = Seed(board, PlayerId.Player1, ref nextId);
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, 1, nextId);
        }

        static PlayerState Seed(Board board, PlayerId id, ref int nextId)
        {
            var roster = new[]
            {
                new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1), // Brute
                new UnitStats(2, 6, 0, 3, 2, 2, 1, 3, 1), // Striker
                new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1), // Sniper
            };
            var flat = new List<HexCoord>();
            foreach (var c in board.DeploymentZone(id))
                if (board.TileAt(c).Elevation == 0) flat.Add(c);
            flat.Sort((x, y) => x.Q != y.Q ? x.Q - y.Q : x.R - y.R);

            var units = new List<Unit>();
            for (int i = 0; i < roster.Length && i < flat.Count; i++)
                units.Add(new Unit(nextId++, id, roster[i], flat[i], 0));
            return new PlayerState(id, 0, null, units, null);
        }
    }

    /// <summary>One live connection: its id + socket, with sends serialized (one SendAsync at a time).</summary>
    sealed class Conn
    {
        public readonly string Id;
        public readonly WebSocket Socket;
        readonly SemaphoreSlim _send = new(1, 1);

        public Conn(string id, WebSocket socket) { Id = id; Socket = socket; }

        public async Task Send(string msg)
        {
            await _send.WaitAsync();
            try { await Socket.SendAsync(Encoding.UTF8.GetBytes(msg), WebSocketMessageType.Text, true, CancellationToken.None); }
            finally { _send.Release(); }
        }
    }
}

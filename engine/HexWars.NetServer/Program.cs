using System.Collections.Concurrent;
using System.Linq;
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
        static readonly MatchHub Hub = new(GameFactory.Build);
        static readonly object HubLock = new();

        public static async Task<int> Main(string[] args)
        {
            if (args.Length > 0 && args[0] == "selftest") return await SelfTest.Run();

            var builder = WebApplication.CreateBuilder(args);
            var port = Environment.GetEnvironmentVariable("PORT");
            if (!string.IsNullOrWhiteSpace(port)) builder.WebHost.UseUrls($"http://0.0.0.0:{port}");

            var app = builder.Build();
            app.UseWebSockets();
            app.UseDefaultFiles();   // serve the WebGL client (index.html) from wwwroot/ when a deploy copies it in
            // Unity WebGL ships .unityweb/.data/.wasm; without these mappings Kestrel 404s them.
            var types = new Microsoft.AspNetCore.StaticFiles.FileExtensionContentTypeProvider();
            types.Mappings[".unityweb"] = "application/octet-stream"; // gzip payload; the loader decompresses (decompressionFallback)
            types.Mappings[".data"] = "application/octet-stream";
            types.Mappings[".wasm"] = "application/wasm";
            app.UseStaticFiles(new StaticFileOptions { ContentTypeProvider = types });
            app.MapGet("/healthz", () => "ok");
            // The lobby browser: open public games as JSON. Same origin as the WebGL client, no CORS.
            app.MapGet("/games", () =>
            {
                IReadOnlyList<OpenGame> open;
                lock (HubLock) open = Hub.OpenGames();
                return Results.Json(new
                {
                    games = open.Select(g => new
                    {
                        code = g.Code,
                        mode = g.Setup.Mode.ToString(),
                        width = g.Setup.Width,
                        height = g.Setup.Height,
                        fog = g.Setup.Fog,
                        pace = g.Setup.TurnActions,
                        army = g.Setup.ArmySize,
                        ageSeconds = g.AgeSeconds,
                    }).ToArray(),
                });
            });
            app.Map("/ws", Handle);
            await app.RunAsync();
            return 0;
        }

        /// <summary>Accept a socket, seat it in the room from ?room=, then pump messages until it closes.</summary>
        internal static async Task Handle(HttpContext ctx)
        {
            if (!ctx.WebSockets.IsWebSocketRequest) { ctx.Response.StatusCode = 400; return; }
            string room = NormalizeRoom(ctx.Request.Query["room"].ToString());
            bool isPrivate = ctx.Request.Query["private"].ToString() == "1";
            var setup = ParseSetup(ctx.Request.Query["setup"].ToString());

            var socket = await ctx.WebSockets.AcceptWebSocketAsync();
            var conn = new Conn(Guid.NewGuid().ToString("N"), socket);
            Conns[conn.Id] = conn;
            Console.WriteLine($"[ws] CONNECT room={room} id={conn.Id[..8]} setup=({setup.Mode} {setup.Width}x{setup.Height} pts{setup.StartingPoints} seed{setup.Seed}) total={Conns.Count}");
            try
            {
                await Dispatch(Locked(() => Hub.Connect(room, conn.Id, setup, isPrivate)));
                while (socket.State == WebSocketState.Open)
                {
                    string? text = await Receive(socket);
                    if (text is null) break;          // closed / errored
                    if (text.Length == 0) continue;
                    Console.WriteLine($"[ws] RECV  room={room} id={conn.Id[..8]}: {text}");
                    await Dispatch(Locked(() => Hub.Receive(room, conn.Id, text)));
                }
            }
            finally
            {
                Console.WriteLine($"[ws] DISCONNECT room={room} id={conn.Id[..8]}");
                Locked(() => Hub.Disconnect(room, conn.Id)); // free the seat so a refresh/rejoin can re-take it
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
            {
                Console.WriteLine($"[ws] SEND  -> {o.ConnectionId[..8]}: {(o.Message.Length > 60 ? o.Message[..60] + "…" : o.Message)}");
                if (Conns.TryGetValue(o.ConnectionId, out var c))
                    try { await c.Send(o.Message); } catch { /* drop a dead socket; cleanup happens on its own loop */ }
            }
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

        /// <summary>Selftest hook: a locked snapshot of the lobby list.</summary>
        internal static IReadOnlyList<OpenGame> OpenGamesSnapshot() { lock (HubLock) return Hub.OpenGames(); }
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

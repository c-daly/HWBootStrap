using System.Net;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using HexWars.Engine;
using HexWars.NetServer.Endpoints;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using NUnit.Framework;

namespace HexWars.NetServer.Tests.Fixtures
{
    /// <summary>
    /// One seat, played the way a real client plays it: an HTTP allocation, a v2 websocket, and a local
    /// <see cref="GameState"/> fast-forwarded from whatever the server dealt.
    ///
    /// The local state is the point of this class. A restart test that only counted frames would pass
    /// against a server that dealt a different game after the restart, because the frame counts would be
    /// identical. Keeping the state the client would actually be looking at - built from START and advanced
    /// by every APPLY, exactly as the Unity client does - lets a test compare what the two halves of a
    /// restart were playing rather than how much they said.
    ///
    /// It is deliberately thin. Nothing here reimplements a server rule: the seat comes from the allocation
    /// response, the start state comes from START, and every command applied locally is one the server
    /// already broadcast.
    /// </summary>
    public sealed class DurableFlowClient : IAsyncDisposable
    {
        public const string CreateRoute = "/api/v1/steam/matches";

        /// <summary>Long enough that a slow Postgres round trip is not a flake, short enough that a frame
        /// which is never coming fails the test rather than hanging the run.</summary>
        public static readonly TimeSpan DefaultWait = TimeSpan.FromSeconds(20);

        readonly WebApplicationFactory<Program> _host;
        readonly HttpClient _http;

        WebSocket? _socket;

        DurableFlowClient(
            WebApplicationFactory<Program> host, HttpClient http, Guid matchId, int seat, string credential)
        {
            _host = host;
            _http = http;
            MatchId = matchId;
            Seat = seat;
            Credential = credential;
        }

        public Guid MatchId { get; }

        public int Seat { get; }

        /// <summary>The credential this seat was issued. Never reused across a restart: the tests ask for a
        /// new one, which is what a reconnecting client does.</summary>
        public string Credential { get; }

        /// <summary>The dealt start state, once START has arrived.</summary>
        public GameState? Start { get; private set; }

        /// <summary>Where this client believes the game is: <see cref="Start"/> plus every APPLY.</summary>
        public GameState? State { get; private set; }

        /// <summary>Every command this client has seen applied, in the order it saw them.</summary>
        public List<Command> Log { get; } = new();

        /// <summary>The wire form of every applied command, for an assertion about ordering.</summary>
        public IReadOnlyList<string> LogWire => Log.Select(CommandWire.Write).ToArray();

        /// <summary>
        /// The exact bytes a START would carry for the position this client is in: the unplayed start state
        /// plus the full command log. Lossless, unlike a serialisation of the played state, so two clients
        /// whose replay text matches are provably playing the same game from the same beginning.
        /// </summary>
        public string ReplayText => ReplayFile.Write(Start!, Log);

        /// <summary>The played state on its own, for comparing against an independent direct replay.</summary>
        public string StateText => ReplayFile.Write(State!, Array.Empty<Command>());

        // ---- allocation ------------------------------------------------------

        /// <summary>Allocates a match as the lobby owner and takes seat 0.</summary>
        public static async Task<DurableFlowClient> CreateAsync(
            WebApplicationFactory<Program> host, string lobbyId, string ticket)
        {
            ArgumentNullException.ThrowIfNull(host);

            HttpClient http = host.CreateClient();
            using HttpResponseMessage response =
                await http.PostAsJsonAsync(CreateRoute, new { steamLobbyId = lobbyId, ticket });

            return await SeatFromAsync(host, http, response, "create");
        }

        /// <summary>Takes, or retakes, a seat in a match somebody already allocated. This is the call a
        /// client makes after a restart: the seat belongs to the account, so it comes back to the same
        /// one.</summary>
        public static async Task<DurableFlowClient> JoinAsync(
            WebApplicationFactory<Program> host, Guid matchId, string ticket)
        {
            ArgumentNullException.ThrowIfNull(host);

            HttpClient http = host.CreateClient();
            using HttpResponseMessage response = await http.PostAsJsonAsync(
                CreateRoute + "/" + matchId.ToString() + "/join", new { ticket });

            return await SeatFromAsync(host, http, response, "join");
        }

        static async Task<DurableFlowClient> SeatFromAsync(
            WebApplicationFactory<Program> host, HttpClient http, HttpResponseMessage response, string what)
        {
            string text = await response.Content.ReadAsStringAsync();
            if (response.StatusCode != HttpStatusCode.OK)
            {
                http.Dispose();
                Assert.Fail(what + " answered " + ((int)response.StatusCode).ToString() + ": " + text);
            }

            JsonElement body = JsonDocument.Parse(text).RootElement;
            return new DurableFlowClient(
                host,
                http,
                Guid.Parse(body.GetProperty("matchId").GetString()!),
                body.GetProperty("seat").GetInt32(),
                body.GetProperty("joinCredential").GetString()!);
        }

        // ---- the socket ------------------------------------------------------

        /// <summary>Opens the v2 socket without saying anything on it.</summary>
        public async Task OpenAsync()
        {
            WebSocketClient client = _host.Server.CreateWebSocketClient();
            _socket = await client.ConnectAsync(
                new Uri("ws://localhost" + SteamMatchEndpoints.WebSocketPath), CancellationToken.None);
        }

        /// <summary>The handshake frame, sent as-is so a test can assert on how it is refused.</summary>
        public Task SendAuthAsync() => SendAsync("AUTH " + MatchId.ToString() + " " + Credential);

        /// <summary>Opens the socket, authenticates, and confirms the seat this client was allocated.</summary>
        public async Task<DurableFlowClient> ConnectAsync()
        {
            await OpenAsync();
            await SendAuthAsync();

            string seat = await ExpectAsync("SEAT ");
            Assert.That(seat, Is.EqualTo("SEAT " + Seat.ToString()),
                "the seat follows the Steam account, not the socket");

            return this;
        }

        /// <summary>The barracks for this seat. Always the default catalog: these tests are about the
        /// journal, and a per-seat army would only make the start state harder to reason about.</summary>
        public Task SendCatalogAsync() =>
            SendAsync(NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));

        public Task SendCmdAsync(Command command) => SendAsync(NetProtocol.Cmd(command));

        public Task SendAsync(string frame)
        {
            ArgumentNullException.ThrowIfNull(frame);

            return _socket!.SendAsync(
                Encoding.UTF8.GetBytes(frame), WebSocketMessageType.Text, true, CancellationToken.None);
        }

        /// <summary>
        /// The next frame, which must start with <paramref name="prefix"/>.
        ///
        /// PING is answered and skipped rather than returned. The heartbeat runs on the real clock in a host
        /// test, so it is noise every assertion here would otherwise have to account for.
        /// </summary>
        public async Task<string> ExpectAsync(string prefix, TimeSpan? within = null)
        {
            ArgumentNullException.ThrowIfNull(prefix);

            DateTimeOffset deadline = DateTimeOffset.UtcNow.Add(within ?? DefaultWait);

            while (true)
            {
                string? frame = await ReceiveAsync(deadline - DateTimeOffset.UtcNow);
                if (frame is null)
                {
                    Assert.Fail("the socket closed while seat " + Seat.ToString()
                        + " was waiting for a frame starting " + prefix);
                    return string.Empty;
                }

                if (frame == "PING")
                {
                    await SendAsync("PONG");
                    continue;
                }

                Track(frame);
                Assert.That(frame, Does.StartWith(prefix));
                return frame;
            }
        }

        /// <summary>Proves a frame did NOT arrive. Bounded rather than instant: a broadcast that reached the
        /// wrong seat would otherwise be missed simply because it was slower than the assertion.</summary>
        public async Task ExpectNothingAsync(TimeSpan within)
        {
            DateTimeOffset deadline = DateTimeOffset.UtcNow.Add(within);

            while (DateTimeOffset.UtcNow < deadline)
            {
                string? frame = await ReceiveAsync(deadline - DateTimeOffset.UtcNow);
                if (frame is null) return;
                if (frame == "PING")
                {
                    await SendAsync("PONG");
                    continue;
                }

                Assert.Fail("seat " + Seat.ToString() + " was told " + frame
                    + " and should have been told nothing");
            }
        }

        /// <summary>The close status the server ended this socket with.</summary>
        public async Task<WebSocketCloseStatus?> ExpectCloseAsync(TimeSpan? within = null)
        {
            DateTimeOffset deadline = DateTimeOffset.UtcNow.Add(within ?? DefaultWait);

            while (true)
            {
                string? frame = await ReceiveAsync(deadline - DateTimeOffset.UtcNow);
                if (frame is null) return _socket!.CloseStatus;
                if (frame == "PING") continue;

                Assert.Fail("expected a close and got " + frame);
            }
        }

        /// <summary>A dead socket: no close handshake, exactly like a client whose process went away.</summary>
        public void Drop()
        {
            _socket?.Abort();
            _socket?.Dispose();
            _socket = null;
        }

        // ---- the local projection --------------------------------------------

        void Track(string frame)
        {
            NetMessage message = NetProtocol.Parse(frame);

            if (message.Type == "START")
            {
                ReplayData dealt = ReplayFile.Read(message.Payload);
                Start = dealt.Start;
                State = dealt.Start;
                Log.Clear();

                foreach (Command command in dealt.Commands) Advance(command);
                return;
            }

            if (message.Type != "APPLY") return;

            Advance(CommandWire.Read(message.Payload));
        }

        void Advance(Command command)
        {
            Result applied = GameEngine.Apply(State!, command);
            Assert.That(applied.Success, Is.True,
                "the server dealt seat " + Seat.ToString() + " a command the engine refuses: "
                + CommandWire.Write(command) + " (" + applied.Reason + ")");

            State = applied.NewState;
            Log.Add(command);
        }

        async Task<string?> ReceiveAsync(TimeSpan within)
        {
            if (within <= TimeSpan.Zero) within = TimeSpan.FromMilliseconds(1);

            using var deadline = new CancellationTokenSource(within);
            var buffer = new byte[65536];
            using var message = new MemoryStream();
            WebSocketReceiveResult result;

            try
            {
                do
                {
                    result = await _socket!.ReceiveAsync(new ArraySegment<byte>(buffer), deadline.Token);
                    if (result.MessageType == WebSocketMessageType.Close) return null;

                    message.Write(buffer, 0, result.Count);
                }
                while (!result.EndOfMessage);
            }
            catch (OperationCanceledException)
            {
                return null;
            }
            catch (WebSocketException)
            {
                // The host went away mid-read, which is what a restart looks like from this side.
                return null;
            }

            return Encoding.UTF8.GetString(message.ToArray());
        }

        public async ValueTask DisposeAsync()
        {
            if (_socket is not null)
            {
                try
                {
                    if (_socket.State == WebSocketState.Open)
                        await _socket.CloseAsync(
                            WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None);
                }
                catch (WebSocketException)
                {
                }
                catch (OperationCanceledException)
                {
                }

                _socket.Dispose();
                _socket = null;
            }

            _http.Dispose();
        }
    }
}

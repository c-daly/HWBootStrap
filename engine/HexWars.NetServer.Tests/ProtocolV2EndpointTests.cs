using System.Net;
using System.Net.WebSockets;
using System.Text;
using HexWars.Engine;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Endpoints;
using HexWars.NetServer.Hosting;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The v2 socket through the real host: a socket really is upgraded, the handshake really refuses
    /// everything that is not an AUTH frame it can validate, and the heartbeat really closes a client that
    /// has stopped answering.
    ///
    /// The clock here is the system one, unlike every other test over this fixture. Liveness is the one
    /// thing in this server that is genuinely about elapsed time rather than about an instant: a frozen
    /// clock would make every socket look permanently fresh, and a test that wound it forward by hand would
    /// be asserting on the winding rather than on the timer. So the intervals are shortened instead - a
    /// one-second ping and a three-second silence window - and the assertions are about real seconds.
    /// </summary>
    [TestFixture]
    public class ProtocolV2EndpointTests
    {
        const string Seat0Steam = "76561190000000001";
        const string Seat1Steam = "76561190000000002";
        const string LobbyId = "109775240000000042";

        static readonly TimeSpan FrameWait = TimeSpan.FromSeconds(10);

        SteamServerFactory _fixture = null!;
        WebApplicationFactory<Program> _host = null!;
        Guid _matchId;
        string _credential0 = null!;
        string _credential1 = null!;

        [TearDown]
        public void DisposeTheHost()
        {
            _host?.Dispose();
            _fixture?.Dispose();
        }

        // ---- fixture ---------------------------------------------------------

        void Start(Action<SteamServerFactory>? configure = null)
        {
            _fixture = new SteamServerFactory();
            _fixture.Settings["MATCH_HEARTBEAT_SECONDS"] = "1";
            _fixture.Settings["MATCH_STALE_CONNECTION_SECONDS"] = "3";
            _fixture.Settings["MATCH_AUTH_TIMEOUT_SECONDS"] = "1";
            _fixture.Settings["MATCH_OUTBOUND_QUEUE_CAPACITY"] = "16";
            configure?.Invoke(_fixture);

            _host = _fixture.WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            {
                services.RemoveAll<TimeProvider>();
                services.AddSingleton(TimeProvider.System);
            }));
        }

        /// <summary>A match with both seats sold, arranged through the store rather than through the HTTP
        /// endpoints: this file is about the socket, and going the long way round would make every test
        /// here also a test of match allocation.</summary>
        async Task SeedAWaitingMatch()
        {
            CreateMatchResult created = await _fixture.Store.CreateMatchForLobbyAsync(
                new CreateMatchRequest(
                    LobbyId,
                    GameSetup.Default.ToWire(),
                    "hexwars-engine/1",
                    2,
                    SteamServerFactory.BuildId,
                    new[] { (Seat0Steam, 0), (Seat1Steam, 1) },
                    DateTimeOffset.UtcNow),
                CancellationToken.None);

            _matchId = created.Match.MatchId;

            var credentials = _host.Services.GetRequiredService<IMatchCredentialService>();
            _credential0 = (await credentials.IssueAsync(_matchId, Seat0Steam, CancellationToken.None)).Credential;
            _credential1 = (await credentials.IssueAsync(_matchId, Seat1Steam, CancellationToken.None)).Credential;
        }

        Task<WebSocket> ConnectAsync(string? origin = null)
        {
            WebSocketClient client = _host.Server.CreateWebSocketClient();
            if (origin is not null)
                client.ConfigureRequest = request => request.Headers["Origin"] = origin;

            return client.ConnectAsync(
                new Uri("ws://localhost" + SteamMatchEndpoints.WebSocketPath), CancellationToken.None);
        }

        string Auth(string credential) => "AUTH " + _matchId + " " + credential;

        static Task SendAsync(WebSocket socket, string text) => socket.SendAsync(
            Encoding.UTF8.GetBytes(text), WebSocketMessageType.Text, true, CancellationToken.None);

        readonly record struct Received(string? Text, WebSocketCloseStatus? CloseStatus);

        static async Task<Received> ReceiveAsync(WebSocket socket, TimeSpan? within = null)
        {
            using var deadline = new CancellationTokenSource(within ?? FrameWait);
            var buffer = new byte[65536];
            using var message = new MemoryStream();
            WebSocketReceiveResult result;

            do
            {
                result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), deadline.Token);
                if (result.MessageType == WebSocketMessageType.Close)
                    return new Received(null, result.CloseStatus);

                message.Write(buffer, 0, result.Count);
            }
            while (!result.EndOfMessage);

            return new Received(Encoding.UTF8.GetString(message.ToArray()), null);
        }

        /// <summary>Authenticates one seat and consumes the two frames a waiting match answers with.</summary>
        async Task<WebSocket> SeatAsync(string credential, int seat)
        {
            WebSocket socket = await ConnectAsync();
            await SendAsync(socket, Auth(credential));

            Assert.That((await ReceiveAsync(socket)).Text, Is.EqualTo("SEAT " + seat));
            Assert.That((await ReceiveAsync(socket)).Text, Is.EqualTo(NetProtocol.CatalogRequest));
            return socket;
        }

        // ---- the handshake ---------------------------------------------------

        [Test]
        public async Task AFirstFrameThatIsNotAuth_IsRefusedAndClosed()
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket socket = await ConnectAsync();
            await SendAsync(socket, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));

            Assert.That((await ReceiveAsync(socket)).Text, Is.EqualTo("AUTH FAIL invalid"));
            Assert.That((await ReceiveAsync(socket)).CloseStatus,
                Is.EqualTo(WebSocketCloseStatus.PolicyViolation));
        }

        [Test]
        public async Task ACredentialThisMatchNeverIssued_IsRefusedAndClosed()
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket socket = await ConnectAsync();
            await SendAsync(socket, "AUTH " + _matchId + " " + new string('a', 43));

            Assert.That((await ReceiveAsync(socket)).Text, Is.EqualTo("AUTH FAIL invalid"));
            Assert.That((await ReceiveAsync(socket)).CloseStatus,
                Is.EqualTo(WebSocketCloseStatus.PolicyViolation));
        }

        [TestCase("AUTH")]
        [TestCase("AUTH one-token")]
        [TestCase("AUTH one two three")]
        public async Task AnAuthFrameThatIsNotAMatchIdAndACredential_IsRefused(string frame)
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket socket = await ConnectAsync();
            await SendAsync(socket, frame);

            Assert.That((await ReceiveAsync(socket)).Text, Is.EqualTo("AUTH FAIL invalid"));
            Assert.That((await ReceiveAsync(socket)).CloseStatus,
                Is.EqualTo(WebSocketCloseStatus.PolicyViolation));
        }

        [Test]
        public async Task ASocketThatNeverAuthenticates_IsClosedWhenTheDeadlinePasses()
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket socket = await ConnectAsync();

            // Nothing is sent. The deadline is one second here; a socket holding a connection slot without
            // proving anything is exactly what it exists to end.
            Received closed = await ReceiveAsync(socket, TimeSpan.FromSeconds(6));

            Assert.That(closed.CloseStatus, Is.EqualTo(WebSocketCloseStatus.PolicyViolation));
        }

        [Test]
        public async Task AValidCredential_IsSeatedAndAskedForACatalog()
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket socket = await ConnectAsync();
            await SendAsync(socket, Auth(_credential0));

            Assert.That((await ReceiveAsync(socket)).Text, Is.EqualTo("SEAT 0"));
            Assert.That((await ReceiveAsync(socket)).Text, Is.EqualTo(NetProtocol.CatalogRequest));
        }

        // ---- playing ---------------------------------------------------------

        [Test]
        public async Task BothCatalogs_StartTheMatchForBothSeats()
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket zero = await SeatAsync(_credential0, 0);
            using WebSocket one = await SeatAsync(_credential1, 1);

            string catalog = NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates));
            await SendAsync(zero, catalog);
            await SendAsync(one, catalog);

            Assert.That((await ReceiveAsync(zero)).Text, Does.StartWith("START "));
            Assert.That((await ReceiveAsync(one)).Text, Does.StartWith("START "));
        }

        [Test]
        public async Task AnAcceptedCommand_ReachesBothSeats()
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket zero = await SeatAsync(_credential0, 0);
            using WebSocket one = await SeatAsync(_credential1, 1);
            await StartTheMatch(zero, one);

            await SendAsync(zero, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));

            Assert.That((await ReceiveAsync(zero)).Text, Is.EqualTo("APPLY E 0"));
            Assert.That((await ReceiveAsync(one)).Text, Is.EqualTo("APPLY E 0"));
        }

        [Test]
        public async Task AFrameOverTheSizeCap_ClosesTheSocketAsTooBig()
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket socket = await SeatAsync(_credential0, 0);

            // Started before the send: the server refuses the message part-way through, so the close can
            // arrive while this client is still writing.
            Task<Received> answer = ReceiveAsync(socket);
            try
            {
                await SendAsync(socket, "CATALOG " + new string('x', ProtocolV2WebSocketServer.MaxIncomingBytes + 1));
            }
            catch (WebSocketException)
            {
            }

            Assert.That((await answer).CloseStatus, Is.EqualTo(WebSocketCloseStatus.MessageTooBig));
        }

        // ---- liveness --------------------------------------------------------

        [Test]
        public async Task ASocketThatAnswersPing_StaysOpenWhileASilentOneIsClosedAsStale()
        {
            Start();
            await SeedAWaitingMatch();

            using WebSocket answering = await SeatAsync(_credential0, 0);
            using WebSocket silent = await SeatAsync(_credential1, 1);

            int pings = 0;
            WebSocketCloseStatus? silentClose = null;
            DateTimeOffset until = DateTimeOffset.UtcNow.AddSeconds(5);

            Task answerer = Task.Run(async () =>
            {
                while (DateTimeOffset.UtcNow < until && answering.State == WebSocketState.Open)
                {
                    Received frame = await ReceiveAsync(answering);
                    if (frame.CloseStatus is not null) return;
                    if (frame.Text != "PING") continue;

                    Interlocked.Increment(ref pings);
                    await SendAsync(answering, "PONG");
                }
            });

            // Bounded rather than "until a close arrives": a server that never closed this socket would
            // otherwise leave the test reading pings forever instead of failing.
            Task watcher = Task.Run(async () =>
            {
                DateTimeOffset giveUp = DateTimeOffset.UtcNow.AddSeconds(10);
                while (DateTimeOffset.UtcNow < giveUp)
                {
                    Received frame = await ReceiveAsync(silent);
                    if (frame.CloseStatus is null) continue;

                    silentClose = frame.CloseStatus;
                    return;
                }
            });

            await watcher;

            // 1001, EndpointUnavailable on the wire: the server is not refusing this client, it has simply
            // concluded there is nobody on the other end any more.
            Assert.That(silentClose, Is.EqualTo(WebSocketCloseStatus.EndpointUnavailable));

            await answerer;
            Assert.That(pings, Is.GreaterThanOrEqualTo(2), "one ping per second was configured");
            Assert.That(answering.State, Is.EqualTo(WebSocketState.Open),
                "a client that answered every ping is not stale, however long it has been idle");
        }

        // ---- routing and origin ----------------------------------------------

        [Test]
        public async Task WithoutTheSteamProvider_TheV2RouteIsNotMapped()
        {
            Start(fixture => fixture.Settings["LOBBY_PROVIDER"] = "Legacy");

            using HttpClient client = _host.CreateClient();
            HttpResponseMessage response = await client.GetAsync(SteamMatchEndpoints.WebSocketPath);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
        }

        [Test]
        public async Task WithoutTheLegacyProvider_TheV1RouteIsNotMapped()
        {
            Start();

            using HttpClient client = _host.CreateClient();
            HttpResponseMessage response = await client.GetAsync("/ws");

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
        }

        [Test]
        public async Task AnOriginFromAnotherSite_IsRefusedBeforeTheUpgrade()
        {
            Start();
            await SeedAWaitingMatch();

            Exception? refused = null;
            try
            {
                using WebSocket socket = await ConnectAsync("https://evil.invalid");
            }
            catch (Exception failure)
            {
                refused = failure;
            }

            Assert.That(refused, Is.Not.Null, "a cross-site page must not be upgraded at all");
            Assert.That(refused!.Message, Does.Contain("403"));
        }

        [Test]
        public async Task AnOriginTheOperatorAllowed_IsUpgraded()
        {
            Start(fixture => fixture.Settings["ALLOWED_WEB_ORIGINS"] = "https://play.hexwars.invalid");
            await SeedAWaitingMatch();

            using WebSocket socket = await ConnectAsync("https://play.hexwars.invalid");
            await SendAsync(socket, Auth(_credential0));

            Assert.That((await ReceiveAsync(socket)).Text, Is.EqualTo("SEAT 0"));
        }

        // ---- back pressure ---------------------------------------------------

        [Test]
        public async Task AClientThatStopsReading_DoesNotStallTheOtherSeat()
        {
            // The end-to-end half of the slow-client rule, and the only half this host can show. Filling a
            // bounded outbound queue in process is not reachable here: every frame a match broadcasts is a
            // few dozen bytes, and the test server's own duplex buffer absorbs thousands of them before a
            // send would block, so the writer keeps draining the queue no matter how little the client
            // reads. What IS reachable, and is the thing the bound exists to protect, is that a seat which
            // stops reading does not hold up the seat that has not. The bound itself is exercised directly
            // against a wedged writer in TheOutboundQueueRefusesRatherThanGrows below.
            Start();
            await SeedAWaitingMatch();

            using WebSocket zero = await SeatAsync(_credential0, 0);
            using WebSocket one = await SeatAsync(_credential1, 1);
            await StartTheMatch(zero, one);

            // Seat one never reads again from here.
            for (int turn = 0; turn < 12; turn++)
            {
                bool zeroToPlay = turn % 2 == 0;
                WebSocket sender = zeroToPlay ? zero : one;
                var command = new EndTurn(zeroToPlay ? PlayerId.Player0 : PlayerId.Player1);

                await SendAsync(sender, NetProtocol.Cmd(command));
                Assert.That((await ReceiveAsync(zero)).Text, Is.EqualTo("APPLY " + CommandWire.Write(command)));
            }

            Assert.That(zero.State, Is.EqualTo(WebSocketState.Open));
        }

        [Test]
        public async Task TheOutboundQueueRefusesRatherThanGrows()
        {
            // A writer that never returns is exactly what a client which has stopped reading looks like
            // from this side. The queue must fill and then refuse, and the refusal must close the socket:
            // dropping frames silently would leave the client replaying a game with a hole in it.
            var wedged = new WedgedWebSocket();
            var connection = new V2Connection("wedged", wedged, "203.0.113.9", 16, DateTimeOffset.UtcNow);

            int accepted = 0;
            for (int frame = 0; frame < 500; frame++)
            {
                if (!connection.TryEnqueue("APPLY E 0")) break;
                accepted++;
            }

            Assert.That(accepted, Is.InRange(16, 17),
                "a capacity of 16, plus at most the one frame the writer had already taken");
            Assert.That(connection.MaxQueueDepth, Is.LessThanOrEqualTo(16));
            Assert.That(connection.IsClosed, Is.True, "a full queue is a decision, not a pause");

            await connection.Closed;
            Assert.That(wedged.AbortCount, Is.GreaterThan(0),
                "there is no close handshake to be had with a peer that is not reading");

            wedged.Release();
        }

        async Task StartTheMatch(WebSocket zero, WebSocket one)
        {
            string catalog = NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates));
            await SendAsync(zero, catalog);
            await SendAsync(one, catalog);

            Assert.That((await ReceiveAsync(zero)).Text, Does.StartWith("START "));
            Assert.That((await ReceiveAsync(one)).Text, Does.StartWith("START "));
        }

        /// <summary>A socket whose sends never complete: the shape of a client that has stopped reading,
        /// without needing a real peer that behaves that way.</summary>
        sealed class WedgedWebSocket : WebSocket
        {
            readonly SemaphoreSlim _wedge = new(0);
            int _aborts;

            public int AbortCount => Volatile.Read(ref _aborts);

            public void Release() => _wedge.Release(int.MaxValue / 2);

            public override WebSocketCloseStatus? CloseStatus => null;
            public override string? CloseStatusDescription => null;
            public override WebSocketState State => WebSocketState.Open;
            public override string? SubProtocol => null;

            public override void Abort() => Interlocked.Increment(ref _aborts);

            public override Task CloseAsync(
                WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken cancellationToken) =>
                Task.CompletedTask;

            public override Task CloseOutputAsync(
                WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken cancellationToken) =>
                Task.CompletedTask;

            public override void Dispose() => _wedge.Dispose();

            public override Task<WebSocketReceiveResult> ReceiveAsync(
                ArraySegment<byte> buffer, CancellationToken cancellationToken) =>
                new TaskCompletionSource<WebSocketReceiveResult>().Task;

            public override Task SendAsync(
                ArraySegment<byte> buffer,
                WebSocketMessageType messageType,
                bool endOfMessage,
                CancellationToken cancellationToken) =>
                _wedge.WaitAsync(cancellationToken);
        }
    }
}

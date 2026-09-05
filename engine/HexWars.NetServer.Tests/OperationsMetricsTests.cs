using System.Net;
using System.Text.Json;
using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The numbers an operator gets to see, and who is allowed to see them.
    ///
    /// The counters are asserted through a real game rather than by calling the meter directly: a counter
    /// that is never reached from the code path it claims to measure is worse than no counter, because it
    /// reads as a healthy zero. So one match is allocated, two sockets are seated, one command is committed
    /// and one is refused, and the snapshot has to agree with what happened.
    /// </summary>
    [TestFixture]
    public class OperationsMetricsTests
    {
        const string Token = "metrics-token-for-tests";

        /// <summary>The opening move of the deterministic seed-7 script the rest of this repository uses.
        /// Its legality is a fact about that seed, which is why the lobby below advertises it.</summary>
        static readonly Command Opening = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));

        readonly List<IDisposable> _disposables = new();

        [TearDown]
        public void DisposeWhatTheTestBuilt()
        {
            for (int i = _disposables.Count - 1; i >= 0; i--)
            {
                try { _disposables[i].Dispose(); }
                catch { /* a host torn down mid-shutdown is not a test failure */ }
            }

            _disposables.Clear();
        }

        T Track<T>(T disposable) where T : IDisposable
        {
            _disposables.Add(disposable);
            return disposable;
        }

        /// <summary>A Steam host whose lobby advertises the seed the scripted opening is legal under.</summary>
        SteamServerFactory Host(string? token = Token)
        {
            SteamServerFactory factory = Track(new SteamServerFactory());
            if (token is not null) factory.Settings[HexWarsConfiguration.MatchMetricsTokenKey] = token;

            factory.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] = FakeSteamWebApiClient.ReadyLobby(
                ruleset: SteamLobbyRules.CustomRuleset, setupWire: GameSetup.Default.ToWire());

            return factory;
        }

        static async Task<JsonElement> SnapshotAsync(HttpClient client)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, HealthEndpoints.MetricsRoute);
            request.Headers.Add(HealthEndpoints.MetricsTokenHeader, Token);

            using HttpResponseMessage response = await client.SendAsync(request);
            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));

            return JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.Clone();
        }

        /// <summary>Both seats seated on a started match, with no commands played.</summary>
        static async Task<(DurableFlowClient Zero, DurableFlowClient One)> StartAsync(SteamServerFactory host)
        {
            DurableFlowClient zero = await DurableFlowClient.CreateAsync(
                host, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket);
            DurableFlowClient one = await DurableFlowClient.JoinAsync(
                host, zero.MatchId, FakeSteamWebApiClient.GuestTicket);

            await zero.ConnectAsync();
            await zero.ExpectAsync(NetProtocol.CatalogRequest);
            await one.ConnectAsync();
            await one.ExpectAsync(NetProtocol.CatalogRequest);

            await zero.SendCatalogAsync();
            await one.SendCatalogAsync();

            await zero.ExpectAsync("START ");
            await one.ExpectAsync("START ");

            return (zero, one);
        }

        // ---- who may read them ------------------------------------------------

        [Test]
        public async Task Metrics_Is404WhenNoTokenIsConfigured()
        {
            SteamServerFactory factory = Host(token: null);
            using HttpClient client = factory.CreateClient();

            using HttpResponseMessage response = await client.GetAsync(HealthEndpoints.MetricsRoute);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.NotFound),
                "a deployment that never turned this on has no such route, and says so");
        }

        [Test]
        public async Task Metrics_Is401WithoutTheTokenAndWithTheWrongOne()
        {
            SteamServerFactory factory = Host();
            using HttpClient client = factory.CreateClient();

            using HttpResponseMessage missing = await client.GetAsync(HealthEndpoints.MetricsRoute);
            Assert.That(missing.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));

            using var wrong = new HttpRequestMessage(HttpMethod.Get, HealthEndpoints.MetricsRoute);
            wrong.Headers.Add(HealthEndpoints.MetricsTokenHeader, Token + "x");

            using HttpResponseMessage refused = await client.SendAsync(wrong);
            Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));
        }

        // ---- what they say ----------------------------------------------------

        [Test]
        public async Task Metrics_CountTheMatchTheCommandAndTheOpenSockets()
        {
            SteamServerFactory factory = Host();
            using HttpClient client = factory.CreateClient();

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(factory);
            await using (zero)
            await using (one)
            {
                await zero.SendCmdAsync(Opening);
                await zero.ExpectAsync("APPLY ");
                await one.ExpectAsync("APPLY ");

                JsonElement snapshot = await SnapshotAsync(client);

                Assert.That(snapshot.GetProperty("matchesCreated").GetInt64(), Is.EqualTo(1));
                Assert.That(snapshot.GetProperty("commandsCommitted").GetInt64(), Is.EqualTo(1));
                Assert.That(snapshot.GetProperty("openSockets").GetInt32(), Is.EqualTo(2));
                Assert.That(snapshot.GetProperty("liveMatches").GetInt32(), Is.EqualTo(1));
                Assert.That(snapshot.GetProperty("commitCount").GetInt64(), Is.EqualTo(1),
                    "the commit histogram saw the append it timed");
                Assert.That(snapshot.GetProperty("broadcastCount").GetInt64(), Is.EqualTo(1));
            }
        }

        [Test]
        public async Task Metrics_CountARefusedCommandUnderTheReasonItWasRefusedFor()
        {
            SteamServerFactory factory = Host();
            using HttpClient client = factory.CreateClient();

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(factory);
            await using (zero)
            await using (one)
            {
                // Seat 0 issuing for seat 1. The seat comes from the credential, so this is refused for a
                // reason an operator can act on, and it is the reason that has to reach the counter.
                await zero.SendCmdAsync(new EndTurn(PlayerId.Player1));
                Assert.That(await zero.ExpectAsync("REJECT "), Is.EqualTo("REJECT WrongSeat"));

                JsonElement snapshot = await SnapshotAsync(client);

                Assert.That(snapshot.GetProperty("commandsRejected").GetInt64(), Is.EqualTo(1));
                Assert.That(
                    snapshot.GetProperty("rejectionsByReason").GetProperty("WrongSeat").GetInt64(),
                    Is.EqualTo(1));
                Assert.That(snapshot.GetProperty("commandsCommitted").GetInt64(), Is.Zero);
            }
        }

        [Test]
        public async Task TheMetricsLogService_WritesTheSnapshotOnItsPeriod()
        {
            var logging = new CapturingLoggerProvider();

            SteamServerFactory factory = Host();
            factory.Logging = logging;
            factory.Clock.VirtualTimers = true;

            using HttpClient client = factory.CreateClient();
            Assert.That(logging.Any("Metrics {"), Is.False, "nothing has been logged before a period passed");

            for (var round = 0; round < 100 && !logging.Any("Metrics {"); round++)
            {
                factory.Clock.Advance(MetricsLogService.Period);
                await Task.Delay(25);
            }

            Assert.That(logging.Any("Metrics {"), Is.True,
                "one structured line per period is what an operator without a scraper gets");
        }
    }
}

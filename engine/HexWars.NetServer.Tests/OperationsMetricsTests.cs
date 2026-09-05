using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
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
        static async Task<(DurableFlowClient Zero, DurableFlowClient One)> StartAsync(
            WebApplicationFactory<Program> host)
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
        public async Task Metrics_CountACatalogRefusalApartFromACommandRefusal()
        {
            SteamServerFactory factory = Host();
            using HttpClient client = factory.CreateClient();

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(factory);
            await using (zero)
            await using (one)
            {
                // The match has started, so a second catalog is refused. It is a different event from a
                // move that was not applied, and an operator reading one counter must not be reading both.
                await zero.SendCatalogAsync();
                Assert.That(await zero.ExpectAsync("REJECT "), Is.EqualTo("REJECT CatalogClosed"));

                JsonElement snapshot = await SnapshotAsync(client);

                Assert.That(snapshot.GetProperty("catalogRejected").GetInt64(), Is.EqualTo(1));
                Assert.That(
                    snapshot.GetProperty("catalogRejectionsByReason").GetProperty("CatalogClosed").GetInt64(),
                    Is.EqualTo(1));
                Assert.That(snapshot.GetProperty("commandsRejected").GetInt64(), Is.Zero,
                    "a catalog that was refused is not a command that was refused");
            }
        }

        [Test]
        public async Task Metrics_CountAStoreFailureUnderTheCallThatFailed()
        {
            SteamServerFactory factory = Host();
            var faults = new FaultInjectingMatchStore(factory.Store);

            WebApplicationFactory<Program> host = Track(factory.WithWebHostBuilder(
                builder => builder.ConfigureServices(services =>
                {
                    services.RemoveAll<IMatchStore>();
                    services.AddSingleton<IMatchStore>(faults);
                })));

            using HttpClient client = host.CreateClient();

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(host);
            await using (zero)
            await using (one)
            {
                faults.FailNextAppend(new InvalidOperationException("the database is not there"));

                await zero.SendCmdAsync(Opening);
                Assert.That(await zero.ExpectAsync("REJECT "), Is.EqualTo("REJECT TemporaryFailure"));

                JsonElement snapshot = await SnapshotAsync(client);

                Assert.That(snapshot.GetProperty("databaseFailures").GetInt64(), Is.GreaterThanOrEqualTo(1));
                Assert.That(
                    snapshot.GetProperty("databaseFailuresByOp").GetProperty("append").GetInt64(),
                    Is.EqualTo(1),
                    "an untagged total says the database is unhappy and nothing an operator can act on");
            }
        }

        [Test]
        public async Task Metrics_CountAnOwnershipRefusalAsASteamFailure()
        {
            SteamServerFactory factory = Host();
            factory.Steam.Ownership.Remove(FakeSteamWebApiClient.OwnerSteamId);

            using HttpClient client = factory.CreateClient();

            using HttpResponseMessage refused = await client.PostAsJsonAsync(
                DurableFlowClient.CreateRoute,
                new
                {
                    steamLobbyId = FakeSteamWebApiClient.LobbyId,
                    ticket = FakeSteamWebApiClient.OwnerTicket,
                });

            Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));

            JsonElement snapshot = await SnapshotAsync(client);

            Assert.That(
                snapshot.GetProperty("steamFailuresByKind").GetProperty("OwnershipMissing").GetInt64(),
                Is.EqualTo(1),
                "a no Valve answered with is still Valve turning a player away");
            Assert.That(snapshot.GetProperty("matchesCreated").GetInt64(), Is.Zero);
        }

        [Test]
        public async Task Metrics_CountAMalformedHandshakeUnderTheFrameStage()
        {
            SteamServerFactory factory = Host();
            using HttpClient client = factory.CreateClient();

            DurableFlowClient seat = await DurableFlowClient.CreateAsync(
                factory, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket);

            await using (seat)
            {
                await seat.OpenAsync();
                await seat.SendAsync("HELLO there");

                Assert.That(await seat.ExpectAsync("AUTH FAIL "), Is.EqualTo("AUTH FAIL invalid"));

                JsonElement snapshot = await SnapshotAsync(client);

                Assert.That(
                    snapshot.GetProperty("authFailuresByStage").GetProperty("frame").GetInt64(),
                    Is.EqualTo(1),
                    "a refusal that never reached the credential store is not a credential failure");
                Assert.That(snapshot.GetProperty("authFailures").GetInt64(), Is.EqualTo(1));
            }
        }

        [Test]
        public async Task TheMetricsLogService_WritesTheSnapshotOnItsPeriod()
        {
            var logging = new CapturingLoggerProvider();

            SteamServerFactory factory = Host();
            factory.Logging = logging;

            using HttpClient client = factory.CreateClient();
            Assert.That(logging.Any("Metrics {"), Is.False, "nothing has been logged before a period passed");

            // The clock only fires what is already armed, so winding it forward before the log service has
            // scheduled its first tick would move it past nothing at all.
            for (var round = 0; round < 200 && factory.Clock.ScheduledTimers == 0; round++)
                await Task.Delay(25);

            Assert.That(factory.Clock.ScheduledTimers, Is.GreaterThan(0), "the period was never armed");

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

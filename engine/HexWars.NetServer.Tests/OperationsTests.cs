using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HexWars.Engine;
using HexWars.NetServer.Contracts;
using HexWars.NetServer.Endpoints;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// What the platform asks this process, and what it does when it is told to go away.
    ///
    /// Liveness and readiness are deliberately different questions and the tests keep them apart: a host
    /// whose database is unreachable is alive and must say so, or the platform kills it and the restart
    /// meets the same database. Readiness is where every reason to hold traffic back is collected, and each
    /// test here names the ONE check it is about rather than the status code alone - a 503 that came from
    /// the wrong check would otherwise pass for the right answer.
    /// </summary>
    [TestFixture]
    public class OperationsTests
    {
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

        // ---- reading the readiness answer -------------------------------------

        sealed record ReadyCheck(string Name, string Status, string? Description);

        static async Task<(HttpStatusCode Code, string Status, IReadOnlyList<ReadyCheck> Checks)> ReadyAsync(
            HttpClient client)
        {
            using HttpResponseMessage response = await client.GetAsync(HealthEndpoints.ReadyRoute);
            JsonElement body = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;

            ReadyCheck[] checks = body.GetProperty("checks").EnumerateArray()
                .Select(entry => new ReadyCheck(
                    entry.GetProperty("name").GetString()!,
                    entry.GetProperty("status").GetString()!,
                    entry.GetProperty("description").ValueKind == JsonValueKind.Null
                        ? null
                        : entry.GetProperty("description").GetString()))
                .ToArray();

            return (response.StatusCode, body.GetProperty("status").GetString()!, checks);
        }

        static ReadyCheck Check(IReadOnlyList<ReadyCheck> checks, string name)
        {
            ReadyCheck? found = checks.SingleOrDefault(check => check.Name == name);
            Assert.That(found, Is.Not.Null,
                "no check named " + name + "; readiness listed "
                + string.Join(", ", checks.Select(check => check.Name)));

            return found!;
        }

        static WebApplicationFactory<Program> LegacyHost() =>
            new WebApplicationFactory<Program>().WithWebHostBuilder(
                builder => builder.UseEnvironment("Development"));

        // ---- liveness ---------------------------------------------------------

        [Test]
        public async Task Live_Answers200OnALegacyHostThatHasNoDatabaseAtAll()
        {
            WebApplicationFactory<Program> factory = Track(
                new WebApplicationFactory<Program>().WithWebHostBuilder(builder => builder
                    .UseEnvironment("Development")
                    .UseSetting("MATCH_BUILD_ID", "legacy-build")));

            using HttpClient client = factory.CreateClient();
            using HttpResponseMessage live = await client.GetAsync(HealthEndpoints.LiveRoute);

            Assert.That(live.StatusCode, Is.EqualTo(HttpStatusCode.OK));

            JsonElement body = JsonDocument.Parse(await live.Content.ReadAsStringAsync()).RootElement;
            Assert.That(body.GetProperty("status").GetString(), Is.EqualTo("live"));
            Assert.That(body.GetProperty("buildId").GetString(), Is.EqualTo("legacy-build"));
        }

        [Test]
        public async Task Live_Answers200OnASteamHostWhoseDatabaseIsUnreachable()
        {
            // The whole point of splitting the two probes. This host cannot serve a match and must not be
            // restarted for it: the restart would come up against the same database.
            SteamServerFactory factory = Track(new SteamServerFactory());
            using HttpClient client = factory.CreateClient();

            using HttpResponseMessage live = await client.GetAsync(HealthEndpoints.LiveRoute);
            Assert.That(live.StatusCode, Is.EqualTo(HttpStatusCode.OK));

            Assert.That((await ReadyAsync(client)).Code, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
        }

        [Test]
        public async Task Healthz_IsStillAnAliasOfLiveness()
        {
            SteamServerFactory factory = Track(new SteamServerFactory());
            using HttpClient client = factory.CreateClient();

            string live = await client.GetStringAsync(HealthEndpoints.LiveRoute);
            string legacy = await client.GetStringAsync(HealthEndpoints.LegacyLiveRoute);

            Assert.That(legacy, Is.EqualTo(live));
            Assert.That(
                JsonDocument.Parse(legacy).RootElement.GetProperty("buildId").GetString(),
                Is.EqualTo(SteamServerFactory.BuildId));
        }

        // ---- readiness on a host that cannot serve ----------------------------

        [Test]
        public async Task Ready_Is503WhileTheStartupRecoveryPassHasNotSucceeded()
        {
            SteamServerFactory factory = Track(new SteamServerFactory());
            factory.Counting.BeforeListOpenMatches =
                () => throw new InvalidOperationException("the database is not there");

            using HttpClient client = factory.CreateClient();

            var state = factory.Services.GetRequiredService<RecoveryState>();
            Assert.That(state.Error, Is.Not.Null, "a pass that could not run has verified nothing");
            Assert.That(state.Report, Is.Null);

            (HttpStatusCode code, string status, IReadOnlyList<ReadyCheck> checks) = await ReadyAsync(client);

            Assert.That(code, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(status, Is.EqualTo("Unhealthy"));
            Assert.That(Check(checks, HealthEndpoints.RecoveryCheck).Status, Is.EqualTo("Unhealthy"));
        }

        [Test]
        public async Task Ready_Is503WhenTheDatabaseWillNotAnswer()
        {
            SteamServerFactory factory = Track(new SteamServerFactory());
            using HttpClient client = factory.CreateClient();

            (HttpStatusCode code, _, IReadOnlyList<ReadyCheck> checks) = await ReadyAsync(client);

            Assert.That(code, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(Check(checks, HealthEndpoints.DatabaseCheck).Status, Is.EqualTo("Unhealthy"));
        }

        [Test]
        public async Task Ready_OnALegacyHost_IsRecoveryAndShutdownAndNothingElse()
        {
            // No database is configured, so there is no database or schema to have an opinion about. What
            // is left still has to be able to say the host is going away.
            WebApplicationFactory<Program> factory = Track(LegacyHost());
            using HttpClient client = factory.CreateClient();

            (HttpStatusCode code, string status, IReadOnlyList<ReadyCheck> checks) = await ReadyAsync(client);

            Assert.That(code, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(status, Is.EqualTo("Healthy"));
            Assert.That(
                checks.Select(check => check.Name),
                Is.EquivalentTo(new[] { HealthEndpoints.RecoveryCheck, HealthEndpoints.ShutdownCheck }));

            factory.Services.GetRequiredService<ServiceReadiness>().BeginShutdown();

            (HttpStatusCode after, _, IReadOnlyList<ReadyCheck> afterChecks) = await ReadyAsync(client);
            Assert.That(after, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(Check(afterChecks, HealthEndpoints.ShutdownCheck).Status, Is.EqualTo("Unhealthy"));
        }

        // ---- readiness over a real database -----------------------------------

        [Test]
        public async Task Ready_Is200WithEveryCheckNamedOnAHealthyHost()
        {
            SteamServerFactory factory = Track(await SteamServerFactory.PostgresAsync());
            using HttpClient client = factory.CreateClient();

            (HttpStatusCode code, string status, IReadOnlyList<ReadyCheck> checks) = await ReadyAsync(client);

            Assert.That(code, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(status, Is.EqualTo("Healthy"));
            Assert.That(checks.Select(check => check.Name), Is.EquivalentTo(new[]
            {
                HealthEndpoints.DatabaseCheck,
                HealthEndpoints.SchemaCheck,
                HealthEndpoints.RecoveryCheck,
                HealthEndpoints.ShutdownCheck,
            }));
            Assert.That(checks.Select(check => check.Status), Is.All.EqualTo("Healthy"));
        }

        [Test]
        public async Task Ready_Is503OnceShutdownHasBegun()
        {
            SteamServerFactory factory = Track(await SteamServerFactory.PostgresAsync());
            using HttpClient client = factory.CreateClient();

            Assert.That((await ReadyAsync(client)).Code, Is.EqualTo(HttpStatusCode.OK));

            factory.Services.GetRequiredService<ServiceReadiness>().BeginShutdown();

            (HttpStatusCode code, string status, IReadOnlyList<ReadyCheck> checks) = await ReadyAsync(client);

            Assert.That(code, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(status, Is.EqualTo("Unhealthy"));
            Assert.That(Check(checks, HealthEndpoints.ShutdownCheck).Status, Is.EqualTo("Unhealthy"));
            Assert.That(Check(checks, HealthEndpoints.DatabaseCheck).Status, Is.EqualTo("Healthy"),
                "the database did not go anywhere; only this process did");
        }

        [Test]
        public async Task Ready_NoticesTheDatabaseGoingAwayAfterStartup()
        {
            SteamServerFactory fixture = Track(await SteamServerFactory.PostgresAsync());

            NpgsqlDataSource? gone = null;
            WebApplicationFactory<Program> host = Track(fixture.WithWebHostBuilder(
                builder => builder.ConfigureServices(services =>
                {
                    services.RemoveAll<DatabaseHealthCheck>();
                    services.AddSingleton(provider => new DatabaseHealthCheck(
                        () => gone ?? provider.GetRequiredService<NpgsqlDataSource>()));
                })));

            using HttpClient client = host.CreateClient();
            Assert.That((await ReadyAsync(client)).Code, Is.EqualTo(HttpStatusCode.OK));

            // Nothing about the process changed; the database did. Readiness is asked again on every probe
            // rather than answered once at startup, and this is the difference that shows it.
            // A refused connection answers immediately and would prove nothing about the deadline. This
            // address is not routable, so the probe hangs exactly the way a database that has gone away
            // hangs, and the only thing that can end it in time is the deadline the check imposes.
            gone = Track(NpgsqlDataSource.Create(
                "Host=10.255.255.1;Port=5432;Username=u;Password=p;Database=fake;Timeout=2"));

            var clock = Stopwatch.StartNew();
            (HttpStatusCode code, _, IReadOnlyList<ReadyCheck> checks) = await ReadyAsync(client);
            clock.Stop();

            Assert.That(code, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(Check(checks, HealthEndpoints.DatabaseCheck).Status, Is.EqualTo("Unhealthy"));
            Assert.That(clock.Elapsed, Is.LessThan(TimeSpan.FromSeconds(4)),
                "a probe that waits as long as the connection attempt is a probe with no deadline");

            // And it let the pool go. A probe that abandoned a lease would make the next one queue behind
            // it, so the second answer arriving in time is the evidence the first one cleaned up.
            var second = Stopwatch.StartNew();
            Assert.That((await ReadyAsync(client)).Code, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            second.Stop();

            Assert.That(second.Elapsed, Is.LessThan(TimeSpan.FromSeconds(4)),
                "the first probe leased a connection it never gave back");
        }

        [Test]
        public async Task Ready_IsDegradedButStillServingWhenOneMatchCannotBeRecovered()
        {
            SteamServerFactory factory = Track(await SteamServerFactory.PostgresAsync());

            // Written straight to the database, under an engine contract this build does not replay, so the
            // startup pass meets a journal it must refuse rather than a double standing in for one.
            var direct = new PostgresMatchStore(
                factory.Database!.DataSource, NullLogger<PostgresMatchStore>.Instance);

            CreateMatchResult stranded = await direct.CreateMatchForLobbyAsync(
                new CreateMatchRequest(
                    FakeSteamWebApiClient.LobbyId,
                    GameSetup.Default.ToWire(),
                    "hexwars-engine/999",
                    2,
                    SteamServerFactory.BuildId,
                    new[]
                    {
                        (FakeSteamWebApiClient.OwnerSteamId, 0),
                        (FakeSteamWebApiClient.GuestSteamId, 1),
                    },
                    SteamServerFactory.Start),
                CancellationToken.None);

            using HttpClient client = factory.CreateClient();

            (HttpStatusCode code, string status, IReadOnlyList<ReadyCheck> checks) = await ReadyAsync(client);

            Assert.That(code, Is.EqualTo(HttpStatusCode.OK),
                "one match that needs a human is not a host that cannot take traffic");
            Assert.That(status, Is.EqualTo("Degraded"));

            ReadyCheck recovery = Check(checks, HealthEndpoints.RecoveryCheck);
            Assert.That(recovery.Status, Is.EqualTo("Degraded"));
            Assert.That(recovery.Description, Does.Contain(stranded.Match.MatchId.ToString()),
                "an operator cannot act on a count; the description names the match");
        }

        [Test]
        public async Task Ready_GoesFromUnreadyToReadyWhenTheRecoveryPassFinallyRuns()
        {
            SteamServerFactory fixture = Track(await SteamServerFactory.PostgresAsync());

            var attempts = 0;
            WebApplicationFactory<Program> host = Track(fixture.WithWebHostBuilder(
                builder => builder.ConfigureServices(services =>
                {
                    services.RemoveAll<IMatchStore>();
                    services.AddSingleton<IMatchStore>(provider => new CountingMatchStore(
                        new PostgresMatchStore(
                            provider.GetRequiredService<NpgsqlDataSource>(),
                            provider.GetRequiredService<ILogger<PostgresMatchStore>>()))
                    {
                        BeforeListOpenMatches = () => ++attempts <= 2
                            ? throw new InvalidOperationException("the database is not there")
                            : Task.CompletedTask,
                    });
                })));

            using HttpClient client = host.CreateClient();

            // The pass that ran during startup failed, so this host is holding traffic back rather than
            // serving matches it has not checked.
            Assert.That((await ReadyAsync(client)).Code, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(attempts, Is.EqualTo(1));

            await AdvanceUntilAsync(fixture, TimeSpan.FromSeconds(30), () => attempts >= 2);
            Assert.That((await ReadyAsync(client)).Code, Is.EqualTo(HttpStatusCode.ServiceUnavailable),
                "the second attempt failed too, so nothing has been verified yet");

            await AdvanceUntilAsync(fixture, TimeSpan.FromSeconds(60), () => attempts >= 3);

            for (var round = 0; round < 200; round++)
            {
                if ((await ReadyAsync(client)).Code == HttpStatusCode.OK) break;
                await Task.Delay(25);
            }

            Assert.That((await ReadyAsync(client)).Code, Is.EqualTo(HttpStatusCode.OK),
                "a host that recovered on its own starts serving without a restart");
            Assert.That(attempts, Is.EqualTo(3), "two refusals and then the pass that worked");
        }

        /// <summary>Winds the clock on once the backoff is actually armed. Advancing before the retry has
        /// scheduled its wait moves the clock past nothing at all.</summary>
        static async Task AdvanceUntilAsync(SteamServerFactory fixture, TimeSpan wait, Func<bool> until)
        {
            for (var round = 0; round < 200 && !until(); round++)
            {
                if (fixture.Clock.ScheduledTimers > 0) fixture.Clock.Advance(wait);
                await Task.Delay(25);
            }

            Assert.That(until(), Is.True, "the retry never came round");
        }

        // ---- going away -------------------------------------------------------

        [Test]
        public async Task Shutdown_TurnsAwayANewMatchAndANewJoinWith503()
        {
            SteamServerFactory factory = Track(new SteamServerFactory());
            using HttpClient client = factory.CreateClient();

            factory.Services.GetRequiredService<ServiceReadiness>().BeginShutdown();

            using HttpResponseMessage created = await client.PostAsJsonAsync(
                SteamMatchEndpoints.CreateRoute,
                new
                {
                    steamLobbyId = FakeSteamWebApiClient.LobbyId,
                    ticket = FakeSteamWebApiClient.OwnerTicket,
                });

            Assert.That(created.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));

            ApiError? body = await created.Content.ReadFromJsonAsync<ApiError>();
            Assert.That(body!.Error, Is.EqualTo(ApiErrors.ServiceUnavailable));

            using HttpResponseMessage joined = await client.PostAsJsonAsync(
                SteamMatchEndpoints.CreateRoute + "/" + Guid.NewGuid().ToString() + "/join",
                new { ticket = FakeSteamWebApiClient.GuestTicket });

            Assert.That(joined.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(factory.Steam.AuthenticateCalls, Is.Zero,
                "a host on its way out spends nothing at Valve on a request it will refuse");
        }

        [Test]
        public async Task Shutdown_RefusesANewV2SocketWith503()
        {
            SteamServerFactory factory = Track(new SteamServerFactory());
            using HttpClient warm = factory.CreateClient();

            factory.Services.GetRequiredService<ServiceReadiness>().BeginShutdown();

            WebSocketClient socket = factory.Server.CreateWebSocketClient();

            InvalidOperationException? refused = Assert.ThrowsAsync<InvalidOperationException>(
                async () => await socket.ConnectAsync(
                    new Uri("ws://localhost" + SteamMatchEndpoints.WebSocketPath), CancellationToken.None));

            Assert.That(refused!.Message, Does.Contain("503"));
        }
    }
}

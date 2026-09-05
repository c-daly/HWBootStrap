using System.Diagnostics;
using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Hosting;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Hosting;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// Shutdown under the conditions that make a budget worth having.
    ///
    /// The happy path proves almost nothing here: every step finishes instantly when the store answers and
    /// the clients read. What has to be proved is the opposite - that a store which has stopped answering,
    /// or a client which has stopped reading, cannot carry this process past the deadline at which the
    /// platform stops asking and starts killing. A host killed mid-shutdown never sends 1012, and every
    /// client it was serving sees a torn connection rather than a restart.
    /// </summary>
    [TestFixture]
    public class OperationsShutdownTests
    {
        /// <summary>The opening move of the deterministic seed-7 script the rest of this repository uses.</summary>
        static readonly Command Opening = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));

        /// <summary>Comfortably past the 20 s budget and short of the 25 s host timeout. A shutdown that
        /// takes longer than this has already lost the race it exists to win.</summary>
        static readonly TimeSpan Deadline = TimeSpan.FromSeconds(22);

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
        SteamServerFactory Fixture()
        {
            SteamServerFactory factory = Track(new SteamServerFactory());
            factory.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] = FakeSteamWebApiClient.ReadyLobby(
                ruleset: SteamLobbyRules.CustomRuleset, setupWire: GameSetup.Default.ToWire());

            return factory;
        }

        /// <summary>Both seats seated on a started match.</summary>
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

        [Test]
        public async Task Shutdown_FinishesInsideTheBudgetWhenTheStoreHasStoppedAnswering()
        {
            SteamServerFactory fixture = Fixture();
            var faults = new FaultInjectingMatchStore(fixture.Store);

            WebApplicationFactory<Program> host = Track(fixture.WithWebHostBuilder(
                builder => builder.ConfigureServices(services =>
                {
                    services.RemoveAll<IMatchStore>();
                    services.AddSingleton<IMatchStore>(faults);
                })));

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(host);
            await using (zero)
            await using (one)
            {
                // The command reaches the store and never comes back, so the match gate stays held for the
                // whole of shutdown. Everything that waits on that gate has to give up rather than wait.
                faults.HangEveryAppend();
                await zero.SendCmdAsync(Opening);
                await faults.AppendReached.WaitAsync(TimeSpan.FromSeconds(10));

                var clock = Stopwatch.StartNew();
                Task stopping = host.Services.GetRequiredService<IHost>().StopAsync(CancellationToken.None);

                // The seat that is NOT wedged is still owed a goodbye, and a 1012 rather than a torn socket.
                Assert.That(await one.ExpectAsync(GracefulShutdownService.RestartNotice),
                    Is.EqualTo(GracefulShutdownService.RestartNotice));
                Assert.That((int)(await one.ExpectCloseAsync())!,
                    Is.EqualTo(GracefulShutdownService.RestartCloseStatus));

                await stopping.WaitAsync(Deadline);
                clock.Stop();

                Assert.That(clock.Elapsed, Is.LessThan(Deadline),
                    "a wedged store must not carry shutdown past the platform kill deadline");

                faults.ReleaseHungAppends();
            }
        }

        [Test]
        public async Task Shutdown_FinishesWhenAClientWillNotCompleteTheCloseHandshake()
        {
            SteamServerFactory fixture = Fixture();
            using HttpClient warm = fixture.CreateClient();

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(fixture);
            await using (zero)
            await using (one)
            {
                // A client whose process went away: the socket is still open as far as this host knows,
                // and nothing will ever answer its close frame. An unbounded close handshake waits on that
                // peer forever, which is the second way shutdown used to overrun its budget.
                zero.Drop();

                var clock = Stopwatch.StartNew();
                await fixture.Services.GetRequiredService<IHost>().StopAsync(CancellationToken.None)
                    .WaitAsync(Deadline);
                clock.Stop();

                Assert.That(clock.Elapsed, Is.LessThan(Deadline),
                    "a peer that will not answer must be aborted, not waited on");
                Assert.That(await one.ExpectAsync(GracefulShutdownService.RestartNotice),
                    Is.EqualTo(GracefulShutdownService.RestartNotice));
                Assert.That((int)(await one.ExpectCloseAsync())!,
                    Is.EqualTo(GracefulShutdownService.RestartCloseStatus));

                one.Drop();
            }
        }

        [Test]
        public async Task Shutdown_TellsBothSeatsClosesThemAndSaysWhatItDid()
        {
            var logging = new CapturingLoggerProvider();
            SteamServerFactory fixture = Fixture();
            fixture.Logging = logging;

            using HttpClient warm = fixture.CreateClient();

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(fixture);
            await using (zero)
            await using (one)
            {
                await fixture.Services.GetRequiredService<IHost>().StopAsync(CancellationToken.None)
                    .WaitAsync(Deadline);

                foreach (DurableFlowClient seat in new[] { zero, one })
                {
                    Assert.That(await seat.ExpectAsync(GracefulShutdownService.RestartNotice),
                        Is.EqualTo(GracefulShutdownService.RestartNotice));
                    Assert.That((int)(await seat.ExpectCloseAsync())!,
                        Is.EqualTo(GracefulShutdownService.RestartCloseStatus));
                }

                Assert.That(logging.Any("Shutdown: readiness false"), Is.True);
                Assert.That(
                    logging.Messages.Any(m => m.Contains("1 live match(es), 2 socket(s) closed")
                        && m.Contains("0 not drained")),
                    Is.True,
                    "the summary is what an operator reads after the process is gone: "
                    + string.Join(" | ", logging.Messages.Where(m => m.Contains("Shutdown:"))));

                zero.Drop();
                one.Drop();
            }
        }

        [Test]
        public async Task Shutdown_RefusesACommandThatArrivesAfterTheBarrier()
        {
            SteamServerFactory fixture = Fixture();
            using HttpClient warm = fixture.CreateClient();

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(fixture);
            await using (zero)
            await using (one)
            {
                var coordinator = fixture.Services.GetRequiredService<DurableMatchCoordinator>();
                Assert.That(coordinator.TryGetLiveMatch(zero.MatchId, out LiveMatch? match), Is.True);

                int before = match!.LastSequence;
                coordinator.BeginShutdown();

                await zero.SendCmdAsync(Opening);

                Assert.That(await zero.ExpectAsync("REJECT "),
                    Is.EqualTo(DurableMatchCoordinator.RejectTemporaryFailure),
                    "the client retries after it reconnects, which is the contract this answer already has");
                Assert.That(match.LastSequence, Is.EqualTo(before), "nothing was appended past the barrier");

                await one.ExpectNothingAsync(TimeSpan.FromMilliseconds(500));
            }
        }

        [Test]
        public async Task Shutdown_ClosesASocketThatArrivesWhileTheSnapshotIsBeingTaken()
        {
            SteamServerFactory fixture = Fixture();
            using HttpClient warm = fixture.CreateClient();

            DurableFlowClient joining = await DurableFlowClient.CreateAsync(
                fixture, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket);

            await using (joining)
            {
                // The upgrade is accepted first, so this socket is past every admission check there was.
                await joining.OpenAsync();

                fixture.Services.GetRequiredService<DurableMatchCoordinator>().BeginShutdown();
                fixture.Services.GetRequiredService<V2ConnectionRegistry>().BeginShutdown();

                await joining.SendAuthAsync();

                Assert.That((int)(await joining.ExpectCloseAsync())!,
                    Is.EqualTo(GracefulShutdownService.RestartCloseStatus),
                    "a socket admitted around the snapshot gets the same 1012 as one that was in it");

                var coordinator = fixture.Services.GetRequiredService<DurableMatchCoordinator>();
                Assert.That(coordinator.ConnectionCount, Is.Zero, "it was never seated");
            }
        }
    }
}

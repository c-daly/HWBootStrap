using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HexWars.Engine;
using HexWars.NetServer.Configuration;
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
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// One row per counter this server publishes, driven through the path that is supposed to move it.
    ///
    /// The assertion that matters is the second one: nothing ELSE moved. A counter that quietly also fires
    /// from a neighbouring path reads as a healthy total and sends whoever is watching it to the wrong
    /// place, and that is a mistake no single-path test can see. Adding a counter without adding a row
    /// here leaves it unproven; changing which path feeds one fails the row that named it.
    /// </summary>
    [TestFixture]
    public class MetricsMatrixTests
    {
        const string Token = "metrics-token-for-tests";

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

        /// <summary>What a row expects to move: the snapshot property, and the tag inside it when the
        /// counter carries one.</summary>
        public sealed record Expectation(string Counter, string? Tag);

        /// <summary>A row: a name, what to do, and the one counter that may move while it happens.</summary>
        public sealed class Row
        {
            public required string Name { get; init; }
            public required Expectation Expect { get; init; }

            /// <summary>Everything this path needs to exist first. Run BEFORE the snapshot, so a match
            /// that had to be created in order to refuse a command is not read as a match created.</summary>
            public Func<MetricsMatrixTests, Fixture, Task>? Arrange { get; init; }

            /// <summary>The one thing being measured.</summary>
            public required Func<MetricsMatrixTests, Fixture, Task> Act { get; init; }

            /// <summary>Totals this path legitimately moves as well. A ticket Valve refused is both a
            /// Steam failure and an authentication failure, and pretending otherwise would mean choosing
            /// which of the two an operator is allowed to see.</summary>
            public string[] AlsoMoves { get; init; } = Array.Empty<string>();

            public override string ToString() => Name;
        }

        /// <summary>The host, its fault seam, and two seats already playing. Everything a row needs, built
        /// before the snapshot so setup never shows up as the thing being measured.</summary>
        public sealed class Fixture
        {
            public required SteamServerFactory Factory { get; init; }
            public required WebApplicationFactory<Program> Host { get; init; }
            public required FaultInjectingMatchStore Faults { get; init; }
            public required HttpClient Client { get; init; }
            public DurableFlowClient? Zero { get; set; }
            public DurableFlowClient? One { get; set; }
        }

        static readonly IReadOnlyList<Row> Rows = new[]
        {
            new Row
            {
                Name = "matchesCreated",
                Expect = new Expectation("matchesCreated", null),
                Act = async (test, f) => await DurableFlowClient
                    .CreateAsync(f.Host, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket)
                    .ContinueWith(t => t.Result.DisposeAsync().AsTask()).Unwrap(),
            },
            new Row
            {
                Name = "commandsCommitted",
                Expect = new Expectation("commandsCommitted", null),
                Arrange = (test, f) => test.SeatBothAsync(f),
                Act = async (test, f) =>
                {
                    await f.Zero!.SendCmdAsync(Opening);
                    await f.Zero.ExpectAsync("APPLY ");
                    await f.One!.ExpectAsync("APPLY ");
                },
            },
            new Row
            {
                Name = "commandsRejected.WrongSeat",
                Expect = new Expectation("rejectionsByReason", "WrongSeat"),
                Arrange = (test, f) => test.SeatBothAsync(f),
                Act = async (test, f) =>
                {
                    await f.Zero!.SendCmdAsync(new EndTurn(PlayerId.Player1));
                    await f.Zero.ExpectAsync("REJECT ");
                },
            },
            new Row
            {
                Name = "catalogRejected.CatalogClosed",
                Expect = new Expectation("catalogRejectionsByReason", "CatalogClosed"),
                Arrange = (test, f) => test.SeatBothAsync(f),
                Act = async (test, f) =>
                {
                    await f.Zero!.SendCatalogAsync();
                    await f.Zero.ExpectAsync("REJECT ");
                },
            },
            new Row
            {
                Name = "dbFailures.append",
                Expect = new Expectation("databaseFailuresByOp", "append"),

                // An append that threw is a database failure AND a command the issuer was refused. Both
                // are true and both are wanted: one says the store is unhappy, the other says a player
                // was told to try again.
                AlsoMoves = new[] { "commandsRejected" },
                Arrange = (test, f) => test.SeatBothAsync(f),
                Act = async (test, f) =>
                {
                    f.Faults.FailNextAppend(new InvalidOperationException("the database is not there"));
                    await f.Zero!.SendCmdAsync(Opening);
                    await f.Zero.ExpectAsync("REJECT ");
                },
            },
            new Row
            {
                Name = "dbFailures.touch",
                Expect = new Expectation("databaseFailuresByOp", "touch"),
                Arrange = async (test, f) => f.Zero = await DurableFlowClient.CreateAsync(
                    f.Host, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket),
                Act = async (test, f) =>
                {
                    f.Faults.BeforeTouch = () => throw new InvalidOperationException("no stamp for you");
                    await f.Zero!.ConnectAsync();
                    f.Faults.BeforeTouch = null;
                },
            },
            new Row
            {
                Name = "steamFailures.OwnershipMissing",
                Expect = new Expectation("steamFailuresByKind", "OwnershipMissing"),
                Act = async (test, f) =>
                {
                    f.Factory.Steam.Ownership.Remove(FakeSteamWebApiClient.OwnerSteamId);
                    using HttpResponseMessage refused = await f.Client.PostAsJsonAsync(
                        DurableFlowClient.CreateRoute,
                        new
                        {
                            steamLobbyId = FakeSteamWebApiClient.LobbyId,
                            ticket = FakeSteamWebApiClient.OwnerTicket,
                        });

                    Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
                },
            },
            new Row
            {
                Name = "steamFailures.lobby_changed",
                Expect = new Expectation("steamFailuresByKind", "lobby_changed"),
                Act = async (test, f) =>
                {
                    // The setup the client asked for is not the setup the lobby now advertises. Refused
                    // locally, without Valve saying anything, which is exactly the gap this row closes.
                    using HttpResponseMessage refused = await f.Client.PostAsJsonAsync(
                        DurableFlowClient.CreateRoute,
                        new
                        {
                            steamLobbyId = FakeSteamWebApiClient.LobbyId,
                            ticket = FakeSteamWebApiClient.OwnerTicket,
                            requestedSetup = new GameSetup(
                                GameSetup.Default.Mode, 11, 9, 0, 4242, 0, 0, 0, 0, 1, false).ToWire(),
                        });

                    Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
                },
            },
            new Row
            {
                Name = "steamFailures.AuthenticationFailed",
                Expect = new Expectation("steamFailuresByKind", "AuthenticationFailed"),
                AlsoMoves = new[] { "authFailures" },
                Act = async (test, f) =>
                {
                    using HttpResponseMessage refused = await f.Client.PostAsJsonAsync(
                        DurableFlowClient.CreateRoute,
                        new { steamLobbyId = FakeSteamWebApiClient.LobbyId, ticket = "deadbeef" });

                    Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));
                },
            },
            new Row
            {
                Name = "authFailures.frame",
                Expect = new Expectation("authFailuresByStage", "frame"),
                Arrange = async (test, f) => f.Zero = await DurableFlowClient.CreateAsync(
                    f.Host, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket),
                Act = async (test, f) =>
                {
                    await f.Zero!.OpenAsync();
                    await f.Zero.SendAsync("HELLO there");
                    await f.Zero.ExpectAsync("AUTH FAIL ");
                },
            },
            new Row
            {
                Name = "authFailures.credential",
                Expect = new Expectation("authFailuresByStage", "credential"),
                Arrange = async (test, f) => f.Zero = await DurableFlowClient.CreateAsync(
                    f.Host, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket),
                Act = async (test, f) =>
                {
                    await f.Zero!.OpenAsync();
                    await f.Zero.SendAsync("AUTH " + Guid.NewGuid() + " " + new string('a', 43));
                    await f.Zero.ExpectAsync("AUTH FAIL ");
                },
            },
            new Row
            {
                Name = "reconnects",
                Expect = new Expectation("reconnects", null),
                Arrange = (test, f) => test.SeatBothAsync(f),
                Act = async (test, f) =>
                {
                    // The same seat, a second socket, inside the window. The first is superseded, which
                    // is what a reconnect looks like from here.
                    DurableFlowClient again = await DurableFlowClient.JoinAsync(
                        f.Host, f.Zero!.MatchId, FakeSteamWebApiClient.OwnerTicket);

                    await again.ConnectAsync();
                    await again.DisposeAsync();
                },
            },
        };

        public static IEnumerable<Row> Cases() => Rows;

        [TestCaseSource(nameof(Cases))]
        public async Task EachPathMovesItsOwnCounterAndNoOther(Row row)
        {
            Fixture fixture = Build();

            if (row.Arrange is not null) await row.Arrange(this, fixture);

            JsonElement before = await SnapshotAsync(fixture.Client);
            await row.Act(this, fixture);
            JsonElement after = await SnapshotAsync(fixture.Client);

            Assert.That(Read(after, row.Expect), Is.EqualTo(Read(before, row.Expect) + 1),
                row.Name + " did not move");

            foreach (string counter in Counters)
            {
                if (counter == row.Expect.Counter
                    || row.AlsoMoves.Contains(counter)
                    || (row.Expect.Tag is not null && Tagged.TryGetValue(counter, out string? owner)
                        && owner == row.Expect.Counter))
                {
                    continue;
                }

                Assert.That(after.GetProperty(counter).GetInt64(),
                    Is.EqualTo(before.GetProperty(counter).GetInt64()),
                    counter + " moved while " + row.Name + " was being driven");
            }

            if (fixture.Zero is not null) await fixture.Zero.DisposeAsync();
            if (fixture.One is not null) await fixture.One.DisposeAsync();
        }

        /// <summary>Every untagged total in the snapshot. A row that moves one of these while claiming
        /// another is the regression this whole file exists to catch.</summary>
        static readonly string[] Counters =
        {
            "matchesCreated", "commandsCommitted", "commandsRejected", "catalogRejected",
            "databaseFailures", "steamFailures", "authFailures", "reconnects", "recoveryFailures",
        };

        /// <summary>Which total a tagged map belongs to, so a row that names the map is not also asked to
        /// hold its own total still.</summary>
        static readonly IReadOnlyDictionary<string, string> Tagged =
            new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["commandsRejected"] = "rejectionsByReason",
                ["catalogRejected"] = "catalogRejectionsByReason",
                ["databaseFailures"] = "databaseFailuresByOp",
                ["steamFailures"] = "steamFailuresByKind",
                ["authFailures"] = "authFailuresByStage",
            };

        static long Read(JsonElement snapshot, Expectation expect)
        {
            JsonElement counter = snapshot.GetProperty(expect.Counter);
            if (expect.Tag is null) return counter.GetInt64();

            return counter.TryGetProperty(expect.Tag, out JsonElement tagged) ? tagged.GetInt64() : 0;
        }

        Fixture Build()
        {
            SteamServerFactory factory = Track(new SteamServerFactory());
            factory.Settings[HexWarsConfiguration.MatchMetricsTokenKey] = Token;
            factory.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] = FakeSteamWebApiClient.ReadyLobby(
                ruleset: SteamLobbyRules.CustomRuleset, setupWire: GameSetup.Default.ToWire());

            var faults = new FaultInjectingMatchStore(factory.Store);

            WebApplicationFactory<Program> host = Track(factory.WithWebHostBuilder(
                builder => builder.ConfigureServices(services =>
                {
                    services.RemoveAll<IMatchStore>();
                    services.AddSingleton<IMatchStore>(faults);
                })));

            return new Fixture
            {
                Factory = factory,
                Host = host,
                Faults = faults,
                Client = Track(host.CreateClient()),
            };
        }

        internal async Task SeatBothAsync(Fixture f)
        {
            DurableFlowClient zero = await DurableFlowClient.CreateAsync(
                f.Host, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket);
            DurableFlowClient one = await DurableFlowClient.JoinAsync(
                f.Host, zero.MatchId, FakeSteamWebApiClient.GuestTicket);

            await zero.ConnectAsync();
            await zero.ExpectAsync(NetProtocol.CatalogRequest);
            await one.ConnectAsync();
            await one.ExpectAsync(NetProtocol.CatalogRequest);

            await zero.SendCatalogAsync();
            await one.SendCatalogAsync();

            await zero.ExpectAsync("START ");
            await one.ExpectAsync("START ");

            f.Zero = zero;
            f.One = one;
        }

        static async Task<JsonElement> SnapshotAsync(HttpClient client)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, HealthEndpoints.MetricsRoute);
            request.Headers.Add(HealthEndpoints.MetricsTokenHeader, Token);

            using HttpResponseMessage response = await client.SendAsync(request);
            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));

            return JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.Clone();
        }
    }
}

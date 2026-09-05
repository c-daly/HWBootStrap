using System.Net.WebSockets;
using HexWars.Engine;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The exit gate for the whole durable design: killing the process at any command boundary and starting
    /// a fresh one over the same database cannot lose, duplicate, reorder or alter an acknowledged action.
    ///
    /// Every test here runs against real Postgres through the real host, because that is the only place the
    /// claim can be tested. The projection, the journal, the credential store and the socket all have to
    /// agree after a restart, and any double that stood in for one of them would be a double that agreed
    /// with itself. What is deliberately NOT faked is the process boundary: host A is disposed - its
    /// sockets, its coordinator and its in-memory projection all go with it - and host B is built from
    /// nothing but the rows A left behind.
    ///
    /// The scripted game is the deterministic default-seed sequence the rest of this repository uses. The
    /// lobby advertises the custom ruleset carrying GameSetup.Default rather than quick-v1, because the
    /// legality of the opening move is a fact about seed 7 and the quick ruleset pins a different seed.
    /// </summary>
    [TestFixture]
    public class MatchRestartRecoveryTests
    {
        const string LobbyId = FakeSteamWebApiClient.LobbyId;

        /// <summary>The tickets the second host knows. A reconnecting client presents a fresh ticket and is
        /// issued a fresh credential; nothing a client held before the restart is reused.</summary>
        const string OwnerTicketAfterRestart = "b1b2c3d4";
        const string GuestTicketAfterRestart = "b5f60718";

        /// <summary>
        /// Five legal commands from the default start state, in order.
        ///
        /// The first two are the pair SelfTest and MatchJournalReplayTests both use: on seed 7 the
        /// deterministic placement puts the Striker of Player0 within reach of one of Player1, so the move
        /// lands and the attack connects. The three passes after them exist so there is a boundary to kill
        /// at for every k the exit gate asks about.
        /// </summary>
        static readonly Command[] Script =
        {
            new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0)),
            new AttackUnit(PlayerId.Player0, 2, 5),
            new EndTurn(PlayerId.Player0),
            new EndTurn(PlayerId.Player1),
            new EndTurn(PlayerId.Player0),
        };

        /// <summary>The barracks both seats send, in the form the server stores and reloads it in.</summary>
        static readonly IReadOnlyList<UnitTemplate> Barracks =
            BarracksWire.Read(BarracksWire.Write(BarracksCatalog.DefaultTemplates));

        /// <summary>How long a frame that must NOT arrive is given to arrive.</summary>
        static readonly TimeSpan Quiet = TimeSpan.FromMilliseconds(500);

        // ---- hosts ------------------------------------------------------------

        /// <summary>
        /// A host over the shared Postgres whose lobby advertises the default setup.
        /// </summary>
        /// <param name="reset">False for the host that takes over: the database it inherits is the point.</param>
        static async Task<SteamServerFactory> HostAsync(bool reset)
        {
            SteamServerFactory factory = await SteamServerFactory.PostgresAsync(reset);

            factory.Steam.Lobbies[LobbyId] = FakeSteamWebApiClient.ReadyLobby(
                ruleset: SteamLobbyRules.CustomRuleset, setupWire: GameSetup.Default.ToWire());

            factory.Steam.Identify(OwnerTicketAfterRestart, FakeSteamWebApiClient.OwnerSteamId);
            factory.Steam.Identify(GuestTicketAfterRestart, FakeSteamWebApiClient.GuestSteamId);

            return factory;
        }

        /// <summary>Starts the host and hands back what its startup recovery pass concluded.</summary>
        static RecoveryReport RecoveryOf(WebApplicationFactory<Program> host)
        {
            using HttpClient started = host.CreateClient();

            var state = host.Services.GetRequiredService<RecoveryState>();
            Assert.That(state.Completed, Is.True, "the startup pass runs before the host serves anything");
            Assert.That(state.Error, Is.Null, state.Error?.Message);

            return state.Report!;
        }

        // ---- playing ----------------------------------------------------------

        /// <summary>Allocates a match, seats both players, starts it, and plays the first
        /// <paramref name="commands"/> commands of the script.</summary>
        static async Task<(DurableFlowClient Zero, DurableFlowClient One)> PlayAsync(
            WebApplicationFactory<Program> host, int commands)
        {
            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(host);

            for (var i = 0; i < commands; i++) await ApplyAsync(zero, one, Script[i]);

            return (zero, one);
        }

        /// <summary>Both seats seated on a freshly started match, with no commands played.</summary>
        static async Task<(DurableFlowClient Zero, DurableFlowClient One)> StartAsync(
            WebApplicationFactory<Program> host)
        {
            DurableFlowClient zero =
                await DurableFlowClient.CreateAsync(host, LobbyId, FakeSteamWebApiClient.OwnerTicket);
            DurableFlowClient one =
                await DurableFlowClient.JoinAsync(host, zero.MatchId, FakeSteamWebApiClient.GuestTicket);

            Assert.That(zero.Seat, Is.EqualTo(0));
            Assert.That(one.Seat, Is.EqualTo(1));

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

        /// <summary>Issues one command from the seat that owns it and waits for both seats to be told.</summary>
        static async Task ApplyAsync(DurableFlowClient zero, DurableFlowClient one, Command command)
        {
            DurableFlowClient issuer = command.Issuer == PlayerId.Player0 ? zero : one;
            await issuer.SendCmdAsync(command);

            string broadcast = NetProtocol.Apply(command);
            Assert.That(await zero.ExpectAsync("APPLY "), Is.EqualTo(broadcast));
            Assert.That(await one.ExpectAsync("APPLY "), Is.EqualTo(broadcast));
        }

        /// <summary>Both seats back on a host that has just come up, each with a brand new credential.</summary>
        static async Task<(DurableFlowClient Zero, DurableFlowClient One)> RejoinAsync(
            WebApplicationFactory<Program> host, Guid matchId)
        {
            DurableFlowClient zero =
                await DurableFlowClient.JoinAsync(host, matchId, OwnerTicketAfterRestart);
            DurableFlowClient one =
                await DurableFlowClient.JoinAsync(host, matchId, GuestTicketAfterRestart);

            Assert.That(zero.Seat, Is.EqualTo(0), "a seat belongs to the account, not to the process");
            Assert.That(one.Seat, Is.EqualTo(1));

            return (zero, one);
        }

        // ---- the database, read directly --------------------------------------

        /// <summary>The command journal as rows, read on a pool this test owns.</summary>
        static async Task<IReadOnlyList<(int Sequence, string Wire)>> JournalAsync(
            PostgresTestDatabase database, Guid matchId)
        {
            await using NpgsqlCommand command = database.DataSource.CreateCommand(
                "SELECT sequence, command_wire FROM match_commands WHERE match_id = @match ORDER BY sequence");
            command.Parameters.AddWithValue("match", matchId);

            var rows = new List<(int Sequence, string Wire)>();
            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync()) rows.Add((reader.GetInt32(0), reader.GetString(1)));

            return rows;
        }

        /// <summary>Asserts the journal is exactly the first <paramref name="commands"/> of the script, in
        /// order, with no gaps and nothing written twice.</summary>
        static async Task AssertJournalIsAsync(
            PostgresTestDatabase database, Guid matchId, int commands)
        {
            IReadOnlyList<(int Sequence, string Wire)> journal = await JournalAsync(database, matchId);

            Assert.That(journal, Has.Count.EqualTo(commands), "a duplicate or a lost command");
            Assert.That(journal.Select(row => row.Sequence),
                Is.EqualTo(Enumerable.Range(1, commands)), "sequences must be 1..n with no gaps");
            Assert.That(journal.Select(row => row.Wire),
                Is.EqualTo(Script.Take(commands).Select(CommandWire.Write)),
                "sequence i must still hold the wire of command i");
        }

        /// <summary>
        /// The state a fresh engine reaches from the same start by applying the first
        /// <paramref name="commands"/> commands, with no server involved at all.
        ///
        /// This is what makes the restart assertions mean something. Comparing the two halves of a restart
        /// against each other would pass for a server that dealt both of them the same wrong game.
        /// </summary>
        static string DirectReplayText(int commands)
        {
            GameState state = GameFactory.Build(GameSetup.Default, Barracks, Barracks);

            for (var i = 0; i < commands; i++)
            {
                Result applied = GameEngine.Apply(state, Script[i]);
                Assert.That(applied.Success, Is.True,
                    "the scripted game is not legal at step " + (i + 1).ToString() + ": " + applied.Reason);
                state = applied.NewState;
            }

            return ReplayFile.Write(state, Array.Empty<Command>());
        }

        /// <summary>
        /// Every step of the scripted game moves the state somewhere new.
        ///
        /// It is the guard on every state comparison below. If two prefixes of the script serialised to the
        /// same text, a server that dealt the wrong number of commands after a restart would still satisfy
        /// the state assertions, and the boundary test would be passing for no reason at all.
        /// </summary>
        [Test]
        public void TheScriptedGameMovesAtEveryStep()
        {
            string[] positions = Enumerable.Range(0, Script.Length + 1).Select(DirectReplayText).ToArray();

            Assert.That(positions, Is.Unique);
        }

        // ---- the exit gate ----------------------------------------------------

        [Test]
        public async Task RestartPreservesJournalAndContinues()
        {
            Guid matchId;
            string playedByA;

            await using (SteamServerFactory a = await HostAsync(reset: true))
            {
                (DurableFlowClient zero, DurableFlowClient one) = await PlayAsync(a, 3);
                await using (zero)
                await using (one)
                {
                    matchId = zero.MatchId;
                    playedByA = zero.ReplayText;

                    Assert.That(zero.Log, Has.Count.EqualTo(3));
                    Assert.That(one.ReplayText, Is.EqualTo(playedByA), "both seats saw the same game");
                    Assert.That(zero.StateText, Is.EqualTo(DirectReplayText(3)));
                }
            }

            // Host A is gone: its sockets, its coordinator and its projection with it. Everything below is
            // built from the rows it left behind and nothing else.
            await using SteamServerFactory b = await HostAsync(reset: false);

            RecoveryReport report = RecoveryOf(b);
            Assert.That(report.Verified, Is.EqualTo(1), "the open match must be replayable before anyone asks");
            Assert.That(report.Failed, Is.Empty);

            (DurableFlowClient zeroB, DurableFlowClient oneB) = await RejoinAsync(b, matchId);
            await using (zeroB)
            await using (oneB)
            {
                await zeroB.ConnectAsync();
                await zeroB.ExpectAsync("START ");
                await oneB.ConnectAsync();
                await oneB.ExpectAsync("START ");

                Assert.That(zeroB.Log, Has.Count.EqualTo(3),
                    "the re-deal carries exactly the commands that were journalled");
                Assert.That(zeroB.ReplayText, Is.EqualTo(playedByA),
                    "the new host deals the same start and the same log the old one was playing");
                Assert.That(zeroB.StateText, Is.EqualTo(DirectReplayText(3)));
                Assert.That(oneB.ReplayText, Is.EqualTo(playedByA));

                await ApplyAsync(zeroB, oneB, Script[3]);

                Assert.That(zeroB.Log, Has.Count.EqualTo(4), "the game continues where it stopped");
                Assert.That(zeroB.StateText, Is.EqualTo(DirectReplayText(4)));
            }

            await AssertJournalIsAsync(b.Database!, matchId, 4);
        }

        /// <summary>
        /// The same restart, taken at every boundary there is: before the first command, between each pair,
        /// and after the last.
        ///
        /// A durability bug that only shows up at one boundary is the normal kind. Losing the command that
        /// was in flight, replaying the last one twice, or dealing a start state that has quietly moved on
        /// would each pass at some values of k and fail at others, which is why every k is run rather than
        /// one representative one.
        /// </summary>
        [TestCase(0)]
        [TestCase(1)]
        [TestCase(2)]
        [TestCase(3)]
        [TestCase(4)]
        [TestCase(5)]
        public async Task KillAtEveryBoundary(int killAfter)
        {
            Guid matchId;
            string beforeTheKill;

            await using (SteamServerFactory a = await HostAsync(reset: true))
            {
                (DurableFlowClient zero, DurableFlowClient one) = await PlayAsync(a, killAfter);
                await using (zero)
                await using (one)
                {
                    matchId = zero.MatchId;
                    beforeTheKill = zero.ReplayText;

                    Assert.That(zero.Log, Has.Count.EqualTo(killAfter));
                    Assert.That(zero.StateText, Is.EqualTo(DirectReplayText(killAfter)));
                }
            }

            await using SteamServerFactory b = await HostAsync(reset: false);

            RecoveryReport report = RecoveryOf(b);
            Assert.That(report.Verified, Is.EqualTo(1));
            Assert.That(report.Failed, Is.Empty);

            (DurableFlowClient zeroB, DurableFlowClient oneB) = await RejoinAsync(b, matchId);
            await using (zeroB)
            await using (oneB)
            {
                await zeroB.ConnectAsync();
                await zeroB.ExpectAsync("START ");
                await oneB.ConnectAsync();
                await oneB.ExpectAsync("START ");

                Assert.That(zeroB.Log, Has.Count.EqualTo(killAfter),
                    "the re-deal must hold exactly the commands the journal holds");
                Assert.That(zeroB.ReplayText, Is.EqualTo(beforeTheKill));
                Assert.That(zeroB.StateText, Is.EqualTo(DirectReplayText(killAfter)));

                for (int i = killAfter; i < Script.Length; i++) await ApplyAsync(zeroB, oneB, Script[i]);

                Assert.That(zeroB.StateText, Is.EqualTo(DirectReplayText(Script.Length)));
                Assert.That(oneB.LogWire, Is.EqualTo(Script.Select(CommandWire.Write)));
            }

            await AssertJournalIsAsync(b.Database!, matchId, Script.Length);
        }

        /// <summary>
        /// A client that saw its command applied and then lost the socket must not resend it.
        ///
        /// This is the rule that keeps a reconnect from forking a game. The command was durable before the
        /// APPLY left the server, so the re-deal already carries it; a client that helpfully sent it again
        /// would be asking for the same move twice.
        /// </summary>
        [Test]
        public async Task DisconnectAfterCommitReconnectSeesCommand()
        {
            await using SteamServerFactory host = await HostAsync(reset: true);

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(host);
            Guid matchId = zero.MatchId;

            await using (one)
            {
                await using (zero)
                {
                    await ApplyAsync(zero, one, Script[0]);

                    // No close handshake, nothing acknowledged: the shape of a client whose process died
                    // between seeing the APPLY and doing anything about it.
                    zero.Drop();
                }

                await using DurableFlowClient back =
                    await DurableFlowClient.JoinAsync(host, matchId, OwnerTicketAfterRestart);

                await back.ConnectAsync();
                await back.ExpectAsync("START ");

                Assert.That(back.Log, Has.Count.EqualTo(1),
                    "the re-deal already carries the command that was committed");
                Assert.That(back.StateText, Is.EqualTo(DirectReplayText(1)));

                // What happens if a client resends anyway, documented rather than assumed: the engine is
                // asked first, and it refuses, because on the state the server actually holds that unit has
                // already moved this turn. The frame never reaches the journal, so there is no second row
                // and no second sequence - the duplicate is stopped by the rules of the game, one layer
                // above the uniqueness constraint that would have stopped it anyway.
                await back.SendCmdAsync(Script[0]);
                string answer = await back.ExpectAsync("REJECT ");

                Assert.That(answer, Is.Not.EqualTo(DurableMatchCoordinator.RejectTemporaryFailure),
                    "this is the engine refusing a repeat, not a write that failed");
                await one.ExpectNothingAsync(Quiet);
            }

            await AssertJournalIsAsync(host.Database!, matchId, 1);
        }

        /// <summary>
        /// A journal write that fails leaves nothing behind: no broadcast, no advanced projection, no row,
        /// and the command still usable exactly as it was.
        ///
        /// The store here is the real Postgres one with a single append made to throw, so the retry is
        /// checked against the journal that actually exists rather than against a double that agreed with
        /// itself about what it had stored.
        /// </summary>
        [Test]
        public async Task DbFailureDuringCommandDoesNotFork()
        {
            await using SteamServerFactory fixture = await HostAsync(reset: true);

            FaultInjectingMatchStore? faulty = null;
            using WebApplicationFactory<Program> host = fixture.WithWebHostBuilder(builder =>
                builder.ConfigureServices(services =>
                {
                    services.RemoveAll<IMatchStore>();
                    services.AddSingleton<IMatchStore>(provider => faulty ??= new FaultInjectingMatchStore(
                        ActivatorUtilities.CreateInstance<PostgresMatchStore>(provider)));
                }));

            (DurableFlowClient zero, DurableFlowClient one) = await StartAsync(host);
            Guid matchId = zero.MatchId;

            await using (zero)
            await using (one)
            {
                Assert.That(faulty, Is.Not.Null, "the host must be resolving the wrapped store");
                faulty!.FailNextAppend(new NpgsqlException("the database went away mid-append"));

                await zero.SendCmdAsync(Script[0]);

                Assert.That(await zero.ExpectAsync("REJECT "),
                    Is.EqualTo(DurableMatchCoordinator.RejectTemporaryFailure));
                await one.ExpectNothingAsync(Quiet);

                Assert.That(faulty.AppendsFailed, Is.EqualTo(1));

                // The identical command, unchanged. A temporary failure means nothing was committed and
                // nobody was told anything, so the command is still the command it always was.
                await ApplyAsync(zero, one, Script[0]);

                Assert.That(zero.Log, Has.Count.EqualTo(1));
                Assert.That(zero.StateText, Is.EqualTo(DirectReplayText(1)));
            }

            await AssertJournalIsAsync(fixture.Database!, matchId, 1);
        }

        /// <summary>
        /// A build that cannot reproduce the rules a journal was written under refuses the whole match, at
        /// startup and at the handshake, and touches none of it.
        ///
        /// Refusing is the only honest answer. A replay under changed rules does not fail - it produces a
        /// perfectly valid-looking different game, which the players would then be judged on.
        /// </summary>
        [Test]
        public async Task UnsupportedContractRefusesAuth()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            Guid matchId;

            await using (SteamServerFactory a = await HostAsync(reset: true))
            {
                (DurableFlowClient zero, DurableFlowClient one) = await PlayAsync(a, 2);
                await using (zero)
                await using (one)
                {
                    matchId = zero.MatchId;
                }
            }

            await using (NpgsqlCommand rewrite = database.DataSource.CreateCommand(
                "UPDATE matches SET engine_version = @version WHERE match_id = @match"))
            {
                rewrite.Parameters.AddWithValue("version", "hexwars-engine/999");
                rewrite.Parameters.AddWithValue("match", matchId);
                Assert.That(await rewrite.ExecuteNonQueryAsync(), Is.EqualTo(1));
            }

            await using SteamServerFactory b = await HostAsync(reset: false);

            RecoveryReport report = RecoveryOf(b);
            Assert.That(report.Verified, Is.EqualTo(0));
            Assert.That(report.Failed.Select(failure => failure.MatchId), Does.Contain(matchId));
            Assert.That(
                report.Failed.Single(failure => failure.MatchId == matchId).Failure,
                Is.EqualTo(MatchRecoveryFailure.UnsupportedEngineContract));

            await using DurableFlowClient refused =
                await DurableFlowClient.JoinAsync(b, matchId, OwnerTicketAfterRestart);

            await refused.OpenAsync();
            await refused.SendAuthAsync();

            Assert.That(await refused.ExpectAsync("AUTH FAIL "), Is.EqualTo("AUTH FAIL unavailable"),
                "unavailable rather than invalid: the credential is fine, this build is not");
            Assert.That(await refused.ExpectCloseAsync(), Is.EqualTo(WebSocketCloseStatus.PolicyViolation));

            await AssertJournalIsAsync(database, matchId, 2);
        }

        /// <summary>
        /// A restart while the match is still waiting for barracks does not lose the barracks that already
        /// arrived: the other seat finishes the start, and both are dealt the same game.
        /// </summary>
        [Test]
        public async Task GracefulRecoveryWhileWaiting()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            Guid matchId;

            await using (SteamServerFactory a = await HostAsync(reset: true))
            {
                DurableFlowClient zero =
                    await DurableFlowClient.CreateAsync(a, LobbyId, FakeSteamWebApiClient.OwnerTicket);
                DurableFlowClient one =
                    await DurableFlowClient.JoinAsync(a, zero.MatchId, FakeSteamWebApiClient.GuestTicket);

                await using (zero)
                await using (one)
                {
                    matchId = zero.MatchId;

                    await zero.ConnectAsync();
                    await zero.ExpectAsync(NetProtocol.CatalogRequest);
                    await one.ConnectAsync();
                    await one.ExpectAsync(NetProtocol.CatalogRequest);

                    // One barracks, and then the process goes away.
                    await zero.SendCatalogAsync();
                    await WaitForStoredCatalogAsync(database, matchId, 0);
                }
            }

            await using SteamServerFactory b = await HostAsync(reset: false);

            RecoveryReport report = RecoveryOf(b);
            Assert.That(report.Verified, Is.EqualTo(1), "a waiting match is an open match");
            Assert.That(report.Failed, Is.Empty);

            (DurableFlowClient zeroB, DurableFlowClient oneB) = await RejoinAsync(b, matchId);
            await using (zeroB)
            await using (oneB)
            {
                await zeroB.ConnectAsync();

                // No CATALOG? for this seat: what it sent before the restart was stored, so it is not
                // asked for it again. Being asked twice is how a player loses an army list.
                await zeroB.ExpectNothingAsync(Quiet);

                await oneB.ConnectAsync();
                await oneB.ExpectAsync(NetProtocol.CatalogRequest);
                await oneB.SendCatalogAsync();

                await zeroB.ExpectAsync("START ");
                await oneB.ExpectAsync("START ");

                Assert.That(zeroB.Log, Is.Empty);
                Assert.That(zeroB.ReplayText, Is.EqualTo(oneB.ReplayText));
                Assert.That(zeroB.StateText, Is.EqualTo(DirectReplayText(0)));
            }
        }

        /// <summary>
        /// Both barracks durable and the start never written: a restart finishes it rather than leaving the
        /// match waiting for a catalog neither client will ever send again.
        /// </summary>
        [Test]
        public async Task BothCatalogsStoredAndNoStart_IsResumedAfterTheRestart()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            Guid matchId;

            await using (SteamServerFactory a = await HostAsync(reset: true))
            {
                DurableFlowClient zero =
                    await DurableFlowClient.CreateAsync(a, LobbyId, FakeSteamWebApiClient.OwnerTicket);
                DurableFlowClient one =
                    await DurableFlowClient.JoinAsync(a, zero.MatchId, FakeSteamWebApiClient.GuestTicket);

                await using (zero)
                await using (one)
                {
                    matchId = zero.MatchId;

                    await zero.ConnectAsync();
                    await zero.ExpectAsync(NetProtocol.CatalogRequest);
                    await zero.SendCatalogAsync();
                    await WaitForStoredCatalogAsync(database, matchId, 0);

                    // The second barracks is made durable behind the coordinator, which is precisely the
                    // window a kill lands in: the row is written and the start that would have followed it
                    // never runs. Sending it through the socket would start the match, which is the state
                    // this test exists to avoid arriving at.
                    await StoreCatalogDirectlyAsync(database, matchId, seat: 1);
                }
            }

            Assert.That(await StatusOfAsync(database, matchId), Is.EqualTo("waiting"));

            await using SteamServerFactory b = await HostAsync(reset: false);

            RecoveryReport report = RecoveryOf(b);
            Assert.That(report.Verified, Is.EqualTo(1));
            Assert.That(report.Failed, Is.Empty);

            (DurableFlowClient zeroB, DurableFlowClient oneB) = await RejoinAsync(b, matchId);
            await using (zeroB)
            await using (oneB)
            {
                await zeroB.ConnectAsync();

                // START, and never CATALOG?: this seat has already answered that question, and nothing
                // else was ever going to start this match.
                await zeroB.ExpectAsync("START ");
                Assert.That(zeroB.Log, Is.Empty);
                Assert.That(zeroB.StateText, Is.EqualTo(DirectReplayText(0)));

                await oneB.ConnectAsync();
                await oneB.ExpectAsync("START ");
                Assert.That(oneB.ReplayText, Is.EqualTo(zeroB.ReplayText),
                    "both seats are dealt the one start state the row now holds");
            }

            Assert.That(await StatusOfAsync(database, matchId), Is.EqualTo("active"));
        }

        /// <summary>Writes a barracks straight into the row, without the coordinator seeing it.</summary>
        static async Task StoreCatalogDirectlyAsync(
            PostgresTestDatabase database, Guid matchId, int seat)
        {
            await using NpgsqlCommand stored = database.DataSource.CreateCommand(
                "UPDATE match_players SET catalog_wire = @wire WHERE match_id = @match AND seat = @seat");
            stored.Parameters.AddWithValue("wire", BarracksWire.Write(Barracks));
            stored.Parameters.AddWithValue("match", matchId);
            stored.Parameters.AddWithValue("seat", seat);

            Assert.That(await stored.ExecuteNonQueryAsync(), Is.EqualTo(1));
        }

        /// <summary>The stored status of one match, as the text the column actually holds.</summary>
        static async Task<string> StatusOfAsync(PostgresTestDatabase database, Guid matchId)
        {
            await using NpgsqlCommand status = database.DataSource.CreateCommand(
                "SELECT status FROM matches WHERE match_id = @match");
            status.Parameters.AddWithValue("match", matchId);

            return (string)(await status.ExecuteScalarAsync())!;
        }

        /// <summary>Waits until a barracks is durable, so a test can drop the host knowing the write landed
        /// rather than hoping it did.</summary>
        static async Task WaitForStoredCatalogAsync(PostgresTestDatabase database, Guid matchId, int seat)
        {
            DateTimeOffset deadline = DateTimeOffset.UtcNow.AddSeconds(15);

            while (DateTimeOffset.UtcNow < deadline)
            {
                await using NpgsqlCommand stored = database.DataSource.CreateCommand(
                    "SELECT count(*) FROM match_players WHERE match_id = @match AND seat = @seat "
                    + "AND catalog_wire IS NOT NULL");
                stored.Parameters.AddWithValue("match", matchId);
                stored.Parameters.AddWithValue("seat", seat);

                if ((long)(await stored.ExecuteScalarAsync())! == 1L) return;

                await Task.Delay(50);
            }

            Assert.Fail("the barracks for seat " + seat.ToString() + " was never stored");
        }
    }
}

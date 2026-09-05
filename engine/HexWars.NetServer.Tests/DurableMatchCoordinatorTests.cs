using HexWars.Engine;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The rule this whole layer exists for: a command is durable before anybody is told about it.
    ///
    /// Every test below is a way of asking the same question from a different side. A write that fails must
    /// leave the game exactly where it was, with nothing broadcast and the same command still usable. Two
    /// players hammering one match at once must produce a journal with no gaps. A projection that has fallen
    /// behind the database must refuse rather than guess. And the moment the last APPLY of a game leaves the
    /// server, the match must already be recorded as finished - a client that sees the winning blow and then
    /// reconnects into a match the database still calls active would be watching two different games.
    /// </summary>
    [TestFixture]
    public class DurableMatchCoordinatorTests
    {
        const string Seat0Steam = "76561190000000001";
        const string Seat1Steam = "76561190000000002";
        const string LobbyId = "109775240000000042";

        static readonly DateTimeOffset Begin = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

        static CancellationToken Ct => CancellationToken.None;

        InMemoryMatchStore _store = null!;
        FakeTimeProvider _clock = null!;
        RecordingConnectionSink _sink = null!;
        MatchCredentialService _credentials = null!;
        DurableMatchCoordinator _coordinator = null!;
        Guid _matchId;
        string _credential0 = null!;
        string _credential1 = null!;

        [SetUp]
        public async Task SeedAWaitingMatchWithACredentialForEachSeat()
        {
            _store = new InMemoryMatchStore();
            _clock = new FakeTimeProvider(Begin);
            _sink = new RecordingConnectionSink();
            _credentials = new MatchCredentialService(
                _store,
                Options.Create(new MatchHostingOptions()),
                _clock,
                NullLogger<MatchCredentialService>.Instance);

            CreateMatchResult created = await _store.CreateMatchForLobbyAsync(new CreateMatchRequest(
                LobbyId, GameSetup.Default.ToWire(), "hexwars-engine/1", 2, "test-build",
                new[] { (Seat0Steam, 0), (Seat1Steam, 1) }, Begin), Ct);
            _matchId = created.Match.MatchId;

            _credential0 = (await _credentials.IssueAsync(_matchId, Seat0Steam, Ct)).Credential;
            _credential1 = (await _credentials.IssueAsync(_matchId, Seat1Steam, Ct)).Credential;

            _coordinator = new DurableMatchCoordinator(
                _store,
                _credentials,
                new JournalLiveMatchLoader(_store),
                _sink,
                Options.Create(new MatchHostingOptions()),
                _clock,
                NullLogger<DurableMatchCoordinator>.Instance);
        }

        // ---- helpers ---------------------------------------------------------

        Task<DurableMatchCoordinator.AuthOutcome> Auth(string connectionId, string credential) =>
            _coordinator.AuthenticateAsync(connectionId, _matchId.ToString(), credential, Ct);

        async Task SeatBothPlayers()
        {
            await Auth("c0", _credential0);
            await Auth("c1", _credential1);
        }

        async Task StartTheMatch()
        {
            await SeatBothPlayers();
            string wire = BarracksWire.Write(BarracksCatalog.DefaultTemplates);
            await _coordinator.ReceiveAsync("c0", NetProtocol.Catalog(wire), Ct);
            await _coordinator.ReceiveAsync("c1", NetProtocol.Catalog(wire), Ct);
        }

        Task Cmd(string connectionId, Command command) =>
            _coordinator.ReceiveAsync(connectionId, NetProtocol.Cmd(command), Ct);

        Task Cmd(Command command) =>
            Cmd(command.Issuer == PlayerId.Player0 ? "c0" : "c1", command);

        /// <summary>The start state the coordinator actually dealt, read back from the row it wrote.</summary>
        async Task<GameState> DealtStartState()
        {
            PersistedMatch row = (await _store.GetMatchAsync(_matchId, Ct))!;
            return ReplayFile.Read(row.StartReplay!).Start;
        }

        LiveMatch Live()
        {
            Assert.That(_coordinator.TryGetLiveMatch(_matchId, out LiveMatch? live), Is.True,
                "the coordinator should still be holding this match");
            return live!;
        }

        // ---- handshake -------------------------------------------------------

        [Test]
        public async Task EachAuthenticatedSeat_IsToldItsSeatAndAskedForACatalog()
        {
            DurableMatchCoordinator.AuthOutcome zero = await Auth("c0", _credential0);
            DurableMatchCoordinator.AuthOutcome one = await Auth("c1", _credential1);

            Assert.That(zero.Ok, Is.True);
            Assert.That(zero.Seat, Is.EqualTo(0));
            Assert.That(zero.FailCode, Is.Null);
            Assert.That(one.Ok, Is.True);
            Assert.That(one.Seat, Is.EqualTo(1));

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "SEAT 0", "CATALOG?" }));
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { "SEAT 1", "CATALOG?" }));
            Assert.That(_coordinator.ConnectionCount, Is.EqualTo(2));
            Assert.That(_coordinator.ConnectionsOf(_matchId), Is.EquivalentTo(new[] { "c0", "c1" }));
        }

        [Test]
        public async Task ACredentialThatWasNeverIssued_IsRefusedWithoutASeat()
        {
            DurableMatchCoordinator.AuthOutcome outcome = await Auth("cX", "this-is-not-a-credential");

            Assert.That(outcome.Ok, Is.False);
            Assert.That(outcome.FailCode, Is.EqualTo("invalid"));
            Assert.That(_sink.Sent, Is.Empty);
            Assert.That(_coordinator.ConnectionCount, Is.EqualTo(0));
        }

        [Test]
        public async Task AMatchIdThatIsNotAGuid_IsRefusedBeforeAnythingIsLoaded()
        {
            DurableMatchCoordinator.AuthOutcome outcome =
                await _coordinator.AuthenticateAsync("cX", "not-a-guid", _credential0, Ct);

            Assert.That(outcome.Ok, Is.False);
            Assert.That(outcome.FailCode, Is.EqualTo("invalid"));
            Assert.That(_coordinator.LiveMatchCount, Is.EqualTo(0));
        }

        // ---- the start -------------------------------------------------------

        [Test]
        public async Task BothCatalogs_StartTheMatchAndDealTheSameStartStateToBothSeats()
        {
            await StartTheMatch();

            PersistedMatch row = (await _store.GetMatchAsync(_matchId, Ct))!;
            Assert.That(row.Status, Is.EqualTo(MatchStatus.Active));
            Assert.That(row.StartReplay, Is.Not.Null);

            string dealt = NetProtocol.Start(row.StartReplay!);
            Assert.That(_sink.MessagesFor("c0").Last(), Is.EqualTo(dealt));
            Assert.That(_sink.MessagesFor("c1").Last(), Is.EqualTo(dealt));

            IReadOnlyList<PersistedPlayer> players = await _store.GetPlayersAsync(_matchId, Ct);
            Assert.That(players.Select(p => p.CatalogWire), Is.All.Not.Null);

            LiveMatch live = Live();
            Assert.That(live.Status, Is.EqualTo(MatchStatus.Active));
            Assert.That(live.LastSequence, Is.EqualTo(0));
            Assert.That(live.Log, Is.Empty);
        }

        [Test]
        public async Task OneCatalogAlone_StartsNothing()
        {
            await SeatBothPlayers();
            _sink.Clear();

            await _coordinator.ReceiveAsync(
                "c0", NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)), Ct);

            Assert.That(_sink.Sent, Is.Empty);
            Assert.That((await _store.GetMatchAsync(_matchId, Ct))!.Status, Is.EqualTo(MatchStatus.Waiting));
        }

        [Test]
        public async Task ACatalogAfterTheStart_IsRefused()
        {
            await StartTheMatch();
            _sink.Clear();

            await _coordinator.ReceiveAsync(
                "c0", NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)), Ct);

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "REJECT CatalogClosed" }));
        }

        [Test]
        public async Task ACommandBeforeTheMatchStarts_IsRefused()
        {
            await SeatBothPlayers();
            _sink.Clear();

            await Cmd("c0", new EndTurn(PlayerId.Player0));

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "REJECT CatalogV1Required" }));
        }

        // ---- commands --------------------------------------------------------

        [Test]
        public async Task AnAcceptedCommand_IsJournalledAndBroadcastToBothSeats()
        {
            await StartTheMatch();
            _sink.Clear();

            await Cmd("c0", new EndTurn(PlayerId.Player0));

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "APPLY E 0" }));
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { "APPLY E 0" }));

            MatchJournal journal = (await _store.LoadJournalAsync(_matchId, Ct))!;
            Assert.That(journal.Commands.Select(c => c.Sequence), Is.EqualTo(new[] { 1 }));
            Assert.That(journal.Commands[0].CommandWire, Is.EqualTo("E 0"));
            Assert.That(journal.Commands[0].IssuerSteamId, Is.EqualTo(Seat0Steam));
            Assert.That(Live().LastSequence, Is.EqualTo(1));
        }

        [Test]
        public async Task TheCommandIsAlreadyInTheJournalWhenTheApplyIsSent()
        {
            await StartTheMatch();

            int journalledWhenTheFirstApplyLeft = -1;
            _sink.OnSend = (_, message) =>
            {
                if (!message.StartsWith("APPLY ", StringComparison.Ordinal)) return;
                if (journalledWhenTheFirstApplyLeft >= 0) return;
                journalledWhenTheFirstApplyLeft =
                    _store.LoadJournalAsync(_matchId, Ct).GetAwaiter().GetResult()!.Commands.Count;
            };

            await Cmd("c0", new EndTurn(PlayerId.Player0));

            Assert.That(journalledWhenTheFirstApplyLeft, Is.EqualTo(1),
                "the APPLY went out before the command was durable");
        }

        [Test]
        public async Task ADurableWriteThatFails_BroadcastsNothingAndLeavesTheCommandStillPlayable()
        {
            await StartTheMatch();
            _sink.Clear();
            _store.InjectedWriteFailure = new InvalidOperationException("the database went away");

            await Cmd("c0", new EndTurn(PlayerId.Player0));

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "REJECT TemporaryFailure" }));
            Assert.That(_sink.MessagesFor("c1"), Is.Empty);
            Assert.That((await _store.LoadJournalAsync(_matchId, Ct))!.Commands, Is.Empty);
            Assert.That(Live().LastSequence, Is.EqualTo(0));

            _sink.Clear();
            await Cmd("c0", new EndTurn(PlayerId.Player0));

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "APPLY E 0" }));
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { "APPLY E 0" }));
            Assert.That((await _store.LoadJournalAsync(_matchId, Ct))!.Commands, Has.Count.EqualTo(1));
        }

        [Test]
        public async Task OneSeatCannotIssueACommandForTheOther()
        {
            await StartTheMatch();
            _sink.Clear();
            int writesBefore = _store.WriteCount;

            await Cmd("c1", new EndTurn(PlayerId.Player0));

            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { "REJECT WrongSeat" }));
            Assert.That(_sink.MessagesFor("c0"), Is.Empty);
            Assert.That(_store.WriteCount, Is.EqualTo(writesBefore));
            Assert.That(Live().LastSequence, Is.EqualTo(0));
        }

        [Test]
        public async Task AnIllegalCommand_IsRefusedWithTheEngineReasonAndWritesNothing()
        {
            await StartTheMatch();
            var illegal = new MoveUnit(PlayerId.Player0, 999, new HexCoord(0, 0));
            Result expected = GameEngine.Apply(await DealtStartState(), illegal);
            Assert.That(expected.Success, Is.False, "the test needs a command the engine actually refuses");

            _sink.Clear();
            int writesBefore = _store.WriteCount;

            await Cmd("c0", illegal);

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { NetProtocol.Reject(expected.Reason) }));
            Assert.That(_sink.MessagesFor("c1"), Is.Empty);
            Assert.That(_store.WriteCount, Is.EqualTo(writesBefore));
        }

        [Test]
        public async Task AMalformedCommand_IsRefusedWithoutTouchingTheJournal()
        {
            await StartTheMatch();
            _sink.Clear();
            int writesBefore = _store.WriteCount;

            await _coordinator.ReceiveAsync("c0", "CMD ZZZ not a command", Ct);

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { NetProtocol.Malformed }));
            Assert.That(_store.WriteCount, Is.EqualTo(writesBefore));
        }

        [Test]
        public async Task AFrameFromAConnectionThatNeverAuthenticated_IsRefused()
        {
            await _coordinator.ReceiveAsync("ghost", NetProtocol.Cmd(new EndTurn(PlayerId.Player0)), Ct);

            Assert.That(_sink.MessagesFor("ghost"), Is.EqualTo(new[] { "REJECT NoSeat" }));
        }

        [Test]
        public async Task APongIsIgnored()
        {
            await StartTheMatch();
            _sink.Clear();

            await _coordinator.ReceiveAsync("c0", "PONG", Ct);

            Assert.That(_sink.Sent, Is.Empty);
        }

        // ---- reconnect -------------------------------------------------------

        [Test]
        public async Task AReconnect_ReDealsTheStartStatePlusEveryCommandSoFar()
        {
            await StartTheMatch();
            await Cmd("c0", new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0)));
            await Cmd("c0", new AttackUnit(PlayerId.Player0, 2, 5));
            _sink.Clear();

            DurableMatchCoordinator.AuthOutcome outcome = await Auth("c0-again", _credential0);

            Assert.That(outcome.Ok, Is.True);
            Assert.That(outcome.Seat, Is.EqualTo(0));

            IReadOnlyList<string> dealt = _sink.MessagesFor("c0-again");
            Assert.That(dealt, Has.Count.EqualTo(2));
            Assert.That(dealt[0], Is.EqualTo("SEAT 0"));
            Assert.That(dealt[1], Does.StartWith("START "));

            ReplayData reDealt = ReplayFile.Read(dealt[1]["START ".Length..]);
            Assert.That(reDealt.Commands, Has.Count.EqualTo(2));

            GameState fastForwarded = reDealt.Start;
            foreach (Command command in reDealt.Commands)
            {
                Result applied = GameEngine.Apply(fastForwarded, command);
                Assert.That(applied.Success, Is.True, "a re-dealt command was refused on replay");
                fastForwarded = applied.NewState;
            }

            Assert.That(ReplayFile.Write(fastForwarded, Array.Empty<Command>()),
                Is.EqualTo(ReplayFile.Write(Live().State!, Array.Empty<Command>())));
        }

        // ---- the end of a game -----------------------------------------------

        [Test]
        public async Task TheMatchIsRecordedAsFinishedBeforeTheFinalApplyIsBroadcast()
        {
            await StartTheMatch();

            // The game is played out offline first, through the same deterministic engine, so the test feeds
            // the coordinator a command sequence that is known to end in a terminal state rather than hoping
            // one turns up. Replaying it through the coordinator must reproduce it exactly.
            MatchRecord played = Match.Record(
                await DealtStartState(), new GreedyAgent(11), new GreedyAgent(29), maxCommands: 4000);
            Assert.That(played.Result.Final.IsGameOver, Is.True,
                "the offline driver never reached a terminal state, so this test proves nothing");

            var statusAtEachApply = new List<MatchStatus>();
            var winnerAtEachApply = new List<int?>();
            _sink.OnSend = (_, message) =>
            {
                if (!message.StartsWith("APPLY ", StringComparison.Ordinal)) return;
                PersistedMatch row = _store.GetMatchAsync(_matchId, Ct).GetAwaiter().GetResult()!;
                statusAtEachApply.Add(row.Status);
                winnerAtEachApply.Add(row.WinnerSeat);
            };

            foreach (Command command in played.Commands) await Cmd(command);

            int? expectedWinner = played.Result.Final.Winner is PlayerId w ? (int)w : null;

            // Two APPLYs per command, one per seat: the last two belong to the winning blow.
            Assert.That(statusAtEachApply, Has.Count.EqualTo(played.Commands.Count * 2),
                "every recorded command should have been accepted by the coordinator");
            Assert.That(statusAtEachApply[^1], Is.EqualTo(MatchStatus.Completed));
            Assert.That(statusAtEachApply[^2], Is.EqualTo(MatchStatus.Completed));
            Assert.That(winnerAtEachApply[^1], Is.EqualTo(expectedWinner));
            Assert.That(statusAtEachApply.Take(statusAtEachApply.Count - 2), Is.All.EqualTo(MatchStatus.Active));

            Assert.That(Live().Status, Is.EqualTo(MatchStatus.Completed));

            _sink.Clear();
            await Cmd("c0", new EndTurn(PlayerId.Player0));
            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "REJECT CatalogV1Required" }));
        }

        // ---- concurrency and staleness ---------------------------------------

        [Test]
        public async Task FortyConcurrentCommands_LeaveAContiguousJournalAndAMatchingProjection()
        {
            await StartTheMatch();

            var submissions = new List<Task>();
            for (int i = 0; i < 40; i++)
            {
                bool seatZero = i % 2 == 0;
                string connection = seatZero ? "c0" : "c1";
                Command command = new EndTurn(seatZero ? PlayerId.Player0 : PlayerId.Player1);
                submissions.Add(Task.Run(() => Cmd(connection, command)));
            }

            await Task.WhenAll(submissions);

            MatchJournal journal = (await _store.LoadJournalAsync(_matchId, Ct))!;
            Assert.That(journal.Commands, Is.Not.Empty);
            Assert.That(journal.Commands.Select(c => c.Sequence),
                Is.EqualTo(Enumerable.Range(1, journal.Commands.Count)));

            LiveMatch live = Live();
            Assert.That(live.LastSequence, Is.EqualTo(journal.Commands.Count));
            Assert.That(LiveMatch.FromJournal(journal).StartReplayText(), Is.EqualTo(live.StartReplayText()));
        }

        [Test]
        public async Task AJournalWrittenBehindTheProjection_ForcesAReloadAndATemporaryFailure()
        {
            await StartTheMatch();
            _sink.Clear();

            var behindTheBack = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));
            AppendResult direct = await _store.AppendCommandAsync(
                _matchId, 1, CommandWire.Write(behindTheBack), Seat0Steam, Begin, Ct);
            Assert.That(direct.Status, Is.EqualTo(AppendStatus.Appended));

            await Cmd("c0", new EndTurn(PlayerId.Player0));

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "REJECT TemporaryFailure" }));
            Assert.That(_sink.MessagesFor("c1"), Is.Empty);

            LiveMatch live = Live();
            Assert.That(live.LastSequence, Is.EqualTo(1));
            Assert.That(live.Log.Select(CommandWire.Write), Is.EqualTo(new[] { "M 0 2 3 0" }));

            _sink.Clear();
            await Cmd("c0", new AttackUnit(PlayerId.Player0, 2, 5));

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "APPLY A 0 2 5" }));
            Assert.That((await _store.LoadJournalAsync(_matchId, Ct))!.Commands.Select(c => c.Sequence),
                Is.EqualTo(new[] { 1, 2 }));
        }

        // ---- lifecycle -------------------------------------------------------

        [Test]
        public async Task DrainReturnsPromptlyWhenNothingIsInFlight()
        {
            await StartTheMatch();

            Task drain = _coordinator.DrainAsync(TimeSpan.FromSeconds(5));
            await drain.WaitAsync(TimeSpan.FromSeconds(5));

            Assert.That(drain.IsCompletedSuccessfully, Is.True);
        }

        [Test]
        public async Task DrainWaitsForACommitThatIsStillInFlight()
        {
            await StartTheMatch();
            LiveMatch live = Live();
            await live.Gate.WaitAsync(Ct);

            Task drain = _coordinator.DrainAsync(TimeSpan.FromSeconds(10));
            await Task.Delay(150);
            Assert.That(drain.IsCompleted, Is.False, "drain returned while a commit still held the gate");

            live.Gate.Release();
            await drain.WaitAsync(TimeSpan.FromSeconds(10));
        }

        [Test]
        public async Task BroadcastAllReachesEveryConnection()
        {
            await StartTheMatch();
            _sink.Clear();

            await _coordinator.BroadcastAllAsync("SERVER RESTART");

            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "SERVER RESTART" }));
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { "SERVER RESTART" }));
        }

        [Test]
        public async Task EvictClosesEveryConnectionAndForgetsTheMatch()
        {
            await StartTheMatch();
            Assert.That(_coordinator.ConnectionCount, Is.EqualTo(2));

            await _coordinator.EvictAsync(new[] { _matchId }, 1012, "SERVER RESTART");

            Assert.That(_sink.Closed.Select(c => c.ConnectionId), Is.EquivalentTo(new[] { "c0", "c1" }));
            Assert.That(_sink.Closed.Select(c => c.CloseStatus), Is.All.EqualTo(1012));
            Assert.That(_sink.Closed.Select(c => c.Reason), Is.All.EqualTo("SERVER RESTART"));
            Assert.That(_coordinator.LiveMatchCount, Is.EqualTo(0));
            Assert.That(_coordinator.ConnectionCount, Is.EqualTo(0));
        }

        [Test]
        public async Task ADisconnectedMatchIsSweptOutOfMemoryButNotOutOfTheDatabase()
        {
            await StartTheMatch();
            await _coordinator.DisconnectAsync("c0");
            await _coordinator.DisconnectAsync("c1");
            Assert.That(_coordinator.ConnectionCount, Is.EqualTo(0));

            _clock.Advance(TimeSpan.FromMinutes(9));
            await _coordinator.SweepAsync(_clock.GetUtcNow());
            Assert.That(_coordinator.LiveMatchCount, Is.EqualTo(1), "nine minutes idle is not ten");

            _clock.Advance(TimeSpan.FromMinutes(2));
            await _coordinator.SweepAsync(_clock.GetUtcNow());
            Assert.That(_coordinator.LiveMatchCount, Is.EqualTo(0));
            Assert.That((await _store.GetMatchAsync(_matchId, Ct))!.Status, Is.EqualTo(MatchStatus.Active),
                "sweeping releases memory and must never end a game");
        }

        [Test]
        public async Task AMatchWithALiveConnectionIsNeverSwept()
        {
            await StartTheMatch();

            _clock.Advance(TimeSpan.FromHours(2));
            await _coordinator.SweepAsync(_clock.GetUtcNow());

            Assert.That(_coordinator.LiveMatchCount, Is.EqualTo(1));
        }
    }
}

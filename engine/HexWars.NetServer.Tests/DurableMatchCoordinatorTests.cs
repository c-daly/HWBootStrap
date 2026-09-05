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

        /// <summary>The store the coordinator actually resolves, wrapping <see cref="_store"/>. Every fault
        /// below is armed on it, and every arrangement a test makes behind the coordinator back is written
        /// straight to the inner store so the fault seams do not see it.</summary>
        FaultInjectingMatchStore _faults = null!;

        MatchHostingOptions _options = null!;
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
            _faults = new FaultInjectingMatchStore(_store);
            _options = new MatchHostingOptions();
            _clock = new FakeTimeProvider(Begin);
            _sink = new RecordingConnectionSink();
            _credentials = new MatchCredentialService(
                _faults,
                Options.Create(_options),
                _clock,
                NullLogger<MatchCredentialService>.Instance);

            CreateMatchResult created = await _store.CreateMatchForLobbyAsync(new CreateMatchRequest(
                LobbyId, GameSetup.Default.ToWire(), "hexwars-engine/1", 2, "test-build",
                new[] { (Seat0Steam, 0), (Seat1Steam, 1) }, Begin), Ct);
            _matchId = created.Match.MatchId;

            _credential0 = (await _credentials.IssueAsync(_matchId, Seat0Steam, Ct)).Credential;
            _credential1 = (await _credentials.IssueAsync(_matchId, Seat1Steam, Ct)).Credential;

            _coordinator = NewCoordinator();
        }

        /// <summary>A coordinator over the same store with no projections of its own: what a restart looks
        /// like from the database, which is the only place a restart is visible.</summary>
        DurableMatchCoordinator NewCoordinator() => new(
            _faults,
            _credentials,
            new JournalLiveMatchLoader(_faults),
            _sink,
            Options.Create(_options),
            _clock,
            NullLogger<DurableMatchCoordinator>.Instance);

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
            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { DurableMatchCoordinator.RejectMatchEnded }),
                "a finished match is a different refusal from one that has not started: this one is final");
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

            // The rebuild found a command these seats had never been told about, so both of them are dealt
            // the log before the issuer is told to try again. Correcting the projection silently would
            // leave two clients playing a position the journal no longer agrees with.
            Assert.That(_sink.MessagesFor("c0").First(), Does.StartWith("START "));
            Assert.That(_sink.MessagesFor("c0").First(), Does.Contain(CommandWire.Write(behindTheBack)));
            Assert.That(_sink.MessagesFor("c0").Last(), Is.EqualTo("REJECT TemporaryFailure"));
            Assert.That(_sink.MessagesFor("c1"),
                Is.EqualTo(new[] { _sink.MessagesFor("c0").First() }),
                "the seat that issued nothing missed the same command and is dealt the same log");

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

        // ---- the end of a game that did not finish closing --------------------

        /// <summary>Plays the scripted game to one command short of the end and hands back what is left.</summary>
        async Task<(MatchRecord Played, GameState Start)> PlayToTheBrinkAsync()
        {
            await StartTheMatch();

            GameState start = await DealtStartState();
            MatchRecord played = Match.Record(start, new GreedyAgent(11), new GreedyAgent(29), maxCommands: 4000);
            Assert.That(played.Result.Final.IsGameOver, Is.True,
                "the offline driver never reached a terminal state, so this test proves nothing");

            for (var i = 0; i < played.Commands.Count - 1; i++) await Cmd(played.Commands[i]);

            return (played, start);
        }

        [Test]
        public async Task AKillBetweenTheWinningAppendAndTheCompletion_IsHealedByTheNextHandshake()
        {
            (MatchRecord played, GameState start) = await PlayToTheBrinkAsync();

            // The winning command commits and no attempt to record the win lands. That is exactly the state
            // a process killed in the gap leaves behind: a game the engine calls over and the database calls
            // active. Every attempt fails, because one that failed once is retried immediately.
            _faults.FailEveryCompletion(new InvalidOperationException("the database went away"));
            _sink.Clear();
            await Cmd(played.Commands[^1]);

            Assert.That(_sink.Sent.Any(s => s.Message.StartsWith("APPLY ", StringComparison.Ordinal)), Is.False,
                "a command whose completion failed must not be broadcast");
            Assert.That(_sink.Sent.Any(s => s.Message == DurableMatchCoordinator.RejectTemporaryFailure),
                Is.False, "the command is durable, so the issuer must never be invited to send it again");
            Assert.That(_sink.Closed, Is.EqualTo(new[]
            {
                ("c0", DurableMatchCoordinator.ResyncCloseStatus,
                    DurableMatchCoordinator.ResyncCloseReason),
            }), "the issuer is disconnected instead, and learns the ending on its reconnect");

            _faults.FailEveryCompletion(null);

            PersistedMatch stranded = (await _store.GetMatchAsync(_matchId, Ct))!;
            Assert.That(stranded.Status, Is.EqualTo(MatchStatus.Active), "this is the state being healed");

            MatchJournal journal = (await _store.LoadJournalAsync(_matchId, Ct))!;
            Assert.That(journal.Commands, Has.Count.EqualTo(played.Commands.Count),
                "the winning command is durable; only the completion is missing");

            // A restart: a coordinator with no projections, over nothing but the rows above.
            DurableMatchCoordinator restarted = NewCoordinator();
            _sink.Clear();

            DurableMatchCoordinator.AuthOutcome back =
                await restarted.AuthenticateAsync("c2", _matchId.ToString(), _credential0, Ct);

            Assert.That(back.Ok, Is.True, back.FailCode);

            PersistedMatch healed = (await _store.GetMatchAsync(_matchId, Ct))!;
            Assert.That(healed.Status, Is.EqualTo(MatchStatus.Completed));
            Assert.That(healed.WinnerSeat,
                Is.EqualTo(played.Result.Final.Winner is PlayerId w ? (int)w : null));

            Assert.That(_sink.MessagesFor("c2"), Is.EqualTo(new[]
            {
                "SEAT 0",
                NetProtocol.Start(ReplayFile.Write(start, played.Commands)),
            }), "the reconnecting seat is fast-forwarded to the position the game ended in");
        }

        [Test]
        public async Task ACompletionThatThrew_IsRetriedByTheNextHandshakeOnTheSameHost()
        {
            (MatchRecord played, GameState start) = await PlayToTheBrinkAsync();

            _faults.FailEveryCompletion(new InvalidOperationException("the database went away"));
            await Cmd(played.Commands[^1]);

            Assert.That((await _store.GetMatchAsync(_matchId, Ct))!.Status, Is.EqualTo(MatchStatus.Active));

            _faults.FailEveryCompletion(null);
            _sink.Clear();

            // No restart at all. The same host, the same projection, the next thing anybody does with it.
            await Auth("c2", _credential1);

            Assert.That((await _store.GetMatchAsync(_matchId, Ct))!.Status, Is.EqualTo(MatchStatus.Completed));
            Assert.That(Live().Status, Is.EqualTo(MatchStatus.Completed));

            // Both seats that were watching are dealt the terminal state. Neither of them ever saw the
            // APPLY that ended the game, so healing the row without telling them would leave two clients
            // sitting in a position that no longer exists.
            string terminal = NetProtocol.Start(ReplayFile.Write(start, played.Commands));
            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { terminal }));
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { terminal }));
        }

        [Test]
        public async Task ACompletionThatFailedOnce_IsRetriedImmediatelyAndTheGameEndsNormally()
        {
            (MatchRecord played, GameState _) = await PlayToTheBrinkAsync();

            // One failure is a connection that has just been replaced, not a database that is gone. The
            // completion is one idempotent UPDATE, so the immediate retry is the cheapest correct answer
            // and the players never learn anything happened.
            _faults.FailNextCompletion(new InvalidOperationException("the connection was reset"));
            _sink.Clear();
            await Cmd(played.Commands[^1]);

            string expected = NetProtocol.Apply(played.Commands[^1]);
            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { expected }));
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { expected }));
            Assert.That(_sink.Closed, Is.Empty);
            Assert.That((await _store.GetMatchAsync(_matchId, Ct))!.Status, Is.EqualTo(MatchStatus.Completed));
        }

        [Test]
        public async Task AMatchAbandonedInTheCompletionWindow_StillGetsItsApplyAndThenLosesItsSockets()
        {
            (MatchRecord played, GameState _) = await PlayToTheBrinkAsync();

            // Somebody else ends this match between the append of the winning command and the completion of
            // it - the reaper, or an operator. TryCompleteMatchAsync then answers false, which is NOT a
            // success: the command is durable and owed to both seats, but the match they are watching is
            // gone.
            var abandonedOnce = false;
            _faults.BeforeCompletion = async () =>
            {
                if (abandonedOnce) return;
                abandonedOnce = true;
                await _store.TryCompleteMatchAsync(
                    _matchId, MatchStatus.Abandoned, null, _clock.GetUtcNow(), Ct);
            };

            _sink.Clear();
            await Cmd(played.Commands[^1]);

            string expected = NetProtocol.Apply(played.Commands[^1]);
            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { expected }));
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { expected }),
                "a durable command reaches both seats whatever the row now says");

            PersistedMatch row = (await _store.GetMatchAsync(_matchId, Ct))!;
            Assert.That(row.Status, Is.EqualTo(MatchStatus.Abandoned), "the row was not overwritten");
            Assert.That(row.WinnerSeat, Is.Null);

            Assert.That(_sink.Closed.Select(c => c.ConnectionId), Is.EquivalentTo(new[] { "c0", "c1" }));
            Assert.That(_sink.Closed.Select(c => c.CloseStatus),
                Is.All.EqualTo(DurableMatchCoordinator.MatchEndedCloseStatus));
            Assert.That(_sink.Closed.Select(c => c.Reason),
                Is.All.EqualTo(DurableMatchCoordinator.MatchEndedCloseReason));
            Assert.That(_coordinator.ConnectionsOf(_matchId), Is.Empty);
        }

        // ---- the terminal reconnect window ------------------------------------

        /// <summary>Plays the scripted game to its end, completion and all.</summary>
        async Task<(MatchRecord Played, GameState Start)> PlayToTheEndAsync()
        {
            (MatchRecord played, GameState start) = await PlayToTheBrinkAsync();
            await Cmd(played.Commands[^1]);

            Assert.That((await _store.GetMatchAsync(_matchId, Ct))!.Status, Is.EqualTo(MatchStatus.Completed));
            return (played, start);
        }

        [Test]
        public async Task ASeatThatMissedTheFinalApply_ReconnectsInsideTheWindowAndIsDealtTheEnding()
        {
            (MatchRecord played, GameState start) = await PlayToTheEndAsync();

            _clock.Advance(TimeSpan.FromSeconds(30));
            _sink.Clear();

            DurableMatchCoordinator.AuthOutcome back = await Auth("c2", _credential1);

            Assert.That(back.Ok, Is.True, back.FailCode);
            Assert.That(back.Seat, Is.EqualTo(1));
            Assert.That(_sink.MessagesFor("c2"), Is.EqualTo(new[]
            {
                "SEAT 1",
                NetProtocol.Start(ReplayFile.Write(start, played.Commands)),
            }));

            _sink.Clear();
            await Cmd("c2", new EndTurn(PlayerId.Player1));
            Assert.That(_sink.MessagesFor("c2"),
                Is.EqualTo(new[] { DurableMatchCoordinator.RejectMatchEnded }),
                "the window is for learning how it ended, not for playing on");
        }

        [Test]
        public async Task AfterTheTerminalWindowHasPassed_TheSameCredentialIsRefused()
        {
            await PlayToTheEndAsync();

            // Past the reconnect window and still inside the credential TTL, so the refusal can only be
            // about the window.
            _clock.Advance(
                TimeSpan.FromSeconds(MatchHostingOptions.DefaultTerminalReconnectSeconds + 60));
            Assert.That(_clock.GetUtcNow() - Begin,
                Is.LessThan(TimeSpan.FromSeconds(MatchHostingOptions.DefaultJoinTokenTtlSeconds)));

            DurableMatchCoordinator.AuthOutcome refused = await Auth("c3", _credential1);

            Assert.That(refused.Ok, Is.False);
            Assert.That(refused.FailCode, Is.EqualTo(DurableMatchCoordinator.AuthFailInvalid));
        }

        [Test]
        public async Task WithTheWindowClosed_AFinishedMatchIsRefusedImmediately()
        {
            _options.TerminalReconnectSeconds = 0;
            await PlayToTheEndAsync();

            DurableMatchCoordinator.AuthOutcome refused = await Auth("c3", _credential1);

            Assert.That(refused.Ok, Is.False);
            Assert.That(refused.FailCode, Is.EqualTo(DurableMatchCoordinator.AuthFailInvalid));
        }

        // ---- a start that was interrupted -------------------------------------

        [Test]
        public async Task BothCatalogsStoredAndNoStart_IsResumedByTheNextHandshake()
        {
            // The gap between the second SaveCatalog and the TryStart. Nothing else will ever start this
            // match: both clients have sent their barracks and no client sends one twice.
            string wire = BarracksWire.Write(BarracksCatalog.DefaultTemplates);
            await _store.SaveCatalogAsync(_matchId, Seat0Steam, wire, Ct);
            await _store.SaveCatalogAsync(_matchId, Seat1Steam, wire, Ct);

            DurableMatchCoordinator.AuthOutcome seated = await Auth("c0", _credential0);

            Assert.That(seated.Ok, Is.True, seated.FailCode);

            PersistedMatch row = (await _store.GetMatchAsync(_matchId, Ct))!;
            Assert.That(row.Status, Is.EqualTo(MatchStatus.Active), "the handshake finished the start");
            Assert.That(row.StartReplay, Is.Not.Null);

            Assert.That(_sink.MessagesFor("c0"),
                Is.EqualTo(new[] { "SEAT 0", NetProtocol.Start(row.StartReplay!) }),
                "and never CATALOG?, which this seat has already answered");
        }

        [Test]
        public async Task BothCatalogsStoredAndSomebodyElseStartedFirst_DealsTheirStartState()
        {
            string wire = BarracksWire.Write(BarracksCatalog.DefaultTemplates);
            await _store.SaveCatalogAsync(_matchId, Seat0Steam, wire, Ct);
            await _store.SaveCatalogAsync(_matchId, Seat1Steam, wire, Ct);

            // A start state that is not the one this host would build, so dealing the wrong one is visible.
            GameState theirs = GameFactory.Build(
                GameSetup.Parse((await _store.GetMatchAsync(_matchId, Ct))!.SetupWire),
                BarracksCatalog.DefaultTemplates,
                BarracksCatalog.DefaultTemplates);
            string theirReplay = ReplayFile.Write(theirs, Array.Empty<Command>());
            Assert.That(await _store.TryStartMatchAsync(_matchId, theirReplay, Begin, Ct), Is.True);

            await Auth("c0", _credential0);

            Assert.That(_sink.MessagesFor("c0"),
                Is.EqualTo(new[] { "SEAT 0", NetProtocol.Start(theirReplay) }),
                "the start state on the row is the game the clients are in");
        }

        // ---- an append that may or may not have landed ------------------------

        [Test]
        public async Task AnAppendThatCommittedAndThenThrew_IsAppliedOnceAndJournalledOnce()
        {
            await StartTheMatch();
            _sink.Clear();

            var opening = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));
            _faults.ThrowAfterNextAppend(new InvalidOperationException("the connection was reset"));

            await Cmd("c0", opening);

            string expected = NetProtocol.Apply(opening);
            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { expected }),
                "the row is there, so the issuer is told once and not asked to retry");
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { expected }));

            MatchJournal journal = (await _store.LoadJournalAsync(_matchId, Ct))!;
            Assert.That(journal.Commands, Has.Count.EqualTo(1));
            Assert.That(journal.Commands[0].Sequence, Is.EqualTo(1));
            Assert.That(Live().LastSequence, Is.EqualTo(1));

            // And a client that retried anyway is evaluated normally rather than waved through.
            _sink.Clear();
            await Cmd("c0", opening);

            Assert.That(_sink.MessagesFor("c0").Single(), Does.StartWith("REJECT "));
            Assert.That((await _store.LoadJournalAsync(_matchId, Ct))!.Commands, Has.Count.EqualTo(1));
        }

        [Test]
        public async Task AnAppendThatNeverLanded_IsStillATemporaryFailure()
        {
            await StartTheMatch();
            _sink.Clear();

            _faults.FailNextAppend(new InvalidOperationException("the database went away"));
            await Cmd("c0", new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0)));

            Assert.That(_sink.MessagesFor("c0"),
                Is.EqualTo(new[] { DurableMatchCoordinator.RejectTemporaryFailure }));
            Assert.That((await _store.LoadJournalAsync(_matchId, Ct))!.Commands, Is.Empty);
        }

        // ---- eviction against the handshake -----------------------------------

        [Test]
        public async Task ASweepBetweenTheLoadAndTheGate_DoesNotStrandTheSeat()
        {
            var swept = 0;
            _coordinator.BeforeGateForTest = async _ =>
            {
                if (swept++ > 0) return;

                // The projection is loaded, nobody is connected to it yet, and the sweeper decides it is
                // idle. Without the re-check after the gate this handshake would seat a player into a
                // projection no later frame can find.
                _clock.Advance(DurableMatchCoordinator.IdleEvictionWindow + TimeSpan.FromMinutes(1));
                await _coordinator.SweepAsync(_clock.GetUtcNow());
            };

            DurableMatchCoordinator.AuthOutcome seated = await Auth("c0", _credential0);

            Assert.That(seated.Ok, Is.True, seated.FailCode);
            Assert.That(swept, Is.GreaterThan(1), "the sweep has to have happened for this to prove anything");
            Assert.That(_sink.MessagesFor("c0"), Is.EqualTo(new[] { "SEAT 0", NetProtocol.CatalogRequest }));

            _coordinator.BeforeGateForTest = null;
            _sink.Clear();

            await _coordinator.ReceiveAsync(
                "c0", NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)), Ct);

            Assert.That(_sink.MessagesFor("c0"), Is.Empty, "the first frame after the race is accepted");
            Assert.That((await _store.GetPlayerAsync(_matchId, Seat0Steam, Ct))!.CatalogWire, Is.Not.Null);
        }

        // ---- one live socket per seat -----------------------------------------

        [Test]
        public async Task ASecondSocketForOneSeat_SupersedesTheFirst()
        {
            await SeatBothPlayers();
            _sink.Clear();

            DurableMatchCoordinator.AuthOutcome again = await Auth("c0-again", _credential0);

            Assert.That(again.Ok, Is.True);
            Assert.That(_sink.Closed, Is.EqualTo(new[]
            {
                ("c0", DurableMatchCoordinator.SupersededCloseStatus,
                    DurableMatchCoordinator.SupersededCloseReason),
            }));
            Assert.That(_coordinator.ConnectionsOf(_matchId), Is.EquivalentTo(new[] { "c1", "c0-again" }));

            _sink.Clear();
            await Cmd("c0", new EndTurn(PlayerId.Player0));
            Assert.That(_sink.MessagesFor("c0"),
                Is.EqualTo(new[] { DurableMatchCoordinator.RejectNoSeat }));
        }

        // ---- an ambiguous commit nobody can settle ----------------------------

        [Test]
        public async Task AnAmbiguousAppendThatCannotBeVerified_ClosesTheIssuerRatherThanInvitingARetry()
        {
            await StartTheMatch();
            _sink.Clear();

            // The row lands and the answer never comes back, and then the journal will not answer either.
            // TemporaryFailure would be a claim that the write did not happen, which is exactly what could
            // not be established - and an obedient client would send the command again.
            var opening = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));
            _faults.ThrowAfterNextAppend(new InvalidOperationException("the connection was reset"));
            _faults.FailNextJournalRead(new InvalidOperationException("the database is not answering"));

            await Cmd("c0", opening);

            Assert.That(_sink.Sent.Any(s => s.Message == DurableMatchCoordinator.RejectTemporaryFailure),
                Is.False, "never an invitation to resend a command that may already be durable");
            Assert.That(_sink.Closed, Is.EqualTo(new[]
            {
                ("c0", DurableMatchCoordinator.ResyncCloseStatus,
                    DurableMatchCoordinator.ResyncCloseReason),
            }));

            MatchJournal journal = (await _store.LoadJournalAsync(_matchId, Ct))!;
            Assert.That(journal.Commands, Has.Count.EqualTo(1), "the write had in fact landed");

            // And the reconnect settles it: START carries the command the client could not be told about.
            _sink.Clear();
            await Auth("c0-again", _credential0);

            Assert.That(_sink.MessagesFor("c0-again").Last(), Does.StartWith("START "));
            Assert.That(_sink.MessagesFor("c0-again").Last(),
                Does.Contain(CommandWire.Write(opening)));
        }

        [Test]
        public async Task AnAppendCancelledMidWriteThatCannotBeVerified_AlsoClosesTheIssuer()
        {
            await StartTheMatch();
            _sink.Clear();

            // A cancellation is not a refusal either: the write may have committed before the token was
            // observed, and there is no way to find out when the journal will not answer.
            _faults.FailNextAppend(new OperationCanceledException("the write was cancelled"));
            _faults.FailNextJournalRead(new InvalidOperationException("the database is not answering"));

            await Cmd("c0", new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0)));

            Assert.That(_sink.Sent.Any(s => s.Message == DurableMatchCoordinator.RejectTemporaryFailure),
                Is.False);
            Assert.That(_sink.Closed.Select(c => c.CloseStatus),
                Is.EqualTo(new[] { DurableMatchCoordinator.ResyncCloseStatus }));
        }

        [Test]
        public async Task AStatusThatCannotBeReReadAfterARefusedCompletion_IsNotTreatedAsCompleted()
        {
            (MatchRecord played, GameState _) = await PlayToTheBrinkAsync();

            // Somebody else ends the match, so TryComplete answers false - and then the row cannot be read
            // to find out what they ended it as. Unknown is not Completed: advancing to it would be
            // inventing an ending, and broadcasting under it would hand the clients a result that may not
            // be the recorded one.
            var abandonedOnce = false;
            _faults.BeforeCompletion = async () =>
            {
                if (abandonedOnce) return;
                abandonedOnce = true;
                await _store.TryCompleteMatchAsync(
                    _matchId, MatchStatus.Abandoned, null, _clock.GetUtcNow(), Ct);
            };

            _faults.FailNextGetMatch(new InvalidOperationException("the database is not answering"));
            _sink.Clear();

            await Cmd(played.Commands[^1]);

            Assert.That(_sink.Sent.Any(s => s.Message.StartsWith("APPLY ", StringComparison.Ordinal)),
                Is.False, "nothing is broadcast under a status this host could not read");
            Assert.That(Live().Status, Is.Not.EqualTo(MatchStatus.Completed));
            Assert.That(_sink.Closed.Select(c => c.ConnectionId), Is.EquivalentTo(new[] { "c0", "c1" }));
            Assert.That(_sink.Closed.Select(c => c.CloseStatus),
                Is.All.EqualTo(DurableMatchCoordinator.ResyncCloseStatus));
        }

        // ---- the window covers however a started match ended -------------------

        [Test]
        public async Task AMatchTheReaperAbandonedAfterItWasOver_StillDealsItsEndingInsideTheWindow()
        {
            (MatchRecord played, GameState start) = await PlayToTheBrinkAsync();

            _faults.FailEveryCompletion(new InvalidOperationException("the database went away"));
            await Cmd(played.Commands[^1]);
            _faults.FailEveryCompletion(null);

            // The reaper gets to the row first and calls it abandoned. The game is still over, the journal
            // still says how, and the seats still deserve to be shown it.
            Assert.That(
                await _store.TryCompleteMatchAsync(
                    _matchId, MatchStatus.Abandoned, null, _clock.GetUtcNow(), Ct),
                Is.True);

            _clock.Advance(TimeSpan.FromSeconds(30));
            _sink.Clear();

            DurableMatchCoordinator.AuthOutcome back = await Auth("c2", _credential1);

            Assert.That(back.Ok, Is.True, back.FailCode);
            Assert.That(_sink.MessagesFor("c2"), Is.EqualTo(new[]
            {
                "SEAT 1",
                NetProtocol.Start(ReplayFile.Write(start, played.Commands)),
            }));
        }

        // ---- a rebuild has to tell the seat that did not reconnect -------------

        [Test]
        public async Task ARebuildThatFindsTheGameOver_TellsTheOpponentTooNotJustTheReconnectingSeat()
        {
            (MatchRecord played, GameState start) = await PlayToTheBrinkAsync();

            // The completion COMMITS and its answer is lost, twice. The row is terminal, this host has no
            // way of knowing, and the issuer is disconnected. What must not happen next is the opponent -
            // still connected, still looking at the position before the winning move - being left there
            // forever because the reconnecting seat was the only one anybody thought to tell.
            _faults.ThrowAfterNextComplete(new InvalidOperationException("the response was lost"), times: 2);
            _sink.Clear();
            await Cmd(played.Commands[^1]);

            Assert.That(_sink.Closed.Select(c => c.ConnectionId), Is.EqualTo(new[] { "c0" }));
            Assert.That((await _store.GetMatchAsync(_matchId, Ct))!.Status, Is.EqualTo(MatchStatus.Completed),
                "the completion did land; only the answer was lost");
            Assert.That(Live().Stale, Is.True);

            // c1 never went away. c0 comes back on a new socket.
            _sink.Clear();
            DurableMatchCoordinator.AuthOutcome back = await Auth("c0-again", _credential0);
            Assert.That(back.Ok, Is.True, back.FailCode);

            string terminal = NetProtocol.Start(ReplayFile.Write(start, played.Commands));
            Assert.That(_sink.MessagesFor("c1"), Is.EqualTo(new[] { terminal }),
                "the seat that stayed connected is dealt the ending it never heard about");
            Assert.That(_sink.MessagesFor("c0-again"), Is.EqualTo(new[] { "SEAT 0", terminal }));

            Assert.That(ReplayFile.Read(terminal["START ".Length..]).Start, Is.Not.Null);
            Assert.That(Live().State!.IsGameOver, Is.True);
            Assert.That(Live().Status, Is.EqualTo(MatchStatus.Completed));
        }

        [Test]
        public async Task ACancelledStatusReRead_LeavesTheProjectionStaleAndTheEndingRecoverable()
        {
            (MatchRecord played, GameState start) = await PlayToTheBrinkAsync();

            // Somebody else ends the match, so TryComplete answers false, and the read that would say what
            // they ended it as is cancelled. A cancellation here is no different from any other failure
            // after a durable append: the journal has moved and this projection has not.
            var abandonedOnce = false;
            _faults.BeforeCompletion = async () =>
            {
                if (abandonedOnce) return;
                abandonedOnce = true;
                await _store.TryCompleteMatchAsync(
                    _matchId, MatchStatus.Abandoned, null, _clock.GetUtcNow(), Ct);
            };

            _faults.FailNextGetMatch(new OperationCanceledException("the request went away"));
            _sink.Clear();

            await Cmd(played.Commands[^1]);

            Assert.That(Live().Stale, Is.True, "a cancellation must not leave a pre-final projection usable");
            Assert.That(_sink.Sent.Any(s => s.Message.StartsWith("APPLY ", StringComparison.Ordinal)),
                Is.False);
            Assert.That(_sink.Closed.Select(c => c.CloseStatus),
                Is.All.EqualTo(DurableMatchCoordinator.ResyncCloseStatus));

            // And the command really was durable, so a reconnect is dealt it.
            _sink.Clear();
            await Auth("c2", _credential1);

            Assert.That(_sink.MessagesFor("c2").Last(),
                Is.EqualTo(NetProtocol.Start(ReplayFile.Write(start, played.Commands))));
        }
    }
}

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
    /// What this process is willing to host after a restart, and what it refuses.
    ///
    /// Recovery is the one place where a journal written by some other build, or by a bug, meets an engine
    /// that has to reproduce it exactly. Every refusal below is a different way of the record being
    /// unplayable, and they are kept apart on purpose: an operator reading the startup report needs to know
    /// whether a match needs a rollback, a schema fix or a different build, and "could not load" tells them
    /// none of those things.
    ///
    /// The refusals are also why the coordinator caches them. Replaying a long journal to discover, again,
    /// that it does not replay is the most expensive way this server can say no, and a client that
    /// reconnects with backoff will ask for it over and over.
    /// </summary>
    [TestFixture]
    public class MatchRecoveryServiceTests
    {
        const string Seat0Steam = "76561190000000001";
        const string Seat1Steam = "76561190000000002";
        const string LobbyId = "109775240000000042";

        static readonly DateTimeOffset Begin = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);
        static readonly Guid MatchId = Guid.Parse("2f6b9f0e-2c1f-4c5a-9d1e-1a2b3c4d5e6f");

        static readonly Command FirstMove = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));
        static readonly Command ThenAttack = new AttackUnit(PlayerId.Player0, 2, 5);

        static CancellationToken Ct => CancellationToken.None;

        JournalStore _store = null!;
        MatchRecoveryService _recovery = null!;

        [SetUp]
        public void AServiceOverAJournalWeControlByHand()
        {
            _store = new JournalStore();
            _recovery = new MatchRecoveryService(
                _store,
                Options.Create(new MatchHostingOptions()),
                NullLogger<MatchRecoveryService>.Instance);
        }

        // ---- fixtures --------------------------------------------------------

        static GameState FreshStart() =>
            GameFactory.Build(GameSetup.Default, BarracksCatalog.DefaultTemplates, BarracksCatalog.DefaultTemplates);

        static string FreshStartReplay() => ReplayFile.Write(FreshStart(), Array.Empty<Command>());

        static PersistedMatch Row(
            Guid matchId,
            MatchStatus status,
            string? startReplay,
            string engineVersion = "hexwars-engine/1",
            int protocolVersion = 2) => new(
            matchId, LobbyId, status, GameSetup.Default.ToWire(), startReplay, engineVersion, protocolVersion,
            "test-build", Begin, startReplay is null ? null : Begin, null, Begin, null);

        static PersistedPlayer Player(Guid matchId, int seat) =>
            new(matchId, seat == 0 ? Seat0Steam : Seat1Steam, seat, null, Begin, null);

        static PersistedCommand Stored(int sequence, Command command, string? issuerSteamId = null) => new(
            MatchId, sequence, CommandWire.Write(command), Begin,
            issuerSteamId ?? (command.Issuer == PlayerId.Player0 ? Seat0Steam : Seat1Steam));

        static PersistedCommand Raw(int sequence, string commandWire, string issuerSteamId) =>
            new(MatchId, sequence, commandWire, Begin, issuerSteamId);

        static MatchJournal Active(string? startReplay, params PersistedCommand[] commands) => new(
            Row(MatchId, MatchStatus.Active, startReplay),
            new[] { Player(MatchId, 0), Player(MatchId, 1) },
            commands);

        MatchJournal Seed(MatchJournal journal)
        {
            _store.Journals[journal.Match.MatchId] = journal;
            return journal;
        }

        MatchRecoveryException Refusal() =>
            Assert.ThrowsAsync<MatchRecoveryException>(() => _recovery.LoadAsync(MatchId, Ct))!;

        // ---- the happy paths -------------------------------------------------

        [Test]
        public async Task AnActiveJournal_ReplaysToExactlyWhatTheEngineProducesDirectly()
        {
            Seed(Active(FreshStartReplay(),
                Stored(1, FirstMove),
                Stored(2, ThenAttack)));

            LiveMatch live = await _recovery.LoadAsync(MatchId, Ct);

            GameState direct = GameEngine.Apply(FreshStart(), FirstMove).NewState;
            direct = GameEngine.Apply(direct, ThenAttack).NewState;

            Assert.That(live.Status, Is.EqualTo(MatchStatus.Active));
            Assert.That(live.LastSequence, Is.EqualTo(2));
            Assert.That(ReplayFile.Write(live.State!, Array.Empty<Command>()),
                Is.EqualTo(ReplayFile.Write(direct, Array.Empty<Command>())));
        }

        [Test]
        public async Task AWaitingJournal_LoadsWithNoGameAndNoComplaint()
        {
            Seed(new MatchJournal(
                Row(MatchId, MatchStatus.Waiting, null),
                new[] { Player(MatchId, 0), Player(MatchId, 1) },
                Array.Empty<PersistedCommand>()));

            LiveMatch live = await _recovery.LoadAsync(MatchId, Ct);

            Assert.That(live.Status, Is.EqualTo(MatchStatus.Waiting));
            Assert.That(live.Start, Is.Null);
            Assert.That(live.Seats, Has.Count.EqualTo(2));
        }

        [Test]
        public async Task AFinishedMatch_StillLoads_BecauseRefusingTheSeatIsTheCoordinatorsJob()
        {
            Seed(new MatchJournal(
                Row(MatchId, MatchStatus.Completed, FreshStartReplay()),
                new[] { Player(MatchId, 0), Player(MatchId, 1) },
                Array.Empty<PersistedCommand>()));

            LiveMatch live = await _recovery.LoadAsync(MatchId, Ct);

            Assert.That(live.Status, Is.EqualTo(MatchStatus.Completed));
        }

        [Test]
        public async Task ACompletedJournal_ReplaysItsWholeLogRatherThanStoppingAtTheStart()
        {
            Seed(new MatchJournal(
                Row(MatchId, MatchStatus.Completed, FreshStartReplay()),
                new[] { Player(MatchId, 0), Player(MatchId, 1) },
                new[] { Stored(1, FirstMove), Stored(2, ThenAttack) }));

            LiveMatch live = await _recovery.LoadAsync(MatchId, Ct);

            GameState direct = GameEngine.Apply(FreshStart(), FirstMove).NewState;
            direct = GameEngine.Apply(direct, ThenAttack).NewState;

            Assert.That(live.LastSequence, Is.EqualTo(2));
            Assert.That(live.Log, Has.Count.EqualTo(2));
            Assert.That(ReplayFile.Write(live.State!, Array.Empty<Command>()),
                Is.EqualTo(ReplayFile.Write(direct, Array.Empty<Command>())),
                "a terminal match is dealt to reconnecting clients, so it has to be the position it ended in");
        }

        [Test]
        public void ACompletedJournalThatWillNotReplay_IsClassifiedRatherThanAccepted()
        {
            Seed(new MatchJournal(
                Row(MatchId, MatchStatus.Completed, FreshStartReplay()),
                new[] { Player(MatchId, 0), Player(MatchId, 1) },
                new[] { Stored(1, FirstMove), Stored(2, FirstMove) }));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.CommandReplayFailed));
            Assert.That(refusal.Detail, Does.Contain("2"));
        }

        [Test]
        public void ACompletedJournalWithASequenceGap_IsRefused()
        {
            Seed(new MatchJournal(
                Row(MatchId, MatchStatus.Completed, FreshStartReplay()),
                new[] { Player(MatchId, 0), Player(MatchId, 1) },
                new[] { Stored(1, FirstMove), Stored(3, ThenAttack) }));

            Assert.That(Refusal().Failure, Is.EqualTo(MatchRecoveryFailure.SequenceGap));
        }

        // ---- the refusals, one per way a record can be unplayable ------------

        [Test]
        public void AMatchThatIsNotThere_IsNotFound()
        {
            Assert.That(Refusal().Failure, Is.EqualTo(MatchRecoveryFailure.NotFound));
        }

        [Test]
        public void AJournalWrittenUnderAnEngineContractThisBuildCannotReplay_IsRefused()
        {
            Seed(new MatchJournal(
                Row(MatchId, MatchStatus.Active, FreshStartReplay(), engineVersion: "hexwars-engine/0"),
                new[] { Player(MatchId, 0), Player(MatchId, 1) },
                Array.Empty<PersistedCommand>()));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.UnsupportedEngineContract));
            Assert.That(refusal.Detail, Does.Contain("hexwars-engine/0"), "the operator needs the stored version");
            Assert.That(refusal.Detail, Does.Contain(EngineContract.Version), "and the one this build speaks");
        }

        [Test]
        public void AJournalFromAnOlderProtocol_IsRefused()
        {
            Seed(new MatchJournal(
                Row(MatchId, MatchStatus.Active, FreshStartReplay(), protocolVersion: 1),
                new[] { Player(MatchId, 0), Player(MatchId, 1) },
                Array.Empty<PersistedCommand>()));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.UnsupportedProtocol));
            Assert.That(refusal.Detail, Does.Contain("1"));
        }

        [Test]
        public void AJournalFromAProtocolThisBuildDoesNotSpeak_IsRefused()
        {
            Seed(new MatchJournal(
                Row(MatchId, MatchStatus.Active, FreshStartReplay(), protocolVersion: 3),
                new[] { Player(MatchId, 0), Player(MatchId, 1) },
                Array.Empty<PersistedCommand>()));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.UnsupportedProtocol));
            Assert.That(refusal.Detail, Does.Contain("3"), "the operator needs the stored version");
            Assert.That(refusal.Detail, Does.Contain(ProtocolContract.SupportedList),
                "and the set this build speaks");
        }

        [Test]
        public void AHostConfiguredForAProtocolThisBuildDoesNotSpeak_RefusesEveryMatch()
        {
            var wrong = new MatchHostingOptions { ProtocolVersion = 3 };
            _recovery = new MatchRecoveryService(
                _store, Options.Create(wrong), NullLogger<MatchRecoveryService>.Instance);

            Seed(Active(FreshStartReplay()));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.UnsupportedProtocol));
            Assert.That(refusal.Detail, Does.Contain(HexWarsConfiguration.MatchProtocolVersionKey));
        }

        [Test]
        public void AStartReplayThatIsNotAReplay_IsACorruptStartState()
        {
            Seed(Active("this is not a HexWars replay"));

            Assert.That(Refusal().Failure, Is.EqualTo(MatchRecoveryFailure.CorruptStartState));
        }

        [Test]
        public void AnActiveMatchWithNoStartReplayAtAll_IsACorruptStartState()
        {
            Seed(Active(null));

            Assert.That(Refusal().Failure, Is.EqualTo(MatchRecoveryFailure.CorruptStartState));
        }

        [Test]
        public void AMissingSequence_IsAGapNamedByItsFirstHole()
        {
            Seed(Active(FreshStartReplay(),
                Stored(1, FirstMove),
                Stored(2, ThenAttack),
                Stored(4, new EndTurn(PlayerId.Player0))));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.SequenceGap));
            Assert.That(refusal.Detail, Does.Contain("3").And.Contain("4"),
                "the detail should name the sequence that is missing and the one found instead");
        }

        [Test]
        public void ASequenceThatDoesNotStartAtOne_IsAGap()
        {
            Seed(Active(FreshStartReplay(), Stored(2, FirstMove)));

            Assert.That(Refusal().Failure, Is.EqualTo(MatchRecoveryFailure.SequenceGap));
        }

        [Test]
        public void ACommandWireThatDoesNotParse_IsACorruptCommand()
        {
            Seed(Active(FreshStartReplay(),
                Stored(1, FirstMove),
                Raw(2, "ZZ not a command", Seat0Steam)));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.CorruptCommand));
            Assert.That(refusal.Detail, Does.Contain("2"));
        }

        [Test]
        public void ACommandStoredAgainstAPlayerWhoDoesNotHoldThatSeat_IsAnIssuerMismatch()
        {
            // The command says Player0 played it; the row says the player in seat 1 wrote it. One of the two
            // is a lie, and a replay that trusted either would deal a game nobody played.
            Seed(Active(FreshStartReplay(), Stored(1, FirstMove, issuerSteamId: Seat1Steam)));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.IssuerSeatMismatch));
            Assert.That(refusal.Detail, Does.Contain("1"));
        }

        [Test]
        public void ACommandFromSomebodyWithNoSeatInThisMatch_IsAnIssuerMismatch()
        {
            Seed(Active(FreshStartReplay(), Stored(1, FirstMove, issuerSteamId: "76561190000000009")));

            Assert.That(Refusal().Failure, Is.EqualTo(MatchRecoveryFailure.IssuerSeatMismatch));
        }

        [Test]
        public void ACommandTheEngineRefuses_IsAFailedReplay()
        {
            Seed(Active(FreshStartReplay(),
                Stored(1, FirstMove),
                Stored(2, new EndTurn(PlayerId.Player1))));

            MatchRecoveryException refusal = Refusal();

            Assert.That(refusal.Failure, Is.EqualTo(MatchRecoveryFailure.CommandReplayFailed));
            Assert.That(refusal.Detail, Does.Contain("2"));
        }

        // ---- the startup pass ------------------------------------------------

        [Test]
        public async Task TheStartupPass_CountsWhatItVerifiedAndNamesWhatItCannotHost()
        {
            Guid good = Guid.Parse("11111111-1111-1111-1111-111111111111");
            Guid waiting = Guid.Parse("22222222-2222-2222-2222-222222222222");
            Guid broken = Guid.Parse("33333333-3333-3333-3333-333333333333");
            Guid finished = Guid.Parse("44444444-4444-4444-4444-444444444444");

            _store.Journals[good] = new MatchJournal(
                Row(good, MatchStatus.Active, FreshStartReplay()),
                new[] { Player(good, 0), Player(good, 1) },
                new[] { new PersistedCommand(good, 1, CommandWire.Write(FirstMove), Begin, Seat0Steam) });

            _store.Journals[waiting] = new MatchJournal(
                Row(waiting, MatchStatus.Waiting, null),
                new[] { Player(waiting, 0), Player(waiting, 1) },
                Array.Empty<PersistedCommand>());

            _store.Journals[broken] = new MatchJournal(
                Row(broken, MatchStatus.Active, FreshStartReplay()),
                new[] { Player(broken, 0), Player(broken, 1) },
                new[] { new PersistedCommand(broken, 2, CommandWire.Write(FirstMove), Begin, Seat0Steam) });

            _store.Journals[finished] = new MatchJournal(
                Row(finished, MatchStatus.Completed, FreshStartReplay()),
                new[] { Player(finished, 0), Player(finished, 1) },
                Array.Empty<PersistedCommand>());

            RecoveryReport report = await _recovery.VerifyOpenMatchesAsync(Ct);

            Assert.That(report.Verified, Is.EqualTo(2), "the finished match is not open, so it is not verified");
            Assert.That(report.Failed, Has.Count.EqualTo(1));
            Assert.That(report.Failed[0].MatchId, Is.EqualTo(broken));
            Assert.That(report.Failed[0].Failure, Is.EqualTo(MatchRecoveryFailure.SequenceGap));
            Assert.That(report.Failed[0].Detail, Is.Not.Empty);
        }

        [Test]
        public void TheStartupPass_LetsAStoreFailureThrough_BecauseADeadDatabaseIsNotABadJournal()
        {
            _store.Failure = new InvalidOperationException("the database is not there");

            Assert.ThrowsAsync<InvalidOperationException>(() => _recovery.VerifyOpenMatchesAsync(Ct));
        }

        [Test]
        public async Task TheStartupService_RecordsTheReportItGot()
        {
            Guid good = Guid.Parse("11111111-1111-1111-1111-111111111111");
            _store.Journals[good] = new MatchJournal(
                Row(good, MatchStatus.Waiting, null),
                new[] { Player(good, 0), Player(good, 1) },
                Array.Empty<PersistedCommand>());

            var state = new RecoveryState();
            var startup = new RecoveryStartupService(
                state, _recovery, NullLogger<RecoveryStartupService>.Instance);

            await startup.StartAsync(Ct);

            Assert.That(state.Completed, Is.True);
            Assert.That(state.Error, Is.Null);
            Assert.That(state.Report!.Verified, Is.EqualTo(1));
        }

        [Test]
        public async Task TheStartupService_RecordsAStoreFailureAndStillCompletes()
        {
            // A database that is down at boot must not crash-loop the process: readiness reports it, so the
            // platform restart policy stops being the thing that decides whether this host ever comes back.
            _store.Failure = new InvalidOperationException("the database is not there");

            var state = new RecoveryState();
            var startup = new RecoveryStartupService(
                state, _recovery, NullLogger<RecoveryStartupService>.Instance);

            await startup.StartAsync(Ct);

            Assert.That(state.Completed, Is.True);
            Assert.That(state.Report, Is.Null);
            Assert.That(state.Error, Is.InstanceOf<InvalidOperationException>());
        }

        [Test]
        public async Task TheStartupService_CompletesImmediatelyWhenThereIsNoDatabaseAtAll()
        {
            var state = new RecoveryState();
            var startup = new RecoveryStartupService(
                state, null, NullLogger<RecoveryStartupService>.Instance);

            await startup.StartAsync(Ct);

            Assert.That(state.Completed, Is.True);
            Assert.That(state.Error, Is.Null);
            Assert.That(state.Report!.Verified, Is.EqualTo(0));
            Assert.That(state.Report!.Failed, Is.Empty);
        }

        // ---- what the coordinator does with a refusal ------------------------

        [Test]
        public async Task AnUnrecoverableMatch_TurnsEveryPlayerAwayAsUnavailableWithoutASeat()
        {
            Fixture seam = await Fixture.WithAnUnrecoverableJournal();

            DurableMatchCoordinator.AuthOutcome outcome = await seam.Auth("c0");

            Assert.That(outcome.Ok, Is.False);
            Assert.That(outcome.Seat, Is.EqualTo(-1));
            Assert.That(outcome.FailCode, Is.EqualTo(DurableMatchCoordinator.AuthFailUnavailable));
            Assert.That(seam.Sink.MessagesFor("c0"), Is.Empty, "nobody is seated in a match we cannot replay");
        }

        [Test]
        public async Task AReconnectStormAgainstAnUnrecoverableMatch_ReplaysTheJournalOnlyOnce()
        {
            // Refusing is the most expensive no this server can say - it costs a full replay - and a client
            // that reconnects with backoff will ask for it again and again.
            Fixture seam = await Fixture.WithAnUnrecoverableJournal();

            await seam.Auth("c0");
            await seam.Auth("c1");
            DurableMatchCoordinator.AuthOutcome third = await seam.Auth("c2");

            Assert.That(seam.Loader.Calls, Is.EqualTo(1));
            Assert.That(third.FailCode, Is.EqualTo(DurableMatchCoordinator.AuthFailUnavailable),
                "the cached answer is still a refusal, not a seat");
        }

        [Test]
        public async Task OnceTheCachedRefusalExpires_TheJournalIsTriedAgain()
        {
            Fixture seam = await Fixture.WithAnUnrecoverableJournal();

            await seam.Auth("c0");
            seam.Clock.Advance(TimeSpan.FromSeconds(61));
            await seam.Auth("c1");

            Assert.That(seam.Loader.Calls, Is.EqualTo(2), "a repaired match must be able to come back");
        }

        // ---- doubles ---------------------------------------------------------

        /// <summary>
        /// A coordinator whose loader refuses this match, over a store that is otherwise perfectly healthy.
        ///
        /// The two are deliberately separate: the credential the player presents has to validate against a
        /// real store, or the handshake would fail before it ever reached the projection, and the test would
        /// be watching the wrong refusal.
        /// </summary>
        sealed class Fixture
        {
            public RecordingConnectionSink Sink { get; private init; } = null!;
            public FakeTimeProvider Clock { get; private init; } = null!;
            public CountingLoader Loader { get; private init; } = null!;
            public DurableMatchCoordinator Coordinator { get; private init; } = null!;
            public Guid Match { get; private init; }
            public string Credential { get; private init; } = null!;

            public Task<DurableMatchCoordinator.AuthOutcome> Auth(string connectionId) =>
                Coordinator.AuthenticateAsync(connectionId, Match.ToString(), Credential, Ct);

            public static async Task<Fixture> WithAnUnrecoverableJournal()
            {
                var store = new InMemoryMatchStore();
                var clock = new FakeTimeProvider(Begin);
                var sink = new RecordingConnectionSink();
                var credentials = new MatchCredentialService(
                    store,
                    Options.Create(new MatchHostingOptions()),
                    clock,
                    NullLogger<MatchCredentialService>.Instance);

                CreateMatchResult created = await store.CreateMatchForLobbyAsync(new CreateMatchRequest(
                    LobbyId, GameSetup.Default.ToWire(), EngineContract.Version, 2, "test-build",
                    new[] { (Seat0Steam, 0), (Seat1Steam, 1) }, Begin), Ct);

                Guid matchId = created.Match.MatchId;

                var journals = new JournalStore();
                journals.Journals[matchId] = new MatchJournal(
                    Row(matchId, MatchStatus.Waiting, null, engineVersion: "hexwars-engine/0"),
                    new[] { Player(matchId, 0), Player(matchId, 1) },
                    Array.Empty<PersistedCommand>());

                var loader = new CountingLoader(new MatchRecoveryService(
                    journals,
                    Options.Create(new MatchHostingOptions()),
                    NullLogger<MatchRecoveryService>.Instance));

                return new Fixture
                {
                    Sink = sink,
                    Clock = clock,
                    Loader = loader,
                    Match = matchId,
                    Credential = (await credentials.IssueAsync(matchId, Seat0Steam, Ct)).Credential,
                    Coordinator = new DurableMatchCoordinator(
                        store,
                        credentials,
                        loader,
                        sink,
                        Options.Create(new MatchHostingOptions()),
                        clock,
                        NullLogger<DurableMatchCoordinator>.Instance),
                };
            }
        }

        /// <summary>Counts how often the coordinator actually went looking for a projection.</summary>
        sealed class CountingLoader(ILiveMatchLoader inner) : ILiveMatchLoader
        {
            public int Calls { get; private set; }

            public Task<LiveMatch> LoadAsync(Guid matchId, CancellationToken ct)
            {
                Calls++;
                return inner.LoadAsync(matchId, ct);
            }
        }

        /// <summary>
        /// A store that serves journals written by hand.
        ///
        /// The in-memory store cannot produce the records this fixture needs: every one of them is a journal
        /// the real write path refuses to create, which is exactly why recovery has to have an opinion about
        /// them. Only the two reads recovery makes are implemented; anything else reaching this double is a
        /// test asking the wrong question.
        /// </summary>
        sealed class JournalStore : IMatchStore
        {
            public Dictionary<Guid, MatchJournal> Journals { get; } = new();

            /// <summary>When set, every read throws it - the database being down rather than a bad record.</summary>
            public Exception? Failure { get; set; }

            public Task<MatchJournal?> LoadJournalAsync(Guid matchId, CancellationToken ct)
            {
                if (Failure is not null) throw Failure;
                return Task.FromResult(Journals.TryGetValue(matchId, out MatchJournal? journal) ? journal : null);
            }

            public Task<IReadOnlyList<Guid>> ListOpenMatchIdsAsync(CancellationToken ct)
            {
                if (Failure is not null) throw Failure;

                IReadOnlyList<Guid> open = Journals
                    .Where(entry => entry.Value.Match.Status is MatchStatus.Waiting or MatchStatus.Active)
                    .Select(entry => entry.Key)
                    .OrderBy(id => id)
                    .ToArray();

                return Task.FromResult(open);
            }

            public Task<CreateMatchResult> CreateMatchForLobbyAsync(CreateMatchRequest request, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<PersistedMatch?> GetMatchAsync(Guid matchId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<PersistedMatch?> FindOpenMatchForLobbyAsync(string steamLobbyId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<IReadOnlyList<PersistedPlayer>> GetPlayersAsync(Guid matchId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<PersistedPlayer?> GetPlayerAsync(Guid matchId, string steamId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task SaveCatalogAsync(Guid matchId, string steamId, string catalogWire, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<bool> TryStartMatchAsync(
                Guid matchId, string startReplay, DateTimeOffset startedAt, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<AppendResult> AppendCommandAsync(
                Guid matchId, int expectedSequence, string commandWire, string issuerSteamId,
                DateTimeOffset acceptedAt, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<bool> TryCompleteMatchAsync(
                Guid matchId, MatchStatus terminal, int? winnerSeat, DateTimeOffset completedAt,
                CancellationToken ct) =>
                throw new NotSupportedException();

            public Task TouchAsync(Guid matchId, string? steamId, DateTimeOffset seenAt, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task StoreJoinCredentialAsync(
                byte[] credentialHash, Guid matchId, string steamId, DateTimeOffset expiresAt,
                CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<JoinCredentialRecord?> FindJoinCredentialAsync(byte[] credentialHash, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task RevokeJoinCredentialsAsync(
                Guid matchId, string steamId, DateTimeOffset revokedAt, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<bool> ReplaceJoinCredentialAsync(
                byte[] credentialHash, Guid matchId, string steamId, DateTimeOffset expiresAt,
                DateTimeOffset now, CancellationToken ct) =>
                throw new NotSupportedException();
        }
    }
}

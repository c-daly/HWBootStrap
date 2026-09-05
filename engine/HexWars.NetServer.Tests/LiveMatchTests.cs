using HexWars.Engine;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Tests.Fakes;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// Rebuilding a match from the journal, which is the only way a restarted process learns what it is
    /// hosting. The projection is allowed to be wrong about nothing: a journal with a gap in it, or one
    /// holding a command the engine refuses, means somebody has written to this match in a way that cannot
    /// have happened, and continuing would deal a state no client can agree with.
    /// </summary>
    [TestFixture]
    public class LiveMatchTests
    {
        const string Seat0Steam = "76561190000000001";
        const string Seat1Steam = "76561190000000002";

        static readonly DateTimeOffset Begin = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);
        static readonly Guid MatchId = Guid.Parse("2f6b9f0e-2c1f-4c5a-9d1e-1a2b3c4d5e6f");

        static readonly Command FirstMove = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));
        static readonly Command ThenAttack = new AttackUnit(PlayerId.Player0, 2, 5);

        static GameState FreshStart() =>
            GameFactory.Build(GameSetup.Default, BarracksCatalog.DefaultTemplates, BarracksCatalog.DefaultTemplates);

        static PersistedMatch Row(MatchStatus status, string? startReplay) => new(
            MatchId, "109775240000000042", status, GameSetup.Default.ToWire(), startReplay,
            "hexwars-engine/1", 2, "test-build", Begin, startReplay is null ? null : Begin,
            null, Begin, null);

        static PersistedPlayer Player(int seat, string? catalogWire) => new(
            MatchId, seat == 0 ? Seat0Steam : Seat1Steam, seat, catalogWire, Begin, null);

        static PersistedCommand Stored(int sequence, Command command, PlayerId issuer) => new(
            MatchId, sequence, CommandWire.Write(command), Begin,
            issuer == PlayerId.Player0 ? Seat0Steam : Seat1Steam);

        [Test]
        public void AWaitingJournal_CarriesSeatsAndCatalogsAndNoGame()
        {
            string wire = BarracksWire.Write(BarracksCatalog.DefaultTemplates);
            var journal = new MatchJournal(
                Row(MatchStatus.Waiting, null),
                new[] { Player(0, wire), Player(1, null) },
                Array.Empty<PersistedCommand>());

            LiveMatch live = LiveMatch.FromJournal(journal);

            Assert.That(live.MatchId, Is.EqualTo(MatchId));
            Assert.That(live.Status, Is.EqualTo(MatchStatus.Waiting));
            Assert.That(live.Seats, Is.EqualTo(new Dictionary<string, int>
            {
                [Seat0Steam] = 0,
                [Seat1Steam] = 1,
            }));
            Assert.That(live.Catalogs.Keys, Is.EqualTo(new[] { 0 }));
            Assert.That(live.Start, Is.Null);
            Assert.That(live.State, Is.Null);
            Assert.That(live.LastSequence, Is.EqualTo(0));
            Assert.That(live.ProtocolVersion, Is.EqualTo(2));
            Assert.That(live.EngineVersion, Is.EqualTo("hexwars-engine/1"));
        }

        [Test]
        public void ACatalogThatNoLongerParses_FallsBackToTheDefaultBarracks()
        {
            var journal = new MatchJournal(
                Row(MatchStatus.Waiting, null),
                new[] { Player(0, "V9 this is not a barracks"), Player(1, null) },
                Array.Empty<PersistedCommand>());

            LiveMatch live = LiveMatch.FromJournal(journal);

            Assert.That(live.Catalogs[0], Is.EqualTo(BarracksCatalog.Normalize(BarracksCatalog.DefaultTemplates)));
        }

        [Test]
        public void AnActiveJournal_ReplaysEveryCommandIntoTheCurrentState()
        {
            string startReplay = ReplayFile.Write(FreshStart(), Array.Empty<Command>());
            var journal = new MatchJournal(
                Row(MatchStatus.Active, startReplay),
                new[] { Player(0, null), Player(1, null) },
                new[]
                {
                    Stored(1, FirstMove, PlayerId.Player0),
                    Stored(2, ThenAttack, PlayerId.Player0),
                });

            LiveMatch live = LiveMatch.FromJournal(journal);

            Assert.That(live.LastSequence, Is.EqualTo(2));
            Assert.That(live.Log.Select(CommandWire.Write),
                Is.EqualTo(new[] { CommandWire.Write(FirstMove), CommandWire.Write(ThenAttack) }));

            GameState direct = GameEngine.Apply(FreshStart(), FirstMove).NewState;
            direct = GameEngine.Apply(direct, ThenAttack).NewState;

            Assert.That(ReplayFile.Write(live.State!, Array.Empty<Command>()),
                Is.EqualTo(ReplayFile.Write(direct, Array.Empty<Command>())));

            // The re-deal a reconnecting client gets is the start plus the log, never the live state: the
            // start-state encoding cannot represent a mid-game position.
            Assert.That(live.StartReplayText(), Is.EqualTo(ReplayFile.Write(live.Start!, live.Log)));
            Assert.That(ReplayFile.Read(live.StartReplayText()).Commands, Has.Count.EqualTo(2));
        }

        [Test]
        public void AJournalWithAMissingSequence_IsRefused()
        {
            string startReplay = ReplayFile.Write(FreshStart(), Array.Empty<Command>());
            var journal = new MatchJournal(
                Row(MatchStatus.Active, startReplay),
                new[] { Player(0, null), Player(1, null) },
                new[]
                {
                    Stored(1, FirstMove, PlayerId.Player0),
                    Stored(3, ThenAttack, PlayerId.Player0),
                });

            var failure = Assert.Throws<InvalidOperationException>(() => LiveMatch.FromJournal(journal));
            Assert.That(failure!.Message, Does.Contain("sequence gap"));
        }

        [Test]
        public void AJournalledCommandTheEngineRefuses_IsRefused()
        {
            string startReplay = ReplayFile.Write(FreshStart(), Array.Empty<Command>());
            var journal = new MatchJournal(
                Row(MatchStatus.Active, startReplay),
                new[] { Player(0, null), Player(1, null) },
                new[]
                {
                    Stored(1, FirstMove, PlayerId.Player0),
                    Stored(2, new EndTurn(PlayerId.Player1), PlayerId.Player1),
                });

            var failure = Assert.Throws<InvalidOperationException>(() => LiveMatch.FromJournal(journal));
            Assert.That(failure!.Message, Does.Contain("replay failed at sequence 2"));
        }

        [Test]
        public void TheLoaderRefusesAMatchThatDoesNotExist()
        {
            var loader = new JournalLiveMatchLoader(new InMemoryMatchStore());

            Assert.ThrowsAsync<KeyNotFoundException>(
                () => loader.LoadAsync(MatchId, CancellationToken.None));
        }
    }
}

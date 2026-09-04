using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.Extensions.Logging.Abstractions;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The parts of the store contract that only mean anything when two connections are talking to the same
    /// database at once: a journal read that must not tear, a catalogue save that must not overtake a start,
    /// and a lobby allocation that must survive the match it collided with closing underneath it.
    ///
    /// These live apart from <see cref="MatchStoreContractTests"/> deliberately. They are not statements
    /// about the interface, they are statements about the isolation levels and row locks the Postgres
    /// implementation uses, and the in-memory double has nothing to say about either.
    /// </summary>
    [TestFixture]
    public sealed class PostgresMatchStoreConcurrencyTests
    {
        const string SetupWire = "annihilation 9 7 0 7 3 1 1 1 3 0";
        const string EngineVersion = "hexwars-engine/1";
        const string BuildId = "test-build";
        const int ProtocolVersion = 2;

        static readonly DateTimeOffset Created = new(2026, 9, 4, 12, 0, 0, TimeSpan.Zero);
        static readonly DateTimeOffset Started = Created.AddMinutes(1);
        static readonly DateTimeOffset Move1 = Created.AddMinutes(2);

        /// <summary>Long enough that a blocked call is really blocked rather than merely slow, short enough
        /// that the suite does not crawl.</summary>
        static readonly TimeSpan BlockedFor = TimeSpan.FromSeconds(1);

        static int _identifiers;

        PostgresTestDatabase _db = null!;
        PostgresMatchStore _store = null!;

        static CancellationToken Ct => CancellationToken.None;

        [SetUp]
        public async Task StartFromAFreshSchema()
        {
            _db = await PostgresTestDatabase.GetAsync();
            await _db.ResetAsync();
            await _db.ApplyMigrationsAsync();
            _store = new PostgresMatchStore(_db.DataSource, NullLogger<PostgresMatchStore>.Instance);
        }

        // ---- 1. the journal is read from one snapshot ---------------------------

        [Test]
        public async Task LoadJournal_ReadsOneSnapshot_SoAStartAndAnAppendMidReadCannotTearIt()
        {
            var match = await NewWaitingMatchAsync();

            await using NpgsqlDataSource otherPool = NpgsqlDataSource.Create(_db.ConnectionString);
            var other = new PostgresMatchStore(otherPool, NullLogger<PostgresMatchStore>.Instance);

            // Fires once the loader has read the match row and before it reads the rest: exactly the window a
            // read-committed loader would have taken three separate snapshots in.
            _store.OnJournalSnapshotForTests = async () =>
            {
                Assert.That(await other.TryStartMatchAsync(match.MatchId, "START-REPLAY", Started, Ct), Is.True);
                Assert.That(
                    (await other.AppendCommandAsync(match.MatchId, 1, "E 0", match.Seat0, Move1, Ct)).Status,
                    Is.EqualTo(AppendStatus.Appended));
            };

            MatchJournal? journal = await _store.LoadJournalAsync(match.MatchId, Ct);

            Assert.That(journal, Is.Not.Null);
            Assert.That(journal!.Match.Status == MatchStatus.Waiting && journal.Commands.Count > 0, Is.False,
                "a waiting match carrying journalled commands is a torn read: the row and the commands came "
                + "from different snapshots");

            if (journal.Match.Status == MatchStatus.Waiting)
            {
                Assert.That(journal.Match.StartReplay, Is.Null);
                Assert.That(journal.Commands, Is.Empty);
            }
            else
            {
                Assert.That(journal.Match.Status, Is.EqualTo(MatchStatus.Active));
                Assert.That(journal.Match.StartReplay, Is.EqualTo("START-REPLAY"));
                Assert.That(journal.Commands, Has.Count.EqualTo(1));
            }
        }

        [Test]
        public async Task LoadJournal_ForAMatchThatDoesNotExist_IsNull()
        {
            Assert.That(await _store.LoadJournalAsync(Guid.NewGuid(), Ct), Is.Null);
        }

        // ---- helpers ------------------------------------------------------------

        static string NextLobbyId() =>
            "1097752" + Interlocked.Increment(ref _identifiers).ToString("D11");

        static string NextSteamId() =>
            "7656119" + Interlocked.Increment(ref _identifiers).ToString("D10");

        static CreateMatchRequest Request(string lobbyId) =>
            new(lobbyId, SetupWire, EngineVersion, ProtocolVersion, BuildId,
                new[] { (NextSteamId(), 0), (NextSteamId(), 1) }, Created);

        async Task<(Guid MatchId, string Lobby, string Seat0, string Seat1)> NewWaitingMatchAsync()
        {
            CreateMatchRequest request = Request(NextLobbyId());
            CreateMatchResult created = await _store.CreateMatchForLobbyAsync(request, Ct);
            return (created.Match.MatchId, request.SteamLobbyId, request.Players[0].SteamId,
                request.Players[1].SteamId);
        }
    }
}

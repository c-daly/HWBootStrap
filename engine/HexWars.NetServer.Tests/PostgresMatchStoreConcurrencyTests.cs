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

        // ---- 2. catalogue saves are serialised with the start -------------------

        [Test]
        public async Task SaveCatalog_QueuesBehindTheMatchRowLock_AndNoOpsOnceTheStartCommits()
        {
            var match = await NewWaitingMatchAsync();
            await _store.SaveCatalogAsync(match.MatchId, match.Seat0, "catalog-before-start", Ct);

            await using NpgsqlConnection holder = await _db.DataSource.OpenConnectionAsync();
            await using NpgsqlTransaction holding = await holder.BeginTransactionAsync();

            await using (var takeLock = new NpgsqlCommand(
                "SELECT status FROM matches WHERE match_id = @matchId FOR UPDATE", holder, holding))
            {
                takeLock.Parameters.AddWithValue("matchId", match.MatchId);
                Assert.That(await takeLock.ExecuteScalarAsync(), Is.EqualTo("waiting"));
            }

            await using NpgsqlDataSource otherPool = NpgsqlDataSource.Create(_db.ConnectionString);
            var other = new PostgresMatchStore(otherPool, NullLogger<PostgresMatchStore>.Instance);
            Task save = other.SaveCatalogAsync(match.MatchId, match.Seat0, "catalog-after-start", Ct);

            Task first = await Task.WhenAny(save, Task.Delay(BlockedFor));
            Assert.That(first, Is.Not.SameAs(save),
                "a catalogue save must take the match row lock, so it cannot run while a start holds it");

            await using (var start = new NpgsqlCommand(
                "UPDATE matches SET status = 'active', start_replay = 'START-REPLAY', "
                + "started_at = now(), last_activity_at = now() WHERE match_id = @matchId", holder, holding))
            {
                start.Parameters.AddWithValue("matchId", match.MatchId);
                Assert.That(await start.ExecuteNonQueryAsync(), Is.EqualTo(1));
            }

            await holding.CommitAsync();
            await save;

            PersistedPlayer? player = await _store.GetPlayerAsync(match.MatchId, match.Seat0, Ct);
            Assert.That(player!.CatalogWire, Is.EqualTo("catalog-before-start"),
                "the save lost the race and must have done nothing at all");
        }

        // ---- 3. allocation survives a close/create race -------------------------

        [Test]
        public async Task CreateMatch_RetriesWhenTheMatchItCollidedWithClosesBeforeItCanBeReadBack()
        {
            string lobby = NextLobbyId();
            CreateMatchResult first = await _store.CreateMatchForLobbyAsync(Request(lobby), Ct);

            int conflicts = 0;
            _store.OnCreateConflictForTests = async () =>
            {
                conflicts++;
                await ExpireAsync(first.Match.MatchId);
            };

            CreateMatchResult second = await _store.CreateMatchForLobbyAsync(Request(lobby), Ct);

            Assert.That(conflicts, Is.EqualTo(1), "the first insert must have collided exactly once");
            Assert.That(second.Created, Is.True, "the lobby was free again, so the retry must have allocated");
            Assert.That(second.Match.MatchId, Is.Not.EqualTo(first.Match.MatchId));
            Assert.That(second.Match.Status, Is.EqualTo(MatchStatus.Waiting));
        }

        [Test]
        public async Task CreateMatch_GivesUpAfterThreeAttempts_WhenTheLobbyKeepsChurning()
        {
            string lobby = NextLobbyId();
            Guid open = (await _store.CreateMatchForLobbyAsync(Request(lobby), Ct)).Match.MatchId;

            int conflicts = 0, retries = 0;

            // Close the match the insert collided with, so the re-read finds nothing...
            _store.OnCreateConflictForTests = async () =>
            {
                conflicts++;
                await ExpireAsync(open);
            };

            // ...then open another one before the retry, so the next insert collides again.
            _store.OnCreateRetryForTests = async () =>
            {
                retries++;
                open = await InsertWaitingMatchAsync(lobby);
            };

            Assert.ThrowsAsync<InvalidOperationException>(
                () => _store.CreateMatchForLobbyAsync(Request(lobby), Ct));

            Assert.That(conflicts, Is.EqualTo(3), "three attempts, three collisions");
            Assert.That(retries, Is.EqualTo(2),
                "the third collision gives up rather than retrying a fourth time");
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

        async Task ExpireAsync(Guid matchId)
        {
            await using var command = _db.DataSource.CreateCommand(
                "UPDATE matches SET status = 'expired', completed_at = now(), last_activity_at = now() "
                + "WHERE match_id = @matchId");
            command.Parameters.AddWithValue("matchId", matchId);
            await command.ExecuteNonQueryAsync();
        }

        async Task<Guid> InsertWaitingMatchAsync(string lobbyId)
        {
            var matchId = Guid.NewGuid();
            await using var command = _db.DataSource.CreateCommand(
                "INSERT INTO matches (match_id, steam_lobby_id, status, setup_wire, engine_version, "
                + "protocol_version, build_id, created_at, last_activity_at) "
                + "VALUES (@matchId, @lobbyId, 'waiting', @setup, @engine, @protocol, @build, now(), now())");
            command.Parameters.AddWithValue("matchId", matchId);
            command.Parameters.AddWithValue("lobbyId", lobbyId);
            command.Parameters.AddWithValue("setup", SetupWire);
            command.Parameters.AddWithValue("engine", EngineVersion);
            command.Parameters.AddWithValue("protocol", ProtocolVersion);
            command.Parameters.AddWithValue("build", BuildId);
            await command.ExecuteNonQueryAsync();
            return matchId;
        }
    }
}

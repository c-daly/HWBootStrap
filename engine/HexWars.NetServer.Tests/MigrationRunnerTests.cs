using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.Extensions.Logging.Abstractions;
using Npgsql;
using NUnit.Framework;
using DbUrl = HexWars.NetServer.Persistence.DatabaseUrl;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// Exercises the migration runner and the schema it applies against a real Postgres. The constraint
    /// tests deliberately go through raw SQL rather than the store: they are asserting that the DATABASE
    /// refuses bad data, which is the guarantee the store layer above is allowed to rely on.
    /// </summary>
    [TestFixture]
    public class MigrationRunnerTests
    {
        const string OnlyMigration = "001_match_journal";
        const string LobbyId = "109775240000000001";

        PostgresTestDatabase _db = null!;

        [SetUp]
        public async Task StartFromAnEmptyDatabase()
        {
            _db = await PostgresTestDatabase.GetAsync();
            await _db.ResetAsync();
        }

        MigrationRunner Runner() => new(_db.DataSource, NullLogger<MigrationRunner>.Instance);

        // ---- discovery -------------------------------------------------------

        [Test]
        public void EmbeddedMigrations_AreReadFromTheAssembly()
        {
            var migrations = MigrationRunner.EmbeddedMigrations();

            Assert.That(migrations.Select(m => m.Version), Is.EqualTo(new[] { OnlyMigration }));
            Assert.That(migrations[0].Sql, Does.Contain("CREATE TABLE IF NOT EXISTS matches"));
            Assert.That(migrations[0].Sql, Does.Contain("ux_matches_open_lobby"));
        }

        // ---- apply / pending --------------------------------------------------

        [Test]
        public async Task PendingAsync_OnAnEmptyDatabase_ListsTheMigrationAndCreatesTheLedger()
        {
            var pending = await Runner().PendingAsync(CancellationToken.None);

            Assert.That(pending, Is.EqualTo(new[] { OnlyMigration }));
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM schema_migrations"), Is.EqualTo(0L));
        }

        [Test]
        public async Task FirstApply_AppliesTheMigration_SecondApplyDoesNothing()
        {
            var first = await Runner().ApplyAsync(CancellationToken.None);
            Assert.That(first, Is.EqualTo(new[] { OnlyMigration }));

            var second = await Runner().ApplyAsync(CancellationToken.None);
            Assert.That(second, Is.Empty);

            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM schema_migrations"), Is.EqualTo(1L));
            Assert.That(await Runner().PendingAsync(CancellationToken.None), Is.Empty);
        }

        [Test]
        public async Task ConcurrentApply_SerializesOnTheAdvisoryLock_AndAppliesTheMigrationOnce()
        {
            var results = await Task.WhenAll(
                Runner().ApplyAsync(CancellationToken.None),
                Runner().ApplyAsync(CancellationToken.None));

            Assert.That(results.SelectMany(r => r), Is.EqualTo(new[] { OnlyMigration }),
                "exactly one of the two racing runners should report the migration as applied");
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM schema_migrations"), Is.EqualTo(1L));
        }

        // ---- schema shape ------------------------------------------------------

        [Test]
        public async Task Migration_CreatesEveryTableAndIndex()
        {
            await _db.ApplyMigrationsAsync();

            var tables = await ListAsync(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema() ORDER BY table_name");
            Assert.That(tables, Is.SupersetOf(new[]
            {
                "match_commands", "match_join_credentials", "match_players", "matches", "schema_migrations",
            }));

            string? openLobby = await IndexDefinitionAsync("ux_matches_open_lobby");
            Assert.That(openLobby, Is.Not.Null, "the partial unique index on the open lobby is missing");
            Assert.That(openLobby, Does.Contain("UNIQUE INDEX"));
            Assert.That(openLobby, Does.Contain("WHERE"), "the index must be partial, not global");
            Assert.That(openLobby, Does.Contain("waiting"));

            Assert.That(await IndexDefinitionAsync("ix_matches_status_activity"), Is.Not.Null);
            Assert.That(await IndexDefinitionAsync("ix_join_credentials_match_player"), Is.Not.Null);
        }

        // ---- constraints --------------------------------------------------------

        [Test]
        public async Task OpenLobbyIndex_BlocksASecondWaitingMatch_AndFreesUpOnceTheFirstCompletes()
        {
            await _db.ApplyMigrationsAsync();
            var first = Guid.NewGuid();
            var second = Guid.NewGuid();
            await InsertMatchAsync(first, LobbyId, "waiting");

            var clash = Assert.ThrowsAsync<PostgresException>(() => InsertMatchAsync(second, LobbyId, "waiting"));
            Assert.That(clash!.SqlState, Is.EqualTo(PostgresErrorCodes.UniqueViolation));

            await ExecuteAsync("UPDATE matches SET status = @status WHERE match_id = @id",
                ("status", "completed"), ("id", first));

            Assert.DoesNotThrowAsync(() => InsertMatchAsync(second, LobbyId, "waiting"));
        }

        [Test]
        public async Task OpenLobbyIndex_ConstrainsOnlyTheSameNonNullLobby()
        {
            await _db.ApplyMigrationsAsync();

            await InsertMatchAsync(Guid.NewGuid(), LobbyId, "waiting");
            Assert.DoesNotThrowAsync(() => InsertMatchAsync(Guid.NewGuid(), "109775240000000002", "waiting"));

            // Legacy (non-Steam) matches carry no lobby id at all and must never collide with each other.
            await InsertMatchAsync(Guid.NewGuid(), null, "waiting");
            Assert.DoesNotThrowAsync(() => InsertMatchAsync(Guid.NewGuid(), null, "waiting"));
        }

        [Test]
        public async Task DuplicateCommandSequence_IsRejected()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, null, "active");
            await InsertCommandAsync(match, 1, "MOVE 0 0 1 1");

            var clash = Assert.ThrowsAsync<PostgresException>(() => InsertCommandAsync(match, 1, "MOVE 1 1 2 2"));

            Assert.That(clash!.SqlState, Is.EqualTo(PostgresErrorCodes.UniqueViolation));
        }

        [Test]
        public async Task NonNumericSteamId_IsRejected()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, null, "waiting");

            var bad = Assert.ThrowsAsync<PostgresException>(() => InsertPlayerAsync(match, "abc", 0));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.CheckViolation));
        }

        [Test]
        public async Task WinnerSeatOutsideTheTwoSeats_IsRejected()
        {
            await _db.ApplyMigrationsAsync();

            var bad = Assert.ThrowsAsync<PostgresException>(
                () => InsertMatchAsync(Guid.NewGuid(), null, "completed", winnerSeat: 2));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.CheckViolation));
        }

        [Test]
        public async Task UnknownStatus_IsRejected()
        {
            await _db.ApplyMigrationsAsync();

            var bad = Assert.ThrowsAsync<PostgresException>(() => InsertMatchAsync(Guid.NewGuid(), null, "bogus"));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.CheckViolation));
        }

        [Test]
        public async Task DeletingAMatch_CascadesToPlayersCommandsAndCredentials()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, LobbyId, "active");
            await InsertPlayerAsync(match, "76561190000000001", 0);
            await InsertPlayerAsync(match, "76561190000000002", 1);
            await InsertCommandAsync(match, 1, "MOVE 0 0 1 1");
            await InsertCredentialAsync(match, "76561190000000001");

            await ExecuteAsync("DELETE FROM matches WHERE match_id = @id", ("id", match));

            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_players"), Is.EqualTo(0L));
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_commands"), Is.EqualTo(0L));
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_join_credentials"), Is.EqualTo(0L));
        }

        // ---- the URL the deploy actually sets -------------------------------------

        [Test]
        public async Task FixtureDatabaseUrl_TranslatesToAConnectionStringThatConnects()
        {
            string connectionString = DbUrl.ToNpgsqlConnectionString(_db.DatabaseUrl);

            await using var dataSource = NpgsqlDataSource.Create(connectionString);
            await using var command = dataSource.CreateCommand("SELECT 1");

            Assert.That(await command.ExecuteScalarAsync(), Is.EqualTo(1));
        }

        // ---- helpers ---------------------------------------------------------------

        Task InsertMatchAsync(Guid matchId, string? lobbyId, string status, int? winnerSeat = null) =>
            ExecuteAsync(
                "INSERT INTO matches (match_id, steam_lobby_id, status, setup_wire, engine_version, "
                + "protocol_version, build_id, created_at, last_activity_at, winner_seat) "
                + "VALUES (@id, @lobby, @status, @setup, @engine, @protocol, @build, @now, @now, @winner)",
                ("id", matchId), ("lobby", lobbyId), ("status", status),
                ("setup", "Annihilation 9 7 0 5 3 1 1 1 3 0"), ("engine", "hexwars-engine/1"),
                ("protocol", 2), ("build", "test-build"), ("now", DateTimeOffset.UtcNow), ("winner", winnerSeat));

        Task InsertPlayerAsync(Guid matchId, string steamId, int seat) =>
            ExecuteAsync(
                "INSERT INTO match_players (match_id, steam_id, seat, joined_at) VALUES (@id, @steam, @seat, @now)",
                ("id", matchId), ("steam", steamId), ("seat", seat), ("now", DateTimeOffset.UtcNow));

        Task InsertCommandAsync(Guid matchId, int sequence, string wire) =>
            ExecuteAsync(
                "INSERT INTO match_commands (match_id, sequence, command_wire, accepted_at, issuer_steam_id) "
                + "VALUES (@id, @seq, @wire, @now, @issuer)",
                ("id", matchId), ("seq", sequence), ("wire", wire), ("now", DateTimeOffset.UtcNow),
                ("issuer", "76561190000000001"));

        Task InsertCredentialAsync(Guid matchId, string steamId) =>
            ExecuteAsync(
                "INSERT INTO match_join_credentials (credential_hash, match_id, steam_id, expires_at) "
                + "VALUES (@hash, @id, @steam, @expires)",
                ("hash", new byte[32]), ("id", matchId), ("steam", steamId),
                ("expires", DateTimeOffset.UtcNow.AddMinutes(15)));

        async Task ExecuteAsync(string sql, params (string Name, object? Value)[] parameters)
        {
            await using var command = _db.DataSource.CreateCommand(sql);
            foreach (var (name, value) in parameters) command.Parameters.AddWithValue(name, value ?? DBNull.Value);
            await command.ExecuteNonQueryAsync();
        }

        async Task<T> ScalarAsync<T>(string sql, params (string Name, object? Value)[] parameters)
        {
            await using var command = _db.DataSource.CreateCommand(sql);
            foreach (var (name, value) in parameters) command.Parameters.AddWithValue(name, value ?? DBNull.Value);
            return (T)(await command.ExecuteScalarAsync())!;
        }

        async Task<string?> IndexDefinitionAsync(string indexName)
        {
            await using var command = _db.DataSource.CreateCommand(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = @name");
            command.Parameters.AddWithValue("name", indexName);
            return await command.ExecuteScalarAsync() as string;
        }

        async Task<IReadOnlyList<string>> ListAsync(string sql)
        {
            var values = new List<string>();
            await using var command = _db.DataSource.CreateCommand(sql);
            await using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync()) values.Add(reader.GetString(0));
            return values;
        }
    }
}

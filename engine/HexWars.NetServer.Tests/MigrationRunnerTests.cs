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
        const string Seat0Steam = "76561190000000001";
        const string Seat1Steam = "76561190000000002";

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

        [Test]
        public async Task ConcurrentPending_OnAFreshDatabase_AllSucceed()
        {
            // Four readiness probes hitting a brand new database at once. Without the advisory lock the
            // racing CREATE TABLE IF NOT EXISTS statements collide in the system catalogue and one of them
            // fails with a duplicate key rather than doing nothing.
            var results = await Task.WhenAll(
                Runner().PendingAsync(CancellationToken.None),
                Runner().PendingAsync(CancellationToken.None),
                Runner().PendingAsync(CancellationToken.None),
                Runner().PendingAsync(CancellationToken.None));

            foreach (var pending in results) Assert.That(pending, Is.EqualTo(new[] { OnlyMigration }));
        }

        [Test]
        public async Task PendingAsync_QueuesBehindTheSameAdvisoryLockThatApplyTakes()
        {
            await using NpgsqlConnection holder = await _db.DataSource.OpenConnectionAsync();

            Task<IReadOnlyList<string>> pending;
            bool blocked;
            try
            {
                await OnConnectionAsync(holder, "SELECT pg_advisory_lock(" + MigrationRunner.AdvisoryLockKey + ")");

                pending = Runner().PendingAsync(CancellationToken.None);
                Task first = await Task.WhenAny(pending, Task.Delay(TimeSpan.FromSeconds(1)));
                blocked = !ReferenceEquals(first, pending);
            }
            finally
            {
                // Released before any assertion runs: a lock left held by a failing test would hang every
                // migration in the rest of the suite rather than failing this one test.
                await OnConnectionAsync(holder, "SELECT pg_advisory_unlock_all()");
            }

            Assert.That(blocked, Is.True,
                "reading the pending list must serialise on the same lock that applying does");
            Assert.That(await pending, Is.EqualTo(new[] { OnlyMigration }));
        }

        [Test]
        public async Task CancelledApply_LeavesNoAdvisoryLockHeld_SoTheNextApplyStillRuns()
        {
            using var cancelled = new CancellationTokenSource();
            await cancelled.CancelAsync();

            Assert.CatchAsync<OperationCanceledException>(() => Runner().ApplyAsync(cancelled.Token));

            // The lock is session level and every instance takes the same one at startup. A run that gave
            // up while holding it would queue every other instance behind a connection nobody will use
            // again, and the deploy would look like a hung migration rather than a cancelled one.
            Assert.That(await AdvisoryLocksAsync(), Is.EqualTo(0L),
                "the cancelled run left an advisory lock held");

            Task<IReadOnlyList<string>> next = Runner().ApplyAsync(CancellationToken.None);
            Task first = await Task.WhenAny(next, Task.Delay(TimeSpan.FromSeconds(10)));

            Assert.That(first, Is.SameAs(next), "the next apply queued behind the cancelled run's lock");
            Assert.That(await next, Is.EqualTo(new[] { OnlyMigration }));
        }

        [Test]
        public async Task ApplyCancelledWhileHoldingTheLock_StillReleasesIt()
        {
            using var cancelling = new CancellationTokenSource();
            MigrationRunner runner = Runner();
            long locksWhileRunning = -1;

            // Cancelling before the lock is taken proves nothing: the interesting moment is the one where
            // the lock is already held, because that is when giving up without releasing would queue every
            // other instance behind a connection nobody is going to use again.
            runner.AfterLockAcquiredForTests = async () =>
            {
                locksWhileRunning = await AdvisoryLocksAsync();
                await cancelling.CancelAsync();
            };

            Assert.CatchAsync<OperationCanceledException>(() => runner.ApplyAsync(cancelling.Token));

            Assert.That(locksWhileRunning, Is.EqualTo(1L), "the hook must run with the lock actually held");
            Assert.That(await AdvisoryLocksAsync(), Is.EqualTo(0L),
                "the cancelled run left the advisory lock held");

            Task<IReadOnlyList<string>> next = Runner().ApplyAsync(CancellationToken.None);
            Task first = await Task.WhenAny(next, Task.Delay(TimeSpan.FromSeconds(10)));

            Assert.That(first, Is.SameAs(next), "the next apply queued behind the cancelled run's lock");
            Assert.That(await next, Is.EqualTo(new[] { OnlyMigration }),
                "and it still had the whole migration left to do");
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

        [Test]
        public async Task Migration_CreatesTheIndexesTheRetentionSweeperNeeds()
        {
            await _db.ApplyMigrationsAsync();

            // One index per sweeper statement in docs/operations/match-data-retention.md, so an hourly sweep
            // over a large table is three range scans rather than three sequential scans.
            string? waiting = await IndexDefinitionAsync("ix_matches_waiting_created");
            Assert.That(waiting, Is.Not.Null, "the waiting-match expiry sweep has no index");
            Assert.That(waiting, Does.Contain("created_at"));
            Assert.That(waiting, Does.Contain("waiting"));

            string? terminal = await IndexDefinitionAsync("ix_matches_terminal_completed");
            Assert.That(terminal, Is.Not.Null, "the 90 day purge has no index");
            Assert.That(terminal, Does.Contain("completed_at"));
            Assert.That(terminal, Does.Contain("completed"));

            string? credentials = await IndexDefinitionAsync("ix_join_credentials_expires");
            Assert.That(credentials, Is.Not.Null, "the expired credential purge has no index");
            Assert.That(credentials, Does.Contain("expires_at"));
        }

        // ---- constraints --------------------------------------------------------

        [Test]
        public async Task OpenLobbyIndex_BlocksASecondWaitingMatch_AndFreesUpOnceTheFirstIsTerminal()
        {
            await _db.ApplyMigrationsAsync();
            var first = Guid.NewGuid();
            var second = Guid.NewGuid();
            await InsertMatchAsync(first, LobbyId, "waiting");

            var clash = Assert.ThrowsAsync<PostgresException>(() => InsertMatchAsync(second, LobbyId, "waiting"));
            Assert.That(clash!.SqlState, Is.EqualTo(PostgresErrorCodes.UniqueViolation));

            // waiting to expired is one of the two edges the store allows out of waiting, and it is the one
            // the retention sweeper uses. A raw jump to completed would break the lifecycle CHECKs.
            await ExecuteAsync(
                "UPDATE matches SET status = 'expired', completed_at = now() WHERE match_id = @id",
                ("id", first));

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
            await InsertPlayerAsync(match, Seat0Steam, 0);
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

        // ---- the lifecycle the schema enforces ------------------------------------

        [Test]
        public async Task AnActiveMatchWithoutAStartReplay_IsRejected()
        {
            await _db.ApplyMigrationsAsync();

            var bad = Assert.ThrowsAsync<PostgresException>(() => InsertMatchRowAsync(
                Guid.NewGuid(), null, "active", winnerSeat: null, startReplay: null, completedAt: null));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.CheckViolation),
                "an active match with no start replay cannot be replayed, so it must not be storable");
        }

        [Test]
        public async Task ACompletedMatchWithoutAStartReplay_IsRejected()
        {
            await _db.ApplyMigrationsAsync();

            var bad = Assert.ThrowsAsync<PostgresException>(() => InsertMatchRowAsync(
                Guid.NewGuid(), null, "completed", winnerSeat: 0, startReplay: null,
                completedAt: DateTimeOffset.UtcNow));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.CheckViolation));
        }

        [Test]
        public async Task ATerminalMatchWithoutACompletedAt_IsRejected()
        {
            await _db.ApplyMigrationsAsync();

            var bad = Assert.ThrowsAsync<PostgresException>(() => InsertMatchRowAsync(
                Guid.NewGuid(), null, "expired", winnerSeat: null, startReplay: null, completedAt: null));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.CheckViolation),
                "the retention purge keys off completed_at, so a terminal row without one never ages out");
        }

        [Test]
        public async Task AWinnerSeatOnAMatchThatDidNotComplete_IsRejected()
        {
            await _db.ApplyMigrationsAsync();

            var bad = Assert.ThrowsAsync<PostgresException>(() => InsertMatchRowAsync(
                Guid.NewGuid(), null, "abandoned", winnerSeat: 0, startReplay: "REPLAY",
                completedAt: DateTimeOffset.UtcNow));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.CheckViolation),
                "only a completed match has a winner; an abandoned one was never scored");
        }

        // ---- commands and credentials belong to a seat -----------------------------

        [Test]
        public async Task ACommandFromASteamIdWithNoSeat_IsRejected()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, null, "active");
            await InsertPlayerAsync(match, Seat0Steam, 0);

            var bad = Assert.ThrowsAsync<PostgresException>(
                () => InsertCommandAsync(match, 1, "MOVE 0 0 1 1", issuer: "76561190000000009"));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.ForeignKeyViolation));
        }

        [Test]
        public async Task ACredentialForASteamIdWithNoSeat_IsRejected()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, null, "waiting");
            await InsertPlayerAsync(match, Seat0Steam, 0);

            var bad = Assert.ThrowsAsync<PostgresException>(
                () => InsertCredentialAsync(match, "76561190000000009"));

            Assert.That(bad!.SqlState, Is.EqualTo(PostgresErrorCodes.ForeignKeyViolation));
        }

        [Test]
        public async Task DeletingAPlayerWhoIssuedCommands_IsRefused()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, LobbyId, "active");
            await InsertPlayerAsync(match, Seat0Steam, 0);
            await InsertPlayerAsync(match, Seat1Steam, 1);
            await InsertCommandAsync(match, 1, "MOVE 0 0 1 1");

            // The journal is the recovery story. Deleting the seat that issued a command out from under a
            // match that still exists would leave a log which replays into a different game, so the seat is
            // pinned by its commands rather than taking them with it.
            var refused = Assert.ThrowsAsync<PostgresException>(
                () => ExecuteAsync("DELETE FROM match_players WHERE match_id = @id AND steam_id = @steam",
                    ("id", match), ("steam", Seat0Steam)));

            Assert.That(refused!.SqlState, Is.EqualTo(PostgresErrorCodes.ForeignKeyViolation));
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_commands"), Is.EqualTo(1L),
                "a refused delete must leave the journal exactly as it was");
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_players"), Is.EqualTo(2L));
        }

        [Test]
        public async Task DeletingAPlayerWhoIssuedNoCommands_TakesOnlyTheirCredentials()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, LobbyId, "active");
            await InsertPlayerAsync(match, Seat0Steam, 0);
            await InsertPlayerAsync(match, Seat1Steam, 1);
            await InsertCommandAsync(match, 1, "MOVE 0 0 1 1", issuer: Seat1Steam);
            await InsertCredentialAsync(match, Seat0Steam);

            await ExecuteAsync("DELETE FROM match_players WHERE match_id = @id AND steam_id = @steam",
                ("id", match), ("steam", Seat0Steam));

            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_join_credentials"), Is.EqualTo(0L),
                "a join token is issued to a seat, so it dies with the seat");
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_commands"), Is.EqualTo(1L),
                "somebody else issued that command");
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM matches"), Is.EqualTo(1L),
                "losing a seat must not take the match row with it");
        }

        [Test]
        public async Task DeletingAMatch_CascadesToPlayersCommandsAndCredentials()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, LobbyId, "active");
            await InsertPlayerAsync(match, Seat0Steam, 0);
            await InsertPlayerAsync(match, Seat1Steam, 1);
            await InsertCommandAsync(match, 1, "MOVE 0 0 1 1");
            await InsertCredentialAsync(match, Seat0Steam);

            await ExecuteAsync("DELETE FROM matches WHERE match_id = @id", ("id", match));

            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_players"), Is.EqualTo(0L));
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_commands"), Is.EqualTo(0L));
            Assert.That(await ScalarAsync<long>("SELECT count(*) FROM match_join_credentials"), Is.EqualTo(0L));
        }

        // ---- the lifecycle is a one-way street ------------------------------------

        [Test]
        public async Task ACompletedDraw_CannotBeSetBackToActive()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, LobbyId, "active");
            await SetStatusAsync(match, "completed", "completed_at = now()");

            // A draw, so no winner_seat CHECK can be the thing that refuses this: the only reason left is
            // the trigger. A finished game that can be nudged back to active starts accepting commands.
            var refused = Assert.ThrowsAsync<PostgresException>(() => SetStatusAsync(match, "active"));

            Assert.That(refused!.SqlState, Is.EqualTo(PostgresErrorCodes.RaiseException));
            Assert.That(refused.MessageText, Does.Contain("illegal match status transition"));
            Assert.That(await StatusAsync(match), Is.EqualTo("completed"));
        }

        [Test]
        public async Task WaitingStraightToCompleted_IsRejectedEvenWithAReplayAndACompletedAt()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, LobbyId, "waiting");

            // Everything the lifecycle CHECKs want is supplied, so this row would satisfy the column
            // constraints. It is still a result for a game nobody played.
            var refused = Assert.ThrowsAsync<PostgresException>(
                () => SetStatusAsync(match, "completed", "start_replay = 'REPLAY', completed_at = now()"));

            Assert.That(refused!.SqlState, Is.EqualTo(PostgresErrorCodes.RaiseException));
            Assert.That(refused.MessageText, Does.Contain("illegal match status transition"));
            Assert.That(await StatusAsync(match), Is.EqualTo("waiting"));
        }

        [TestCase("waiting", "active")]
        [TestCase("waiting", "expired")]
        [TestCase("waiting", "abandoned")]
        [TestCase("active", "completed")]
        [TestCase("active", "expired")]
        [TestCase("active", "abandoned")]
        public async Task EveryEdgeTheStoreTakes_IsAccepted(string from, string to)
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, LobbyId, from);

            // Whatever the destination status needs, so the trigger is the only thing under test.
            string extra = to switch
            {
                "active" => "start_replay = 'REPLAY'",
                _ => "start_replay = COALESCE(start_replay, 'REPLAY'), completed_at = now()",
            };

            Assert.DoesNotThrowAsync(() => SetStatusAsync(match, to, extra));
            Assert.That(await StatusAsync(match), Is.EqualTo(to));
        }

        [Test]
        public async Task RewritingTheStatusAMatchAlreadyHas_IsNotATransition()
        {
            await _db.ApplyMigrationsAsync();
            var match = Guid.NewGuid();
            await InsertMatchAsync(match, LobbyId, "active");

            // The trigger fires on any UPDATE that assigns the column, including one that assigns the value
            // already there. That is a rewrite, not a move, and it must not be refused.
            Assert.DoesNotThrowAsync(() => SetStatusAsync(match, "active"));
            Assert.That(await StatusAsync(match), Is.EqualTo("active"));
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

        /// <summary>Inserts a row that satisfies the lifecycle CHECKs for the status asked for, so a test
        /// about one constraint is never tripped by another.</summary>
        Task InsertMatchAsync(Guid matchId, string? lobbyId, string status, int? winnerSeat = null) =>
            InsertMatchRowAsync(matchId, lobbyId, status, winnerSeat,
                startReplay: status is "active" or "completed" ? "REPLAY" : null,
                completedAt: status is "waiting" or "active" ? null : DateTimeOffset.UtcNow);

        /// <summary>Every column spelled out, for the tests that mean to violate something.</summary>
        Task InsertMatchRowAsync(Guid matchId, string? lobbyId, string status, int? winnerSeat,
            string? startReplay, DateTimeOffset? completedAt) =>
            ExecuteAsync(
                "INSERT INTO matches (match_id, steam_lobby_id, status, setup_wire, start_replay, "
                + "engine_version, protocol_version, build_id, created_at, completed_at, last_activity_at, "
                + "winner_seat) "
                + "VALUES (@id, @lobby, @status, @setup, @replay, @engine, @protocol, @build, @now, "
                + "@completed, @now, @winner)",
                ("id", matchId), ("lobby", lobbyId), ("status", status),
                ("setup", "Annihilation 9 7 0 5 3 1 1 1 3 0"), ("replay", startReplay),
                ("engine", "hexwars-engine/1"), ("protocol", 2), ("build", "test-build"),
                ("now", DateTimeOffset.UtcNow), ("completed", completedAt), ("winner", winnerSeat));

        /// <summary>Moves a match to a status, optionally setting whatever else that status needs. The
        /// status column is always assigned, so the transition trigger always fires.</summary>
        Task SetStatusAsync(Guid matchId, string status, string? alsoSet = null) =>
            ExecuteAsync(
                "UPDATE matches SET status = @status"
                + (alsoSet is null ? string.Empty : ", " + alsoSet)
                + " WHERE match_id = @id",
                ("status", status), ("id", matchId));

        Task<string> StatusAsync(Guid matchId) =>
            ScalarAsync<string>("SELECT status FROM matches WHERE match_id = @id", ("id", matchId));

        Task InsertPlayerAsync(Guid matchId, string steamId, int seat) =>
            ExecuteAsync(
                "INSERT INTO match_players (match_id, steam_id, seat, joined_at) VALUES (@id, @steam, @seat, @now)",
                ("id", matchId), ("steam", steamId), ("seat", seat), ("now", DateTimeOffset.UtcNow));

        Task InsertCommandAsync(Guid matchId, int sequence, string wire, string issuer = Seat0Steam) =>
            ExecuteAsync(
                "INSERT INTO match_commands (match_id, sequence, command_wire, accepted_at, issuer_steam_id) "
                + "VALUES (@id, @seq, @wire, @now, @issuer)",
                ("id", matchId), ("seq", sequence), ("wire", wire), ("now", DateTimeOffset.UtcNow),
                ("issuer", issuer));

        Task InsertCredentialAsync(Guid matchId, string steamId) =>
            ExecuteAsync(
                "INSERT INTO match_join_credentials (credential_hash, match_id, steam_id, expires_at) "
                + "VALUES (@hash, @id, @steam, @expires)",
                ("hash", new byte[32]), ("id", matchId), ("steam", steamId),
                ("expires", DateTimeOffset.UtcNow.AddMinutes(15)));

        /// <summary>How many advisory locks this database is holding right now, read from a connection
        /// other than the one under test. The suite runs one fixture at a time against one database, so
        /// anything counted here belongs to the code being tested.</summary>
        Task<long> AdvisoryLocksAsync() =>
            ScalarAsync<long>("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'");

        static async Task OnConnectionAsync(NpgsqlConnection connection, string sql)
        {
            await using var command = new NpgsqlCommand(sql, connection);
            await command.ExecuteNonQueryAsync();
        }

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

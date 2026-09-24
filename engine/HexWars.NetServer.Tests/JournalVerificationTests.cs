using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.Extensions.Logging.Abstractions;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The read-only verb an operator points at a restored backup.
    ///
    /// Two claims have to hold, or the procedure that uses it is worse than nothing: it must report what is
    /// actually wrong with the journals it finds, and it must not change a byte of the database it was
    /// pointed at. The second needs proving rather than reading, so it is proved by counting the rows
    /// Postgres itself says were inserted, updated and deleted.
    /// </summary>
    [TestFixture]
    public class JournalVerificationTests
    {
        const string Seat0 = "76561198000000001";
        const string Seat1 = "76561198000000002";

        PostgresTestDatabase _database = null!;

        [SetUp]
        public async Task AFreshSchema()
        {
            _database = await PostgresTestDatabase.GetAsync();
            await _database.ResetAsync();
            await _database.ApplyMigrationsAsync();
        }

        PostgresMatchStore Store() => new(_database.DataSource, NullLogger<PostgresMatchStore>.Instance);

        async Task<Guid> SeedAsync(
            string lobbyId, string engineVersion, bool corruptCommand, NpgsqlDataSource? into = null)
        {
            NpgsqlDataSource source = into ?? _database.DataSource;
            var store = new PostgresMatchStore(source, NullLogger<PostgresMatchStore>.Instance);

            CreateMatchResult created = await store.CreateMatchForLobbyAsync(
                new CreateMatchRequest(
                    lobbyId,
                    GameSetup.Default.ToWire(),
                    engineVersion,
                    2,
                    "test-build",
                    new[] { (Seat0, 0), (Seat1, 1) },
                    DateTimeOffset.UtcNow),
                CancellationToken.None);

            Guid matchId = created.Match.MatchId;

            GameState start = GameFactory.Build(GameSetup.Default);
            await store.TryStartMatchAsync(
                matchId,
                ReplayFile.Write(start, Array.Empty<Command>()),
                DateTimeOffset.UtcNow,
                CancellationToken.None);

            if (!corruptCommand) return matchId;

            // Written through the store so the row is the shape production writes, then made unreadable in
            // place: a journal that parses everywhere except at the one command it holds.
            await store.AppendCommandAsync(
                matchId, 1, CommandWire.Write(new EndTurn(PlayerId.Player0)), Seat0,
                DateTimeOffset.UtcNow, CancellationToken.None);

            await using NpgsqlConnection connection =
                await source.OpenConnectionAsync(CancellationToken.None);
            await using NpgsqlCommand corrupt = connection.CreateCommand();
            corrupt.CommandText =
                "UPDATE match_commands SET command_wire = 'NOT A COMMAND' WHERE match_id = @id";
            corrupt.Parameters.AddWithValue("id", matchId);
            await corrupt.ExecuteNonQueryAsync(CancellationToken.None);

            return matchId;
        }

        async Task<(int Exit, string Output)> RunAsync(bool openOnly = false)
        {
            await using NpgsqlDataSource source = JournalVerification.ReadOnlySource(_database.DatabaseUrl);

            var written = new StringWriter();
            int exit = await JournalVerification.VerifyAsync(source, openOnly, written, CancellationToken.None);

            return (exit, written.ToString());
        }

        /// <summary>
        /// Everything the verb could possibly have changed, as one string.
        ///
        /// Content rather than pg_stat counters: those are flushed asynchronously, so a write from the
        /// SEEDING can still be in flight when the first reading is taken and land before the second,
        /// which fails a test about the verb for something the test itself did.
        /// </summary>
        async Task<string> FingerprintAsync()
        {
            await using NpgsqlConnection connection =
                await _database.DataSource.OpenConnectionAsync(CancellationToken.None);
            await using NpgsqlCommand read = connection.CreateCommand();
            read.CommandText =
                "SELECT md5(coalesce(string_agg(row, '|' ORDER BY row), '')) FROM ("
                + "  SELECT matches::text AS row FROM matches UNION ALL"
                + "  SELECT match_players::text FROM match_players UNION ALL"
                + "  SELECT match_commands::text FROM match_commands UNION ALL"
                + "  SELECT schema_migrations::text FROM schema_migrations"
                + ") AS everything";

            return (string)(await read.ExecuteScalarAsync(CancellationToken.None))!;
        }

        [Test]
        public async Task AHealthySetPasses()
        {
            await SeedAsync("109775240000000001", EngineContract.Version, corruptCommand: false);
            await SeedAsync("109775240000000002", EngineContract.Version, corruptCommand: false);

            (int exit, string output) = await RunAsync();

            Assert.That(exit, Is.Zero, output);
            Assert.That(output, Does.Contain(JournalVerification.PassPrefix + " 2/2"));
        }

        [Test]
        public async Task OneBadJournalFailsAndBothMatchesAreReported()
        {
            Guid healthy = await SeedAsync("109775240000000001", EngineContract.Version, false);
            Guid broken = await SeedAsync("109775240000000002", EngineContract.Version, true);

            (int exit, string output) = await RunAsync();

            Assert.That(exit, Is.EqualTo(1), output);
            Assert.That(output, Does.Contain(JournalVerification.FailPrefix + " 1/2"));
            Assert.That(output, Does.Contain(healthy.ToString()));
            Assert.That(output, Does.Contain(broken.ToString()));
            Assert.That(output, Does.Contain("CorruptCommand"),
                "the line has to say what is wrong, not only that something is");
        }

        [Test]
        public async Task AnUnsupportedEngineContractIsReportedRatherThanReplayed()
        {
            await SeedAsync("109775240000000003", "hexwars-engine/999", false);

            (int exit, string output) = await RunAsync();

            Assert.That(exit, Is.EqualTo(1), output);
            Assert.That(output, Does.Contain("UnsupportedEngineContract"));
        }

        [Test]
        public async Task ItWritesNothingAtAll()
        {
            await SeedAsync("109775240000000001", EngineContract.Version, false);
            await SeedAsync("109775240000000002", EngineContract.Version, true);

            string before = await FingerprintAsync();
            (int exit, _) = await RunAsync();
            string after = await FingerprintAsync();

            Assert.That(exit, Is.EqualTo(1));
            Assert.That(after, Is.EqualTo(before),
                "something in this database changed during a verb that promises to change nothing");
        }

        [Test]
        public async Task ItRefusesAWriteEvenWhenSomethingTries()
        {
            await using NpgsqlDataSource source = JournalVerification.ReadOnlySource(_database.DatabaseUrl);

            await using NpgsqlConnection connection =
                await source.OpenConnectionAsync(CancellationToken.None);
            await using NpgsqlCommand write = connection.CreateCommand();
            write.CommandText = "UPDATE matches SET build_id = 'nope'";

            // The guarantee is the server refusing, not this code remembering to be careful. A write added
            // below this verb later - a migration, a heal - fails in exactly the same way.
            Assert.ThrowsAsync<PostgresException>(
                async () => await write.ExecuteNonQueryAsync(CancellationToken.None));
        }

        [Test]
        public async Task ItAcceptsADatabaseNameThatIsNotDisposable()
        {
            // The whole reason this verb exists. selftest-durable refuses anything not marked disposable,
            // and drops the public schema when it does run, so it can never be pointed at the data an
            // operator actually needs verified.
            Assert.That(
                Fixtures.DisposableDatabaseGuard.IsDisposable(
                    "postgres://u:p@db.internal:5432/hexwars", _ => null, out _),
                Is.False,
                "the production database name is not disposable, which is what makes this verb necessary");

            await SeedAsync("109775240000000001", EngineContract.Version, false);

            (int exit, string output) = await RunAsync();
            Assert.That(exit, Is.Zero, output);
        }

        [Test]
        public async Task AnUnmigratedDatabaseIsAConfigurationError()
        {
            await _database.ResetAsync();

            (int exit, string output) = await RunAsync();

            Assert.That(exit, Is.EqualTo(2), output);
            Assert.That(output, Does.Contain("the schema is behind this build"));
        }

        [Test]
        public async Task OpenOnlySkipsTheMatchesThatHaveFinished()
        {
            Guid open = await SeedAsync("109775240000000001", EngineContract.Version, false);
            Guid broken = await SeedAsync("109775240000000002", EngineContract.Version, true);

            await Store().TryCompleteMatchAsync(
                broken, MatchStatus.Completed, 0, DateTimeOffset.UtcNow, CancellationToken.None);

            (int exit, string output) = await RunAsync(openOnly: true);

            Assert.That(exit, Is.Zero, output);
            Assert.That(output, Does.Contain(open.ToString()));
            Assert.That(output, Does.Not.Contain(broken.ToString()));
        }

        /// <summary>The name this test needs: no delimited "test" token anywhere in it, so
        /// <see cref="DisposableDatabaseGuard"/> refuses it the way it refuses a production database.</summary>
        const string NotDisposableDatabase = "hexwars_verify";

        /// <summary>
        /// Program.Main, against a database that could not be handed to anything which drops schemas.
        ///
        /// The test below runs the same boundary against hexwars_test, and that name IS disposable: a guard
        /// reintroduced in front of this verb would pass it and the operator procedure would still be
        /// broken, silently. So this one builds a sibling database on the same server, named the way real
        /// data is named, and verifies THAT.
        /// </summary>
        [Test]
        public async Task ThroughTheCommandBoundary_ItVerifiesADatabaseNoGuardWouldCallDisposable()
        {
            await using PostgresTestDatabase.SiblingDatabase real =
                await _database.CreateSiblingAsync(NotDisposableDatabase);

            Assert.That(
                Fixtures.DisposableDatabaseGuard.IsDisposable(real.DatabaseUrl, _ => null, out string reason),
                Is.False,
                "this test is only worth anything against a database a guard would refuse: " + reason);

            await real.ApplyMigrationsAsync();
            await SeedAsync("109775240000000001", EngineContract.Version, false, real.DataSource);

            string? verify = Environment.GetEnvironmentVariable(JournalVerification.DatabaseVariable);
            string? database = Environment.GetEnvironmentVariable("DATABASE_URL");
            TextWriter console = Console.Out;
            var written = new StringWriter();
            int exit;

            try
            {
                Environment.SetEnvironmentVariable(JournalVerification.DatabaseVariable, real.DatabaseUrl);
                Environment.SetEnvironmentVariable("DATABASE_URL", null);

                // Main writes to Console.Out rather than to a writer it was handed, so the console is where
                // the operator's answer has to be read from.
                Console.SetOut(written);
                exit = await Program.Main(new[] { "verify-journals" });
            }
            finally
            {
                Console.SetOut(console);
                Environment.SetEnvironmentVariable(JournalVerification.DatabaseVariable, verify);
                Environment.SetEnvironmentVariable("DATABASE_URL", database);
            }

            Assert.That(exit, Is.Zero, written.ToString());
            Assert.That(written.ToString(), Does.Contain(JournalVerification.PassPrefix + " 1/1"));
        }

        [Test]
        public async Task ThroughTheCommandBoundary_ItAcceptsANonDisposableNameAndFallsBackToDatabaseUrl()
        {
            await SeedAsync("109775240000000001", EngineContract.Version, false);

            string? verify = Environment.GetEnvironmentVariable(JournalVerification.DatabaseVariable);
            string? database = Environment.GetEnvironmentVariable("DATABASE_URL");

            try
            {
                // Through Program.Main, which is where an operator actually reaches it. The database is
                // named hexwars_test here only because the fixture names it that; what matters is that no
                // disposable-name guard stands between this verb and a real database, because standing
                // between them is exactly what made the previous procedure impossible to follow.
                Environment.SetEnvironmentVariable(
                    JournalVerification.DatabaseVariable, _database.DatabaseUrl);
                Environment.SetEnvironmentVariable("DATABASE_URL", null);

                Assert.That(await Program.Main(new[] { "verify-journals" }), Is.Zero);

                // And the fallback: no verify variable, so it uses the address the service itself runs on.
                Environment.SetEnvironmentVariable(JournalVerification.DatabaseVariable, null);
                Environment.SetEnvironmentVariable("DATABASE_URL", _database.DatabaseUrl);

                Assert.That(await Program.Main(new[] { "verify-journals" }), Is.Zero);

                // With neither, it says so rather than guessing.
                Environment.SetEnvironmentVariable("DATABASE_URL", null);
                Assert.That(await Program.Main(new[] { "verify-journals" }), Is.EqualTo(2));
            }
            finally
            {
                Environment.SetEnvironmentVariable(JournalVerification.DatabaseVariable, verify);
                Environment.SetEnvironmentVariable("DATABASE_URL", database);
            }
        }


        [Test]
        public async Task TurningTheSessionDefaultOffDoesNotDefeatTheVerifierQueries()
        {
            await using NpgsqlDataSource source = JournalVerification.ReadOnlySource(_database.DatabaseUrl);

            // The bypass the session default cannot survive: SET as its OWN command, so it commits before
            // the write is even parsed. This is what makes the default a default rather than a guarantee.
            await using (NpgsqlConnection loosened = await source.OpenConnectionAsync(CancellationToken.None))
            await using (NpgsqlCommand off = loosened.CreateCommand())
            {
                off.CommandText = "SET default_transaction_read_only = off";
                await off.ExecuteNonQueryAsync(CancellationToken.None);

                await using NpgsqlCommand write = loosened.CreateCommand();
                write.CommandText = "UPDATE matches SET build_id = 'nope'";

                // Honest about what the session default is worth on its own: nothing, once something has
                // turned it off. The guarantee has to come from the transaction instead.
                Assert.DoesNotThrowAsync(
                    async () => await write.ExecuteNonQueryAsync(CancellationToken.None));
            }

            // And the same two commands through the path the verb actually uses. SET TRANSACTION READ ONLY
            // cannot be lifted from inside the transaction it applies to, so the write is refused however
            // the session was left.
            Assert.ThrowsAsync<PostgresException>(async () => await JournalVerification.ReadOnlyQueryAsync(
                source,
                "UPDATE matches SET build_id = 'nope'",
                async (command, token) => await command.ExecuteNonQueryAsync(token),
                CancellationToken.None));

            Assert.ThrowsAsync<PostgresException>(async () => await JournalVerification.ReadOnlyQueryAsync(
                source,
                "CREATE TABLE verify_should_not_exist (id int)",
                async (command, token) => await command.ExecuteNonQueryAsync(token),
                CancellationToken.None));
        }

        [Test]
        public void EverySqlStatementInTheVerbGoesThroughTheReadOnlyHelper()
        {
            // A guarantee that rests on one method is only as good as the promise that nothing else opens
            // a command. That promise is checkable, so it is checked: the two CreateCommand calls in the
            // file are the two inside the helper itself.
            string source = File.ReadAllText(VerbSourcePath());

            int commands = source.Split("CreateCommand()").Length - 1;
            int inHelper = source[source.IndexOf("ReadOnlyQueryAsync<T>", StringComparison.Ordinal)..]
                .Split("CreateCommand()").Length - 1;

            Assert.That(commands, Is.EqualTo(inHelper),
                "a statement outside ReadOnlyQueryAsync would run without a read-only transaction");
        }

        static string VerbSourcePath()
        {
            var directory = new DirectoryInfo(TestContext.CurrentContext.TestDirectory);
            while (directory is not null && directory.Name != "engine") directory = directory.Parent;

            Assert.That(directory, Is.Not.Null, "could not find the engine directory from the test output");
            return Path.Combine(
                directory!.FullName, "HexWars.NetServer", "Operations", "JournalVerification.cs");
        }
    }
}

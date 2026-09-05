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

        async Task<Guid> SeedAsync(string lobbyId, string engineVersion, bool corruptCommand)
        {
            PostgresMatchStore store = Store();

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
                await _database.DataSource.OpenConnectionAsync(CancellationToken.None);
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

        /// <summary>What Postgres says it has written to this database, ever.</summary>
        async Task<long> WritesAsync()
        {
            await using NpgsqlConnection connection =
                await _database.DataSource.OpenConnectionAsync(CancellationToken.None);
            await using NpgsqlCommand read = connection.CreateCommand();
            read.CommandText =
                "SELECT coalesce(sum(n_tup_ins + n_tup_upd + n_tup_del), 0) "
                + "FROM pg_stat_user_tables WHERE schemaname = 'public'";

            object? answer = await read.ExecuteScalarAsync(CancellationToken.None);
            return Convert.ToInt64(answer);
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

            long before = await WritesAsync();
            (int exit, _) = await RunAsync();
            long after = await WritesAsync();

            Assert.That(exit, Is.EqualTo(1));
            Assert.That(after, Is.EqualTo(before),
                "Postgres counted an insert, update or delete during a verb that promises none");
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
    }
}

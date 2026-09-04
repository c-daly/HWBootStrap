using System.Globalization;
using HexWars.NetServer.Persistence;
using Microsoft.Extensions.Logging.Abstractions;
using Npgsql;
using NUnit.Framework;
using Testcontainers.PostgreSql;
using DbUrl = HexWars.NetServer.Persistence.DatabaseUrl;

namespace HexWars.NetServer.Tests.Fixtures
{
    /// <summary>
    /// The single Postgres that every database-backed test in this process shares. Starting a container
    /// costs seconds, so it is started once and reused; tests get isolation from <see cref="ResetAsync"/>
    /// instead, which is why this class hands out a database it is allowed to drop the schema of.
    ///
    /// Set HEXWARS_TEST_DATABASE_URL to run against an existing Postgres (CI with a service container, or
    /// a developer without Docker). Anything else starts postgres:16-alpine through Testcontainers. There
    /// is deliberately no third path: a missing database fails the tests loudly rather than skipping them,
    /// because a silently skipped persistence suite is how a broken schema reaches production.
    /// </summary>
    public sealed class PostgresTestDatabase : IAsyncDisposable
    {
        public const string OverrideEnvironmentVariable = "HEXWARS_TEST_DATABASE_URL";
        public const string ContainerImage = "postgres:16-alpine";

        static readonly SemaphoreSlim Gate = new(1, 1);
        static PostgresTestDatabase? _instance;

        readonly PostgreSqlContainer? _container;

        PostgresTestDatabase(PostgreSqlContainer? container, string databaseUrl, string connectionString)
        {
            _container = container;
            DatabaseUrl = databaseUrl;
            ConnectionString = connectionString;
            DataSource = NpgsqlDataSource.Create(connectionString);
        }

        /// <summary>postgres://user:password@host:port/database — exactly what a deploy puts in DATABASE_URL,
        /// so a test can feed it straight to UseSetting and exercise the real translation path.</summary>
        public string DatabaseUrl { get; }

        /// <summary>The same target in the Key=Value form Npgsql consumes.</summary>
        public string ConnectionString { get; }

        /// <summary>A pool the test owns, independent of whatever the server under test builds.</summary>
        public NpgsqlDataSource DataSource { get; }

        /// <summary>True when this run is driving a container rather than a supplied database.</summary>
        public bool UsesContainer => _container is not null;

        public static async Task<PostgresTestDatabase> GetAsync()
        {
            if (_instance is not null) return _instance;

            await Gate.WaitAsync().ConfigureAwait(false);
            try
            {
                return _instance ??= await CreateAsync().ConfigureAwait(false);
            }
            finally
            {
                Gate.Release();
            }
        }

        static async Task<PostgresTestDatabase> CreateAsync()
        {
            string? supplied = Environment.GetEnvironmentVariable(OverrideEnvironmentVariable);
            if (!string.IsNullOrWhiteSpace(supplied))
            {
                string connectionString = DbUrl.ToNpgsqlConnectionString(supplied!);
                return new PostgresTestDatabase(null, ComposeUrl(connectionString), connectionString);
            }

            var container = new PostgreSqlBuilder().WithImage(ContainerImage).Build();
            try
            {
                await container.StartAsync().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    "Could not start a disposable Postgres for the persistence tests. Either run a Docker daemon "
                    + "that can pull " + ContainerImage + ", or point " + OverrideEnvironmentVariable
                    + " at a throwaway Postgres (postgres://user:password@host:port/database). Note that the test "
                    + "run DROPs and recreates the public schema of whatever database it is given.", ex);
            }

            string fromContainer = container.GetConnectionString();
            return new PostgresTestDatabase(container, ComposeUrl(fromContainer), fromContainer);
        }

        /// <summary>Npgsql key=value back to the postgres:// URL a hosting platform would hand us.</summary>
        static string ComposeUrl(string connectionString)
        {
            var parts = new NpgsqlConnectionStringBuilder(connectionString);
            string host = string.IsNullOrEmpty(parts.Host) ? "localhost" : parts.Host!;
            string database = string.IsNullOrEmpty(parts.Database) ? "postgres" : parts.Database!;
            string user = Uri.EscapeDataString(parts.Username ?? string.Empty);
            string password = Uri.EscapeDataString(parts.Password ?? string.Empty);
            string port = parts.Port.ToString(CultureInfo.InvariantCulture);
            return "postgres://" + user + ":" + password + "@" + host + ":" + port + "/" + database;
        }

        /// <summary>Back to an empty database. Every fixture that touches Postgres starts here, so no test
        /// can depend on rows or tables another one left behind.</summary>
        public async Task ResetAsync()
        {
            await using var connection = await DataSource.OpenConnectionAsync().ConfigureAwait(false);
            await using var command = new NpgsqlCommand(
                "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;", connection);
            await command.ExecuteNonQueryAsync().ConfigureAwait(false);
        }

        /// <summary>Runs the real production migration runner, not a test copy of the schema.</summary>
        public Task<IReadOnlyList<string>> ApplyMigrationsAsync() =>
            new MigrationRunner(DataSource, NullLogger<MigrationRunner>.Instance).ApplyAsync(CancellationToken.None);

        internal static async Task ShutdownAsync()
        {
            var instance = Interlocked.Exchange(ref _instance, null);
            if (instance is not null) await instance.DisposeAsync().ConfigureAwait(false);
        }

        public async ValueTask DisposeAsync()
        {
            await DataSource.DisposeAsync().ConfigureAwait(false);
            if (_container is not null) await _container.DisposeAsync().ConfigureAwait(false);
        }
    }
}

namespace HexWars.NetServer.Tests
{
    /// <summary>Assembly-wide teardown: without this the Testcontainers Postgres outlives the test run.</summary>
    [SetUpFixture]
    public sealed class PostgresTestDatabaseLifetime
    {
        [OneTimeTearDown]
        public async Task StopSharedDatabase() => await Fixtures.PostgresTestDatabase.ShutdownAsync();
    }
}

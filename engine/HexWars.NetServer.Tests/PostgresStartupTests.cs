using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// Proves the composition root reaches Postgres exactly when DATABASE_URL says so, and that a database
    /// it cannot reach stops the process rather than letting it serve traffic against no schema. The
    /// no-database case matters just as much: the legacy deployment has never had a database and must keep
    /// booting untouched.
    /// </summary>
    [TestFixture]
    public class PostgresStartupTests
    {
        [Test]
        public async Task WithDatabaseUrl_TheHostAppliesTheMigrationsBeforeServingTraffic()
        {
            var db = await PostgresTestDatabase.GetAsync();
            await db.ResetAsync();

            using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            {
                builder.UseEnvironment("Development");
                builder.UseSetting("DATABASE_URL", db.DatabaseUrl);
            });
            using var client = factory.CreateClient();

            Assert.That(await client.GetStringAsync("/healthz"), Is.EqualTo("ok"));

            await using var command = db.DataSource.CreateCommand("SELECT version FROM schema_migrations ORDER BY version");
            var applied = new List<string>();
            await using (var reader = await command.ExecuteReaderAsync())
            {
                while (await reader.ReadAsync()) applied.Add(reader.GetString(0));
            }

            Assert.That(applied, Is.EqualTo(new[] { "001_match_journal" }));
        }

        [Test]
        public void WithAnUnreachableDatabase_TheHostRefusesToStart()
        {
            var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            {
                builder.UseEnvironment("Development");
                builder.UseSetting("DATABASE_URL", "postgres://u:p@127.0.0.1:1/db");
            });

            try
            {
                // Fail closed: a match host that cannot reach its journal must not accept players.
                Assert.That(() => factory.CreateClient(), Throws.Exception);
            }
            finally
            {
                factory.Dispose();
            }
        }

        [Test]
        public void WithoutADatabaseUrl_NoPostgresServicesAreRegistered()
        {
            using var factory = new WebApplicationFactory<Program>();
            using var scope = factory.Services.CreateScope();

            Assert.That(scope.ServiceProvider.GetService<NpgsqlDataSource>(), Is.Null);
            Assert.That(scope.ServiceProvider.GetService<MigrationRunner>(), Is.Null);
            Assert.That(scope.ServiceProvider.GetServices<IHostedService>().OfType<MigrationHostedService>(),
                Is.Empty);
        }
    }
}

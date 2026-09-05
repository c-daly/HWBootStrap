using HexWars.NetServer.Persistence;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace HexWars.NetServer.Tests.Fixtures
{
    /// <summary>
    /// The real server, composed as production composes it, with only the three edges a test cannot own
    /// swapped out: Steam, the clock and (by default) storage. Everything between the socket and those
    /// edges - configuration binding, options validation, the rate limiter, routing, the endpoints, the
    /// credential service - is the shipped code, which is the point: an endpoint test that reimplemented
    /// the pipeline would pass while the deployed server refused every request.
    ///
    /// DATABASE_URL is set but never connected to. The Postgres branch of composition is what registers
    /// <see cref="HexWars.NetServer.Auth.IMatchCredentialService"/>, so a host without it cannot issue a
    /// join credential at all; the startup migration that would (correctly) refuse to reach that address
    /// is removed instead, and the store behind it is replaced.
    ///
    /// Each test should build its own factory. The rate limiter partitions on the client address, which
    /// under the test server is the same for every request, so two tests sharing a host would share a
    /// window and the second one would start seeing 429s that have nothing to do with it.
    /// </summary>
    public sealed class SteamServerFactory : WebApplicationFactory<Program>
    {
        public const string PublicBaseUrl = "https://match.test";
        public const string WebsocketUrl = "wss://match.test/ws/v2";
        public const string BuildId = "test-build";
        public const int JoinTokenTtlSeconds = 900;

        /// <summary>An address that parses as a database and refuses to connect. Port 1 on loopback is
        /// chosen so a stray connection attempt fails immediately rather than hanging on a DNS lookup.</summary>
        public const string UnreachableDatabaseUrl = "postgres://u:p@127.0.0.1:1/fake";

        /// <summary>A fixed instant, so an expiry assertion is arithmetic rather than a tolerance.</summary>
        public static readonly DateTimeOffset Start = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

        /// <summary>The flat environment the server binds. Mutate before the first request to change it.</summary>
        public Dictionary<string, string> Settings { get; } = new(StringComparer.Ordinal)
        {
            ["LOBBY_PROVIDER"] = "Steam",
            ["STEAM_APP_ID"] = "480000",
            ["STEAM_PUBLISHER_WEB_API_KEY"] = "test-key",
            ["DATABASE_URL"] = UnreachableDatabaseUrl,
            ["MATCH_PUBLIC_BASE_URL"] = PublicBaseUrl,
            ["MATCH_BUILD_ID"] = BuildId,
            ["MATCH_JOIN_TOKEN_TTL_SECONDS"] = "900",
        };

        /// <summary>The store the host uses unless <see cref="UsePostgres"/> is set. Write to it directly to
        /// arrange state a request could not reach, and read its WriteCount to assert a request wrote
        /// nothing.</summary>
        public InMemoryMatchStore Store { get; } = new();

        /// <summary>
        /// The decorator the host actually resolves, wrapping <see cref="Store"/>. It carries the seam a
        /// race needs: a hook that runs inside a seat lookup, so a test can end a match in the window
        /// between a join checking the status and issuing the credential.
        /// </summary>
        public CountingMatchStore Counting { get; }

        public SteamServerFactory() => Counting = new CountingMatchStore(Store);

        /// <summary>The scripted Steam API. Configure its tables before the first request.</summary>
        public FakeSteamWebApiClient Steam { get; } = FakeSteamWebApiClient.Ready();

        /// <summary>The clock every part of the host shares, including the credential service.</summary>
        public FakeTimeProvider Clock { get; } = new(Start);

        /// <summary>When set, the host keeps the real PostgresMatchStore and talks to <see cref=\"Database\"/>.</summary>
        public bool UsePostgres { get; private set; }

        /// <summary>The database behind a <see cref="PostgresAsync"/> factory, for asserting on rows.</summary>
        public PostgresTestDatabase? Database { get; private set; }

        /// <summary>
        /// A factory whose host writes to a real, migrated Postgres. Used by the one end-to-end test that
        /// has to prove the rows the endpoints claim to write are actually there; every other test uses
        /// the in-memory store, which has the same semantics and does not need Docker.
        /// </summary>
        public static async Task<SteamServerFactory> PostgresAsync()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            await database.ResetAsync();
            await database.ApplyMigrationsAsync();

            var factory = new SteamServerFactory { UsePostgres = true, Database = database };
            factory.Settings["DATABASE_URL"] = database.DatabaseUrl;
            return factory;
        }

        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            builder.UseEnvironment("Development");
            foreach (KeyValuePair<string, string> setting in Settings)
            {
                builder.UseSetting(setting.Key, setting.Value);
            }

            builder.ConfigureServices(services =>
            {
                // The schema is applied by the fixture (Postgres) or is not needed at all (in-memory);
                // either way the startup migration must not run. Everything else the Postgres branch
                // registered stays: NpgsqlDataSource and MigrationRunner are lazy, and nothing resolves
                // them once this is gone.
                ServiceDescriptor? migration =
                    services.FirstOrDefault(d => d.ImplementationType == typeof(MigrationHostedService));
                if (migration is not null) services.Remove(migration);

                if (!UsePostgres)
                {
                    services.RemoveAll<IMatchStore>();
                    services.AddSingleton<IMatchStore>(Counting);
                }

                services.RemoveAll<ISteamWebApiClient>();
                services.AddSingleton<ISteamWebApiClient>(Steam);

                services.RemoveAll<TimeProvider>();
                services.AddSingleton<TimeProvider>(Clock);
            });
        }
    }
}

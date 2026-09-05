using System.Net;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Npgsql;

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

        long _requestBodyBytesRead;

        /// <summary>
        /// Bytes pulled off request bodies since this host started. It is the only way to tell a body that
        /// was refused for being too large from one that was read in full and then measured, which is the
        /// difference between a bound and a comment claiming there is one.
        /// </summary>
        public long RequestBodyBytesRead => Interlocked.Read(ref _requestBodyBytesRead);

        /// <summary>
        /// The peer address every request appears to arrive from. The test server sets none, so without
        /// this a known-network check has nothing to judge and any test of the forwarded-header trust
        /// boundary would pass for the wrong reason.
        /// </summary>
        public IPAddress? RemoteIpAddress { get; set; }

        /// <summary>When set, receives everything the host logs. Assign before the first request.</summary>
        public ILoggerProvider? Logging { get; set; }

        /// <summary>
        /// The hosting environment name. Development by default, because that is what lets the suite reach
        /// these endpoints over the plaintext transport of the test host; set Production for the rules
        /// that only apply there.
        /// </summary>
        public string Environment { get; init; } = "Development";

        /// <summary>Wraps every request body in a counting stream, ahead of the whole application
        /// pipeline.</summary>
        sealed class CountingBodyStartupFilter(SteamServerFactory owner) : IStartupFilter
        {
            public Action<IApplicationBuilder> Configure(Action<IApplicationBuilder> next) => app =>
            {
                app.Use(async (context, following) =>
                {
                    context.Request.Body = new CountingStream(context.Request.Body, owner);

                    // The test server leaves the connection with no address, which would make every
                    // known-network check trivially fail. Set before the forwarded-headers middleware, so
                    // what a test says is the peer address is what that middleware judges.
                    if (owner.RemoteIpAddress is not null)
                    {
                        context.Connection.RemoteIpAddress = owner.RemoteIpAddress;
                        context.Connection.RemotePort = 51000;
                    }

                    await following();
                });

                next(app);
            };
        }

        /// <summary>A read-through stream that counts. Only the read paths are overridden: nothing writes
        /// to a request body, and a write path nobody exercises could only ever be wrong.</summary>
        sealed class CountingStream(Stream inner, SteamServerFactory owner) : Stream
        {
            public override bool CanRead => inner.CanRead;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => inner.Length;

            public override long Position
            {
                get => inner.Position;
                set => throw new NotSupportedException();
            }

            public override int Read(byte[] buffer, int offset, int count) =>
                Count(inner.Read(buffer, offset, count));

            public override async ValueTask<int> ReadAsync(
                Memory<byte> buffer, CancellationToken cancellationToken = default) =>
                Count(await inner.ReadAsync(buffer, cancellationToken));

            public override async Task<int> ReadAsync(
                byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
                Count(await inner.ReadAsync(buffer.AsMemory(offset, count), cancellationToken));

            int Count(int read)
            {
                if (read > 0) Interlocked.Add(ref owner._requestBodyBytesRead, read);
                return read;
            }

            public override void Flush() => inner.Flush();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        }

        /// <summary>The scripted Steam API. Configure its tables before the first request.</summary>
        public FakeSteamWebApiClient Steam { get; } = FakeSteamWebApiClient.Ready();

        /// <summary>The clock every part of the host shares, including the credential service.</summary>
        public FakeTimeProvider Clock { get; } = new(Start);

        /// <summary>When set, the host keeps the real PostgresMatchStore and talks to <see cref=\"Database\"/>.</summary>
        public bool UsePostgres { get; private set; }

        /// <summary>
        /// When set, the host gets a real PostgresMatchStore pointed at a port nothing is listening on.
        ///
        /// The in-memory store can be made to throw, but only exceptions a test chose. This produces the
        /// real ones - the Npgsql failures an outage actually raises - which is the only way to show an
        /// endpoint turns them into the fixed error body rather than leaking a connection string or a
        /// stack trace into the response.
        /// </summary>
        public bool UseUnreachableDatabase { get; init; }

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
            builder.UseEnvironment(Environment);
            foreach (KeyValuePair<string, string> setting in Settings)
            {
                builder.UseSetting(setting.Key, setting.Value);
            }

            if (Logging is not null)
            {
                builder.ConfigureLogging(logging => logging.AddProvider(Logging));
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

                if (UseUnreachableDatabase)
                {
                    // A short connect timeout so a refused connection is an immediate failure rather than
                    // a test that spends its life waiting for one.
                    services.RemoveAll<IMatchStore>();
                    services.AddSingleton<IMatchStore>(provider => new PostgresMatchStore(
                        NpgsqlDataSource.Create(
                            "Host=127.0.0.1;Port=1;Username=u;Password=p;Database=fake;Timeout=1"),
                        provider.GetRequiredService<ILogger<PostgresMatchStore>>()));
                }
                else if (!UsePostgres)
                {
                    services.RemoveAll<IMatchStore>();
                    services.AddSingleton<IMatchStore>(Counting);
                }

                services.RemoveAll<ISteamWebApiClient>();
                services.AddSingleton<ISteamWebApiClient>(Steam);

                services.RemoveAll<TimeProvider>();
                services.AddSingleton<TimeProvider>(Clock);

                services.AddSingleton<IStartupFilter>(new CountingBodyStartupFilter(this));
            });
        }
    }
}

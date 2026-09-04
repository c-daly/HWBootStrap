using HexWars.NetServer.Configuration;
using Microsoft.Extensions.Options;
using Npgsql;

namespace HexWars.NetServer.Persistence
{
    /// <summary>
    /// Registers the Postgres side of the server. Deliberately opt-in: the composition root calls this only
    /// when DATABASE_URL is set, so the legacy deployment that has never had a database keeps starting with
    /// no Npgsql in the container at all.
    /// </summary>
    public static class PostgresRegistration
    {
        /// <summary>Pool sized for a small match host, not a reporting workload: connections are held only
        /// for the length of a journal append. The short Timeout matters more than the pool size — a match
        /// that cannot commit within five seconds should tell the player to retry, not hang the socket.</summary>
        public const int MaxPoolSize = 20;
        public const int ConnectTimeoutSeconds = 5;
        public const int CommandTimeoutSeconds = 10;
        public const string ApplicationName = "hexwars-match";

        public static IServiceCollection AddHexWarsPostgres(this IServiceCollection services)
        {
            services.AddSingleton(serviceProvider =>
            {
                var options = serviceProvider.GetRequiredService<IOptions<MatchHostingOptions>>().Value;
                var builder = new NpgsqlDataSourceBuilder(DatabaseUrl.ToNpgsqlConnectionString(options.DatabaseUrl));
                builder.ConnectionStringBuilder.MaxPoolSize = MaxPoolSize;
                builder.ConnectionStringBuilder.Timeout = ConnectTimeoutSeconds;
                builder.ConnectionStringBuilder.CommandTimeout = CommandTimeoutSeconds;
                builder.ConnectionStringBuilder.ApplicationName = ApplicationName;
                builder.UseLoggerFactory(serviceProvider.GetRequiredService<ILoggerFactory>());
                return builder.Build();
            });

            services.AddSingleton<MigrationRunner>();
            services.AddSingleton<IHostedService, MigrationHostedService>();
            return services;
        }
    }

    /// <summary>
    /// Migrates on startup and fails the host if it cannot. Serving traffic against a schema we could not
    /// verify is worse than not serving at all: it would accept players into matches it cannot journal.
    ///
    /// A test that swaps in a fake store and wants nothing to do with a real database removes this by type:
    /// <c>services.Remove(services.Single(d =&gt; d.ImplementationType == typeof(MigrationHostedService)))</c>.
    /// </summary>
    public sealed class MigrationHostedService(MigrationRunner runner, ILogger<MigrationHostedService> logger)
        : IHostedService
    {
        public async Task StartAsync(CancellationToken cancellationToken)
        {
            IReadOnlyList<string> applied = await runner.ApplyAsync(cancellationToken).ConfigureAwait(false);
            if (applied.Count > 0)
                logger.LogInformation("Applied {Count} schema migration(s): {Versions}",
                    applied.Count, string.Join(", ", applied));
        }

        public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    }
}

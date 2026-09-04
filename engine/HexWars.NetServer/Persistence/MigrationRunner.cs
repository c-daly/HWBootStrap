using System.Reflection;
using Npgsql;

namespace HexWars.NetServer.Persistence
{
    /// <summary>
    /// Applies the embedded schema migrations, once, in order, whoever gets there first.
    ///
    /// A match host scales to more than one instance and every instance runs this at startup, so the whole
    /// design point is the session-level advisory lock: racing instances queue behind it and only the first
    /// through does any work. Each migration commits in its own transaction together with its ledger row,
    /// so a crash halfway through a series leaves the applied ones recorded and the rest still pending.
    /// </summary>
    public sealed class MigrationRunner(NpgsqlDataSource dataSource, ILogger<MigrationRunner> logger)
    {
        /// <summary>Arbitrary but fixed: every HexWars instance must pick the same number or the lock is
        /// pointless. Changing it would let an old and a new build migrate concurrently.</summary>
        public const long AdvisoryLockKey = 7331001;

        const string ResourcePrefix = "HexWars.NetServer.Migrations.";

        const string CreateLedger =
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            + "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())";

        /// <summary>The migrations compiled into this assembly, ordered by version. Ordinal ordering is why
        /// the files are numbered: 002 must sort after 001 as text, not just as intent.</summary>
        public static IReadOnlyList<(string Version, string Sql)> EmbeddedMigrations()
        {
            Assembly assembly = typeof(MigrationRunner).Assembly;
            var migrations = new List<(string Version, string Sql)>();

            foreach (string resource in assembly.GetManifestResourceNames())
            {
                if (!resource.StartsWith(ResourcePrefix, StringComparison.Ordinal)) continue;

                string version = Path.GetFileNameWithoutExtension(resource.Substring(ResourcePrefix.Length));
                using Stream stream = assembly.GetManifestResourceStream(resource)
                    ?? throw new InvalidOperationException("Migration resource " + resource + " could not be opened.");
                using var reader = new StreamReader(stream);
                migrations.Add((version, reader.ReadToEnd()));
            }

            migrations.Sort((left, right) => string.CompareOrdinal(left.Version, right.Version));
            return migrations;
        }

        /// <summary>Versions this database has not seen yet. Creates the ledger table if it is missing so a
        /// readiness probe can ask the question on a brand new database without failing.</summary>
        public async Task<IReadOnlyList<string>> PendingAsync(CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            await ExecuteAsync(connection, CreateLedger, ct).ConfigureAwait(false);

            HashSet<string> applied = await AppliedVersionsAsync(connection, ct).ConfigureAwait(false);
            return EmbeddedMigrations()
                .Select(migration => migration.Version)
                .Where(version => !applied.Contains(version))
                .ToArray();
        }

        /// <summary>Brings the database up to date and returns what THIS call applied. A second caller, or a
        /// second instance, gets an empty list rather than an error.</summary>
        public async Task<IReadOnlyList<string>> ApplyAsync(CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);

            // Taken BEFORE the ledger is created: two instances racing on CREATE TABLE IF NOT EXISTS can
            // otherwise collide in the system catalogue, which is a unique violation rather than a no-op.
            await ExecuteAsync(connection, "SELECT pg_advisory_lock(" + AdvisoryLockKey + ")", ct).ConfigureAwait(false);
            try
            {
                await ExecuteAsync(connection, CreateLedger, ct).ConfigureAwait(false);
                HashSet<string> applied = await AppliedVersionsAsync(connection, ct).ConfigureAwait(false);

                var appliedNow = new List<string>();
                foreach ((string version, string sql) in EmbeddedMigrations())
                {
                    if (applied.Contains(version)) continue;

                    await using NpgsqlTransaction transaction =
                        await connection.BeginTransactionAsync(ct).ConfigureAwait(false);

                    await using (var migration = new NpgsqlCommand(sql, connection, transaction))
                    {
                        await migration.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
                    }

                    await using (var record = new NpgsqlCommand(
                        "INSERT INTO schema_migrations (version) VALUES (@version)", connection, transaction))
                    {
                        record.Parameters.AddWithValue("version", version);
                        await record.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
                    }

                    await transaction.CommitAsync(ct).ConfigureAwait(false);
                    appliedNow.Add(version);
                    logger.LogInformation("Applied schema migration {Version}", version);
                }

                if (appliedNow.Count == 0) logger.LogInformation("Database schema is already current.");
                return appliedNow;
            }
            finally
            {
                // Not the caller token: the lock must be released even when the migration was cancelled.
                // Closing the connection would drop it anyway, but leaving it held until then blocks peers.
                await ExecuteAsync(connection, "SELECT pg_advisory_unlock(" + AdvisoryLockKey + ")",
                    CancellationToken.None).ConfigureAwait(false);
            }
        }

        static async Task<HashSet<string>> AppliedVersionsAsync(NpgsqlConnection connection, CancellationToken ct)
        {
            var applied = new HashSet<string>(StringComparer.Ordinal);
            await using var command = new NpgsqlCommand("SELECT version FROM schema_migrations", connection);
            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            while (await reader.ReadAsync(ct).ConfigureAwait(false)) applied.Add(reader.GetString(0));
            return applied;
        }

        static async Task ExecuteAsync(NpgsqlConnection connection, string sql, CancellationToken ct)
        {
            await using var command = new NpgsqlCommand(sql, connection);
            await command.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
        }
    }
}

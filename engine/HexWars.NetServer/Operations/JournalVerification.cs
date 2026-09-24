using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Steam;
using Microsoft.Extensions.Logging.Abstractions;
using Npgsql;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// The read-only answer to one question: can this build host the matches in that database.
    ///
    /// It exists because the only other tool that replayed real journals could not be pointed at real
    /// data. selftest-durable refuses any database whose name is not marked disposable, and if the name
    /// were made disposable it drops the public schema before it starts - so the one procedure that most
    /// needs to run against a restored backup was the one procedure that would destroy it.
    ///
    /// So this writes nothing, and does not rely on being careful about it. The connection asks Postgres
    /// for read-only sessions, which turns any write anywhere below - a migration, a heal, a stray UPDATE
    /// added later - into an error from the server rather than into a change to somebody backup.
    /// </summary>
    public static class JournalVerification
    {
        /// <summary>The database to read. Separate from DATABASE_URL so that pointing this at a restored
        /// copy does not mean editing the variable the service itself runs on.</summary>
        public const string DatabaseVariable = "HEXWARS_VERIFY_DATABASE_URL";

        public const string PassPrefix = "VERIFY-JOURNALS PASS";
        public const string FailPrefix = "VERIFY-JOURNALS FAIL";

        /// <summary>Everything Postgres needs to refuse a write on our behalf. Applied as a session option
        /// rather than per transaction, so it covers every command on the connection whether or not the
        /// code that issues it remembered to open a transaction.</summary>
        internal const string ReadOnlyOptions = "-c default_transaction_read_only=on";

        /// <summary>Exit codes: 0 every journal replays, 1 at least one does not, 2 it could not look.</summary>
        public static async Task<int> RunAsync(string[] args, TextWriter output)
        {
            ArgumentNullException.ThrowIfNull(args);
            ArgumentNullException.ThrowIfNull(output);

            bool openOnly = args.Any(a => string.Equals(a, "--open-only", StringComparison.Ordinal));

            string? databaseUrl = Environment.GetEnvironmentVariable(DatabaseVariable);
            if (string.IsNullOrWhiteSpace(databaseUrl))
                databaseUrl = Environment.GetEnvironmentVariable("DATABASE_URL");

            if (string.IsNullOrWhiteSpace(databaseUrl))
            {
                output.WriteLine("VERIFY-JOURNALS CONFIGURATION: set " + DatabaseVariable + " or DATABASE_URL");
                return 2;
            }

            NpgsqlDataSource source;
            try
            {
                source = ReadOnlySource(databaseUrl);
            }
            catch (Exception malformed)
            {
                output.WriteLine("VERIFY-JOURNALS CONFIGURATION: that is not a database address");
                output.WriteLine("  " + SteamLogRedaction.Redact(malformed.Message));
                return 2;
            }

            await using (source)
            {
                return await VerifyAsync(source, openOnly, output, CancellationToken.None)
                    .ConfigureAwait(false);
            }
        }

        /// <summary>A data source that cannot write, whatever is asked of it.</summary>
        internal static NpgsqlDataSource ReadOnlySource(string databaseUrl)
        {
            var builder = new NpgsqlDataSourceBuilder(DatabaseUrl.ToNpgsqlConnectionString(databaseUrl));
            builder.ConnectionStringBuilder.Options = ReadOnlyOptions;
            builder.ConnectionStringBuilder.ApplicationName = "hexwars-verify-journals";

            return builder.Build();
        }

        internal static async Task<int> VerifyAsync(
            NpgsqlDataSource source, bool openOnly, TextWriter output, CancellationToken ct)
        {
            IReadOnlyList<string> missing;
            try
            {
                missing = await PendingMigrationsAsync(source, ct).ConfigureAwait(false);
            }
            catch (Exception unreachable)
            {
                output.WriteLine("VERIFY-JOURNALS CONFIGURATION: the database could not be read");
                output.WriteLine("  " + SteamLogRedaction.Redact(unreachable.Message));
                return 2;
            }

            if (missing.Count > 0)
            {
                // Refused rather than attempted. A journal read against a schema this build does not
                // recognise would report failures that are about the schema, and an operator acting on
                // them would go looking at matches that are perfectly intact.
                output.WriteLine("VERIFY-JOURNALS CONFIGURATION: the schema is behind this build");
                output.WriteLine("  not applied: " + string.Join(", ", missing));
                return 2;
            }

            var store = new PostgresMatchStore(source, NullLogger<PostgresMatchStore>.Instance);

            IReadOnlyList<(Guid MatchId, string Status)> matches;
            try
            {
                matches = await ListAsync(source, openOnly, ct).ConfigureAwait(false);
            }
            catch (Exception unreadable)
            {
                output.WriteLine("VERIFY-JOURNALS CONFIGURATION: the matches could not be listed");
                output.WriteLine("  " + SteamLogRedaction.Redact(unreadable.Message));
                return 2;
            }

            output.WriteLine("VERIFY-JOURNALS SCOPE " + (openOnly ? "open matches" : "all retained matches"));
            return await VerifyMatchesAsync(store, matches, output, ct).ConfigureAwait(false);
        }

        internal static async Task<int> VerifyMatchesAsync(
            IMatchStore store, IReadOnlyList<(Guid MatchId, string Status)> matches,
            TextWriter output, CancellationToken ct)
        {
            var refused = 0;

            foreach ((Guid matchId, string status) in matches)
            {
                MatchJournal? journal;
                try
                {
                    journal = await store.LoadJournalAsync(matchId, ct).ConfigureAwait(false);
                }
                catch (Exception unreadable)
                {
                    output.WriteLine(Line(matchId, status, 0, "UNREADABLE",
                        SteamLogRedaction.Redact(unreadable.Message)));
                    output.WriteLine("VERIFY-JOURNALS INCOMPLETE: a journal could not be read; retry verification");
                    return 2;
                }

                if (journal is null)
                {
                    output.WriteLine(Line(matchId, status, 0, "MISSING", "the journal is gone"));
                    output.WriteLine("VERIFY-JOURNALS INCOMPLETE: the match set changed; retry on a stable copy");
                    return 2;
                }

                int commands = journal.Commands.Count;

                try
                {
                    JournalVerifier.Verify(journal, ProtocolContractVersion);
                    output.WriteLine(Line(matchId, status, commands, "OK", null));
                }
                catch (MatchRecoveryException refusal)
                {
                    refused++;
                    output.WriteLine(
                        Line(matchId, status, commands, refusal.Failure.ToString(), refusal.Detail));
                }
            }

            int total = matches.Count;

            if (refused == 0)
            {
                output.WriteLine(PassPrefix + " " + total + "/" + total);
                return 0;
            }

            output.WriteLine(FailPrefix + " " + refused + "/" + total);
            return 1;
        }

        /// <summary>The protocol this build speaks, read from the contract rather than from configuration:
        /// this verb has no service to be configured for, and the contract is what a journal is judged
        /// against anyway.</summary>
        static int ProtocolContractVersion => Configuration.ProtocolContract.Version;

        static string Line(Guid matchId, string status, int commands, string result, string? detail) =>
            matchId.ToString() + " " + status + " " + commands + " " + result
            + (detail is null ? string.Empty : " " + detail);

        /// <summary>
        /// Migrations this build carries that the database has not recorded.
        ///
        /// Written here rather than reusing MigrationRunner.PendingAsync, which creates the ledger table
        /// when it is missing. That is the right thing for a readiness probe on a live database and
        /// exactly the wrong thing here: it is a write, on somebody restored backup, from a verb whose
        /// whole promise is that it makes none.
        /// </summary>
        internal static async Task<IReadOnlyList<string>> PendingMigrationsAsync(
            NpgsqlDataSource source, CancellationToken ct)
        {
            bool ledger = await ReadOnlyQueryAsync(
                source,
                "SELECT to_regclass('public.schema_migrations') IS NOT NULL",
                async (command, token) =>
                    await command.ExecuteScalarAsync(token).ConfigureAwait(false) is true,
                ct).ConfigureAwait(false);

            // No ledger at all is not an empty ledger: it is a database that has never been migrated, and
            // every migration this build carries is missing from it.
            if (!ledger) return MigrationRunner.EmbeddedMigrations().Select(m => m.Version).ToArray();

            HashSet<string> applied = await ReadOnlyQueryAsync(
                source,
                "SELECT version FROM schema_migrations",
                async (command, token) =>
                {
                    var seen = new HashSet<string>(StringComparer.Ordinal);
                    await using NpgsqlDataReader rows =
                        await command.ExecuteReaderAsync(token).ConfigureAwait(false);

                    while (await rows.ReadAsync(token).ConfigureAwait(false)) seen.Add(rows.GetString(0));
                    return seen;
                },
                ct).ConfigureAwait(false);

            // Only what is missing. A version in the ledger this build has never heard of means the
            // database is AHEAD, which is a rollback and is fine: the older build simply does not use it.
            return MigrationRunner.EmbeddedMigrations()
                .Select(migration => migration.Version)
                .Where(version => !applied.Contains(version))
                .ToArray();
        }

        static Task<IReadOnlyList<(Guid MatchId, string Status)>> ListAsync(
            NpgsqlDataSource source, bool openOnly, CancellationToken ct) =>
            ReadOnlyQueryAsync(
                source,
                openOnly
                    ? "SELECT match_id, status FROM matches WHERE status IN ('waiting','active') ORDER BY created_at"
                    : "SELECT match_id, status FROM matches ORDER BY created_at",
                async (command, token) =>
                {
                    var matches = new List<(Guid, string)>();
                    await using NpgsqlDataReader rows =
                        await command.ExecuteReaderAsync(token).ConfigureAwait(false);

                    while (await rows.ReadAsync(token).ConfigureAwait(false))
                        matches.Add((rows.GetGuid(0), rows.GetString(1)));

                    return (IReadOnlyList<(Guid, string)>)matches;
                },
                ct);

        /// <summary>
        /// The only place this verb runs SQL of its own, and the only reason its promise is worth
        /// anything.
        ///
        /// The session default the connection asks for is a DEFAULT: one SET turns it off, and the next
        /// statement can write. An explicit READ ONLY transaction cannot be turned off from inside itself,
        /// so every statement that runs in here is refused a write by the server no matter what ran
        /// before it. The transaction is rolled back rather than committed, because there is by
        /// construction nothing to commit.
        /// </summary>
        internal static async Task<T> ReadOnlyQueryAsync<T>(
            NpgsqlDataSource source,
            string sql,
            Func<NpgsqlCommand, CancellationToken, Task<T>> read,
            CancellationToken ct)
        {
            await using NpgsqlConnection connection =
                await source.OpenConnectionAsync(ct).ConfigureAwait(false);
            await using NpgsqlTransaction transaction =
                await connection.BeginTransactionAsync(ct).ConfigureAwait(false);

            await using (NpgsqlCommand mark = connection.CreateCommand())
            {
                mark.Transaction = transaction;
                mark.CommandText = "SET TRANSACTION READ ONLY";
                await mark.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
            }

            await using NpgsqlCommand command = connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText = sql;

            T answer = await read(command, ct).ConfigureAwait(false);

            await transaction.RollbackAsync(ct).ConfigureAwait(false);
            return answer;
        }
    }
}

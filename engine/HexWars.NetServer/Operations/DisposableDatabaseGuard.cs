using HexWars.NetServer.Persistence;
using Npgsql;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// Decides whether a database is one a destructive self-test is allowed to destroy.
    ///
    /// This is a deliberate copy of the rule the test fixture enforces, not a shared implementation. The
    /// server assembly cannot reference the test project, and the alternative - trusting an operator who
    /// typed a URL - is not an alternative: the durable self-test opens by dropping and recreating the
    /// public schema of whatever it is given, and a pasted URL for the wrong database reads exactly like
    /// one for the right database. The two copies are kept identical on purpose, and each is tested where
    /// it lives.
    ///
    /// A database qualifies one of two ways. Either its name says what it is - anything containing "test"
    /// - or somebody named it explicitly in the confirmation variable. The confirmation is by name rather
    /// than a bare yes: a stale yes left in a shell would authorise the next database that came along,
    /// while a stale name only ever authorises the same one.
    ///
    /// Pure and static, with the environment as a parameter, so the rule can be exercised without a
    /// database and without mutating the environment of the process asking.
    /// </summary>
    internal static class DisposableDatabaseGuard
    {
        /// <summary>Set to the exact database name to authorise a database whose name does not say it is a
        /// test database.</summary>
        public const string ConfirmationEnvironmentVariable = "HEXWARS_TEST_DATABASE_DISPOSABLE";

        /// <summary>What a database name has to contain to authorise itself.</summary>
        public const string TestNameMarker = "test";

        /// <summary>True when this target may be dropped and recreated. <paramref name="reason"/> always
        /// explains the answer and never contains a username or a password, because it is written to a
        /// console that may well be a CI log.</summary>
        public static bool IsDisposable(
            string databaseUrlOrConnectionString, Func<string, string?> env, out string reason)
        {
            ArgumentNullException.ThrowIfNull(env);

            string? database = DatabaseName(databaseUrlOrConnectionString);

            if (database is null)
            {
                reason = "that value does not parse into a connection that names a database, so there is "
                    + "nothing here that could be checked.";
                return false;
            }

            if (database.Contains(TestNameMarker, StringComparison.OrdinalIgnoreCase))
            {
                reason = "database \"" + database + "\" is named as a test database.";
                return true;
            }

            if (string.Equals(env(ConfirmationEnvironmentVariable), database, StringComparison.Ordinal))
            {
                reason = "database \"" + database + "\" is confirmed disposable by name in "
                    + ConfirmationEnvironmentVariable + ".";
                return true;
            }

            reason = "database \"" + database + "\" is not marked disposable. This self-test DROPs and "
                + "recreates the public schema of whatever it is given, so it will only do that to a "
                + "database whose name contains \"" + TestNameMarker + "\", or to one named explicitly in "
                + ConfirmationEnvironmentVariable + "=" + database + ".";
            return false;
        }

        /// <summary>The database a target actually points at, or null when it names none or cannot be
        /// parsed at all. Npgsql does the parsing, so this is the same database a connection would open
        /// rather than whatever a substring search guessed at.</summary>
        public static string? DatabaseName(string databaseUrlOrConnectionString)
        {
            if (string.IsNullOrWhiteSpace(databaseUrlOrConnectionString)) return null;

            try
            {
                var parts = new NpgsqlConnectionStringBuilder(
                    DatabaseUrl.ToNpgsqlConnectionString(databaseUrlOrConnectionString));
                return string.IsNullOrEmpty(parts.Database) ? null : parts.Database;
            }
            catch (Exception ex) when (ex is FormatException or ArgumentException)
            {
                return null;
            }
        }
    }
}

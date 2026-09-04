using System.Globalization;
using Npgsql;

namespace HexWars.NetServer.Persistence
{
    /// <summary>
    /// Translates the DATABASE_URL that hosting platforms hand out (Render, Heroku, Fly all give a
    /// postgres:// URI) into the connection string Npgsql wants, and describes it without leaking
    /// credentials.
    ///
    /// Both forms end up in an <see cref="NpgsqlConnectionStringBuilder"/>, which is the only component
    /// that agrees with Npgsql about what a valid connection string is. A hand-rolled semicolon split does
    /// not: it cuts a quoted password in half, so Password="foo;Port=SECRET;X=bar" used to be described as
    /// host:SECRET/database and printed in the startup log. Letting Npgsql parse also means an invalid
    /// port or an unsupported keyword fails validation at boot instead of at the first connection attempt.
    /// </summary>
    public static class DatabaseUrl
    {
        const int DefaultPort = 5432;

        /// <summary>Accepts a postgres:// or postgresql:// URI, or an Npgsql key=value connection string
        /// that actually names a host and a database, and returns the normalised connection string.
        /// Anything Npgsql would reject throws <see cref="FormatException"/>, so a bad value fails startup
        /// instead of letting the server come up without a reachable database.</summary>
        public static string ToNpgsqlConnectionString(string databaseUrl) => Build(databaseUrl).ConnectionString;

        /// <summary>host:port/database, for logs and the environment report. Never a username, never a
        /// password, and never throws: an unusable value is reported as invalid so a health report still
        /// renders.</summary>
        public static string DescribeTarget(string databaseUrl)
        {
            string raw = (databaseUrl ?? string.Empty).Trim();
            if (raw.Length == 0) return "none";
            try
            {
                var builder = Build(raw);
                string target = builder.Host + ":" + builder.Port.ToString(CultureInfo.InvariantCulture);
                return string.IsNullOrEmpty(builder.Database) ? target : target + "/" + builder.Database;
            }
            catch (FormatException)
            {
                return "invalid";
            }
        }

        static NpgsqlConnectionStringBuilder Build(string databaseUrl)
        {
            string raw = (databaseUrl ?? string.Empty).Trim();
            if (raw.Length == 0) throw new FormatException("The database URL is empty.");
            // The SCHEME PREFIX selects the format, not the mere presence of a scheme separator: a key=value
            // string may legitimately carry :// inside a value, such as a password or a search_path option.
            if (IsPostgresUri(raw)) return FromUri(raw);
            if (raw.Contains("=", StringComparison.Ordinal)) return FromKeyValue(raw);
            throw new FormatException("The database URL is neither a postgres:// URL nor a key=value connection string.");
        }

        static bool IsPostgresUri(string raw) =>
            raw.StartsWith("postgres://", StringComparison.OrdinalIgnoreCase)
            || raw.StartsWith("postgresql://", StringComparison.OrdinalIgnoreCase);

        /// <summary>Npgsql is the parser. Its failure messages name the offending KEYWORD and never echo a
        /// value, so quoting one here cannot leak a password.</summary>
        static NpgsqlConnectionStringBuilder FromKeyValue(string raw)
        {
            NpgsqlConnectionStringBuilder builder;
            try
            {
                builder = new NpgsqlConnectionStringBuilder(raw);
            }
            catch (Exception ex) when (ex is ArgumentException or KeyNotFoundException or FormatException)
            {
                throw new FormatException("DATABASE_URL is not a valid Npgsql connection string: " + ex.Message);
            }

            if (string.IsNullOrEmpty(builder.Host) || string.IsNullOrEmpty(builder.Database))
                throw new FormatException("DATABASE_URL key=value form must include Host and Database");

            return builder;
        }

        static NpgsqlConnectionStringBuilder FromUri(string raw)
        {
            var parts = ParseUri(raw);
            var builder = new NpgsqlConnectionStringBuilder
            {
                Host = parts.Host,
                Port = parts.Port,
            };
            if (!string.IsNullOrEmpty(parts.Database)) builder.Database = parts.Database;
            if (!string.IsNullOrEmpty(parts.User)) builder.Username = parts.User;
            if (!string.IsNullOrEmpty(parts.Password)) builder.Password = parts.Password;

            if (parts.Query.TryGetValue("sslmode", out string? sslMode))
            {
                builder.SslMode = ParseSslMode(sslMode);
                // Managed Postgres (Render, Heroku) presents a chain the container does not trust;
                // sslmode=require means encrypt-but-do-not-verify. verify-ca and verify-full do verify.
                // Npgsql 8 folds that meaning into SslMode.Require itself and marks the flag obsolete, so
                // this keeps the emitted connection string explicit about the intent; it changes nothing.
#pragma warning disable CS0618
                if (builder.SslMode == SslMode.Require) builder.TrustServerCertificate = true;
#pragma warning restore CS0618
            }

            return builder;
        }

        /// <summary>libpq spells the modes with hyphens (verify-ca), Npgsql spells them as enum members
        /// (VerifyCA). An unrecognised mode is a configuration error, not something to silently ignore.</summary>
        static SslMode ParseSslMode(string raw)
        {
            string name = raw.Trim().Replace("-", string.Empty, StringComparison.Ordinal);
            if (!Enum.TryParse(name, ignoreCase: true, out SslMode mode) || !Enum.IsDefined(mode))
                throw new FormatException("The database URL has an unsupported sslmode.");
            return mode;
        }

        readonly record struct UriParts(string? User, string? Password, string Host, int Port, string? Database,
            IReadOnlyDictionary<string, string> Query);

        /// <summary>Hand-rolled rather than System.Uri: a postgres userinfo routinely carries
        /// percent-encoded reserved characters and we need the exact decoded password back.</summary>
        static UriParts ParseUri(string raw)
        {
            int schemeEnd = raw.IndexOf("://", StringComparison.Ordinal);
            string scheme = raw.Substring(0, schemeEnd).ToLowerInvariant();
            if (scheme != "postgres" && scheme != "postgresql")
                throw new FormatException("Unsupported database URL scheme.");

            string rest = raw.Substring(schemeEnd + 3);

            var query = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            int questionMark = rest.IndexOf("?", StringComparison.Ordinal);
            if (questionMark >= 0)
            {
                foreach (string pair in rest.Substring(questionMark + 1).Split("&", StringSplitOptions.RemoveEmptyEntries))
                {
                    int equals = pair.IndexOf("=", StringComparison.Ordinal);
                    if (equals <= 0) continue;
                    query[Uri.UnescapeDataString(pair.Substring(0, equals))] =
                        Uri.UnescapeDataString(pair.Substring(equals + 1));
                }
                rest = rest.Substring(0, questionMark);
            }

            string? database = null;
            int slash = rest.IndexOf("/", StringComparison.Ordinal);
            if (slash >= 0)
            {
                string path = rest.Substring(slash + 1);
                if (path.Length > 0) database = Uri.UnescapeDataString(path);
                rest = rest.Substring(0, slash);
            }

            string? user = null;
            string? password = null;
            int at = rest.LastIndexOf("@", StringComparison.Ordinal);
            if (at >= 0)
            {
                string userInfo = rest.Substring(0, at);
                rest = rest.Substring(at + 1);
                int colon = userInfo.IndexOf(":", StringComparison.Ordinal);
                if (colon >= 0)
                {
                    user = Uri.UnescapeDataString(userInfo.Substring(0, colon));
                    password = Uri.UnescapeDataString(userInfo.Substring(colon + 1));
                }
                else if (userInfo.Length > 0)
                {
                    user = Uri.UnescapeDataString(userInfo);
                }
            }

            string host;
            int port = DefaultPort;
            if (rest.StartsWith("[", StringComparison.Ordinal))
            {
                int close = rest.IndexOf("]", StringComparison.Ordinal);
                if (close < 0) throw new FormatException("Malformed IPv6 host in the database URL.");
                host = rest.Substring(1, close - 1);
                string tail = rest.Substring(close + 1);
                if (tail.StartsWith(":", StringComparison.Ordinal)) port = ParsePort(tail.Substring(1));
            }
            else
            {
                int colon = rest.LastIndexOf(":", StringComparison.Ordinal);
                if (colon >= 0)
                {
                    host = rest.Substring(0, colon);
                    port = ParsePort(rest.Substring(colon + 1));
                }
                else
                {
                    host = rest;
                }
            }

            if (host.Length == 0) throw new FormatException("The database URL has no host.");
            return new UriParts(user, password, host, port, database, query);
        }

        static int ParsePort(string raw)
        {
            if (!int.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out int port)
                || port <= 0 || port > 65535)
                throw new FormatException("The database URL has an invalid port.");
            return port;
        }
    }
}

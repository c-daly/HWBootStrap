using System.Globalization;
using System.Text;

namespace HexWars.NetServer.Persistence
{
    /// <summary>
    /// Translates the DATABASE_URL that hosting platforms hand out (Render, Heroku, Fly all give a
    /// postgres:// URI) into the Key=Value string Npgsql wants, and describes it without leaking
    /// credentials. Pure string handling: this file deliberately takes no Npgsql dependency, so the
    /// configuration layer can reject a bad connection string long before anything tries to connect.
    /// </summary>
    public static class DatabaseUrl
    {
        const int DefaultPort = 5432;

        /// <summary>Npgsql accepts Server as an alias for Host and DB as an alias for Database; a
        /// connection string that names neither a host nor a database cannot reach a server.</summary>
        static readonly string[] HostKeys = { "host", "server" };
        static readonly string[] DatabaseKeys = { "database", "db" };
        const string DoubleQuote = "\u0022";
        const string SingleQuote = "\u0027";

        /// <summary>Accepts a postgres:// or postgresql:// URI, or an Npgsql Key=Value string that actually
        /// names a host and a database (returned unchanged, trimmed). Anything else — including a stray
        /// "foo=bar" that merely contains an equals sign — throws <see cref="FormatException"/>, so a bad
        /// value fails startup instead of letting the server come up without a reachable database.</summary>
        public static string ToNpgsqlConnectionString(string databaseUrl)
        {
            string raw = (databaseUrl ?? string.Empty).Trim();
            if (raw.Length == 0) throw new FormatException("The database URL is empty.");
            if (raw.Contains("://", StringComparison.Ordinal)) return Compose(ParseUri(raw));
            if (raw.Contains("=", StringComparison.Ordinal))
            {
                RequireHostAndDatabase(ParseKeyValue(raw));
                return raw;
            }
            throw new FormatException("The database URL is neither a postgres:// URL nor a key=value connection string.");
        }

        static void RequireHostAndDatabase(IReadOnlyDictionary<string, string> pairs)
        {
            if (Lookup(pairs, HostKeys) is null || Lookup(pairs, DatabaseKeys) is null)
                throw new FormatException("DATABASE_URL key=value form must include Host and Database");
        }

        /// <summary>host:port/database, for logs and the environment report. Never credentials, and never
        /// throws: an unusable value is reported as invalid so a health report still renders.</summary>
        public static string DescribeTarget(string databaseUrl)
        {
            string raw = (databaseUrl ?? string.Empty).Trim();
            if (raw.Length == 0) return "none";
            try
            {
                if (raw.Contains("://", StringComparison.Ordinal))
                {
                    var parsed = ParseUri(raw);
                    return Format(parsed.Host, parsed.Port.ToString(CultureInfo.InvariantCulture), parsed.Database);
                }
                if (raw.Contains("=", StringComparison.Ordinal))
                {
                    var pairs = ParseKeyValue(raw);
                    RequireHostAndDatabase(pairs);
                    string host = Lookup(pairs, HostKeys) ?? "unknown";
                    string port = Lookup(pairs, "port") ?? DefaultPort.ToString(CultureInfo.InvariantCulture);
                    string? database = Lookup(pairs, DatabaseKeys);
                    return Format(host, port, database);
                }
            }
            catch (FormatException)
            {
                return "invalid";
            }
            return "invalid";
        }

        static string Format(string host, string port, string? database) =>
            string.IsNullOrEmpty(database) ? host + ":" + port : host + ":" + port + "/" + database;

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

        static string Compose(UriParts parts)
        {
            var sb = new StringBuilder();
            Append(sb, "Host", parts.Host);
            Append(sb, "Port", parts.Port.ToString(CultureInfo.InvariantCulture));
            if (!string.IsNullOrEmpty(parts.Database)) Append(sb, "Database", parts.Database!);
            if (!string.IsNullOrEmpty(parts.User)) Append(sb, "Username", parts.User!);
            if (!string.IsNullOrEmpty(parts.Password)) Append(sb, "Password", parts.Password!);

            if (parts.Query.TryGetValue("sslmode", out string? sslMode))
            {
                switch (sslMode.Trim().ToLowerInvariant())
                {
                    case "require":
                        Append(sb, "SSL Mode", "Require");
                        // Managed Postgres (Render, Heroku) presents a chain the container does not trust;
                        // sslmode=require means encrypt-but-do-not-verify. verify-ca/verify-full verify.
                        Append(sb, "Trust Server Certificate", "true");
                        break;
                    case "prefer": Append(sb, "SSL Mode", "Prefer"); break;
                    case "disable": Append(sb, "SSL Mode", "Disable"); break;
                    case "verify-ca": Append(sb, "SSL Mode", "VerifyCA"); break;
                    case "verify-full": Append(sb, "SSL Mode", "VerifyFull"); break;
                    default: break;   // unknown modes fall through to the Npgsql default rather than failing startup
                }
            }

            return sb.ToString();
        }

        static void Append(StringBuilder sb, string key, string value)
        {
            if (sb.Length > 0) sb.Append(";");
            sb.Append(key).Append("=").Append(Escape(value));
        }

        /// <summary>Npgsql (like every DbConnectionStringBuilder) reads a value verbatim unless it is
        /// quoted, so only wrap the values that would otherwise re-split the string.</summary>
        static string Escape(string value)
        {
            bool needsQuoting = value.Length == 0
                || value.Contains(";", StringComparison.Ordinal)
                || value.Contains("=", StringComparison.Ordinal)
                || value.Contains(SingleQuote, StringComparison.Ordinal)
                || value.Contains(DoubleQuote, StringComparison.Ordinal)
                || char.IsWhiteSpace(value[0])
                || char.IsWhiteSpace(value[value.Length - 1]);
            if (!needsQuoting) return value;
            return DoubleQuote + value.Replace(DoubleQuote, DoubleQuote + DoubleQuote) + DoubleQuote;
        }

        /// <summary>Strict on purpose: a segment that is not key=value means the whole string was never a
        /// connection string, and silently skipping it is how "foo=bar" used to pass validation.</summary>
        static Dictionary<string, string> ParseKeyValue(string raw)
        {
            var pairs = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (string part in raw.Split(";", StringSplitOptions.RemoveEmptyEntries))
            {
                string segment = part.Trim();
                if (segment.Length == 0) continue;
                int equals = segment.IndexOf("=", StringComparison.Ordinal);
                if (equals <= 0)
                    throw new FormatException("The database connection string has a segment that is not key=value.");
                string key = segment.Substring(0, equals).Trim();
                string value = segment.Substring(equals + 1).Trim();
                if (value.Length >= 2
                    && value.StartsWith(DoubleQuote, StringComparison.Ordinal)
                    && value.EndsWith(DoubleQuote, StringComparison.Ordinal))
                {
                    value = value.Substring(1, value.Length - 2)
                                 .Replace(DoubleQuote + DoubleQuote, DoubleQuote);
                }
                if (key.Length == 0)
                    throw new FormatException("The database connection string has a segment with an empty key.");
                pairs[key] = value;
            }
            return pairs;
        }

        static string? Lookup(IReadOnlyDictionary<string, string> pairs, params string[] keys)
        {
            foreach (string key in keys)
                if (pairs.TryGetValue(key, out string? value) && value.Length > 0) return value;
            return null;
        }
    }
}

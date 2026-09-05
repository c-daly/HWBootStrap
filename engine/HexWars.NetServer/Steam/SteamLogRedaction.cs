using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// Log hygiene for anything that has been near a Steam URL. Transport exception messages routinely
    /// quote the request they failed on, and that request carries the publisher key and the auth ticket,
    /// so every such string passes through here before it reaches a log sink or an exception detail.
    /// </summary>
    public static class SteamLogRedaction
    {
        const string Mask = "<redacted>";

        static readonly Regex SecretParameter = new(
            @"\b(key|ticket|token|access_token|credential|password|pwd)\s*=\s*[^&;\s<>]*",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

        // A connection string in URI form carries its password in the userinfo section rather than in
        // a named parameter, so the rule above cannot see it. Npgsql and the platform both quote the
        // target they failed to reach, and that target is DATABASE_URL.
        static readonly Regex UriCredentials = new(
            @"(?<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^\s/@:]+:[^\s/@]*@",
            RegexOptions.CultureInvariant | RegexOptions.Compiled);

        // Belt and braces: even an unrecognised parameter name cannot survive, because the whole query
        // string goes. Stopping at whitespace keeps the surrounding sentence readable.
        static readonly Regex QueryString = new(
            @"\?\S*", RegexOptions.CultureInvariant | RegexOptions.Compiled);

        public static string Redact(string text)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;

            var masked = SecretParameter.Replace(text, m => m.Groups[1].Value + "=" + Mask);
            masked = UriCredentials.Replace(masked, m => m.Groups["scheme"].Value + Mask + "@");
            return QueryString.Replace(masked, "?" + Mask);
        }

        /// <summary>
        /// An exception as a log line, with everything it might be quoting taken out of it.
        ///
        /// It exists because passing an exception object to a logger hands the sink the whole thing -
        /// message, inner messages and all - and the exceptions that reach the outer catch of an endpoint are
        /// precisely the ones raised by code that was holding a secret when it failed. A transport error
        /// quotes the URL it could not reach, which for the Steam client is the publisher key and the ticket;
        /// a connection failure quotes its target, which is DATABASE_URL. The stack trace survives, because
        /// it names methods rather than values and is the part actually worth reading.
        /// </summary>
        public static string Describe(Exception? failure) =>
            failure is null ? string.Empty : Redact(failure.ToString());

        /// <summary>
        /// The secret behind the log pseudonyms. Steam account ids sit in a small, enumerable namespace,
        /// so an unkeyed digest of one is not a pseudonym at all: anyone holding the log can precompute
        /// the candidates and read the accounts straight back out. The handle is therefore an HMAC, and
        /// the key is a secret. Absent a configured one this is random per process, which keeps handles
        /// correlatable inside a single process lifetime and meaningless across a restart.
        /// </summary>
        static byte[] _pseudonymKey = RandomNumberGenerator.GetBytes(32);

        /// <summary>
        /// Installs the configured pseudonym key. Called once at startup, before anything is logged;
        /// production sets it from the secret store so handles stay comparable across restarts and
        /// across instances.
        /// </summary>
        public static void ConfigureKey(byte[] key)
        {
            if (key is null || key.Length == 0)
            {
                throw new ArgumentException("a log pseudonym key must not be empty", nameof(key));
            }

            _pseudonymKey = (byte[])key.Clone();
        }

        /// <summary>
        /// A stable, keyed 64-bit handle for a Steam ID, rendered as sid: plus 16 lowercase hex, so
        /// operators can correlate the log lines for one player without the log becoming a list of
        /// accounts. Sixty-four bits rather than thirty-two because a handle narrow enough to collide is
        /// a handle that merges two players and misleads whoever is reading.
        /// </summary>
        public static string HashSteamId(string steamId)
        {
            var digest = HMACSHA256.HashData(_pseudonymKey, Encoding.UTF8.GetBytes(steamId ?? string.Empty));
            return "sid:" + Convert.ToHexString(digest, 0, 8).ToLowerInvariant();
        }
    }
}

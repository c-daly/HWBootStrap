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
            @"\b(key|ticket|token|access_token|credential)=[^&\s<>]*",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

        // Belt and braces: even an unrecognised parameter name cannot survive, because the whole query
        // string goes. Stopping at whitespace keeps the surrounding sentence readable.
        static readonly Regex QueryString = new(
            @"\?\S*", RegexOptions.CultureInvariant | RegexOptions.Compiled);

        public static string Redact(string text)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;

            var masked = SecretParameter.Replace(text, m => m.Groups[1].Value + "=" + Mask);
            return QueryString.Replace(masked, "?" + Mask);
        }

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

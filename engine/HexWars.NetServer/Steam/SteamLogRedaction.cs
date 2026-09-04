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
        /// A stable, non-reversible 12-character handle for a Steam ID, so operators can correlate log
        /// lines for one player without the log becoming a list of accounts.
        /// </summary>
        public static string HashSteamId(string steamId)
        {
            var digest = SHA256.HashData(Encoding.UTF8.GetBytes(steamId ?? string.Empty));
            return "sid:" + Convert.ToHexString(digest, 0, 4).ToLowerInvariant();
        }
    }
}

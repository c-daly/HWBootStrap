using System.Globalization;
using System.Net;
using System.Net.Sockets;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// Reads the MATCH_TRUSTED_PROXY_CIDRS entries that say whose X-Forwarded-For this server will
    /// believe.
    ///
    /// One parser, used by both configuration validation and the middleware wiring, so a value that
    /// starts the process is exactly a value that will be honoured. Two parsers would eventually differ,
    /// and the direction they would differ in is an entry that validates and then silently trusts nobody.
    /// </summary>
    public static class TrustedProxies
    {
        const string PrefixSeparator = "/";

        /// <summary>
        /// Parses one entry: a bare IPv4 or IPv6 address, or an address with a prefix length. A bare
        /// address comes back as a full-length prefix, so a caller has one shape to handle.
        /// </summary>
        public static bool TryParse(string? raw, out IPAddress address, out int prefixLength)
        {
            address = IPAddress.None;
            prefixLength = 0;

            if (string.IsNullOrWhiteSpace(raw)) return false;
            string text = raw.Trim();

            int separator = text.IndexOf(PrefixSeparator, StringComparison.Ordinal);
            if (separator < 0)
            {
                if (!IPAddress.TryParse(text, out IPAddress? single)) return false;

                address = single;
                prefixLength = FullPrefixLength(single);
                return true;
            }

            if (!IPAddress.TryParse(text[..separator], out IPAddress? prefix)) return false;

            if (!int.TryParse(
                    text[(separator + 1)..], NumberStyles.None, CultureInfo.InvariantCulture, out int length))
            {
                return false;
            }

            if (length < 0 || length > FullPrefixLength(prefix)) return false;

            address = prefix;
            prefixLength = length;
            return true;
        }

        public static bool IsValid(string? raw) => TryParse(raw, out _, out _);

        static int FullPrefixLength(IPAddress address) =>
            address.AddressFamily == AddressFamily.InterNetworkV6 ? 128 : 32;
    }
}

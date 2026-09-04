using System.Globalization;

namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// Every Steam identifier in this server is a canonical decimal SteamID64 string, never a long: the
    /// values exceed long.MaxValue only in theory but they are routinely round-tripped through JSON, URLs
    /// and Postgres text columns, and one accidental numeric narrowing would silently corrupt an account.
    /// This is the single gate that turns anything user supplied into that canonical form.
    /// </summary>
    public static class SteamId64
    {
        /// <summary>The lowest SteamID64 an individual account can have (universe 1, type 1, instance 1).</summary>
        public const ulong IndividualBase = 76561197960265728UL;

        /// <summary>Longest decimal representation of a ulong, used to reject overflow before parsing.</summary>
        const int MaxDigits = 20;

        /// <summary>
        /// True when <paramref name="raw"/> is a SteamID64 for an individual account. Surrounding
        /// whitespace and leading zeros are tolerated; the canonical form has neither. A JSON number must
        /// be converted to its string form by the caller before it gets here.
        /// </summary>
        public static bool TryNormalize(string? raw, out string canonical)
        {
            canonical = string.Empty;
            if (string.IsNullOrWhiteSpace(raw)) return false;

            var trimmed = raw.Trim();
            if (trimmed.Length == 0 || trimmed.Length > MaxDigits) return false;

            foreach (var c in trimmed)
            {
                if (!char.IsAsciiDigit(c)) return false;
            }

            if (!ulong.TryParse(trimmed, NumberStyles.None, CultureInfo.InvariantCulture, out var value)) return false;
            if (value < IndividualBase) return false;

            canonical = value.ToString(CultureInfo.InvariantCulture);
            return true;
        }

        public static bool IsValid(string? raw) => TryNormalize(raw, out _);
    }
}

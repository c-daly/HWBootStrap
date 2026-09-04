using System.Globalization;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// Which browser Origins may upgrade to a websocket. Rejecting a mismatched Origin before Accept is
    /// what closes cross-site websocket hijacking of a logged-in session (audit M13). Comparing ONLY
    /// against the request Host, as this used to, also made ALLOWED_WEB_ORIGINS dead configuration: a
    /// client served from another domain was refused no matter what an operator configured. This keeps
    /// the same-origin allowance and honours the configured list alongside it.
    /// </summary>
    public static class OriginPolicy
    {
        /// <summary>True when the request may be upgraded. An absent or unparseable Origin is allowed
        /// through unchanged (non-browser clients and the in-process selftest send none), which matches
        /// the scope of the rule: it applies when both an Origin and a Host are present.</summary>
        public static bool IsAllowed(HttpContext context, IReadOnlyList<string> allowedOrigins)
        {
            string origin = context.Request.Headers.Origin.ToString();
            if (string.IsNullOrEmpty(origin)) return true;
            if (!Uri.TryCreate(origin, UriKind.Absolute, out Uri? originUri)) return true;

            string originAuthority = Authority(originUri);
            string host = context.Request.Host.Value ?? string.Empty;
            if (host.Length > 0 && string.Equals(originAuthority, host, StringComparison.OrdinalIgnoreCase))
                return true;

            foreach (string entry in allowedOrigins)
                if (Matches(entry, originUri.Scheme, originAuthority)) return true;

            return false;
        }

        /// <summary>A configured entry is compared as scheme plus authority, so a trailing slash, an
        /// explicitly written default port and letter case all make no difference. An entry that is not an
        /// absolute URL matches nothing rather than matching everything.</summary>
        static bool Matches(string entry, string originScheme, string originAuthority)
        {
            string candidate = entry.Trim();
            if (candidate.Length == 0) return false;
            if (!Uri.TryCreate(candidate, UriKind.Absolute, out Uri? allowed)) return false;
            return string.Equals(allowed.Scheme, originScheme, StringComparison.OrdinalIgnoreCase)
                && string.Equals(Authority(allowed), originAuthority, StringComparison.OrdinalIgnoreCase);
        }

        static string Authority(Uri uri) => uri.IsDefaultPort
            ? uri.Host
            : uri.Host + ":" + uri.Port.ToString(CultureInfo.InvariantCulture);
    }
}

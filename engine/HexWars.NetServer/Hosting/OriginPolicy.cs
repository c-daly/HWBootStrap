using System.Globalization;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// Which browser Origins may upgrade to a websocket. Rejecting a mismatched Origin before Accept is
    /// what closes cross-site websocket hijacking of a logged-in session (audit M13). Comparing ONLY
    /// against the request Host, as this used to, also made ALLOWED_WEB_ORIGINS dead configuration: a
    /// client served from another domain was refused no matter what an operator configured. This keeps
    /// the same-origin allowance and honours the configured list alongside it.
    ///
    /// It is a browser rule and only a browser rule. A native client - the Steam build, a script, a
    /// test - chooses what it sends here, so for those callers this is not a security boundary and must
    /// never be treated as one. What protects a seat from a native client is the join credential, which
    /// is issued to one Steam account for one match and proved on the socket. This rule exists because a
    /// BROWSER does not let a page choose its Origin, which is what makes it worth checking there.
    /// </summary>
    public static class OriginPolicy
    {
        /// <summary>
        /// True when the request may be upgraded.
        ///
        /// An ABSENT Origin is allowed: non-browser clients and the in-process selftest send none, and this
        /// rule only has anything to say when a browser has attached one. An Origin that is present and
        /// cannot be parsed as an absolute URL is REFUSED, which is not the same case. The value a browser
        /// sends there is the literal word null, from a sandboxed frame or a document loaded from a file or
        /// a data URL - exactly the contexts a cross-site upgrade would be launched from - and letting an
        /// unreadable Origin through would leave the rule with a hole shaped like the attack.
        /// </summary>
        public static bool IsAllowed(HttpContext context, IReadOnlyList<string> allowedOrigins)
        {
            string origin = context.Request.Headers.Origin.ToString();
            if (string.IsNullOrEmpty(origin)) return true;
            if (!Uri.TryCreate(origin, UriKind.Absolute, out Uri? originUri)) return false;

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

namespace HexWars.NetServer.Configuration
{
    /// <summary>Everything the match host needs that is not a Steam credential. Bound from the flat
    /// DATABASE_URL / MATCH_* / ALLOWED_WEB_ORIGINS / LOBBY_PROVIDER environment keys.</summary>
    public sealed class MatchHostingOptions
    {
        public const int DefaultJoinTokenTtlSeconds = 900;
        public const int MinJoinTokenTtlSeconds = 60;
        public const int MaxJoinTokenTtlSeconds = 86400;
        public const int DefaultProtocolVersion = 2;

        /// <summary>A postgres:// URI or an Npgsql key=value string. A secret (it carries a password).</summary>
        public string DatabaseUrl { get; set; } = string.Empty;

        /// <summary>Absolute public URL of this server, used to build the client websocket URL.</summary>
        public Uri? PublicBaseUrl { get; set; }

        public int JoinTokenTtlSeconds { get; set; } = DefaultJoinTokenTtlSeconds;

        public string BuildId { get; set; } = string.Empty;

        public int ProtocolVersion { get; set; } = DefaultProtocolVersion;

        public string[] AllowedWebOrigins { get; set; } = Array.Empty<string>();

        public LobbyProviders LobbyProvider { get; set; } = LobbyProviders.Legacy;

        /// <summary>Empty means every client build is accepted.</summary>
        public string[] CompatibleClientBuilds { get; set; } = Array.Empty<string>();

        public bool TrustForwardedHeaders { get; set; }

        /// <summary>
        /// The addresses and networks whose X-Forwarded-For this server will believe, as bare IP addresses
        /// or CIDR prefixes. Only consulted when <see cref="TrustForwardedHeaders"/> is on.
        ///
        /// Empty means every peer is trusted to name the client, which is only safe when nothing can reach
        /// this process except the platform proxy. The server says so at startup rather than assuming the
        /// operator meant it.
        /// </summary>
        public string[] TrustedProxyCidrs { get; set; } = Array.Empty<string>();

        /// <summary>
        /// The operator saying, in as many words, that trusting every peer to name the client is what they
        /// meant. Required when <see cref="TrustForwardedHeaders"/> is on and
        /// <see cref="TrustedProxyCidrs"/> is empty, because that combination is a real deployment on a
        /// platform whose proxy addresses are not published, and also exactly what an unfinished
        /// configuration looks like. Startup should not have to guess which one it is looking at.
        /// </summary>
        public bool TrustAllProxies { get; set; }

        public string[] BlockedSteamIds { get; set; } = Array.Empty<string>();

        public string? MetricsToken { get; set; }

        /// <summary>The shortest key worth having behind a log pseudonym.</summary>
        public const int MinLogPseudonymKeyLength = 16;

        /// <summary>
        /// Secret behind the Steam-id pseudonyms in logs. A secret: it is what stops a log reader
        /// precomputing the handles for an enumerable id namespace. Null means a random per-process key,
        /// so handles correlate within one process lifetime and never across a restart or an instance.
        /// </summary>
        public string? LogPseudonymKey { get; set; }
    }
}

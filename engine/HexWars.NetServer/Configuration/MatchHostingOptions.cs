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

        public string[] BlockedSteamIds { get; set; } = Array.Empty<string>();

        public string? MetricsToken { get; set; }
    }
}

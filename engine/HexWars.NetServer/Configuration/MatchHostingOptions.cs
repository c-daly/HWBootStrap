namespace HexWars.NetServer.Configuration
{
    /// <summary>Everything the match host needs that is not a Steam credential. Bound from the flat
    /// DATABASE_URL / MATCH_* / ALLOWED_WEB_ORIGINS / LOBBY_PROVIDER environment keys.</summary>
    public sealed class MatchHostingOptions
    {
        public const int DefaultJoinTokenTtlSeconds = 900;
        public const int MinJoinTokenTtlSeconds = 60;
        public const int MaxJoinTokenTtlSeconds = 86400;
        public const int DefaultProtocolVersion = ProtocolContract.Version;

        public const int DefaultTerminalReconnectSeconds = 600;
        public const int MinTerminalReconnectSeconds = 0;
        public const int MaxTerminalReconnectSeconds = 86400;

        public const int DefaultMaxSocketsPerIp = 8;
        public const int MinMaxSocketsPerIp = 1;
        public const int MaxMaxSocketsPerIp = 256;

        public const int DefaultMaxRechecksPerCadence = 256;
        public const int MinMaxRechecksPerCadence = 8;
        public const int MaxMaxRechecksPerCadence = 4096;

        public const int DefaultCredentialRecheckSeconds = 60;
        public const int MinCredentialRecheckSeconds = 5;
        public const int MaxCredentialRecheckSeconds = 3600;

        public const int DefaultOutboundQueueBytes = 4 * 1024 * 1024;
        public const int MinOutboundQueueBytes = 64 * 1024;
        public const int MaxOutboundQueueBytes = 64 * 1024 * 1024;

        public const int DefaultHeartbeatIntervalSeconds = 20;
        public const int MinHeartbeatIntervalSeconds = 1;
        public const int MaxHeartbeatIntervalSeconds = 300;

        public const int DefaultStaleConnectionSeconds = 60;
        public const int MinStaleConnectionSeconds = 2;
        public const int MaxStaleConnectionSeconds = 900;

        public const int DefaultOutboundQueueCapacity = 256;
        public const int MinOutboundQueueCapacity = 16;
        public const int MaxOutboundQueueCapacity = 4096;

        public const int DefaultAuthFrameTimeoutSeconds = 10;
        public const int MinAuthFrameTimeoutSeconds = 1;
        public const int MaxAuthFrameTimeoutSeconds = 120;

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

        /// <summary>
        /// How often the v2 host sends PING on every authenticated socket.
        ///
        /// It is the only traffic on an idle match, so it is also the only thing that keeps an intermediary
        /// from quietly dropping a socket both ends still believe in, and the only thing that tells this
        /// process a client has gone away without closing.
        /// </summary>
        public int HeartbeatIntervalSeconds { get; set; } = DefaultHeartbeatIntervalSeconds;

        /// <summary>
        /// How long a socket may go without a single inbound frame before it is closed as stale.
        ///
        /// Must be longer than <see cref="HeartbeatIntervalSeconds"/>, and by enough to survive one lost
        /// ping: a window equal to the cadence would judge silence over an interval the client was never
        /// given a chance to answer in.
        /// </summary>
        public int StaleConnectionSeconds { get; set; } = DefaultStaleConnectionSeconds;

        /// <summary>
        /// Frames one connection may have waiting to go out before the server gives up on it.
        ///
        /// The queue is bounded because the alternative is not "no limit" but "the limit is this process's
        /// memory": a client that stops reading while its match keeps playing would otherwise be paid for
        /// by every other match on the host.
        /// </summary>
        public int OutboundQueueCapacity { get; set; } = DefaultOutboundQueueCapacity;

        /// <summary>How long a freshly accepted socket has to send its AUTH frame. An unauthenticated
        /// socket costs a connection slot and holds no seat, so this is deliberately short.</summary>
        public int AuthFrameTimeoutSeconds { get; set; } = DefaultAuthFrameTimeoutSeconds;

        /// <summary>
        /// How long after a match finishes a player may still reconnect to it and be dealt the final state.
        ///
        /// The last APPLY of a game is the one most likely to be lost: it is broadcast at the instant the
        /// match becomes terminal, and a client whose socket dropped a moment earlier has no way to learn
        /// how the game it was playing ended. Without a window it is told its credential is invalid, which
        /// is true and useless. Zero closes the window entirely.
        /// </summary>
        public int TerminalReconnectSeconds { get; set; } = DefaultTerminalReconnectSeconds;

        /// <summary>The same value as a span, and zero when the window is closed. Every place that judges
        /// the window reads it from here so they cannot drift apart on the arithmetic.</summary>
        public TimeSpan TerminalReconnectWindow =>
            TerminalReconnectSeconds <= 0 ? TimeSpan.Zero : TimeSpan.FromSeconds(TerminalReconnectSeconds);

        /// <summary>
        /// Sockets one address may hold on /ws/v2 at once, counted before the upgrade is accepted.
        ///
        /// A match needs two, and a household behind one NAT might plausibly want a handful. The cap is not
        /// about them: an unauthenticated socket costs a connection slot, a receive buffer and a writer task
        /// before it has proved anything, and without a per-address ceiling one client can open as many as
        /// the process has memory for.
        /// </summary>
        public int MaxSocketsPerIp { get; set; } = DefaultMaxSocketsPerIp;

        /// <summary>
        /// Bytes one connection may have queued to go out before it is closed as a slow client.
        ///
        /// The frame count is not a memory bound on its own. A START carrying a long journal is orders of
        /// magnitude bigger than an APPLY, so a queue that is well inside its frame limit can still be
        /// holding megabytes, and the number of those queues is the number of sockets on the host.
        /// </summary>
        public int OutboundQueueBytes { get; set; } = DefaultOutboundQueueBytes;

        /// <summary>
        /// How often a live socket has its credential checked again against the store.
        ///
        /// The handshake is a moment and a match is an hour. A credential revoked by the same player
        /// reconnecting elsewhere, or simply expired, leaves a socket that was legitimate when it opened
        /// and is not any more, and nothing else on this host would ever notice.
        /// </summary>
        public int CredentialRecheckSeconds { get; set; } = DefaultCredentialRecheckSeconds;

        /// <summary>
        /// Sockets one heartbeat will re-check the credential of.
        /// </summary>
        /// <remarks>
        /// It has to be large enough that the whole socket budget fits inside one recheck interval, or the
        /// documented cadence is a fiction: with a cap of 32 and 100 live sockets the last of them is asked
        /// about minutes late. Eight concurrent checks over a 20 second heartbeat drain 256 sub-second
        /// queries comfortably, so that is the default. It is still a cap rather than no cap, because a
        /// host that has just come up finds every socket due at once and the tick that discovers that also
        /// has pings to send.
        /// </remarks>
        public int MaxRechecksPerCadence { get; set; } = DefaultMaxRechecksPerCadence;
    }
}

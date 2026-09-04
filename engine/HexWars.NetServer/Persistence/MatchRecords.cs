namespace HexWars.NetServer.Persistence
{
    /// <summary>
    /// The lifecycle of a match. Stored as lowercase text rather than an integer so a person reading the
    /// table during an incident can see what happened without a lookup, and so adding a status later cannot
    /// silently renumber the existing rows.
    /// </summary>
    public enum MatchStatus
    {
        /// <summary>Seats allocated, no start replay yet.</summary>
        Waiting,

        /// <summary>The start replay is written and the command journal is open.</summary>
        Active,

        /// <summary>The game reached a real end state, with a winner seat recorded.</summary>
        Completed,

        /// <summary>Aged out before it ever started. No winner.</summary>
        Expired,

        /// <summary>Started and then went quiet. No winner.</summary>
        Abandoned
    }

    /// <summary>One row of <c>matches</c>. Timestamps are always UTC.</summary>
    public sealed record PersistedMatch(Guid MatchId, string? SteamLobbyId, MatchStatus Status, string SetupWire, string? StartReplay,
        string EngineVersion, int ProtocolVersion, string BuildId, DateTimeOffset CreatedAt, DateTimeOffset? StartedAt,
        DateTimeOffset? CompletedAt, DateTimeOffset LastActivityAt, int? WinnerSeat);

    /// <summary>One seat of a match. <paramref name="CatalogWire"/> is the normalized barracks the player
    /// chose while the match was waiting; it is null until they send one.</summary>
    public sealed record PersistedPlayer(Guid MatchId, string SteamId, int Seat, string? CatalogWire, DateTimeOffset JoinedAt, DateTimeOffset? LastSeenAt);

    /// <summary>One accepted command. Sequence numbers start at 1 and never have gaps.</summary>
    public sealed record PersistedCommand(Guid MatchId, int Sequence, string CommandWire, DateTimeOffset AcceptedAt, string IssuerSteamId);

    /// <summary>Everything needed to rebuild a match from nothing: the setup and start replay in
    /// <paramref name="Match"/>, plus the commands to apply in order. <paramref name="Commands"/> is ordered
    /// by sequence ascending and <paramref name="Players"/> by seat.</summary>
    public sealed record MatchJournal(PersistedMatch Match, IReadOnlyList<PersistedPlayer> Players, IReadOnlyList<PersistedCommand> Commands);

    /// <summary><paramref name="Players"/> must be exactly two entries holding seats 0 and 1 with distinct
    /// Steam ids; anything else is a programming error and is rejected with <see cref="ArgumentException"/>.</summary>
    public sealed record CreateMatchRequest(string SteamLobbyId, string SetupWire, string EngineVersion, int ProtocolVersion, string BuildId,
        IReadOnlyList<(string SteamId, int Seat)> Players, DateTimeOffset CreatedAt);

    /// <summary><paramref name="Created"/> is false when an existing waiting or active match for that lobby
    /// was returned instead, which is the normal answer when both players race to allocate.</summary>
    public sealed record CreateMatchResult(PersistedMatch Match, bool Created);

    /// <summary>What happened to an append.</summary>
    public enum AppendStatus
    {
        /// <summary>The command is now durable at the requested sequence.</summary>
        Appended,

        /// <summary>That exact command (same wire, same issuer) is already stored at that sequence, so the
        /// caller is retrying a write that actually succeeded. Treat it as success.</summary>
        AlreadyApplied,

        /// <summary>The requested sequence is not the next one, or a different command already holds it. The
        /// view of the journal held by the caller is stale and must be rebuilt.</summary>
        Conflict,

        /// <summary>The match is missing, still waiting, or already terminal. Nothing was written.</summary>
        MatchNotActive
    }

    /// <summary>
    /// The outcome of an append. <paramref name="Sequence"/> is the sequence the command occupies for
    /// <see cref="AppendStatus.Appended"/> and <see cref="AppendStatus.AlreadyApplied"/>; for
    /// <see cref="AppendStatus.Conflict"/> it is the sequence the store expects NEXT, which is what a caller
    /// rebuilding its projection wants to know; for <see cref="AppendStatus.MatchNotActive"/> it is simply
    /// the sequence that was asked for, because the store knows nothing else about a match it will not write.
    /// </summary>
    public sealed record AppendResult(AppendStatus Status, int Sequence);

    /// <summary>A stored join credential. Only the SHA-256 hash of the credential is ever persisted, so this
    /// record cannot be used to reconstruct the token a client holds.</summary>
    public sealed record JoinCredentialRecord(byte[] CredentialHash, Guid MatchId, string SteamId, DateTimeOffset ExpiresAt, DateTimeOffset? RevokedAt);

    /// <summary>
    /// The single place that knows how <see cref="MatchStatus"/> is spelled in the database. The schema has a
    /// CHECK constraint on the same five words, so a mismatch here is a startup-time failure rather than a
    /// silently wrong row.
    /// </summary>
    public static class MatchStatusText
    {
        public const string Waiting = "waiting";
        public const string Active = "active";
        public const string Completed = "completed";
        public const string Expired = "expired";
        public const string Abandoned = "abandoned";

        public static string ToDb(MatchStatus status) => status switch
        {
            MatchStatus.Waiting => Waiting,
            MatchStatus.Active => Active,
            MatchStatus.Completed => Completed,
            MatchStatus.Expired => Expired,
            MatchStatus.Abandoned => Abandoned,
            _ => throw new ArgumentOutOfRangeException(nameof(status), status, "Unknown match status.")
        };

        public static MatchStatus FromDb(string status) => status switch
        {
            Waiting => MatchStatus.Waiting,
            Active => MatchStatus.Active,
            Completed => MatchStatus.Completed,
            Expired => MatchStatus.Expired,
            Abandoned => MatchStatus.Abandoned,
            _ => throw new ArgumentOutOfRangeException(nameof(status), status,
                "Unknown match status text; the database holds a status this build does not understand.")
        };
    }

    /// <summary>
    /// Argument checks shared by every <see cref="IMatchStore"/> implementation. They live here rather than in
    /// each store so the in-memory double cannot quietly accept a request Postgres would reject: a test that
    /// passes against the double has to be a test that would pass against the database.
    /// </summary>
    internal static class MatchStoreGuard
    {
        /// <summary>SHA-256, so 32 bytes. The schema enforces the same length.</summary>
        public const int CredentialHashBytes = 32;

        public static void ValidatePlayers(IReadOnlyList<(string SteamId, int Seat)> players)
        {
            ArgumentNullException.ThrowIfNull(players);

            if (players.Count != 2)
                throw new ArgumentException(
                    "A match needs exactly two players, one in each seat, but " + players.Count + " were given.",
                    nameof(players));

            var seats = new HashSet<int>();
            var steamIds = new HashSet<string>(StringComparer.Ordinal);

            foreach ((string steamId, int seat) in players)
            {
                if (string.IsNullOrWhiteSpace(steamId))
                    throw new ArgumentException("A player was given with no Steam id.", nameof(players));
                if (seat is not (0 or 1))
                    throw new ArgumentException("Seat " + seat + " is not a seat; only 0 and 1 exist.", nameof(players));
                if (!seats.Add(seat))
                    throw new ArgumentException("Two players were given seat " + seat + ".", nameof(players));
                if (!steamIds.Add(steamId))
                    throw new ArgumentException("The same Steam id was given both seats.", nameof(players));
            }
        }

        public static void ValidateCredentialHash(byte[] credentialHash, string parameterName)
        {
            ArgumentNullException.ThrowIfNull(credentialHash, parameterName);

            if (credentialHash.Length != CredentialHashBytes)
                throw new ArgumentException(
                    "A join credential hash is " + CredentialHashBytes + " bytes (SHA-256), not "
                    + credentialHash.Length + ".", parameterName);
        }
    }
}

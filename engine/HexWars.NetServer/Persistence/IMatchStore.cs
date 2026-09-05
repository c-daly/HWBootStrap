namespace HexWars.NetServer.Persistence
{
    /// <summary>
    /// The durable side of a match: the row, the seats, the command journal and the join credentials.
    ///
    /// The rules below are not implementation notes, they are the contract. Every method is safe to call
    /// concurrently and safe to call twice with the same arguments, because the match coordinator retries
    /// after a dropped connection and must never end up with a duplicated or reordered command. Where a
    /// guarantee cannot be expressed in the type system it is enforced by the schema (the partial unique
    /// index on open lobbies, the primary key on match id plus sequence), so an implementation that talks to
    /// Postgres inherits it and a test double has to reproduce it deliberately.
    ///
    /// Timestamps are supplied by the caller rather than taken from a clock inside the store, so a replay or
    /// a recovery pass can write the times things actually happened. They are stored and returned as UTC.
    /// </summary>
    public interface IMatchStore
    {
        /// <summary>
        /// Allocates a match for a Steam lobby, or hands back the one that is already open for it.
        ///
        /// Idempotent per lobby: while a waiting or active match exists for <c>SteamLobbyId</c> this returns
        /// that match with <see cref="CreateMatchResult.Created"/> false and writes nothing. That is the
        /// normal outcome when both players ask at once. Once the match reaches a terminal status the lobby
        /// is free and the next call allocates a new one.
        /// </summary>
        /// <exception cref="ArgumentException">The request does not carry exactly two players holding seats
        /// 0 and 1 with distinct Steam ids.</exception>
        Task<CreateMatchResult> CreateMatchForLobbyAsync(CreateMatchRequest request, CancellationToken ct);

        /// <summary>The match row, or null when no match has that id.</summary>
        Task<PersistedMatch?> GetMatchAsync(Guid matchId, CancellationToken ct);

        /// <summary>The waiting or active match for that lobby, or null. Terminal matches are invisible here:
        /// this is the question "can this lobby still be joined", not "what did this lobby ever host".</summary>
        Task<PersistedMatch?> FindOpenMatchForLobbyAsync(string steamLobbyId, CancellationToken ct);

        /// <summary>Both seats, ordered by seat number. Empty when the match does not exist.</summary>
        Task<IReadOnlyList<PersistedPlayer>> GetPlayersAsync(Guid matchId, CancellationToken ct);

        /// <summary>One seat by Steam id, or null when that player is not in that match.</summary>
        Task<PersistedPlayer?> GetPlayerAsync(Guid matchId, string steamId, CancellationToken ct);

        /// <summary>
        /// Records the barracks a player picked. A no-op unless the match is still waiting: once the start
        /// replay is written the catalogue is baked into it, and letting a late frame rewrite the row would
        /// make the stored setup disagree with the game everyone is playing.
        /// </summary>
        Task SaveCatalogAsync(Guid matchId, string steamId, string catalogWire, CancellationToken ct);

        /// <summary>
        /// Moves waiting to active and stores the start replay, atomically. Returns false when the match was
        /// not waiting, which is how two racing callers agree on a single start: exactly one gets true, and
        /// the loser reads the replay the winner wrote rather than dealing a second, different game.
        /// </summary>
        Task<bool> TryStartMatchAsync(Guid matchId, string startReplay, DateTimeOffset startedAt, CancellationToken ct);

        /// <summary>
        /// Appends one accepted command to the journal.
        ///
        /// <paramref name="expectedSequence"/> must be exactly one past the highest stored sequence (so the
        /// first command of a match is 1). This is the whole ordering guarantee: a caller working from a
        /// stale projection cannot slip a command in, and a caller that never saw its own success cannot
        /// write it twice.
        ///
        /// <list type="bullet">
        /// <item><see cref="AppendStatus.Appended"/> when it is now stored.</item>
        /// <item><see cref="AppendStatus.AlreadyApplied"/> when that sequence already holds a command with
        /// the same wire AND the same issuer, which means an earlier attempt committed after all.</item>
        /// <item><see cref="AppendStatus.Conflict"/> when the sequence is not the next one, or a different
        /// command holds it. The result carries the sequence the store expects next.</item>
        /// <item><see cref="AppendStatus.MatchNotActive"/> when the match is missing, still waiting, or
        /// already terminal. Nothing is written.</item>
        /// </list>
        /// </summary>
        Task<AppendResult> AppendCommandAsync(Guid matchId, int expectedSequence, string commandWire, string issuerSteamId, DateTimeOffset acceptedAt, CancellationToken ct);

        /// <summary>Everything needed to rebuild the match: the row, both seats by seat number, and the whole
        /// command journal in sequence order. Null when the match does not exist.</summary>
        Task<MatchJournal?> LoadJournalAsync(Guid matchId, CancellationToken ct);

        /// <summary>
        /// Moves the match to a terminal status, atomically, and stamps <c>completed_at</c> and the winner.
        ///
        /// Only these edges are allowed: active to completed, expired or abandoned; waiting to expired or
        /// abandoned. Everything else returns false and changes nothing, so the reaper cannot expire a match
        /// that just finished and a finished match can never be reopened or re-scored.
        /// </summary>
        Task<bool> TryCompleteMatchAsync(Guid matchId, MatchStatus terminal, int? winnerSeat, DateTimeOffset completedAt, CancellationToken ct);

        /// <summary>Marks the match as alive at <paramref name="seenAt"/>, and the player too when
        /// <paramref name="steamId"/> is given. This is what keeps the abandonment reaper away from a match
        /// whose players are simply thinking.</summary>
        Task TouchAsync(Guid matchId, string? steamId, DateTimeOffset seenAt, CancellationToken ct);

        /// <summary>Every waiting or active match, oldest first. The startup recovery pass and the reaper both
        /// start here.</summary>
        Task<IReadOnlyList<Guid>> ListOpenMatchIdsAsync(CancellationToken ct);

        /// <summary>Stores the SHA-256 hash of a join credential, bound to one seat of one match. The raw
        /// credential is never given to the store.</summary>
        /// <exception cref="ArgumentException"><paramref name="credentialHash"/> is not 32 bytes.</exception>
        Task StoreJoinCredentialAsync(byte[] credentialHash, Guid matchId, string steamId, DateTimeOffset expiresAt, CancellationToken ct);

        /// <summary>
        /// Looks a credential up by hash. Returns the row whatever its expiry or revocation state: deciding
        /// whether an expired or revoked credential may be used belongs to the credential service, and a
        /// store that hid those rows would leave it unable to tell "never issued" from "issued and expired".
        /// Null when no credential has that hash.
        /// </summary>
        /// <exception cref="ArgumentException"><paramref name="credentialHash"/> is not 32 bytes.</exception>
        Task<JoinCredentialRecord?> FindJoinCredentialAsync(byte[] credentialHash, CancellationToken ct);

        /// <summary>Revokes every credential this player still holds for this match. Already revoked rows keep
        /// the timestamp they were revoked at.</summary>
        Task RevokeJoinCredentialsAsync(Guid matchId, string steamId, DateTimeOffset revokedAt, CancellationToken ct);

        /// <summary>
        /// Retires whatever this seat still holds and issues it one new credential, in a single transaction.
        ///
        /// This exists because revoke-then-store as two calls is not the operation the protocol needs. Two
        /// clients reconnecting at once can interleave into two live credentials for one seat, and a store
        /// failure between the two calls leaves the player with none - having destroyed the one they had.
        /// Doing both under one lock on the seat makes replacement true rather than merely usually true.
        ///
        /// The match status is checked inside the same transaction for the same reason: a match that
        /// completes while a join is in flight must not hand out a credential that would seat someone in a
        /// finished game. Returns false, having changed nothing, when the match is no longer waiting or
        /// active; the caller turns that into a refusal rather than a retry.
        /// </summary>
        /// <param name="now">The instant the revocations are stamped with.</param>
        /// <returns>True when the credential was stored; false when the match is not open.</returns>
        /// <exception cref="ArgumentException"><paramref name="credentialHash"/> is not 32 bytes, the Steam id
        /// is malformed, or it holds no seat in this match.</exception>
        /// <param name="allowTerminalWithin">When set, a match in a terminal status that has a start replay
        /// and finished within this window is still issued a credential. It is how a seat that missed the
        /// final APPLY gets back in to learn how the game ended; null means open matches only.</param>
        Task<bool> ReplaceJoinCredentialAsync(byte[] credentialHash, Guid matchId, string steamId,
            DateTimeOffset expiresAt, DateTimeOffset now, CancellationToken ct,
            TimeSpan? allowTerminalWithin = null);
    }
}

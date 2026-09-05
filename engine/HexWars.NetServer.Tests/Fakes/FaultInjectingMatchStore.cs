using HexWars.NetServer.Persistence;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// A store that forwards everything, except for the one write a test has armed it to fail.
    ///
    /// It exists because the durable ordering rule is only interesting when the durable step does not work.
    /// An in-memory double can be made to throw, but the thing under test here is the real Postgres store
    /// wrapped in the real host: what has to be proved is that a failed append leaves NOTHING behind - no
    /// broadcast, no advanced projection, and no row - and that the same command sent again is still the
    /// command it always was. Wrapping the real store rather than replacing it keeps every other call on the
    /// database, so the retry is checked against the journal that actually exists.
    ///
    /// The failure is armed once and consumed by the first call that meets it, so a test says "the next
    /// append fails" rather than trying to guess which attempt to break.
    /// </summary>
    public sealed class FaultInjectingMatchStore(IMatchStore inner) : IMatchStore
    {
        Exception? _nextAppendFailure;

        /// <summary>How many appends actually reached the inner store.</summary>
        public int AppendsForwarded { get; private set; }

        /// <summary>How many appends were refused by the injected fault.</summary>
        public int AppendsFailed { get; private set; }

        /// <summary>Arms the next AppendCommandAsync to throw, once. Null disarms it.</summary>
        public void FailNextAppend(Exception? failure) => _nextAppendFailure = failure;

        public Task<AppendResult> AppendCommandAsync(Guid matchId, int expectedSequence, string commandWire,
            string issuerSteamId, DateTimeOffset acceptedAt, CancellationToken ct)
        {
            Exception? failure = _nextAppendFailure;
            if (failure is not null)
            {
                _nextAppendFailure = null;
                AppendsFailed++;

                // Thrown rather than returned: an outage is not one of the outcomes AppendResult can carry,
                // and turning it into one here would test a code path production never takes.
                return Task.FromException<AppendResult>(failure);
            }

            AppendsForwarded++;
            return inner.AppendCommandAsync(
                matchId, expectedSequence, commandWire, issuerSteamId, acceptedAt, ct);
        }

        // ---- everything else is a plain forward -------------------------------

        public Task<CreateMatchResult> CreateMatchForLobbyAsync(CreateMatchRequest request, CancellationToken ct) =>
            inner.CreateMatchForLobbyAsync(request, ct);

        public Task<PersistedMatch?> GetMatchAsync(Guid matchId, CancellationToken ct) =>
            inner.GetMatchAsync(matchId, ct);

        public Task<PersistedMatch?> FindOpenMatchForLobbyAsync(string steamLobbyId, CancellationToken ct) =>
            inner.FindOpenMatchForLobbyAsync(steamLobbyId, ct);

        public Task<IReadOnlyList<PersistedPlayer>> GetPlayersAsync(Guid matchId, CancellationToken ct) =>
            inner.GetPlayersAsync(matchId, ct);

        public Task<PersistedPlayer?> GetPlayerAsync(Guid matchId, string steamId, CancellationToken ct) =>
            inner.GetPlayerAsync(matchId, steamId, ct);

        public Task SaveCatalogAsync(Guid matchId, string steamId, string catalogWire, CancellationToken ct) =>
            inner.SaveCatalogAsync(matchId, steamId, catalogWire, ct);

        public Task<bool> TryStartMatchAsync(
            Guid matchId, string startReplay, DateTimeOffset startedAt, CancellationToken ct) =>
            inner.TryStartMatchAsync(matchId, startReplay, startedAt, ct);

        public Task<MatchJournal?> LoadJournalAsync(Guid matchId, CancellationToken ct) =>
            inner.LoadJournalAsync(matchId, ct);

        public Task<bool> TryCompleteMatchAsync(Guid matchId, MatchStatus terminal, int? winnerSeat,
            DateTimeOffset completedAt, CancellationToken ct) =>
            inner.TryCompleteMatchAsync(matchId, terminal, winnerSeat, completedAt, ct);

        public Task TouchAsync(Guid matchId, string? steamId, DateTimeOffset seenAt, CancellationToken ct) =>
            inner.TouchAsync(matchId, steamId, seenAt, ct);

        public Task<IReadOnlyList<Guid>> ListOpenMatchIdsAsync(CancellationToken ct) =>
            inner.ListOpenMatchIdsAsync(ct);

        public Task StoreJoinCredentialAsync(byte[] credentialHash, Guid matchId, string steamId,
            DateTimeOffset expiresAt, CancellationToken ct) =>
            inner.StoreJoinCredentialAsync(credentialHash, matchId, steamId, expiresAt, ct);

        public Task<JoinCredentialRecord?> FindJoinCredentialAsync(byte[] credentialHash, CancellationToken ct) =>
            inner.FindJoinCredentialAsync(credentialHash, ct);

        public Task RevokeJoinCredentialsAsync(
            Guid matchId, string steamId, DateTimeOffset revokedAt, CancellationToken ct) =>
            inner.RevokeJoinCredentialsAsync(matchId, steamId, revokedAt, ct);

        public Task<bool> ReplaceJoinCredentialAsync(byte[] credentialHash, Guid matchId, string steamId,
            DateTimeOffset expiresAt, DateTimeOffset now, CancellationToken ct) =>
            inner.ReplaceJoinCredentialAsync(credentialHash, matchId, steamId, expiresAt, now, ct);
    }
}

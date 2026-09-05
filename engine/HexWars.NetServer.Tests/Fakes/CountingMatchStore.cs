using HexWars.NetServer.Persistence;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// An <see cref="IMatchStore"/> that forwards every call to another one and counts what went past.
    ///
    /// It exists for an assertion nothing else can make. Whether a request the server should refuse on
    /// sight nevertheless became a database query is invisible in the return value: both paths answer no.
    /// The difference only shows up under load, where a lookup per junk credential turns an unauthenticated
    /// socket into a free amplifier against the database. Counting is how a test can see it.
    ///
    /// Reads and writes are counted apart because they are refused for different reasons and by different
    /// parts of the system, so a test that means one of them should not pass because of the other.
    /// </summary>
    public sealed class CountingMatchStore(IMatchStore inner) : IMatchStore
    {
        public int Reads { get; private set; }

        public int Writes { get; private set; }

        public int Calls => Reads + Writes;

        /// <summary>
        /// When set, seat lookups answer null however the inner store would have answered, while still
        /// counting as a read. It stands in for the one state the real stores cannot be talked into: a
        /// credential row whose seat is gone. Postgres refuses to store a credential for a seat that does
        /// not exist, and deleting the match takes the credential with it, so the only way a validator can
        /// meet that pair in production is a race, and the only way a test can is here.
        /// </summary>
        public bool HideSeats { get; set; }

        public void ResetCounts()
        {
            Reads = 0;
            Writes = 0;
        }

        public Task<CreateMatchResult> CreateMatchForLobbyAsync(CreateMatchRequest request, CancellationToken ct)
        {
            Writes++;
            return inner.CreateMatchForLobbyAsync(request, ct);
        }

        public Task<PersistedMatch?> GetMatchAsync(Guid matchId, CancellationToken ct)
        {
            Reads++;
            return inner.GetMatchAsync(matchId, ct);
        }

        public Task<PersistedMatch?> FindOpenMatchForLobbyAsync(string steamLobbyId, CancellationToken ct)
        {
            Reads++;
            return inner.FindOpenMatchForLobbyAsync(steamLobbyId, ct);
        }

        public Task<IReadOnlyList<PersistedPlayer>> GetPlayersAsync(Guid matchId, CancellationToken ct)
        {
            Reads++;
            return inner.GetPlayersAsync(matchId, ct);
        }

        /// <summary>
        /// Runs before every seat lookup, once per assignment. It is the seam for the races an endpoint has
        /// to survive: a seat lookup is the last read a join does before it issues a credential, so a hook
        /// that completes the match here reproduces a game ending in the window between the status check and
        /// the issue, which is otherwise only reachable by getting the timing right by luck.
        /// </summary>
        public Func<Guid, string, Task>? BeforeGetPlayer { get; set; }

        public async Task<PersistedPlayer?> GetPlayerAsync(Guid matchId, string steamId, CancellationToken ct)
        {
            Reads++;

            Func<Guid, string, Task>? hook = BeforeGetPlayer;
            if (hook is not null)
            {
                // Cleared before it runs: the hook usually writes through this same store, and one that
                // re-entered itself would recurse rather than simulate a race.
                BeforeGetPlayer = null;
                await hook(matchId, steamId).ConfigureAwait(false);
            }

            return HideSeats
                ? null
                : await inner.GetPlayerAsync(matchId, steamId, ct).ConfigureAwait(false);
        }

        public Task SaveCatalogAsync(Guid matchId, string steamId, string catalogWire, CancellationToken ct)
        {
            Writes++;
            return inner.SaveCatalogAsync(matchId, steamId, catalogWire, ct);
        }

        public Task<bool> TryStartMatchAsync(Guid matchId, string startReplay, DateTimeOffset startedAt, CancellationToken ct)
        {
            Writes++;
            return inner.TryStartMatchAsync(matchId, startReplay, startedAt, ct);
        }

        public Task<AppendResult> AppendCommandAsync(Guid matchId, int expectedSequence, string commandWire,
            string issuerSteamId, DateTimeOffset acceptedAt, CancellationToken ct)
        {
            Writes++;
            return inner.AppendCommandAsync(matchId, expectedSequence, commandWire, issuerSteamId, acceptedAt, ct);
        }

        public Task<MatchJournal?> LoadJournalAsync(Guid matchId, CancellationToken ct)
        {
            Reads++;
            return inner.LoadJournalAsync(matchId, ct);
        }

        public Task<bool> TryCompleteMatchAsync(Guid matchId, MatchStatus terminal, int? winnerSeat,
            DateTimeOffset completedAt, CancellationToken ct)
        {
            Writes++;
            return inner.TryCompleteMatchAsync(matchId, terminal, winnerSeat, completedAt, ct);
        }

        public Task TouchAsync(Guid matchId, string? steamId, DateTimeOffset seenAt, CancellationToken ct)
        {
            Writes++;
            return inner.TouchAsync(matchId, steamId, seenAt, ct);
        }

        public Task<IReadOnlyList<Guid>> ListOpenMatchIdsAsync(CancellationToken ct)
        {
            Reads++;
            return inner.ListOpenMatchIdsAsync(ct);
        }

        public Task StoreJoinCredentialAsync(byte[] credentialHash, Guid matchId, string steamId,
            DateTimeOffset expiresAt, CancellationToken ct)
        {
            Writes++;
            return inner.StoreJoinCredentialAsync(credentialHash, matchId, steamId, expiresAt, ct);
        }

        public Task<JoinCredentialRecord?> FindJoinCredentialAsync(byte[] credentialHash, CancellationToken ct)
        {
            Reads++;
            return inner.FindJoinCredentialAsync(credentialHash, ct);
        }

        public Task RevokeJoinCredentialsAsync(Guid matchId, string steamId, DateTimeOffset revokedAt, CancellationToken ct)
        {
            Writes++;
            return inner.RevokeJoinCredentialsAsync(matchId, steamId, revokedAt, ct);
        }

        public Task<bool> ReplaceJoinCredentialAsync(byte[] credentialHash, Guid matchId, string steamId,
            DateTimeOffset expiresAt, DateTimeOffset now, CancellationToken ct,
            TimeSpan? allowTerminalWithin = null)
        {
            Writes++;
            return inner.ReplaceJoinCredentialAsync(
                credentialHash, matchId, steamId, expiresAt, now, ct, allowTerminalWithin);
        }
    }
}

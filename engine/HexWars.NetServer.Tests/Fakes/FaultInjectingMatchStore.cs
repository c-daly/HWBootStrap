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
        Exception? _afterNextAppendFailure;
        Exception? _nextCompletionFailure;
        int _nextCompletionRemaining;
        Exception? _everyCompletionFailure;
        Exception? _afterCompletionFailure;
        int _afterCompletionRemaining;
        Exception? _nextJournalReadFailure;
        Exception? _nextGetMatchFailure;

        /// <summary>How many appends actually reached the inner store.</summary>
        public int AppendsForwarded { get; private set; }

        /// <summary>How many appends were refused by the injected fault.</summary>
        public int AppendsFailed { get; private set; }

        /// <summary>How many completions actually reached the inner store.</summary>
        public int CompletionsForwarded { get; private set; }

        /// <summary>Runs just before a completion is forwarded. The seam a test needs to end a match from
        /// somewhere else in the window the coordinator is trying to complete it in.</summary>
        public Func<Task>? BeforeCompletion { get; set; }

        /// <summary>Arms the next AppendCommandAsync to throw, once. Null disarms it.</summary>
        public void FailNextAppend(Exception? failure) => _nextAppendFailure = failure;

        /// <summary>
        /// Arms the next AppendCommandAsync to commit and THEN throw, once.
        ///
        /// This is the ambiguous commit, and it is a different fault from FailNextAppend in the only way
        /// that matters: the row is there. A statement that timed out on the way back, or a connection lost
        /// after the server committed, both reach the caller as an exception over a journal that has
        /// already moved. Answering it with TemporaryFailure invites the client to send the command again,
        /// which is how one move becomes two.
        /// </summary>
        public void ThrowAfterNextAppend(Exception? failure) => _afterNextAppendFailure = failure;

        /// <summary>Arms the next TryCompleteMatchAsync calls to throw, <paramref name="times"/> of them.
        /// Two is the interesting number: the coordinator retries a failed completion once immediately, so
        /// one failure is invisible to everybody and two is a database it cannot finish the game against.</summary>
        public void FailNextCompletion(Exception? failure, int times = 1)
        {
            _nextCompletionFailure = failure;
            _nextCompletionRemaining = failure is null ? 0 : times;
        }

        /// <summary>Fails EVERY TryCompleteMatchAsync until disarmed with null: a database this host cannot
        /// finish a game against, rather than one bad moment it can retry through.</summary>
        public void FailEveryCompletion(Exception? failure) => _everyCompletionFailure = failure;

        /// <summary>
        /// Lets the next completions COMMIT and then throws, <paramref name="times"/> of them.
        ///
        /// The difference from FailEveryCompletion is the whole point: the row really is terminal and the
        /// caller has no way of knowing. It is the shape of a response lost on the way back, and it is how
        /// a projection ends up stale over a match that has already finished.
        /// </summary>
        public void ThrowAfterNextComplete(Exception? failure, int times = 1)
        {
            _afterCompletionFailure = failure;
            _afterCompletionRemaining = failure is null ? 0 : times;
        }

        /// <summary>Arms the next LoadJournalAsync to throw, once. What makes an ambiguous commit
        /// unverifiable rather than merely uncertain.</summary>
        public void FailNextJournalRead(Exception? failure) => _nextJournalReadFailure = failure;

        /// <summary>Arms the next GetMatchAsync to throw, once.</summary>
        public void FailNextGetMatch(Exception? failure) => _nextGetMatchFailure = failure;

        public async Task<AppendResult> AppendCommandAsync(Guid matchId, int expectedSequence,
            string commandWire, string issuerSteamId, DateTimeOffset acceptedAt, CancellationToken ct)
        {
            Exception? failure = _nextAppendFailure;
            if (failure is not null)
            {
                _nextAppendFailure = null;
                AppendsFailed++;

                // Thrown rather than returned: an outage is not one of the outcomes AppendResult can carry,
                // and turning it into one here would test a code path production never takes.
                throw failure;
            }

            AppendsForwarded++;
            AppendResult result = await inner
                .AppendCommandAsync(matchId, expectedSequence, commandWire, issuerSteamId, acceptedAt, ct)
                .ConfigureAwait(false);

            Exception? afterwards = _afterNextAppendFailure;
            if (afterwards is null) return result;

            _afterNextAppendFailure = null;
            AppendsFailed++;
            throw afterwards;
        }

        // ---- everything else is a plain forward -------------------------------

        public Task<CreateMatchResult> CreateMatchForLobbyAsync(CreateMatchRequest request, CancellationToken ct) =>
            inner.CreateMatchForLobbyAsync(request, ct);

        public Task<PersistedMatch?> GetMatchAsync(Guid matchId, CancellationToken ct)
        {
            Exception? failure = _nextGetMatchFailure;
            if (failure is null) return inner.GetMatchAsync(matchId, ct);

            _nextGetMatchFailure = null;
            return Task.FromException<PersistedMatch?>(failure);
        }

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

        public Task<MatchJournal?> LoadJournalAsync(Guid matchId, CancellationToken ct)
        {
            Exception? failure = _nextJournalReadFailure;
            if (failure is null) return inner.LoadJournalAsync(matchId, ct);

            _nextJournalReadFailure = null;
            return Task.FromException<MatchJournal?>(failure);
        }

        public async Task<bool> TryCompleteMatchAsync(Guid matchId, MatchStatus terminal, int? winnerSeat,
            DateTimeOffset completedAt, CancellationToken ct)
        {
            if (_everyCompletionFailure is not null) throw _everyCompletionFailure;

            Exception? failure = _nextCompletionFailure;
            if (failure is not null && _nextCompletionRemaining > 0)
            {
                _nextCompletionRemaining--;
                if (_nextCompletionRemaining == 0) _nextCompletionFailure = null;
                throw failure;
            }

            if (BeforeCompletion is not null) await BeforeCompletion().ConfigureAwait(false);

            CompletionsForwarded++;
            bool completed = await inner
                .TryCompleteMatchAsync(matchId, terminal, winnerSeat, completedAt, ct)
                .ConfigureAwait(false);

            if (_afterCompletionRemaining <= 0 || _afterCompletionFailure is null) return completed;

            _afterCompletionRemaining--;
            throw _afterCompletionFailure;
        }

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

        public Task<CredentialReplacement> ReplaceJoinCredentialAsync(byte[] credentialHash, Guid matchId,
            string steamId, DateTimeOffset expiresAt, DateTimeOffset now, CancellationToken ct,
            TimeSpan? allowTerminalWithin = null) =>
            inner.ReplaceJoinCredentialAsync(
                credentialHash, matchId, steamId, expiresAt, now, ct, allowTerminalWithin);
    }
}

using HexWars.NetServer.Persistence;

namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// Where a match comes from when this process does not have it yet.
    ///
    /// It is an interface rather than a method on the coordinator because the answer changes: today it is a
    /// read of the journal, and once startup recovery exists it also has to decide whether a match is worth
    /// hosting at all - too old, an engine version this build cannot replay, a journal it refuses. The
    /// coordinator does not need to know which of those is happening; it needs a projection or an exception.
    /// </summary>
    public interface ILiveMatchLoader
    {
        /// <summary>The current projection of one match.</summary>
        /// <exception cref="KeyNotFoundException">No match has that id.</exception>
        /// <exception cref="InvalidOperationException">The journal cannot be replayed.</exception>
        Task<LiveMatch> LoadAsync(Guid matchId, CancellationToken ct);
    }

    /// <summary>The plain loader: read the journal, replay it, hand it over.</summary>
    public sealed class JournalLiveMatchLoader(IMatchStore store) : ILiveMatchLoader
    {
        public async Task<LiveMatch> LoadAsync(Guid matchId, CancellationToken ct)
        {
            MatchJournal? journal = await store.LoadJournalAsync(matchId, ct).ConfigureAwait(false);

            if (journal is null)
                throw new KeyNotFoundException("no match has id " + matchId);

            return LiveMatch.FromJournal(journal);
        }
    }
}

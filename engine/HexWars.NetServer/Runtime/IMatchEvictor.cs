namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// Closing the sockets of matches that are over, seen from the outside.
    ///
    /// It exists as an interface for one reason: the retention sweeper has to keep running when eviction
    /// does not, and a service that can only be handed the real coordinator cannot be shown to survive an
    /// eviction that throws or one that never returns. The implementation is
    /// <see cref="DurableMatchCoordinator"/> and there is exactly one instance of it.
    /// </summary>
    public interface IMatchEvictor
    {
        /// <summary>
        /// Closes every connection of the named matches and drops them from memory.
        ///
        /// Bounded rather than best-effort: a match whose gate cannot be taken within the implementation
        /// deadline is left exactly as it was and named in the result, so the caller can try it again rather
        /// than lose it. That is the difference between a slow eviction and a lost one.
        /// </summary>
        /// <returns>The ids that could NOT be evicted, in the order they were offered. Empty when every
        /// match named was either evicted or was not live on this host at all.</returns>
        Task<IReadOnlyList<Guid>> EvictAsync(
            IEnumerable<Guid> matchIds, int closeStatus, string reason, CancellationToken ct);
    }
}

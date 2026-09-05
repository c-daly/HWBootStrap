namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// Why a stored match cannot be turned back into a game.
    ///
    /// These are kept apart because they call for different answers, not because the code needs the
    /// distinction. A sequence gap or a command that will not replay means the record itself is damaged and
    /// somebody has to look at it. An unsupported engine contract or protocol means the record is fine and
    /// this build is the wrong one to host it - rolling forward, or back, fixes every affected match at
    /// once. A startup report that said only "could not load" would leave an operator unable to tell a
    /// deploy mistake from data loss.
    /// </summary>
    public enum MatchRecoveryFailure
    {
        /// <summary>No journal has this id.</summary>
        NotFound,

        /// <summary>Written under engine rules this build cannot reproduce.</summary>
        UnsupportedEngineContract,

        /// <summary>Written for a wire protocol this host does not speak.</summary>
        UnsupportedProtocol,

        /// <summary>The match is active but its start state is missing or does not parse.</summary>
        CorruptStartState,

        /// <summary>A stored command is not a command.</summary>
        CorruptCommand,

        /// <summary>A stored command parses, and the engine refuses to apply it.</summary>
        CommandReplayFailed,

        /// <summary>The stored sequences are not 1..n, so commands are missing or duplicated.</summary>
        SequenceGap,

        /// <summary>A command claims a seat its stored issuer does not hold.</summary>
        IssuerSeatMismatch,
    }

    /// <summary>
    /// A match this process refuses to host, and the reason an operator needs.
    ///
    /// Thrown rather than returned because every caller has the same choice: a projection, or no seat.
    /// There is no partial recovery worth having - a match replayed up to the point it stopped making sense
    /// would deal clients a state that disagrees with the record they will be judged against.
    /// </summary>
    public sealed class MatchRecoveryException(MatchRecoveryFailure failure, Guid matchId, string detail)
        : Exception("match " + matchId + " cannot be recovered (" + failure + "): " + detail)
    {
        public MatchRecoveryFailure Failure { get; } = failure;

        public Guid MatchId { get; } = matchId;

        /// <summary>The specifics: which sequence, which version, which seat. Safe to log - it never
        /// carries a credential, and it names seats rather than Steam ids.</summary>
        public string Detail { get; } = detail;
    }
}

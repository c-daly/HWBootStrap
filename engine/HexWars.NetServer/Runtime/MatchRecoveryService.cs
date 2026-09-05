using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Runtime
{
    /// <summary>What one startup verification pass found.</summary>
    /// <param name="Verified">Open matches this build can replay and would host.</param>
    /// <param name="Failed">The ones it would refuse, with the reason each needs.</param>
    /// <param name="CompletedAt">When the pass finished. Readiness reports it so an operator can tell a
    /// pass that ran from one that never got the chance.</param>
    public sealed record RecoveryReport(
        int Verified,
        IReadOnlyList<(Guid MatchId, MatchRecoveryFailure Failure, string Detail)> Failed,
        DateTimeOffset CompletedAt);

    /// <summary>
    /// The loader that has an opinion. It reads the journal, decides whether this build is entitled to host
    /// the match at all, and replays it - refusing with a reason rather than an exception nobody can act on.
    ///
    /// The two contract checks come first and cost nothing. A journal written under engine rules this build
    /// no longer reproduces, or for a protocol it does not speak, would replay to a state that quietly
    /// disagrees with the record; catching that by version is the only way to catch it at all, because a
    /// changed rule produces a perfectly valid-looking wrong answer.
    ///
    /// Everything after that is checked before the projection is built, not while it is being built. The
    /// difference matters at startup: this class is what tells an operator which matches need attention and
    /// why, and a half-built projection thrown away mid-replay can only report that something went wrong.
    /// </summary>
    public sealed class MatchRecoveryService(
        IMatchStore store,
        IOptions<MatchHostingOptions> options,
        ILogger<MatchRecoveryService> logger) : ILiveMatchLoader
    {
        /// <summary>
        /// The current projection of one match, or a refusal naming what is wrong with it.
        ///
        /// Terminal matches load like any other. Whether somebody may sit down in a finished game is the
        /// a question for the coordinator, and answering it here would mean a completed match could not be read
        /// back for a scoreboard or an investigation.
        /// </summary>
        /// <exception cref="MatchRecoveryException">This build will not host this match.</exception>
        public async Task<LiveMatch> LoadAsync(Guid matchId, CancellationToken ct)
        {
            MatchJournal? journal = await store.LoadJournalAsync(matchId, ct).ConfigureAwait(false);

            if (journal is null)
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.NotFound, matchId, "no match has this id");

            return Verify(journal);
        }

        /// <summary>
        /// Replays every open match once, at startup, so the answer to "can this host serve the games it is
        /// responsible for" is known before the first player asks rather than at their expense.
        ///
        /// A refusal is collected and logged; a store failure is not. They look alike from here and mean
        /// opposite things: a bad journal is one match that needs a human, a database that will not answer
        /// is every match, and the pass has learned nothing about any of them. Letting the second one
        /// through is what keeps readiness false instead of reporting an all-clear it did not earn.
        /// </summary>
        public async Task<RecoveryReport> VerifyOpenMatchesAsync(CancellationToken ct)
        {
            IReadOnlyList<Guid> open = await store.ListOpenMatchIdsAsync(ct).ConfigureAwait(false);

            var failed = new List<(Guid MatchId, MatchRecoveryFailure Failure, string Detail)>();
            int verified = 0;

            foreach (Guid matchId in open)
            {
                try
                {
                    await LoadAsync(matchId, ct).ConfigureAwait(false);
                    verified++;
                }
                catch (MatchRecoveryException refusal)
                {
                    failed.Add((matchId, refusal.Failure, refusal.Detail));
                    logger.LogError(
                        "Match {MatchId} cannot be recovered: {Failure} {Detail} - maintenance required",
                        Short(matchId), refusal.Failure, refusal.Detail);
                }
            }

            logger.LogInformation(
                "Startup recovery verified {Verified} open match(es) and refused {Refused}",
                verified, failed.Count);

            return new RecoveryReport(verified, failed, DateTimeOffset.UtcNow);
        }

        /// <summary>
        /// Judges a journal, then builds it. The build is <see cref="LiveMatch.FromJournal"/> and not a
        /// second implementation on purpose: two replays that could drift apart would make the verified
        /// projection and the hosted one different objects with the same name. The cost is one extra pass
        /// over a command list that is already in memory, against a database round trip that is not.
        /// </summary>
        LiveMatch Verify(MatchJournal journal)
        {
            PersistedMatch row = journal.Match;
            Guid matchId = row.MatchId;

            if (!EngineContract.SupportedVersions.Contains(row.EngineVersion))
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.UnsupportedEngineContract, matchId,
                    "the journal was written under " + row.EngineVersion + ", this build replays "
                    + string.Join(", ", EngineContract.SupportedVersions));

            if (row.ProtocolVersion != options.Value.ProtocolVersion)
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.UnsupportedProtocol, matchId,
                    "the journal speaks protocol " + row.ProtocolVersion + ", this host speaks "
                    + options.Value.ProtocolVersion);

            if (row.Status == MatchStatus.Active) VerifyReplay(journal);

            try
            {
                return LiveMatch.FromJournal(journal);
            }
            catch (Exception unexpected)
            {
                // Unreachable if the checks above are exhaustive, which is exactly why it is here: an
                // unclassified refusal escaping as a raw InvalidOperationException would reach a player as
                // a bare unavailable, and the startup report would not mention the match at all.
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.CommandReplayFailed, matchId,
                    "the journal would not rebuild: " + unexpected.Message);
            }
        }

        /// <summary>Every way an active journal can fail to be a game, in the order they can be detected.</summary>
        static void VerifyReplay(MatchJournal journal)
        {
            Guid matchId = journal.Match.MatchId;

            if (journal.Match.StartReplay is null)
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.CorruptStartState, matchId,
                    "the match is active and has no start state");

            GameState state;
            try
            {
                state = ReplayFile.Read(journal.Match.StartReplay).Start;
            }
            catch (Exception malformed)
            {
                // Deliberately every exception, not FormatException: a truncated replay reaches the reader
                // as an index or a parse failure depending on which line it stops at, and all of them mean
                // the same thing to an operator.
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.CorruptStartState, matchId,
                    "the stored start state does not parse: " + malformed.Message);
            }

            // The whole sequence first. A hole means commands are missing from the middle of a game, and
            // reporting the command that happens to sit at the hole would send an operator to the wrong row.
            int expected = 1;
            foreach (PersistedCommand stored in journal.Commands)
            {
                if (stored.Sequence != expected)
                    throw new MatchRecoveryException(
                        MatchRecoveryFailure.SequenceGap, matchId,
                        "expected sequence " + expected + " but the journal holds " + stored.Sequence);

                expected++;
            }

            var seats = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (PersistedPlayer player in journal.Players) seats[player.SteamId] = player.Seat;

            foreach (PersistedCommand stored in journal.Commands)
            {
                if (!CommandWire.TryRead(stored.CommandWire, out Command? command) || command is null)
                    throw new MatchRecoveryException(
                        MatchRecoveryFailure.CorruptCommand, matchId,
                        "the command at sequence " + stored.Sequence + " does not parse");

                // The row says who sent it and the payload says who played it. When they disagree the
                // journal is describing a move somebody was not entitled to make, and replaying it would
                // hand that move to whichever seat the payload names.
                if (!seats.TryGetValue(stored.IssuerSteamId, out int seat))
                    throw new MatchRecoveryException(
                        MatchRecoveryFailure.IssuerSeatMismatch, matchId,
                        "the command at sequence " + stored.Sequence
                        + " was written by a player who holds no seat in this match");

                if (seat != (int)command.Issuer)
                    throw new MatchRecoveryException(
                        MatchRecoveryFailure.IssuerSeatMismatch, matchId,
                        "the command at sequence " + stored.Sequence + " claims seat "
                        + (int)command.Issuer + " but its row belongs to seat " + seat);

                Result applied = GameEngine.Apply(state, command);
                if (!applied.Success)
                    throw new MatchRecoveryException(
                        MatchRecoveryFailure.CommandReplayFailed, matchId,
                        "the engine refused the command at sequence " + stored.Sequence + " with "
                        + applied.Reason);

                state = applied.NewState;
            }
        }

        /// <summary>Match ids reach logs as their first eight hex characters, the same shortening the
        /// coordinator and the credential service use, so one match can be followed across all three.</summary>
        static string Short(Guid matchId) => matchId.ToString("N")[..8];
    }
}

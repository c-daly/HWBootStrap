using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;

namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// Whether a journal is a game this build can host, and the projection it makes if it is.
    ///
    /// Pure, static and free of any store, clock or logger, because two things have to agree about it: the
    /// startup recovery pass, which decides whether this host may serve a match, and the read-only
    /// verify-journals verb an operator points at a restored backup. Two implementations of the same
    /// judgement would let a restore be verified against rules the running host does not apply, which is
    /// the one answer an operator must be able to trust.
    /// </summary>
    public static class JournalVerifier
    {
        /// <summary>
        /// Judges a journal, then builds it. The build is <see cref="LiveMatch.FromJournal"/> and not a
        /// second implementation on purpose: two replays that could drift apart would make the verified
        /// projection and the hosted one different objects with the same name. The cost is one extra pass
        /// over a command list that is already in memory, against a database round trip that is not.
        /// </summary>
        public static LiveMatch Verify(MatchJournal journal, int configuredProtocolVersion)
        {
            PersistedMatch row = journal.Match;
            Guid matchId = row.MatchId;

            if (!EngineContract.SupportedVersions.Contains(row.EngineVersion))
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.UnsupportedEngineContract, matchId,
                    "the journal was written under " + row.EngineVersion + ", this build replays "
                    + string.Join(", ", EngineContract.SupportedVersions));

            // The supported set rather than the configured number. A host has the code for every version
            // in the set, so refusing a row it can replay perfectly would strand matches over a deployment
            // setting rather than over anything about the journal.
            if (!ProtocolContract.SupportedVersions.Contains(row.ProtocolVersion))
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.UnsupportedProtocol, matchId,
                    "the journal speaks protocol " + row.ProtocolVersion + ", this build speaks "
                    + ProtocolContract.SupportedList);

            // And the configured value has to be one this build speaks. Options validation refuses anything
            // else at startup; this is the second lock, for a host that reached here some other way.
            if (!ProtocolContract.SupportedVersions.Contains(configuredProtocolVersion))
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.UnsupportedProtocol, matchId,
                    HexWarsConfiguration.MatchProtocolVersionKey + " is " + configuredProtocolVersion
                    + ", which is not one of " + ProtocolContract.SupportedList);

            if (row.Status == MatchStatus.Active && row.StartReplay is null)
                throw new MatchRecoveryException(
                    MatchRecoveryFailure.CorruptStartState, matchId,
                    "the match is active and has no start state");

            // Every journal that has a start state is verified, terminal ones included. A completed match
            // is served for the length of the reconnect window and an abandoned one may still be read back,
            // so a corrupt log in either has to be classified here rather than surface later as a
            // projection that quietly disagrees with the record.
            if (row.StartReplay is not null) VerifyReplay(journal);

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
    }
}

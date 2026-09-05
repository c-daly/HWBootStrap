using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Runtime
{
    /// <summary>What one startup verification pass found.</summary>
    /// <param name="Verified">Open matches this build can replay and would host.</param>
    /// <param name="Healed">Matches the pass found finished in the engine and unfinished in the store, and
    /// closed. Reported separately from Verified because it is a write: an operator seeing a non-zero count
    /// is being told this host repaired something, which is worth knowing even though nothing went wrong.</param>
    /// <param name="Failed">The ones it would refuse, with the reason each needs.</param>
    /// <param name="CompletedAt">When the pass finished. Readiness reports it so an operator can tell a
    /// pass that ran from one that never got the chance.</param>
    public sealed record RecoveryReport(
        int Verified,
        int Healed,
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
        TimeProvider time,
        MatchMetrics metrics,
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
            // Untagged here on purpose. This method is the shared ILiveMatchLoader: the startup pass
            // calls it, and so does every live handshake and every stale reload. Counting a recovery
            // failure here tagged an outage during a normal reconnect as a startup problem, and counted
            // it twice, because the caller counts it as load or reload as well. The startup pass tags its
            // own attempt instead, where the context is actually known.
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
            IReadOnlyList<Guid> open;
            try
            {
                open = await store.ListOpenMatchIdsAsync(ct).ConfigureAwait(false);
            }
            catch (Exception unreadable)
            {
                metrics.DbFailure(MatchMetrics.DbOp.RecoveryList);
                throw;
            }


            var failed = new List<(Guid MatchId, MatchRecoveryFailure Failure, string Detail)>();
            int verified = 0;
            int healed = 0;

            foreach (Guid matchId in open)
            {
                LiveMatch live;
                try
                {
                    live = await LoadAsync(matchId, ct).ConfigureAwait(false);
                    verified++;
                }
                catch (MatchRecoveryException refusal)
                {
                    failed.Add((matchId, refusal.Failure, refusal.Detail));
                    logger.LogError(
                        "Match {MatchId} cannot be recovered: {Failure} {Detail} - maintenance required",
                        Short(matchId), refusal.Failure, refusal.Detail);
                    continue;
                }
                catch (Exception)
                {
                    // A store that will not answer, on the startup path specifically. Rethrown so the
                    // caller keeps telling an outage apart from a bad journal, and tagged here so the
                    // database counter says which pass it happened to.
                    metrics.DbFailure(MatchMetrics.DbOp.RecoveryLoad);
                    throw;
                }

                if (await HealAsync(live, ct).ConfigureAwait(false)) healed++;

            }

            // Counted here rather than where the report is read, so a pass that nobody consults is still
            // on the record. A refused match is the one finding in this whole class that needs a human.
            metrics.RecoveryFailure(failed.Count);

            logger.LogInformation(
                "Startup recovery verified {Verified} open match(es), healed {Healed} and refused {Refused}",
                verified, healed, failed.Count);

            return new RecoveryReport(verified, healed, failed, time.GetUtcNow());
        }

        /// <summary>
        /// Closes a match the journal says is still being played and the engine says is over.
        ///
        /// It is the same repair the coordinator performs on the next handshake, done at startup instead so
        /// it does not wait for one. Nothing guarantees a player ever comes back to a finished game, and a
        /// match left active is one the retention reaper will eventually mark abandoned - turning a game
        /// somebody won into a game the record says nobody finished.
        ///
        /// A failure here is deliberately not fatal and not collected as a refusal: the journal is intact,
        /// the match is hostable, and the next handshake tries the same write again.
        /// </summary>
        async Task<bool> HealAsync(LiveMatch live, CancellationToken ct)
        {
            if (live.Status != MatchStatus.Active) return false;
            if (live.State is null || !live.State.IsGameOver) return false;

            int? winnerSeat = live.State.Winner is PlayerId winner ? (int)winner : null;

            try
            {
                bool closed = await store
                    .TryCompleteMatchAsync(
                        live.MatchId, MatchStatus.Completed, winnerSeat, time.GetUtcNow(), ct)
                    .ConfigureAwait(false);

                if (!closed) return false;

                logger.LogWarning(
                    "Match {MatchId} was finished in the journal and open in the store; closed it at startup",
                    Short(live.MatchId));

                return true;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                metrics.DbFailure(MatchMetrics.DbOp.RecoveryHeal);
                logger.LogError(failure,
                    "Match {MatchId} is finished and could not be closed at startup", Short(live.MatchId));
                return false;
            }
        }

        /// <summary>Judges a journal and builds it, through the verifier the read-only verify-journals
        /// verb also uses. One implementation on purpose: two that could drift apart would let an
        /// operator verify a restored copy against rules the running host does not apply.</summary>
        LiveMatch Verify(MatchJournal journal) =>
            JournalVerifier.Verify(journal, options.Value.ProtocolVersion);

        /// <summary>Match ids reach logs as their first eight hex characters, the same shortening the
        /// coordinator and the credential service use, so one match can be followed across all three.</summary>
        static string Short(Guid matchId) => matchId.ToString("N")[..8];
    }
}

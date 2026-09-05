using System.Collections.Concurrent;
using HexWars.Engine;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Steam;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// The authoritative brain of protocol v2: it seats players, deals the start state, and turns each
    /// command into a durable journal entry and then a broadcast - in that order, always.
    ///
    /// The order is the entire design. A player who is told APPLY has seen the future of the match, so if
    /// the write that was supposed to make it true never landed, this process and every client have
    /// permanently disagreed about the game. Appending first means the only failure a player can experience
    /// is being told to try again, which is recoverable, instead of watching a move that the next reconnect
    /// will erase. The same reasoning ends a game: the match is recorded as finished before the winning
    /// APPLY goes out, never after.
    ///
    /// It owns no socket. Everything it decides to send goes through <see cref="IConnectionSink"/>, which is
    /// what lets a test stand between the journal and the broadcast and check which happened first.
    ///
    /// Concurrency is one gate per match, held across evaluate-append-advance-broadcast. Anything narrower
    /// would let two commands agree on the same sequence; anything wider would make one slow database write
    /// stall every other game this process is hosting.
    /// </summary>
    public sealed class DurableMatchCoordinator(
        IMatchStore store,
        IMatchCredentialService credentials,
        ILiveMatchLoader loader,
        IConnectionSink sink,
        IOptions<MatchHostingOptions> options,
        TimeProvider time,
        ILogger<DurableMatchCoordinator> logger)
    {
        /// <summary>How long a match with no connections is kept in memory before it is released. It is not
        /// an abandonment rule: nothing durable changes, the next player to arrive simply reloads it.</summary>
        public static readonly TimeSpan IdleEvictionWindow = TimeSpan.FromMinutes(10);

        /// <summary>
        /// How long a match this build refuses to replay stays refused without being tried again.
        ///
        /// Refusing is the most expensive answer this server has - it costs a full journal read and replay
        /// to reach - and it is the answer a reconnecting client asks for repeatedly, because from its side
        /// an unavailable match looks exactly like a server that is coming back. Long enough that a backoff
        /// storm reads the journal once; short enough that a match repaired by hand starts working again on
        /// its own rather than after a deploy.
        /// </summary>
        public static readonly TimeSpan UnrecoverableRetryWindow = TimeSpan.FromSeconds(60);

        /// <summary>The frame has no seat behind it: the connection never authenticated, or its match has
        /// been released.</summary>
        public const string RejectNoSeat = "REJECT NoSeat";

        /// <summary>A seat tried to issue a command on behalf of the other one.</summary>
        public const string RejectWrongSeat = "REJECT WrongSeat";

        /// <summary>A barracks arrived after the start state was dealt, which is baked into it.</summary>
        public const string RejectCatalogClosed = "REJECT CatalogClosed";

        /// <summary>A command arrived for a match that has not started yet. Temporary by nature: the same
        /// command will be accepted once both barracks are in.</summary>
        public const string RejectCatalogV1Required = "REJECT CatalogV1Required";

        /// <summary>The durable write did not happen. The client may send the same command again.</summary>
        public const string RejectTemporaryFailure = "REJECT TemporaryFailure";

        /// <summary>
        /// A command arrived for a match the journal calls finished.
        ///
        /// Kept apart from CatalogV1Required because the two invite opposite behaviour. A match that has
        /// not started yet will accept this command once it does; a match that has ended never will, and a
        /// client that cannot tell them apart retries a move into a game that is over.
        /// </summary>
        public const string RejectMatchEnded = "REJECT MatchEnded";

        /// <summary>The close a socket gets when the match under it has ended in a status this host cannot
        /// go on serving. 1000 rather than an error code: nothing went wrong, the game is simply over.</summary>
        public const int MatchEndedCloseStatus = 1000;

        public const string MatchEndedCloseReason = "match ended";

        /// <summary>The close the older socket of a seat gets when that seat authenticates again.</summary>
        public const int SupersededCloseStatus = 1000;

        public const string SupersededCloseReason = "superseded";

        /// <summary>
        /// The close a socket gets when this host can no longer say what it has seen.
        ///
        /// 1011 rather than a refusal: the client did nothing wrong, the server did. It is the one answer
        /// that cannot be misread as an invitation to send the command again - the protocol rule after a
        /// disconnect is to reconnect and fast-forward from START, which is exactly what resolves the
        /// ambiguity.
        /// </summary>
        public const int ResyncCloseStatus = 1011;

        public const string ResyncCloseReason = "resync required";

        /// <summary>The credential, the match id or the match itself will not do. Deliberately one code for
        /// all of them: the caller is unauthenticated, so a finer answer would only help a guesser.</summary>
        public const string AuthFailInvalid = "invalid";

        /// <summary>The match exists but this process cannot load it right now. A retry may work.</summary>
        public const string AuthFailUnavailable = "unavailable";

        const string CatalogMessage = "CATALOG";
        const string CommandMessage = "CMD";

        /// <summary>
        /// Matches this process is hosting, each behind a Lazy so two connections arriving together load it
        /// once rather than racing to build two projections of the same game.
        /// </summary>
        readonly ConcurrentDictionary<Guid, Lazy<Task<LiveMatch>>> _matches = new();

        /// <summary>Connection id to match id, so an inbound frame can find its game without the socket
        /// layer having to remember anything.</summary>
        readonly ConcurrentDictionary<string, Guid> _connections = new(StringComparer.Ordinal);

        /// <summary>Matches this build has refused to replay, and when. Read on every handshake so a
        /// journal that cannot be recovered is read once per window rather than once per reconnect.</summary>
        readonly ConcurrentDictionary<Guid, DateTimeOffset> _unrecoverable = new();

        /// <summary>The protocol version this host speaks, as the websocket route reports it.</summary>
        public int ProtocolVersion { get; } = options.Value.ProtocolVersion;

        /// <summary>
        /// Whether the handshake worked, the seat it seated, and - when it did not - why not.
        ///
        /// The credential hash and its expiry ride along on a success so the socket layer can keep asking,
        /// for as long as the connection lives, whether the credential behind it is still good. The hash
        /// rather than the credential: it answers that one question and is worth nothing else.
        /// </summary>
        public sealed record AuthOutcome(
            bool Ok,
            int Seat,
            string? FailCode,
            byte[]? CredentialHash = null,
            DateTimeOffset? CredentialExpiresAt = null);

        /// <summary>What recording the end of a game concluded. <c>Advance</c> false means the caller must
        /// change nothing: the issuer has already been told why.</summary>
        readonly record struct CompletionOutcome(bool Advance, MatchStatus Status);

        /// <summary>What the journal says about a command this process could not confirm it wrote.</summary>
        enum JournalCheck
        {
            /// <summary>The row is there. The write landed and the clients are owed it.</summary>
            Present,

            /// <summary>The row is not there. The write did not land and the command is still usable.</summary>
            Absent,

            /// <summary>The journal could not be read, so neither answer can be given honestly.</summary>
            Unknown,
        }

        /// <summary>
        /// Runs between finding a cached match and taking its gate. Null in production.
        ///
        /// A seam for the one race that cannot be provoked from outside: the sweeper releasing a projection
        /// a handshake is already committed to. Without a hook a test would have to win a scheduling race
        /// to observe it, which is the same as not testing it.
        /// </summary>
        internal Func<Guid, Task>? BeforeGateForTest { get; set; }

        // ---- diagnostics -----------------------------------------------------

        /// <summary>Matches held in memory right now.</summary>
        public int LiveMatchCount => _matches.Count;

        /// <summary>Sockets seated across every match held in memory.</summary>
        public int ConnectionCount
        {
            get
            {
                int total = 0;
                foreach (LiveMatch match in Cached()) total += match.Connections.Count;
                return total;
            }
        }

        /// <summary>The connections watching one match, or nothing when it is not held in memory.</summary>
        public IReadOnlyCollection<string> ConnectionsOf(Guid matchId) =>
            TryGetLiveMatch(matchId, out LiveMatch? match)
                ? match!.Connections.Keys.ToArray()
                : Array.Empty<string>();

        /// <summary>The projection of one match, when this process is holding it and has finished loading it.</summary>
        public bool TryGetLiveMatch(Guid matchId, out LiveMatch? match)
        {
            match = null;
            if (!_matches.TryGetValue(matchId, out Lazy<Task<LiveMatch>>? entry)) return false;
            if (!entry.IsValueCreated || !entry.Value.IsCompletedSuccessfully) return false;

            match = entry.Value.Result;
            return true;
        }

        // ---- the handshake ---------------------------------------------------

        /// <summary>
        /// Turns the first frame of a v2 socket into a seat.
        ///
        /// The credential decides who this is; the projection decides what they get told. A player joining a
        /// match that has not started is asked for their barracks, and one joining a match already in
        /// progress is dealt the start state plus every command since - which is the same message a first
        /// connect gets, because a reconnect is not a special case, it is the only case.
        /// </summary>
        public async Task<AuthOutcome> AuthenticateAsync(
            string connectionId, string matchIdText, string credential, CancellationToken ct)
        {
            if (!Guid.TryParse(matchIdText, out Guid matchId))
            {
                logger.LogDebug("A v2 socket offered something that is not a match id");
                return Failed(AuthFailInvalid);
            }

            using IDisposable? scope = MatchScope(matchId);

            // The credential is never logged, at any level, and neither is the fact that this particular
            // string was offered: the value below is only ever the Steam id it turned out to belong to.
            CredentialValidation? validation =
                await credentials.ValidateAsync(matchId, credential, ct).ConfigureAwait(false);

            if (validation is null)
            {
                logger.LogDebug("A v2 socket offered a credential this match will not accept");
                return Failed(AuthFailInvalid);
            }

            // Checked after the credential, so a bad credential is still answered as invalid rather than
            // as this match being unavailable, and before the load, which is the expensive part being
            // avoided.
            if (IsStillRefused(matchId))
            {
                logger.LogDebug("Turned a player away: this match was refused recently and is not retried yet");
                return Failed(AuthFailUnavailable);
            }

            LiveMatch match;
            try
            {
                match = await EnterAsync(matchId, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (MatchRecoveryException refusal)
            {
                // Logged at Error and only once per window: this is an operator problem, not a player one,
                // and a reconnect storm against a broken journal must not also be a log storm.
                _unrecoverable[matchId] = time.GetUtcNow();
                logger.LogError(
                    "Refusing every seat in this match: {Failure} {Detail} - maintenance required",
                    refusal.Failure, refusal.Detail);
                return Failed(AuthFailUnavailable);
            }
            catch (Exception failure)
            {
                logger.LogWarning(failure,
                    "Turned {Player} away: this match could not be loaded",
                    SteamLogRedaction.HashSteamId(validation.SteamId));
                return Failed(AuthFailUnavailable);
            }

            try
            {
                var dealt = false;

                if (match.Stale)
                {
                    ReloadOutcome rebuilt = await TryReloadAsync(match, ct).ConfigureAwait(false);
                    if (!rebuilt.Ok) return Failed(AuthFailUnavailable);

                    dealt = rebuilt.Redealt;
                }

                // Before anything is served. A process killed between the append of a winning command and
                // the row that records the win leaves a game the engine calls over and the database calls
                // active; the next player through the door finishes closing it, rather than being dealt a
                // match that can never end.
                await HealCompletionAsync(match, dealt, ct).ConfigureAwait(false);

                if (!CanSeat(match))
                {
                    logger.LogDebug("Turned a player away: this match is {Status}", match.Status);
                    return Failed(AuthFailInvalid);
                }

                int seat = validation.Seat;
                DateTimeOffset now = time.GetUtcNow();

                // One live socket per seat. A second AUTH on the same seat is far more often a reconnect
                // whose predecessor has not been noticed yet than it is two clients - and when it is two,
                // one credential fanning out into an unbounded number of sockets is the shape of an abuse
                // rather than of a game.
                SupersedeSeat(match, connectionId, validation.SteamId);

                match.Connections[connectionId] = validation.SteamId;
                match.LastConnectionAt = now;
                _connections[connectionId] = matchId;

                try
                {
                    await store.TouchAsync(matchId, validation.SteamId, now, ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception failure)
                {
                    // A liveness stamp that did not land costs this match a little of its reaper budget and
                    // nothing else. Refusing the connection over it would be strictly worse for the player.
                    logger.LogWarning(failure, "Could not record that a player is here");
                }

                sink.Send(connectionId, NetProtocol.Seat((PlayerId)seat));

                if (match.Status == MatchStatus.Waiting)
                {
                    // Both barracks already durable and no start state: this host, or the one before it,
                    // died between the second SaveCatalog and the TryStart. Nothing else will ever start
                    // this match - the catalogues have been sent and no client sends one twice - so the
                    // handshake finishes what the catalog frame began.
                    if (match.Catalogs.TryGetValue(0, out IReadOnlyList<UnitTemplate>? seat0Barracks)
                        && match.Catalogs.TryGetValue(1, out IReadOnlyList<UnitTemplate>? seat1Barracks))
                    {
                        logger.LogInformation(
                            "Both barracks were already stored and the match never started; resuming it");
                        await StartTheMatchAsync(match, connectionId, seat0Barracks, seat1Barracks, ct)
                            .ConfigureAwait(false);
                    }
                    else if (!match.Catalogs.ContainsKey(seat))
                    {
                        sink.Send(connectionId, NetProtocol.CatalogRequest);
                    }
                }
                else
                {
                    sink.Send(connectionId, NetProtocol.Start(match.StartReplayText()));
                }

                logger.LogInformation("Seated {Player} in seat {Seat} of a {Status} match",
                    SteamLogRedaction.HashSteamId(validation.SteamId), seat, match.Status);

                return new AuthOutcome(
                    true, seat, null, validation.CredentialHash, validation.ExpiresAt);
            }
            finally
            {
                match.Gate.Release();
            }
        }

        // ---- inbound frames --------------------------------------------------

        /// <summary>One frame from a seated connection.</summary>
        public async Task ReceiveAsync(string connectionId, string raw, CancellationToken ct)
        {
            if (!_connections.TryGetValue(connectionId, out Guid matchId)
                || !TryGetLiveMatch(matchId, out _))
            {
                sink.Send(connectionId, RejectNoSeat);
                return;
            }

            NetMessage message = NetProtocol.Parse(raw);

            // PONG and anything this build does not know about are liveness, nothing more. Answering an
            // unknown type would make adding one to the client a breaking change.
            if (message.Type is not (CatalogMessage or CommandMessage)) return;

            using IDisposable? scope = MatchScope(matchId);

            LiveMatch match;
            try
            {
                match = await EnterAsync(matchId, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                logger.LogWarning(failure, "A frame arrived for a match that could not be loaded");
                sink.Send(connectionId, RejectTemporaryFailure);
                return;
            }

            try
            {
                var dealt = false;

                if (match.Stale)
                {
                    ReloadOutcome rebuilt = await TryReloadAsync(match, ct).ConfigureAwait(false);
                    if (!rebuilt.Ok)
                    {
                        sink.Send(connectionId, RejectTemporaryFailure);
                        return;
                    }

                    dealt = rebuilt.Redealt;
                }

                // The same self-healing pass the handshake runs. A completion that threw left this match
                // finished in the engine and active in the database; the next frame either way is the one
                // that closes it.
                await HealCompletionAsync(match, dealt, ct).ConfigureAwait(false);

                if (!match.Connections.TryGetValue(connectionId, out string? steamId)
                    || !match.Seats.TryGetValue(steamId, out int seat))
                {
                    sink.Send(connectionId, RejectNoSeat);
                    return;
                }

                if (message.Type == CatalogMessage)
                    await ReceiveCatalogAsync(match, connectionId, steamId, seat, message.Payload, ct)
                        .ConfigureAwait(false);
                else
                    await ReceiveCommandAsync(match, connectionId, steamId, seat, message.Payload, ct)
                        .ConfigureAwait(false);
            }
            finally
            {
                match.Gate.Release();
            }
        }

        async Task ReceiveCatalogAsync(LiveMatch match, string connectionId, string steamId, int seat,
            string payload, CancellationToken ct)
        {
            if (match.Status != MatchStatus.Waiting)
            {
                sink.Send(connectionId, RejectCatalogClosed);
                return;
            }

            IReadOnlyList<UnitTemplate> catalog;
            try
            {
                catalog = BarracksWire.Read(payload);
            }
            catch (FormatException)
            {
                // The same forgiveness the legacy hub shows a garbled catalogue: a player with a broken
                // client still gets a playable army rather than a match that never starts.
                catalog = BarracksCatalog.Normalize(BarracksCatalog.DefaultTemplates);
            }

            // What is stored is what was understood, not what arrived. A payload that fell back to the
            // default barracks has to reload as the default barracks, or a restart would rebuild a
            // different game from the one the players are in.
            string stored = BarracksWire.Write(catalog);

            try
            {
                await store.SaveCatalogAsync(match.MatchId, steamId, stored, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                logger.LogWarning(failure, "Could not store the barracks {Player} chose",
                    SteamLogRedaction.HashSteamId(steamId));
                match.Stale = true;
                sink.Send(connectionId, RejectTemporaryFailure);
                return;
            }

            match.Catalogs[seat] = catalog;

            if (!match.Catalogs.TryGetValue(0, out IReadOnlyList<UnitTemplate>? seat0Barracks)
                || !match.Catalogs.TryGetValue(1, out IReadOnlyList<UnitTemplate>? seat1Barracks))
                return;

            await StartTheMatchAsync(match, connectionId, seat0Barracks, seat1Barracks, ct).ConfigureAwait(false);
        }

        async Task StartTheMatchAsync(LiveMatch match, string connectionId,
            IReadOnlyList<UnitTemplate> seat0Barracks, IReadOnlyList<UnitTemplate> seat1Barracks,
            CancellationToken ct)
        {
            GameState start = GameFactory.Build(match.Setup, seat0Barracks, seat1Barracks);
            string startReplay = ReplayFile.Write(start, Array.Empty<Command>());
            DateTimeOffset now = time.GetUtcNow();

            bool started;
            try
            {
                started = await store
                    .TryStartMatchAsync(match.MatchId, startReplay, now, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                logger.LogWarning(failure, "The start state could not be written, so it was not dealt");
                match.Stale = true;
                sink.Send(connectionId, RejectTemporaryFailure);
                return;
            }

            if (started)
            {
                match.Start = start;
                match.State = start;
                match.Status = MatchStatus.Active;
                match.Log.Clear();
                match.LastSequence = 0;
                DealTheStart(match);
                logger.LogInformation("Both barracks are in; the match is active");
                return;
            }

            // Somebody else started this match first - another host, or this one before a restart. Their
            // start replay is the game the clients will be playing, so take it rather than deal a second,
            // differently seeded one that nobody else would agree with.
            logger.LogInformation("Another host started this match first; adopting the start state it wrote");
            match.Stale = true;

            ReloadOutcome adopted = await TryReloadAsync(match, ct).ConfigureAwait(false);
            if (!adopted.Ok)
            {
                sink.Send(connectionId, RejectTemporaryFailure);
                return;
            }

            if (!adopted.Redealt && match.Status == MatchStatus.Active) DealTheStart(match);
        }

        async Task ReceiveCommandAsync(LiveMatch match, string connectionId, string steamId, int seat,
            string payload, CancellationToken ct)
        {
            if (match.Status != MatchStatus.Active)
            {
                sink.Send(connectionId, RejectForStatus(match.Status));
                return;
            }

            if (!CommandWire.TryRead(payload, out Command? command) || command is null)
            {
                sink.Send(connectionId, NetProtocol.Malformed);
                return;
            }

            // The seat comes from the credential, the issuer from the frame. A client that disagrees with
            // the server about which one it is does not get to move the other army.
            if (seat != (int)command.Issuer)
            {
                logger.LogWarning("{Player} holds seat {Seat} and tried to issue for seat {Claimed}",
                    SteamLogRedaction.HashSteamId(steamId), seat, (int)command.Issuer);
                sink.Send(connectionId, RejectWrongSeat);
                return;
            }

            Result applied = GameEngine.Apply(match.State!, command);
            if (!applied.Success)
            {
                sink.Send(connectionId, NetProtocol.Reject(applied.Reason));
                return;
            }

            DateTimeOffset now = time.GetUtcNow();
            string wire = CommandWire.Write(command);
            int sequence = match.LastSequence + 1;

            // The durable step, before anything in memory moves and before anybody is told.
            AppendResult append;
            try
            {
                append = await store.AppendCommandAsync(
                    match.MatchId, sequence, wire, steamId, now, ct)
                    .ConfigureAwait(false);
            }
            catch (Exception failure)
            {
                // An exception is not a refusal, and a cancellation is not one either. A statement that
                // timed out, a connection dropped after the server had already committed, a token cancelled
                // mid-write: all of them leave this process unable to tell a command that landed from one
                // that did not, and TemporaryFailure invites the client to send it again, which is how one
                // move becomes two. The journal is the only thing that knows, so it is asked.
                logger.LogWarning(failure, "The command from {Player} may not have been journalled",
                    SteamLogRedaction.HashSteamId(steamId));

                JournalCheck check =
                    await CheckJournalAsync(match, sequence, wire, steamId, ct).ConfigureAwait(false);

                if (check == JournalCheck.Absent)
                {
                    match.Stale = true;
                    sink.Send(connectionId, RejectTemporaryFailure);
                    return;
                }

                if (check == JournalCheck.Unknown)
                {
                    // Nothing here can be said honestly. TemporaryFailure would be a claim that the write
                    // did not happen, which is exactly what could not be established; the socket is closed
                    // instead, and the protocol rule after a disconnect - reconnect, take START, do not
                    // resend - resolves the ambiguity without anybody having to guess.
                    logger.LogError(
                        "Sequence {Sequence} cannot be confirmed either way; the issuer must resync",
                        sequence);
                    match.Stale = true;
                    sink.Close(connectionId, ResyncCloseStatus, ResyncCloseReason);
                    return;
                }

                logger.LogInformation(
                    "The append that failed had committed after all; sequence {Sequence} counts as applied",
                    sequence);
                append = new AppendResult(AppendStatus.Appended, sequence);
            }

            switch (append.Status)
            {
                case AppendStatus.Appended:
                case AppendStatus.AlreadyApplied:
                    break;

                case AppendStatus.Conflict:
                    // The journal has moved on without this process. Nothing here can be trusted until it
                    // has been rebuilt, and the player is told to try again rather than told a lie.
                    logger.LogWarning(
                        "The journal expects sequence {Expected}, this projection offered {Offered}",
                        append.Sequence, match.LastSequence + 1);
                    match.Stale = true;
                    await TryReloadAsync(match, ct).ConfigureAwait(false);
                    sink.Send(connectionId, RejectTemporaryFailure);
                    return;

                default:
                    logger.LogInformation("A command arrived for a match the journal no longer calls active");
                    match.Stale = true;
                    await TryReloadAsync(match, ct).ConfigureAwait(false);
                    sink.Send(connectionId, RejectForStatus(match.Status));
                    return;
            }

            MatchStatus terminalStatus = MatchStatus.Completed;
            bool endTheSockets = false;

            if (applied.NewState.IsGameOver)
            {
                CompletionOutcome outcome = await CompleteTheMatchAsync(
                        match, connectionId, sequence, wire, steamId, applied.NewState, now, ct)
                    .ConfigureAwait(false);

                if (!outcome.Advance) return;

                terminalStatus = outcome.Status;

                // Completed is the game ending. Anything else terminal means somebody took this match away
                // underneath a live game - the reaper, or an operator - and the sockets have to be told.
                endTheSockets = outcome.Status != MatchStatus.Completed;
            }

            match.State = applied.NewState;
            match.Log.Add(command);
            match.LastSequence = sequence;
            if (applied.NewState.IsGameOver) match.Status = terminalStatus;

            string broadcast = NetProtocol.Apply(command);
            foreach (string connection in match.Connections.Keys) sink.Send(connection, broadcast);

            // After the APPLY, never before it. The command is durable, so both seats are entitled to see
            // it; what they are not entitled to is a socket into a match that no longer exists.
            if (endTheSockets) CloseEveryConnection(match, MatchEndedCloseStatus, MatchEndedCloseReason);
        }

        /// <summary>
        /// Records the end of the game, before the winning command is broadcast.
        ///
        /// Three answers, and the difference between them is the whole point. Recorded: advance and
        /// broadcast. Refused (the row was already terminal): the append that preceded this may still have
        /// landed, so the journal is asked, and a durable command is broadcast whatever the row now says
        /// while one that never landed is not. Threw: advance nothing - the command is already durable, so
        /// the same frame sent again finds it AlreadyApplied and tries the completion once more, and if
        /// nobody sends anything again the next handshake heals it. Advancing on a completion that failed
        /// would leave a finished game the database still calls active, which the reaper would eventually
        /// mark abandoned.
        /// </summary>
        async Task<CompletionOutcome> CompleteTheMatchAsync(LiveMatch match, string connectionId,
            int sequence, string wire, string issuerSteamId, GameState final, DateTimeOffset now,
            CancellationToken ct)
        {
            int? winnerSeat = final.Winner is PlayerId winner ? (int)winner : null;

            // Twice, and only twice. A completion is one idempotent UPDATE, so the first failure is usually
            // a connection that has just been replaced rather than a database that is gone; a second one is
            // a database this host cannot finish the game against, and no number of further attempts inside
            // one command is going to change that.
            bool completed;
            for (var attempt = 1; ; attempt++)
            {
                try
                {
                    completed = await store
                        .TryCompleteMatchAsync(match.MatchId, MatchStatus.Completed, winnerSeat, now, ct)
                        .ConfigureAwait(false);
                    break;
                }
                catch (Exception failure) when (attempt == 1)
                {
                    logger.LogWarning(failure,
                        "The winning command is journalled and the match would not close; trying once more");
                }
                catch (Exception again)
                {
                    // The command is durable, so the issuer must NOT be told to retry it: TemporaryFailure
                    // would invite exactly the duplicate the append-first order exists to prevent. It is
                    // disconnected instead, and its reconnect is dealt the terminal state through the
                    // window.
                    logger.LogError(again,
                        "The winning command is journalled and the match could not be closed at all");
                    match.Stale = true;
                    sink.Close(connectionId, ResyncCloseStatus, ResyncCloseReason);
                    return new CompletionOutcome(false, match.Status);
                }
            }

            if (completed)
            {
                logger.LogInformation("Match finished, winner {WinnerSeat}",
                    winnerSeat is null ? "draw" : winnerSeat.ToString());
                return new CompletionOutcome(true, MatchStatus.Completed);
            }

            logger.LogWarning(
                "The winning command is journalled but the match was already in a terminal status");

            JournalCheck durable =
                await CheckJournalAsync(match, sequence, wire, issuerSteamId, ct).ConfigureAwait(false);

            if (durable == JournalCheck.Absent)
            {
                match.Stale = true;
                sink.Send(connectionId, RejectTemporaryFailure);
                return new CompletionOutcome(false, match.Status);
            }

            if (durable == JournalCheck.Unknown)
            {
                match.Stale = true;
                sink.Close(connectionId, ResyncCloseStatus, ResyncCloseReason);
                return new CompletionOutcome(false, match.Status);
            }

            PersistedMatch? row;
            try
            {
                row = await store.GetMatchAsync(match.MatchId, ct).ConfigureAwait(false);
            }
            catch (Exception failure)
            {
                // Every failure, cancellation included. Past a durable append there is no such thing as an
                // exception this projection can survive unchanged: the journal has moved and this copy of
                // it has not, so letting a cancellation propagate would leave a projection that is
                // pre-final and NOT marked stale, which the next command would then build on.
                //
                // Unknown is also not Completed. The row is terminal in some way this host cannot read, so
                // advancing to Completed would be inventing an ending, and broadcasting an APPLY under it
                // would hand the clients a result that may not be the recorded one. The sockets are sent
                // back through the reconnect path, which reads the row.
                logger.LogError(failure, "The status of a finished match could not be re-read");
                match.Stale = true;
                CloseEveryConnection(match, ResyncCloseStatus, ResyncCloseReason);
                return new CompletionOutcome(false, match.Status);
            }

            if (row is null)
            {
                logger.LogError("A match that was being completed is no longer in the store at all");
                match.Stale = true;
                CloseEveryConnection(match, ResyncCloseStatus, ResyncCloseReason);
                return new CompletionOutcome(false, match.Status);
            }

            // Read rather than assumed: a match that ends at the same moment the reaper abandons it is
            // abandoned, and the sockets watching it have to be told that rather than told they won.
            return new CompletionOutcome(true, row.Status);
        }

        /// <summary>
        /// Finishes a completion that was started and never landed.
        ///
        /// A process killed between the append of a winning command and the row that records the win leaves
        /// a game the engine calls over and the database calls active. Nothing in the journal is wrong -
        /// replaying it produces exactly this position - so this is not a repair: it is the same completion,
        /// attempted again by whoever arrives next. It is safe to run on every load because
        /// TryCompleteMatchAsync is idempotent, and it runs before anything is served so a reconnecting
        /// player is dealt a match whose status matches the position in front of them.
        /// </summary>
        /// <param name="alreadyDealt">True when the rebuild that preceded this has already handed every
        /// seat the current state. The replay is the same bytes either way, so dealing it twice in one
        /// operation would only make the client parse a game it has just parsed.</param>
        async Task HealCompletionAsync(LiveMatch match, bool alreadyDealt, CancellationToken ct)
        {
            if (match.Status != MatchStatus.Active) return;
            if (match.State is null || !match.State.IsGameOver) return;

            int? winnerSeat = match.State.Winner is PlayerId winner ? (int)winner : null;

            try
            {
                if (await store
                        .TryCompleteMatchAsync(
                            match.MatchId, MatchStatus.Completed, winnerSeat, time.GetUtcNow(), ct)
                        .ConfigureAwait(false))
                {
                    match.Status = MatchStatus.Completed;
                    logger.LogInformation(
                        "Closed a finished match the journal still called active, winner {WinnerSeat}",
                        winnerSeat is null ? "draw" : winnerSeat.ToString());

                    // Every seat watching this match is one that never heard how it ended - the APPLY that
                    // would have told them is the write that did not land. The replay ends in the terminal
                    // position, so re-dealing START is the whole answer.
                    if (!alreadyDealt) DealTheStart(match);
                    return;
                }

                // False means somebody moved the row first, and what they moved it to decides what this
                // host serves. It is read rather than assumed.
                PersistedMatch? row = await store.GetMatchAsync(match.MatchId, ct).ConfigureAwait(false);
                if (row is null) return;

                match.Status = row.Status;

                // The same re-deal for a match somebody else ended. The seats are watching a game that is
                // over either way, and they have to be shown the position it ended in.
                if (!alreadyDealt && match.Status is not (MatchStatus.Waiting or MatchStatus.Active))
                    DealTheStart(match);
            }
            catch (OperationCanceledException)
            {
                // Stale BEFORE it propagates. A cancelled heal leaves a projection that says active over a
                // game the engine calls over, and a caller that saw only the cancellation would have no
                // reason to rebuild it.
                match.Stale = true;
                throw;
            }
            catch (Exception failure)
            {
                // Left stale on purpose: the next operation on this match rebuilds it and tries again,
                // which is the entire self-healing loop.
                logger.LogError(failure, "A finished match could not be closed; it will be tried again");
                match.Stale = true;
            }
        }

        /// <summary>
        /// Whether the command this process just tried to append is in the journal after all.
        ///
        /// Matched on all three of sequence, wire and issuer. Sequence alone would accept a different
        /// command that happens to sit at the same number, which is the one case where answering yes turns
        /// a failed write into a lie both clients then act on.
        ///
        /// Three answers rather than two, because a journal that cannot be read is not a journal that says
        /// no. Collapsing Unknown into Absent is what would produce a TemporaryFailure over a command that
        /// had in fact committed - and an obedient client would then send it again.
        /// </summary>
        async Task<JournalCheck> CheckJournalAsync(
            LiveMatch match, int sequence, string wire, string issuerSteamId, CancellationToken ct)
        {
            try
            {
                MatchJournal? reloaded =
                    await store.LoadJournalAsync(match.MatchId, ct).ConfigureAwait(false);

                if (reloaded is null) return JournalCheck.Absent;

                foreach (PersistedCommand stored in reloaded.Commands)
                {
                    if (stored.Sequence != sequence) continue;

                    return string.Equals(stored.CommandWire, wire, StringComparison.Ordinal)
                        && string.Equals(stored.IssuerSteamId, issuerSteamId, StringComparison.Ordinal)
                        ? JournalCheck.Present
                        : JournalCheck.Absent;
                }

                return JournalCheck.Absent;
            }
            catch (Exception failure)
            {
                logger.LogWarning(failure, "The journal could not be re-read after an ambiguous append");
                return JournalCheck.Unknown;
            }
        }

        // ---- lifecycle -------------------------------------------------------

        /// <summary>Forgets a socket. The seat itself belongs to the Steam account, not to the socket, so
        /// nothing durable changes and the same player can come straight back.</summary>
        public async Task DisconnectAsync(string connectionId)
        {
            if (!_connections.TryRemove(connectionId, out Guid matchId)) return;
            if (!TryGetLiveMatch(matchId, out LiveMatch? match)) return;

            await match!.Gate.WaitAsync().ConfigureAwait(false);
            try
            {
                match.Connections.TryRemove(connectionId, out _);
                match.LastConnectionAt = time.GetUtcNow();
            }
            finally
            {
                match.Gate.Release();
            }
        }

        /// <summary>Releases matches nobody has been watching for a while. Memory only: the journal is
        /// untouched and the next player to arrive loads it again.</summary>
        public Task SweepAsync(DateTimeOffset now)
        {
            foreach (KeyValuePair<Guid, Lazy<Task<LiveMatch>>> entry in _matches)
            {
                if (!entry.Value.IsValueCreated || !entry.Value.Value.IsCompletedSuccessfully) continue;

                LiveMatch match = entry.Value.Value.Result;
                if (!match.Connections.IsEmpty) continue;
                if (now - match.LastConnectionAt < IdleEvictionWindow) continue;

                // A match somebody is mid-commit on is not idle, whatever the clock says. Skipping it costs
                // one sweep interval; taking it away under a commit would cost the commit.
                if (!match.Gate.Wait(0)) continue;
                try
                {
                    if (!match.Connections.IsEmpty) continue;
                    if (now - match.LastConnectionAt < IdleEvictionWindow) continue;

                    _matches.TryRemove(entry);
                    using IDisposable? scope = MatchScope(entry.Key);
                    logger.LogInformation("Released an idle match from memory");
                }
                finally
                {
                    match.Gate.Release();
                }
            }

            // Cached refusals expire on read, which is enough for a match somebody keeps asking for and
            // nothing at all for one they gave up on. Dropping them here keeps a host that has outlived a
            // few broken matches from carrying them for the rest of its life.
            foreach (KeyValuePair<Guid, DateTimeOffset> refused in _unrecoverable)
                if (now - refused.Value >= UnrecoverableRetryWindow)
                    _unrecoverable.TryRemove(refused);

            return Task.CompletedTask;
        }

        /// <summary>
        /// Waits for every in-flight commit to finish, up to a deadline.
        ///
        /// This is what makes a graceful shutdown honest. A process that stopped while a command was between
        /// its append and its broadcast would leave a player who never heard about a move that is in the
        /// journal - recoverable on reconnect, but only because the reconnect re-deals the whole log.
        /// </summary>
        public async Task DrainAsync(TimeSpan timeout)
        {
            using var deadline = new CancellationTokenSource(timeout);

            foreach (LiveMatch match in Cached().ToArray())
            {
                try
                {
                    await match.Gate.WaitAsync(deadline.Token).ConfigureAwait(false);
                    match.Gate.Release();
                }
                catch (OperationCanceledException)
                {
                    logger.LogWarning("Gave up waiting for in-flight commits after {Timeout}", timeout);
                    return;
                }
            }
        }

        /// <summary>Sends one frame to every connection of every match held in memory.</summary>
        public async Task BroadcastAllAsync(string message)
        {
            foreach (LiveMatch match in Cached().ToArray())
            {
                await match.Gate.WaitAsync().ConfigureAwait(false);
                try
                {
                    foreach (string connection in match.Connections.Keys) sink.Send(connection, message);
                }
                finally
                {
                    match.Gate.Release();
                }
            }
        }

        /// <summary>Closes every connection of the named matches and drops them from memory.</summary>
        public async Task EvictAsync(IEnumerable<Guid> matchIds, int closeStatus, string reason)
        {
            ArgumentNullException.ThrowIfNull(matchIds);

            foreach (Guid matchId in matchIds)
            {
                if (!_matches.TryRemove(matchId, out Lazy<Task<LiveMatch>>? entry)) continue;
                if (!entry.IsValueCreated || !entry.Value.IsCompletedSuccessfully) continue;

                LiveMatch match = entry.Value.Result;
                await match.Gate.WaitAsync().ConfigureAwait(false);
                try
                {
                    CloseEveryConnection(match, closeStatus, reason);
                }
                finally
                {
                    match.Gate.Release();
                }
            }
        }

        // ---- internals -------------------------------------------------------

        void DealTheStart(LiveMatch match)
        {
            string message = NetProtocol.Start(match.StartReplayText());
            foreach (string connection in match.Connections.Keys) sink.Send(connection, message);
        }

        /// <summary>Closes every socket watching one match and forgets them. The caller holds the gate.</summary>
        void CloseEveryConnection(LiveMatch match, int closeStatus, string reason)
        {
            foreach (string connection in match.Connections.Keys)
            {
                _connections.TryRemove(connection, out _);
                sink.Close(connection, closeStatus, reason);
            }

            match.Connections.Clear();
        }

        /// <summary>Closes any socket this seat already holds, so one credential is one live connection.
        /// The caller holds the gate.</summary>
        void SupersedeSeat(LiveMatch match, string connectionId, string steamId)
        {
            foreach (KeyValuePair<string, string> seated in match.Connections)
            {
                if (string.Equals(seated.Key, connectionId, StringComparison.Ordinal)) continue;
                if (!string.Equals(seated.Value, steamId, StringComparison.Ordinal)) continue;

                match.Connections.TryRemove(seated.Key, out _);
                _connections.TryRemove(seated.Key, out _);
                sink.Close(seated.Key, SupersededCloseStatus, SupersededCloseReason);

                logger.LogInformation("Closed the earlier socket of a seat that authenticated again");
            }
        }

        /// <summary>
        /// Whether a seat may be taken in this match right now.
        ///
        /// Waiting and active are the game. A match that started and has since ended - completed, expired
        /// or abandoned - is served too, for as long as the credential service is willing to validate into
        /// it: the final APPLY is the frame most likely to be lost, and a player whose socket dropped a
        /// moment before it has no other way to learn how the game they were playing ended. A match that
        /// never started has no game in it and is never served once it is over.
        /// </summary>
        static bool CanSeat(LiveMatch match) =>
            match.Status is MatchStatus.Waiting or MatchStatus.Active || match.Start is not null;

        /// <summary>What a command arriving at a match that is not being played is told. A match that has
        /// not started will take this command once it does; a terminal one never will, and a client that
        /// cannot tell those apart retries a move into a game that is over.</summary>
        static string RejectForStatus(MatchStatus status) =>
            status == MatchStatus.Waiting ? RejectCatalogV1Required : RejectMatchEnded;

        async Task<LiveMatch> GetOrLoadAsync(Guid matchId, CancellationToken ct)
        {
            // The load runs without the caller cancellation token on purpose: the projection is shared, so
            // one client giving up must not cancel the load every other client is waiting on.
            Lazy<Task<LiveMatch>> entry = _matches.GetOrAdd(matchId,
                id => new Lazy<Task<LiveMatch>>(() => LoadAndStampAsync(id)));

            try
            {
                return await entry.Value.WaitAsync(ct).ConfigureAwait(false);
            }
            catch
            {
                // A failed load must not be cached, or one bad moment would refuse this match for the life
                // of the process.
                if (entry.Value.IsFaulted || entry.Value.IsCanceled)
                    _matches.TryRemove(new KeyValuePair<Guid, Lazy<Task<LiveMatch>>>(matchId, entry));

                throw;
            }
        }

        /// <summary>Loads a projection and stamps it as freshly arrived. Without the stamp a match that has
        /// been loaded and not yet connected to looks, to the sweeper, like one nobody has watched since the
        /// epoch - and it can be released between the load and the handshake that asked for it.</summary>
        async Task<LiveMatch> LoadAndStampAsync(Guid matchId)
        {
            LiveMatch match =
                await loader.LoadAsync(matchId, CancellationToken.None).ConfigureAwait(false);

            match.LastConnectionAt = time.GetUtcNow();
            return match;
        }

        /// <summary>
        /// Takes the gate of the match this process is holding, reloading when the sweeper released it
        /// while this caller was waiting. The returned projection is the cached one and its gate is held.
        ///
        /// The cache and the gate are two facts that can disagree. The sweeper takes a gate, drops the
        /// match from memory and releases; a caller that was already waiting on that gate then owns a
        /// projection nobody else can find, and a player seated into it holds a seat in a game this process
        /// no longer believes it is hosting. Re-checking after the wait costs a dictionary lookup.
        /// </summary>
        async Task<LiveMatch> EnterAsync(Guid matchId, CancellationToken ct)
        {
            while (true)
            {
                LiveMatch match = await GetOrLoadAsync(matchId, ct).ConfigureAwait(false);

                if (BeforeGateForTest is not null)
                    await BeforeGateForTest(matchId).ConfigureAwait(false);

                await match.Gate.WaitAsync(ct).ConfigureAwait(false);

                if (TryGetLiveMatch(matchId, out LiveMatch? cached) && ReferenceEquals(cached, match))
                    return match;

                match.Gate.Release();
            }
        }

        /// <summary>Whether a rebuild worked, and whether it re-dealt START in the course of it.</summary>
        readonly record struct ReloadOutcome(bool Ok, bool Redealt);

        /// <summary>
        /// Rebuilds a stale projection, and tells the seats when the rebuild changed the game under them.
        ///
        /// The re-deal is the part that is easy to leave out and impossible to do without. A projection
        /// goes stale precisely when this host could not finish telling people what happened - so by the
        /// time it is rebuilt, the connections still attached to it are looking at a position that is
        /// behind the record, and nothing else will ever tell them. A terminal status or a sequence that
        /// has moved are the two ways to know that from here, and both are answered the same way: START
        /// with the whole log, to everyone, because the seat that reconnected is not the only one that
        /// missed something.
        /// </summary>
        async Task<ReloadOutcome> TryReloadAsync(LiveMatch match, CancellationToken ct)
        {
            MatchStatus before = match.Status;
            int seenBefore = match.LastSequence;

            try
            {
                LiveMatch rebuilt = await loader.LoadAsync(match.MatchId, ct).ConfigureAwait(false);
                match.ReplaceProjectionWith(rebuilt);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                logger.LogError(failure,
                    "This projection is stale and could not be rebuilt from the journal");
                return new ReloadOutcome(false, false);
            }

            bool nowTerminal = match.Status is not (MatchStatus.Waiting or MatchStatus.Active);
            bool moved = match.LastSequence > seenBefore
                || (nowTerminal && before is MatchStatus.Waiting or MatchStatus.Active);

            if (!moved || match.Start is null) return new ReloadOutcome(true, false);

            logger.LogInformation(
                "The rebuild moved this match on; re-dealing the state to every connected seat");

            DealTheStart(match);
            return new ReloadOutcome(true, true);
        }

        /// <summary>Whether this match was refused recently enough that reading its journal again would
        /// only produce the same refusal. The entry is dropped once it expires, so the next handshake pays
        /// for one honest retry and a repaired match comes back without a restart.</summary>
        bool IsStillRefused(Guid matchId)
        {
            if (!_unrecoverable.TryGetValue(matchId, out DateTimeOffset refusedAt)) return false;
            if (time.GetUtcNow() - refusedAt < UnrecoverableRetryWindow) return true;

            _unrecoverable.TryRemove(new KeyValuePair<Guid, DateTimeOffset>(matchId, refusedAt));
            return false;
        }

        IEnumerable<LiveMatch> Cached()
        {
            foreach (KeyValuePair<Guid, Lazy<Task<LiveMatch>>> entry in _matches)
                if (entry.Value.IsValueCreated && entry.Value.Value.IsCompletedSuccessfully)
                    yield return entry.Value.Value.Result;
        }

        IDisposable? MatchScope(Guid matchId) =>
            logger.BeginScope(new Dictionary<string, object> { ["MatchId"] = Short(matchId) });

        static AuthOutcome Failed(string code) => new(false, -1, code);

        /// <summary>Match ids reach logs as their first eight hex characters, the same shortening the
        /// credential service uses, so one match can be followed across both.</summary>
        static string Short(Guid matchId) => matchId.ToString("N")[..8];
    }
}

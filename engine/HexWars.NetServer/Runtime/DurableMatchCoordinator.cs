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

        /// <summary>A command arrived for a match that is not being played: still waiting, or finished.</summary>
        public const string RejectCatalogV1Required = "REJECT CatalogV1Required";

        /// <summary>The durable write did not happen. The client may send the same command again.</summary>
        public const string RejectTemporaryFailure = "REJECT TemporaryFailure";

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

        /// <summary>Whether the handshake worked, the seat it seated, and - when it did not - why not.</summary>
        public sealed record AuthOutcome(bool Ok, int Seat, string? FailCode);

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
                match = await GetOrLoadAsync(matchId, ct).ConfigureAwait(false);
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

            await match.Gate.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                if (match.Stale && !await TryReloadAsync(match, ct).ConfigureAwait(false))
                    return Failed(AuthFailUnavailable);

                if (match.Status is not (MatchStatus.Waiting or MatchStatus.Active))
                {
                    logger.LogDebug("Turned a player away: this match is {Status}", match.Status);
                    return Failed(AuthFailInvalid);
                }

                int seat = validation.Seat;
                DateTimeOffset now = time.GetUtcNow();

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
                    if (!match.Catalogs.ContainsKey(seat))
                        sink.Send(connectionId, NetProtocol.CatalogRequest);
                }
                else
                {
                    sink.Send(connectionId, NetProtocol.Start(match.StartReplayText()));
                }

                logger.LogInformation("Seated {Player} in seat {Seat} of a {Status} match",
                    SteamLogRedaction.HashSteamId(validation.SteamId), seat, match.Status);

                return new AuthOutcome(true, seat, null);
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
                || !TryGetLiveMatch(matchId, out LiveMatch? cached))
            {
                sink.Send(connectionId, RejectNoSeat);
                return;
            }

            NetMessage message = NetProtocol.Parse(raw);

            // PONG and anything this build does not know about are liveness, nothing more. Answering an
            // unknown type would make adding one to the client a breaking change.
            if (message.Type is not (CatalogMessage or CommandMessage)) return;

            LiveMatch match = cached!;
            using IDisposable? scope = MatchScope(matchId);

            await match.Gate.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                if (match.Stale && !await TryReloadAsync(match, ct).ConfigureAwait(false))
                {
                    sink.Send(connectionId, RejectTemporaryFailure);
                    return;
                }

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

            if (!await TryReloadAsync(match, ct).ConfigureAwait(false))
            {
                sink.Send(connectionId, RejectTemporaryFailure);
                return;
            }

            if (match.Status == MatchStatus.Active) DealTheStart(match);
        }

        async Task ReceiveCommandAsync(LiveMatch match, string connectionId, string steamId, int seat,
            string payload, CancellationToken ct)
        {
            if (match.Status != MatchStatus.Active)
            {
                sink.Send(connectionId, RejectCatalogV1Required);
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

            // The durable step, before anything in memory moves and before anybody is told.
            AppendResult append;
            try
            {
                append = await store.AppendCommandAsync(
                    match.MatchId, match.LastSequence + 1, CommandWire.Write(command), steamId, now, ct)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                logger.LogWarning(failure, "The command from {Player} could not be journalled",
                    SteamLogRedaction.HashSteamId(steamId));
                match.Stale = true;
                sink.Send(connectionId, RejectTemporaryFailure);
                return;
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
                    sink.Send(connectionId, RejectCatalogV1Required);
                    return;
            }

            if (applied.NewState.IsGameOver
                && !await CompleteTheMatchAsync(match, connectionId, applied.NewState, now, ct)
                    .ConfigureAwait(false))
                return;

            match.State = applied.NewState;
            match.Log.Add(command);
            match.LastSequence++;
            if (applied.NewState.IsGameOver) match.Status = MatchStatus.Completed;

            string broadcast = NetProtocol.Apply(command);
            foreach (string connection in match.Connections.Keys) sink.Send(connection, broadcast);
        }

        /// <summary>
        /// Records the end of the game, before the winning command is broadcast.
        ///
        /// Returns false when it could not be recorded, and the caller then advances nothing: the command is
        /// already durable, so the same frame sent again finds it AlreadyApplied and tries the completion
        /// once more. Advancing here and failing to record it would leave a finished game the database still
        /// calls active, which the reaper would eventually mark abandoned.
        /// </summary>
        async Task<bool> CompleteTheMatchAsync(LiveMatch match, string connectionId, GameState final,
            DateTimeOffset now, CancellationToken ct)
        {
            int? winnerSeat = final.Winner is PlayerId winner ? (int)winner : null;

            try
            {
                bool completed = await store
                    .TryCompleteMatchAsync(match.MatchId, MatchStatus.Completed, winnerSeat, now, ct)
                    .ConfigureAwait(false);

                if (!completed)
                    logger.LogWarning(
                        "The winning command is journalled but the match was already in a terminal status");
                else
                    logger.LogInformation("Match finished, winner {WinnerSeat}",
                        winnerSeat is null ? "draw" : winnerSeat.ToString());

                return true;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                logger.LogError(failure,
                    "The winning command is journalled but the match could not be closed");
                match.Stale = true;
                sink.Send(connectionId, RejectTemporaryFailure);
                return false;
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
                    foreach (string connection in match.Connections.Keys)
                    {
                        _connections.TryRemove(connection, out _);
                        sink.Close(connection, closeStatus, reason);
                    }

                    match.Connections.Clear();
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

        async Task<LiveMatch> GetOrLoadAsync(Guid matchId, CancellationToken ct)
        {
            // The load runs without the caller cancellation token on purpose: the projection is shared, so
            // one client giving up must not cancel the load every other client is waiting on.
            Lazy<Task<LiveMatch>> entry = _matches.GetOrAdd(matchId,
                id => new Lazy<Task<LiveMatch>>(() => loader.LoadAsync(id, CancellationToken.None)));

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

        async Task<bool> TryReloadAsync(LiveMatch match, CancellationToken ct)
        {
            try
            {
                LiveMatch rebuilt = await loader.LoadAsync(match.MatchId, ct).ConfigureAwait(false);
                match.ReplaceProjectionWith(rebuilt);
                return true;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception failure)
            {
                logger.LogError(failure,
                    "This projection is stale and could not be rebuilt from the journal");
                return false;
            }
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

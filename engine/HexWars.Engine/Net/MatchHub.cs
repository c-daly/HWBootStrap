using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>A message the transport should deliver to one connection.</summary>
    public readonly struct Outbound
    {
        public readonly string ConnectionId;
        public readonly string Message;
        public Outbound(string connectionId, string message) { ConnectionId = connectionId; Message = message; }
    }

    /// <summary>One joinable lobby entry: a public room with a host waiting and a game not yet begun.</summary>
    public readonly struct OpenGame
    {
        public readonly string Code;
        public readonly GameSetup Setup;
        public readonly int AgeSeconds;
        public OpenGame(string code, GameSetup setup, int ageSeconds) { Code = code; Setup = setup; AgeSeconds = ageSeconds; }
    }

    /// <summary>
    /// Routes connections into rooms and drives each room's <see cref="GameSession"/>. Pure logic: every
    /// method returns the <see cref="Outbound"/> messages a transport should send, so the entire server
    /// brain is unit-testable without a socket. A new room mints a fresh game via the injected factory;
    /// once both seats are filled it deals the start state to both; thereafter validated commands are
    /// broadcast to everyone and rejections go back to just the issuer.
    ///
    /// Seats are keyed by a client-minted TOKEN, not the transport connection id (a token survives a
    /// refresh/reconnect; a connection id does not). Each room keeps its own connection→token map so a
    /// dropped-and-reconnected socket with the same token reclaims the same <see cref="GameSession"/>
    /// seat. A room that has Started and drops to zero live connections is HELD (its Session — and so
    /// its token→seat assignments — kept alive) for <see cref="HoldWindowTicks"/> instead of being
    /// deleted, so both players can survive a simultaneous drop; an un-started room (never dealt START)
    /// still cleans up the instant it empties, as before this feature.
    ///
    /// Every re-deal (initial deal AND reconnect) sends <c>Room.Start</c> + <c>Room.Log</c> (every
    /// Accepted command, in order) through <see cref="ReplayFile.Write(GameState, System.Collections.Generic.IReadOnlyList{Command})"/>,
    /// never the live <c>Session.State</c> directly — <see cref="ReplayFile"/>'s start-state encoding
    /// assumes full-health units and omits per-turn tracking (MovedUnitIds/AttackedUnitIds/MovementSpent/
    /// IsGameOver), which is exact for a FRESH state (that's all it's ever had to represent) but would
    /// silently corrupt a mid-game re-deal. Replaying the command log through the same engine the client
    /// already uses (GameEngine.Apply) reconstructs the exact state instead, by construction. The initial
    /// deal's Log is empty, so this is byte-identical to writing Session.State directly, as before.
    /// </summary>
    public sealed class MatchHub
    {
        /// <summary>How long a Started room with zero live connections is kept alive for a reconnect.</summary>
        private static readonly long HoldWindowTicks = TimeSpan.FromMinutes(10).Ticks;

        private sealed class Room
        {
            public GameState? Start;       // assigned once both catalogs arrive; paired with Log, replays exactly
            public readonly List<Command> Log = new List<Command>(); // every Accepted command, in order
            public GameSession? Session;
            public readonly List<string> Members = new List<string>(); // seated connections, broadcast targets
            public readonly Dictionary<string, string> ConnToToken = new Dictionary<string, string>();
            public readonly Dictionary<string, PlayerId> TokenToSeat = new Dictionary<string, PlayerId>();
            public readonly Dictionary<PlayerId, IReadOnlyList<UnitTemplate>> Catalogs =
                new Dictionary<PlayerId, IReadOnlyList<UnitTemplate>>();
            public readonly GameSetup Setup;       // the host's picks — shown in the lobby browser
            public readonly bool IsPrivate;        // private rooms are joinable by code/link only
            public readonly long CreatedAtTicks;
            public bool Started;                   // set when the start state is dealt; never cleared —
                                                   // a started room that drops to one member must not re-list
            public long? EmptySinceTicks;          // set when a Started room's Members hits zero; null while
                                                   // occupied or un-started — the held-room expiry clock
            public Room(GameSetup setup, bool isPrivate, long nowTicks)
            { Setup = setup; IsPrivate = isPrivate; CreatedAtTicks = nowTicks; }
        }

        private readonly Func<GameSetup, IReadOnlyList<UnitTemplate>, IReadOnlyList<UnitTemplate>, GameState> _newGame;
        private readonly Func<long> _now;
        private readonly Dictionary<string, Room> _rooms = new Dictionary<string, Room>();

        /// <summary>The clock is injectable so lobby ages (and hold-window expiry) are exactly testable;
        /// production uses UTC.</summary>
        public MatchHub(Func<GameSetup, GameState> newGame, Func<long>? utcNowTicks = null,
            Func<GameSetup, IReadOnlyList<UnitTemplate>, IReadOnlyList<UnitTemplate>, GameState>? newCatalogGame = null)
        {
            _newGame = newCatalogGame ?? ((setup, _, __) => newGame(setup));
            _now = utcNowTicks ?? (() => DateTime.UtcNow.Ticks);
        }

        /// <summary>Drop any held room whose hold window has elapsed. Cheap (a linear scan of rooms);
        /// called opportunistically at the top of every public method — there are no timers in this
        /// pure, transport-agnostic hub.</summary>
        private void Sweep()
        {
            if (_rooms.Count == 0) return;
            long now = _now();
            List<string>? expired = null;
            foreach (var kv in _rooms)
                if (kv.Value.EmptySinceTicks is long since && now - since > HoldWindowTicks)
                    (expired ??= new List<string>()).Add(kv.Key);
            if (expired != null)
                foreach (var code in expired) _rooms.Remove(code);
        }

        /// <summary>Seat a connection under its identity token. The first connection to a room creates it
        /// from <paramref name="setup"/> (the host's lobby picks); later joiners' setups are ignored —
        /// they join the host's game. <paramref name="joinOnly"/> marks a joiner (link/code/browser row,
        /// never a host): a joiner must never mint a room, since a typo'd code would otherwise create a
        /// phantom public game and strand the joiner with a seat in it. Missing room + joinOnly turns the
        /// connection away instead. <paramref name="token"/> is the client's persistent identity (null
        /// falls back to <paramref name="connectionId"/> — a fresh/garbled token is a fresh identity, same
        /// as today's per-socket behaviour). Any connect into an already-Started room — including this
        /// same reconnect — gets a personal START re-deal of the current session state.</summary>
        public IReadOnlyList<Outbound> Connect(string roomCode, string connectionId, GameSetup setup = default,
            bool isPrivate = false, bool joinOnly = false, string? token = null)
        {
            token ??= connectionId;
            Sweep();

            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room))
            {
                if (joinOnly)
                {
                    outs.Add(new Outbound(connectionId, NetProtocol.SeatFull));
                    return outs;
                }
                room = new Room(setup.Sanitized(), isPrivate, _now());
                _rooms[roomCode] = room;
            }

            var seat = Join(room, token);
            if (seat == null)
            {
                outs.Add(new Outbound(connectionId, NetProtocol.SeatFull));
                return outs;
            }

            room.ConnToToken[connectionId] = token;
            room.EmptySinceTicks = null; // any successful connect cancels a pending hold-expiry
            if (!room.Members.Contains(connectionId)) room.Members.Add(connectionId);
            outs.Add(new Outbound(connectionId, NetProtocol.Seat(seat.Value)));

            if (room.Started)
            {
                // (Re)connect into an already-started room: a personal re-deal — the start state PLUS
                // every command accepted since, so the replay reconstructs the current state exactly
                // (see class doc). Same resync-by-replay mechanism used for the initial deal below, now
                // also serving a reconnect after a drop.
                string startMsg = NetProtocol.Start(ReplayFile.Write(room.Start!, room.Log));
                outs.Add(new Outbound(connectionId, startMsg));
            }
            else if (!room.Catalogs.ContainsKey(seat.Value))
            {
                // The server explicitly requests setup data only while the room is waiting. This makes
                // reconnect unambiguous: a started reconnect receives START, never a timing-based client
                // guess that can race the re-deal and provoke CatalogClosed.
                outs.Add(new Outbound(connectionId, NetProtocol.CatalogRequest));
            }
            return outs;
        }

        private static PlayerId? Join(Room room, string token)
        {
            if (room.TokenToSeat.TryGetValue(token, out var existing)) return existing;
            if (!room.TokenToSeat.ContainsValue(PlayerId.Player0))
                return room.TokenToSeat[token] = PlayerId.Player0;
            if (!room.TokenToSeat.ContainsValue(PlayerId.Player1))
                return room.TokenToSeat[token] = PlayerId.Player1;
            return null;
        }

        /// <summary>The lobby browser's view: public rooms with a waiting host and no game started,
        /// newest first. Callers own thread-safety (the transport already serializes hub access).</summary>
        public IReadOnlyList<OpenGame> OpenGames()
        {
            Sweep();
            var list = new List<OpenGame>();
            foreach (var kv in _rooms)
            {
                var r = kv.Value;
                // "A waiting host" is one claimed SEAT, not one connection — a host with two tabs open
                // is still a lone waiting player and must stay browsable.
                if (r.IsPrivate || r.Started || r.TokenToSeat.Count != 1) continue;
                int age = (int)((_now() - r.CreatedAtTicks) / TimeSpan.TicksPerSecond);
                list.Add(new OpenGame(kv.Key, r.Setup, age));
            }
            list.Sort((a, b) => a.AgeSeconds.CompareTo(b.AgeSeconds)); // newest (smallest age) first
            return list;
        }

        public IReadOnlyList<Outbound> Receive(string roomCode, string connectionId, string raw)
        {
            Sweep();
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room)) return outs;

            var msg = NetProtocol.Parse(raw);
            if (msg.Type == "CATALOG")
            {
                if (!room.ConnToToken.TryGetValue(connectionId, out var catalogToken)
                    || !room.TokenToSeat.TryGetValue(catalogToken, out var catalogSeat))
                {
                    outs.Add(new Outbound(connectionId, "REJECT NoSeat"));
                    return outs;
                }
                if (room.Started)
                {
                    outs.Add(new Outbound(connectionId, "REJECT CatalogClosed"));
                    return outs;
                }

                IReadOnlyList<UnitTemplate> catalog;
                try { catalog = BarracksWire.Read(msg.Payload); }
                catch (FormatException) { catalog = BarracksCatalog.DefaultTemplates; }
                room.Catalogs[catalogSeat] = catalog;
                TryStart(room, outs);
                return outs;
            }

            if (msg.Type != "CMD") return outs;
            if (!room.ConnToToken.TryGetValue(connectionId, out var commandToken)
                || !room.TokenToSeat.ContainsKey(commandToken))
            {
                outs.Add(new Outbound(connectionId, "REJECT NoSeat"));
                return outs;
            }
            if (!room.Started)
            {
                outs.Add(new Outbound(connectionId, "REJECT CatalogV1Required"));
                return outs;
            }

            if (!CommandWire.TryRead(msg.Payload, out var cmd) || cmd == null)
            {
                outs.Add(new Outbound(connectionId, NetProtocol.Malformed));
                return outs;
            }

            var outcome = room.Session!.Submit(commandToken, cmd);
            switch (outcome.Status)
            {
                case SubmitStatus.Accepted:
                    room.Log.Add(cmd);
                    string applyMsg = NetProtocol.Apply(cmd);
                    foreach (var m in room.Members) outs.Add(new Outbound(m, applyMsg));
                    break;
                case SubmitStatus.Rejected:
                    outs.Add(new Outbound(connectionId, NetProtocol.Reject(outcome.Reason)));
                    break;
                default: // NoSeat / WrongSeat — tell only the offender, never touch the game
                    outs.Add(new Outbound(connectionId, "REJECT " + outcome.Status));
                    break;
            }
            return outs;
        }

        private void TryStart(Room room, List<Outbound> outs)
        {
            if (room.Started || room.TokenToSeat.Count != 2
                || !room.Catalogs.TryGetValue(PlayerId.Player0, out var p0Catalog)
                || !room.Catalogs.TryGetValue(PlayerId.Player1, out var p1Catalog))
                return;

            room.Start = _newGame(room.Setup, p0Catalog, p1Catalog);
            room.Session = new GameSession(room.Start, room.TokenToSeat);
            room.Started = true;
            string startMsg = NetProtocol.Start(ReplayFile.Write(room.Start, room.Log));
            foreach (var member in room.Members)
                outs.Add(new Outbound(member, startMsg));
        }

        /// <summary>Drop a connection. Its room's seat is NOT freed here — seats belong to tokens for the
        /// life of a Started room, so the same token can always reclaim it (see class doc). A Started room
        /// that drops to zero live connections starts its hold-window clock instead of being removed; an
        /// un-started room (nobody ever dealt START) is still removed the instant it empties.</summary>
        public IReadOnlyList<Outbound> Disconnect(string roomCode, string connectionId)
        {
            Sweep();
            if (_rooms.TryGetValue(roomCode, out var room))
            {
                room.ConnToToken.TryGetValue(connectionId, out var token);
                room.Members.Remove(connectionId);
                room.ConnToToken.Remove(connectionId);
                if (!room.Started && token != null && !room.ConnToToken.ContainsValue(token)
                    && room.TokenToSeat.TryGetValue(token, out var waitingSeat))
                {
                    // Before START, a vanished identity must not reserve a seat forever. Keep the seat
                    // while any same-token tab remains, but otherwise let the waiting player be replaced.
                    room.TokenToSeat.Remove(token);
                    room.Catalogs.Remove(waitingSeat);
                }
                if (room.Members.Count == 0)
                {
                    if (room.Started) room.EmptySinceTicks = _now();
                    else _rooms.Remove(roomCode);
                }
            }
            return Array.Empty<Outbound>();
        }
    }
}

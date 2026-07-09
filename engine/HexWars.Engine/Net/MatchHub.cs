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
    /// </summary>
    public sealed class MatchHub
    {
        /// <summary>How long a Started room with zero live connections is kept alive for a reconnect.</summary>
        private static readonly long HoldWindowTicks = TimeSpan.FromMinutes(10).Ticks;

        private sealed class Room
        {
            public readonly GameSession Session;
            public readonly List<string> Members = new List<string>(); // seated connections, broadcast targets
            public readonly Dictionary<string, string> ConnToToken = new Dictionary<string, string>();
            public readonly GameSetup Setup;       // the host's picks — shown in the lobby browser
            public readonly bool IsPrivate;        // private rooms are joinable by code/link only
            public readonly long CreatedAtTicks;
            public bool Started;                   // set when the start state is dealt; never cleared —
                                                   // a started room that drops to one member must not re-list
            public long? EmptySinceTicks;          // set when a Started room's Members hits zero; null while
                                                   // occupied or un-started — the held-room expiry clock
            public Room(GameState start, GameSetup setup, bool isPrivate, long nowTicks)
            { Session = new GameSession(start); Setup = setup; IsPrivate = isPrivate; CreatedAtTicks = nowTicks; }
        }

        private readonly Func<GameSetup, GameState> _newGame;
        private readonly Func<long> _now;
        private readonly Dictionary<string, Room> _rooms = new Dictionary<string, Room>();

        /// <summary>The clock is injectable so lobby ages (and hold-window expiry) are exactly testable;
        /// production uses UTC.</summary>
        public MatchHub(Func<GameSetup, GameState> newGame, Func<long>? utcNowTicks = null)
        { _newGame = newGame; _now = utcNowTicks ?? (() => DateTime.UtcNow.Ticks); }

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
                room = new Room(_newGame(setup), setup.Sanitized(), isPrivate, _now());
                _rooms[roomCode] = room;
            }

            var seat = room.Session.Join(token);
            if (seat == null)
            {
                outs.Add(new Outbound(connectionId, NetProtocol.SeatFull));
                return outs;
            }

            room.ConnToToken[connectionId] = token;
            room.EmptySinceTicks = null; // any successful connect cancels a pending hold-expiry
            bool added = false;
            if (!room.Members.Contains(connectionId)) { room.Members.Add(connectionId); added = true; }
            outs.Add(new Outbound(connectionId, NetProtocol.Seat(seat.Value)));

            if (room.Started)
            {
                // (Re)connect into an already-started room: a personal re-deal of the current state — the
                // same resync-by-replay mechanism used for the initial deal below, now also serving a
                // reconnect after a drop.
                string startMsg = NetProtocol.Start(ReplayFile.Write(room.Session.State, Array.Empty<Command>()));
                outs.Add(new Outbound(connectionId, startMsg));
            }
            else if (added && !room.Started && room.Session.SeatedCount == 2)
            {
                // Start when the second SEAT fills, not the second connection: seats are token-keyed,
                // so one player's extra tabs are extra connections but the same single seat — counting
                // Members here would fire a bogus START at a lone host with two tabs open (and Started,
                // never cleared, would silently delist the room from the lobby forever).
                room.Started = true;
                string startMsg = NetProtocol.Start(ReplayFile.Write(room.Session.State, Array.Empty<Command>()));
                foreach (var m in room.Members) outs.Add(new Outbound(m, startMsg));
            }
            return outs;
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
                if (r.IsPrivate || r.Started || r.Session.SeatedCount != 1) continue;
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
            if (msg.Type != "CMD") return outs; // v0: CMD is the only client→server message that does anything

            if (!CommandWire.TryRead(msg.Payload, out var cmd) || cmd == null)
            {
                outs.Add(new Outbound(connectionId, NetProtocol.Malformed));
                return outs;
            }

            string token = room.ConnToToken.TryGetValue(connectionId, out var t) ? t : connectionId;
            var outcome = room.Session.Submit(token, cmd);
            switch (outcome.Status)
            {
                case SubmitStatus.Accepted:
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

        /// <summary>Drop a connection. Its room's seat is NOT freed here — seats belong to tokens for the
        /// life of a Started room, so the same token can always reclaim it (see class doc). A Started room
        /// that drops to zero live connections starts its hold-window clock instead of being removed; an
        /// un-started room (nobody ever dealt START) is still removed the instant it empties.</summary>
        public IReadOnlyList<Outbound> Disconnect(string roomCode, string connectionId)
        {
            Sweep();
            if (_rooms.TryGetValue(roomCode, out var room))
            {
                room.Members.Remove(connectionId);
                room.ConnToToken.Remove(connectionId);
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

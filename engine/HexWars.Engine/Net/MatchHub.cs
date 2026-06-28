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

    /// <summary>
    /// Routes connections into rooms and drives each room's <see cref="GameSession"/>. Pure logic: every
    /// method returns the <see cref="Outbound"/> messages a transport should send, so the entire server
    /// brain is unit-testable without a socket. A new room mints a fresh game via the injected factory;
    /// once both seats are filled it deals the start state to both; thereafter validated commands are
    /// broadcast to everyone and rejections go back to just the issuer.
    /// </summary>
    public sealed class MatchHub
    {
        private sealed class Room
        {
            public readonly GameSession Session;
            public readonly List<string> Members = new List<string>(); // seated connections, broadcast targets
            public Room(GameState start) { Session = new GameSession(start); }
        }

        private readonly Func<GameState> _newGame;
        private readonly Dictionary<string, Room> _rooms = new Dictionary<string, Room>();

        public MatchHub(Func<GameState> newGame) { _newGame = newGame; }

        public IReadOnlyList<Outbound> Connect(string roomCode, string connectionId)
        {
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room))
            {
                room = new Room(_newGame());
                _rooms[roomCode] = room;
            }

            var seat = room.Session.Join(connectionId);
            if (seat == null)
            {
                outs.Add(new Outbound(connectionId, NetProtocol.SeatFull));
                return outs;
            }

            bool added = false;
            if (!room.Members.Contains(connectionId)) { room.Members.Add(connectionId); added = true; }
            outs.Add(new Outbound(connectionId, NetProtocol.Seat(seat.Value)));

            // The moment the second distinct player takes a seat, deal the authoritative start state to both.
            if (added && room.Members.Count == 2)
            {
                string startMsg = NetProtocol.Start(ReplayFile.Write(room.Session.State, Array.Empty<Command>()));
                foreach (var m in room.Members) outs.Add(new Outbound(m, startMsg));
            }
            return outs;
        }

        public IReadOnlyList<Outbound> Receive(string roomCode, string connectionId, string raw)
        {
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room)) return outs;

            var msg = NetProtocol.Parse(raw);
            if (msg.Type != "CMD") return outs; // v0: CMD is the only client→server message that does anything

            var cmd = CommandWire.Read(msg.Payload);
            var outcome = room.Session.Submit(connectionId, cmd);
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
    }
}

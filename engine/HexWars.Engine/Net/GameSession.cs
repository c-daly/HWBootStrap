using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>Why a <see cref="GameSession.Submit"/> did or didn't land.</summary>
    public enum SubmitStatus
    {
        /// <summary>Applied; <see cref="SubmitOutcome.State"/> holds the new state to broadcast.</summary>
        Accepted,
        /// <summary>The connection holds no seat in this session (e.g., a spectator or stranger).</summary>
        NoSeat,
        /// <summary>The connection tried to issue a command as the seat it does not hold (impersonation).</summary>
        WrongSeat,
        /// <summary>The engine rejected it; <see cref="SubmitOutcome.Reason"/> says why.</summary>
        Rejected,
    }

    /// <summary>Result of submitting a command to a <see cref="GameSession"/>.</summary>
    public sealed class SubmitOutcome
    {
        public SubmitStatus Status { get; }
        public RejectionReason Reason { get; } // meaningful when Status == Rejected
        public GameState State { get; }        // the post-command state when Status == Accepted

        private SubmitOutcome(SubmitStatus status, RejectionReason reason, GameState state)
        {
            Status = status;
            Reason = reason;
            State = state;
        }

        internal static SubmitOutcome Accepted(GameState state) => new SubmitOutcome(SubmitStatus.Accepted, RejectionReason.None, state);
        internal static SubmitOutcome NoSeat() => new SubmitOutcome(SubmitStatus.NoSeat, RejectionReason.None, null);
        internal static SubmitOutcome WrongSeat() => new SubmitOutcome(SubmitStatus.WrongSeat, RejectionReason.None, null);
        internal static SubmitOutcome Rejected(RejectionReason reason) => new SubmitOutcome(SubmitStatus.Rejected, reason, null);
    }

    /// <summary>
    /// One authoritative head-to-head match, independent of any transport. It seats two connections
    /// (P0 then P1), and on every command enforces the one rule the engine can't: a connection may only
    /// issue as the seat it actually holds (anti-impersonation). Everything else — turn order, legality,
    /// win/elimination — is delegated to <see cref="GameEngine.Apply"/>, the single source of truth.
    /// A WebSocket layer just turns sockets into <see cref="Join"/>/<see cref="Submit"/> calls and
    /// relays the accepted command (via <see cref="CommandWire"/>) to both seats.
    /// </summary>
    public sealed class GameSession
    {
        private readonly Dictionary<string, PlayerId> _seats = new Dictionary<string, PlayerId>();

        public GameState State { get; private set; }

        public GameSession(GameState start) { State = start; }

        /// <summary>Seat a connection as P0, then P1; a returning connection keeps its seat; null once full.</summary>
        public PlayerId? Join(string connectionId)
        {
            if (_seats.TryGetValue(connectionId, out var existing)) return existing;
            if (!_seats.ContainsValue(PlayerId.Player0)) return _seats[connectionId] = PlayerId.Player0;
            if (!_seats.ContainsValue(PlayerId.Player1)) return _seats[connectionId] = PlayerId.Player1;
            return null;
        }

        /// <summary>Validate the issuer owns its seat, then apply through the engine. On Accepted, advances State.</summary>
        public SubmitOutcome Submit(string connectionId, Command cmd)
        {
            if (!_seats.TryGetValue(connectionId, out var seat)) return SubmitOutcome.NoSeat();
            if (cmd.Issuer != seat) return SubmitOutcome.WrongSeat();

            var result = GameEngine.Apply(State, cmd);
            if (!result.Success) return SubmitOutcome.Rejected(result.Reason);

            State = result.NewState;
            return SubmitOutcome.Accepted(State);
        }
    }
}

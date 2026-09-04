#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>Where a protocol-v2 match socket stands.</summary>
    public enum SteamMatchSessionState
    {
        /// <summary>A socket is being opened; nothing has been sent yet.</summary>
        Connecting,

        /// <summary>The socket is open and AUTH went out; only SEAT can move the flow on.</summary>
        Authenticating,

        /// <summary>The server accepted the credential and dealt a seat.</summary>
        Seated,

        /// <summary>This attempt is over.</summary>
        Closed,
    }

    /// <summary>What the driver must do next.</summary>
    public enum SteamMatchSessionOutputKind
    {
        /// <summary>Send the output text over the socket.</summary>
        Send,

        /// <summary>The server asked for the starting barracks catalog.</summary>
        CatalogRequested,

        /// <summary>A seat was dealt; the output seat is its index.</summary>
        Seat,

        /// <summary>The match is full.</summary>
        SeatFull,

        /// <summary>The authoritative start state, in the output text.</summary>
        Start,

        /// <summary>A validated command to apply, as its wire line.</summary>
        Apply,

        /// <summary>A rejection, with the server payload.</summary>
        Reject,

        /// <summary>The credential was refused; the code is the text. No retry can fix it.</summary>
        AuthFailed,

        /// <summary>A drop is being retried; the output seat is the attempt number.</summary>
        Reconnecting,

        /// <summary>A seat came back after a drop.</summary>
        Reconnected,

        /// <summary>Wait the output delay, refresh the credential, and reopen.</summary>
        Retry,

        /// <summary>The socket died before a match began; stop, the way v1 does.</summary>
        Closed,

        /// <summary>No fresh credential could be had: the match is unrecoverable.</summary>
        GiveUp,
    }

    /// <summary>One queued instruction from <see cref="SteamMatchSession"/>.</summary>
    public sealed class SteamMatchSessionOutput
    {
        public SteamMatchSessionOutput(SteamMatchSessionOutputKind kind, string? text = null,
                                       int seat = 0, double delaySeconds = 0)
        {
            Kind = kind;
            Text = text ?? string.Empty;
            Seat = seat;
            DelaySeconds = delaySeconds;
        }

        public SteamMatchSessionOutputKind Kind { get; }

        /// <summary>The frame to send, the payload received, or the auth failure code.</summary>
        public string Text { get; }

        /// <summary>The seat index for Seat, or the attempt number for Reconnecting.</summary>
        public int Seat { get; }

        /// <summary>How long to wait before the next attempt, for Retry.</summary>
        public double DelaySeconds { get; }

        public override string ToString()
        {
            return Kind + "(" + Text + ", seat=" + Seat + ", delay="
                 + DelaySeconds.ToString(CultureInfo.InvariantCulture) + ")";
        }
    }

    /// <summary>
    /// The protocol-v2 transport decision machine, with no Unity and no socket in it, so every rule
    /// below is covered by a dotnet test instead of by a play session.
    /// <para>
    /// The state is Connecting -> Authenticating -> Seated -> Closed. The driver feeds it
    /// <see cref="Attempting"/>, <see cref="Opened"/>, <see cref="Frame"/>, <see cref="Closed"/> and
    /// <see cref="Tick"/>, then executes whatever <see cref="Drain"/> hands back.
    /// </para>
    /// <para>
    /// The reason it exists: a socket that opened and then never answered AUTH used to hang the game
    /// on "Connecting..." forever, because the only thing that ended an attempt was the socket close
    /// event. <see cref="HandshakeTimeoutSeconds"/> now ends it, and a hung handshake is always worth
    /// another attempt: nothing has reached the player yet, so stopping there is the very hang the
    /// deadline exists to break.
    /// </para>
    /// </summary>
    public sealed class SteamMatchSession
    {
        /// <summary>How long AUTH may go unanswered before the attempt is abandoned and retried.</summary>
        public const double HandshakeTimeoutSeconds = 15;

        /// <summary>Capped exponential backoff between attempts; the last entry repeats.</summary>
        static readonly double[] BackoffSeconds = { 1, 2, 4, 8, 15 };

        static readonly SteamMatchSessionOutput[] NoOutputs = new SteamMatchSessionOutput[0];

        readonly List<SteamMatchSessionOutput> _outputs = new List<SteamMatchSessionOutput>();

        string _matchId = string.Empty;
        string _credential = string.Empty;

        bool _handshakeArmed;
        double? _handshakeAnchor;

        int _attempt;
        bool _authFailed;
        bool _expectRestart;

        public SteamMatchSession()
        {
        }

        public SteamMatchSession(SteamMatchTicket ticket)
        {
            UseTicket(ticket);
        }

        public SteamMatchSessionState State { get; private set; } = SteamMatchSessionState.Connecting;

        /// <summary>0 for the first attempt, then one more for every drop that was retried.</summary>
        public int Attempt { get { return _attempt; } }

        /// <summary>
        /// True once a match actually started. A first drop with no game behind it keeps the v1
        /// toast-and-stop behaviour: there is nothing to reconnect into.
        /// </summary>
        public bool GameStarted { get; set; }

        /// <summary>True when the credential was refused, which no retry can fix.</summary>
        public bool AuthFailed { get { return _authFailed; } }

        /// <summary>Adopts a (re-issued) ticket for the next AUTH frame.</summary>
        public void UseTicket(SteamMatchTicket? ticket)
        {
            _matchId = ticket == null ? string.Empty : ticket.MatchId;
            _credential = ticket == null ? string.Empty : ticket.JoinCredential;
        }

        /// <summary>A new socket is being opened. Call before <see cref="Opened"/>.</summary>
        public void Attempting()
        {
            if (_authFailed) return;
            State = SteamMatchSessionState.Connecting;
            ClearHandshake();
        }

        /// <summary>The socket opened: AUTH goes out first, and the handshake deadline starts.</summary>
        public void Opened()
        {
            if (_authFailed) return;
            State = SteamMatchSessionState.Authenticating;
            _handshakeArmed = true;
            _handshakeAnchor = null;
            Emit(SteamMatchSessionOutputKind.Send, SteamMatchProtocol.AuthFrame(_matchId, _credential));
        }

        /// <summary>One inbound frame.</summary>
        public void Frame(string? raw)
        {
            if (raw == null) return;
            if (State == SteamMatchSessionState.Closed) return;

            string code;
            if (SteamMatchProtocol.TryParseAuthFail(raw, out code))
            {
                _authFailed = true;
                State = SteamMatchSessionState.Closed;
                ClearHandshake();
                Emit(SteamMatchSessionOutputKind.AuthFailed, code);
                return;
            }

            if (string.Equals(raw, SteamMatchProtocol.Ping, StringComparison.Ordinal))
            {
                Emit(SteamMatchSessionOutputKind.Send, SteamMatchProtocol.Pong);
                return;
            }

            if (string.Equals(raw, SteamMatchProtocol.ServerRestart, StringComparison.Ordinal))
            {
                _expectRestart = true;   // the close right behind this is planned, so retry at once
                return;
            }

            var msg = NetProtocol.Parse(raw);

            // SEAT is the server accepting AUTH. Nothing before it may touch the game, or a socket
            // that was never authenticated could deal a board.
            if (State != SteamMatchSessionState.Seated
                && !string.Equals(msg.Type, "SEAT", StringComparison.Ordinal)) return;

            switch (msg.Type)
            {
                case "SEAT":
                    OnSeat(msg.Payload);
                    return;
                case "CATALOG?":
                    Emit(SteamMatchSessionOutputKind.CatalogRequested);
                    return;
                case "START":
                    GameStarted = true;
                    Emit(SteamMatchSessionOutputKind.Start, msg.Payload);
                    return;
                case "APPLY":
                    Emit(SteamMatchSessionOutputKind.Apply, msg.Payload);
                    return;
                case "REJECT":
                    Emit(SteamMatchSessionOutputKind.Reject, msg.Payload);
                    return;
            }
        }

        /// <summary>The socket closed.</summary>
        public void Closed()
        {
            EndAttempt(false);
        }

        /// <summary>
        /// Advances the clock. The handshake deadline is relative: it is anchored on the first tick
        /// that observes it, so a driver whose clock starts long after zero is never late.
        /// </summary>
        public void Tick(double nowSeconds)
        {
            if (!_handshakeArmed) return;
            if (_handshakeAnchor == null) { _handshakeAnchor = nowSeconds; return; }
            if (nowSeconds < _handshakeAnchor.Value + HandshakeTimeoutSeconds) return;
            EndAttempt(true);
        }

        /// <summary>A fresh ticket arrived (or null, which is a give-up).</summary>
        public void CredentialRefreshed(SteamMatchTicket? ticket)
        {
            if (ticket == null) { CredentialRefreshFailed(); return; }
            UseTicket(ticket);
        }

        /// <summary>No fresh credential could be had. Replaying a dead one can never work.</summary>
        public void CredentialRefreshFailed()
        {
            State = SteamMatchSessionState.Closed;
            ClearHandshake();
            Emit(SteamMatchSessionOutputKind.GiveUp);
        }

        /// <summary>Takes every queued output, leaving the queue empty.</summary>
        public IReadOnlyList<SteamMatchSessionOutput> Drain()
        {
            if (_outputs.Count == 0) return NoOutputs;
            var taken = _outputs.ToArray();
            _outputs.Clear();
            return taken;
        }

        // ----- internals ---------------------------------------------------------------------

        void OnSeat(string payload)
        {
            ClearHandshake();

            if (string.Equals(payload, "FULL", StringComparison.Ordinal))
            {
                State = SteamMatchSessionState.Closed;
                Emit(SteamMatchSessionOutputKind.SeatFull);
                return;
            }

            int seat;
            if (!int.TryParse(payload, NumberStyles.Integer, CultureInfo.InvariantCulture, out seat)) return;

            State = SteamMatchSessionState.Seated;
            if (_attempt > 0)
            {
                _attempt = 0;
                Emit(SteamMatchSessionOutputKind.Reconnected);
            }
            Emit(SteamMatchSessionOutputKind.Seat, payload, seat);
        }

        void EndAttempt(bool handshakeTimedOut)
        {
            if (State == SteamMatchSessionState.Closed) return;

            State = SteamMatchSessionState.Closed;
            ClearHandshake();
            if (_authFailed) return;

            var plannedRestart = _expectRestart;
            _expectRestart = false;

            // A first drop with nothing started is the pre-start case: no seat to come back to, so it
            // keeps the v1 toast-and-stop. A planned restart, and a handshake that simply never
            // answered, are both always worth another attempt.
            if (!handshakeTimedOut && _attempt == 0 && !GameStarted && !plannedRestart)
            {
                Emit(SteamMatchSessionOutputKind.Closed);
                return;
            }

            Emit(SteamMatchSessionOutputKind.Reconnecting, null, _attempt);
            var delay = plannedRestart ? 0 : BackoffSeconds[Math.Min(_attempt, BackoffSeconds.Length - 1)];
            _attempt++;
            Emit(SteamMatchSessionOutputKind.Retry, null, 0, delay);
        }

        void ClearHandshake()
        {
            _handshakeArmed = false;
            _handshakeAnchor = null;
        }

        void Emit(SteamMatchSessionOutputKind kind, string? text = null, int seat = 0, double delaySeconds = 0)
        {
            _outputs.Add(new SteamMatchSessionOutput(kind, text, seat, delaySeconds));
        }
    }
}

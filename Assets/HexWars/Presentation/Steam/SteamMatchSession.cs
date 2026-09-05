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

        /// <summary>The far end broke the protocol. The output text is the close code to use.</summary>
        ProtocolViolation,
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
            var rendered = Kind == SteamMatchSessionOutputKind.Send ? Redact(Text) : Text;
            return Kind + "(" + rendered + ", seat=" + Seat + ", delay="
                 + DelaySeconds.ToString(CultureInfo.InvariantCulture) + ")";
        }

        /// <summary>
        /// An outbound frame is rendered as its verb and nothing else. AUTH carries the single-use
        /// join credential and the match id, and no diagnostic is worth putting those in a log or a
        /// crash report. The same rule covers every other Send, so a frame added later cannot leak
        /// a payload just because nobody remembered to redact it.
        /// </summary>
        static string Redact(string text)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;
            var space = text.IndexOf(" ", StringComparison.Ordinal);
            return space < 0 ? text : text.Substring(0, space) + " <redacted>";
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
        int _currentAttempt;
        int _staleEventsIgnored;
        bool _authFailed;
        bool _protocolViolated;
        bool _expectRestart;

        public SteamMatchSession()
        {
            ArmHandshake();
        }

        public SteamMatchSession(SteamMatchTicket ticket)
        {
            UseTicket(ticket);
            ArmHandshake();
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

        /// <summary>
        /// The id of the socket attempt that is live now. Every event the driver reports carries the
        /// attempt it belongs to, and anything that is not this one is dropped.
        /// </summary>
        public int CurrentAttempt { get { return _currentAttempt; } }

        /// <summary>How many driver events were dropped because they belonged to an old attempt.</summary>
        public int StaleEventsIgnored { get { return _staleEventsIgnored; } }

        /// <summary>True when the far end sent something the protocol does not allow.</summary>
        public bool ProtocolViolated { get { return _protocolViolated; } }

        /// <summary>Adopts a (re-issued) ticket for the next AUTH frame.</summary>
        public void UseTicket(SteamMatchTicket? ticket)
        {
            _matchId = ticket == null ? string.Empty : ticket.MatchId;
            _credential = ticket == null ? string.Empty : ticket.JoinCredential;
        }

        /// <summary>
        /// A new socket is being opened. Returns the id every event for that socket must carry.
        /// <para>
        /// The handshake deadline is armed here rather than on <see cref="Opened"/>, so that ONE
        /// deadline covers connect, AUTH and SEAT together. A socket that hangs on connect used to be
        /// covered by nothing at all: the deadline only started once the socket opened, so a connect
        /// that never completed left the game on "Connecting..." for good.
        /// </para>
        /// </summary>
        public int BeginAttempt()
        {
            _currentAttempt++;
            if (_authFailed || _protocolViolated) return _currentAttempt;

            State = SteamMatchSessionState.Connecting;
            ArmHandshake();
            return _currentAttempt;
        }

        /// <summary>The current attempt socket opened. AUTH is the very first frame out.</summary>
        public void Opened()
        {
            Opened(_currentAttempt);
        }

        /// <summary>The socket for <paramref name="attemptId"/> opened. A stale id is ignored.</summary>
        public void Opened(int attemptId)
        {
            if (!IsCurrentAttempt(attemptId)) return;
            if (_authFailed || _protocolViolated) return;
            // An attempt the deadline already ended is over even though its id is still the current
            // one: an open arriving now would re-send AUTH with a credential that is already spent.
            if (State == SteamMatchSessionState.Closed) return;

            State = SteamMatchSessionState.Authenticating;
            // The deadline is NOT restarted here: it was armed for the whole attempt in BeginAttempt.
            Emit(SteamMatchSessionOutputKind.Send, SteamMatchProtocol.AuthFrame(_matchId, _credential));
        }

        /// <summary>One inbound frame on the current attempt.</summary>
        public void Frame(string? raw)
        {
            Frame(_currentAttempt, raw);
        }

        /// <summary>One inbound frame on <paramref name="attemptId"/>. A stale id is ignored.</summary>
        public void Frame(int attemptId, string? raw)
        {
            if (!IsCurrentAttempt(attemptId)) return;
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

        /// <summary>The current attempt socket closed.</summary>
        public void Closed()
        {
            Closed(_currentAttempt);
        }

        /// <summary>
        /// The socket for <paramref name="attemptId"/> closed. A stale id is ignored, which is what
        /// stops the late OnClose of an abandoned attempt from ending the one that replaced it.
        /// </summary>
        public void Closed(int attemptId)
        {
            if (!IsCurrentAttempt(attemptId)) return;
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

        /// <summary>
        /// SEAT is the server accepting AUTH, so it is validated before anything is acted on and
        /// before the handshake deadline is touched.
        /// <para>
        /// Three rules, each of which was a way in. It is accepted ONLY in Authenticating, so a SEAT
        /// that arrives before our own AUTH went out cannot seat an unauthenticated socket. The
        /// payload must be exactly 0, 1 or FULL, so a seat index the game has no player for can never
        /// reach the barracks cache. And the deadline is cleared only once the payload is known good:
        /// clearing it first meant a single malformed SEAT disarmed the timeout and hung the attempt
        /// with nothing left to end it.
        /// </para>
        /// </summary>
        void OnSeat(string payload)
        {
            if (State != SteamMatchSessionState.Authenticating) return;

            if (string.Equals(payload, "FULL", StringComparison.Ordinal))
            {
                ClearHandshake();
                State = SteamMatchSessionState.Closed;
                Emit(SteamMatchSessionOutputKind.SeatFull);
                return;
            }

            if (!string.Equals(payload, "0", StringComparison.Ordinal)
                && !string.Equals(payload, "1", StringComparison.Ordinal))
            {
                ProtocolViolation();
                return;
            }

            var seat = string.Equals(payload, "1", StringComparison.Ordinal) ? 1 : 0;

            ClearHandshake();
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

        /// <summary>The far end broke the protocol: end the attempt and never try it again.</summary>
        void ProtocolViolation()
        {
            _protocolViolated = true;
            State = SteamMatchSessionState.Closed;
            ClearHandshake();
            Emit(SteamMatchSessionOutputKind.ProtocolViolation, SteamMatchProtocol.ProtocolCloseCode);
            Emit(SteamMatchSessionOutputKind.GiveUp);
        }

        /// <summary>True while the event belongs to the live attempt; a stale one is counted here.</summary>
        bool IsCurrentAttempt(int attemptId)
        {
            if (attemptId == _currentAttempt) return true;
            _staleEventsIgnored++;
            return false;
        }

        void ArmHandshake()
        {
            _handshakeArmed = true;
            _handshakeAnchor = null;
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

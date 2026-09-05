#nullable enable
using System;
using System.Collections;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using NativeWebSocket;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// The protocol-v2 link to a durable Steam match: the socket, and nothing else. Every decision
    /// about what a frame means, when an attempt is over and how long to wait before the next one
    /// lives in <see cref="SteamMatchSession"/>, which is plain C# and unit tested; this class opens
    /// sockets, feeds the session, and carries out what it asks for.
    /// <para>
    /// Three things differ from the legacy v1 <see cref="NetClient"/> path, which is untouched. One:
    /// there is no room in the URL and no token in the query string, so the first frame is
    /// <c>AUTH matchId credential</c> and the seat only counts once SEAT comes back. Two: a join
    /// credential is single-use, so every retry asks the caller for a fresh one first. Three: the
    /// server speaks PING and announces a planned restart with SERVER RESTART.
    /// </para>
    /// <para>
    /// Attempts are isolated by id. A socket that is being abandoned keeps firing for a while: its
    /// OnClose lands late, and so can a SEAT the server sent before it noticed. Every event carries
    /// the attempt it belongs to (see <see cref="ISteamSocketDriver"/>), the session ignores anything
    /// that is not the live attempt, and the lifecycle waits - bounded - for the previous socket to
    /// finish closing before the next one opens.
    /// </para>
    /// <para>
    /// The socket callbacks may arrive off the main thread, so they only queue into
    /// <see cref="SteamMatchSocketPump"/>. <see cref="Update"/> is the single place the session is
    /// driven and its outputs are carried out, which keeps every Unity call on the main thread.
    /// </para>
    /// </summary>
    public sealed class SteamMatchConnection : MonoBehaviour
    {
        /// <summary>How long a credential refresh may take before the drop counts as unrecoverable.</summary>
        public const float RefreshTimeoutSeconds = 30f;

        /// <summary>How long the previous attempt gets to finish closing before the next one opens.</summary>
        public const float AttemptCloseTimeoutSeconds = 2f;

        readonly SteamMatchSession _session = new SteamMatchSession();

        NativeWebSocketDriver? _driver;
        SteamMatchSocketPump? _pump;
        GameBootstrap? _game;
        SteamMatchTicket? _ticket;
        Func<Action<SteamMatchTicket?>, bool>? _refresh;

        /// <summary>The seat the match service assigned (null until SEAT arrives, or if the match is full).</summary>
        public PlayerId? Seat { get; private set; }

        /// <summary>True only after the server accepted the credential and dealt a seat.</summary>
        public bool Connected { get; private set; }

        bool _closing;      // deliberate teardown (Cancel / Main menu), not an error
        bool _stopped;      // the session will not try again
        bool _retryRequested;
        double _retryDelay;
        bool _refreshPending;
        int _refreshGeneration;

        /// <summary>
        /// Starts the connection lifecycle. <paramref name="refreshCredential"/> is called before every
        /// retry: it must start a reissue and answer on the main thread with a fresh ticket, or with null
        /// to give up. Returning false means no reissue could even be started, which also gives up.
        /// </summary>
        public void Connect(GameBootstrap game, SteamMatchTicket ticket, Func<Action<SteamMatchTicket?>, bool> refreshCredential)
        {
            _game = game;
            _ticket = ticket;
            _refresh = refreshCredential;
            _session.UseTicket(ticket);
            _driver = new NativeWebSocketDriver(message => Debug.LogWarning("[SteamNet] " + message));
            _pump = new SteamMatchSocketPump(_session, _driver, message => Debug.LogError("[SteamNet] " + message));
            StartCoroutine(Lifecycle());
        }

        /// <summary>
        /// One attempt per pass. An attempt ends on its own socket close event OR on the session
        /// handshake deadline, which covers connect as well as the handshake: a socket that never
        /// finishes connecting, and one that opens and then never answers AUTH, both used to hang the
        /// game on "Connecting..." for good.
        /// </summary>
        IEnumerator Lifecycle()
        {
            while (true)
            {
                var attempt = OpenOnce();

                while (!_retryRequested && !_stopped && !_closing) yield return null;
                if (_closing || _stopped) yield break;

                _retryRequested = false;
                var wait = _retryDelay;

                // Wait for the abandoned socket to finish closing before the next one opens, so its
                // last events land while it is still the attempt they belong to. Bounded: a socket
                // that will not close must not hold the reconnect up for ever.
                yield return CloseAttempt(attempt);
                if (_closing) yield break;

                if (wait > 0) yield return new WaitForSeconds((float)wait);
                if (_closing) yield break;

                if (BeginRefresh())
                {
                    var deadline = Time.unscaledTime + RefreshTimeoutSeconds;
                    while (_refreshPending && !_closing && Time.unscaledTime < deadline) yield return null;
                    if (_closing) yield break;
                    if (_refreshPending)
                    {
                        _refreshPending = false;
                        _session.CredentialRefreshFailed();   // no answer in time is a give-up
                    }
                    ExecuteOutputs();
                }
                if (_closing || _stopped) yield break;
            }
        }

        /// <summary>Closes one attempt and waits, bounded, for the close to finish.</summary>
        IEnumerator CloseAttempt(int attemptId)
        {
            var driver = _driver;
            if (driver == null) yield break;

            Task? closing = null;
            try { closing = driver.CloseAsync(attemptId); }
            catch (Exception e) { Debug.LogWarning("[SteamNet] close failed: " + e.Message); }
            if (closing == null) yield break;

            var deadline = Time.unscaledTime + AttemptCloseTimeoutSeconds;
            while (!closing.IsCompleted && Time.unscaledTime < deadline) yield return null;

            if (!closing.IsCompleted)
            {
                Debug.LogWarning("[SteamNet] attempt " + attemptId + " did not close in time; abandoning its socket");
            }
            else if (closing.IsFaulted)
            {
                var reason = closing.Exception == null ? "unknown" : closing.Exception.GetBaseException().Message;
                Debug.LogWarning("[SteamNet] close failed: " + reason);
            }
        }

        /// <summary>Asks for a fresh credential. Returns false when there is nothing to wait for.</summary>
        bool BeginRefresh()
        {
            var refresh = _refresh;
            if (refresh == null) return false;

            var generation = ++_refreshGeneration;
            _refreshPending = true;

            bool started;
            try { started = refresh(ticket => OnRefreshed(generation, ticket)); }
            catch (Exception e)
            {
                Debug.LogError("[SteamNet] credential refresh threw: " + e.Message);
                started = false;
            }

            if (!started)
            {
                _refreshPending = false;
                _session.CredentialRefreshFailed();   // never replay a credential that is already spent
            }
            return true;
        }

        void OnRefreshed(int generation, SteamMatchTicket? ticket)
        {
            if (generation != _refreshGeneration) return;   // a stale answer from an abandoned attempt
            _refreshPending = false;
            if (ticket != null) _ticket = ticket;
            _session.CredentialRefreshed(ticket);
        }

        /// <summary>Begins one attempt and hands back the id every event for it must carry.</summary>
        int OpenOnce()
        {
            var attempt = _session.BeginAttempt();

            var ticket = _ticket;
            var driver = _driver;
            if (driver == null || ticket == null || string.IsNullOrEmpty(ticket.WebsocketUrl))
            {
                Debug.LogError("[SteamNet] no match socket url");
                _session.Closed(attempt);
                ExecuteOutputs();
                return attempt;
            }

            Debug.Log("[SteamNet] connecting to " + SafeUrl(ticket.WebsocketUrl));
            driver.Open(ticket.WebsocketUrl, attempt);
            return attempt;
        }

        public void Send(string message)
        {
            var driver = _driver;
            if (driver == null) return;
            driver.Send(_session.CurrentAttempt, message);
        }

        void Update()
        {
            var driver = _driver;
            if (driver != null) driver.DispatchMessages();
            if (_closing) return;

            // a started game means a drop has something to reconnect into
            if (_game != null && _game.State != null) _session.GameStarted = true;

            var pump = _pump;
            if (pump != null) pump.Pump();

            _session.Tick(Time.unscaledTimeAsDouble);
            ExecuteOutputs();
        }

        /// <summary>Carries out whatever the session asked for. Main thread only.</summary>
        void ExecuteOutputs()
        {
            var game = _game;
            foreach (var output in _session.Drain())
            {
                switch (output.Kind)
                {
                    case SteamMatchSessionOutputKind.Send:
                        Send(output.Text);
                        break;
                    case SteamMatchSessionOutputKind.CatalogRequested:
                        Send(StartingCatalogMessage());
                        break;
                    case SteamMatchSessionOutputKind.Seat:
                        Seat = (PlayerId)output.Seat;
                        Connected = true;
                        if (game != null) game.OnNetSeat(Seat.Value);
                        break;
                    case SteamMatchSessionOutputKind.Reconnected:
                        if (game != null) game.OnNetReconnected();
                        break;
                    case SteamMatchSessionOutputKind.SeatFull:
                        Seat = null;
                        Connected = false;
                        _stopped = true;
                        if (game != null) game.OnNetSeatFull();
                        break;
                    case SteamMatchSessionOutputKind.Start:
                        if (game != null) game.OnNetStart(output.Text);
                        break;
                    case SteamMatchSessionOutputKind.Apply:
                        if (game != null) game.OnNetApply(CommandWire.Read(output.Text));
                        break;
                    case SteamMatchSessionOutputKind.Reject:
                        if (game != null) game.OnNetReject(output.Text);
                        break;
                    case SteamMatchSessionOutputKind.AuthFailed:
                        Connected = false;
                        _stopped = true;
                        Debug.LogWarning("[SteamNet] auth rejected: " + output.Text);
                        if (game != null) game.OnNetAuthFailed(output.Text);
                        break;
                    case SteamMatchSessionOutputKind.ProtocolViolation:
                        // The far end sent something the protocol does not allow. Drop the socket at
                        // once rather than keep reading from a peer that is not speaking v2.
                        Connected = false;
                        _stopped = true;
                        Debug.LogWarning("[SteamNet] protocol violation; closing the socket (code "
                                         + output.Text + ")");
                        CloseCurrentAttempt();
                        break;
                    case SteamMatchSessionOutputKind.Reconnecting:
                        Connected = false;
                        if (game != null) game.OnNetReconnecting(output.Seat);
                        break;
                    case SteamMatchSessionOutputKind.Retry:
                        _retryRequested = true;
                        _retryDelay = output.DelaySeconds;
                        break;
                    case SteamMatchSessionOutputKind.Closed:
                        Connected = false;
                        _stopped = true;
                        if (game != null) game.OnNetClosed();
                        break;
                    case SteamMatchSessionOutputKind.GiveUp:
                        // No fresh credential, or a peer that broke the protocol: this connection can
                        // never come back, so it takes itself down and the bootstrap puts the player
                        // back on the title.
                        Connected = false;
                        _stopped = true;
                        CloseCurrentAttempt();
                        if (game != null) game.OnSteamReconnectAbandoned();
                        Destroy(this);
                        return;
                }
            }
        }

        void CloseCurrentAttempt()
        {
            var driver = _driver;
            if (driver == null) return;
            try { driver.CloseAsync(_session.CurrentAttempt); }
            catch (Exception e) { Debug.LogWarning("[SteamNet] close failed: " + e.Message); }
        }

        string StartingCatalogMessage()
        {
            var seat = Seat ?? PlayerId.Player0;
            var catalog = SessionBarracksCache.ForLocalPlayer((int)seat).Snapshot();
            return NetProtocol.Catalog(BarracksWire.Write(catalog));
        }

        void OnDestroy()
        {
            _closing = true;        // deliberate teardown (Cancel / Main menu), not an error
            StopAllCoroutines();

            var pump = _pump;
            _pump = null;
            if (pump != null) pump.Dispose();

            var driver = _driver;
            _driver = null;
            if (driver != null) driver.CloseAll();
        }

        /// <summary>Log-safe form of the socket URL. The credential is never in a URL, and this keeps it
        /// that way even if a future service starts adding query strings.</summary>
        internal static string SafeUrl(string? url)
        {
            if (string.IsNullOrEmpty(url)) return string.Empty;
            try
            {
                var uri = new Uri(url!);
                return uri.Scheme + "://" + uri.Authority + uri.AbsolutePath;
            }
            catch (Exception) { return "(unparsed)"; }
        }
    }

    /// <summary>
    /// <see cref="ISteamSocketDriver"/> on NativeWebSocket. Every handler closes over the attempt id
    /// its socket was opened for, which is what makes a late event from a dying socket identifiable
    /// as such instead of being fed to whichever attempt happens to be live.
    /// </summary>
    sealed class NativeWebSocketDriver : ISteamSocketDriver
    {
        readonly object _gate = new object();
        readonly Action<string> _log;

        WebSocket? _socket;
        int _socketAttempt;

        public NativeWebSocketDriver(Action<string>? log = null)
        {
            _log = log ?? (_ => { });
        }

        public event Action<int>? Opened;
        public event Action<int, string>? Message;
        public event Action<int, string>? Closed;
        public event Action<int, string>? Error;

        public void Open(string url, int attemptId)
        {
            CloseAll();   // an attempt never inherits the socket of the one before it

            WebSocket socket;
            try { socket = new WebSocket(url); }
            catch (Exception e)
            {
                RaiseError(attemptId, "socket create failed: " + e.Message);
                RaiseClosed(attemptId, "create-failed");
                return;
            }

            lock (_gate)
            {
                _socket = socket;
                _socketAttempt = attemptId;
            }

            socket.OnOpen += () => RaiseOpened(attemptId);
            socket.OnError += message => RaiseError(attemptId, message);
            socket.OnClose += code => RaiseClosed(attemptId, code.ToString());
            socket.OnMessage += data => RaiseMessage(attemptId, Encoding.UTF8.GetString(data));

            // The Connect Task is only observed for faults, because on WebGL it completes the moment
            // the JS socket is kicked; the open itself arrives as OnOpen.
            try
            {
                socket.Connect().ContinueWith(task =>
                {
                    if (!task.IsFaulted) return;
                    var reason = task.Exception == null ? "unknown" : task.Exception.GetBaseException().Message;
                    RaiseError(attemptId, "connect faulted: " + reason);
                    RaiseClosed(attemptId, "connect-faulted");
                }, TaskScheduler.Default);
            }
            catch (Exception e)
            {
                RaiseError(attemptId, "connect failed: " + e.Message);
                RaiseClosed(attemptId, "connect-failed");
            }
        }

        public void Send(int attemptId, string text)
        {
            WebSocket? socket;
            lock (_gate)
            {
                if (_socketAttempt != attemptId) return;
                socket = _socket;
            }
            if (socket == null || socket.State != WebSocketState.Open) return;
            SendText(socket, text);
        }

        public Task CloseAsync(int attemptId)
        {
            WebSocket? socket;
            lock (_gate)
            {
                if (_socket == null || _socketAttempt != attemptId) return Task.CompletedTask;
                socket = _socket;
                _socket = null;
                _socketAttempt = 0;
            }
            return CloseSocket(socket);
        }

        /// <summary>Closes whatever socket is still held, whichever attempt it belonged to.</summary>
        public void CloseAll()
        {
            WebSocket? socket;
            lock (_gate)
            {
                socket = _socket;
                _socket = null;
                _socketAttempt = 0;
            }
            if (socket == null) return;
            var _ = CloseSocket(socket);   // fire and forget: nobody is waiting on this one
        }

        /// <summary>Drains the socket queue. WebGL dispatches its own, so this is a no-op there.</summary>
        public void DispatchMessages()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            WebSocket? socket;
            lock (_gate) { socket = _socket; }
            if (socket != null) socket.DispatchMessageQueue();
#endif
        }

        async Task CloseSocket(WebSocket socket)
        {
            try { await socket.Close(); }
            catch (Exception e) { _log("close failed: " + e.Message); }
        }

        async void SendText(WebSocket socket, string text)
        {
            try { await socket.SendText(text); }
            catch (Exception e) { _log("send failed: " + e.Message); }
        }

        void RaiseOpened(int attemptId)
        {
            var handler = Opened;
            if (handler != null) handler(attemptId);
        }

        void RaiseMessage(int attemptId, string text)
        {
            var handler = Message;
            if (handler != null) handler(attemptId, text);
        }

        void RaiseClosed(int attemptId, string reason)
        {
            var handler = Closed;
            if (handler != null) handler(attemptId, reason);
        }

        void RaiseError(int attemptId, string message)
        {
            var handler = Error;
            if (handler != null) handler(attemptId, message);
        }
    }
}

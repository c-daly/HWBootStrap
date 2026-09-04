#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
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
    /// The socket callbacks may arrive off the main thread, so they only set volatile flags and queue
    /// raw frames. <see cref="Update"/> is the single place the session is driven and its outputs are
    /// carried out, which keeps every Unity call on the main thread.
    /// </para>
    /// </summary>
    public sealed class SteamMatchConnection : MonoBehaviour
    {
        /// <summary>How long a credential refresh may take before the drop counts as unrecoverable.</summary>
        public const float RefreshTimeoutSeconds = 30f;

        readonly SteamMatchSession _session = new SteamMatchSession();
        readonly Queue<string> _inbound = new Queue<string>();

        WebSocket? _ws;
        GameBootstrap? _game;
        SteamMatchTicket? _ticket;
        Func<Action<SteamMatchTicket?>, bool>? _refresh;

        /// <summary>The seat the match service assigned (null until SEAT arrives, or if the match is full).</summary>
        public PlayerId? Seat { get; private set; }

        /// <summary>True only after the server accepted the credential and dealt a seat.</summary>
        public bool Connected { get; private set; }

        bool _closing;                 // deliberate teardown (Cancel / Main menu), not an error
        volatile bool _socketOpened;   // the socket opened; the session has not been told yet
        volatile bool _attemptClosed;  // this attempt is over (OnClose fired, or it never opened)
        bool _stopped;                 // the session will not try again
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
            StartCoroutine(Lifecycle());
        }

        /// <summary>
        /// One attempt per pass. An attempt ends on the socket close event OR on the session handshake
        /// deadline, which is what stops a socket that opens and then never answers AUTH from hanging
        /// the game on "Connecting..." for good.
        /// </summary>
        IEnumerator Lifecycle()
        {
            while (true)
            {
                OpenOnce();
                while (!_attemptClosed && !_retryRequested && !_stopped && !_closing) yield return null;
                if (_closing) yield break;

                // The deadline may already have ended this attempt; only a real close needs telling.
                if (_attemptClosed && !_retryRequested && !_stopped)
                {
                    _session.Closed();
                    ExecuteOutputs();
                }
                if (_closing || _stopped) yield break;
                if (!_retryRequested) yield break;

                _retryRequested = false;
                var wait = _retryDelay;
                CloseSocket();
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

        /// <summary>One attempt, event-driven exactly as in <see cref="NetClient"/>: the Connect Task is
        /// only observed for faults, because on WebGL it completes the moment the JS socket is kicked.</summary>
        void OpenOnce()
        {
            _attemptClosed = false;
            _socketOpened = false;
            _session.Attempting();

            var ticket = _ticket;
            if (ticket == null || string.IsNullOrEmpty(ticket.WebsocketUrl))
            {
                Debug.LogError("[SteamNet] no match socket url");
                _attemptClosed = true;
                return;
            }

            Debug.Log("[SteamNet] connecting to " + SafeUrl(ticket.WebsocketUrl));
            try { _ws = new WebSocket(ticket.WebsocketUrl); }
            catch (Exception e)
            {
                Debug.LogError("[SteamNet] socket create failed: " + e.Message);
                _attemptClosed = true;
                return;
            }

            _ws.OnOpen += () =>
            {
                Debug.Log("[SteamNet] open");
                _socketOpened = true;   // Update sends AUTH, on the main thread
            };
            _ws.OnError += e => Debug.LogError("[SteamNet] error: " + e);
            _ws.OnClose += c =>
            {
                Connected = false;
                Debug.Log("[SteamNet] closed: " + c);
                _attemptClosed = true;
            };
            _ws.OnMessage += data =>
            {
                var raw = Encoding.UTF8.GetString(data);
                lock (_inbound) _inbound.Enqueue(raw);
            };

            try
            {
                _ws.Connect().ContinueWith(t =>
                {
                    if (!t.IsFaulted) return;
                    Debug.LogError("[SteamNet] connect faulted: " + t.Exception.GetBaseException().Message);
                    _attemptClosed = true;
                }, System.Threading.Tasks.TaskScheduler.Default);
            }
            catch (Exception e)
            {
                Debug.LogError("[SteamNet] connect failed: " + e.Message);
                _attemptClosed = true;
            }
        }

        public async void Send(string message)
        {
            var ws = _ws;
            if (ws != null && ws.State == WebSocketState.Open) await ws.SendText(message);
        }

        void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            _ws?.DispatchMessageQueue();
#endif
            if (_closing) return;

            // a started game means a drop has something to reconnect into
            if (_game != null && _game.State != null) _session.GameStarted = true;

            if (_socketOpened) { _socketOpened = false; _session.Opened(); }

            while (true)
            {
                string? frame = null;
                lock (_inbound) { if (_inbound.Count > 0) frame = _inbound.Dequeue(); }
                if (frame == null) break;
                _session.Frame(frame);
            }

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
                        // No fresh credential: this connection can never come back, so it takes itself
                        // down and the bootstrap puts the player back on the title.
                        Connected = false;
                        _stopped = true;
                        if (game != null) game.OnSteamReconnectAbandoned();
                        Destroy(this);
                        return;
                }
            }
        }

        string StartingCatalogMessage()
        {
            var seat = Seat ?? PlayerId.Player0;
            var catalog = SessionBarracksCache.ForLocalPlayer((int)seat).Snapshot();
            return NetProtocol.Catalog(BarracksWire.Write(catalog));
        }

        async void CloseSocket()
        {
            var ws = _ws;
            _ws = null;
            if (ws == null) return;
            try { await ws.Close(); }
            catch (Exception e) { Debug.LogWarning("[SteamNet] close failed: " + e.Message); }
        }

        async void OnDestroy()
        {
            _closing = true;        // deliberate teardown (Cancel / Main menu), not an error
            StopAllCoroutines();
            var ws = _ws;
            _ws = null;
            if (ws != null) await ws.Close();
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
}

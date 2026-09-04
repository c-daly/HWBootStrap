#nullable enable
using System;
using System.Collections;
using System.Globalization;
using System.Text;
using UnityEngine;
using NativeWebSocket;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// The protocol-v2 link to a durable Steam match. Same shape as <see cref="NetClient"/> (which still
    /// owns the legacy v1 browser path, untouched): one event-driven attempt at a time, the attempt ends
    /// on the socket OnClose event rather than on the Connect Task, and retries use capped exponential
    /// backoff. Three things differ from v1.
    /// <para>
    /// One: there is no room in the URL and no token in the query string. The socket opens on the URL the
    /// match service handed out and the very first frame is <c>AUTH matchId credential</c>; the seat only
    /// counts as live once SEAT comes back.
    /// </para>
    /// <para>
    /// Two: a join credential is single-use and short-lived, so every RETRY asks the caller for a fresh
    /// one first (the lobby coordinator reissues it through the match service). A caller that answers null
    /// has given up, and the drop becomes an ordinary connection-lost.
    /// </para>
    /// <para>
    /// Three: the server speaks PING (answered with PONG) and announces a planned restart with
    /// SERVER RESTART. A restart close is always retryable, immediately, even before a game started.
    /// </para>
    /// </summary>
    public sealed class SteamMatchConnection : MonoBehaviour
    {
        /// <summary>How long a credential refresh may take before the drop counts as unrecoverable.</summary>
        public const float RefreshTimeoutSeconds = 30f;

        static readonly float[] BackoffSeconds = { 1f, 2f, 4f, 8f, 15f }; // caps at 15s, then repeats

        WebSocket? _ws;
        GameBootstrap? _game;
        SteamMatchTicket? _ticket;
        Func<Action<SteamMatchTicket?>, bool>? _refresh;

        /// <summary>The seat the match service assigned (null until SEAT arrives, or if the match is full).</summary>
        public PlayerId? Seat { get; private set; }

        /// <summary>True only after the server accepted the credential and dealt a seat.</summary>
        public bool Connected { get; private set; }

        bool _closing;
        int _attempt;                  // 0 = first-ever attempt; >0 = a retry after a drop
        volatile bool _attemptClosed;  // this attempt is over (OnClose fired, or it never opened)
        bool _authFailed;              // the credential was rejected: retrying cannot fix that
        bool _expectRestart;           // SERVER RESTART arrived, so the close behind it is planned
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
            StartCoroutine(Lifecycle());
        }

        IEnumerator Lifecycle()
        {
            while (true)
            {
                OpenOnce();
                while (!_attemptClosed) yield return null;
                if (_closing) yield break;

                var game = _game;
                if (game == null) yield break;
                if (_authFailed) yield break;   // GameBootstrap.OnNetAuthFailed already took the player home

                var plannedRestart = _expectRestart;
                _expectRestart = false;

                // A first drop with no game yet is the pre-start case: nothing to reconnect into, so it
                // keeps the v1 toast-and-stop behaviour. A restart is always worth retrying.
                if (_attempt == 0 && game.State == null && !plannedRestart)
                {
                    game.OnNetClosed();
                    yield break;
                }

                game.OnNetReconnecting(_attempt);
                var wait = plannedRestart ? 0f : BackoffSeconds[Mathf.Min(_attempt, BackoffSeconds.Length - 1)];
                _attempt++;
                if (wait > 0f) yield return new WaitForSeconds(wait);
                if (_closing) yield break;

                if (BeginRefresh())
                {
                    var deadline = Time.unscaledTime + RefreshTimeoutSeconds;
                    while (_refreshPending && !_closing && Time.unscaledTime < deadline) yield return null;
                    if (_closing) yield break;
                    if (_refreshPending)
                    {
                        _refreshPending = false;
                        _ticket = null;   // no answer in time: treat it as a give-up
                    }
                }

                if (_ticket == null) { game.OnNetClosed(); yield break; }
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
                _ticket = null;   // no reissue possible: give up rather than replay a dead credential
            }
            return true;
        }

        void OnRefreshed(int generation, SteamMatchTicket? ticket)
        {
            if (generation != _refreshGeneration) return;   // a stale answer from an abandoned attempt
            _refreshPending = false;
            _ticket = ticket;
        }

        /// <summary>One attempt, event-driven exactly as in <see cref="NetClient"/>: the Connect Task is
        /// only observed for faults, because on WebGL it completes the moment the JS socket is kicked.</summary>
        void OpenOnce()
        {
            _attemptClosed = false;

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
                var current = _ticket;
                if (current != null) Send(SteamMatchProtocol.AuthFrame(current.MatchId, current.JoinCredential));
            };
            _ws.OnError += e => Debug.LogError("[SteamNet] error: " + e);
            _ws.OnClose += c =>
            {
                Connected = false;
                Debug.Log("[SteamNet] closed: " + c);
                _attemptClosed = true;
            };
            _ws.OnMessage += OnMessage;

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

        void OnMessage(byte[] data)
        {
            var game = _game;
            if (game == null) return;

            var raw = Encoding.UTF8.GetString(data);

            string code;
            if (SteamMatchProtocol.TryParseAuthFail(raw, out code))
            {
                _authFailed = true;
                Connected = false;
                Debug.LogWarning("[SteamNet] auth rejected: " + code);
                game.OnNetAuthFailed(code);
                return;
            }

            if (string.Equals(raw, SteamMatchProtocol.Ping, StringComparison.Ordinal))
            {
                Send(SteamMatchProtocol.Pong);
                return;
            }

            if (string.Equals(raw, SteamMatchProtocol.ServerRestart, StringComparison.Ordinal))
            {
                _expectRestart = true;   // the close right behind this is planned, so retry at once
                Debug.Log("[SteamNet] server restarting, will reconnect");
                return;
            }

            var msg = NetProtocol.Parse(raw);
            switch (msg.Type)
            {
                case "SEAT":
                    if (msg.Payload == "FULL") { Seat = null; game.OnNetSeatFull(); }
                    else
                    {
                        int seat;
                        if (!int.TryParse(msg.Payload, NumberStyles.Integer, CultureInfo.InvariantCulture, out seat))
                        {
                            Debug.LogError("[SteamNet] unreadable seat: " + msg.Payload);
                            break;
                        }
                        Seat = (PlayerId)seat;
                        Connected = true;
                        if (_attempt > 0) { _attempt = 0; game.OnNetReconnected(); }
                        game.OnNetSeat(Seat.Value);
                    }
                    break;
                case "CATALOG?": Send(StartingCatalogMessage()); break;
                case "START":  game.OnNetStart(msg.Payload); break;
                case "APPLY":  game.OnNetApply(CommandWire.Read(msg.Payload)); break;
                case "REJECT": game.OnNetReject(msg.Payload); break;
            }
        }

        string StartingCatalogMessage()
        {
            var seat = Seat ?? PlayerId.Player0;
            var catalog = SessionBarracksCache.ForLocalPlayer((int)seat).Snapshot();
            return NetProtocol.Catalog(BarracksWire.Write(catalog));
        }

        void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            _ws?.DispatchMessageQueue();
#endif
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

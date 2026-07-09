using System;
using System.Collections;
using System.Text;
using UnityEngine;
using NativeWebSocket;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Browser↔server link for online play. Connects to the authoritative server's <c>/ws</c> endpoint,
    /// sends the local player's commands, and feeds server messages (seat, start state, validated moves,
    /// rejections) back into <see cref="GameBootstrap"/>. WebGL-safe: the socket queue is pumped from
    /// Update (only needed off-WebGL; on WebGL the jslib callbacks already run on the main thread).
    /// The server URL is derived from the page origin, with the room read from <c>?room=</c>.
    /// </summary>
    public sealed class NetClient : MonoBehaviour
    {
        WebSocket _ws;
        GameBootstrap _game;

        public PlayerId? Seat { get; private set; }
        public bool Connected { get; private set; }

        bool _closing;
        int _attempt;                 // 0 = first-ever attempt; >0 = a retry after a drop
        volatile bool _attemptClosed; // current attempt's socket session ended (OnClose fired, or connect failed)
        string _room, _setupWire;
        bool _isPrivate;

        const string TokenPrefKey = "HexWars.SeatToken";
        const string TokenAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ23456789";
        static readonly float[] BackoffSeconds = { 1f, 2f, 4f, 8f, 15f }; // caps at 15s, then repeats forever

        /// <summary>One token per browser, minted once and kept in PlayerPrefs. The server seats by
        /// token, not by socket (spec §3), so presenting the same token — after a refresh, a background
        /// tab drop, or this class's own reconnect loop — reclaims the same seat.</summary>
        public static string Token()
        {
            string t = PlayerPrefs.GetString(TokenPrefKey, "");
            if (!string.IsNullOrEmpty(t)) return t;
            var chars = new char[16];
            for (int i = 0; i < chars.Length; i++) chars[i] = TokenAlphabet[UnityEngine.Random.Range(0, TokenAlphabet.Length)];
            t = new string(chars);
            PlayerPrefs.SetString(TokenPrefKey, t);
            PlayerPrefs.Save();
            return t;
        }

        /// <summary>Start the connection lifecycle for a room. Remembers the args so a dropped socket
        /// can retry with the exact same request.</summary>
        public void Connect(GameBootstrap game, string room, string setupWire, bool isPrivate = false)
        {
            _game = game;
            _room = room;
            _setupWire = setupWire;
            _isPrivate = isPrivate;
            StartCoroutine(Lifecycle());
        }

        /// <summary>Owns every connection attempt for this component's lifetime. A drop BEFORE a game
        /// started (still on the host/join screen) is never retried — un-started rooms clean up
        /// instantly server-side, so there's nothing left to reconnect into; that case keeps today's
        /// toast-and-stop behavior via <see cref="GameBootstrap.OnNetClosed"/>. A drop AFTER a game
        /// started retries with capped exponential backoff (1s, 2s, 4s, 8s, cap 15s) indefinitely —
        /// until it reconnects or this component is destroyed (Cancel / Main menu both call
        /// <c>Destroy(_net)</c>, which stops this coroutine along with everything else). "This attempt
        /// is over" is signalled by the socket's OnClose EVENT (via <see cref="_attemptClosed"/>), never
        /// by Connect()'s Task — on WebGL that Task completes immediately (see <see cref="OpenOnce"/>),
        /// so a Task-driven loop would treat every attempt as instantly over on the deployed platform:
        /// spurious "Connection lost" on every host/join, and a new socket spun up per iteration while
        /// the previous ones were still live. Waiting on the event also guarantees the next attempt's
        /// socket is only constructed after the previous session actually ended — no stacking.</summary>
        IEnumerator Lifecycle()
        {
            while (true)
            {
                OpenOnce();
                while (!_attemptClosed) yield return null;
                if (_closing) yield break;

                if (_attempt == 0 && _game.State == null)
                {
                    _game.OnNetClosed();   // pre-start drop: existing toast + SetupForm status path, no retry
                    yield break;
                }

                _game.OnNetReconnecting(_attempt);
                float wait = BackoffSeconds[Mathf.Min(_attempt, BackoffSeconds.Length - 1)];
                _attempt++;
                yield return new WaitForSeconds(wait);
                if (_closing) yield break;
            }
        }

        /// <summary>One connection attempt, event-driven: the attempt ends when this socket's OnClose
        /// fires (sets <see cref="_attemptClosed"/>), or when construction/connect fails outright.
        /// NativeWebSocket's <c>Connect()</c> Task must NOT be treated as the attempt's lifetime: on
        /// standalone/editor it runs the whole read loop, but on WebGL it kicks the JS socket and
        /// returns <c>Task.CompletedTask</c> immediately. The OnOpen/OnClose events — fired by the
        /// JSLIB bridge on WebGL, and from Connect()'s own read loop/catch on standalone — are the only
        /// signals that mean the same thing on both platforms, and they are exactly what this class's
        /// pre-reconnect code relied on in the live WebGL deployment. Connect()'s Task is still observed
        /// for faults so a faulted connect counts as a closed attempt instead of vanishing as an
        /// unobserved exception (standalone Connect() routes all its errors to OnError/OnClose and never
        /// faults; WebGL's throws synchronously — both covered — so the continuation is a backstop).</summary>
        void OpenOnce()
        {
            _attemptClosed = false;
            string url = ServerWsUrl(_room, _setupWire, _isPrivate, Token());
            Debug.Log("[Net] connecting to " + url);
            try { _ws = new WebSocket(url); }
            catch (Exception e)
            {
                Debug.LogError("[Net] socket create failed: " + e.Message);
                _attemptClosed = true;   // an instantly-closed attempt: the normal backoff path applies
                return;
            }
            _ws.OnOpen += () =>
            {
                Connected = true;
                Debug.Log("[Net] open");
                if (_attempt > 0) { _attempt = 0; _game.OnNetReconnected(); }
            };
            _ws.OnError += e => Debug.LogError("[Net] error: " + e);
            _ws.OnClose += c => { Connected = false; Debug.Log("[Net] closed: " + c); _attemptClosed = true; };
            _ws.OnMessage += OnMessage;
            try
            {
                _ws.Connect().ContinueWith(t =>
                {
                    if (!t.IsFaulted) return;
                    Debug.LogError("[Net] connect faulted: " + t.Exception.GetBaseException().Message);
                    _attemptClosed = true;
                }, System.Threading.Tasks.TaskScheduler.Default);
            }
            catch (Exception e)
            {
                Debug.LogError("[Net] connect failed: " + e.Message);
                _attemptClosed = true;
            }
        }

        public async void Send(string message)
        {
            if (_ws != null && _ws.State == WebSocketState.Open)
                await _ws.SendText(message);
        }

        void OnMessage(byte[] data)
        {
            var msg = NetProtocol.Parse(Encoding.UTF8.GetString(data));
            switch (msg.Type)
            {
                case "SEAT":
                    if (msg.Payload == "FULL") { Seat = null; _game.OnNetSeatFull(); }
                    else { Seat = (PlayerId)int.Parse(msg.Payload); _game.OnNetSeat(Seat.Value); }
                    break;
                case "START":  _game.OnNetStart(msg.Payload); break;
                case "APPLY":  _game.OnNetApply(CommandWire.Read(msg.Payload)); break;
                case "REJECT": _game.OnNetReject(msg.Payload); break;
            }
        }

        void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            _ws?.DispatchMessageQueue();
#endif
        }

        async void OnDestroy()
        {
            _closing = true;             // deliberate teardown (Cancel / ReturnToMenu) — not an error
            StopAllCoroutines();         // Unity would stop Lifecycle() on destroy anyway; explicit for clarity
            if (_ws != null) await _ws.Close();
        }

        /// <summary>Build the WebSocket URL for a room from the page origin: https://host → wss://host/ws?room=…
        /// (&amp;setup=… for the host). Falls back to ws://127.0.0.1:5234 in the editor (no page URL).
        /// <paramref name="token"/> is always appended — the server seats by token, not connection id.</summary>
        static string ServerWsUrl(string room, string setupWire, bool isPrivate, string token)
        {
            string origin = "ws://127.0.0.1:5234"; // dev default when there's no page URL (editor)
            string page = Application.absoluteURL;
            if (!string.IsNullOrEmpty(page))
            {
                try
                {
                    var uri = new Uri(page);
                    origin = (uri.Scheme == "https" ? "wss" : "ws") + "://" + uri.Authority;
                }
                catch { /* keep dev default */ }
            }
            string url = origin + "/ws?room=" + Uri.EscapeDataString(room);
            if (!string.IsNullOrEmpty(setupWire)) url += "&setup=" + Uri.EscapeDataString(setupWire);
            else url += "&join=1"; // a joiner (link/code/browser row) never carries setup — flag it so a
                                    // missing room turns the connection away instead of minting a phantom game
            if (isPrivate) url += "&private=1";
            url += "&token=" + Uri.EscapeDataString(token);
            return url;
        }
    }
}

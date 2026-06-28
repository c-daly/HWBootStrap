using System;
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

        public async void Connect(GameBootstrap game)
        {
            _game = game;
            string url = ServerWsUrl();
            Debug.Log("[Net] connecting to " + url);
            _ws = new WebSocket(url);
            _ws.OnOpen += () => { Connected = true; Debug.Log("[Net] open"); };
            _ws.OnError += e => Debug.LogError("[Net] error: " + e);
            _ws.OnClose += c => { Connected = false; Debug.Log("[Net] closed: " + c); };
            _ws.OnMessage += OnMessage;
            await _ws.Connect();
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
            if (_ws != null) await _ws.Close();
        }

        /// <summary>https://host/?room=X → wss://host/ws?room=X (ws://127.0.0.1:5234 in the editor).</summary>
        static string ServerWsUrl()
        {
            string page = Application.absoluteURL;
            string room = "default";
            string origin = "ws://127.0.0.1:5234"; // dev default when there's no page URL (editor)

            if (!string.IsNullOrEmpty(page))
            {
                int q = page.IndexOf('?');
                if (q >= 0)
                    foreach (var kv in page.Substring(q + 1).Split('&'))
                    {
                        var p = kv.Split('=');
                        if (p.Length == 2 && p[0] == "room") room = Uri.UnescapeDataString(p[1]);
                    }
                try
                {
                    var uri = new Uri(page);
                    origin = (uri.Scheme == "https" ? "wss" : "ws") + "://" + uri.Authority;
                }
                catch { /* keep dev default */ }
            }
            return origin + "/ws?room=" + Uri.EscapeDataString(room);
        }
    }
}

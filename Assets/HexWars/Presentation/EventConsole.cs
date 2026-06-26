using System.Collections.Generic;
using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// On-screen overlay for AI matches: a live <b>scoreboard</b> (round, units + value per side, winner)
    /// plus a scrolling <b>event log</b> (kills, deploys, designs — from <see cref="CombatLog"/>). Drivers
    /// call <see cref="Report"/> on each state transition. Auto-created per play session (no scene wiring);
    /// drawn with OnGUI so it also updates while paused.
    /// </summary>
    public sealed class EventConsole : MonoBehaviour
    {
        const int MaxLines = 30;
        static EventConsole _inst;

        readonly Queue<string> _lines = new Queue<string>();
        GameState _state;
        bool _collapsed;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            var go = new GameObject("EventConsole");
            go.AddComponent<EventConsole>();
            DontDestroyOnLoad(go);
        }

        void Awake() => _inst = this;

        /// <summary>Update the scoreboard to <paramref name="cur"/> and append any event lines.</summary>
        public static void Report(GameState cur, IEnumerable<string> events)
        {
            if (_inst == null) return;
            _inst._state = cur;
            if (events != null)
                foreach (var line in events)
                {
                    _inst._lines.Enqueue(line);
                    while (_inst._lines.Count > MaxLines) _inst._lines.Dequeue();
                }
        }

        /// <summary>Reset for a new game (clears the log; scoreboard refreshes on the next Report).</summary>
        public static void Clear()
        {
            if (_inst == null) return;
            _inst._lines.Clear();
            _inst._state = null;
        }

        void OnGUI()
        {
            var e = Event.current;
            if (e != null && e.type == EventType.KeyDown && e.keyCode == KeyCode.BackQuote) { _collapsed = !_collapsed; e.Use(); }

            // OnGUI doesn't DPI-scale, so it's tiny on 4K — scale the whole sidebar by screen height
            // (≈2x at 2160p) and draw in 1080p-logical coordinates.
            float s = Mathf.Max(1f, Screen.height / 1080f);
            var prevMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(s, s, 1f));
            DrawSidebar(Screen.width / s, Screen.height / s);
            GUI.matrix = prevMatrix;
        }

        // Right-edge panel: a header scoreboard + a scrolling, color-coded narration of events.
        void DrawSidebar(float w, float h)
        {
            var btn = new GUIStyle(GUI.skin.button) { fontSize = 16 };

            // collapsed: just a small re-open tab top-right (also toggle with the ` key)
            if (_collapsed)
            {
                if (GUI.Button(new Rect(w - 96f, 6f, 90f, 30f), "◀ Log", btn)) _collapsed = false;
                return;
            }
            if (_state == null && _lines.Count == 0) return;

            const float pad = 12f, panelW = 430f;
            float x = w - panelW;

            var prevColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.78f);
            GUI.DrawTexture(new Rect(x, 0f, panelW, h), Texture2D.whiteTexture);
            GUI.color = prevColor;

            if (GUI.Button(new Rect(x + panelW - 36f, 6f, 30f, 28f), "▶", btn)) { _collapsed = true; return; }

            float y = pad;
            if (_state != null) y = DrawHeader(x + pad, y, panelW - 2f * pad - 34f); // leave room for the collapse button
            if (_lines.Count > 0) DrawLog(x + pad, y + 6f, panelW - 2f * pad, h - y - pad - 6f);
        }

        float DrawHeader(float x, float y, float w)
        {
            int u0 = AliveUnits(PlayerId.Player0), u1 = AliveUnits(PlayerId.Player1);
            int v0 = WinCheck.Evaluate(_state, PlayerId.Player0), v1 = WinCheck.Evaluate(_state, PlayerId.Player1);

            string result = "";
            if (_state.IsGameOver)
                result = _state.Winner == null ? "  ·  DRAW"
                       : (_state.Winner == PlayerId.Player0 ? "  ·  P1 WINS" : "  ·  P2 WINS");

            var h1 = new GUIStyle(GUI.skin.label) { fontSize = 26, fontStyle = FontStyle.Bold, richText = true };
            h1.normal.textColor = Color.white;
            GUI.Label(new Rect(x, y, w, 34f), $"Round {_state.Round}{result}", h1);

            var h2 = new GUIStyle(GUI.skin.label) { fontSize = 19, richText = true };
            h2.normal.textColor = new Color(0.9f, 0.92f, 0.95f);
            GUI.Label(new Rect(x, y + 36f, w, 26f),
                      $"<color=#6FB1FF>P1</color>  {u0}u · {v0}v       <color=#FF7B6B>P2</color>  {u1}u · {v1}v", h2);

            return y + 70f;
        }

        void DrawLog(float x, float y, float w, float h)
        {
            var style = new GUIStyle(GUI.skin.label)
            { fontSize = 16, alignment = TextAnchor.LowerLeft, richText = true, wordWrap = true };
            style.normal.textColor = new Color(0.92f, 0.93f, 0.96f);
            GUI.Label(new Rect(x, y, w, h), string.Join("\n", _lines), style);
        }

        int AliveUnits(PlayerId p)
        {
            int n = 0;
            foreach (var u in _state.Player(p).UnitsOnBoard) if (u.IsAlive) n++;
            return n;
        }
    }
}

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
        PlayerId? _viewer;
        bool _collapsed;

        // cached rendering data, rebuilt only when Report/Clear touch it — never per-OnGUI-frame (OnGUI
        // runs every frame, including while paused, so this was audit F1/U1's allocation source)
        GUIStyle _btnStyle, _h1Style, _h2Style, _h3Style, _logStyle;
        string _joinedLog = "";
        string _headerRound = "", _headerArmies = "", _headerSettings = "";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            var go = new GameObject("EventConsole");
            go.AddComponent<EventConsole>();
            DontDestroyOnLoad(go);
        }

        void Awake() => _inst = this;

        /// <summary>Update the scoreboard to <paramref name="cur"/> and append any event lines. The
        /// header strings and the joined log text are rebuilt HERE (on a state/event change) instead of
        /// every OnGUI frame — Report fires on game turns/actions, not 60 times a second.</summary>
        public static void Report(GameState cur, IEnumerable<string> events, PlayerId? viewer = null)
        {
            if (_inst == null) return;
            _inst._state = cur;
            _inst._viewer = viewer;
            if (events != null)
            {
                bool changed = false;
                foreach (var line in events)
                {
                    _inst._lines.Enqueue(line);
                    changed = true;
                    while (_inst._lines.Count > MaxLines) _inst._lines.Dequeue();
                }
                if (changed) _inst._joinedLog = string.Join("\n", _inst._lines);
            }
            if (cur != null) _inst.RebuildHeader();
        }

        /// <summary>Reset for a new game (clears the log; scoreboard refreshes on the next Report).</summary>
        public static void Clear()
        {
            if (_inst == null) return;
            _inst._lines.Clear();
            _inst._state = null;
            _inst._viewer = null;
            _inst._joinedLog = "";
            _inst._headerRound = _inst._headerArmies = _inst._headerSettings = "";
        }

        void RebuildHeader()
        {
            string result = "";
            if (_state.IsGameOver)
                result = _state.Winner == null ? "  ·  DRAW"
                       : (_state.Winner == PlayerId.Player0 ? "  ·  P1 WINS" : "  ·  P2 WINS");

            _headerRound = $"Round {_state.Round}{result}";
            _headerArmies = FormatArmySummary(_state, _viewer);
            string mode = _state.Config.TerritoryMode ? "Territory" : "Annihilation";
            _headerSettings = $"{mode} · {_state.Board.Tiles.Count} tiles · {_state.Config.StartingPoints} start pts";
        }

        void OnGUI()
        {
            // portrait: at 1080p-logical scale the 430-wide sidebar covers ~86% of a 390px-wide phone
            // screen (audit F1/U1) — the combat log stays a desktop/landscape feature, per spec §4.
            if (Screen.width < Screen.height) return;

            var e = Event.current;
            if (e != null && e.type == EventType.KeyDown && e.keyCode == KeyCode.BackQuote) { _collapsed = !_collapsed; e.Use(); }

            EnsureStyles();

            // OnGUI doesn't DPI-scale, so it's tiny on 4K — scale the whole sidebar by screen height
            // (≈2x at 2160p) and draw in 1080p-logical coordinates.
            float s = Mathf.Max(1f, Screen.height / 1080f);
            var prevMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(s, s, 1f));
            DrawSidebar(Screen.width / s, Screen.height / s);
            GUI.matrix = prevMatrix;
        }

        /// <summary>GUIStyle construction reads GUI.skin, which is only valid inside OnGUI — built once,
        /// the first time OnGUI actually runs (audit F1/U1: these were rebuilt every frame before).</summary>
        void EnsureStyles()
        {
            if (_btnStyle != null) return;
            _btnStyle = new GUIStyle(GUI.skin.button) { fontSize = 16 };
            _h1Style = new GUIStyle(GUI.skin.label) { fontSize = 26, fontStyle = FontStyle.Bold, richText = true };
            _h1Style.normal.textColor = Color.white;
            _h2Style = new GUIStyle(GUI.skin.label) { fontSize = 19, richText = true };
            _h2Style.normal.textColor = new Color(0.9f, 0.92f, 0.95f);
            _h3Style = new GUIStyle(GUI.skin.label) { fontSize = 15, richText = true };
            _h3Style.normal.textColor = new Color(0.62f, 0.66f, 0.74f);
            _logStyle = new GUIStyle(GUI.skin.label)
                { fontSize = 16, alignment = TextAnchor.LowerLeft, richText = true, wordWrap = true };
            _logStyle.normal.textColor = new Color(0.92f, 0.93f, 0.96f);
        }

        // Right-edge panel: a header scoreboard + a scrolling, color-coded narration of events.
        void DrawSidebar(float w, float h)
        {
            // collapsed: just a small re-open tab top-right (also toggle with the ` key)
            if (_collapsed)
            {
                if (GUI.Button(new Rect(w - 96f, 6f, 90f, 30f), "◀ Log", _btnStyle)) _collapsed = false;
                return;
            }
            if (_state == null && _lines.Count == 0) return;

            const float pad = 12f, panelW = 430f;
            float x = w - panelW;

            var prevColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.78f);
            GUI.DrawTexture(new Rect(x, 0f, panelW, h), Texture2D.whiteTexture);
            GUI.color = prevColor;

            if (GUI.Button(new Rect(x + panelW - 36f, 6f, 30f, 28f), "▶", _btnStyle)) { _collapsed = true; return; }

            float y = pad;
            if (_state != null) y = DrawHeader(x + pad, y, panelW - 2f * pad - 34f); // leave room for the collapse button
            if (_lines.Count > 0) DrawLog(x + pad, y + 6f, panelW - 2f * pad, h - y - pad - 6f);
        }

        float DrawHeader(float x, float y, float w)
        {
            GUI.Label(new Rect(x, y, w, 34f), _headerRound, _h1Style);
            GUI.Label(new Rect(x, y + 36f, w, 26f), _headerArmies, _h2Style);
            GUI.Label(new Rect(x, y + 64f, w, 22f), _headerSettings, _h3Style);
            return y + 90f;
        }

        void DrawLog(float x, float y, float w, float h)
        {
            GUI.Label(new Rect(x, y, w, h), _joinedLog, _logStyle);
        }

        public static string FormatArmySummary(GameState state, PlayerId? viewer = null)
        {
            string Totals(PlayerId player)
            {
                if (viewer.HasValue && state.Config.FogOfWar && viewer.Value != player)
                    return "?u · ?v";
                int units = 0;
                foreach (var unit in state.Player(player).UnitsOnBoard) if (unit.IsAlive) units++;
                return $"{units}u · {WinCheck.Evaluate(state, player)}v";
            }
            return $"<color=#6FB1FF>P1</color>  {Totals(PlayerId.Player0)}       " +
                   $"<color=#FF7B6B>P2</color>  {Totals(PlayerId.Player1)}";
        }
    }
}

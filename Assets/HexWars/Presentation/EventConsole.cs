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
        const int MaxLines = 14;
        static EventConsole _inst;

        readonly Queue<string> _lines = new Queue<string>();
        GameState _state;

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
            if (_state != null) DrawScoreboard();
            if (_lines.Count > 0) DrawLog();
        }

        void DrawScoreboard()
        {
            int u0 = AliveUnits(PlayerId.Player0), u1 = AliveUnits(PlayerId.Player1);
            int v0 = WinCheck.Evaluate(_state, PlayerId.Player0), v1 = WinCheck.Evaluate(_state, PlayerId.Player1);

            string result = "";
            if (_state.IsGameOver)
                result = _state.Winner == null ? "  —  DRAW"
                       : (_state.Winner == PlayerId.Player0 ? "  —  P1 WINS" : "  —  P2 WINS");

            string text = $"Round {_state.Round}{result}\n"
                        + $"P1:  {u0} units · {v0} val          P2:  {u1} units · {v1} val";

            var style = new GUIStyle(GUI.skin.label)
            {
                fontSize = 16,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.UpperCenter,
            };
            style.normal.textColor = Color.white;

            var rect = new Rect(Screen.width / 2f - 280f, 6f, 560f, 52f);
            GUI.Box(rect, GUIContent.none);
            GUI.Label(rect, text, style);
        }

        void DrawLog()
        {
            var style = new GUIStyle(GUI.skin.label) { fontSize = 13, alignment = TextAnchor.LowerLeft, richText = false };
            style.normal.textColor = new Color(0.9f, 0.92f, 0.95f);

            var rect = new Rect(8f, Screen.height - 232f, 380f, 224f);
            GUI.Box(rect, GUIContent.none);
            GUI.Label(new Rect(rect.x + 6f, rect.y + 4f, rect.width - 12f, rect.height - 8f),
                      string.Join("\n", _lines), style);
        }

        int AliveUnits(PlayerId p)
        {
            int n = 0;
            foreach (var u in _state.Player(p).UnitsOnBoard) if (u.IsAlive) n++;
            return n;
        }
    }
}

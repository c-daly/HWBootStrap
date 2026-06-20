using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Plays the live game with two engine agents (no human) so you can just watch them fight — Player 1
    /// is a <see cref="GreedyAgent"/>, Player 2 a <see cref="RandomAgent"/>. Steps one command every
    /// <see cref="SecondsPerAction"/> via the bootstrap's TryApply, so the board, HP bars, and turn
    /// banner update through the normal rendering path. Human input stays in read-only mode, so you can
    /// still hover units for tooltips and click them to inspect — you just can't issue commands.
    /// </summary>
    public sealed class SpectatorDriver : MonoBehaviour
    {
        public float SecondsPerAction = 0.35f;

        GameBootstrap _game;
        IAgent _p0, _p1;
        float _timer;

        void Start()
        {
            _game = FindAnyObjectByType<GameBootstrap>();
            _p0 = new GreedyAgent(1);
            _p1 = new RandomAgent(2);

            var input = FindAnyObjectByType<UnitInputController>();
            if (input != null) input.ReadOnly = true; // keep hover + click-to-inspect, just block human commands

            var barracks = FindAnyObjectByType<BarracksPanel>();
            if (barracks != null) barracks.ReadOnly = true; // show barracks, but no human deploys
        }

        void Update()
        {
            if (_game == null || _game.State == null || _game.State.IsGameOver) return;

            _timer += Time.deltaTime;
            if (_timer < SecondsPerAction) return;
            _timer = 0f;

            var s = _game.State;
            var agent = s.ActivePlayer == PlayerId.Player0 ? _p0 : _p1;
            _game.TryApply(agent.Decide(s));
        }

#if UNITY_EDITOR
        // Attaches itself when the "Watch AI vs AI" menu set the flag, surviving the play-mode domain
        // reload via EditorPrefs (so the saved scene is never modified).
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoAttachInEditor()
        {
            if (!UnityEditor.EditorPrefs.GetBool("HexWars.Spectate", false)) return;
            UnityEditor.EditorPrefs.SetBool("HexWars.Spectate", false);

            var game = FindAnyObjectByType<GameBootstrap>();
            if (game != null && game.GetComponent<SpectatorDriver>() == null)
                game.gameObject.AddComponent<SpectatorDriver>();
        }
#endif
    }
}

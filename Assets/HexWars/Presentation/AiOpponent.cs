using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>Difficulty of the built-in AI opponent. Easy = Random, Hard = Greedy (the audit shows
    /// greedy plays a real, decisive game). Pure C# agents, so they ship in a standalone build; a
    /// model-backed "Expert" tier could be added later via the policy bridge.</summary>
    public enum AiLevel { Easy, Hard }

    /// <summary>
    /// A single AI-controlled seat in the playable game, so a human can challenge the computer. On the AI
    /// seat's turn it steps a scripted agent through the normal <see cref="GameBootstrap.TryApply"/> path
    /// (board, HP bars, turn banner, event sidebar all update as usual); on the human's turn it hands
    /// control back. Human input is gated to the human's turn by toggling the input/barracks ReadOnly flags.
    /// </summary>
    public sealed class AiOpponent : MonoBehaviour
    {
        public PlayerId AiSeat = PlayerId.Player1; // the human plays the other seat
        public AiLevel Level = AiLevel.Hard;
        public float SecondsPerAction = 0.35f;

        GameBootstrap _game;
        IAgent _agent;
        UnitInputController _input;
        BarracksPanel _barracks;
        float _timer;

        void Start()
        {
            _game = FindAnyObjectByType<GameBootstrap>();
            _agent = Level == AiLevel.Hard ? new GreedyAgent(7) : (IAgent)new RandomAgent(7);
            _input = FindAnyObjectByType<UnitInputController>();
            _barracks = FindAnyObjectByType<BarracksPanel>();
        }

        void Update()
        {
            if (_game == null || _game.State == null) return;
            var s = _game.State;
            bool aiTurn = !s.IsGameOver && s.ActivePlayer == AiSeat;

            // the human can only issue commands on their own turn
            if (_input != null) _input.ReadOnly = aiTurn;
            if (_barracks != null) _barracks.ReadOnly = aiTurn;
            if (!aiTurn) return;

            _timer += Time.deltaTime;
            if (_timer < SecondsPerAction) return;
            _timer = 0f;
            _game.TryApply(_agent.Decide(s));
        }

#if UNITY_EDITOR
        // The "HexWars > Play vs AI" menu sets these prefs, then this attaches on play — so the saved scene
        // is never modified (mirrors SpectatorDriver). In a build, use GameBootstrap's VsAI fields instead.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoAttachInEditor()
        {
            if (!UnityEditor.EditorPrefs.GetBool("HexWars.VsAI", false)) return;
            UnityEditor.EditorPrefs.SetBool("HexWars.VsAI", false);

            var game = FindAnyObjectByType<GameBootstrap>();
            if (game == null) return;
            var ai = game.GetComponent<AiOpponent>() ?? game.gameObject.AddComponent<AiOpponent>();
            ai.Level = (AiLevel)UnityEditor.EditorPrefs.GetInt("HexWars.AiLevel", (int)AiLevel.Hard);
        }
#endif
    }
}

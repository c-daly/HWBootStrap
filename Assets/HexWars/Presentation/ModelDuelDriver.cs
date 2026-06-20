using UnityEngine;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Presentation
{
    /// <summary>
    /// Watches a tactical duel where one or both seats are <b>trained models</b>, driven via the
    /// <see cref="PolicyBridge"/> (Windows-venv policy_server.py). Uses the engine's <see cref="DuelEnv"/>
    /// so the observation/action/mask are exactly what the models trained on. Model seats are "external"
    /// (Unity supplies their actions from the bridge); a greedy/random seat is auto-played inside DuelEnv.
    /// Configure the fields (the editor menu does this), then it runs on Start.
    /// </summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class ModelDuelDriver : MonoBehaviour
    {
        public string PythonExe;     // ...\python\winenv\Scripts\python.exe
        public string ServerScript;  // ...\python\policy_server.py
        public string WorkingDir;    // ...\python
        public string P0Spec = "ppo:sp6base.zip"; // "ppo:FILE"/"dqn:FILE" (model) or "greedy"/"random"
        public string P1Spec = "greedy";
        public int Seed = 0;
        public float SecondsPerAction = 0.4f;

        BoardRenderer _board;
        DuelEnv _duel;
        PolicyBridge _bridge;
        DuelEnv.View _view;
        bool _p0Model, _p1Model, _done;
        float _timer;

        void Start()
        {
            _board = GetComponent<BoardRenderer>();
            _p0Model = IsModel(P0Spec);
            _p1Model = IsModel(P1Spec);

            if (_p0Model || _p1Model)
            {
                _bridge = new PolicyBridge();
                bool ok = _bridge.Start(PythonExe, ServerScript,
                                        _p0Model ? P0Spec : null, _p1Model ? P1Spec : null, WorkingDir);
                if (!ok) { Debug.LogError("ModelDuelDriver: policy bridge failed to start."); _done = true; return; }
            }

            // model seats are external (null controller -> we feed actions); scripted seats run inside DuelEnv
            IAgent c0 = _p0Model ? null : Scripted(P0Spec, Seed * 2 + 1);
            IAgent c1 = _p1Model ? null : Scripted(P1Spec, Seed * 2 + 2);

            _duel = new DuelEnv();
            _view = _duel.Reset(Seed, c0, c1, PlayerId.Player0);
            _board.Render(_duel.State.Board);
            _board.RenderEntities(_duel.State);

            var input = FindAnyObjectByType<UnitInputController>();
            if (input != null) input.ReadOnly = true; // hover + inspect, no commands

            Debug.Log($"ModelDuelDriver: P0={P0Spec} vs P1={P1Spec}, seed {Seed}");
        }

        void Update()
        {
            if (_done || _duel == null) return;
            if (_view.Terminated || _view.Truncated)
            {
                if (!_done) Debug.Log($"Duel over: winner={(_view.Winner < 0 ? "DRAW" : "P" + _view.Winner)}");
                _done = true;
                return;
            }

            _timer += Time.deltaTime;
            if (_timer < SecondsPerAction) return;
            _timer = 0f;

            int seat = _view.Seat;
            bool seatIsModel = seat == 0 ? _p0Model : _p1Model;
            if (!seatIsModel) { _done = true; return; } // scripted seats auto-advance inside Step; shouldn't land here

            try
            {
                int action = _bridge.Act(seat, _view.Observation, _view.ActionMask);
                _view = _duel.Step(action);
                _board.RenderEntities(_duel.State);
            }
            catch (System.Exception e)
            {
                Debug.LogError("ModelDuelDriver: bridge error, stopping. " + e.Message);
                _done = true;
            }
        }

        void OnDestroy() => _bridge?.Dispose();

        static bool IsModel(string spec) => spec != null && (spec.StartsWith("ppo:") || spec.StartsWith("dqn:"));
        static IAgent Scripted(string spec, int seed) => spec == "random" ? new RandomAgent(seed) : (IAgent)new GreedyAgent(seed);
    }
}

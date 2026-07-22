using System;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Presentation
{
    public enum ModelControllerKind { Greedy, Random, FixedCheckpoint, FixedRun, LiveRun }
    public enum MlModelAlgorithm { MaskablePpo, MaskedDqn }

    [Serializable]
    public sealed class ModelSeatConfiguration
    {
        public ModelControllerKind Kind = ModelControllerKind.Greedy;
        public string Path = string.Empty;
        public MlModelAlgorithm Algorithm = MlModelAlgorithm.MaskablePpo;

        public string BuildSpec()
        {
            switch (Kind)
            {
                case ModelControllerKind.Random: return "random";
                case ModelControllerKind.FixedCheckpoint:
                    return (Algorithm == MlModelAlgorithm.MaskedDqn ? "dqn:" : "ppo:") + Path;
                case ModelControllerKind.FixedRun: return "run:" + Path;
                case ModelControllerKind.LiveRun:
                    return JsonUtility.ToJson(new RunSpec { kind = "run", path = Path, mode = "live" });
                default: return "greedy";
            }
        }

        public bool IsModel => Kind != ModelControllerKind.Greedy && Kind != ModelControllerKind.Random;
        public bool IsLive => Kind == ModelControllerKind.LiveRun;

        public string ValidationError(string label)
        {
            if (IsModel && string.IsNullOrWhiteSpace(Path)) return label + " requires a model or run path.";
            return string.Empty;
        }

        [Serializable] sealed class RunSpec { public string kind; public string path; public string mode; }
    }

    [Serializable]
    public sealed class ModelDuelConfiguration
    {
        public ModelSeatConfiguration P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.LiveRun };
        public ModelSeatConfiguration P1 = new ModelSeatConfiguration { Kind = ModelControllerKind.Greedy };
        public int Seed;
        public float SecondsPerAction = 0.4f;
        public bool Loop = true;

        public IReadOnlyList<string> Validate()
        {
            var errors = new List<string>();
            string p0 = P0?.ValidationError("Seat 0") ?? "Seat 0 configuration is missing.";
            string p1 = P1?.ValidationError("Seat 1") ?? "Seat 1 configuration is missing.";
            if (!string.IsNullOrEmpty(p0)) errors.Add(p0);
            if (!string.IsNullOrEmpty(p1)) errors.Add(p1);
            if (SecondsPerAction <= 0) errors.Add("Action pacing must be greater than zero.");
            return errors;
        }

        public bool ShouldReload(bool gameEnded) => gameEnded &&
            ((P0 != null && P0.IsLive) || (P1 != null && P1.IsLive));
    }

    [RequireComponent(typeof(BoardRenderer))]
    [RequireComponent(typeof(ModelArenaIdentityOverlay))]
    public sealed class ModelDuelDriver : MonoBehaviour
    {
        public string PythonExe;
        public string ServerScript;
        public string WorkingDir;
        public string P0Spec = "greedy";
        public string P1Spec = "greedy";
        public int Seed;
        public float SecondsPerAction = 0.4f;
        public bool Loop;
        public float SecondsBetweenGames = 1.5f;

        public bool Paused { get; private set; }
        public bool IsDone => _done;
        public int CurrentSeat => _view.Seat;
        public int GamesPlayed { get; private set; }
        public int P0Wins { get; private set; }
        public int P1Wins { get; private set; }
        public int Draws { get; private set; }
        public PolicySeatInfo P0Resolved => _bridge?.Seat0;
        public PolicySeatInfo P1Resolved => _bridge?.Seat1;

        public ModelArenaSeatIdentity[] IdentitySnapshot() => ModelArenaIdentity.Build(
            P0Spec, P1Spec, P0Resolved, P1Resolved, CurrentSeat,
            P0Wins, P1Wins, Draws);

        BoardRenderer _board;
        DuelEnv _duel;
        PolicyBridge _bridge;
        DuelEnv.View _view;
        bool _p0Model, _p1Model, _p0Live, _p1Live, _done, _ended;
        float _timer, _restTimer;
        CancellationTokenSource _startupCancellation;
        public bool IsStarting { get; private set; }

        async void Start()
        {
            _board = GetComponent<BoardRenderer>();
            _p0Model = IsModel(P0Spec);
            _p1Model = IsModel(P1Spec);
            _p0Live = IsLiveRun(P0Spec);
            _p1Live = IsLiveRun(P1Spec);
            if (_p0Model || _p1Model)
            {
                _bridge = new PolicyBridge();
                _startupCancellation = new CancellationTokenSource();
                IsStarting = true;
                bool ok = await _bridge.StartAsync(PythonExe, ServerScript,
                    _p0Model ? P0Spec : null, _p1Model ? P1Spec : null, WorkingDir,
                    PolicyBridge.DefaultStartupTimeoutMs, _startupCancellation.Token);
                IsStarting = false;
                if (_done || this == null) return;
                if (!ok) { Debug.LogError("ModelDuelDriver: policy bridge failed to start."); _done = true; return; }
            }
            var input = FindAnyObjectByType<UnitInputController>();
            if (input != null) input.ReadOnly = true;
            Debug.Log($"ModelDuelDriver: P0={P0Spec} vs P1={P1Spec}, loop={Loop}");
            BeginGame();
        }

        void BeginGame()
        {
            IAgent c0 = _p0Model ? null : Scripted(P0Spec, Seed * 2 + 1);
            IAgent c1 = _p1Model ? null : Scripted(P1Spec, Seed * 2 + 2);
            _duel = new DuelEnv();
            _view = _duel.Reset(Seed, c0, c1, PlayerId.Player0);
            _board.Render(_duel.State.Board);
            _board.RenderEntities(_duel.State);
            EventConsole.Clear();
            EventConsole.Report(_duel.State, null);
            _timer = 0;
        }

        void Update()
        {
            if (_done || Paused || _duel == null) return;
            if (_view.Terminated || _view.Truncated)
            {
                if (!_ended)
                {
                    RecordResult();
                    _ended = true;
                    _restTimer = 0f;
                    if (!Loop) { _done = true; return; }
                }
                _restTimer += Time.deltaTime;
                if (_restTimer < SecondsBetweenGames) return;
                if ((_p0Live || _p1Live) && _bridge != null)
                {
                    var reload = _bridge.Reload();
                    if (!string.IsNullOrWhiteSpace(reload.Error))
                    {
                        Debug.LogError("ModelDuelDriver: live reload failed. " + reload.Error);
                        _done = true;
                        return;
                    }
                }
                Seed++;
                _ended = false;
                BeginGame();
                return;
            }

            _timer += Time.deltaTime;
            if (_timer < SecondsPerAction) return;
            _timer = 0f;
            int seat = _view.Seat;
            bool seatIsModel = seat == 0 ? _p0Model : _p1Model;
            if (!seatIsModel) { _done = true; return; }
            try
            {
                int action = _bridge.Act(seat, _view.Observation, _view.ActionMask);
                var prev = _duel.State;
                _view = _duel.Step(action);
                _board.RenderEntities(_duel.State);
                EventConsole.Report(_duel.State, CombatLog.Diff(prev, _duel.State));
            }
            catch (Exception error)
            {
                Debug.LogError("ModelDuelDriver: bridge error, stopping. " + error.Message);
                _done = true;
            }
        }

        void RecordResult()
        {
            GamesPlayed++;
            if (_view.Winner == 0) P0Wins++;
            else if (_view.Winner == 1) P1Wins++;
            else Draws++;
            Debug.Log($"Game over: winner={(_view.Winner < 0 ? "DRAW" : "P" + _view.Winner)}; " +
                      $"tally={P0Wins}-{P1Wins}-{Draws}");
        }

        public void SetPaused(bool paused) => Paused = paused;
        public void StopDuel()
        {
            _done = true;
            _startupCancellation?.Cancel();
            _startupCancellation?.Dispose();
            _startupCancellation = null;
            IsStarting = false;
            _bridge?.Dispose();
            _bridge = null;
        }
        void OnDestroy() => StopDuel();

        public static bool IsModel(string spec) => !string.IsNullOrWhiteSpace(spec) && spec != "greedy" && spec != "random";
        public static bool IsLiveRun(string spec, Func<string, bool> directoryExists = null)
        {
            if (string.IsNullOrWhiteSpace(spec)) return false;
            string trimmed = spec.TrimStart();
            if (trimmed.StartsWith("ppo:", StringComparison.Ordinal) || trimmed.StartsWith("dqn:", StringComparison.Ordinal))
                return (directoryExists ?? System.IO.Directory.Exists)(trimmed.Substring(4));
            if (!trimmed.StartsWith("{", StringComparison.Ordinal)) return false;
            try { return JsonUtility.FromJson<LiveSpec>(trimmed)?.mode == "live"; }
            catch (Exception) { return false; }
        }

        static IAgent Scripted(string spec, int seed) => spec == "random" ? new RandomAgent(seed) : (IAgent)new GreedyAgent(seed);
        [Serializable] sealed class LiveSpec { public string mode; }
    }
}

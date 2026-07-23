using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Presentation
{
    public enum ModelControllerKind { Greedy, Random, FixedRun, LiveRun }
    public enum ModelInferenceMode { Deterministic, Stochastic }
    public enum ModelDuelObserverSeat { Player1, Player2 }

    public static class ModelDuelObserver
    {
        public static PlayerId Resolve(ModelDuelObserverSeat observer) =>
            observer == ModelDuelObserverSeat.Player2 ? PlayerId.Player1 : PlayerId.Player0;
    }

    [Serializable]
    public sealed class MlPresentationGame
    {
        public MlPresentationGame(
            string p0Spec,
            string p1Spec,
            int learnerSeat,
            ModelDuelObserverSeat observer,
            string opponentLabel,
            TrainingScenario scenario)
        {
            P0Spec = p0Spec;
            P1Spec = p1Spec;
            LearnerSeat = learnerSeat;
            Observer = observer;
            OpponentLabel = opponentLabel;
            Scenario = scenario;
        }

        public string P0Spec;
        public string P1Spec;
        public int LearnerSeat;
        public ModelDuelObserverSeat Observer;
        public string OpponentLabel;
        public TrainingScenario Scenario;
    }

    [Serializable]
    public sealed class MlPresentationSchedule
    {
        public MlPresentationGame[] Games = Array.Empty<MlPresentationGame>();

        public MlPresentationGame NextPresentationGame(int gamesPlayed)
        {
            if (gamesPlayed < 0)
                throw new ArgumentOutOfRangeException(nameof(gamesPlayed));
            if (Games == null || Games.Length == 0)
                throw new InvalidOperationException(
                    "presentation schedule must contain at least one game");
            MlPresentationGame game = Games[gamesPlayed % Games.Length];
            return game ?? throw new InvalidOperationException(
                "presentation schedule contains an empty game");
        }
    }

    [Serializable]
    public sealed class ModelSeatConfiguration
    {
        public ModelControllerKind Kind = ModelControllerKind.Greedy;
        public string Path = string.Empty;
        public ModelInferenceMode InferenceMode = ModelInferenceMode.Deterministic;

        public string BuildSpec()
        {
            switch (Kind)
            {
                case ModelControllerKind.Random: return "random";
                case ModelControllerKind.FixedRun: return "run:" + Path;
                case ModelControllerKind.LiveRun:
                    return JsonUtility.ToJson(new RunSpec
                    {
                        kind = "run",
                        path = Path,
                        mode = "live",
                        inference_mode = InferenceValue(InferenceMode),
                    });
                default: return "greedy";
            }
        }

        static string InferenceValue(ModelInferenceMode value) =>
            value == ModelInferenceMode.Stochastic ? "stochastic" : "deterministic";

        public bool IsModel => Kind != ModelControllerKind.Greedy && Kind != ModelControllerKind.Random;
        public bool IsLive => Kind == ModelControllerKind.LiveRun;

        public string ValidationError(string label)
        {
            if (IsModel && string.IsNullOrWhiteSpace(Path)) return label + " requires a model or run path.";
            return string.Empty;
        }

        [Serializable] sealed class RunSpec
        {
            public string kind;
            public string path;
            public string mode;
            public string inference_mode;
        }
    }

    [Serializable]
    public sealed class ModelDuelConfiguration
    {
        public ModelSeatConfiguration P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.LiveRun };
        public ModelSeatConfiguration P1 = new ModelSeatConfiguration { Kind = ModelControllerKind.Greedy };
        public MlEnvironmentContract Environment = MlEnvironmentContract.TacticalV1;
        public string ScenarioRunPath = string.Empty;
        public ModelDuelObserverSeat Observer = ModelDuelObserverSeat.Player1;
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
        public MlEnvironmentContract Environment = MlEnvironmentContract.TacticalV1;
        public ModelDuelObserverSeat Observer = ModelDuelObserverSeat.Player1;
        public TrainingScenario Scenario;
        public MlPresentationSchedule PresentationPlan;
        public int Seed;
        public float SecondsPerAction = 0.4f;
        public bool Loop;
        public float SecondsBetweenGames = 1.5f;

        public bool Paused { get; private set; }
        public bool IsDone => _done;
        public int CurrentSeat => _duel == null ? -1 : _view.Seat;
        public int GamesPlayed { get; private set; }
        public int P0Wins { get; private set; }
        public int P1Wins { get; private set; }
        public int Draws { get; private set; }
        public int LearnerWins { get; private set; }
        public int LearnerLosses { get; private set; }
        public int LearnerDraws { get; private set; }
        public PlayerId ObserverPlayer => ModelDuelObserver.Resolve(Observer);
        public bool ShouldShowArenaOverlays => Environment == MlEnvironmentContract.TacticalV1
            || (_duel != null && _view.DeploymentComplete);
        public PolicySeatInfo P0Resolved => _bridge?.Seat0;
        public PolicySeatInfo P1Resolved => _bridge?.Seat1;
        public string P0ArenaStatus { get; private set; }
        public string P1ArenaStatus { get; private set; }
        public int CurrentLearnerSeat => _activePresentationGame?.LearnerSeat ?? -1;

        public ModelArenaSeatIdentity[] IdentitySnapshot() => ModelArenaIdentity.Build(
            P0Spec, P1Spec, P0Resolved, P1Resolved, CurrentSeat,
            P0Wins, P1Wins, Draws, P0ArenaStatus, P1ArenaStatus,
            CurrentLearnerSeat, LearnerWins, LearnerLosses, LearnerDraws);

        public MlPresentationGame NextPresentationGame(int gamesPlayed) =>
            PresentationPlan?.NextPresentationGame(gamesPlayed);

        public static bool ShouldReconfigure(
            MlPresentationGame previous,
            MlPresentationGame next,
            bool gameEnded) =>
            gameEnded && previous != null && next != null &&
            (!string.Equals(previous.P0Spec, next.P0Spec, StringComparison.Ordinal) ||
             !string.Equals(previous.P1Spec, next.P1Spec, StringComparison.Ordinal));

        BoardRenderer _board;
        IModelDuelEnvironment _duel;
        TrainingScenario _activeScenario;
        PolicyBridge _bridge;
        ModelDuelContractIdentity _contractIdentity;
        ModelDuelView _view;
        ModelDuelPresentationState _presentation;
        bool _p0Model, _p1Model, _p0Live, _p1Live, _done, _ended;
        float _timer, _restTimer;
        CancellationTokenSource _startupCancellation;
        MlPresentationGame _activePresentationGame;
        public bool IsStarting { get; private set; }

        async void Start()
        {
            _board = GetComponent<BoardRenderer>();
            try
            {
                _activePresentationGame = NextPresentationGame(0);
                if (_activePresentationGame != null)
                    ApplyPresentationGame(_activePresentationGame);
                RefreshControllerFlags();
                _activeScenario = ResolveScenario();
                _contractIdentity =
                    ModelDuelEnvironmentFactory.ContractIdentity(_activeScenario);
            }
            catch (Exception error)
            {
                Debug.LogError("ModelDuelDriver: invalid scenario. " + error.Message);
                _done = true;
                return;
            }
            if (!await StartPolicyBridge("load failed")) return;
            var input = FindAnyObjectByType<UnitInputController>();
            if (input != null) input.ReadOnly = true;
            Debug.Log($"ModelDuelDriver: P0={P0Spec} vs P1={P1Spec}, loop={Loop}");
            BeginGame();
        }

        public TrainingScenario ResolveScenario()
        {
            TrainingScenario scenario = Scenario ??
                TrainingScenario.CreateStandard(
                    MlEnvironmentContracts.CliValue(Environment));
            if (!string.Equals(
                scenario.Environment,
                MlEnvironmentContracts.CliValue(Environment),
                StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "arena scenario environment " + scenario.Environment +
                    " does not match selected environment " +
                    MlEnvironmentContracts.CliValue(Environment) + ".");
            // Unity's by-value serializer materializes absent nested reference fields.
            // Restore the schema-v1 environment invariant before validating the reload.
            if (scenario.Environment == MlContract.CurrentVersion)
            {
                scenario.AdaptiveReward = null;
                scenario.Adaptive = null;
            }
            else if (scenario.Environment == MlContract.AdaptiveVersion)
            {
                scenario.TacticalReward = null;
            }
            IReadOnlyList<string> errors = scenario.Validate();
            if (errors.Count > 0)
                throw new InvalidOperationException(
                    "arena scenario is invalid after serialization: " +
                    string.Join("; ", errors));
            return scenario;
        }

        void BeginGame()
        {
            IAgent c0 = _p0Model ? null : Scripted(P0Spec, Seed * 2 + 1);
            IAgent c1 = _p1Model ? null : Scripted(P1Spec, Seed * 2 + 2);
            _duel = ModelDuelEnvironmentFactory.Create(_activeScenario);
            _presentation = new ModelDuelPresentationState(Environment);
            _view = _duel.Reset(Seed, c0, c1);
            Present(previous: null);
            _timer = 0;
        }

        void Update()
        {
            if (_done || Paused || _duel == null) return;
            if (_duel.RequiresContinuation)
            {
                var previous = _duel.CurrentState;
                _view = _duel.Continue();
                Present(previous);
                return;
            }
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
                AdvanceAtGameBoundary();
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
                var prev = _duel.CurrentState;
                _view = _duel.Step(action);
                Present(prev);
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
            if (_activePresentationGame != null)
            {
                if (_view.Winner < 0) LearnerDraws++;
                else if (_view.Winner == _activePresentationGame.LearnerSeat) LearnerWins++;
                else LearnerLosses++;
            }
            Debug.Log($"Game over: winner={(_view.Winner < 0 ? "DRAW" : "P" + _view.Winner)}; " +
                      $"tally={P0Wins}-{P1Wins}-{Draws}");
        }

        public void SetPaused(bool paused) => Paused = paused;
        bool ValidateResolvedContracts()
        {
            var errors = ModelDuelContractCompatibility.Validate(
                _contractIdentity, _p0Model, _bridge?.Seat0, _p1Model, _bridge?.Seat1);
            if (errors.Count == 0) return true;
            if (_p0Model) P0ArenaStatus = "contract mismatch";
            if (_p1Model) P1ArenaStatus = "contract mismatch";
            Debug.LogError("ModelDuelDriver: " + string.Join(" ", errors));
            _done = true;
            return false;
        }

        void Present(GameState previous)
        {
            ModelDuelRenderDirective directive = _presentation.Next(_view.DeploymentComplete);
            if (directive == ModelDuelRenderDirective.Suppress) return;
            GameState current = _duel.CurrentState;
            if (current == null) throw new InvalidOperationException(
                "arena presentation became visible before the environment exposed a revealed state");
            PlayerId viewer = ObserverPlayer;
            if (directive == ModelDuelRenderDirective.Initialize)
            {
                _board.Render(current.Board);
                _board.RenderEntities(current, viewer);
                EventConsole.Clear();
                EventConsole.Report(current, null, viewer);
                FindAnyObjectByType<CameraRig>()?.Frame();
                return;
            }
            _board.RenderEntities(current, viewer);
            EventConsole.Report(current, CombatLog.Diff(previous, current, viewer), viewer);
        }

        void MarkLiveReloadStatus(string status)
        {
            if (_p0Live) P0ArenaStatus = status;
            if (_p1Live) P1ArenaStatus = status;
        }

        async void AdvanceAtGameBoundary()
        {
            if (IsStarting || _done) return;
            IsStarting = true;
            _duel = null;
            MlPresentationGame next = NextPresentationGame(GamesPlayed);
            try
            {
                if (ShouldReconfigure(_activePresentationGame, next, gameEnded: true))
                {
                    _bridge?.Dispose();
                    _bridge = null;
                    ApplyPresentationGame(next);
                    _activePresentationGame = next;
                    RefreshControllerFlags();
                    _activeScenario = ResolveScenario();
                    _contractIdentity =
                        ModelDuelEnvironmentFactory.ContractIdentity(_activeScenario);
                    if (!await StartPolicyBridge("restart failed")) return;
                }
                else
                {
                    if (!ReloadLiveModelsAtBoundary()) return;
                    if (next != null)
                    {
                        ApplyPresentationGame(next);
                        _activePresentationGame = next;
                    }
                }

                if (_done || this == null) return;
                Seed++;
                _ended = false;
                BeginGame();
            }
            catch (Exception error)
            {
                MarkModelStatus("restart failed");
                Debug.LogError(
                    "ModelDuelDriver: game-boundary restart failed. " +
                    error.Message);
                _done = true;
            }
            finally
            {
                IsStarting = false;
            }
        }

        bool ReloadLiveModelsAtBoundary()
        {
            if ((!_p0Live && !_p1Live) || _bridge == null) return true;
            try
            {
                PolicyReloadResult reload = _bridge.Reload();
                if (!string.IsNullOrWhiteSpace(reload.Error))
                {
                    MarkLiveReloadStatus("reload failed");
                    Debug.LogError(
                        "ModelDuelDriver: live reload failed. " + reload.Error);
                    _done = true;
                    return false;
                }
                MarkLiveReloadStatus(string.Empty);
                return ValidateResolvedContracts();
            }
            catch (Exception error)
            {
                MarkLiveReloadStatus("reload failed");
                Debug.LogError(
                    "ModelDuelDriver: live reload failed. " + error.Message);
                _done = true;
                return false;
            }
        }

        async Task<bool> StartPolicyBridge(string failureStatus)
        {
            if (!_p0Model && !_p1Model)
            {
                _bridge?.Dispose();
                _bridge = null;
                P0ArenaStatus = P1ArenaStatus = string.Empty;
                return true;
            }

            _bridge = new PolicyBridge();
            _startupCancellation?.Dispose();
            _startupCancellation = new CancellationTokenSource();
            IsStarting = true;
            bool ok = await _bridge.StartAsync(
                PythonExe, ServerScript,
                _p0Model ? P0Spec : null,
                _p1Model ? P1Spec : null,
                WorkingDir,
                _contractIdentity.Environment,
                _contractIdentity.Version,
                _contractIdentity.EncodingHash,
                PolicyBridge.DefaultStartupTimeoutMs,
                _startupCancellation.Token);
            IsStarting = false;
            if (_done || this == null) return false;
            if (!ok)
            {
                MarkModelStatus(failureStatus);
                Debug.LogError(
                    "ModelDuelDriver: policy bridge failed to start.");
                _done = true;
                return false;
            }
            P0ArenaStatus = P1ArenaStatus = string.Empty;
            return ValidateResolvedContracts();
        }

        void ApplyPresentationGame(MlPresentationGame game)
        {
            if (game == null) return;
            if (game.LearnerSeat != 0 && game.LearnerSeat != 1)
                throw new InvalidOperationException(
                    "presentation learner seat must be 0 or 1");
            if (game.Scenario == null)
                throw new InvalidOperationException(
                    "presentation game scenario is required");
            P0Spec = game.P0Spec;
            P1Spec = game.P1Spec;
            Observer = game.Observer;
            Scenario = game.Scenario;
            Environment = game.Scenario.Environment == MlContract.AdaptiveVersion
                ? MlEnvironmentContract.AdaptiveV1
                : MlEnvironmentContract.TacticalV1;
        }

        void RefreshControllerFlags()
        {
            _p0Model = IsModel(P0Spec);
            _p1Model = IsModel(P1Spec);
            _p0Live = IsLiveRun(P0Spec);
            _p1Live = IsLiveRun(P1Spec);
        }

        void MarkModelStatus(string status)
        {
            P0ArenaStatus = _p0Model ? status : string.Empty;
            P1ArenaStatus = _p1Model ? status : string.Empty;
        }

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
        public static bool IsLiveRun(string spec)
        {
            if (string.IsNullOrWhiteSpace(spec)) return false;
            string trimmed = spec.TrimStart();
            if (!trimmed.StartsWith("{", StringComparison.Ordinal)) return false;
            try { return JsonUtility.FromJson<LiveSpec>(trimmed)?.mode == "live"; }
            catch (Exception) { return false; }
        }

        static IAgent Scripted(string spec, int seed) => spec == "random" ? new RandomAgent(seed) : (IAgent)new GreedyAgent(seed);
        [Serializable] sealed class LiveSpec { public string mode; }
    }
}

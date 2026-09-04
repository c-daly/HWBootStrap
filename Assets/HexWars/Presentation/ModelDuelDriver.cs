using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Presentation
{
    // Explicit values preserve serialized ML Lab Arena selections across enum growth.
    public enum ModelControllerKind
    {
        Greedy = 0,
        Random = 1,
        FixedRun = 2,
        LiveRun = 3,
        Passive = 4,
    }
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
        // No runtime reader (see ModelDuelDriver.Seed's comment); recorded/serialized metadata only.
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
                case ModelControllerKind.Passive: return "passive";
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

        public bool IsModel => Kind != ModelControllerKind.Greedy &&
            Kind != ModelControllerKind.Random &&
            Kind != ModelControllerKind.Passive;
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
        public MlEnvironmentContract Environment = MlEnvironmentContract.TacticalV2;
        public string ScenarioRunPath = string.Empty;
        // No longer editable via the Arena tab UI (its Observer dropdown was removed — review-fix
        // pass); retained for this class's EditorWindow-serialized-state round-trip and existing
        // ModelDuelConfigurationTests coverage.
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
    [RequireComponent(typeof(EventConsole))]
    [RequireComponent(typeof(ModelArenaIdentityOverlay))]
    public sealed class ModelDuelDriver : MonoBehaviour
    {
        const int LaunchStateSnapshotVersion = 1;

        static readonly JsonSerializerSettings LaunchStateJsonSettings =
            new JsonSerializerSettings
            {
                Culture = CultureInfo.InvariantCulture,
                CheckAdditionalContent = true,
                DateParseHandling = DateParseHandling.None,
                FloatParseHandling = FloatParseHandling.Double,
                MaxDepth = 128,
                MetadataPropertyHandling = MetadataPropertyHandling.Ignore,
                MissingMemberHandling = MissingMemberHandling.Error,
                ObjectCreationHandling = ObjectCreationHandling.Replace,
                TypeNameHandling = TypeNameHandling.None,
            };

        [JsonObject(MemberSerialization.OptIn)]
        sealed class LaunchStateSnapshot
        {
            [JsonProperty("version", Required = Required.Always)]
            public int Version;

            [JsonProperty("scenario", Required = Required.AllowNull)]
            public TrainingScenario Scenario;

            [JsonProperty("presentation_plan", Required = Required.AllowNull)]
            public MlPresentationSchedule PresentationPlan;
        }

        public string PythonExe;
        public string ServerScript;
        public string WorkingDir;
        public string P0Spec = "greedy";
        public string P1Spec = "greedy";
        public MlEnvironmentContract Environment = MlEnvironmentContract.TacticalV1;
        public TrainingScenario Scenario;
        /// <summary>Spec §"Fog-of-War Indicator" (amended 2026-07-25): the single on/off toggle for the
        /// acting-player fog marking. Default on — the marking is the point of watching a fog run.</summary>
        public bool ShowFogMarking = true;
        [SerializeReference]
        public MlPresentationSchedule PresentationPlan;
        [SerializeField, HideInInspector]
        string _launchStateSnapshot = string.Empty;
        // Removed dead ModelDuelDriver.Observer/ObserverPlayer (Task C review carry): omniscient
        // presentation always passes viewer: null (RenderEntities/InitializeBoard), so the field had no
        // remaining reader besides one test. ModelDuelObserverSeat/ModelDuelObserver.Resolve, and the
        // ModelDuelConfiguration/MlPresentationGame/MlArenaLaunchPlan.Observer, are untouched even
        // though the ML Lab Arena tab's "Observer" dropdown that used to feed them is now ALSO gone
        // (review-fix pass) — ReplayViewerMenu.LaunchDuel dropped its own now-parameterless `observer`
        // argument to match. These three Observer members remain purely recorded/serialized metadata
        // (e.g. MlRunPresentationPlan still derives MlPresentationGame.Observer from the recorded
        // learner seat) with no runtime reader left anywhere in the presentation pipeline — kept alive
        // only by existing test coverage (ModelDuelConfigurationTests, MlRunPresentationPlanTests).
        public int Seed;
        public float SecondsPerAction = 0.4f;
        public bool Loop;
        public float SecondsBetweenGames = 1.5f;

        public bool Paused { get; private set; }
        public bool IsDone => _done;
        public int CurrentSeat => _duel == null ? -1 : _view.Seat;
        /// <summary>Whose turn the identity rows' "▶" indicator marks: derived from
        /// <see cref="PresentedState"/> (what is actually on screen), never from the live simulation
        /// seat, which can already be several commands ahead while presentation catches up.</summary>
        public int PresentedActiveSeat => _presentedState == null ? -1 : (int)_presentedState.ActivePlayer;
        /// <summary>The last game state whose transition has fully finished presenting — board tokens,
        /// point totals, the active-player indicator, and console text all read from here, never from a
        /// simulation state that may already be ahead. Null before the first render (or, for adaptive,
        /// before the atomic reveal).</summary>
        public GameState PresentedState => _presentedState;
        public int GamesPlayed { get; private set; }
        public int P0Wins { get; private set; }
        public int P1Wins { get; private set; }
        public int Draws { get; private set; }
        public int LearnerWins { get; private set; }
        public int LearnerLosses { get; private set; }
        public int LearnerDraws { get; private set; }
        /// <summary>Every board cell outside the acting player's (<see cref="PresentedState"/>'s
        /// <see cref="GameState.ActivePlayer"/>) current visibility — spec §"Fog-of-War Indicator"
        /// (amended 2026-07-25). Empty while <see cref="ShowFogMarking"/> is off, before the first
        /// render, or when the scenario's fog of war is disabled. A live derived read of
        /// <see cref="PresentedState"/> (never cached), so it is always already correct for whatever the
        /// board currently shows — <see cref="RefreshFogMarking"/> exists only to push it into the
        /// renderer at the two points <see cref="PresentedState"/> actually advances.</summary>
        public IReadOnlyCollection<HexCoord> MarkedFogCells => ShowFogMarking
            ? FogMarkingOverlay.MarkedCells(_presentedState)
            : Array.Empty<HexCoord>();
        public bool ShouldShowArenaOverlays => Environment == MlEnvironmentContract.TacticalV1
            || Environment == MlEnvironmentContract.TacticalV2
            || Environment == MlEnvironmentContract.TacticalV3
            || (_duel != null && _view.DeploymentComplete);
        public PolicySeatInfo P0Resolved => _bridge?.Seat0;
        public PolicySeatInfo P1Resolved => _bridge?.Seat1;
        public string P0ArenaStatus { get; private set; }
        public string P1ArenaStatus { get; private set; }
        public int CurrentLearnerSeat => _activePresentationGame?.LearnerSeat ?? -1;

        public ModelArenaSeatIdentity[] IdentitySnapshot() => ModelArenaIdentity.Build(
            P0Spec, P1Spec, P0Resolved, P1Resolved, PresentedActiveSeat,
            P0Wins, P1Wins, Draws, P0ArenaStatus, P1ArenaStatus,
            CurrentLearnerSeat, LearnerWins, LearnerLosses, LearnerDraws,
            _activePresentationGame?.OpponentLabel,
            PresentedPoints(PlayerId.Player0), PresentedPoints(PlayerId.Player1));

        int PresentedPoints(PlayerId player) => _presentedState?.Player(player).Points ?? 0;

        public MlPresentationGame NextPresentationGame(int gamesPlayed) =>
            PresentationPlan?.NextPresentationGame(gamesPlayed);

        public void ConfigureLaunchState(
            TrainingScenario scenario,
            MlPresentationSchedule presentationPlan)
        {
            string snapshot = PolicyJson.Serialize(new LaunchStateSnapshot
            {
                Version = LaunchStateSnapshotVersion,
                Scenario = scenario,
                PresentationPlan = presentationPlan,
            });
            Scenario = scenario;
            PresentationPlan = presentationPlan;
            _launchStateSnapshot = snapshot;
        }

        void RestoreLaunchState()
        {
            if (string.IsNullOrWhiteSpace(_launchStateSnapshot)) return;

            LaunchStateSnapshot snapshot;
            try
            {
                snapshot = JsonConvert.DeserializeObject<LaunchStateSnapshot>(
                    _launchStateSnapshot, LaunchStateJsonSettings);
                if (snapshot == null)
                    throw new JsonSerializationException(
                        "arena launch-state snapshot is empty");
            }
            catch (JsonException error)
            {
                throw new InvalidOperationException(
                    "arena launch-state snapshot could not be restored: " +
                    error.Message, error);
            }
            if (snapshot.Version != LaunchStateSnapshotVersion)
                throw new InvalidOperationException(
                    "unsupported arena launch-state snapshot version " +
                    snapshot.Version);

            Scenario = snapshot.Scenario;
            PresentationPlan = snapshot.PresentationPlan;
        }

        public static bool ShouldReconfigure(
            MlPresentationGame previous,
            MlPresentationGame next,
            bool gameEnded) =>
            gameEnded && previous != null && next != null &&
            (!string.Equals(previous.P0Spec, next.P0Spec, StringComparison.Ordinal) ||
             !string.Equals(previous.P1Spec, next.P1Spec, StringComparison.Ordinal));

        BoardRenderer _board;
        ActionPresenter _presenter;
        IModelDuelEnvironment _duel;
        TrainingScenario _activeScenario;
        PolicyBridge _bridge;
        ModelDuelContractIdentity _contractIdentity;
        ModelDuelView _view;
        ModelDuelPresentationState _presentation;
        GameState _presentedState;
        bool _p0Model, _p1Model, _p0Live, _p1Live, _done, _ended;
        float _timer, _restTimer;
        CancellationTokenSource _startupCancellation;
        MlPresentationGame _activePresentationGame;
        public bool IsStarting { get; private set; }

        void Awake()
        {
            _board = GetComponent<BoardRenderer>();
            _presenter = GetComponent<ActionPresenter>() ?? gameObject.AddComponent<ActionPresenter>();
            _presenter.ItemCommitted += OnItemCommitted;
            _presenter.RenderFault += OnPresenterRenderFault;
        }

        async void Start()
        {
            try
            {
                RestoreLaunchState();
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
                scenario.TacticalV2 = null;
                scenario.TacticalV3Reward = null;
                scenario.TacticalV3 = null;
            }
            else if (scenario.Environment == MlContract.AdaptiveVersion)
            {
                scenario.TacticalReward = null;
                scenario.TacticalV2 = null;
                scenario.TacticalV3Reward = null;
                scenario.TacticalV3 = null;
            }
            else if (scenario.Environment == MlContract.TacticalV2Version)
            {
                scenario.AdaptiveReward = null;
                scenario.Adaptive = null;
                scenario.TacticalV3Reward = null;
                scenario.TacticalV3 = null;
            }
            else if (scenario.Environment == MlContract.TacticalV3Version)
            {
                scenario.AdaptiveReward = null;
                scenario.Adaptive = null;
                scenario.TacticalReward = null;
                scenario.TacticalV2 = null;
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
            _duel.CaptureTransitions = true;
            _presentation = new ModelDuelPresentationState(Environment);
            _presentedState = null;
            _presenter.ResetQueue();
            _view = _duel.Reset(Seed, c0, c1);
            HandlePresentation();
            _timer = 0;
        }

        void Update()
        {
            if (_done || Paused || _duel == null) return;
            // Pacing gate: never request the next policy/step action while the last batch of
            // transitions is still queued or mid-animation — spec §"Viewer Playback". This governs the
            // Unity viewing duel only; headless training never touches this driver.
            if (_presenter.IsBusy) { _timer = 0f; return; }
            if (_duel.RequiresContinuation)
            {
                _view = _duel.Continue();
                HandlePresentation();
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
                if (_duel is IStructuredModelDuelEnvironment structured)
                {
                    TacticalV3View decision = _view.StructuredDecision ??
                        throw new InvalidOperationException(
                            "structured environment did not expose a tactical-v3 decision");
                    TacticalV3ViewDto payload = TacticalV3PolicyPayload.From(decision);
                    PolicyCandidateResult selected =
                        _bridge.ActStructured(seat, payload);
                    int matches = decision.Decision.Candidates.Count(candidate =>
                        candidate.DecisionId == selected.DecisionId &&
                        candidate.CandidateId == selected.CandidateId);
                    if (matches != 1)
                        throw new InvalidOperationException(
                            "structured policy selected an unknown candidate identity");
                    _view = structured.Step(
                        selected.DecisionId, selected.CandidateId);
                }
                else
                {
                    var legacy = _duel as ILegacyModelDuelEnvironment ??
                        throw new InvalidOperationException(
                            "environment exposes neither structured nor legacy stepping");
                    int action = _bridge.Act(
                        seat, _view.Observation, _view.ActionMask);
                    _view = legacy.Step(action);
                }
                HandlePresentation();
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

        /// <summary>Drains every transition the engine accepted since the last drain and hands them to
        /// the shared gameplay animation pipeline (<see cref="ActionPresenter"/>) — omnisciently
        /// (<c>viewer: null</c>): the arena is a full spectator, never fog-limited like a seated player,
        /// even when the underlying scenario trains with fog of war on. Adaptive-v1 stays pre-empted
        /// (<see cref="ModelDuelRenderDirective.Suppress"/>) until its atomic reveal; the engine never
        /// produces pregame-deployment transitions, so nothing is lost by not draining while suppressed.
        /// A transition that cannot be queued or rendered stops the run with an explicit status — spec
        /// §"Validation and Failure Behavior" — rather than skipping it and continuing with stale
        /// visuals.</summary>
        void HandlePresentation()
        {
            ModelDuelRenderDirective directive = _presentation.Next(_view.DeploymentComplete);
            if (directive == ModelDuelRenderDirective.Suppress) return;
            try
            {
                IReadOnlyList<DuelTransition> transitions = _duel.DrainTransitions();
                if (directive == ModelDuelRenderDirective.Initialize)
                {
                    GameState anchor = transitions.Count > 0 ? transitions[0].Previous : _duel.CurrentState;
                    if (anchor == null) throw new InvalidOperationException(
                        "arena presentation became visible before the environment exposed a revealed state");
                    InitializeBoard(anchor);
                }
                foreach (DuelTransition transition in transitions)
                {
                    _presenter.Enqueue(transition.Previous, transition.Command, transition.Resulting, isLocal: false);
                    // A render fault routes synchronously back into MarkPresentationError (RenderFault ->
                    // OnPresenterRenderFault) before Enqueue returns, setting _done. Keep draining anyway
                    // and every remaining transition would fault too — multi-fault log spam plus
                    // presentation continuing past a run the driver already declared dead.
                    if (_done) break;
                }
            }
            catch (Exception error)
            {
                MarkPresentationError("presentation error: " + error.Message, error);
            }
        }

        /// <summary>Routes a render-path fault (mid-animation, or <see cref="ActionPresenter"/>'s own
        /// post-item <c>Commit</c>) into the same stop-the-run path as an unpresentable transition —
        /// spec §"Validation and Failure Behavior". Without this, <see cref="ActionPresenter"/> would
        /// still recover on its own (queue cleared, <c>IsBusy</c> false), but the driver would keep
        /// ticking forever with a dead board and no surfaced status. No exception object is forwarded
        /// to <see cref="MarkPresentationError"/> here — <see cref="ActionPresenter"/> already logged
        /// this exact exception via <c>Debug.LogException</c> before raising this event, and logging
        /// it a second time would just duplicate the same stack trace in the console.</summary>
        void OnPresenterRenderFault(Exception error) =>
            MarkPresentationError("render error: " + error.Message);

        void InitializeBoard(GameState anchor)
        {
            _board.Render(anchor.Board);
            _board.RenderEntities(anchor, viewer: null);
            EventConsole.Clear();
            EventConsole.Report(anchor, null, null);
            _presentedState = anchor;
            RefreshFogMarking();
            FindAnyObjectByType<CameraRig>()?.Frame();
        }

        /// <summary>Advances the lagging <see cref="PresentedState"/> exactly when <see cref="ActionPresenter"/>
        /// finishes presenting each queued transition — never sooner. Console text reads from the same
        /// transition (omnisciently), so narration always matches what just finished on screen.</summary>
        void OnItemCommitted(GameState prev, Command cmd, GameState next)
        {
            _presentedState = next;
            EventConsole.Report(next, CombatLog.Diff(prev, next, null), null);
            RefreshFogMarking();
        }

        /// <summary>Pushes <see cref="MarkedFogCells"/> into the board's rendering — the acting player
        /// can change with a presented turn transition, so this must run every time
        /// <see cref="PresentedState"/> actually advances (<see cref="InitializeBoard"/> and
        /// <see cref="OnItemCommitted"/>), never sooner.</summary>
        void RefreshFogMarking()
        {
            if (_board != null) _board.UpdateFogMarking(_presentedState, MarkedFogCells);
        }

        /// <summary>UI entry point for the single fog-marking toggle (spec §"Fog-of-War Indicator") — also
        /// pushes the change to the board immediately, so hiding/showing the marking doesn't wait for the
        /// next presented transition.</summary>
        public void SetShowFogMarking(bool show)
        {
            ShowFogMarking = show;
            RefreshFogMarking();
        }

        void MarkPresentationError(string message, Exception error = null)
        {
            P0ArenaStatus = message;
            P1ArenaStatus = message;
            Debug.LogError("ModelDuelDriver: " + message);
            if (error != null) Debug.LogException(error); // preserve the stack trace, not just Message
            _done = true;
        }

        void MarkLiveReloadStatus(string status)
        {
            if (_p0Live) P0ArenaStatus = status;
            if (_p1Live) P1ArenaStatus = status;
        }

        /// <summary>Called by MlLabWindow (D2, "Lab stops lying about trainers") each repaint while its
        /// Arena tab is spectating a live run: an honest, non-interrupting status line for a live seat
        /// whose backing trainer process has exited or stalled, exactly like <see
        /// cref="MarkLiveReloadStatus"/> already does for weight-reload failures -- never pauses or
        /// resets playback, and clears back to empty once the trainer looks healthy again. A no-op for
        /// a seat that isn't a live run (fixed run / greedy / random controllers never reload weights,
        /// so trainer liveness is not their concern).</summary>
        public void MarkTrainerLivenessStatus(string status)
        {
            if (_p0Live) P0ArenaStatus = status ?? string.Empty;
            if (_p1Live) P1ArenaStatus = status ?? string.Empty;
        }

        async void AdvanceAtGameBoundary()
        {
            if (IsStarting || _done) return;
            IsStarting = true;
            _duel = null;
            try
            {
                MlPresentationGame next = NextPresentationGame(GamesPlayed);
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
                _contractIdentity.CapacityHash,
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
            Scenario = game.Scenario;
            Environment = game.Scenario.Environment == MlContract.AdaptiveVersion
                ? MlEnvironmentContract.AdaptiveV1
                : game.Scenario.Environment == MlContract.TacticalV2Version
                    ? MlEnvironmentContract.TacticalV2
                    : game.Scenario.Environment == MlContract.TacticalV3Version
                        ? MlEnvironmentContract.TacticalV3
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
        void OnDestroy()
        {
            if (_presenter != null)
            {
                _presenter.ItemCommitted -= OnItemCommitted;
                _presenter.RenderFault -= OnPresenterRenderFault;
            }
            StopDuel();
        }

        public static bool IsModel(string spec) =>
            !string.IsNullOrWhiteSpace(spec) &&
            spec != "greedy" &&
            spec != "random" &&
            spec != "passive";
        public static bool IsLiveRun(string spec)
        {
            if (string.IsNullOrWhiteSpace(spec)) return false;
            string trimmed = spec.TrimStart();
            if (!trimmed.StartsWith("{", StringComparison.Ordinal)) return false;
            try { return JsonUtility.FromJson<LiveSpec>(trimmed)?.mode == "live"; }
            catch (Exception) { return false; }
        }

        static IAgent Scripted(string spec, int seed)
        {
            if (spec == "random") return new RandomAgent(seed);
            if (spec == "passive") return new PassiveAgent();
            return new GreedyAgent(seed);
        }
        [Serializable] sealed class LiveSpec { public string mode; }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using HexWars.Engine.Rl;
using UnityEditor;
using UnityEngine;

namespace HexWars.Presentation.EditorTools.MlLab
{
    public enum MlLabUiPhase { Idle, Validating, Running, Stopping, Completed, Failed, ExternallyRunning }

    public sealed class MlLabWindowState
    {
        public MlLabUiPhase Phase { get; private set; } = MlLabUiPhase.Idle;
        public bool LaunchedHere { get; private set; }
        public string Error { get; private set; } = string.Empty;

        public void BeginValidation() { Phase = MlLabUiPhase.Validating; Error = string.Empty; }
        public void MarkLaunched() { LaunchedHere = true; Phase = MlLabUiPhase.Running; Error = string.Empty; }
        public void BeginStopping() { Phase = MlLabUiPhase.Stopping; Error = string.Empty; }
        public void Fail(string error) { Phase = MlLabUiPhase.Failed; Error = error ?? "Unknown ML Lab error."; }
        public void Reset() { Phase = MlLabUiPhase.Idle; LaunchedHere = false; Error = string.Empty; }

        public void Apply(MlRunState state, int pid)
        {
            switch (state)
            {
                case MlRunState.Created:
                    Phase = MlLabUiPhase.Validating;
                    break;
                case MlRunState.Running:
                    Phase = LaunchedHere ? MlLabUiPhase.Running : MlLabUiPhase.ExternallyRunning;
                    break;
                case MlRunState.Stopping:
                    Phase = MlLabUiPhase.Stopping;
                    break;
                case MlRunState.Stopped:
                case MlRunState.Completed:
                    Phase = MlLabUiPhase.Completed;
                    LaunchedHere = false;
                    break;
                case MlRunState.Failed:
                    Phase = MlLabUiPhase.Failed;
                    LaunchedHere = false;
                    break;
            }
        }
    }

    public sealed class MlDoctorReport
    {
        public bool Healthy { get; private set; }
        public string Summary { get; private set; }

        public static MlDoctorReport Parse(string json)
        {
            DoctorEnvelope envelope;
            try { envelope = JsonUtility.FromJson<DoctorEnvelope>(json); }
            catch (Exception) { envelope = null; }
            if (envelope == null || envelope.result == null)
                return new MlDoctorReport { Healthy = false, Summary = "Doctor returned an unreadable response." };
            var checks = envelope.result.checks ?? Array.Empty<DoctorCheck>();
            var failedRequired = checks.Where(check => check != null && check.required && !check.ok)
                .Select(check => check.name + ": " + check.detail).ToArray();
            var optional = checks.Where(check => check != null && !check.required && !check.ok)
                .Select(check => check.name).ToArray();
            bool healthy = envelope.ok && envelope.result.ok && failedRequired.Length == 0;
            string summary = healthy
                ? "Doctor passed all required checks."
                : "Doctor failed required checks: " + (failedRequired.Length == 0
                    ? envelope.result.message ?? "unknown failure"
                    : string.Join("; ", failedRequired));
            if (optional.Length > 0) summary += " Optional unavailable: " + string.Join(", ", optional) + ".";
            return new MlDoctorReport { Healthy = healthy, Summary = summary };
        }

        [Serializable] sealed class DoctorEnvelope { public bool ok; public DoctorResult result; }
        [Serializable] sealed class DoctorResult { public bool ok; public string message; public DoctorCheck[] checks; }
        [Serializable] sealed class DoctorCheck { public string name; public bool ok; public bool required; public string detail; }
    }

    public sealed class MlTrainingScenarioSession
    {
        MlTrainingScenarioLibrary _library;
        readonly string _libraryPath;

        MlTrainingScenarioSession(string libraryPath, Exception error)
        {
            _libraryPath = libraryPath;
            LibraryException = error;
            LibraryError = libraryPath + ": " + error.Message;
        }

        public MlTrainingScenarioSession(MlTrainingScenarioLibrary library)
            : this(library, string.Empty)
        {
        }

        MlTrainingScenarioSession(MlTrainingScenarioLibrary library, string libraryPath)
        {
            _library = library ?? throw new ArgumentNullException(nameof(library));
            _libraryPath = libraryPath ?? string.Empty;
            SelectEnvironment(MlEnvironmentContract.TacticalV1);
        }

        public MlEnvironmentContract Environment { get; private set; }
        public string SelectedTemplateId { get; private set; } = string.Empty;
        public MlTrainingScenario WorkingCopy { get; private set; }
        public string SaveName { get; private set; } = string.Empty;
        public string SaveId { get; private set; } = string.Empty;
        public bool OverwriteArmed { get; private set; }
        public Exception LibraryException { get; }
        public string LibraryError { get; } = string.Empty;
        public bool CanLaunch
        {
            get
            {
                if (!string.IsNullOrWhiteSpace(LibraryError) ||
                    WorkingCopy == null ||
                    WorkingCopy.Validate().Count > 0)
                    return false;
                try
                {
                    MlTrainingScenarioPreflight.Create(WorkingCopy);
                    return true;
                }
                catch (Exception)
                {
                    return false;
                }
            }
        }
        public IReadOnlyList<MlTrainingScenario> AvailableTemplates =>
            _library?.Filter(Environment) ?? Array.Empty<MlTrainingScenario>();

        public static MlTrainingScenarioSession Load(string libraryPath)
        {
            try
            {
                return new MlTrainingScenarioSession(
                    MlTrainingScenarioLibrary.Load(libraryPath), libraryPath);
            }
            catch (Exception error)
            {
                return new MlTrainingScenarioSession(libraryPath, error);
            }
        }

        public void SelectEnvironment(MlEnvironmentContract environment)
        {
            EnsureLibrary();
            Environment = environment;
            IReadOnlyList<MlTrainingScenario> available = _library.Filter(environment);
            if (available.Count == 0)
                throw new InvalidDataException(
                    "No templates are available for " +
                    MlEnvironmentContracts.CliValue(environment) + ".");
            string standardId = environment == MlEnvironmentContract.AdaptiveV1
                ? "adaptive-standard"
                : "tactical-standard";
            MlTrainingScenario selected =
                available.FirstOrDefault(item => item.Id == standardId) ?? available[0];
            Select(selected);
        }

        public void SelectTemplate(string id)
        {
            EnsureLibrary();
            MlTrainingScenario selected = _library.Filter(Environment).FirstOrDefault(
                item => string.Equals(item.Id, id, StringComparison.Ordinal));
            if (selected == null)
                throw new ArgumentException(
                    "Template '" + id + "' is not available for " +
                    MlEnvironmentContracts.CliValue(Environment) + ".", nameof(id));
            Select(selected);
        }

        public void Reload()
        {
            EnsureLibrary();
            SelectTemplate(SelectedTemplateId);
        }

        public void ReloadTemplates()
        {
            if (string.IsNullOrWhiteSpace(_libraryPath))
                throw new InvalidOperationException(
                    "This session does not have a template-library path.");
            _library = MlTrainingScenarioLibrary.Load(_libraryPath);
            string selectedId = SelectedTemplateId;
            MlTrainingScenario selected = _library.Filter(Environment).FirstOrDefault(
                item => string.Equals(item.Id, selectedId, StringComparison.Ordinal));
            if (selected == null)
            {
                SelectEnvironment(Environment);
                return;
            }
            Select(selected);
        }

        public void SetSaveIdentity(string name, string id)
        {
            SaveName = name ?? string.Empty;
            SaveId = id ?? string.Empty;
            OverwriteArmed = false;
        }

        public bool SaveAsTemplate()
        {
            EnsureWritableLibrary();
            MlTrainingScenario candidate = SaveCandidate();
            bool collision = _library.Templates.Any(
                item => string.Equals(item.Id, candidate.Id, StringComparison.Ordinal));
            if (collision)
            {
                OverwriteArmed = true;
                return false;
            }
            MlTrainingScenarioStore.SaveAsTemplate(
                _libraryPath, candidate, overwrite: false);
            CompleteSave(candidate.Id);
            return true;
        }

        public bool ConfirmOverwrite()
        {
            EnsureWritableLibrary();
            if (!OverwriteArmed) return false;
            MlTrainingScenario candidate = SaveCandidate();
            MlTrainingScenarioStore.SaveAsTemplate(
                _libraryPath, candidate, overwrite: true);
            CompleteSave(candidate.Id);
            return true;
        }

        void CompleteSave(string id)
        {
            _library = MlTrainingScenarioLibrary.Load(_libraryPath);
            Environment = WorkingCopy.Environment;
            SelectTemplate(id);
            OverwriteArmed = false;
        }

        MlTrainingScenario SaveCandidate()
        {
            if (string.IsNullOrWhiteSpace(SaveName))
                throw new InvalidDataException("Template name is required.");
            if (string.IsNullOrWhiteSpace(SaveId))
                throw new InvalidDataException("Template ID is required.");
            string prefix = Environment == MlEnvironmentContract.AdaptiveV1
                ? "adaptive-"
                : "tactical-";
            if (!SaveId.StartsWith(prefix, StringComparison.Ordinal))
                throw new InvalidDataException(
                    "Template ID must start with '" + prefix + "'.");
            MlTrainingScenario candidate = WorkingCopy.Clone();
            candidate.Name = SaveName.Trim();
            candidate.Id = SaveId.Trim();
            return candidate;
        }

        void Select(MlTrainingScenario selected)
        {
            SelectedTemplateId = selected.Id;
            WorkingCopy = selected.Clone();
            SaveName = selected.Name;
            SaveId = selected.Id;
            OverwriteArmed = false;
        }

        void EnsureLibrary()
        {
            if (_library == null)
                throw new InvalidOperationException(LibraryError);
        }

        void EnsureWritableLibrary()
        {
            EnsureLibrary();
            if (string.IsNullOrWhiteSpace(_libraryPath))
                throw new InvalidOperationException(
                    "This session does not have a template-library path.");
        }
    }

    public sealed class MlTrainingScenarioPreflight
    {
        MlTrainingScenarioPreflight(
            MlTrainingScenario scenario, int observationSize, int actionSize)
        {
            TemplateId = scenario.Id;
            TemplateName = scenario.Name;
            Environment = scenario.Environment;
            BoardWidth = scenario.Board.Width;
            BoardHeight = scenario.Board.Height;
            ZoneDepth = scenario.Board.ZoneDepth;
            ActionsPerTurn = scenario.Rules.ActionsPerTurn;
            RoundCap = scenario.Rules.RoundCap;
            MaxSteps = scenario.Episode.MaxSteps;
            FogOfWar = scenario.Rules.FogOfWar;
            ObservationSize = observationSize;
            ActionSize = actionSize;
            LargeScenarioWarning =
                (long)scenario.Board.Width * scenario.Board.Height > 13L * 9L;
        }

        public string TemplateId { get; }
        public string TemplateName { get; }
        public MlEnvironmentContract Environment { get; }
        public int BoardWidth { get; }
        public int BoardHeight { get; }
        public int ZoneDepth { get; }
        public int ActionsPerTurn { get; }
        public int RoundCap { get; }
        public int MaxSteps { get; }
        public bool FogOfWar { get; }
        public int ObservationSize { get; }
        public int ActionSize { get; }
        public bool LargeScenarioWarning { get; }
        public string DisplayText => Describe("not selected", "not selected");

        public static MlTrainingScenarioPreflight Create(
            MlTrainingScenario scenario)
        {
            if (scenario == null) throw new ArgumentNullException(nameof(scenario));
            TrainingScenario engineScenario = ToEngine(scenario);
            MlContract contract = scenario.Environment == MlEnvironmentContract.AdaptiveV1
                ? MlContract.CreateAdaptive(engineScenario.BuildAdaptive())
                : MlContract.Create(engineScenario.BuildTactical());
            return new MlTrainingScenarioPreflight(
                scenario, contract.ObservationSize, contract.ActionSize);
        }

        public static MlTrainingScenarioPreflight LoadSourceRun(
            string runDirectory)
        {
            if (string.IsNullOrWhiteSpace(runDirectory))
                throw new ArgumentException(
                    "Source run is required.", nameof(runDirectory));
            string path = Path.Combine(runDirectory, "scenario.json");
            return Create(MlTrainingScenarioFile.Load(path));
        }

        public string Describe(string opponent, string learnerSeats)
        {
            string actions = ActionsPerTurn == 0
                ? "Whole team"
                : ActionsPerTurn.ToString(CultureInfo.InvariantCulture);
            return TemplateName + " \u00b7 " +
                   MlEnvironmentContracts.CliValue(Environment) + "\n" +
                   "Board " + BoardWidth + "\u00d7" + BoardHeight +
                   " \u00b7 zone depth " + ZoneDepth +
                   " \u00b7 actions/turn " + actions + "\n" +
                   "Round cap " + RoundCap +
                   " \u00b7 max steps " + MaxSteps +
                   " \u00b7 fog " + (FogOfWar ? "on" : "off") +
                   " \u00b7 opponent " + opponent +
                   " \u00b7 learner seats " + learnerSeats + "\n" +
                   "Observation " + ObservationSize +
                   " \u00b7 actions " + ActionSize;
        }

        public static TrainingScenario ToEngine(MlTrainingScenario scenario)
        {
            if (scenario == null) throw new ArgumentNullException(nameof(scenario));
            var converted = new TrainingScenario
            {
                SchemaVersion = scenario.SchemaVersion,
                Id = scenario.Id,
                Name = scenario.Name,
                Environment = MlEnvironmentContracts.CliValue(scenario.Environment),
                Board = scenario.Board == null
                    ? null
                    : new TrainingBoardConfig
                    {
                        Width = scenario.Board.Width,
                        Height = scenario.Board.Height,
                        MaxElevation = scenario.Board.MaxElevation,
                        ZoneDepth = scenario.Board.ZoneDepth,
                        FlatChance = scenario.Board.FlatChance,
                        PlainsWeight = scenario.Board.PlainsWeight,
                        ForestWeight = scenario.Board.ForestWeight,
                        RoughWeight = scenario.Board.RoughWeight,
                        WaterWeight = scenario.Board.WaterWeight,
                    },
                Rules = scenario.Rules == null
                    ? null
                    : new TrainingRuleConfig
                    {
                        ActionsPerTurn = scenario.Rules.ActionsPerTurn,
                        RoundCap = scenario.Rules.RoundCap,
                        StartingPoints = scenario.Rules.StartingPoints,
                        FogOfWar = scenario.Rules.FogOfWar,
                        BiomesEnabled = scenario.Rules.BiomesEnabled,
                        BountyRate = scenario.Rules.BountyRate,
                        DeployCostMultiplier = scenario.Rules.DeployCostMultiplier,
                        GeneratorCost = scenario.Rules.GeneratorCost,
                        GeneratorOutput = scenario.Rules.GeneratorOutput,
                        GeneratorHealth = scenario.Rules.GeneratorHealth,
                    },
                Episode = scenario.Episode == null
                    ? null
                    : new TrainingEpisodeConfig
                    {
                        MaxSteps = scenario.Episode.MaxSteps,
                    },
                TacticalReward = scenario.TacticalReward == null
                    ? null
                    : new TacticalRewardConfig
                    {
                        ShapeScale = scenario.TacticalReward.ShapeScale,
                        StepPenalty = scenario.TacticalReward.StepPenalty,
                        ClosingWeight = scenario.TacticalReward.ClosingWeight,
                        DrawCreditWeight = scenario.TacticalReward.DrawCreditWeight,
                        PointsWeight = scenario.TacticalReward.PointsWeight,
                    },
                AdaptiveReward = scenario.AdaptiveReward == null
                    ? null
                    : new AdaptiveRewardConfig
                    {
                        IntermediateDecisionPenalty =
                            scenario.AdaptiveReward.IntermediateDecisionPenalty,
                        DeploymentCompletionBonus =
                            scenario.AdaptiveReward.DeploymentCompletionBonus,
                    },
                Adaptive = scenario.Adaptive == null
                    ? null
                    : new TrainingAdaptiveConfig
                    {
                        StartingUnitCount = scenario.Adaptive.StartingUnitCount,
                        StartingArmyBudget = scenario.Adaptive.StartingArmyBudget,
                        MaxDesignPointCost = scenario.Adaptive.MaxDesignPointCost,
                    },
            };
            IReadOnlyList<string> errors = converted.Validate();
            if (errors.Count > 0)
                throw new InvalidDataException(string.Join("; ", errors));
            return converted;
        }
    }

    public sealed class MlLabWindow : EditorWindow
    {
        const string SelectedRunKey = "HexWars.MlLab.SelectedRun";
        const double PollIntervalSeconds = 1.0;

        [SerializeField] MlLabConfig _config = new MlLabConfig();
        [SerializeField] bool _resume;
        [SerializeField] bool _useTensorBoard = true;
        [SerializeField] bool _useWandb;
        [SerializeField] bool _useCustomTracker;
        [SerializeField] string _customTrackerAdapter = string.Empty;
        [SerializeField] bool _showAdvanced;
        [SerializeField] bool _showGameSettings;
        [SerializeField] int _tab;
        [SerializeField] ModelDuelConfiguration _arena = new ModelDuelConfiguration();

        MlLabWindowState _state = new MlLabWindowState();
        MlTrainingScenarioSession _scenarioSession;
        MlCliProcess _training;
        MlCliProcess _statusQuery;
        MlCliProcess _command;
        MlRunStatus _status;
        Vector2 _scroll;
        Vector2 _logScroll;
        string _selectedRun = string.Empty;
        string[] _knownRuns = Array.Empty<string>();
        string[] _knownRunLabels = Array.Empty<string>();
        int _knownRunIndex;
        double _nextPoll;
        double _throughput;
        string _lastMetricTime = string.Empty;
        bool _watchWhenReady;
        bool _watchLaunched;
        string _notice = string.Empty;
        string _arenaError = string.Empty;
        string _arenaNotice = string.Empty;
        CommandKind _activeCommand;

        enum CommandKind { None, Doctor, Control }

        string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;
        string PythonDir => Path.Combine(ProjectRoot, "python");
        string PythonExe => MlLabPaths.ResolvePythonExecutable(ProjectRoot);
        string CliScript => Path.Combine(PythonDir, "hexwars_ml.py");
        string RunsRoot => Path.Combine(PythonDir, "runs");
        string TemplateLibraryPath =>
            Path.Combine(ProjectRoot, "python", "config", "training-game-templates.json");
        string SessionScenarioPath =>
            Path.Combine(ProjectRoot, "Library", "HexWars", "MLLab", "scenario.json");
        string GymServer => Path.Combine(ProjectRoot, "engine", "HexWars.GymServer", "bin", "Release", "net8.0", "HexWars.GymServer.dll");

        [MenuItem("HexWars/ML Lab", priority = 20)]
        public static void Open()
        {
            var window = GetWindow<MlLabWindow>("HexWars ML Lab");
            window.minSize = new Vector2(620, 700);
            window.Show();
        }

        void OnEnable()
        {
            _state = new MlLabWindowState();
            LoadScenarioLibrary();
            CreateProcessOwners();
            _selectedRun = SessionState.GetString(SelectedRunKey, string.Empty);
            var attached = MlRunAttachment.Restore();
            if (string.IsNullOrWhiteSpace(_selectedRun) && attached.Exists)
                _selectedRun = attached.RunDirectory;
            RefreshKnownRuns();
            EditorApplication.update += Tick;
            _nextPoll = 0;
        }

        void OnDisable()
        {
            EditorApplication.update -= Tick;
            SessionState.SetString(SelectedRunKey, _selectedRun ?? string.Empty);
            DisposeProcessOwners();
        }

        void CreateProcessOwners()
        {
            DisposeProcessOwners();
            _training = new MlCliProcess(500);
            _statusQuery = new MlCliProcess(60);
            _command = new MlCliProcess(160);
            _training.Changed += Repaint;
            _training.Exited += OnTrainingExited;
            _statusQuery.Changed += Repaint;
            _statusQuery.StatusReceived += OnStatusReceived;
            _command.Changed += Repaint;
            _command.Exited += OnCommandExited;
        }

        void DisposeProcessOwners()
        {
            if (_training != null)
            {
                _training.Changed -= Repaint;
                _training.Exited -= OnTrainingExited;
                _training.Dispose();
            }
            if (_statusQuery != null)
            {
                _statusQuery.Changed -= Repaint;
                _statusQuery.StatusReceived -= OnStatusReceived;
                _statusQuery.Dispose();
            }
            if (_command != null)
            {
                _command.Changed -= Repaint;
                _command.Exited -= OnCommandExited;
                _command.Dispose();
            }
            _training = null;
            _statusQuery = null;
            _command = null;
        }

        void OnGUI()
        {
            EditorGUILayout.LabelField("HexWars ML Lab", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Configure, launch, monitor, reconnect to, and watch headless experiments.", EditorStyles.wordWrappedMiniLabel);
            EditorGUILayout.Space(4);
            _tab = GUILayout.Toolbar(_tab, new[] { "Train", "Arena" });
            EditorGUILayout.Space(4);
            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            if (_tab == 0)
            {
                DrawRunPicker();
                EditorGUILayout.Space(6);
                DrawTrainingForm();
                EditorGUILayout.Space(8);
                DrawControls();
                EditorGUILayout.Space(8);
                DrawStatus();
                EditorGUILayout.Space(8);
                DrawLogs();
            }
            else DrawArena();
            EditorGUILayout.EndScrollView();
        }

        void DrawArena()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Arbitrary model arena", EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                "Each seat may be scripted, a metadata-backed fixed run, or a live run. Live weights refresh only between games.",
                EditorStyles.wordWrappedMiniLabel);
            _arena.Environment = (MlEnvironmentContract)EditorGUILayout.EnumPopup(
                "Environment", _arena.Environment);
            _arena.Observer = (ModelDuelObserverSeat)EditorGUILayout.EnumPopup(
                "Observer", _arena.Observer);
            DrawEnvironmentSummary(MlEnvironmentSummary.ForSelection(_arena.Environment), "Arena preflight");
            DrawArenaSeat("Seat 0", _arena.P0);
            EditorGUILayout.Space(4);
            DrawArenaSeat("Seat 1", _arena.P1);
            EditorGUILayout.Space(5);
            _arena.Seed = EditorGUILayout.IntField("Initial seed", _arena.Seed);
            _arena.SecondsPerAction = EditorGUILayout.FloatField("Seconds per action", _arena.SecondsPerAction);
            _arena.Loop = EditorGUILayout.ToggleLeft("Loop games", _arena.Loop);
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Resolved before launch", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Seat 0", DescribeSeat(_arena.P0), EditorStyles.wordWrappedLabel);
            EditorGUILayout.LabelField("Seat 1", DescribeSeat(_arena.P1), EditorStyles.wordWrappedLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Launch arena", GUILayout.Height(28))) LaunchArena();
            var driver = EditorApplication.isPlaying ? FindAnyObjectByType<ModelDuelDriver>() : null;
            using (new EditorGUI.DisabledScope(driver == null))
            {
                if (GUILayout.Button(driver != null && driver.Paused ? "Resume" : "Pause", GUILayout.Height(28)))
                    driver.SetPaused(!driver.Paused);
                if (GUILayout.Button("Stop", GUILayout.Height(28)))
                {
                    driver.StopDuel();
                    EditorApplication.ExitPlaymode();
                }
            }
            EditorGUILayout.EndHorizontal();
            if (!string.IsNullOrWhiteSpace(_arenaNotice)) EditorGUILayout.HelpBox(_arenaNotice, MessageType.Info);
            if (!string.IsNullOrWhiteSpace(_arenaError)) EditorGUILayout.HelpBox(_arenaError, MessageType.Error);
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);
            DrawArenaRuntime(driver);
        }

        void DrawArenaSeat(string label, ModelSeatConfiguration seat)
        {
            EditorGUILayout.LabelField(label, EditorStyles.boldLabel);
            seat.Kind = (ModelControllerKind)EditorGUILayout.EnumPopup("Controller", seat.Kind);
            if (seat.IsModel)
            {
                EditorGUILayout.BeginHorizontal();
                seat.Path = EditorGUILayout.TextField("Run directory", seat.Path);
                if ((seat.Kind == ModelControllerKind.FixedRun || seat.Kind == ModelControllerKind.LiveRun) &&
                    GUILayout.Button("Use selected", GUILayout.Width(92))) seat.Path = _selectedRun;
                EditorGUILayout.EndHorizontal();
            }
        }

        string DescribeSeat(ModelSeatConfiguration seat)
        {
            if (seat == null) return "missing configuration";
            if (!seat.IsModel) return seat.Kind.ToString();
            if (string.IsNullOrWhiteSpace(seat.Path)) return "model path required";
            string manifest = Path.Combine(seat.Path, "run.json");
            if (!File.Exists(manifest)) return "run metadata not found · " + manifest;
            try
            {
                var data = JsonUtility.FromJson<ArenaRunManifest>(File.ReadAllText(manifest));
                var environment = MlEnvironmentSummary.FromRunManifest(File.ReadAllText(manifest));
                string mode = seat.Kind == ModelControllerKind.LiveRun ? "live (reloads between games)" : "fixed";
                return $"{mode} · {environment.ContractVersion} · {data?.config?.algorithm ?? "unknown algorithm"} · " +
                       $"step {data?.latest_checkpoint_step ?? 0:N0} · {data?.latest_checkpoint ?? "no checkpoint"}";
            }
            catch (Exception error) { return "invalid run metadata · " + error.Message; }
        }

        void LaunchArena()
        {
            _arenaError = string.Empty;
            _arenaNotice = string.Empty;
            var errors = new List<string>(_arena.Validate());
            ValidateSeatFiles(_arena.P0, "Seat 0", errors);
            ValidateSeatFiles(_arena.P1, "Seat 1", errors);
            if (!File.Exists(PythonExe)) errors.Add("Python environment not found: " + PythonExe);
            if (errors.Count > 0) { _arenaError = string.Join("\n", errors); return; }
            ReplayViewerMenu.LaunchDuel(
                PythonDir, _arena.P0.BuildSpec(), _arena.P1.BuildSpec(), _arena.Loop,
                _arena.Seed, _arena.SecondsPerAction, _arena.Environment, _arena.Observer);
            _arenaNotice = "Arena launched. Resolved checkpoints appear here after the policy bridge is ready.";
        }

        static void ValidateSeatFiles(ModelSeatConfiguration seat, string label, List<string> errors)
        {
            if (seat == null || !seat.IsModel || string.IsNullOrWhiteSpace(seat.Path)) return;
            if ((seat.Kind == ModelControllerKind.FixedRun || seat.Kind == ModelControllerKind.LiveRun) &&
                !File.Exists(Path.Combine(seat.Path, "run.json")))
                errors.Add(label + " run.json does not exist: " + seat.Path);
        }

        void DrawArenaRuntime(ModelDuelDriver driver)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Arena runtime", EditorStyles.boldLabel);
            if (driver == null)
            {
                EditorGUILayout.LabelField("Arena is not running.");
                EditorGUILayout.EndVertical();
                return;
            }
            driver.SecondsPerAction = Mathf.Max(0.01f, _arena.SecondsPerAction);
            driver.Loop = _arena.Loop;
            EditorGUILayout.LabelField("State", driver.IsDone ? "stopped" : driver.IsStarting ? "loading models" : driver.Paused ? "paused" : "playing");
            EditorGUILayout.LabelField("Environment", MlEnvironmentContracts.CliValue(driver.Environment));
            EditorGUILayout.LabelField("Seed", driver.Seed.ToString());
            EditorGUILayout.LabelField("Current seat", driver.CurrentSeat.ToString());
            EditorGUILayout.LabelField("Tally (P0 / P1 / Draw)", $"{driver.P0Wins} / {driver.P1Wins} / {driver.Draws}");
            EditorGUILayout.LabelField("Seat 0 loaded", FormatResolved(driver.P0Resolved), EditorStyles.wordWrappedLabel);
            EditorGUILayout.LabelField("Seat 1 loaded", FormatResolved(driver.P1Resolved), EditorStyles.wordWrappedLabel);
            EditorGUILayout.EndVertical();
            Repaint();
        }

        static string FormatResolved(PolicySeatInfo info) => info == null
            ? "scripted or loading"
            : $"{info.Algorithm} · step {info.Step:N0} · {info.Path} · " +
              $"{info.ContractVersion} contract {info.ContractHash}";

        [Serializable] sealed class ArenaRunManifest
        {
            public ArenaRunConfig config;
            public string latest_checkpoint;
            public long latest_checkpoint_step;
        }
        [Serializable] sealed class ArenaRunConfig { public string algorithm; }

        void DrawRunPicker()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Runs", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            using (new EditorGUI.DisabledScope(_knownRuns.Length == 0))
            {
                int selected = EditorGUILayout.Popup("Known local runs", _knownRunIndex, _knownRunLabels);
                if (selected != _knownRunIndex && selected >= 0 && selected < _knownRuns.Length)
                {
                    _knownRunIndex = selected;
                    SelectRun(_knownRuns[selected]);
                }
            }
            if (GUILayout.Button("Refresh", GUILayout.Width(72))) RefreshKnownRuns();
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.SelectableLabel(
                string.IsNullOrWhiteSpace(_selectedRun) ? "No run selected" : _selectedRun,
                EditorStyles.textField, GUILayout.Height(EditorGUIUtility.singleLineHeight));
            EditorGUILayout.EndVertical();
        }

        void DrawTrainingForm()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Training configuration", EditorStyles.boldLabel);
            _resume = EditorGUILayout.ToggleLeft("Resume a metadata-backed run as a new run", _resume);
            if (_resume)
            {
                EditorGUILayout.BeginHorizontal();
                _config.ResumeSource = EditorGUILayout.TextField("Source run", _config.ResumeSource);
                if (GUILayout.Button("Use selected", GUILayout.Width(92))) _config.ResumeSource = _selectedRun;
                EditorGUILayout.EndHorizontal();
                DrawSourceRunScenarioPreflight();
            }
            else
            {
                DrawScenarioEditor();
            }
            _config.RunName = EditorGUILayout.TextField("New run name", _config.RunName);
            _config.Algorithm = (MlAlgorithm)EditorGUILayout.EnumPopup("SB3 algorithm", _config.Algorithm);
            _config.TotalTimesteps = EditorGUILayout.LongField("Target timesteps", _config.TotalTimesteps);
            _config.Seed = EditorGUILayout.IntField("Seed", _config.Seed);
            _config.CheckpointInterval = EditorGUILayout.IntField("Checkpoint interval", _config.CheckpointInterval);
            _config.Workers = EditorGUILayout.IntField("Workers", _config.Workers);
            _config.Device = EditorGUILayout.TextField("Device", _config.Device);
            _config.LearnerSeat = (MlLearnerSeat)EditorGUILayout.EnumPopup("Learner seat", _config.LearnerSeat);
            if (!_resume)
            {
                _config.OpponentKind = (MlOpponentKind)EditorGUILayout.EnumPopup("Opponent", _config.OpponentKind);
                if (_config.OpponentKind == MlOpponentKind.FixedRun || _config.OpponentKind == MlOpponentKind.LiveRun)
                {
                    _config.OpponentPath = EditorGUILayout.TextField("Opponent path", _config.OpponentPath);
                }
            }

            EditorGUILayout.Space(3);
            EditorGUILayout.LabelField("Tracking (local files are always authoritative)", EditorStyles.boldLabel);
            _useTensorBoard = EditorGUILayout.ToggleLeft("TensorBoard", _useTensorBoard);
            _useWandb = EditorGUILayout.ToggleLeft("Weights & Biases", _useWandb);
            if (_useWandb)
            {
                _config.WandbProject = EditorGUILayout.TextField("W&B project", _config.WandbProject);
                _config.WandbEntity = EditorGUILayout.TextField("W&B entity", _config.WandbEntity);
                _config.WandbMode = EditorGUILayout.TextField("W&B mode", _config.WandbMode);
                _config.WandbGroup = EditorGUILayout.TextField("W&B group", _config.WandbGroup);
                _config.WandbUploadArtifacts = EditorGUILayout.ToggleLeft("Upload checkpoint artifacts", _config.WandbUploadArtifacts);
                EditorGUILayout.HelpBox("W&B credentials remain in the environment or W&B login state; ML Lab never saves them.", MessageType.Info);
            }
            _useCustomTracker = EditorGUILayout.ToggleLeft("Custom tracker adapter", _useCustomTracker);
            if (_useCustomTracker)
                _customTrackerAdapter = EditorGUILayout.TextField("module:function", _customTrackerAdapter);

            _showAdvanced = EditorGUILayout.Foldout(_showAdvanced, "Resolved developer paths");
            if (_showAdvanced)
            {
                EditorGUILayout.SelectableLabel("Python: " + PythonExe, EditorStyles.miniLabel, GUILayout.Height(18));
                EditorGUILayout.SelectableLabel("GymServer: " + GymServer, EditorStyles.miniLabel, GUILayout.Height(18));
                EditorGUILayout.SelectableLabel("Runs: " + RunsRoot, EditorStyles.miniLabel, GUILayout.Height(18));
            }
            EditorGUILayout.EndVertical();
        }

        void LoadScenarioLibrary()
        {
            _scenarioSession = MlTrainingScenarioSession.Load(TemplateLibraryPath);
            if (_scenarioSession.CanLaunch)
            {
                _scenarioSession.SelectEnvironment(_config.Environment);
                _config.Environment = _scenarioSession.Environment;
            }
        }

        void DrawScenarioEditor()
        {
            if (_scenarioSession == null) LoadScenarioLibrary();
            if (!string.IsNullOrWhiteSpace(_scenarioSession.LibraryError))
            {
                EditorGUILayout.HelpBox(
                    _scenarioSession.LibraryError, MessageType.Error);
                return;
            }

            MlEnvironmentContract environment =
                (MlEnvironmentContract)EditorGUILayout.EnumPopup(
                    "Environment", _scenarioSession.Environment);
            if (environment != _scenarioSession.Environment)
            {
                try
                {
                    _scenarioSession.SelectEnvironment(environment);
                    _config.Environment = environment;
                }
                catch (Exception error)
                {
                    _state.Fail(error.Message);
                }
            }

            IReadOnlyList<MlTrainingScenario> templates =
                _scenarioSession.AvailableTemplates;
            string[] labels = templates.Select(
                item => item.Name + " (" + item.Id + ")").ToArray();
            int selectedIndex = Math.Max(
                0, Array.FindIndex(
                    templates.ToArray(),
                    item => item.Id == _scenarioSession.SelectedTemplateId));
            int chosenIndex = EditorGUILayout.Popup(
                "Template", selectedIndex, labels);
            if (chosenIndex != selectedIndex &&
                chosenIndex >= 0 && chosenIndex < templates.Count)
            {
                _scenarioSession.SelectTemplate(templates[chosenIndex].Id);
            }

            _showGameSettings = EditorGUILayout.Foldout(
                _showGameSettings, "Advanced game settings");
            if (_showGameSettings) DrawScenarioFields(_scenarioSession.WorkingCopy);

            string saveName = EditorGUILayout.TextField(
                "Template name", _scenarioSession.SaveName);
            string saveId = EditorGUILayout.TextField(
                "Template ID", _scenarioSession.SaveId);
            if (!string.Equals(saveName, _scenarioSession.SaveName, StringComparison.Ordinal) ||
                !string.Equals(saveId, _scenarioSession.SaveId, StringComparison.Ordinal))
            {
                _scenarioSession.SetSaveIdentity(saveName, saveId);
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Save as template"))
            {
                try
                {
                    bool saved = _scenarioSession.SaveAsTemplate();
                    _notice = saved
                        ? "Template saved to " + TemplateLibraryPath
                        : "Template ID already exists. Confirm overwrite inline.";
                }
                catch (Exception error)
                {
                    _state.Fail(
                        TemplateLibraryPath + ": " + error.Message);
                }
            }
            if (GUILayout.Button("Reload templates"))
            {
                try
                {
                    _scenarioSession.ReloadTemplates();
                    _notice = "Templates reloaded from " + TemplateLibraryPath;
                }
                catch (Exception error)
                {
                    LoadScenarioLibrary();
                    _state.Fail(
                        TemplateLibraryPath + ": " + error.Message);
                }
            }
            EditorGUILayout.EndHorizontal();
            if (_scenarioSession.OverwriteArmed &&
                GUILayout.Button("Confirm overwrite"))
            {
                try
                {
                    _scenarioSession.ConfirmOverwrite();
                    _notice = "Template overwritten in " + TemplateLibraryPath;
                }
                catch (Exception error)
                {
                    _state.Fail(
                        TemplateLibraryPath + ": " + error.Message);
                }
            }

            IReadOnlyList<string> errors =
                _scenarioSession.WorkingCopy.Validate();
            foreach (string error in errors)
                EditorGUILayout.HelpBox(error, MessageType.Error);
            if (errors.Count == 0)
            {
                try
                {
                    MlTrainingScenarioPreflight preflight =
                        MlTrainingScenarioPreflight.Create(
                            _scenarioSession.WorkingCopy);
                    EditorGUILayout.LabelField(
                        "Training preflight", EditorStyles.boldLabel);
                    EditorGUILayout.HelpBox(
                        preflight.Describe(
                            OpponentLabel(_config.OpponentKind),
                            LearnerSeatLabel(_config.LearnerSeat)),
                        MessageType.None);
                    if (preflight.LargeScenarioWarning)
                        EditorGUILayout.HelpBox(
                            "Warning: large scenario may reduce headless throughput",
                            MessageType.Warning);
                }
                catch (Exception error)
                {
                    EditorGUILayout.HelpBox(error.Message, MessageType.Error);
                }
            }
        }

        static void DrawScenarioFields(MlTrainingScenario scenario)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Board", EditorStyles.boldLabel);
            scenario.Board.Width = EditorGUILayout.IntField(
                "Width", scenario.Board.Width);
            scenario.Board.Height = EditorGUILayout.IntField(
                "Height", scenario.Board.Height);
            scenario.Board.MaxElevation = EditorGUILayout.IntField(
                "Max elevation", scenario.Board.MaxElevation);
            scenario.Board.ZoneDepth = EditorGUILayout.IntField(
                "Zone depth", scenario.Board.ZoneDepth);
            scenario.Board.FlatChance = EditorGUILayout.FloatField(
                "Flat chance", scenario.Board.FlatChance);
            scenario.Board.PlainsWeight = EditorGUILayout.IntField(
                "Plains weight", scenario.Board.PlainsWeight);
            scenario.Board.ForestWeight = EditorGUILayout.IntField(
                "Forest weight", scenario.Board.ForestWeight);
            scenario.Board.RoughWeight = EditorGUILayout.IntField(
                "Rough weight", scenario.Board.RoughWeight);
            scenario.Board.WaterWeight = EditorGUILayout.IntField(
                "Water weight", scenario.Board.WaterWeight);
            EditorGUILayout.EndVertical();

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Match rules", EditorStyles.boldLabel);
            scenario.Rules.ActionsPerTurn = EditorGUILayout.IntField(
                "Actions per turn", scenario.Rules.ActionsPerTurn);
            scenario.Rules.RoundCap = EditorGUILayout.IntField(
                "Round cap", scenario.Rules.RoundCap);
            scenario.Rules.StartingPoints = EditorGUILayout.IntField(
                "Starting points", scenario.Rules.StartingPoints);
            scenario.Rules.FogOfWar = EditorGUILayout.Toggle(
                "Fog of war", scenario.Rules.FogOfWar);
            scenario.Rules.BiomesEnabled = EditorGUILayout.Toggle(
                "Biomes enabled", scenario.Rules.BiomesEnabled);
            scenario.Rules.BountyRate = EditorGUILayout.FloatField(
                "Bounty rate", scenario.Rules.BountyRate);
            scenario.Rules.DeployCostMultiplier = EditorGUILayout.FloatField(
                "Deploy cost multiplier", scenario.Rules.DeployCostMultiplier);
            scenario.Rules.GeneratorCost = EditorGUILayout.IntField(
                "Generator cost", scenario.Rules.GeneratorCost);
            scenario.Rules.GeneratorOutput = EditorGUILayout.IntField(
                "Generator output", scenario.Rules.GeneratorOutput);
            scenario.Rules.GeneratorHealth = EditorGUILayout.IntField(
                "Generator health", scenario.Rules.GeneratorHealth);
            EditorGUILayout.EndVertical();

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Episode", EditorStyles.boldLabel);
            scenario.Episode.MaxSteps = EditorGUILayout.IntField(
                "Max steps", scenario.Episode.MaxSteps);
            EditorGUILayout.EndVertical();

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField(
                scenario.Environment == MlEnvironmentContract.AdaptiveV1
                    ? "Adaptive reward"
                    : "Tactical reward",
                EditorStyles.boldLabel);
            if (scenario.Environment == MlEnvironmentContract.AdaptiveV1)
            {
                scenario.AdaptiveReward.IntermediateDecisionPenalty =
                    EditorGUILayout.FloatField(
                        "Intermediate decision penalty",
                        scenario.AdaptiveReward.IntermediateDecisionPenalty);
                scenario.AdaptiveReward.DeploymentCompletionBonus =
                    EditorGUILayout.FloatField(
                        "Deployment completion bonus",
                        scenario.AdaptiveReward.DeploymentCompletionBonus);
            }
            else
            {
                scenario.TacticalReward.ShapeScale = EditorGUILayout.FloatField(
                    "Shape scale", scenario.TacticalReward.ShapeScale);
                scenario.TacticalReward.StepPenalty = EditorGUILayout.FloatField(
                    "Step penalty", scenario.TacticalReward.StepPenalty);
                scenario.TacticalReward.ClosingWeight = EditorGUILayout.FloatField(
                    "Closing weight", scenario.TacticalReward.ClosingWeight);
                scenario.TacticalReward.DrawCreditWeight = EditorGUILayout.FloatField(
                    "Draw credit weight", scenario.TacticalReward.DrawCreditWeight);
                scenario.TacticalReward.PointsWeight = EditorGUILayout.FloatField(
                    "Points weight", scenario.TacticalReward.PointsWeight);
            }
            EditorGUILayout.EndVertical();

            if (scenario.Environment == MlEnvironmentContract.AdaptiveV1)
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.LabelField(
                    "Adaptive deployment", EditorStyles.boldLabel);
                scenario.Adaptive.StartingUnitCount = EditorGUILayout.IntField(
                    "Starting unit count", scenario.Adaptive.StartingUnitCount);
                scenario.Adaptive.StartingArmyBudget = EditorGUILayout.IntField(
                    "Starting army budget", scenario.Adaptive.StartingArmyBudget);
                scenario.Adaptive.MaxDesignPointCost = EditorGUILayout.IntField(
                    "Max design point cost", scenario.Adaptive.MaxDesignPointCost);
                EditorGUILayout.EndVertical();
            }
        }

        void DrawSourceRunScenarioPreflight()
        {
            EditorGUILayout.LabelField(
                "Source run scenario", EditorStyles.boldLabel);
            try
            {
                MlTrainingScenarioPreflight preflight =
                    MlTrainingScenarioPreflight.LoadSourceRun(
                        _config.ResumeSource);
                EditorGUILayout.HelpBox(
                    preflight.Describe(
                        "source run", "source run"),
                    MessageType.None);
                if (preflight.LargeScenarioWarning)
                    EditorGUILayout.HelpBox(
                        "Warning: large scenario may reduce headless throughput",
                        MessageType.Warning);
            }
            catch (Exception error)
            {
                string path = string.IsNullOrWhiteSpace(_config.ResumeSource)
                    ? "Source run scenario path unavailable"
                    : Path.Combine(_config.ResumeSource, "scenario.json");
                EditorGUILayout.HelpBox(
                    path + ": " + error.Message, MessageType.Warning);
            }
        }

        static string OpponentLabel(MlOpponentKind opponent)
        {
            switch (opponent)
            {
                case MlOpponentKind.Random: return "Random";
                case MlOpponentKind.FixedRun: return "Fixed run";
                case MlOpponentKind.LiveRun: return "Live run";
                default: return "Greedy";
            }
        }

        static string LearnerSeatLabel(MlLearnerSeat learnerSeat)
        {
            switch (learnerSeat)
            {
                case MlLearnerSeat.Seat0: return "Seat 0";
                case MlLearnerSeat.Seat1: return "Seat 1";
                default: return "Alternating";
            }
        }

        void DrawControls()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Controls", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Doctor")) RunDoctor();
            using (new EditorGUI.DisabledScope(
                       !_resume &&
                       (_scenarioSession == null || !_scenarioSession.CanLaunch)))
            {
                if (GUILayout.Button(_resume ? "Resume" : "Start"))
                    StartTraining(false);
                if (GUILayout.Button("Start & Watch")) StartTraining(true);
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.BeginHorizontal();
            using (new EditorGUI.DisabledScope(string.IsNullOrWhiteSpace(_selectedRun)))
            {
                if (GUILayout.Button("Stop after checkpoint")) RequestStop(false);
                if (GUILayout.Button("Stop now")) RequestStop(true);
                if (GUILayout.Button("Open run folder")) EditorUtility.RevealInFinder(_selectedRun);
            }
            EditorGUILayout.EndHorizontal();
            if (!string.IsNullOrWhiteSpace(_notice))
                EditorGUILayout.HelpBox(_notice, MessageType.Info);
            if (!string.IsNullOrWhiteSpace(_state.Error))
                EditorGUILayout.HelpBox(_state.Error, MessageType.Error);
            EditorGUILayout.EndVertical();
        }

        void DrawStatus()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Live status", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("UI state", _state.Phase.ToString());
            EditorGUILayout.LabelField("Run state", _status == null ? "unknown" : _status.State.ToString());
            EditorGUILayout.LabelField("PID", _status == null || _status.Pid <= 0 ? "-" : _status.Pid.ToString());
            EditorGUILayout.LabelField("Progress", _status == null
                ? "-"
                : string.Format(CultureInfo.InvariantCulture, "{0:N0} / {1:N0}", _status.Step, _status.TargetStep));
            EditorGUILayout.LabelField("Elapsed", RunElapsed());
            EditorGUILayout.LabelField("Throughput", _throughput > 0
                ? _throughput.ToString("N1", CultureInfo.InvariantCulture) + " steps/s"
                : "-");
            EditorGUILayout.LabelField("Latest checkpoint", _status == null || string.IsNullOrWhiteSpace(_status.LatestCheckpoint)
                ? "none"
                : _status.LatestCheckpoint);
            string evaluation = !string.IsNullOrWhiteSpace(_selectedRun)
                ? Path.Combine(_selectedRun, "evaluation.json")
                : string.Empty;
            EditorGUILayout.LabelField("Latest evaluation", File.Exists(evaluation) ? evaluation : "none");
            EditorGUILayout.LabelField("Trackers", _status != null && _status.TrackerDegraded ? "degraded (see run.json/log)" : "healthy or local-only");
            if (!string.IsNullOrWhiteSpace(_selectedRun))
                DrawEnvironmentSummary(LoadRunEnvironmentSummary(_selectedRun), "Run contract");
            if (!string.IsNullOrWhiteSpace(_lastMetricTime))
                EditorGUILayout.LabelField("Last metric", _lastMetricTime);
            EditorGUILayout.EndVertical();
        }

        static MlEnvironmentSummary LoadRunEnvironmentSummary(string runDirectory)
        {
            if (string.IsNullOrWhiteSpace(runDirectory)) return new MlEnvironmentSummary();
            try
            {
                string manifest = Path.Combine(runDirectory, "run.json");
                return File.Exists(manifest)
                    ? MlEnvironmentSummary.FromRunManifest(File.ReadAllText(manifest))
                    : new MlEnvironmentSummary();
            }
            catch (Exception) { return new MlEnvironmentSummary(); }
        }

        static void DrawEnvironmentSummary(MlEnvironmentSummary summary, string label)
        {
            EditorGUILayout.LabelField(label, EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(summary.DisplayText, MessageType.None);
        }

        void DrawLogs()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Bounded live log", EditorStyles.boldLabel);
            if (GUILayout.Button("Clear", GUILayout.Width(60)))
            {
                _training?.Log.Clear();
                _statusQuery?.Log.Clear();
                _command?.Log.Clear();
            }
            EditorGUILayout.EndHorizontal();
            var lines = CombinedLog();
            _logScroll = EditorGUILayout.BeginScrollView(_logScroll, GUILayout.MinHeight(150), GUILayout.MaxHeight(260));
            EditorGUILayout.SelectableLabel(string.Join("\n", lines), EditorStyles.textArea,
                GUILayout.ExpandHeight(true), GUILayout.MinHeight(140));
            EditorGUILayout.EndScrollView();
            EditorGUILayout.EndVertical();
        }

        void StartTraining(bool watch)
        {
            _notice = string.Empty;
            _state.BeginValidation();
            SyncTrackers();
            var errors = new List<string>(_config.Validate());
            if (!_resume)
            {
                if (_scenarioSession == null)
                    errors.Add("Training scenario session is unavailable.");
                else if (!string.IsNullOrWhiteSpace(_scenarioSession.LibraryError))
                    errors.Add(_scenarioSession.LibraryError);
                else if (_scenarioSession.WorkingCopy == null)
                    errors.Add("Training scenario is unavailable.");
                else
                {
                    errors.AddRange(_scenarioSession.WorkingCopy.Validate());
                    _config.Environment =
                        _scenarioSession.WorkingCopy.Environment;
                    try
                    {
                        MlTrainingScenarioPreflight.Create(
                            _scenarioSession.WorkingCopy);
                    }
                    catch (Exception error)
                    {
                        errors.Add(
                            "Training scenario preflight failed: " +
                            error.Message);
                    }
                }
            }
            if (!File.Exists(PythonExe)) errors.Add("Python environment was not found: " + PythonExe);
            if (!File.Exists(CliScript)) errors.Add("ML CLI was not found: " + CliScript);
            if (!File.Exists(GymServer)) errors.Add("Release GymServer was not found: " + GymServer);
            if (_resume && string.IsNullOrWhiteSpace(_config.ResumeSource)) errors.Add("A source run is required to resume.");
            string targetRun = Path.Combine(RunsRoot, _config.RunName ?? string.Empty);
            if (Directory.Exists(targetRun)) errors.Add("Run already exists; choose a new run name. Existing runs are never overwritten.");
            if (errors.Count > 0)
            {
                _state.Fail(string.Join("\n", errors));
                return;
            }

            string args;
            try
            {
                if (_resume)
                {
                    args = _config.BuildResumeArguments();
                }
                else
                {
                    string scenarioPath =
                        MlTrainingScenarioStore.WriteSessionScenario(
                            ProjectRoot, _scenarioSession.WorkingCopy);
                    args = _config.BuildTrainArguments(scenarioPath);
                }
            }
            catch (Exception error)
            {
                _state.Fail(SessionScenarioPath + ": " + error.Message);
                return;
            }
            args += " --runs-root " + MlCliProcess.QuoteArgument(RunsRoot) +
                    " --server " + MlCliProcess.QuoteArgument(GymServer);
            try
            {
                var info = MlCliProcess.BuildDetachedStartInfo(PythonExe, CliScript, args, PythonDir);
                _training.Start(info, targetRun);
                _state.MarkLaunched();
                _selectedRun = targetRun;
                SessionState.SetString(SelectedRunKey, _selectedRun);
                _watchWhenReady = watch;
                _watchLaunched = false;
                _nextPoll = 0;
                RefreshKnownRuns();
                _notice = watch
                    ? "Training started. The Arena viewer will open after the first validated checkpoint."
                    : "Training started headlessly; this window remains attached to local run truth.";
            }
            catch (Exception error) { _state.Fail(error.Message); }
        }

        void RunDoctor()
        {
            if (_command.IsRunning) { _state.Fail("Another ML Lab command is still running."); return; }
            if (!File.Exists(PythonExe) || !File.Exists(CliScript))
            {
                _state.Fail("Python ML environment or hexwars_ml.py is missing.");
                return;
            }
            SyncTrackers();
            string args = _config.BuildDoctorArguments(RunsRoot, GymServer);
            RunCommand(args, "Environment doctor started.", CommandKind.Doctor);
        }

        void RequestStop(bool immediate)
        {
            if (string.IsNullOrWhiteSpace(_selectedRun)) return;
            _state.BeginStopping();
            RunCommand(MlCliProcess.BuildStopArguments(_selectedRun, immediate),
                immediate ? "Immediate stop requested." : "Stop requested after the next validated checkpoint.",
                CommandKind.Control);
            _nextPoll = 0;
        }

        void RunCommand(string args, string notice, CommandKind kind)
        {
            try
            {
                if (_command.IsRunning) { _state.Fail("Another ML Lab command is still running."); return; }
                _command.Start(MlCliProcess.BuildStartInfo(PythonExe, CliScript, args, PythonDir));
                _activeCommand = kind;
                _notice = notice;
            }
            catch (Exception error) { _state.Fail(error.Message); }
        }

        void SyncTrackers()
        {
            _config.Trackers = new List<MlTrackerConfig> { new MlTrackerConfig("local") };
            if (_useTensorBoard) _config.Trackers.Add(new MlTrackerConfig("tensorboard"));
            if (_useWandb) _config.Trackers.Add(new MlTrackerConfig("wandb"));
            if (_useCustomTracker) _config.Trackers.Add(new MlTrackerConfig("custom", _customTrackerAdapter));
        }

        void Tick()
        {
            if (string.IsNullOrWhiteSpace(_selectedRun) || EditorApplication.timeSinceStartup < _nextPoll) return;
            _nextPoll = EditorApplication.timeSinceStartup + PollIntervalSeconds;
            if (_statusQuery != null && !_statusQuery.IsRunning) QueryStatus();
        }

        void QueryStatus()
        {
            if (!Directory.Exists(_selectedRun)) return;
            try
            {
                var info = MlCliProcess.BuildStartInfo(
                    PythonExe, CliScript, MlCliProcess.BuildStatusArguments(_selectedRun), PythonDir);
                _statusQuery.Start(info);
            }
            catch (Exception error) { _state.Fail(error.Message); }
        }

        void OnStatusReceived(MlRunStatus status)
        {
            _status = status;
            if (!status.Ok)
            {
                _state.Fail(string.IsNullOrWhiteSpace(status.Error) ? "Status query failed." : status.Error);
                Repaint();
                return;
            }
            _state.Apply(status.State, status.Pid);
            MlRunAttachment.Remember(_selectedRun);
            ReadLatestMetric();
            if (_watchWhenReady && !_watchLaunched && !string.IsNullOrWhiteSpace(status.LatestCheckpoint))
            {
                _watchLaunched = true;
                EditorApplication.delayCall += () => ReplayViewerMenu.WatchLiveRun(_selectedRun);
            }
            Repaint();
        }

        void OnTrainingExited(int exitCode)
        {
            if (exitCode != 0) _state.Fail("Training process exited with code " + exitCode + ". See the live log and train.log.");
            _nextPoll = 0;
            Repaint();
        }

        void OnCommandExited(int exitCode)
        {
            if (exitCode != 0)
                _state.Fail("ML Lab command exited with code " + exitCode + ". See the command log.");
            else if (_activeCommand == CommandKind.Doctor)
            {
                string json = _command.Log.Lines.LastOrDefault(line => line.StartsWith("{", StringComparison.Ordinal));
                var report = MlDoctorReport.Parse(json ?? string.Empty);
                if (report.Healthy) _notice = report.Summary;
                else _state.Fail(report.Summary);
            }
            _activeCommand = CommandKind.None;
            _nextPoll = 0;
            Repaint();
        }

        void RefreshKnownRuns()
        {
            try
            {
                Directory.CreateDirectory(RunsRoot);
                _knownRuns = Directory.GetDirectories(RunsRoot)
                    .Where(path => File.Exists(Path.Combine(path, "run.json")))
                    .OrderByDescending(path => File.GetLastWriteTimeUtc(Path.Combine(path, "run.json")))
                    .ToArray();
                _knownRunLabels = _knownRuns.Select(Path.GetFileName).ToArray();
                _knownRunIndex = Math.Max(0, Array.IndexOf(_knownRuns, _selectedRun));
            }
            catch (Exception error)
            {
                _knownRuns = Array.Empty<string>();
                _knownRunLabels = Array.Empty<string>();
                _state.Fail("Could not scan runs: " + error.Message);
            }
        }

        void SelectRun(string path)
        {
            _selectedRun = path ?? string.Empty;
            _status = null;
            _throughput = 0;
            _lastMetricTime = string.Empty;
            _state.Reset();
            SessionState.SetString(SelectedRunKey, _selectedRun);
            if (!string.IsNullOrWhiteSpace(_selectedRun)) MlRunAttachment.Remember(_selectedRun);
            _nextPoll = 0;
        }

        void ReadLatestMetric()
        {
            try
            {
                string path = Path.Combine(_selectedRun, "progress.csv");
                string line = MlSharedFileSnapshot.ReadLastNonEmptyLine(path);
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("timestamp", StringComparison.Ordinal)) return;
                string[] fields = line.Split(',');
                if (fields.Length >= 5)
                {
                    double.TryParse(fields[4], NumberStyles.Float, CultureInfo.InvariantCulture, out _throughput);
                    _lastMetricTime = fields[0];
                }
            }
            catch (IOException) { }
        }

        string RunElapsed()
        {
            try
            {
                string manifest = Path.Combine(_selectedRun ?? string.Empty, "run.json");
                if (!File.Exists(manifest)) return "-";
                TimeSpan elapsed = DateTime.UtcNow - File.GetCreationTimeUtc(manifest);
                return string.Format(CultureInfo.InvariantCulture, "{0:00}:{1:00}:{2:00}",
                    (int)elapsed.TotalHours, elapsed.Minutes, elapsed.Seconds);
            }
            catch (Exception) { return "-"; }
        }

        string[] CombinedLog()
        {
            var lines = new List<string>();
            lines.AddRange(MlRunLog.ReadTail(_selectedRun, 500));
            if (_training != null) lines.AddRange(_training.Log.Lines);
            if (_command != null) lines.AddRange(_command.Log.Lines.Select(line => "[command] " + line));
            if (_statusQuery != null)
                lines.AddRange(_statusQuery.Log.Lines.Where(line => line.StartsWith("ERROR:", StringComparison.Ordinal))
                    .Select(line => "[status] " + line));
            return lines.Count == 0 ? new[] { "No output yet." } : lines.TakeLast(500).ToArray();
        }
    }
}

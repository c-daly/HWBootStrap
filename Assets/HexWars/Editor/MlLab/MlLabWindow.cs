using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
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
        [SerializeField] int _tab;
        [SerializeField] ModelDuelConfiguration _arena = new ModelDuelConfiguration();

        MlLabWindowState _state = new MlLabWindowState();
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
            }
            _config.RunName = EditorGUILayout.TextField("New run name", _config.RunName);
            using (new EditorGUI.DisabledScope(_resume))
                _config.Environment = (MlEnvironmentContract)EditorGUILayout.EnumPopup(
                    "Environment", _config.Environment);
            MlEnvironmentSummary trainingSummary = _resume
                ? LoadRunEnvironmentSummary(_config.ResumeSource)
                : MlEnvironmentSummary.ForSelection(_config.Environment);
            DrawEnvironmentSummary(trainingSummary, _resume ? "Source run contract" : "Training preflight");
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

        void DrawControls()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Controls", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Doctor")) RunDoctor();
            if (GUILayout.Button(_resume ? "Resume" : "Start")) StartTraining(false);
            if (GUILayout.Button("Start & Watch")) StartTraining(true);
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

            string args = _resume ? _config.BuildResumeArguments() : _config.BuildTrainArguments();
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

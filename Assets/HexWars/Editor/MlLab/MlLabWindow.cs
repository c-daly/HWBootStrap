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

    public sealed class MlArenaLaunchPlan
    {
        MlArenaLaunchPlan(
            TrainingScenario scenario,
            string p0Spec,
            string p1Spec,
            ModelDuelObserverSeat observer)
        {
            Scenario = scenario;
            P0Spec = p0Spec;
            P1Spec = p1Spec;
            Observer = observer;
        }

        public TrainingScenario Scenario { get; }
        public string P0Spec { get; }
        public string P1Spec { get; }
        // LaunchArena() no longer reads this (the Arena tab's Observer dropdown is gone); kept only
        // because ManualArena_LoadsSelectedRunScenarioWithoutChangingSeatsOrObserver still asserts it.
        public ModelDuelObserverSeat Observer { get; }

        public static MlArenaLaunchPlan Create(ModelDuelConfiguration config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.Environment == MlEnvironmentContract.TacticalV3)
            {
                if (HasDotPathComponent(config.ScenarioRunPath))
                    throw new InvalidOperationException(
                        "tactical-v3 scenario run path must not contain . or .. components.");
                var scenarioErrors = new List<string>();
                RequireContainedRegularFile(
                    config.ScenarioRunPath,
                    Path.Combine(config.ScenarioRunPath, "scenario.json"),
                    "tactical-v3 scenario", scenarioErrors);
                if (scenarioErrors.Count > 0)
                    throw new InvalidOperationException(string.Join("\n", scenarioErrors));
            }
            TrainingScenario scenario = LoadScenario(config);
            ModelDuelContractIdentity expected =
                ModelDuelEnvironmentFactory.ContractIdentity(scenario);
            TacticalV3Contract structuredExpected =
                config.Environment == MlEnvironmentContract.TacticalV3
                    ? TacticalV3Contract.Create(
                        scenario.BuildTacticalV3(), MlEnvironmentKind.Duel)
                    : null;
            var errors = new List<string>();
            ValidateSeatContract(config.P0, "Seat 0", expected, structuredExpected, errors);
            ValidateSeatContract(config.P1, "Seat 1", expected, structuredExpected, errors);
            if (errors.Count > 0)
                throw new InvalidOperationException(string.Join("\n", errors));
            return new MlArenaLaunchPlan(
                scenario,
                config.P0.BuildSpec(),
                config.P1.BuildSpec(),
                config.Observer);
        }

        public static TrainingScenario LoadScenario(ModelDuelConfiguration config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (string.IsNullOrWhiteSpace(config.ScenarioRunPath))
                return TrainingScenario.CreateStandard(
                    MlEnvironmentContracts.CliValue(config.Environment));

            string path = Path.Combine(config.ScenarioRunPath, "scenario.json");
            try
            {
                MlTrainingScenario recorded = MlTrainingScenarioFile.Load(path);
                TrainingScenario scenario =
                    MlTrainingScenarioPreflight.ToEngine(recorded);
                string selected = MlEnvironmentContracts.CliValue(config.Environment);
                if (!string.Equals(scenario.Environment, selected, StringComparison.Ordinal))
                    throw new InvalidOperationException(
                        "scenario environment " + scenario.Environment +
                        " does not match selected environment " + selected + ".");
                return scenario;
            }
            catch (Exception error) when (
                error is ArgumentException ||
                error is IOException ||
                error is InvalidOperationException ||
                error is UnauthorizedAccessException)
            {
                throw new InvalidOperationException(
                    path + ": " + error.Message, error);
            }
        }

        static void ValidateSeatContract(
            ModelSeatConfiguration seat,
            string label,
            ModelDuelContractIdentity expected,
            TacticalV3Contract structuredExpected,
            List<string> errors)
        {
            if (seat == null || !seat.IsModel ||
                string.IsNullOrWhiteSpace(seat.Path))
                return;
            if (structuredExpected != null && HasDotPathComponent(seat.Path))
            {
                errors.Add(label + " run path must not contain . or .. components.");
                return;
            }
            string manifestPath = Path.Combine(seat.Path, "run.json");
            try
            {
                string manifestJson = File.ReadAllText(manifestPath);
                ArenaContractManifest manifest = JsonUtility.FromJson<ArenaContractManifest>(
                    manifestJson);
                if (structuredExpected != null)
                {
                    ValidateStructuredRun(
                        seat.Path, label, manifestPath, manifestJson, manifest,
                        structuredExpected, errors);
                    return;
                }
                ArenaContract contract = manifest?.contract;
                if (contract == null)
                {
                    errors.Add(label + " run metadata is missing contract identity: " +
                        manifestPath);
                    return;
                }
                if (!string.Equals(contract.environment, expected.Environment, StringComparison.Ordinal))
                    errors.Add(label + " environment " + contract.environment +
                        " does not match selected environment " + expected.Environment + ".");
                if (!string.Equals(contract.version, expected.Version, StringComparison.Ordinal))
                    errors.Add(label + " contract " + contract.version +
                        " does not match selected environment " + expected.Version + ".");
                if (!string.Equals(contract.encoding_hash, expected.EncodingHash, StringComparison.Ordinal))
                    errors.Add(label + " encoding hash " + contract.encoding_hash +
                        " does not match expected " + expected.EncodingHash + ".");
            }
            catch (Exception error) when (
                error is ArgumentException ||
                error is InvalidDataException ||
                error is IOException ||
                error is UnauthorizedAccessException)
            {
                errors.Add(label + " run metadata could not be read: " +
                    manifestPath + ": " + error.Message);
            }
        }

        static void RequireContainedRegularFile(
            string runPath, string path, string label, List<string> errors)
        {
            string root = Path.GetFullPath(runPath)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string file = Path.GetFullPath(path);
            string prefix = root + Path.DirectorySeparatorChar;
            if (!file.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                errors.Add(label + " must be contained by the selected run directory.");
                return;
            }
            if (!File.Exists(file))
            {
                errors.Add(label + " does not exist: " + file);
                return;
            }
            if ((File.GetAttributes(file) & FileAttributes.ReparsePoint) != 0)
                errors.Add(label + " must be a regular file, not a reparse point.");

            DirectoryInfo directory = new FileInfo(file).Directory;
            while (directory != null)
            {
                if ((directory.Attributes & FileAttributes.ReparsePoint) != 0)
                    errors.Add(label + " must not traverse a reparse-point directory.");
                if (string.Equals(directory.FullName, root, StringComparison.OrdinalIgnoreCase))
                    return;
                directory = directory.Parent;
            }
            errors.Add(label + " must be contained by the selected run directory.");
        }

        static bool HasDotPathComponent(string path) =>
            !string.IsNullOrWhiteSpace(path) && path.Split(
                new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar },
                StringSplitOptions.RemoveEmptyEntries).Any(
                    part => part == "." || part == "..");

        static void ValidateStructuredRun(
            string runPath, string label, string manifestPath, string manifestJson,
            ArenaContractManifest manifest, TacticalV3Contract expected,
            List<string> errors)
        {
            RequireContainedRegularFile(runPath, manifestPath, label + " run metadata", errors);
            if (manifest == null)
            {
                errors.Add(label + " run metadata is empty: " + manifestPath);
                return;
            }
            if (manifest.schema_version != 2)
                errors.Add(label + " run schema version must be 2.");
            if (!string.Equals(manifest.config?.algorithm, "structured_imitation", StringComparison.Ordinal))
                errors.Add(label + " algorithm must be structured_imitation.");
            if (!string.Equals(manifest.evidence_status, "unsealed-experimental", StringComparison.Ordinal))
                errors.Add(label + " evidence status must be unsealed-experimental.");
            ArenaContract contract = manifest.contract;
            if (contract == null)
            {
                errors.Add(label + " run metadata is missing contract identity: " + manifestPath);
                return;
            }
            if (!string.Equals(contract.environment, "tactical-v3", StringComparison.Ordinal))
                errors.Add(label + " environment must be tactical-v3.");
            if (!string.Equals(contract.version, expected.Version, StringComparison.Ordinal))
                errors.Add(label + " contract version does not match " + expected.Version + ".");
            if (!string.Equals(contract.environment_kind, "duel", StringComparison.Ordinal))
                errors.Add(label + " environment kind must be duel.");
            if (!IsLowerSha256(contract.contract_hash))
                errors.Add(label + " contract hash must be a lowercase SHA-256 hash.");
            if (!string.Equals(contract.encoding_hash, expected.EncodingHash, StringComparison.Ordinal))
                errors.Add(label + " encoding hash does not match the tactical-v3 encoding.");
            if (!string.Equals(contract.capacity_hash, expected.CapacityHash, StringComparison.Ordinal))
                errors.Add(label + " capacity hash does not match the selected scenario.");
            if (MlStrictScenarioJson.ObjectContainsAnyMember(
                    manifestJson, manifestPath, "contract",
                    "observation_size", "action_size"))
                errors.Add(label + " tactical-v3 contract must not declare fixed observation_size or action_size.");
            if (!string.Equals(manifest.latest_checkpoint, "checkpoints/best.pt", StringComparison.Ordinal))
                errors.Add(label + " latest checkpoint must be checkpoints/best.pt.");
            RequireContainedRegularFile(runPath,
                Path.Combine(runPath, "checkpoints", "best.pt"),
                label + " checkpoint", errors);
            if (!string.Equals(
                    manifest.policy_identity, "policy-identity.json",
                    StringComparison.Ordinal))
            {
                errors.Add(label + " policy identity must be policy-identity.json.");
                return;
            }
            string policyPath = Path.Combine(runPath, manifest.policy_identity);
            int errorCount = errors.Count;
            RequireContainedRegularFile(
                runPath, policyPath, label + " policy identity", errors);
            if (errors.Count != errorCount) return;
            MlTacticalV3PolicyIdentity policy =
                MlStrictScenarioJson.ValidatePolicyIdentity(
                    File.ReadAllText(policyPath), policyPath);
            if (!string.Equals(policy.Version, contract.version, StringComparison.Ordinal) ||
                !string.Equals(policy.EnvironmentKind, contract.environment_kind, StringComparison.Ordinal) ||
                !string.Equals(policy.ContractHash, contract.contract_hash, StringComparison.Ordinal) ||
                !string.Equals(policy.EncodingHash, contract.encoding_hash, StringComparison.Ordinal) ||
                !string.Equals(policy.CapacityHash, contract.capacity_hash, StringComparison.Ordinal))
                errors.Add(label + " policy identity does not match run contract.");
        }

        static bool IsLowerSha256(string value) =>
            value != null && value.Length == 64 &&
            value.All(character =>
                (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f'));

        [Serializable] sealed class ArenaContractManifest
        {
            public int schema_version;
            public string evidence_status;
            public ArenaRunConfig config;
            public ArenaContract contract;
            public string policy_identity;
            public string latest_checkpoint;
        }

        [Serializable] sealed class ArenaRunConfig
        {
            public string algorithm;
        }

        [Serializable] sealed class ArenaContract
        {
            public string environment;
            public string version;
            public string environment_kind;
            public string contract_hash;
            public string encoding_hash;
            public string capacity_hash;
            public int observation_size;
            public int action_size;
        }

    }

    public sealed class MlLabWindowState
    {
        public MlLabUiPhase Phase { get; private set; } = MlLabUiPhase.Idle;
        public bool LaunchedHere { get; private set; }
        public string Error { get; private set; } = string.Empty;

        public void BeginValidation() { Phase = MlLabUiPhase.Validating; Error = string.Empty; }
        public void MarkLaunched() { LaunchedHere = true; Phase = MlLabUiPhase.Running; Error = string.Empty; }
        public void BeginStopping() { Phase = MlLabUiPhase.Stopping; Error = string.Empty; }
        public void Fail(string error) { Phase = MlLabUiPhase.Failed; Error = error ?? "Unknown ML Lab error."; }
        public void ClearError() { Error = string.Empty; }
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
            SelectEnvironment(MlEnvironmentContract.TacticalV2);
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
                : environment == MlEnvironmentContract.TacticalV2
                    ? "tactical-v2-standard"
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

        /// <summary>Refreshes the working copy's tactical-v2 template catalog from
        /// <see cref="MlTacticalRosterSource"/>'s current snapshot for <paramref name="localPlayer"/>
        /// (canonical defaults plus that seat's saved custom designs), while preserving the working
        /// copy's own <see cref="MlTrainingTacticalV2.StartingUnitCount"/> — the working scenario, not
        /// live session/cache state, is what drives preflight and launch.</summary>
        public void RefreshTacticalRoster(int localPlayer)
        {
            if (WorkingCopy == null)
                throw new InvalidOperationException(
                    "No working training scenario is selected.");
            if (WorkingCopy.Environment != MlEnvironmentContract.TacticalV2 ||
                WorkingCopy.TacticalV2 == null)
                throw new InvalidOperationException(
                    "Roster refresh is only available for tactical-v2 scenarios.");

            int preservedStartingUnitCount = WorkingCopy.TacticalV2.StartingUnitCount;
            WorkingCopy.TacticalV2.Templates =
                MlTacticalRosterSource.Snapshot(localPlayer).ToList();
            WorkingCopy.TacticalV2.StartingUnitCount = preservedStartingUnitCount;
            WorkingCopy.TacticalV2.MaxControllableUnits = preservedStartingUnitCount;
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
            MlTrainingScenario scenario, int? observationSize, int? actionSize,
            bool usesStructuredCandidates,
            ModelDuelContractIdentity contractIdentity,
            string contractHash = null)
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
            UsesStructuredCandidates = usesStructuredCandidates;
            ContractIdentity = contractIdentity;
            ContractHash = contractHash ?? string.Empty;
            LargeScenarioWarning =
                (long)scenario.Board.Width * scenario.Board.Height > 13L * 9L;
            if (scenario.Environment == MlEnvironmentContract.TacticalV2 &&
                scenario.TacticalV2 != null)
            {
                TacticalV2StartingUnitCount = scenario.TacticalV2.StartingUnitCount;
                TacticalV2ControllableSlots = scenario.TacticalV2.MaxControllableUnits;
                TacticalV2TemplateCount = scenario.TacticalV2.Templates?.Count ?? 0;
            }
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
        public int? ObservationSize { get; }
        public int? ActionSize { get; }
        public bool UsesStructuredCandidates { get; }
        public ModelDuelContractIdentity ContractIdentity { get; }
        public string ContractHash { get; }
        public bool LargeScenarioWarning { get; }
        public int TacticalV2StartingUnitCount { get; }
        public int TacticalV2ControllableSlots { get; }
        public int TacticalV2TemplateCount { get; }
        public string DisplayText => Describe("not selected", "not selected");

        public static MlTrainingScenarioPreflight Create(
            MlTrainingScenario scenario)
        {
            if (scenario == null) throw new ArgumentNullException(nameof(scenario));
            TrainingScenario engineScenario = ToEngine(scenario);
            if (scenario.Environment == MlEnvironmentContract.TacticalV3)
            {
                TacticalV3Contract structuredContract = TacticalV3Contract.Create(
                    engineScenario.BuildTacticalV3(), MlEnvironmentKind.Duel);
                return new MlTrainingScenarioPreflight(
                    scenario, null, null, true,
                    new ModelDuelContractIdentity(
                        structuredContract.Version, structuredContract.Version,
                        structuredContract.EncodingHash, structuredContract.CapacityHash),
                    structuredContract.ContractHash);
            }
            MlContract contract = scenario.Environment == MlEnvironmentContract.AdaptiveV1
                ? MlContract.CreateAdaptive(engineScenario.BuildAdaptive())
                : scenario.Environment == MlEnvironmentContract.TacticalV2
                    ? MlContract.CreateTacticalV2(engineScenario.BuildTacticalV2())
                    : MlContract.Create(engineScenario.BuildTactical());
            return new MlTrainingScenarioPreflight(
                scenario, contract.ObservationSize, contract.ActionSize, false,
                new ModelDuelContractIdentity(
                    contract.Version, contract.Version, contract.EncodingHash));
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
            string geometry = UsesStructuredCandidates
                ? "variable structured candidates\n" +
                  "Contract " + ContractHash +
                  " \u00b7 encoding " + ContractIdentity.EncodingHash +
                  " \u00b7 capacity " + ContractIdentity.CapacityHash
                : "Observation " + ObservationSize +
                  " \u00b7 actions " + ActionSize;
            string text = TemplateName + " \u00b7 " +
                   MlEnvironmentContracts.CliValue(Environment) + "\n" +
                   "Board " + BoardWidth + "\u00d7" + BoardHeight +
                   " \u00b7 zone depth " + ZoneDepth +
                   " \u00b7 actions/turn " + actions + "\n" +
                   "Round cap " + RoundCap +
                   " \u00b7 max steps " + MaxSteps +
                   " \u00b7 fog " + (FogOfWar ? "on" : "off") +
                   " \u00b7 opponent " + opponent +
                   " \u00b7 learner seats " + learnerSeats + "\n" +
                   geometry;
            if (Environment == MlEnvironmentContract.TacticalV2)
                text += "\n" +
                    "Starting units " + TacticalV2StartingUnitCount +
                    " \u00b7 controllable slots " + TacticalV2ControllableSlots +
                    " \u00b7 templates " + TacticalV2TemplateCount + "\n" +
                    "Roster source snapshotted \u00b7 automatic symmetric placement";
            return text;
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
                TacticalV3Reward = scenario.TacticalV3Reward == null
                    ? null
                    : new TrainingTacticalV3RewardConfig
                    {
                        TerminalWin = scenario.TacticalV3Reward.TerminalWin,
                        TerminalNonWin = scenario.TacticalV3Reward.TerminalNonWin,
                        MaterialAdjustmentBound =
                            scenario.TacticalV3Reward.MaterialAdjustmentBound,
                        TimePressureBound = scenario.TacticalV3Reward.TimePressureBound,
                        PointsWeight = scenario.TacticalV3Reward.PointsWeight,
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
                TacticalV2 = scenario.TacticalV2 == null
                    ? null
                    : new TrainingTacticalV2Config
                    {
                        StartingUnitCount = scenario.TacticalV2.StartingUnitCount,
                        MaxControllableUnits = scenario.TacticalV2.MaxControllableUnits,
                        PlacementPolicy = scenario.TacticalV2.PlacementPolicy,
                        StartProfiles = (scenario.TacticalV2.StartProfiles ??
                                new List<MlTrainingTacticalV2StartProfile>())
                            .Select(item => new TacticalV2StartProfile(
                                item.Id,
                                item.LearnerUnitCount,
                                item.OpponentUnitCount,
                                item.Separation))
                            .ToList(),
                        StartDistribution = (scenario.TacticalV2.StartDistribution ??
                                new List<MlTrainingTacticalV2StartWeight>())
                            .Select(item => new TacticalV2StartWeight(
                                item.ProfileId, item.BasisPoints))
                            .ToList(),
                        Templates = (scenario.TacticalV2.Templates ??
                                new List<MlTrainingUnitTemplate>())
                            .Select(item => new TrainingUnitTemplateConfig
                            {
                                Id = item.Id,
                                Name = item.Name,
                                Health = item.Stats.Health,
                                Damage = item.Stats.Damage,
                                Defense = item.Stats.Defense,
                                Movement = item.Stats.Movement,
                                VerticalMovement = item.Stats.VerticalMovement,
                                Range = item.Stats.Range,
                                RangeArc = item.Stats.RangeArc,
                                Vision = item.Stats.Vision,
                                VisionArc = item.Stats.VisionArc,
                            })
                            .ToList(),
                    },
                TacticalV3 = scenario.TacticalV3 == null
                    ? null
                    : new TrainingTacticalV3Config
                    {
                        StartingUnitCount = scenario.TacticalV3.StartingUnitCount,
                        MaxControllableUnits = scenario.TacticalV3.MaxControllableUnits,
                        PlacementPolicy = scenario.TacticalV3.PlacementPolicy,
                        Capacity = scenario.TacticalV3.Capacity == null ? null
                            : new TrainingTacticalV3CapacityConfig
                            {
                                MaxCells = scenario.TacticalV3.Capacity.MaxCells,
                                MaxUnits = scenario.TacticalV3.Capacity.MaxUnits,
                                MaxTemplates = scenario.TacticalV3.Capacity.MaxTemplates,
                                MaxCapabilityDefinitions = scenario.TacticalV3.Capacity.MaxCapabilityDefinitions,
                                MaxCapabilityAllocations = scenario.TacticalV3.Capacity.MaxCapabilityAllocations,
                                MaxRules = scenario.TacticalV3.Capacity.MaxRules,
                                MaxMemoryRecords = scenario.TacticalV3.Capacity.MaxMemoryRecords,
                                MaxRelations = scenario.TacticalV3.Capacity.MaxRelations,
                                MaxCandidates = scenario.TacticalV3.Capacity.MaxCandidates,
                            },
                        StartProfiles = (scenario.TacticalV3.StartProfiles ??
                                new List<MlTrainingTacticalV2StartProfile>())
                            .Select(item => new TacticalV2StartProfile(item.Id,
                                item.LearnerUnitCount, item.OpponentUnitCount,
                                item.Separation)).ToList(),
                        StartDistribution = (scenario.TacticalV3.StartDistribution ??
                                new List<MlTrainingTacticalV2StartWeight>())
                            .Select(item => new TacticalV2StartWeight(
                                item.ProfileId, item.BasisPoints)).ToList(),
                        Templates = (scenario.TacticalV3.Templates ??
                                new List<MlTrainingUnitTemplate>())
                            .Select(item => new TrainingUnitTemplateConfig
                            {
                                Id = item.Id, Name = item.Name,
                                Health = item.Stats.Health, Damage = item.Stats.Damage,
                                Defense = item.Stats.Defense, Movement = item.Stats.Movement,
                                VerticalMovement = item.Stats.VerticalMovement,
                                Range = item.Stats.Range, RangeArc = item.Stats.RangeArc,
                                Vision = item.Stats.Vision, VisionArc = item.Stats.VisionArc,
                            }).ToList(),
                    },
            };
            IReadOnlyList<string> errors = converted.Validate();
            if (errors.Count > 0)
                throw new InvalidDataException(string.Join("; ", errors));
            return converted;
        }
    }

    public sealed class MlTrackerSelectionSnapshot
    {
        readonly bool _useTensorBoard;
        readonly bool _useWandb;
        readonly bool _useCustomTracker;
        readonly string _customTrackerAdapter;

        MlTrackerSelectionSnapshot(
            bool useTensorBoard,
            bool useWandb,
            bool useCustomTracker,
            string customTrackerAdapter)
        {
            _useTensorBoard = useTensorBoard;
            _useWandb = useWandb;
            _useCustomTracker = useCustomTracker;
            _customTrackerAdapter = customTrackerAdapter ?? string.Empty;
        }

        public static MlTrackerSelectionSnapshot Capture(
            bool useTensorBoard,
            bool useWandb,
            bool useCustomTracker,
            string customTrackerAdapter) =>
            new MlTrackerSelectionSnapshot(
                useTensorBoard,
                useWandb,
                useCustomTracker,
                customTrackerAdapter);

        public List<MlTrackerConfig> CreateTrackers()
        {
            var trackers = new List<MlTrackerConfig>
            {
                new MlTrackerConfig("local"),
            };
            if (_useTensorBoard)
                trackers.Add(new MlTrackerConfig("tensorboard"));
            if (_useWandb)
                trackers.Add(new MlTrackerConfig("wandb"));
            if (_useCustomTracker)
                trackers.Add(
                    new MlTrackerConfig(
                        "custom", _customTrackerAdapter));
            return trackers;
        }
    }

    public sealed class MlTrainingLaunchFormState
    {
        MlTrainingLaunchFormState(IReadOnlyList<string> errors)
        {
            Errors = errors;
        }

        public IReadOnlyList<string> Errors { get; }
        public bool CanLaunch => Errors.Count == 0;

        public static MlTrainingLaunchFormState Evaluate(
            MlLabConfig config,
            MlTrainingScenarioSession scenarioSession,
            bool resume,
            MlTrackerSelectionSnapshot trackerSelection = null)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            IReadOnlyList<MlTrackerConfig> trackers =
                trackerSelection == null
                    ? config.Trackers
                    : trackerSelection.CreateTrackers();
            var errors = new List<string>(config.Validate(trackers));
            if (resume)
            {
                if (string.IsNullOrWhiteSpace(config.ResumeSource))
                    errors.Add("A source run is required to resume.");
                return new MlTrainingLaunchFormState(errors);
            }

            if (scenarioSession == null)
            {
                errors.Add("Training scenario session is unavailable.");
            }
            else if (!string.IsNullOrWhiteSpace(
                         scenarioSession.LibraryError))
            {
                errors.Add(scenarioSession.LibraryError);
            }
            else if (scenarioSession.WorkingCopy == null)
            {
                errors.Add("Training scenario is unavailable.");
            }
            else
            {
                errors.AddRange(
                    scenarioSession.WorkingCopy.Validate());
                try
                {
                    MlTrainingScenarioPreflight.Create(
                        scenarioSession.WorkingCopy);
                }
                catch (Exception error)
                {
                    errors.Add(
                        "Training scenario preflight failed: " +
                        error.Message);
                }
            }
            return new MlTrainingLaunchFormState(errors);
        }
    }

    /// <summary>Outcome of <see cref="MlWatchStartPolicy.Decide"/>: whether the Start & Watch
    /// auto-trigger should launch the viewer now, wait and poll again, give up because training ended
    /// (or the safety-net ceiling passed) before a checkpoint became ready, or silently drop a retry
    /// whose target run directory is no longer the current selection.</summary>
    public enum MlWatchStartDecision { WaitAndRetry, Watch, GiveUp, Stale }

    /// <summary>Pure decision for the Start & Watch auto-trigger. run.json is written (with
    /// <c>latest_checkpoint: null</c>) well before the Python trainer publishes its first checkpoint,
    /// so gating on manifest existence alone let <see cref="ReplayViewerMenu.WatchLiveRun"/> launch
    /// against a run that had no checkpoint yet — the policy server then fails closed with "run
    /// manifest is missing latest_checkpoint metadata" (see <c>controllers.py</c>'s
    /// <c>_resolve_run</c>, which requires that exact field). <see cref="Decide"/> instead gates on
    /// <paramref name="checkpointReady"/> the caller has already validated the same way (manifest
    /// exists, <c>latest_checkpoint</c> is a non-blank string naming a checkpoint file that actually
    /// exists). While not ready, this waits and retries for as long as training is still alive
    /// (<paramref name="trainingAlive"/>) rather than failing on a fixed short timeout, since a slow
    /// first checkpoint (long checkpoint interval, slow device) is not an error; a <paramref
    /// name="ceilingDeadlinePassed"/> ceiling remains only as a safety net against a genuinely stuck
    /// wait.</summary>
    public static class MlWatchStartPolicy
    {
        public static MlWatchStartDecision Decide(
            bool pendingRunDirectoryMatchesSelection,
            bool checkpointReady,
            bool trainingAlive,
            bool ceilingDeadlinePassed)
        {
            if (!pendingRunDirectoryMatchesSelection) return MlWatchStartDecision.Stale;
            if (checkpointReady) return MlWatchStartDecision.Watch;
            if (!trainingAlive) return MlWatchStartDecision.GiveUp;
            return ceilingDeadlinePassed ? MlWatchStartDecision.GiveUp : MlWatchStartDecision.WaitAndRetry;
        }
    }

    /// <summary>Outcome of <see cref="MlWatchResumePolicy.Decide"/>: what OnEnable should do about a
    /// Start & Watch retry that may have been mid-flight when a domain reload happened.</summary>
    public enum MlWatchResumeDecision { NoPendingWatch, Resume, ResetToRetryable }

    /// <summary>Pure decision for resuming a Start & Watch retry after a domain reload. The pending
    /// run directory and retry deadline that <see cref="MlLabWindow.AttemptWatch"/> tracks are plain
    /// fields wiped by a script recompile, but <see cref="MlStartAndWatchState.LaunchPending"/> is
    /// [SerializeField] and survives on its own -- so without this, LaunchPending can be stuck true
    /// forever with nothing left to resume it (TryQueue refuses to re-arm, CanRetry never sets, the
    /// Retry button never appears). MlLabWindow persists the pending run directory through
    /// SessionState (same mechanism as its selected-run field) so it can be compared against
    /// LaunchPending here on the next OnEnable.</summary>
    public static class MlWatchResumePolicy
    {
        public static MlWatchResumeDecision Decide(
            bool hasPersistedPendingRunDirectory,
            bool persistedPendingRunDirectoryMatchesSelection,
            bool launchPending)
        {
            if (hasPersistedPendingRunDirectory &&
                persistedPendingRunDirectoryMatchesSelection && launchPending)
                return MlWatchResumeDecision.Resume;
            return launchPending
                ? MlWatchResumeDecision.ResetToRetryable
                : MlWatchResumeDecision.NoPendingWatch;
        }
    }

    [Serializable]
    public sealed class MlStartAndWatchState
    {
        [SerializeField] string _lastAttemptedCheckpoint = string.Empty;
        [SerializeField] bool _requested;
        [SerializeField] bool _launchPending;
        [SerializeField] bool _launched;
        [SerializeField] bool _canRetry;
        [SerializeField] string _presentationStatus = string.Empty;

        public bool Requested => _requested;
        public bool LaunchPending => _launchPending;
        public bool Launched => _launched;
        public bool CanRetry => _canRetry;
        public string PresentationStatus => _presentationStatus;

        public void Begin(bool requested)
        {
            _requested = requested;
            _launchPending = false;
            _launched = false;
            _canRetry = false;
            _presentationStatus = string.Empty;
            _lastAttemptedCheckpoint = string.Empty;
        }

        public bool TryQueue(string latestCheckpoint)
        {
            if (!_requested || _launchPending || _launched ||
                string.IsNullOrWhiteSpace(latestCheckpoint) ||
                string.Equals(
                    latestCheckpoint,
                    _lastAttemptedCheckpoint,
                    StringComparison.Ordinal))
                return false;
            _lastAttemptedCheckpoint = latestCheckpoint;
            _launchPending = true;
            return true;
        }

        public void Retry()
        {
            if (!_canRetry || _launchPending || _launched) return;
            _canRetry = false;
            _lastAttemptedCheckpoint = string.Empty;
        }

        public void Apply(
            MlViewerLaunchResult result, MlLabWindowState uiState)
        {
            if (uiState == null)
                throw new ArgumentNullException(nameof(uiState));
            _launchPending = false;
            if (!result.Success)
            {
                _launched = false;
                _canRetry = true;
                _presentationStatus = string.Empty;
                uiState.Fail(result.Error);
                return;
            }

            _launched = true;
            _canRetry = false;
            uiState.ClearError();
            _presentationStatus =
                $"Scenario {result.Scenario} · schedule {result.SeatSchedule} · " +
                $"learner P{result.LearnerSeat + 1} " +
                $"(seat {result.LearnerSeat}) · opponent {result.Opponent}";
        }

        /// <summary>Recovery for a LaunchPending flag left stuck true by a domain reload that wiped
        /// MlLabWindow's plain-field retry target (see MlWatchResumePolicy). A no-op when nothing is
        /// actually pending, so it never fabricates a Retry affordance that wasn't earned.</summary>
        public void ResetStuckLaunch()
        {
            if (!_launchPending) return;
            _launchPending = false;
            _canRetry = true;
        }
    }

    public sealed class MlLabWindow : EditorWindow
    {
        static readonly MlEnvironmentContract[] TrainEnvironmentValues =
        {
            MlEnvironmentContract.TacticalV1,
            MlEnvironmentContract.AdaptiveV1,
            MlEnvironmentContract.TacticalV2,
        };

        public static IReadOnlyList<MlEnvironmentContract> TrainEnvironmentChoices =>
            Array.AsReadOnly(TrainEnvironmentValues);

        const string SelectedRunKey = "HexWars.MlLab.SelectedRun";
        const string PendingWatchRunDirectoryKey = "HexWars.MlLab.PendingWatchRunDirectory";
        const string PendingWatchDeadlineKey = "HexWars.MlLab.PendingWatchDeadline";
        const double PollIntervalSeconds = 1.0;
        // Start & Watch retries until a validated checkpoint appears (see MlWatchStartPolicy) for as
        // long as training is still alive; this ceiling is only a safety net against a genuinely
        // stuck wait (e.g. training hung without ever transitioning to a terminal state), not the
        // normal way out — a slow first checkpoint is not itself an error.
        const double WatchCeilingTimeoutSeconds = 600.0;

        [SerializeField] MlLabConfig _config = new MlLabConfig();
        [SerializeField] bool _resume;
        [SerializeField] bool _useTensorBoard = true;
        [SerializeField] bool _useWandb;
        [SerializeField] bool _useCustomTracker;
        [SerializeField] string _customTrackerAdapter = string.Empty;
        [SerializeField] bool _showAdvanced;
        [SerializeField] bool _showGameSettings;
        [SerializeField] int _tab;
        [SerializeField] int _tacticalRosterPlayer;
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
        [SerializeField] MlStartAndWatchState _watch =
            new MlStartAndWatchState();
        // Run directory for a Start & Watch attempt currently retrying for run.json to appear; empty
        // when nothing is pending. Keyed to the run directory (not captured via closure) so a stale
        // retry is dropped if the selection moves to a different run mid-wait. Plain fields, so a
        // domain reload wipes them in memory; OnDisable persists them to SessionState (same
        // mechanism as _selectedRun) and OnEnable restores + resumes them, because
        // MlStartAndWatchState.LaunchPending is [SerializeField] and would otherwise survive the
        // reload with nothing left able to resume or retry it. See MlWatchResumePolicy.
        string _pendingWatchRunDirectory = string.Empty;
        double _watchRetryDeadline;
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
            RestorePendingWatch();
            EditorApplication.update += Tick;
            _nextPoll = 0;
        }

        void OnDisable()
        {
            EditorApplication.update -= Tick;
            SessionState.SetString(SelectedRunKey, _selectedRun ?? string.Empty);
            PersistPendingWatch();
            DisposeProcessOwners();
        }

        // Restores a Start & Watch retry that may have been mid-flight across a domain reload (see
        // the field comment on _pendingWatchRunDirectory) and either resumes it, recovers a stuck
        // LaunchPending into a retryable state, or does nothing if there was never anything pending.
        void RestorePendingWatch()
        {
            _pendingWatchRunDirectory = SessionState.GetString(PendingWatchRunDirectoryKey, string.Empty);
            _watchRetryDeadline = ParseTimeSinceStartup(
                SessionState.GetString(PendingWatchDeadlineKey, string.Empty));
            bool hasPersistedPending = !string.IsNullOrEmpty(_pendingWatchRunDirectory);
            bool matchesSelection = hasPersistedPending &&
                string.Equals(_pendingWatchRunDirectory, _selectedRun, StringComparison.Ordinal);

            switch (MlWatchResumePolicy.Decide(hasPersistedPending, matchesSelection, _watch.LaunchPending))
            {
                case MlWatchResumeDecision.Resume:
                    // Defer past OnEnable: WatchLiveRun can create a scene / enter Play Mode, which is
                    // not safe to do synchronously while the editor is still finishing a domain reload.
                    EditorApplication.delayCall += AttemptWatch;
                    break;

                case MlWatchResumeDecision.ResetToRetryable:
                    _pendingWatchRunDirectory = string.Empty;
                    SessionState.EraseString(PendingWatchRunDirectoryKey);
                    SessionState.EraseString(PendingWatchDeadlineKey);
                    _watch.ResetStuckLaunch();
                    _notice =
                        "A pending Start & Watch retry was interrupted by a script reload. " +
                        "Click Retry viewer to try again.";
                    break;

                case MlWatchResumeDecision.NoPendingWatch:
                    _pendingWatchRunDirectory = string.Empty;
                    SessionState.EraseString(PendingWatchRunDirectoryKey);
                    SessionState.EraseString(PendingWatchDeadlineKey);
                    break;
            }
        }

        void PersistPendingWatch()
        {
            if (string.IsNullOrEmpty(_pendingWatchRunDirectory))
            {
                SessionState.EraseString(PendingWatchRunDirectoryKey);
                SessionState.EraseString(PendingWatchDeadlineKey);
                return;
            }
            SessionState.SetString(PendingWatchRunDirectoryKey, _pendingWatchRunDirectory);
            SessionState.SetString(
                PendingWatchDeadlineKey,
                _watchRetryDeadline.ToString("R", CultureInfo.InvariantCulture));
        }

        static double ParseTimeSinceStartup(string text) =>
            double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value)
                ? value
                : 0.0;

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
            // No "Observer" control here anymore (review-fix pass): presentation is always omniscient
            // (BoardRenderer.RenderEntities/InitializeBoard pass viewer: null), so an observer seat had
            // no effect on what the Arena tab actually showed — see ModelDuelConfiguration.Observer.
            EditorGUILayout.BeginHorizontal();
            _arena.ScenarioRunPath = EditorGUILayout.TextField(
                "Scenario run", _arena.ScenarioRunPath);
            if (GUILayout.Button("Use selected run scenario", GUILayout.Width(174)))
                _arena.ScenarioRunPath = _selectedRun;
            EditorGUILayout.EndHorizontal();
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
            MlArenaLaunchPlan plan;
            try
            {
                plan = MlArenaLaunchPlan.Create(_arena);
            }
            catch (Exception error)
            {
                _arenaError = error.Message;
                return;
            }
            ReplayViewerMenu.LaunchDuel(
                PythonDir, plan.P0Spec, plan.P1Spec, _arena.Loop,
                _arena.Seed, _arena.SecondsPerAction, _arena.Environment,
                plan.Scenario);
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
            UpdateArenaTrainerLivenessStatus(driver);
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

        // D2 ("Lab stops lying about trainers"): while the Arena is spectating a live run, feed the
        // same process+file trainer-liveness truth the Train tab's status line uses into the Arena
        // identity rows (ModelDuelDriver.MarkTrainerLivenessStatus -> IdentitySnapshot -> the on-screen
        // overlay) -- an honest "trainer exited"/"trainer stalled" line beside a live seat, without
        // pausing or resetting playback. A no-op when neither seat is configured as a live run.
        void UpdateArenaTrainerLivenessStatus(ModelDuelDriver driver)
        {
            string liveRunDirectory =
                _arena.P0.Kind == ModelControllerKind.LiveRun ? _arena.P0.Path :
                _arena.P1.Kind == ModelControllerKind.LiveRun ? _arena.P1.Path :
                string.Empty;
            if (string.IsNullOrWhiteSpace(liveRunDirectory)) return;
            ComputeTrainerLiveness(liveRunDirectory, out bool confirmedExited, out double minutesSinceProgress);
            (long step, long targetStep) = ReadRunProgress(liveRunDirectory);
            driver.MarkTrainerLivenessStatus(
                MlTrainerStatusFormatter.Describe(confirmedExited, minutesSinceProgress, step, targetStep));
        }

        static (long Step, long TargetStep) ReadRunProgress(string runDirectory)
        {
            try
            {
                string manifestPath = Path.Combine(runDirectory, "run.json");
                if (!File.Exists(manifestPath)) return (0, 0);
                var manifest = JsonUtility.FromJson<ArenaRunManifest>(File.ReadAllText(manifestPath));
                return (manifest?.latest_checkpoint_step ?? 0, manifest?.config?.total_timesteps ?? 0);
            }
            catch (Exception) { return (0, 0); }
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
        [Serializable] sealed class ArenaRunConfig { public string algorithm; public long total_timesteps; }

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

            int environmentIndex = Math.Max(0,
                Array.IndexOf(TrainEnvironmentValues, _scenarioSession.Environment));
            environmentIndex = EditorGUILayout.Popup(
                "Environment", environmentIndex,
                TrainEnvironmentValues.Select(MlEnvironmentContracts.CliValue).ToArray());
            MlEnvironmentContract environment = TrainEnvironmentValues[environmentIndex];
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

        void DrawScenarioFields(MlTrainingScenario scenario)
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
            scenario.Board.FlatChance = EditorGUILayout.DoubleField(
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
            scenario.Rules.BountyRate = EditorGUILayout.DoubleField(
                "Bounty rate", scenario.Rules.BountyRate);
            scenario.Rules.DeployCostMultiplier = EditorGUILayout.DoubleField(
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

            // Exhaustive over the three environments: tactical-v2 gets the configurable roster/count
            // box, tactical-v1 a read-only legacy notice, adaptive-v1 keeps its existing deployment
            // fields. Never fall through an unhandled environment to the tactical-v1 case.
            if (scenario.Environment == MlEnvironmentContract.TacticalV2)
            {
                DrawTacticalV2Setup(scenario);
            }
            else if (scenario.Environment == MlEnvironmentContract.TacticalV1)
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.LabelField("Tactical setup", EditorStyles.boldLabel);
                EditorGUILayout.HelpBox(
                    "tactical-v1 uses a fixed legacy roster; starting unit count and roster " +
                    "source are not configurable.",
                    MessageType.None);
                EditorGUILayout.EndVertical();
            }
            else if (scenario.Environment == MlEnvironmentContract.AdaptiveV1)
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

        static readonly string[] TacticalRosterPlayerLabels = { "Local player 1", "Local player 2" };

        void DrawTacticalV2Setup(MlTrainingScenario scenario)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Tactical setup", EditorStyles.boldLabel);
            _tacticalRosterPlayer = EditorGUILayout.Popup(
                "Roster source", _tacticalRosterPlayer, TacticalRosterPlayerLabels);
            int startingUnitCount = EditorGUILayout.IntSlider(
                "Starting unit count", scenario.TacticalV2.StartingUnitCount, 1, 12);
            scenario.TacticalV2.StartingUnitCount = startingUnitCount;
            scenario.TacticalV2.MaxControllableUnits = startingUnitCount;
            List<MlTrainingUnitTemplate> templates =
                scenario.TacticalV2.Templates ?? new List<MlTrainingUnitTemplate>();
            EditorGUILayout.LabelField(
                "Available templates",
                templates.Count.ToString(CultureInfo.InvariantCulture));
            EditorGUILayout.LabelField(
                templates.Count == 0
                    ? "No templates available."
                    : string.Join(", ", templates.Select(item => item.Name)),
                EditorStyles.wordWrappedLabel);
            if (GUILayout.Button("Refresh saved roster"))
            {
                try
                {
                    _scenarioSession.RefreshTacticalRoster(_tacticalRosterPlayer);
                    _notice = "Roster refreshed from local player " +
                        (_tacticalRosterPlayer + 1) + "'s saved templates.";
                }
                catch (Exception error)
                {
                    _state.Fail(error.Message);
                }
            }
            EditorGUILayout.EndVertical();
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
            MlTrackerSelectionSnapshot trackerSelection =
                CaptureTrackerSelection();
            MlTrainingLaunchFormState formState =
                MlTrainingLaunchFormState.Evaluate(
                    _config,
                    _scenarioSession,
                    _resume,
                    trackerSelection);
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Controls", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Doctor")) RunDoctor();
            using (new EditorGUI.DisabledScope(!formState.CanLaunch))
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
            if (_watch.CanRetry && GUILayout.Button("Retry viewer"))
            {
                _watch.Retry();
                _state.ClearError();
                if (_status != null && _status.Ok)
                    _state.Apply(_status.State, _status.Pid);
                _nextPoll = 0;
                _notice =
                    "Viewer retry requested; the selected run and checkpoint " +
                    "will be validated again.";
            }
            foreach (string error in formState.Errors)
                EditorGUILayout.HelpBox(error, MessageType.Error);
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
            EditorGUILayout.LabelField("Learner episodes (Seat 0 / Seat 1)", _status == null
                ? "-"
                : string.Format(CultureInfo.InvariantCulture, "{0:N0} / {1:N0}",
                    _status.Seat0Episodes, _status.Seat1Episodes));
            if (_status != null && _status.SeatAuditShowsWarning)
                EditorGUILayout.HelpBox(_status.SeatAuditWarning, MessageType.Warning);
            else if (_status != null && _status.SeatAuditShowsInfo)
                EditorGUILayout.HelpBox(
                    !_status.SeatAuditReadable
                        ? (string.IsNullOrWhiteSpace(_status.SeatAuditWarning)
                            ? "Learner seat audit is not readable yet."
                            : _status.SeatAuditWarning)
                        : "Learner seat balance is still in progress; counts can differ while workers finish episodes.",
                    MessageType.Info);
            EditorGUILayout.LabelField("Elapsed", RunElapsed());
            EditorGUILayout.LabelField("Throughput", _throughput > 0
                ? _throughput.ToString("N1", CultureInfo.InvariantCulture) + " steps/s"
                : "-");
            EditorGUILayout.LabelField("Latest checkpoint", _status == null || string.IsNullOrWhiteSpace(_status.LatestCheckpoint)
                ? "none"
                : _status.LatestCheckpoint);
            if (!string.IsNullOrWhiteSpace(_watch.PresentationStatus))
                EditorGUILayout.LabelField(
                    "Viewer presentation", _watch.PresentationStatus);
            string evaluation = !string.IsNullOrWhiteSpace(_selectedRun)
                ? Path.Combine(_selectedRun, "evaluation.json")
                : string.Empty;
            EditorGUILayout.LabelField("Latest evaluation", File.Exists(evaluation) ? evaluation : "none");
            EditorGUILayout.LabelField("Trackers", _status != null && _status.TrackerDegraded ? "degraded (see run.json/log)" : "healthy or local-only");
            if (!string.IsNullOrWhiteSpace(_selectedRun))
            {
                // D2 ("Lab stops lying about trainers"): an honest status line, derived from process
                // liveness + progress.csv staleness -- never a modal, never interrupts anything, just
                // surfaced text so a dead or hung trainer isn't only discoverable via TensorBoard.
                ComputeTrainerLiveness(_selectedRun, out bool confirmedExited, out double minutesSinceProgress);
                string trainerStatus = MlTrainerStatusFormatter.Describe(
                    confirmedExited, minutesSinceProgress, _status?.Step ?? 0, _status?.TargetStep ?? 0);
                if (!string.IsNullOrWhiteSpace(trainerStatus))
                    EditorGUILayout.HelpBox(trainerStatus, MessageType.Warning);
                DrawEnvironmentSummary(LoadRunEnvironmentSummary(_selectedRun), "Run contract");
            }
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
            MlTrackerSelectionSnapshot trackerSelection =
                CaptureTrackerSelection();
            var errors = new List<string>(
                MlTrainingLaunchFormState.Evaluate(
                    _config,
                    _scenarioSession,
                    _resume,
                    trackerSelection).Errors);
            if (!_resume && _scenarioSession?.WorkingCopy != null)
                _config.Environment =
                    _scenarioSession.WorkingCopy.Environment;
            if (!File.Exists(PythonExe)) errors.Add("Python environment was not found: " + PythonExe);
            if (!File.Exists(CliScript)) errors.Add("ML CLI was not found: " + CliScript);
            if (!File.Exists(GymServer)) errors.Add("Release GymServer was not found: " + GymServer);
            string targetRun = Path.Combine(RunsRoot, _config.RunName ?? string.Empty);
            if (Directory.Exists(targetRun)) errors.Add("Run already exists; choose a new run name. Existing runs are never overwritten.");
            if (errors.Count > 0)
            {
                _state.Fail(string.Join("\n", errors));
                return;
            }
            SyncTrackers(trackerSelection);

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
                _watch.Begin(watch);
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
            SyncTrackers(CaptureTrackerSelection());
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

        MlTrackerSelectionSnapshot CaptureTrackerSelection() =>
            MlTrackerSelectionSnapshot.Capture(
                _useTensorBoard,
                _useWandb,
                _useCustomTracker,
                _customTrackerAdapter);

        void SyncTrackers(
            MlTrackerSelectionSnapshot trackerSelection)
        {
            _config.Trackers = trackerSelection.CreateTrackers();
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
            if (_watch.TryQueue(status.LatestCheckpoint))
            {
                _pendingWatchRunDirectory = _selectedRun;
                _watchRetryDeadline =
                    EditorApplication.timeSinceStartup + WatchCeilingTimeoutSeconds;
                AttemptWatch();
            }
            Repaint();
        }

        // Retries the Start & Watch launch until a validated checkpoint is ready, training is no
        // longer alive, the safety-net ceiling passes, or the pending run directory is superseded by a
        // different selection. See MlWatchStartPolicy for the pure decision; this method is just the
        // editor-main-thread plumbing (file reads, live training/UI state) around it.
        void AttemptWatch()
        {
            if (this == null || string.IsNullOrEmpty(_pendingWatchRunDirectory)) return;
            string runDirectory = _pendingWatchRunDirectory;
            bool matchesSelection = string.Equals(runDirectory, _selectedRun, StringComparison.Ordinal);
            bool checkpointReady = RunHasValidatedCheckpoint(runDirectory);
            bool trainingAlive = MlTrainerLivenessPolicy.IsAlive(
                ComputeTrainerLiveness(runDirectory, out _, out _));
            bool ceilingPassed = EditorApplication.timeSinceStartup >= _watchRetryDeadline;

            switch (MlWatchStartPolicy.Decide(matchesSelection, checkpointReady, trainingAlive, ceilingPassed))
            {
                case MlWatchStartDecision.Stale:
                    _pendingWatchRunDirectory = string.Empty;
                    break;

                case MlWatchStartDecision.WaitAndRetry:
                    EditorApplication.delayCall += AttemptWatch;
                    break;

                case MlWatchStartDecision.Watch:
                    _pendingWatchRunDirectory = string.Empty;
                    MlViewerLaunchResult result = ReplayViewerMenu.WatchLiveRun(runDirectory);
                    _watch.Apply(result, _state);
                    if (result.Success)
                        _notice = "Arena viewer launched. " + _watch.PresentationStatus;
                    Repaint();
                    break;

                case MlWatchStartDecision.GiveUp:
                    _pendingWatchRunDirectory = string.Empty;
                    _watch.Apply(
                        MlViewerLaunchResult.Failed(trainingAlive
                            ? "Start & Watch gave up: no validated checkpoint appeared within " +
                              (WatchCeilingTimeoutSeconds / 60.0).ToString("0", CultureInfo.InvariantCulture) +
                              " minute(s) of the first status."
                            : "Start & Watch gave up: training ended before a validated checkpoint appeared."),
                        _state);
                    Repaint();
                    break;
            }
        }

        // Mirrors python/ml_lab/controllers.py's ControllerResolver._resolve_run check (the exact
        // check that raises "run manifest is missing latest_checkpoint metadata"): the manifest must
        // exist and its latest_checkpoint must be a non-blank string naming a checkpoint file that
        // actually exists in the run directory. Checking this here — not just run.json's existence —
        // is what closes the race that let Start & Watch launch a viewer before the policy server had
        // anything loadable to serve.
        static bool RunHasValidatedCheckpoint(string runDirectory)
        {
            string manifestPath = Path.Combine(runDirectory, "run.json");
            if (!File.Exists(manifestPath)) return false;
            try
            {
                var manifest = JsonUtility.FromJson<ArenaRunManifest>(File.ReadAllText(manifestPath));
                if (manifest == null || string.IsNullOrWhiteSpace(manifest.latest_checkpoint)) return false;
                return File.Exists(Path.Combine(runDirectory, manifest.latest_checkpoint));
            }
            catch (Exception) { return false; }
        }

        // Ground truth for D1 ("Lab stops lying about trainers"): whether the trainer tracked for
        // runDirectory is actually still alive, independent of _state.Phase (see
        // MlTrainerLivenessPolicy's class comment for why that ephemeral phase is not trustworthy on
        // its own). Reattaches via the PID persisted at launch time (MlRunAttachment.RememberProcess),
        // guarded against PID reuse by an existence+name check, and falls back to progress.csv mtime
        // freshness whenever no valid process handle is available.
        MlTrainerLivenessState ComputeTrainerLiveness(
            string runDirectory, out bool confirmedExited, out double minutesSinceProgress)
        {
            MlRunAttachment attachment = MlRunAttachment.Restore();
            bool hasPersistedPid = attachment.HasPid && !string.IsNullOrEmpty(runDirectory) &&
                string.Equals(attachment.RunDirectory, runDirectory, StringComparison.Ordinal);
            bool processExists = false;
            bool processIdentityMatches = false;
            if (hasPersistedPid &&
                MlTrainerProcessLookup.TryGetRunningProcessName(attachment.Pid, out string processName))
            {
                processExists = true;
                processIdentityMatches = MlTrainerProcessLookup.MatchesExpectedExecutable(processName, PythonExe);
            }
            minutesSinceProgress = MinutesSinceProgressUpdate(runDirectory);
            bool progressFresh = MlTrainerProgressFreshness.IsFresh(minutesSinceProgress);
            confirmedExited = hasPersistedPid && !(processExists && processIdentityMatches);
            return MlTrainerLivenessPolicy.Decide(
                hasPersistedPid, processExists, processIdentityMatches, progressFresh);
        }

        // No progress.csv yet is not evidence of a stall (training may not have written its first
        // metrics row); treated as fresh so a brand-new run is never mistaken for stalled.
        static double MinutesSinceProgressUpdate(string runDirectory)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(runDirectory)) return 0.0;
                string path = Path.Combine(runDirectory, "progress.csv");
                if (!File.Exists(path)) return 0.0;
                return (DateTime.UtcNow - File.GetLastWriteTimeUtc(path)).TotalMinutes;
            }
            catch (IOException) { return 0.0; }
            catch (UnauthorizedAccessException) { return 0.0; }
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

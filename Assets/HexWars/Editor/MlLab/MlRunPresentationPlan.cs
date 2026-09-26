using System;
using System.Collections.Generic;
using System.IO;
using HexWars.Engine.Rl;
using HexWars.Presentation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace HexWars.Presentation.EditorTools.MlLab
{
    public sealed class MlRunPresentationPlan
    {
        const string OutcomeAlgorithm = "structured_policy_gradient";

        readonly string _learnerSpec;
        readonly string _learnerSeat;
        readonly IReadOnlyList<OpponentPlan> _opponents;

        MlRunPresentationPlan(
            string runDirectory,
            string learnerRunDirectory,
            string learnerSpec,
            string learnerSeat,
            IReadOnlyList<OpponentPlan> opponents,
            TrainingScenario scenario)
        {
            RunDirectory = runDirectory;
            LearnerRunDirectory = learnerRunDirectory;
            _learnerSpec = learnerSpec;
            _learnerSeat = learnerSeat;
            _opponents = opponents;
            Scenario = scenario;
        }

        public string RunDirectory { get; }
        public string LearnerRunDirectory { get; }
        public TrainingScenario Scenario { get; }
        public string LearnerSeatSchedule => _learnerSeat;

        public static MlRunPresentationPlan Load(string runDirectory)
        {
            string runPath;
            try
            {
                if (string.IsNullOrWhiteSpace(runDirectory))
                    throw new InvalidDataException("run directory is required");
                runPath = Path.GetFullPath(runDirectory);
            }
            catch (Exception error) when (
                error is ArgumentException ||
                error is NotSupportedException ||
                error is PathTooLongException ||
                error is InvalidDataException)
            {
                throw new InvalidOperationException(
                    (runDirectory ?? "<null>") + ": " + error.Message, error);
            }

            string manifestPath = Path.Combine(runPath, "run.json");
            try
            {
                if (!File.Exists(manifestPath))
                    throw new InvalidDataException("run manifest does not exist");
                string json = File.ReadAllText(manifestPath);
                bool hasScenarioMetadata = ValidateRawManifest(json);
                RunManifestDto manifest = JsonUtility.FromJson<RunManifestDto>(json);
                if (manifest == null)
                    throw new InvalidDataException("run manifest must be a JSON object");
                if (manifest.schema_version != 1)
                    throw new InvalidDataException("schema_version must be 1");
                string learnerSeat = manifest.config?.learner_seat;
                if (learnerSeat != "0" &&
                    learnerSeat != "1" &&
                    learnerSeat != "alternating")
                    throw new InvalidDataException(
                        "config.learner_seat must be '0', '1', or 'alternating'");
                if (manifest.opponent_snapshot == null)
                    throw new InvalidDataException("opponent_snapshot is required");

                TrainingScenario scenario = LoadScenario(
                    runPath, manifest, hasScenarioMetadata);
                if (string.Equals(
                        manifest.config?.algorithm,
                        "structured_dagger",
                        StringComparison.Ordinal))
                {
                    return LoadStructuredLifecycle(
                        runPath, manifest, learnerSeat, scenario);
                }
                if (string.Equals(
                        manifest.config?.algorithm,
                        OutcomeAlgorithm,
                        StringComparison.Ordinal))
                {
                    return LoadOutcomeCandidate(
                        runPath, manifest, learnerSeat, scenario);
                }
                ContractDto learnerContract = manifest.contract;
                IReadOnlyList<OpponentPlan> opponents = ResolveOpponents(
                    manifest.opponent_snapshot, learnerContract, "opponent_snapshot");
                return new MlRunPresentationPlan(
                    runPath,
                    runPath,
                    ReplayViewerMenu.BuildLiveTrainingSpec(runPath),
                    learnerSeat,
                    opponents,
                    scenario);
            }
            catch (InvalidOperationException error)
            {
                throw new InvalidOperationException(
                    manifestPath + ": " + error.Message, error);
            }
            catch (Exception error) when (
                error is ArgumentException ||
                error is InvalidDataException ||
                error is IOException ||
                error is UnauthorizedAccessException)
            {
                throw new InvalidOperationException(
                    manifestPath + ": " + error.Message, error);
            }
        }

        /// <summary>
        /// A tactical-v3 DAgger directory describes a training lifecycle, not a controller the
        /// policy server can load.  Presentation deliberately keeps the lifecycle's recorded
        /// scenario, seat schedule, and opponent, while using its declared published strict model,
        /// or its authenticated initialization source only before publication. MlArenaLaunchPlan.Create
        /// supplies the exact same structured-model/scenario validation used by the Arena tab.
        /// </summary>
        static MlRunPresentationPlan LoadStructuredLifecycle(
            string runPath,
            RunManifestDto manifest,
            string learnerSeat,
            TrainingScenario scenario)
        {
            if (!string.Equals(
                    scenario.Environment, "tactical-v3", StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "structured_dagger presentation requires a tactical-v3 scenario");
            ModelSeatConfiguration opponent = ResolveStructuredOpponent(
                manifest.opponent_snapshot, "opponent_snapshot", out string opponentLabel);
            string candidate = MlArenaLaunchPlan.ResolveModelRunSelection(runPath);
            if (SamePath(candidate, runPath))
                throw new InvalidOperationException(
                    "structured_dagger presentation has no valid published or source model");
            try
            {
                var learner = new ModelSeatConfiguration
                {
                    Kind = ModelControllerKind.LiveRun,
                    Path = candidate,
                    InferenceMode = ModelInferenceMode.Deterministic,
                };
                var arena = new ModelDuelConfiguration
                {
                    Environment = MlEnvironmentContract.TacticalV3,
                    ScenarioRunPath = runPath,
                    P0 = learner,
                    P1 = opponent,
                };
                MlArenaLaunchPlan launch = MlArenaLaunchPlan.Create(arena);
                return new MlRunPresentationPlan(
                    runPath,
                    candidate,
                    launch.P0Spec,
                    learnerSeat,
                    new[] { new OpponentPlan(launch.P1Spec, opponentLabel) },
                    launch.Scenario);
            }
            catch (Exception error) when (
                error is ArgumentException ||
                error is InvalidDataException ||
                error is IOException ||
                error is InvalidOperationException ||
                error is UnauthorizedAccessException)
            {
                throw new InvalidOperationException(
                    candidate + ": " + error.Message, error);
            }
        }

        /// <summary>
        /// An outcome candidate is itself an inference-bearing tactical-v3 run. Present its own
        /// latest validated checkpoint in live mode, while retaining its recorded scenario, seat
        /// schedule, and opponent. Unlike a DAgger lifecycle, it must never resolve through an
        /// initialization source or a separately published model.
        /// </summary>
        static MlRunPresentationPlan LoadOutcomeCandidate(
            string runPath,
            RunManifestDto manifest,
            string learnerSeat,
            TrainingScenario scenario)
        {
            if (!string.Equals(
                    scenario.Environment, "tactical-v3", StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    OutcomeAlgorithm +
                    " presentation requires a tactical-v3 scenario");
            }
            ModelSeatConfiguration opponent = ResolveStructuredOpponent(
                manifest.opponent_snapshot,
                "opponent_snapshot",
                OutcomeAlgorithm,
                out string opponentLabel);
            try
            {
                var learner = new ModelSeatConfiguration
                {
                    Kind = ModelControllerKind.LiveRun,
                    Path = runPath,
                    InferenceMode = ModelInferenceMode.Deterministic,
                };
                var arena = new ModelDuelConfiguration
                {
                    Environment = MlEnvironmentContract.TacticalV3,
                    ScenarioRunPath = runPath,
                    P0 = learner,
                    P1 = opponent,
                };
                MlArenaLaunchPlan launch = MlArenaLaunchPlan.Create(arena);
                return new MlRunPresentationPlan(
                    runPath,
                    runPath,
                    launch.P0Spec,
                    learnerSeat,
                    new[] { new OpponentPlan(launch.P1Spec, opponentLabel) },
                    launch.Scenario);
            }
            catch (Exception error) when (
                error is ArgumentException ||
                error is InvalidDataException ||
                error is IOException ||
                error is InvalidOperationException ||
                error is UnauthorizedAccessException)
            {
                throw new InvalidOperationException(
                    runPath + ": " + error.Message, error);
            }
        }

        static ModelSeatConfiguration ResolveStructuredOpponent(
            OpponentDto opponent,
            string metadataPath,
            out string label)
        {
            return ResolveStructuredOpponent(
                opponent,
                metadataPath,
                "structured_dagger",
                out label);
        }

        static ModelSeatConfiguration ResolveStructuredOpponent(
            OpponentDto opponent,
            string metadataPath,
            string presentationAlgorithm,
            out string label)
        {
            if (opponent.kind == "scripted")
            {
                if (opponent.name == "greedy")
                {
                    label = "Greedy";
                    return new ModelSeatConfiguration {
                        Kind = ModelControllerKind.Greedy };
                }
                if (opponent.name == "random")
                {
                    label = "Random";
                    return new ModelSeatConfiguration {
                        Kind = ModelControllerKind.Random };
                }
                if (opponent.name == "passive")
                {
                    label = "Passive";
                    return new ModelSeatConfiguration {
                        Kind = ModelControllerKind.Passive };
                }
                throw new InvalidOperationException(
                    metadataPath +
                    ".name must be 'greedy', 'random', or 'passive' for a scripted opponent");
            }
            bool live = opponent.kind == "live_run";
            if (!live && opponent.kind != "fixed_run")
                throw new InvalidOperationException(
                    metadataPath + ".kind '" +
                    (opponent.kind ?? "<missing>") +
                    "' is not supported for " +
                    presentationAlgorithm + " presentation");
            string expectedMode = live ? "live" : "fixed";
            if (!string.Equals(opponent.mode, expectedMode, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    metadataPath + ".mode must be '" + expectedMode + "'");
            if (!string.Equals(
                    opponent.algorithm,
                    "structured_imitation",
                    StringComparison.Ordinal) &&
                !string.Equals(
                    opponent.algorithm,
                    OutcomeAlgorithm,
                    StringComparison.Ordinal))
                throw new InvalidOperationException(
                    metadataPath +
                    ".algorithm must be structured_imitation or " +
                    OutcomeAlgorithm);
            if (string.IsNullOrWhiteSpace(opponent.source_run))
                throw new InvalidOperationException(
                    metadataPath + ".source_run is required");
            label = Path.GetFileName(opponent.source_run.TrimEnd(
                Path.DirectorySeparatorChar,
                Path.AltDirectorySeparatorChar)) +
                (string.Equals(
                    opponent.algorithm,
                    OutcomeAlgorithm,
                    StringComparison.Ordinal)
                    ? " · outcome candidate"
                    : string.Empty) +
                (live ? " · live" : " · fixed");
            return new ModelSeatConfiguration
            {
                Kind = live
                    ? ModelControllerKind.LiveRun
                    : ModelControllerKind.FixedRun,
                Path = opponent.source_run,
            };
        }

        public MlPresentationGame PlanGame(int gameIndex)
        {
            if (gameIndex < 0)
                throw new ArgumentOutOfRangeException(
                    nameof(gameIndex), gameIndex, "game index must be non-negative");
            int learnerSeat = _learnerSeat == "1"
                ? 1
                : _learnerSeat == "alternating"
                    ? gameIndex % 2
                    : 0;
            OpponentPlan opponent = _opponents[gameIndex % _opponents.Count];
            return new MlPresentationGame(
                learnerSeat == 0 ? _learnerSpec : opponent.Spec,
                learnerSeat == 0 ? opponent.Spec : _learnerSpec,
                learnerSeat,
                learnerSeat == 0
                    ? ModelDuelObserverSeat.Player1
                    : ModelDuelObserverSeat.Player2,
                opponent.Label,
                Scenario);
        }

        public MlPresentationSchedule BuildRuntimeSchedule()
        {
            var games = new MlPresentationGame[_opponents.Count * 2];
            for (int index = 0; index < games.Length; index++)
                games[index] = PlanGame(index);
            return new MlPresentationSchedule { Games = games };
        }

        static bool ValidateRawManifest(string json)
        {
            JObject root;
            try
            {
                root = JToken.Parse(json) as JObject;
            }
            catch (JsonException error)
            {
                throw new InvalidDataException("run manifest contains invalid JSON", error);
            }
            if (root == null)
                throw new InvalidDataException("run manifest must be a JSON object");

            JProperty scenario = root.Property(
                "scenario", StringComparison.Ordinal);
            if (scenario != null && scenario.Value.Type != JTokenType.Object)
                throw new InvalidDataException(
                    "scenario must be a non-null JSON object");

            JProperty opponent = root.Property(
                "opponent_snapshot", StringComparison.Ordinal);
            if (opponent != null)
                ValidateRawOpponent(
                    opponent.Value, "opponent_snapshot");
            return scenario != null;
        }

        static void ValidateRawOpponent(JToken token, string metadataPath)
        {
            JObject opponent = token as JObject;
            if (opponent == null)
                throw new InvalidDataException(
                    metadataPath + " must be a non-null JSON object");
            string kind = opponent.Value<string>("kind");
            if (kind == "snapshot")
            {
                JProperty step = opponent.Property(
                    "step", StringComparison.Ordinal);
                if (step == null || step.Value.Type != JTokenType.Integer)
                    throw new InvalidDataException(
                        metadataPath + ".step must be a non-negative integer");
                try
                {
                    long value = step.Value.Value<long>();
                    if (value < 0 || value > int.MaxValue)
                        throw new InvalidDataException(
                            metadataPath +
                            ".step must be a non-negative 32-bit integer");
                }
                catch (OverflowException error)
                {
                    throw new InvalidDataException(
                        metadataPath +
                        ".step must be a non-negative 32-bit integer",
                        error);
                }
                return;
            }
            if (kind != "pool") return;
            JArray controllers = opponent["controllers"] as JArray;
            if (controllers == null)
                throw new InvalidDataException(
                    metadataPath + ".controllers must be an array");
            for (int index = 0; index < controllers.Count; index++)
                ValidateRawOpponent(
                    controllers[index],
                    metadataPath + ".controllers[" + index + "]");
        }

        static TrainingScenario LoadScenario(
            string runDirectory,
            RunManifestDto manifest,
            bool hasScenarioMetadata)
        {
            if (!hasScenarioMetadata)
            {
                string environment = manifest.config?.environment;
                if (string.IsNullOrWhiteSpace(environment))
                    environment = manifest.contract?.version;
                try
                {
                    return TrainingScenario.CreateStandard(
                        environment, "legacy-default");
                }
                catch (Exception error)
                {
                    throw new InvalidOperationException(
                        "legacy scenario environment is invalid: " + error.Message,
                        error);
                }
            }

            ScenarioDto recorded = manifest.scenario;
            if (recorded.schema_version != 1)
                throw new InvalidOperationException(
                    "scenario.schema_version must be 1");
            if (string.IsNullOrWhiteSpace(recorded.path))
                throw new InvalidOperationException(
                    "scenario.path must be a non-empty string");
            string scenarioPath = ResolveContainedPath(
                runDirectory, recorded.path, "scenario.path");
            try
            {
                MlTrainingScenario scenario = MlTrainingScenarioFile.Load(scenarioPath);
                if (!string.IsNullOrWhiteSpace(recorded.template_id) &&
                    !string.Equals(
                        scenario.Id, recorded.template_id, StringComparison.Ordinal))
                    throw new InvalidDataException(
                        "scenario id does not match run metadata");
                string expectedEnvironment = manifest.contract?.version;
                if (!string.IsNullOrWhiteSpace(expectedEnvironment) &&
                    !string.Equals(
                        MlEnvironmentContracts.CliValue(scenario.Environment),
                        expectedEnvironment,
                        StringComparison.Ordinal))
                    throw new InvalidDataException(
                        "scenario environment does not match run contract");
                return MlTrainingScenarioPreflight.ToEngine(scenario);
            }
            catch (Exception error) when (
                error is ArgumentException ||
                error is InvalidDataException ||
                error is IOException ||
                error is UnauthorizedAccessException)
            {
                throw new InvalidOperationException(
                    scenarioPath + ": " + error.Message, error);
            }
        }

        static IReadOnlyList<OpponentPlan> ResolveOpponents(
            OpponentDto opponent,
            ContractDto learnerContract,
            string metadataPath)
        {
            if (opponent.kind == "pool")
            {
                if (opponent.controllers == null || opponent.controllers.Length == 0)
                    throw new InvalidOperationException(
                        metadataPath + ".controllers must contain at least one opponent");
                var plans = new List<OpponentPlan>(opponent.controllers.Length);
                for (int index = 0; index < opponent.controllers.Length; index++)
                {
                    OpponentEntryDto item = opponent.controllers[index];
                    if (item == null)
                        throw new InvalidOperationException(
                            metadataPath + ".controllers[" + index + "] is required");
                    plans.Add(ResolveOpponent(
                        item, learnerContract,
                        metadataPath + ".controllers[" + index + "]"));
                }
                return plans;
            }
            return new[]
            {
                ResolveOpponent(
                    OpponentEntryDto.From(opponent),
                    learnerContract,
                    metadataPath)
            };
        }

        static OpponentPlan ResolveOpponent(
            OpponentEntryDto opponent,
            ContractDto learnerContract,
            string metadataPath)
        {
            if (opponent.kind == "scripted")
            {
                if (opponent.name != "greedy" &&
                    opponent.name != "random" &&
                    opponent.name != "passive")
                    throw new InvalidOperationException(
                        metadataPath +
                        ".name must be 'greedy', 'random', or 'passive' for a scripted opponent");
                return new OpponentPlan(
                    opponent.name,
                    opponent.name == "greedy"
                        ? "Greedy"
                        : opponent.name == "random" ? "Random" : "Passive");
            }
            if (opponent.kind == "snapshot")
                return ResolveSnapshot(opponent, learnerContract, metadataPath);
            if (opponent.kind == "run")
                return ResolveLiveRun(opponent, learnerContract, metadataPath);
            throw new InvalidOperationException(
                metadataPath + ".kind '" + (opponent.kind ?? "<missing>") +
                "' is not a supported opponent");
        }

        static OpponentPlan ResolveSnapshot(
            OpponentEntryDto opponent,
            ContractDto learnerContract,
            string metadataPath)
        {
            if (string.IsNullOrWhiteSpace(opponent.path))
                throw new InvalidOperationException(metadataPath + ".path is required");
            if (string.IsNullOrWhiteSpace(opponent.source_run))
                throw new InvalidOperationException(
                    metadataPath + ".source_run is required");
            if (opponent.algorithm != "maskable_ppo" &&
                opponent.algorithm != "masked_dqn")
                throw new InvalidOperationException(
                    metadataPath + ".algorithm is invalid");
            if (opponent.step < 0)
                throw new InvalidOperationException(
                    metadataPath + ".step must be non-negative");

            string sourceRun = Path.GetFullPath(opponent.source_run);
            string checkpoint = Path.GetFullPath(opponent.path);
            string checkpointsDirectory = Path.GetFullPath(
                Path.Combine(sourceRun, "checkpoints"));
            if (!string.Equals(
                    Path.GetDirectoryName(checkpoint),
                    checkpointsDirectory,
                    PathComparison) ||
                !string.Equals(
                    Path.GetFileName(checkpoint),
                    "step_" + opponent.step.ToString("D9") + ".zip",
                    StringComparison.OrdinalIgnoreCase) ||
                !File.Exists(checkpoint))
                throw new InvalidOperationException(
                    metadataPath +
                    ".path must name its recorded checkpoint inside source_run");

            SourceManifestDto source = LoadSourceManifest(
                sourceRun, learnerContract, metadataPath);
            if (!string.Equals(
                    source.config?.algorithm,
                    opponent.algorithm,
                    StringComparison.Ordinal))
                throw new InvalidOperationException(
                    metadataPath +
                    ".algorithm does not match source_run/run.json");

            string inferenceMode = ValidateInferenceMode(
                opponent.inference_mode, metadataPath);
            string spec = JsonUtility.ToJson(new SnapshotSpecDto
            {
                kind = "snapshot",
                path = checkpoint,
                source_run = sourceRun,
                algorithm = opponent.algorithm,
                step = opponent.step,
                inference_mode = inferenceMode,
            });
            return new OpponentPlan(
                spec,
                Path.GetFileName(sourceRun) + " · step " + opponent.step);
        }

        static OpponentPlan ResolveLiveRun(
            OpponentEntryDto opponent,
            ContractDto learnerContract,
            string metadataPath)
        {
            if (opponent.mode != "live")
                throw new InvalidOperationException(
                    metadataPath + ".mode must be 'live'");
            if (string.IsNullOrWhiteSpace(opponent.path))
                throw new InvalidOperationException(metadataPath + ".path is required");
            string sourceRun = Path.GetFullPath(opponent.path);
            LoadSourceManifest(sourceRun, learnerContract, metadataPath);
            string inferenceMode = ValidateInferenceMode(
                opponent.inference_mode, metadataPath);
            string spec = JsonUtility.ToJson(new RunSpecDto
            {
                kind = "run",
                path = sourceRun,
                mode = "live",
                inference_mode = inferenceMode,
            });
            return new OpponentPlan(
                spec, Path.GetFileName(sourceRun) + " · live");
        }

        static SourceManifestDto LoadSourceManifest(
            string sourceRun,
            ContractDto learnerContract,
            string metadataPath)
        {
            string manifestPath = Path.Combine(sourceRun, "run.json");
            try
            {
                SourceManifestDto source = JsonUtility.FromJson<SourceManifestDto>(
                    File.ReadAllText(manifestPath));
                if (source == null || source.schema_version != 1)
                    throw new InvalidDataException(
                        "source run schema_version must be 1");
                if (source.config == null ||
                    string.IsNullOrWhiteSpace(source.config.algorithm))
                    throw new InvalidDataException(
                        "source run config.algorithm is required");
                ValidateContractCompatibility(
                    learnerContract, source.contract, metadataPath);
                return source;
            }
            catch (Exception error) when (
                error is ArgumentException ||
                error is InvalidDataException ||
                error is IOException ||
                error is UnauthorizedAccessException)
            {
                throw new InvalidOperationException(
                    manifestPath + ": " + error.Message, error);
            }
        }

        static void ValidateContractCompatibility(
            ContractDto learner,
            ContractDto opponent,
            string metadataPath)
        {
            if (learner == null ||
                string.IsNullOrWhiteSpace(learner.version) ||
                string.IsNullOrWhiteSpace(learner.encoding_hash))
                throw new InvalidOperationException(
                    metadataPath + ": learner run contract metadata is incomplete");
            if (opponent == null ||
                string.IsNullOrWhiteSpace(opponent.version) ||
                string.IsNullOrWhiteSpace(opponent.encoding_hash))
                throw new InvalidOperationException(
                    metadataPath + ": opponent source run contract metadata is incomplete");
            if (!string.Equals(
                    learner.version, opponent.version, StringComparison.Ordinal) ||
                !string.Equals(
                    learner.encoding_hash,
                    opponent.encoding_hash,
                    StringComparison.Ordinal))
                throw new InvalidOperationException(
                    metadataPath +
                    ": opponent source run contract is incompatible with the learner run");
        }

        static string ValidateInferenceMode(string value, string metadataPath)
        {
            if (string.IsNullOrWhiteSpace(value)) return "deterministic";
            if (value == "deterministic" || value == "stochastic") return value;
            throw new InvalidOperationException(
                metadataPath +
                ".inference_mode must be 'deterministic' or 'stochastic'");
        }

        static string ResolveContainedPath(
            string root, string recordedPath, string metadataPath)
        {
            if (Path.IsPathRooted(recordedPath))
                throw new InvalidOperationException(
                    metadataPath + " must be relative to the run directory");
            string fullRoot = Path.GetFullPath(root).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string fullPath = Path.GetFullPath(Path.Combine(fullRoot, recordedPath));
            if (!fullPath.StartsWith(
                    fullRoot + Path.DirectorySeparatorChar,
                    PathComparison))
                throw new InvalidOperationException(
                    metadataPath + " escapes the run directory");
            return fullPath;
        }

        static StringComparison PathComparison =>
            Environment.OSVersion.Platform == PlatformID.Win32NT
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal;

        static bool SamePath(string left, string right)
        {
            if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
                return string.Equals(left, right, StringComparison.Ordinal);
            string normalizedLeft = Path.GetFullPath(left).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string normalizedRight = Path.GetFullPath(right).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return string.Equals(normalizedLeft, normalizedRight, PathComparison);
        }

        sealed class OpponentPlan
        {
            public OpponentPlan(string spec, string label)
            {
                Spec = spec;
                Label = label;
            }

            public string Spec { get; }
            public string Label { get; }
        }

        [Serializable]
        sealed class RunManifestDto
        {
            public int schema_version;
            public ConfigDto config;
            public ContractDto contract;
            public ScenarioDto scenario;
            public OpponentDto opponent_snapshot;
        }

        [Serializable]
        sealed class SourceManifestDto
        {
            public int schema_version;
            public ConfigDto config;
            public ContractDto contract;
        }

        [Serializable]
        sealed class ConfigDto
        {
            public string algorithm;
            public string environment;
            public string learner_seat;
        }

        [Serializable]
        sealed class ContractDto
        {
            public string version;
            public string encoding_hash;
        }

        [Serializable]
        sealed class ScenarioDto
        {
            public string path;
            public string template_id;
            public int schema_version;
        }

        [Serializable]
        sealed class OpponentDto
        {
            public string kind;
            public string name;
            public string path;
            public string source_run;
            public string algorithm;
            public int step;
            public string mode;
            public string inference_mode;
            public OpponentEntryDto[] controllers;
        }

        [Serializable]
        sealed class OpponentEntryDto
        {
            public string kind;
            public string name;
            public string path;
            public string source_run;
            public string algorithm;
            public int step;
            public string mode;
            public string inference_mode;

            public static OpponentEntryDto From(OpponentDto value) =>
                new OpponentEntryDto
                {
                    kind = value.kind,
                    name = value.name,
                    path = value.path,
                    source_run = value.source_run,
                    algorithm = value.algorithm,
                    step = value.step,
                    mode = value.mode,
                    inference_mode = value.inference_mode,
                };
        }

        [Serializable]
        sealed class SnapshotSpecDto
        {
            public string kind;
            public string path;
            public string source_run;
            public string algorithm;
            public int step;
            public string inference_mode;
        }

        [Serializable]
        sealed class RunSpecDto
        {
            public string kind;
            public string path;
            public string mode;
            public string inference_mode;
        }
    }
}

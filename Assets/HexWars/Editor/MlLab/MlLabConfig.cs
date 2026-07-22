using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using HexWars.Presentation;
using UnityEngine;

namespace HexWars.Presentation.EditorTools.MlLab
{
    public enum MlAlgorithm { MaskablePpo, MaskedDqn }
    public enum MlOpponentKind { Greedy, Random, FixedRun, LiveRun }
    public enum MlLearnerSeat { Alternating, Seat0, Seat1 }

    [Serializable]
    public sealed class MlTrackerConfig
    {
        public string Kind;
        public string Settings;

        public MlTrackerConfig(string kind, string settings = "")
        {
            Kind = kind ?? string.Empty;
            Settings = settings ?? string.Empty;
        }

        public string ToCliValue() => string.Equals(Kind, "custom", StringComparison.OrdinalIgnoreCase)
            ? "custom=" + Settings
            : Kind;
    }

    [Serializable]
    public sealed class MlLabConfig
    {
        static readonly Regex SafeRunName = new Regex(
            "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", RegexOptions.CultureInvariant);

        public string RunName = "run1";
        public MlEnvironmentContract Environment = MlEnvironmentContract.TacticalV1;
        public MlAlgorithm Algorithm = MlAlgorithm.MaskablePpo;
        public long TotalTimesteps = 300000;
        public int Seed = 1;
        public int CheckpointInterval = 10000;
        public int Workers = 1;
        public string Device = "auto";
        public MlLearnerSeat LearnerSeat = MlLearnerSeat.Alternating;
        public MlOpponentKind OpponentKind = MlOpponentKind.Greedy;
        public string OpponentPath = string.Empty;
        public string ResumeSource = string.Empty;
        public List<MlTrackerConfig> Trackers = new List<MlTrackerConfig> { new MlTrackerConfig("local") };
        public string WandbProject = string.Empty;
        public string WandbEntity = string.Empty;
        public string WandbMode = string.Empty;
        public string WandbGroup = string.Empty;
        public List<string> WandbTags = new List<string>();
        public bool WandbUploadArtifacts;

        public static MlLabConfig Default() => new MlLabConfig();

        public IReadOnlyList<string> Validate()
        {
            var errors = new List<string>();
            if (string.IsNullOrWhiteSpace(RunName) || !SafeRunName.IsMatch(RunName))
                errors.Add("Run name must use 1-64 letters, numbers, dots, underscores, or dashes.");
            if (TotalTimesteps <= 0) errors.Add("Timesteps must be greater than zero.");
            if (CheckpointInterval <= 0) errors.Add("Checkpoint interval must be greater than zero.");
            if (Workers <= 0) errors.Add("Workers must be at least one.");
            if (string.IsNullOrWhiteSpace(Device)) errors.Add("Device is required.");
            if ((OpponentKind == MlOpponentKind.FixedRun || OpponentKind == MlOpponentKind.LiveRun) &&
                string.IsNullOrWhiteSpace(OpponentPath))
                errors.Add("Opponent path is required for a model or live run.");
            if (!string.IsNullOrEmpty(ResumeSource) && string.IsNullOrWhiteSpace(ResumeSource))
                errors.Add("Resume source cannot be blank.");
            foreach (var tracker in Trackers ?? new List<MlTrackerConfig>())
                if (tracker != null && string.Equals(tracker.Kind, "custom", StringComparison.OrdinalIgnoreCase) &&
                    string.IsNullOrWhiteSpace(tracker.Settings))
                    errors.Add("Custom tracker requires a module:function adapter.");
            return errors;
        }

        public string BuildTrainArguments()
        {
            var args = new List<string>
            {
                "train",
                "--run", Q(RunName),
                "--environment", MlEnvironmentContracts.CliValue(Environment),
                "--algorithm", AlgorithmValue(Algorithm),
                "--opponent", Q(OpponentValue()),
                "--timesteps", TotalTimesteps.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--checkpoint-every", CheckpointInterval.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--workers", Workers.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--seed", Seed.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--device", Q(Device),
                "--learner-seat", SeatValue(LearnerSeat),
            };
            var trackers = Trackers ?? new List<MlTrackerConfig>();
            foreach (var tracker in trackers)
            {
                if (tracker == null || string.IsNullOrWhiteSpace(tracker.Kind)) continue;
                args.Add("--tracker");
                args.Add(Q(tracker.ToCliValue()));
            }
            bool hasWandb = trackers.Exists(tracker => tracker != null &&
                string.Equals(tracker.Kind, "wandb", StringComparison.OrdinalIgnoreCase));
            if (hasWandb)
            {
                AddOption(args, "--wandb-project", WandbProject);
                AddOption(args, "--wandb-entity", WandbEntity);
                AddOption(args, "--wandb-mode", WandbMode);
                AddOption(args, "--wandb-group", WandbGroup);
                foreach (var tag in WandbTags ?? new List<string>()) AddOption(args, "--wandb-tag", tag);
                if (WandbUploadArtifacts) args.Add("--wandb-upload-artifacts");
            }
            args.Add("--no-console-output");
            args.Add("--json");
            return string.Join(" ", args);
        }

        public string BuildResumeArguments()
        {
            return string.Join(" ", new[]
            {
                "resume", Q(ResumeSource), "--run", Q(RunName), "--timesteps",
                TotalTimesteps.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--no-console-output", "--json"
            });
        }

        public string BuildDoctorArguments(string runsRoot, string server)
        {
            var args = new List<string>
            {
                "doctor",
                "--environment", MlEnvironmentContracts.CliValue(Environment),
                "--runs-root", Q(runsRoot),
                "--server", Q(server),
            };
            foreach (var tracker in Trackers ?? new List<MlTrackerConfig>())
            {
                if (tracker == null || string.IsNullOrWhiteSpace(tracker.Kind)) continue;
                args.Add("--tracker");
                args.Add(Q(tracker.ToCliValue()));
            }
            args.Add("--json");
            return string.Join(" ", args);
        }

        string OpponentValue()
        {
            switch (OpponentKind)
            {
                case MlOpponentKind.Random: return "random";
                case MlOpponentKind.FixedRun: return "run:" + OpponentPath;
                case MlOpponentKind.LiveRun:
                    return new ModelSeatConfiguration
                    {
                        Kind = ModelControllerKind.LiveRun,
                        Path = OpponentPath,
                    }.BuildSpec();
                default: return "greedy";
            }
        }

        static string AlgorithmValue(MlAlgorithm value) =>
            value == MlAlgorithm.MaskablePpo ? "maskable_ppo" : "masked_dqn";

        static string SeatValue(MlLearnerSeat value)
        {
            if (value == MlLearnerSeat.Seat0) return "0";
            if (value == MlLearnerSeat.Seat1) return "1";
            return "alternating";
        }

        static string Q(string value) => MlCliProcess.QuoteArgument(value ?? string.Empty);

        static void AddOption(List<string> args, string name, string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return;
            args.Add(name);
            args.Add(Q(value));
        }
    }

    public sealed class MlEnvironmentSummary
    {
        static readonly string[] AdaptiveFixedRoles =
        {
            "Frontline", "Assault", "Marksman", "Artillery", "Recon", "Support",
        };

        public string ContractVersion { get; private set; } = string.Empty;
        public int FixedTemplateCount { get; private set; }
        public int CustomTemplateCount { get; private set; }
        public int MaxControllableUnits { get; private set; }
        public int StartingUnitCount { get; private set; }
        public int StartingArmyBudget { get; private set; }
        public string[] FixedRoles { get; private set; } = Array.Empty<string>();
        public bool HiddenDeployment { get; private set; }
        public string DisplayText
        {
            get
            {
                if (!string.Equals(ContractVersion, "adaptive-v1", StringComparison.Ordinal))
                    return string.IsNullOrWhiteSpace(ContractVersion)
                        ? "Environment contract metadata unavailable."
                        : ContractVersion + " · immediate tactical setup";
                string roles = FixedRoles.Length == 0 ? "not recorded" : string.Join(", ", FixedRoles);
                return ContractVersion + "\n" +
                    "Fixed roles: " + roles + "\n" +
                    CustomTemplateCount + " custom slots · " + MaxControllableUnits + " maximum units · " +
                    StartingUnitCount + " starting units · " + StartingArmyBudget + " setup points\n" +
                    "combined-arms scripted deployment · " +
                    (HiddenDeployment ? "hidden deployment" : "deployment visibility not recorded");
            }
        }

        public static MlEnvironmentSummary ForSelection(MlEnvironmentContract environment)
        {
            if (environment == MlEnvironmentContract.TacticalV1)
                return new MlEnvironmentSummary { ContractVersion = "tactical-v1" };
            return new MlEnvironmentSummary
            {
                ContractVersion = "adaptive-v1",
                FixedTemplateCount = 6,
                CustomTemplateCount = 3,
                MaxControllableUnits = 24,
                StartingUnitCount = 6,
                StartingArmyBudget = 132,
                FixedRoles = (string[])AdaptiveFixedRoles.Clone(),
                HiddenDeployment = true,
            };
        }

        public static MlEnvironmentSummary FromRunManifest(string json)
        {
            var manifest = JsonUtility.FromJson<RunManifestDto>(json);
            var contract = manifest?.contract;
            if (contract == null) return new MlEnvironmentSummary();
            var semantics = contract.semantics;
            var roles = new List<string>();
            if (semantics?.templates != null)
                foreach (var template in semantics.templates)
                    if (template != null && template.@fixed && !string.IsNullOrWhiteSpace(template.name))
                        roles.Add(template.name);
            return new MlEnvironmentSummary
            {
                ContractVersion = contract.version ?? string.Empty,
                FixedTemplateCount = semantics?.fixed_template_count ?? 0,
                CustomTemplateCount = semantics?.custom_template_count ?? 0,
                MaxControllableUnits = semantics?.max_controllable_units ?? 0,
                StartingUnitCount = semantics?.starting_unit_count ?? 0,
                StartingArmyBudget = semantics?.starting_army_budget ?? 0,
                FixedRoles = roles.ToArray(),
                HiddenDeployment = string.Equals(contract.version, "adaptive-v1", StringComparison.Ordinal)
                    && !string.IsNullOrWhiteSpace(semantics?.fog_rule),
            };
        }

        [Serializable] sealed class RunManifestDto { public ContractDto contract; }
        [Serializable] sealed class ContractDto { public string version; public SemanticsDto semantics; }
        [Serializable] sealed class SemanticsDto
        {
            public int fixed_template_count;
            public int custom_template_count;
            public int max_controllable_units;
            public int starting_unit_count;
            public int starting_army_budget;
            public string fog_rule;
            public TemplateDto[] templates;
        }
        [Serializable] sealed class TemplateDto { public string name; public bool @fixed; }
    }
}

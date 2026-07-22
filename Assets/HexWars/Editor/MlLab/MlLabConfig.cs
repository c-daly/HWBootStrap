using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace HexWars.Presentation.EditorTools.MlLab
{
    public enum MlAlgorithm { MaskablePpo, MaskedDqn }
    public enum MlOpponentKind { Greedy, Random, FixedCheckpoint, LiveRun }
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
        public MlAlgorithm Algorithm = MlAlgorithm.MaskablePpo;
        public long TotalTimesteps = 300000;
        public int Seed = 1;
        public int CheckpointInterval = 10000;
        public int Workers = 1;
        public string Device = "auto";
        public MlLearnerSeat LearnerSeat = MlLearnerSeat.Alternating;
        public MlOpponentKind OpponentKind = MlOpponentKind.Greedy;
        public string OpponentPath = string.Empty;
        public MlAlgorithm OpponentAlgorithm = MlAlgorithm.MaskablePpo;
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
            if ((OpponentKind == MlOpponentKind.FixedCheckpoint || OpponentKind == MlOpponentKind.LiveRun) &&
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
                "--algorithm", AlgorithmValue(Algorithm),
                "--opponent", Q(OpponentValue()),
                "--timesteps", TotalTimesteps.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--checkpoint-every", CheckpointInterval.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--workers", Workers.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--seed", Seed.ToString(System.Globalization.CultureInfo.InvariantCulture),
                "--device", Q(Device),
                "--learner-seat", SeatValue(LearnerSeat),
            };
            foreach (var tracker in Trackers ?? new List<MlTrackerConfig>())
            {
                if (tracker == null || string.IsNullOrWhiteSpace(tracker.Kind)) continue;
                args.Add("--tracker");
                args.Add(Q(tracker.ToCliValue()));
            }
            AddOption(args, "--wandb-project", WandbProject);
            AddOption(args, "--wandb-entity", WandbEntity);
            AddOption(args, "--wandb-mode", WandbMode);
            AddOption(args, "--wandb-group", WandbGroup);
            foreach (var tag in WandbTags ?? new List<string>()) AddOption(args, "--wandb-tag", tag);
            if (WandbUploadArtifacts) args.Add("--wandb-upload-artifacts");
            args.Add("--json");
            return string.Join(" ", args);
        }

        public string BuildResumeArguments()
        {
            return string.Join(" ", new[]
            {
                "resume", Q(ResumeSource), "--run", Q(RunName), "--timesteps",
                TotalTimesteps.ToString(System.Globalization.CultureInfo.InvariantCulture), "--json"
            });
        }

        string OpponentValue()
        {
            switch (OpponentKind)
            {
                case MlOpponentKind.Random: return "random";
                case MlOpponentKind.FixedCheckpoint:
                    return (OpponentAlgorithm == MlAlgorithm.MaskablePpo ? "ppo:" : "dqn:") + OpponentPath;
                case MlOpponentKind.LiveRun: return "run:" + OpponentPath;
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
}

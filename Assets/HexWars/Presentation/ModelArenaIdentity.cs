using System;
using System.IO;

namespace HexWars.Presentation
{
    public sealed class ModelArenaSeatIdentity
    {
        public string Player { get; internal set; }
        public string Controller { get; internal set; }
        public string Algorithm { get; internal set; }
        public string Checkpoint { get; internal set; }
        public string Step { get; internal set; }
        public string Record { get; internal set; }
        public bool IsActive { get; internal set; }
    }

    public static class ModelArenaIdentity
    {
        public static ModelArenaSeatIdentity[] Build(
            string p0Spec, string p1Spec, PolicySeatInfo p0, PolicySeatInfo p1,
            int currentSeat, int p0Wins, int p1Wins, int draws) => new[]
        {
            BuildSeat(0, p0Spec, p0, currentSeat == 0, p0Wins, p1Wins, draws),
            BuildSeat(1, p1Spec, p1, currentSeat == 1, p1Wins, p0Wins, draws),
        };

        static ModelArenaSeatIdentity BuildSeat(
            int seat, string spec, PolicySeatInfo resolved, bool active,
            int wins, int losses, int draws)
        {
            bool scripted = string.Equals(spec, "greedy", StringComparison.OrdinalIgnoreCase)
                           || string.Equals(spec, "random", StringComparison.OrdinalIgnoreCase);
            return new ModelArenaSeatIdentity
            {
                Player = seat == 0 ? "P1" : "P2",
                Controller = scripted ? Capitalize(spec) : RunName(spec, resolved),
                Algorithm = resolved == null ? string.Empty : FriendlyAlgorithm(resolved.Algorithm),
                Checkpoint = resolved == null ? (scripted ? string.Empty : "loading checkpoint") : Path.GetFileName(resolved.Path),
                Step = resolved == null || resolved.Step <= 0 ? string.Empty : $"step {resolved.Step:N0}",
                Record = FormatRecord(wins, losses, draws),
                IsActive = active,
            };
        }

        public static string FormatRecord(int wins, int losses, int draws)
        {
            int total = wins + losses + draws;
            return total == 0 ? $"{wins}-{losses}-{draws} (—)"
                              : $"{wins}-{losses}-{draws} ({100.0 * wins / total:0.#}%)";
        }

        public static string MiddleTruncate(string value, int max)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= max) return value ?? string.Empty;
            int left = (max - 2) / 2;
            return value.Substring(0, left) + "…" + value.Substring(value.Length - (max - left - 1));
        }

        static string FriendlyAlgorithm(string value) => value switch
        {
            "maskable_ppo" => "Maskable PPO",
            "masked_dqn" => "Masked DQN",
            _ => string.IsNullOrWhiteSpace(value) ? "unknown algorithm" : value,
        };

        static string RunName(string spec, PolicySeatInfo resolved)
        {
            string path = resolved?.Kind == "run" ? Directory.GetParent(resolved.Path)?.Parent?.FullName : null;
            if (string.IsNullOrWhiteSpace(path))
            {
                int colon = (spec ?? string.Empty).IndexOf(':');
                path = colon >= 0 ? spec.Substring(colon + 1) : spec;
            }
            return string.IsNullOrWhiteSpace(path) ? "model" : new DirectoryInfo(path).Name;
        }

        static string Capitalize(string value) => string.IsNullOrEmpty(value)
            ? string.Empty : char.ToUpperInvariant(value[0]) + value.Substring(1).ToLowerInvariant();
    }
}

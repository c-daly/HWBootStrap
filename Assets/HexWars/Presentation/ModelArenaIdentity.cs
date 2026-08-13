using System;
using UnityEngine;

namespace HexWars.Presentation
{
    public sealed class ModelArenaSeatIdentity
    {
        public string Player { get; internal set; }
        public string Role { get; internal set; }
        public string Controller { get; internal set; }
        public string Algorithm { get; internal set; }
        public string Checkpoint { get; internal set; }
        public string Step { get; internal set; }
        public string Record { get; internal set; }
        public string Status { get; internal set; }
        public bool IsActive { get; internal set; }
        /// <summary>The player's point total from the viewer's <c>PresentedState</c> — never the live
        /// simulation state, which may already be ahead of what is on screen.</summary>
        public int Points { get; internal set; }
    }

    public static class ModelArenaIdentity
    {
        public static ModelArenaSeatIdentity[] Build(
            string p0Spec, string p1Spec, PolicySeatInfo p0, PolicySeatInfo p1,
            int currentSeat, int p0Wins, int p1Wins, int draws,
            string p0Status = null, string p1Status = null,
            int learnerSeat = -1, int learnerWins = 0,
            int learnerLosses = 0, int learnerDraws = 0,
            string opponentLabel = null,
            int p0Points = 0, int p1Points = 0) => new[]
        {
            BuildSeat(
                0, p0Spec, p0, currentSeat == 0,
                learnerSeat < 0 ? p0Wins : learnerSeat == 0 ? learnerWins : learnerLosses,
                learnerSeat < 0 ? p1Wins : learnerSeat == 0 ? learnerLosses : learnerWins,
                learnerSeat < 0 ? draws : learnerDraws,
                p0Status, Role(0, learnerSeat),
                learnerSeat >= 0 && learnerSeat != 0 ? opponentLabel : null, p0Points),
            BuildSeat(
                1, p1Spec, p1, currentSeat == 1,
                learnerSeat < 0 ? p1Wins : learnerSeat == 1 ? learnerWins : learnerLosses,
                learnerSeat < 0 ? p0Wins : learnerSeat == 1 ? learnerLosses : learnerWins,
                learnerSeat < 0 ? draws : learnerDraws,
                p1Status, Role(1, learnerSeat),
                learnerSeat >= 0 && learnerSeat != 1 ? opponentLabel : null, p1Points),
        };

        static ModelArenaSeatIdentity BuildSeat(
            int seat, string spec, PolicySeatInfo resolved, bool active,
            int wins, int losses, int draws, string status, string role,
            string controllerLabel, int points)
        {
            bool scripted = string.Equals(spec, "greedy", StringComparison.OrdinalIgnoreCase)
                           || string.Equals(spec, "random", StringComparison.OrdinalIgnoreCase);
            return new ModelArenaSeatIdentity
            {
                Player = seat == 0 ? "P1" : "P2",
                Role = role,
                Controller = !string.IsNullOrWhiteSpace(controllerLabel)
                    ? controllerLabel
                    : scripted ? Capitalize(spec) : RunName(spec, resolved),
                Algorithm = resolved == null ? string.Empty : FriendlyAlgorithm(resolved.Algorithm),
                Checkpoint = resolved == null ? (scripted ? string.Empty : "loading checkpoint") : SafePathLeaf(resolved.Path),
                Step = resolved == null ? string.Empty : resolved.HasStep ? $"step {resolved.Step:N0}" : "step unknown",
                Record = FormatRecord(wins, losses, draws),
                Status = status ?? string.Empty,
                IsActive = active,
                Points = points,
            };
        }

        static string Role(int seat, int learnerSeat)
        {
            if (learnerSeat < 0) return string.Empty;
            return seat == learnerSeat ? "Learner" : "Opponent";
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
            "structured_imitation" => "Structured PyTorch policy",
            _ => string.IsNullOrWhiteSpace(value) ? "unknown algorithm" : value,
        };

        static string RunName(string spec, PolicySeatInfo resolved)
        {
            if (resolved?.Kind == "run" && !string.IsNullOrWhiteSpace(resolved.Path))
            {
                string root = ParentPath(ParentPath(resolved.Path));
                string leaf = SafePathLeaf(root);
                if (!string.IsNullOrWhiteSpace(leaf)) return leaf;
            }
            string path = SpecPath(spec);
            string name = SafePathLeaf(path);
            return string.IsNullOrWhiteSpace(name) ? "model" : name;
        }

        static string SpecPath(string spec)
        {
            if (string.IsNullOrWhiteSpace(spec)) return string.Empty;
            string trimmed = spec.Trim();
            if (trimmed.StartsWith("{", StringComparison.Ordinal))
            {
                try { return JsonUtility.FromJson<ModelSpecDto>(trimmed)?.path ?? string.Empty; }
                catch (Exception) { return string.Empty; }
            }
            foreach (string prefix in new[] { "run:", "ppo:", "dqn:" })
                if (trimmed.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    return trimmed.Substring(prefix.Length).Trim();
            return string.Empty;
        }

        static string ParentPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return string.Empty;
            string value = path.Trim().TrimEnd('/', '\\');
            int split = Math.Max(value.LastIndexOf('/'), value.LastIndexOf('\\'));
            return split <= 0 ? string.Empty : value.Substring(0, split);
        }

        static string SafePathLeaf(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return string.Empty;
            string value = path.Trim().TrimEnd('/', '\\');
            int split = Math.Max(value.LastIndexOf('/'), value.LastIndexOf('\\'));
            return split >= 0 ? value.Substring(split + 1) : value;
        }

        [Serializable] sealed class ModelSpecDto { public string kind; public string path; public string mode; }

        static string Capitalize(string value) => string.IsNullOrEmpty(value)
            ? string.Empty : char.ToUpperInvariant(value[0]) + value.Substring(1).ToLowerInvariant();
    }
}

using System;
using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    public sealed class TacticalV3CapacityProfile
    {
        public TacticalV3CapacityProfile(
            int maxCells, int maxUnits, int maxTemplates,
            int maxCapabilityDefinitions, int maxCapabilityAllocations,
            int maxRules, int maxMemoryRecords, int maxRelations, int maxCandidates)
        {
            MaxCells = RequirePositive(maxCells, nameof(maxCells));
            MaxUnits = RequirePositive(maxUnits, nameof(maxUnits));
            MaxTemplates = RequirePositive(maxTemplates, nameof(maxTemplates));
            MaxCapabilityDefinitions = RequirePositive(maxCapabilityDefinitions, nameof(maxCapabilityDefinitions));
            MaxCapabilityAllocations = RequirePositive(maxCapabilityAllocations, nameof(maxCapabilityAllocations));
            MaxRules = RequirePositive(maxRules, nameof(maxRules));
            MaxMemoryRecords = RequirePositive(maxMemoryRecords, nameof(maxMemoryRecords));
            MaxRelations = RequirePositive(maxRelations, nameof(maxRelations));
            MaxCandidates = RequirePositive(maxCandidates, nameof(maxCandidates));
        }

        public int MaxCells { get; }
        public int MaxUnits { get; }
        public int MaxTemplates { get; }
        public int MaxCapabilityDefinitions { get; }
        public int MaxCapabilityAllocations { get; }
        public int MaxRules { get; }
        public int MaxMemoryRecords { get; }
        public int MaxRelations { get; }
        public int MaxCandidates { get; }

        public static TacticalV3CapacityProfile ExperimentalDefault() =>
            new TacticalV3CapacityProfile(512, 64, 32, 128, 2048, 128, 64, 65536, 32768);

        private static int RequirePositive(int value, string parameterName)
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(parameterName, "capacity must be positive");
            return value;
        }
    }

    public sealed class TacticalV3RewardConfig
    {
        public TacticalV3RewardConfig(
            float terminalWin, float terminalNonWin,
            float materialAdjustmentBound, float timePressureBound, float pointsWeight)
        {
            if (terminalWin != 1f || terminalNonWin != -1f)
                throw new ArgumentException("tactical-v3 terminal rewards must be +1/-1");
            if (materialAdjustmentBound != 0.20f || timePressureBound != 0.05f)
                throw new ArgumentException("tactical-v3 shaping bounds must be 0.20/0.05");
            if (pointsWeight != 0.5f)
                throw new ArgumentException("tactical-v3 points weight must be 0.5");

            TerminalWin = terminalWin;
            TerminalNonWin = terminalNonWin;
            MaterialAdjustmentBound = materialAdjustmentBound;
            TimePressureBound = timePressureBound;
            PointsWeight = pointsWeight;
        }

        public float TerminalWin { get; }
        public float TerminalNonWin { get; }
        public float MaterialAdjustmentBound { get; }
        public float TimePressureBound { get; }
        public float PointsWeight { get; }
    }

    public sealed class TacticalV3Config
    {
        public TacticalV3Config(
            TacticalV2Config match,
            TacticalV3CapacityProfile capacity,
            TacticalV3RewardConfig reward)
        {
            Match = match ?? throw new ArgumentNullException(nameof(match));
            Capacity = capacity ?? throw new ArgumentNullException(nameof(capacity));
            Reward = reward ?? throw new ArgumentNullException(nameof(reward));
        }

        public TacticalV2Config Match { get; }
        public TacticalV3CapacityProfile Capacity { get; }
        public TacticalV3RewardConfig Reward { get; }

        public IReadOnlyList<string> Validate()
        {
            var errors = new List<string>();
            errors.AddRange(Match.Validate());

            if (Match.Game.FogOfWar) errors.Add("tactical-v3 stage one requires fog_of_war=false");
            if (Match.Game.WinConditions != WinBy.Annihilation)
                errors.Add("tactical-v3 stage one requires annihilation");
            if (Match.Game.GeneratorsEnabled)
                errors.Add("tactical-v3 stage one requires generators disabled");
            if (Reward.TerminalWin != 1f || Reward.TerminalNonWin != -1f)
                errors.Add("tactical-v3 terminal rewards must be +1/-1");
            if (Reward.MaterialAdjustmentBound != 0.20f || Reward.TimePressureBound != 0.05f)
                errors.Add("tactical-v3 shaping bounds must be 0.20/0.05");
            if (Reward.PointsWeight != 0.5f)
                errors.Add("tactical-v3 points weight must be 0.5");

            return errors.AsReadOnly();
        }
    }
}

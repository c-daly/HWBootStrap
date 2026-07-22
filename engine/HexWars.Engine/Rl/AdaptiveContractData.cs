using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace HexWars.Engine.Rl
{
    public enum AdaptiveStat
    {
        Health,
        Damage,
        Defense,
        Movement,
        VerticalMovement,
        Range,
        RangeArc,
        Vision,
        VisionArc,
    }

    /// <summary>Immutable roster and design catalogs shared by every adaptive environment.</summary>
    public static class AdaptiveContractData
    {
        public static readonly IReadOnlyList<UnitTemplate> Templates = Array.AsReadOnly(new[]
        {
            new UnitTemplate("Frontline", new UnitStats(7, 2, 3, 2, 2, 1, 1, 3, 1)),
            new UnitTemplate("Assault",   new UnitStats(3, 6, 0, 3, 2, 2, 1, 3, 1)),
            new UnitTemplate("Marksman",  new UnitStats(2, 3, 0, 2, 2, 6, 1, 5, 1)),
            new UnitTemplate("Artillery", new UnitStats(3, 6, 0, 1, 1, 5, 2, 3, 1)),
            new UnitTemplate("Recon",     new UnitStats(2, 1, 0, 5, 3, 1, 0, 7, 2)),
            new UnitTemplate("Support",   new UnitStats(4, 3, 2, 3, 2, 3, 1, 4, 1)),
            new UnitTemplate("Custom A",  new UnitStats(4, 3, 1, 3, 2, 2, 1, 3, 1)),
            new UnitTemplate("Custom B",  new UnitStats(5, 2, 2, 2, 2, 3, 1, 3, 1)),
            new UnitTemplate("Custom C",  new UnitStats(3, 4, 1, 3, 2, 2, 1, 4, 1)),
        });

        public static readonly IReadOnlyDictionary<AdaptiveStat, IReadOnlyList<int>> StatValues =
            new ReadOnlyDictionary<AdaptiveStat, IReadOnlyList<int>>(
                new Dictionary<AdaptiveStat, IReadOnlyList<int>>
                {
                    [AdaptiveStat.Health] = Values(1, 8),
                    [AdaptiveStat.Damage] = Values(0, 9),
                    [AdaptiveStat.Defense] = Values(0, 9),
                    [AdaptiveStat.Movement] = Values(0, 7),
                    [AdaptiveStat.VerticalMovement] = Values(0, 5),
                    [AdaptiveStat.Range] = Values(0, 9),
                    [AdaptiveStat.RangeArc] = Values(0, 5),
                    [AdaptiveStat.Vision] = Values(0, 11),
                    [AdaptiveStat.VisionArc] = Values(0, 5),
                });

        private static IReadOnlyList<int> Values(int start, int count) =>
            Array.AsReadOnly(Enumerable.Range(start, count).ToArray());
    }

    /// <summary>Configuration pinned by the adaptive-v1 semantic contract.</summary>
    public sealed class AdaptiveEnvConfig
    {
        public BoardGenConfig BoardGen { get; set; } = BoardGenConfig.Default();
        public GameConfig Game { get; set; } = GameConfig.Default(biomesEnabled: false, fogOfWar: true);
        public IReadOnlyList<UnitTemplate> Templates { get; set; } = AdaptiveContractData.Templates;
        public IReadOnlyDictionary<AdaptiveStat, IReadOnlyList<int>> StatValues { get; set; } = AdaptiveContractData.StatValues;
        public int FixedTemplateCount { get; set; } = 6;
        public int CustomTemplateCount { get; set; } = 3;
        public int MaxControllableUnits { get; set; } = 24;
        public int StartingUnitCount { get; set; } = 6;
        public int StartingArmyBudget { get; set; } = 132;
        public int MaxDesignPointCost { get; set; } = 24;
        public int MaxSteps { get; set; } = 900;
        public float IntermediateDecisionPenalty { get; set; } = 0.001f;
        public float DeploymentCompletionBonus { get; set; } = 0f;

        public static AdaptiveEnvConfig Default() => new AdaptiveEnvConfig();

        public IReadOnlyList<string> Validate(Board board)
        {
            if (board == null) throw new ArgumentNullException(nameof(board));

            var errors = new List<string>();
            int cells = Math.Min(
                board.DeploymentZone(PlayerId.Player0).Count,
                board.DeploymentZone(PlayerId.Player1).Count);
            int cheapest = Templates.Count == 0 ? 0 : Templates.Min(t => t.Stats.PointCost);
            if (cells < StartingUnitCount)
                errors.Add($"starting deployment requires {StartingUnitCount} cells per seat but only {cells} are available");
            if (StartingArmyBudget < cheapest * StartingUnitCount)
                errors.Add($"starting deployment requires at least {cheapest * StartingUnitCount} points but only {StartingArmyBudget} are available");
            if (Templates.Count != FixedTemplateCount + CustomTemplateCount)
                errors.Add("adaptive roster must contain exactly 9 templates");
            if (MaxControllableUnits < StartingUnitCount)
                errors.Add("maximum controllable units must cover the starting army");
            return errors;
        }
    }
}

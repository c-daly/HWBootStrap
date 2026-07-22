using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>Fixed adaptive-v1 action regions and seat-relative observation geometry.</summary>
    public sealed class AdaptiveLayout
    {
        internal const int PhaseTotal = 14;
        internal const int StatTotal = 9;
        internal const int ValueTotal = 11;
        internal const int GlobalCount = 124;

        public IReadOnlyList<HexCoord> Cells { get; }
        public IReadOnlyDictionary<HexCoord, int> CellIndex { get; }
        public BoardGenConfig BoardGen { get; }
        public GameConfig Game { get; }
        public IReadOnlyList<UnitTemplate> Templates { get; }
        public IReadOnlyDictionary<AdaptiveStat, IReadOnlyList<int>> StatValues { get; }
        public int FixedTemplateCount { get; }
        public int StartingUnitCount { get; }
        public int StartingArmyBudget { get; }
        public int MaxDesignPointCost { get; }
        public int CellCount => Cells.Count;

        public int CommandOffset => 0;
        public int CommandCount => 12;
        public int UnitOffset => CommandOffset + CommandCount;
        public int UnitCount => 24;
        public int TemplateOffset => UnitOffset + UnitCount;
        public int TemplateCount => 9;
        public int CellOffset => TemplateOffset + TemplateCount;
        public int StatOffset => CellOffset + CellCount;
        public int StatCount => StatTotal;
        public int ValueOffset => StatOffset + StatCount;
        public int ValueCount => ValueTotal;
        public int ActionCount => ValueOffset + ValueCount;

        public int ElevationPlane => 0;
        public int PlainsPlane => 1;
        public int ForestPlane => 2;
        public int RoughPlane => 3;
        public int WaterPlane => 4;
        public int DeploymentZonePlane => 5;
        public int CurrentVisibilityPlane => 6;
        public int PreviouslySeenPlane => 7;
        public int FriendlyUnitPlane(int role) => 8 + role;
        public int EnemyUnitPlane(int role) => 17 + role;
        public int FriendlySlotPlane(int slot) => 26 + slot;
        public int ObservationChannels => 26 + UnitCount;
        public int ObservationGlobals => GlobalCount;
        public int ObservationLength => checked(ObservationChannels * CellCount + ObservationGlobals);

        public AdaptiveLayout(AdaptiveEnvConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.MaxControllableUnits != 24)
                throw new ArgumentException("adaptive-v1 requires exactly 24 controllable unit slots", nameof(config));
            if (config.Templates.Count != 9)
                throw new ArgumentException("adaptive-v1 requires exactly 9 template slots", nameof(config));
            BoardGen = config.BoardGen;
            Game = config.Game;
            Templates = Array.AsReadOnly(config.Templates.ToArray());
            StatValues = new ReadOnlyDictionary<AdaptiveStat, IReadOnlyList<int>>(
                config.StatValues.ToDictionary(
                    pair => pair.Key,
                    pair => (IReadOnlyList<int>)Array.AsReadOnly(pair.Value.ToArray())));
            FixedTemplateCount = config.FixedTemplateCount;
            StartingUnitCount = config.StartingUnitCount;
            StartingArmyBudget = config.StartingArmyBudget;
            MaxDesignPointCost = config.MaxDesignPointCost;
            var cells = new List<HexCoord>();
            var cellIndex = new Dictionary<HexCoord, int>();
            for (int row = 0; row < BoardGen.Height; row++)
                for (int col = 0; col < BoardGen.Width; col++)
                {
                    var cell = HexLayout.OffsetToAxial(col, row);
                    if (cellIndex.ContainsKey(cell)) continue;
                    cellIndex[cell] = cells.Count;
                    cells.Add(cell);
                }
            Cells = cells.AsReadOnly();
            CellIndex = new ReadOnlyDictionary<HexCoord, int>(cellIndex);
        }
    }
}

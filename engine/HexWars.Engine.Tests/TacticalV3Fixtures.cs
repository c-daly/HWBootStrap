using System.Collections.Generic;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Engine.Tests
{
    internal static class TacticalV3Fixtures
    {
        public static GameConfig CloneGame(GameConfig source, bool? fogOfWar = null)
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>();
            foreach (TerrainType terrainType in System.Enum.GetValues(typeof(TerrainType)))
                terrain.Add(terrainType, source.Terrain(terrainType));

            return new GameConfig(
                terrain,
                startingPoints: source.StartingPoints,
                bountyRate: source.BountyRate,
                generatorCost: source.GeneratorCost,
                generatorOutput: source.GeneratorOutput,
                generatorHealth: source.GeneratorHealth,
                damageFloor: source.DamageFloor,
                dmgHighGroundBonus: source.DmgHighGroundBonus,
                rangeHighGroundBonus: source.RangeHighGroundBonus,
                roundCap: source.RoundCap,
                designFee: source.DesignFee,
                deployCostMultiplier: source.DeployCostMultiplier,
                turnPolicy: source.TurnPolicy,
                biomesEnabled: source.BiomesEnabled,
                winConditions: source.WinConditions,
                captureCost: source.CaptureCost,
                economyWinThreshold: source.EconomyWinThreshold,
                scoreKills: source.ScoreKills,
                scorePoints: source.ScorePoints,
                scoreArmy: source.ScoreArmy,
                scoreTerritory: source.ScoreTerritory,
                upkeepFactor: source.UpkeepFactor,
                captureFactor: source.CaptureFactor,
                buildFactor: source.BuildFactor,
                territoryMode: source.TerritoryMode,
                claimEndsTurn: source.ClaimEndsTurn,
                buildAnywhere: source.BuildAnywhere,
                territoryIncome: source.TerritoryIncome,
                generatorsEnabled: source.GeneratorsEnabled,
                pointDecay: source.PointDecay,
                fogOfWar: fogOfWar ?? source.FogOfWar,
                maxDesignPointCost: source.MaxDesignPointCost,
                fixedTemplateCount: source.FixedTemplateCount,
                templateSlotCount: source.TemplateSlotCount);
        }

        public static TacticalV3CapacityProfile ExperimentalCapacity(
            int? maxCells = null, int? maxCandidates = null) =>
            new TacticalV3CapacityProfile(
                maxCells ?? 512,
                maxUnits: 64,
                maxTemplates: 32,
                maxCapabilityDefinitions: 128,
                maxCapabilityAllocations: 2048,
                maxRules: 128,
                maxMemoryRecords: 64,
                maxRelations: 65536,
                maxCandidates: maxCandidates ?? 32768);

        public static TacticalV2Config Match(int width = 13, int height = 9)
        {
            TacticalV2Config match = TacticalV2Config.Default();
            match.BoardGen = new BoardGenConfig(width: width, height: height);
            return match;
        }

        public static TacticalV3Config Config(int width = 13, int height = 9) =>
            new TacticalV3Config(
                Match(width, height),
                ExperimentalCapacity(),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));
    }
}

using System;
using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>Serializable schema-v1 training scenario. JSON parsing stays outside the engine so
    /// Unity and headless callers construct the exact same environment configuration here.</summary>
    [Serializable]
    public sealed class TrainingScenario
    {
        public int SchemaVersion = 1;
        public string Id = "legacy-default";
        public string Name = "Standard";
        public string Environment = MlContract.CurrentVersion;
        public TrainingBoardConfig Board = new TrainingBoardConfig();
        public TrainingRuleConfig Rules = new TrainingRuleConfig();
        public TrainingEpisodeConfig Episode = new TrainingEpisodeConfig();
        public TacticalRewardConfig TacticalReward = null!;
        public AdaptiveRewardConfig AdaptiveReward = null!;
        public TrainingAdaptiveConfig Adaptive = null!;

        public static TrainingScenario CreateStandard(string environment, string id = "legacy-default")
        {
            if (environment != MlContract.CurrentVersion && environment != MlContract.AdaptiveVersion)
                throw new ArgumentException($"unsupported environment '{environment}'", nameof(environment));

            var scenario = new TrainingScenario
            {
                Id = id,
                Environment = environment,
            };

            if (environment == MlContract.CurrentVersion)
            {
                scenario.Episode.MaxSteps = 600;
                scenario.Rules.FogOfWar = false;
                scenario.TacticalReward = new TacticalRewardConfig();
            }
            else
            {
                scenario.Episode.MaxSteps = 900;
                scenario.Rules.FogOfWar = true;
                scenario.AdaptiveReward = new AdaptiveRewardConfig();
                scenario.Adaptive = new TrainingAdaptiveConfig();
            }

            return scenario;
        }

        public IReadOnlyList<string> Validate()
        {
            var errors = new List<string>();
            if (SchemaVersion != 1) errors.Add("schema version must be 1");
            bool tactical = Environment == MlContract.CurrentVersion;
            bool adaptive = Environment == MlContract.AdaptiveVersion;
            if (!tactical && !adaptive) errors.Add("environment must be tactical-v1 or adaptive-v1");

            ValidateBoard(errors);
            ValidateRules(errors);
            ValidateEpisode(errors);

            if (tactical)
            {
                if (TacticalReward == null) errors.Add("tactical-v1 requires a tactical reward section");
                if (AdaptiveReward != null) errors.Add("adaptive reward section is not valid for tactical-v1");
                if (Adaptive != null) errors.Add("adaptive section is not valid for tactical-v1");
            }
            else if (adaptive)
            {
                if (TacticalReward != null) errors.Add("tactical reward section is not valid for adaptive-v1");
                if (AdaptiveReward == null) errors.Add("adaptive-v1 requires an adaptive reward section");
                if (Adaptive == null) errors.Add("adaptive-v1 requires an adaptive section");
                if (Adaptive != null) ValidateAdaptive(errors, Adaptive);
            }

            return errors;
        }

        public EnvConfig BuildTactical()
        {
            ThrowIfInvalid();
            if (Environment != MlContract.CurrentVersion)
                throw new ArgumentException("scenario environment must be tactical-v1", nameof(Environment));

            TacticalRewardConfig reward = TacticalReward!;
            return new EnvConfig
            {
                BoardGen = BuildBoardGen(),
                Game = BuildGameConfig(),
                Roster = EnvConfig.DefaultRoster(),
                MaxSteps = Episode.MaxSteps,
                ShapeScale = reward.ShapeScale,
                StepPenalty = reward.StepPenalty,
                ClosingWeight = reward.ClosingWeight,
                DrawCreditWeight = reward.DrawCreditWeight,
                PointsWeight = reward.PointsWeight,
            };
        }

        public AdaptiveEnvConfig BuildAdaptive()
        {
            ThrowIfInvalid();
            if (Environment != MlContract.AdaptiveVersion)
                throw new ArgumentException("scenario environment must be adaptive-v1", nameof(Environment));

            TrainingAdaptiveConfig adaptive = Adaptive!;
            AdaptiveRewardConfig reward = AdaptiveReward!;
            return new AdaptiveEnvConfig
            {
                BoardGen = BuildBoardGen(),
                Game = BuildGameConfig(
                    maxDesignPointCost: adaptive.MaxDesignPointCost,
                    fixedTemplateCount: 6,
                    templateSlotCount: 9),
                Templates = AdaptiveContractData.Templates,
                StatValues = AdaptiveContractData.StatValues,
                FixedTemplateCount = 6,
                CustomTemplateCount = 3,
                MaxControllableUnits = 24,
                StartingUnitCount = adaptive.StartingUnitCount,
                StartingArmyBudget = adaptive.StartingArmyBudget,
                MaxDesignPointCost = adaptive.MaxDesignPointCost,
                MaxSteps = Episode.MaxSteps,
                IntermediateDecisionPenalty = reward.IntermediateDecisionPenalty,
                DeploymentCompletionBonus = reward.DeploymentCompletionBonus,
            };
        }

        private void ValidateBoard(List<string> errors)
        {
            if (Board == null)
            {
                errors.Add("board section is required");
                return;
            }

            if (Board.Width <= 0) errors.Add("board width must be positive");
            if (Board.Height <= 0) errors.Add("board height must be positive");
            if (Board.MaxElevation <= 0) errors.Add("board max elevation must be positive");
            if (Board.ZoneDepth <= 0) errors.Add("board zone depth must be positive");
            if (Board.Width > 0 && Board.ZoneDepth > 0
                && Board.ZoneDepth > Board.Width - Board.ZoneDepth)
                errors.Add("deployment zones overlap");
            if (double.IsNaN(Board.FlatChance) || Board.FlatChance < 0 || Board.FlatChance > 1)
                errors.Add("board flat chance must be within [0,1]");
            if (Board.PlainsWeight < 0) errors.Add("plains weight must be non-negative");
            if (Board.ForestWeight < 0) errors.Add("forest weight must be non-negative");
            if (Board.RoughWeight < 0) errors.Add("rough weight must be non-negative");
            if (Board.WaterWeight < 0) errors.Add("water weight must be non-negative");
            if ((long)Board.PlainsWeight + Board.ForestWeight + Board.RoughWeight + Board.WaterWeight <= 0)
                errors.Add("terrain weight sum must be positive");
        }

        private void ValidateRules(List<string> errors)
        {
            if (Rules == null)
            {
                errors.Add("rules section is required");
                return;
            }

            if (Rules.ActionsPerTurn < 0) errors.Add("actions per turn must be non-negative");
            if (Rules.RoundCap <= 0) errors.Add("round cap must be positive");
        }

        private void ValidateEpisode(List<string> errors)
        {
            if (Episode == null)
            {
                errors.Add("episode section is required");
                return;
            }

            if (Episode.MaxSteps <= 0) errors.Add("max steps must be positive");
        }

        private void ValidateAdaptive(List<string> errors, TrainingAdaptiveConfig adaptive)
        {
            if (adaptive.StartingUnitCount <= 0) errors.Add("adaptive starting unit count must be positive");
            if (adaptive.StartingUnitCount > 24)
                errors.Add("adaptive starting unit count must not exceed 24 controllable slots");

            if (Board != null && Board.Height > 0 && Board.ZoneDepth > 0)
            {
                long cellsPerSeat = (long)Board.Height * Board.ZoneDepth;
                if (cellsPerSeat < adaptive.StartingUnitCount)
                    errors.Add("adaptive deployment cells must cover the starting unit count");
            }

            int cheapest = AdaptiveContractData.Templates.Min(template => template.Stats.PointCost);
            long minimumBudget = (long)cheapest * adaptive.StartingUnitCount;
            if (adaptive.StartingArmyBudget < minimumBudget)
                errors.Add("adaptive starting army budget is insufficient for the starting unit count");
            if (adaptive.MaxDesignPointCost <= 0)
                errors.Add("adaptive max design point cost must be positive");
        }

        private void ThrowIfInvalid()
        {
            IReadOnlyList<string> errors = Validate();
            if (errors.Count > 0) throw new ArgumentException(string.Join("; ", errors));
        }

        private BoardGenConfig BuildBoardGen() => new BoardGenConfig(
            width: Board.Width,
            height: Board.Height,
            maxElevation: Board.MaxElevation,
            zoneDepth: Board.ZoneDepth,
            flatChance: Board.FlatChance,
            plainsWeight: Board.PlainsWeight,
            forestWeight: Board.ForestWeight,
            roughWeight: Board.RoughWeight,
            waterWeight: Board.WaterWeight);

        private GameConfig BuildGameConfig(
            int maxDesignPointCost = 0,
            int fixedTemplateCount = 0,
            int templateSlotCount = 0)
        {
            ITurnPolicy turnPolicy = Rules.ActionsPerTurn == 0
                ? (ITurnPolicy)new AllUnitsPolicy()
                : new KActionsPolicy(Rules.ActionsPerTurn);

            return new GameConfig(
                new Dictionary<TerrainType, TerrainDef>
                {
                    { TerrainType.Plains, new TerrainDef(moveCost: 1, concealment: 0, defense: 0, passable: true) },
                    { TerrainType.Forest, new TerrainDef(moveCost: 2, concealment: 2, defense: 1, passable: true) },
                    { TerrainType.Rough, new TerrainDef(moveCost: 2, concealment: 1, defense: 1, passable: true) },
                    { TerrainType.Water, new TerrainDef(moveCost: 3, concealment: 0, defense: 0, passable: true) },
                },
                startingPoints: Rules.StartingPoints,
                bountyRate: Rules.BountyRate,
                generatorCost: Rules.GeneratorCost,
                generatorOutput: Rules.GeneratorOutput,
                generatorHealth: Rules.GeneratorHealth,
                damageFloor: 0,
                dmgHighGroundBonus: 1,
                rangeHighGroundBonus: 1,
                roundCap: Rules.RoundCap,
                designFee: 0,
                deployCostMultiplier: Rules.DeployCostMultiplier,
                turnPolicy: turnPolicy,
                biomesEnabled: Rules.BiomesEnabled,
                winConditions: WinBy.Annihilation,
                captureCost: 3,
                economyWinThreshold: 200,
                scoreKills: 1,
                scorePoints: 1,
                scoreArmy: 1,
                scoreTerritory: 1,
                upkeepFactor: 0.25,
                captureFactor: 4.0,
                buildFactor: 4.0,
                territoryMode: false,
                claimEndsTurn: true,
                buildAnywhere: false,
                territoryIncome: 0,
                generatorsEnabled: true,
                pointDecay: 0.0,
                fogOfWar: Rules.FogOfWar,
                maxDesignPointCost: maxDesignPointCost,
                fixedTemplateCount: fixedTemplateCount,
                templateSlotCount: templateSlotCount);
        }
    }

    [Serializable]
    public sealed class TrainingBoardConfig
    {
        public int Width = 13;
        public int Height = 9;
        public int MaxElevation = 4;
        public int ZoneDepth = 3;
        public double FlatChance = 0.6;
        public int PlainsWeight = 70;
        public int ForestWeight = 15;
        public int RoughWeight = 10;
        public int WaterWeight = 5;
    }

    [Serializable]
    public sealed class TrainingRuleConfig
    {
        public int ActionsPerTurn;
        public int RoundCap = 100;
        public int StartingPoints = 12;
        public bool FogOfWar;
        public bool BiomesEnabled;
        public double BountyRate = 0.5;
        public double DeployCostMultiplier = 1.0;
        public int GeneratorCost = 2;
        public int GeneratorOutput = 1;
        public int GeneratorHealth = 3;
    }

    [Serializable]
    public sealed class TrainingEpisodeConfig
    {
        public int MaxSteps = 600;
    }

    [Serializable]
    public sealed class TacticalRewardConfig
    {
        public float ShapeScale = 0.01f;
        public float StepPenalty = 0.005f;
        public float ClosingWeight = 0.02f;
        public float DrawCreditWeight = 0.25f;
        public float PointsWeight = 0.5f;
    }

    [Serializable]
    public sealed class AdaptiveRewardConfig
    {
        public float IntermediateDecisionPenalty = 0.001f;
        public float DeploymentCompletionBonus;
    }

    [Serializable]
    public sealed class TrainingAdaptiveConfig
    {
        public int StartingUnitCount = 6;
        public int StartingArmyBudget = 132;
        public int MaxDesignPointCost = 24;
    }
}

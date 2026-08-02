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
        public TrainingTacticalV2Config TacticalV2 = null!;

        public static TrainingScenario CreateStandard(string environment, string id = "legacy-default")
        {
            if (environment != MlContract.CurrentVersion && environment != MlContract.AdaptiveVersion
                && environment != MlContract.TacticalV2Version)
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
            else if (environment == MlContract.AdaptiveVersion)
            {
                scenario.Episode.MaxSteps = 900;
                scenario.Rules.FogOfWar = true;
                scenario.AdaptiveReward = new AdaptiveRewardConfig();
                scenario.Adaptive = new TrainingAdaptiveConfig();
            }
            else
            {
                scenario.Rules.FogOfWar = false;
                scenario.TacticalReward = new TacticalRewardConfig();
                scenario.TacticalV2 = new TrainingTacticalV2Config
                {
                    StartingUnitCount = 3,
                    MaxControllableUnits = 3,
                    PlacementPolicy = "symmetric-random-v1",
                    Templates = DefaultTacticalV2Templates(),
                };
                // Derived, not a magic constant: the RL step budget must never pre-empt the engine's own
                // round-cap backstop (see GameConfig.DefaultRoundCap / TacticalV2Config.DefaultMaxSteps).
                scenario.Episode.MaxSteps = TacticalV2Config.DefaultMaxSteps(
                    scenario.TacticalV2.StartingUnitCount, scenario.Rules.RoundCap);
            }

            return scenario;
        }

        /// <summary>The five canonical barracks templates, carried over verbatim (same stable ids,
        /// names, and stats) from <see cref="TacticalV2Config.Default"/> — so the standard scenario and
        /// the engine's own default always build byte-identical <see cref="TacticalV2Config"/> contract
        /// identities.</summary>
        private static List<TrainingUnitTemplateConfig> DefaultTacticalV2Templates()
        {
            var result = new List<TrainingUnitTemplateConfig>();
            foreach (TacticalV2Template template in TacticalV2Config.Default().Templates)
            {
                UnitStats stats = template.Template.Stats;
                result.Add(new TrainingUnitTemplateConfig
                {
                    Id = template.Id,
                    Name = template.Template.Name,
                    Health = stats.Health,
                    Damage = stats.Damage,
                    Defense = stats.Defense,
                    Movement = stats.Movement,
                    VerticalMovement = stats.VerticalMovement,
                    Range = stats.Range,
                    RangeArc = stats.RangeArc,
                    Vision = stats.Vision,
                    VisionArc = stats.VisionArc,
                });
            }
            return result;
        }

        public IReadOnlyList<string> Validate()
        {
            var errors = new List<string>();
            if (SchemaVersion != 1) errors.Add("schema version must be 1");
            bool tactical = Environment == MlContract.CurrentVersion;
            bool adaptive = Environment == MlContract.AdaptiveVersion;
            bool tacticalV2 = Environment == MlContract.TacticalV2Version;
            if (!tactical && !adaptive && !tacticalV2)
                errors.Add("environment must be tactical-v1, tactical-v2, or adaptive-v1");

            ValidateBoard(errors);
            ValidateRules(errors);
            ValidateEpisode(errors);

            if (tactical)
            {
                if (TacticalReward == null) errors.Add("tactical-v1 requires a tactical reward section");
                if (AdaptiveReward != null) errors.Add("adaptive reward section is not valid for tactical-v1");
                if (Adaptive != null) errors.Add("adaptive section is not valid for tactical-v1");
                if (TacticalV2 != null) errors.Add("tactical-v2 section is not valid for tactical-v1");
            }
            else if (adaptive)
            {
                if (TacticalReward != null) errors.Add("tactical reward section is not valid for adaptive-v1");
                if (AdaptiveReward == null) errors.Add("adaptive-v1 requires an adaptive reward section");
                if (Adaptive == null) errors.Add("adaptive-v1 requires an adaptive section");
                if (Adaptive != null) ValidateAdaptive(errors, Adaptive);
                if (TacticalV2 != null) errors.Add("tactical-v2 section is not valid for adaptive-v1");
            }
            else if (tacticalV2)
            {
                if (TacticalReward == null) errors.Add("tactical-v2 requires a tactical reward section");
                if (AdaptiveReward != null) errors.Add("adaptive reward section is not valid for tactical-v2");
                if (Adaptive != null) errors.Add("adaptive section is not valid for tactical-v2");
                if (TacticalV2 == null) errors.Add("tactical-v2 requires a tactical-v2 section");
                if (TacticalV2 != null) ValidateTacticalV2(errors, TacticalV2);
            }

            return errors;
        }

        /// <summary>Non-fatal advisories that never invalidate the scenario (contrast <see cref="Validate"/>,
        /// whose errors are hard rejections). Today this covers exactly one case: a tactical-v2
        /// <see cref="TrainingEpisodeConfig.MaxSteps"/> too small for the configured army to ever reach the
        /// engine's own round cap (<see cref="TrainingRuleConfig.RoundCap"/>), so the RL step budget would
        /// truncate the episode first and report a draw the game itself never reached. This is surfaced as
        /// a warning — not an error — so scenario.json files written before this check existed (including
        /// long-running, already-checkpointed training runs) keep loading for resume/Arena unchanged; a
        /// hard rejection belongs only at new-run creation time, one layer up.</summary>
        public IReadOnlyList<string> Warnings()
        {
            var warnings = new List<string>();
            if (Environment == MlContract.TacticalV2Version && Rules != null && Episode != null && TacticalV2 != null)
            {
                int minimum = TacticalV2Config.MinimumMaxSteps(TacticalV2.StartingUnitCount, Rules.RoundCap);
                if (Episode.MaxSteps < minimum)
                {
                    warnings.Add(
                        $"tactical-v2 episode.max_steps ({Episode.MaxSteps}) is insufficient to reach the " +
                        $"round cap ({Rules.RoundCap}) for {TacticalV2.StartingUnitCount} starting units; " +
                        $"minimum required is {minimum}");
                }
            }
            return warnings;
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

        /// <summary>Maps the DTO into a <see cref="TacticalV2Config"/> without reading Unity state —
        /// same boundary as <see cref="BuildTactical"/>/<see cref="BuildAdaptive"/>. Template ids are
        /// carried over verbatim from the DTO (never re-derived), so a caller-assigned id such as a
        /// saved custom design's id round-trips exactly.</summary>
        public TacticalV2Config BuildTacticalV2()
        {
            ThrowIfInvalid();
            if (Environment != MlContract.TacticalV2Version)
                throw new ArgumentException("scenario environment must be tactical-v2", nameof(Environment));

            TacticalRewardConfig reward = TacticalReward!;
            TrainingTacticalV2Config tacticalV2 = TacticalV2!;
            var templates = new List<TacticalV2Template>(tacticalV2.Templates.Count);
            foreach (TrainingUnitTemplateConfig item in tacticalV2.Templates)
            {
                var stats = new UnitStats(
                    item.Health, item.Damage, item.Defense,
                    item.Movement, item.VerticalMovement,
                    item.Range, item.RangeArc,
                    item.Vision, item.VisionArc);
                templates.Add(new TacticalV2Template(item.Id, new UnitTemplate(UnitTemplate.Sanitize(item.Name), stats)));
            }

            return new TacticalV2Config
            {
                BoardGen = BuildBoardGen(),
                Game = BuildGameConfig(
                    fixedTemplateCount: templates.Count,
                    templateSlotCount: templates.Count,
                    captureCost: int.MaxValue,
                    generatorsEnabled: false),
                Templates = templates.AsReadOnly(),
                StartingUnitCount = tacticalV2.StartingUnitCount,
                MaxControllableUnits = tacticalV2.MaxControllableUnits,
                MaxSteps = Episode.MaxSteps,
                ShapeScale = reward.ShapeScale,
                StepPenalty = reward.StepPenalty,
                ClosingWeight = reward.ClosingWeight,
                DrawCreditWeight = reward.DrawCreditWeight,
                PointsWeight = reward.PointsWeight,
                PlacementPolicy = tacticalV2.PlacementPolicy,
                StartProfiles = tacticalV2.StartProfiles.AsReadOnly(),
                StartDistribution = new TacticalV2StartDistribution(tacticalV2.StartDistribution),
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

        /// <summary>Mirrors <see cref="TacticalV2Config.Validate"/>'s own invariants (count range,
        /// starting-count/controllable-unit-cap lockstep, placement policy, non-empty and duplicate-free
        /// catalog) at the DTO layer, plus a deployment-cell capacity check like
        /// <see cref="ValidateAdaptive"/>'s — so a scenario can never reach <see cref="TacticalV2Layout"/>
        /// with a controllable-unit cap the board or the starting roster can't honor (the crash surface
        /// is <see cref="TacticalV2UnitRegistry"/>, which throws rather than silently truncating).</summary>
        private void ValidateTacticalV2(List<string> errors, TrainingTacticalV2Config tacticalV2)
        {
            if (tacticalV2.StartingUnitCount < 1 || tacticalV2.StartingUnitCount > 12)
                errors.Add("tactical-v2 starting unit count must be between 1 and 12");
            if (tacticalV2.MaxControllableUnits != tacticalV2.StartingUnitCount)
                errors.Add("tactical-v2 max controllable units must equal starting unit count");
            if (tacticalV2.PlacementPolicy == "profiled-seeded-v1")
            {
                TacticalV2Config profiled = TacticalV2Config.Default();
                profiled.StartingUnitCount = tacticalV2.StartingUnitCount;
                profiled.MaxControllableUnits = tacticalV2.MaxControllableUnits;
                profiled.PlacementPolicy = tacticalV2.PlacementPolicy;
                profiled.StartProfiles = tacticalV2.StartProfiles.AsReadOnly();
                profiled.StartDistribution = new TacticalV2StartDistribution(tacticalV2.StartDistribution);
                foreach (string error in profiled.Validate())
                    if (error.Contains("profile") || error.Contains("start distribution") ||
                        error.Contains("starting unit count and max controllable units"))
                        errors.Add("tactical-v2 " + error);
            }
            else if (tacticalV2.PlacementPolicy != "symmetric-random-v1")
                errors.Add("tactical-v2 placement policy must be 'symmetric-random-v1'");

            if (tacticalV2.Templates == null || tacticalV2.Templates.Count == 0)
            {
                errors.Add("tactical-v2 template catalog must not be empty");
            }
            else
            {
                var seenIds = new HashSet<string>(StringComparer.Ordinal);
                foreach (TrainingUnitTemplateConfig template in tacticalV2.Templates)
                {
                    if (string.IsNullOrEmpty(template.Id))
                        errors.Add("tactical-v2 template ids must not be empty");
                    else if (!seenIds.Add(template.Id))
                        errors.Add($"duplicate tactical-v2 template id '{template.Id}'");

                    if (template.Health < 0 || template.Damage < 0 || template.Defense < 0
                        || template.Movement < 0 || template.VerticalMovement < 0
                        || template.Range < 0 || template.RangeArc < 0
                        || template.Vision < 0 || template.VisionArc < 0)
                        errors.Add($"tactical-v2 template '{template.Id}' has an invalid stat");
                }
            }

            if (Board != null && Board.Height > 0 && Board.ZoneDepth > 0)
            {
                long cellsPerSeat = (long)Board.Height * Board.ZoneDepth;
                if (cellsPerSeat < tacticalV2.StartingUnitCount)
                    errors.Add("tactical-v2 deployment cells must cover the starting unit count");
            }
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
            int templateSlotCount = 0,
            int captureCost = 3,
            bool generatorsEnabled = true)
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
                captureCost: captureCost,
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
                generatorsEnabled: generatorsEnabled,
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
        public int RoundCap = GameConfig.DefaultRoundCap;
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

    [Serializable]
    public sealed class TrainingTacticalV2Config
    {
        public int StartingUnitCount = 3;
        public List<TacticalV2StartProfile> StartProfiles = new List<TacticalV2StartProfile>();
        public List<TacticalV2StartWeight> StartDistribution = new List<TacticalV2StartWeight>();
        public int MaxControllableUnits = 3;
        public string PlacementPolicy = "symmetric-random-v1";
        public List<TrainingUnitTemplateConfig> Templates = new List<TrainingUnitTemplateConfig>();
    }

    [Serializable]
    public sealed class TrainingUnitTemplateConfig
    {
        public string Id = string.Empty;
        public string Name = string.Empty;
        public int Health;
        public int Damage;
        public int Defense;
        public int Movement;
        public int VerticalMovement;
        public int Range;
        public int RangeArc;
        public int Vision;
        public int VisionArc;
    }
}

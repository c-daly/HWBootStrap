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
        public TrainingTacticalV3RewardConfig TacticalV3Reward = null!;
        public TrainingTacticalV3Config TacticalV3 = null!;

        public static TrainingScenario CreateStandard(string environment, string id = "legacy-default")
        {
            if (environment != MlContract.CurrentVersion && environment != MlContract.AdaptiveVersion
                && environment != MlContract.TacticalV2Version && environment != MlContract.TacticalV3Version)
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
            else if (environment == MlContract.TacticalV2Version)
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
            else
            {
                scenario.Rules.FogOfWar = false;
                scenario.TacticalV3Reward = new TrainingTacticalV3RewardConfig();
                scenario.TacticalV3 = new TrainingTacticalV3Config
                {
                    StartingUnitCount = 3,
                    MaxControllableUnits = 3,
                    PlacementPolicy = "symmetric-random-v1",
                    Templates = DefaultTacticalV2Templates(),
                };
                scenario.Episode.MaxSteps = TacticalV2Config.DefaultMaxSteps(
                    scenario.TacticalV3.StartingUnitCount, scenario.Rules.RoundCap);
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
            bool tacticalV3 = Environment == MlContract.TacticalV3Version;
            if (!tactical && !adaptive && !tacticalV2 && !tacticalV3)
                errors.Add("environment must be tactical-v1, tactical-v2, tactical-v3, or adaptive-v1");

            ValidateBoard(errors);
            ValidateRules(errors);
            ValidateEpisode(errors);

            if (tactical)
            {
                if (TacticalReward == null) errors.Add("tactical-v1 requires a tactical reward section");
                if (AdaptiveReward != null) errors.Add("adaptive reward section is not valid for tactical-v1");
                if (Adaptive != null) errors.Add("adaptive section is not valid for tactical-v1");
                if (TacticalV2 != null) errors.Add("tactical-v2 section is not valid for tactical-v1");
                if (TacticalV3Reward != null) errors.Add("tactical-v3 reward section is not valid for tactical-v1");
                if (TacticalV3 != null) errors.Add("tactical-v3 section is not valid for tactical-v1");
            }
            else if (adaptive)
            {
                if (TacticalReward != null) errors.Add("tactical reward section is not valid for adaptive-v1");
                if (AdaptiveReward == null) errors.Add("adaptive-v1 requires an adaptive reward section");
                if (Adaptive == null) errors.Add("adaptive-v1 requires an adaptive section");
                if (Adaptive != null) ValidateAdaptive(errors, Adaptive);
                if (TacticalV2 != null) errors.Add("tactical-v2 section is not valid for adaptive-v1");
                if (TacticalV3Reward != null) errors.Add("tactical-v3 reward section is not valid for adaptive-v1");
                if (TacticalV3 != null) errors.Add("tactical-v3 section is not valid for adaptive-v1");
            }
            else if (tacticalV2)
            {
                if (TacticalReward == null) errors.Add("tactical-v2 requires a tactical reward section");
                if (AdaptiveReward != null) errors.Add("adaptive reward section is not valid for tactical-v2");
                if (Adaptive != null) errors.Add("adaptive section is not valid for tactical-v2");
                if (TacticalV2 == null) errors.Add("tactical-v2 requires a tactical-v2 section");
                if (TacticalV2 != null) ValidateTacticalV2(errors, TacticalV2);
                if (TacticalV3Reward != null) errors.Add("tactical-v3 reward section is not valid for tactical-v2");
                if (TacticalV3 != null) errors.Add("tactical-v3 section is not valid for tactical-v2");
            }
            else if (tacticalV3)
            {
                if (TacticalReward != null) errors.Add("tactical reward section is not valid for tactical-v3");
                if (AdaptiveReward != null) errors.Add("adaptive reward section is not valid for tactical-v3");
                if (Adaptive != null) errors.Add("adaptive section is not valid for tactical-v3");
                if (TacticalV2 != null) errors.Add("tactical-v2 section is not valid for tactical-v3");
                if (TacticalV3Reward == null) errors.Add("tactical-v3 requires a tactical-v3 reward section");
                if (TacticalV3 == null) errors.Add("tactical-v3 requires a tactical-v3 section");
                if (TacticalV3Reward != null) ValidateTacticalV3Reward(errors, TacticalV3Reward);
                if (TacticalV3 != null) ValidateTacticalV3(errors, TacticalV3);
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
            return BuildTacticalMatch(TacticalV2!, reward.PointsWeight, reward.ShapeScale,
                reward.StepPenalty, reward.ClosingWeight, reward.DrawCreditWeight);
        }

        public TacticalV3Config BuildTacticalV3()
        {
            ThrowIfInvalid();
            if (Environment != MlContract.TacticalV3Version)
                throw new ArgumentException("scenario environment must be tactical-v3", nameof(Environment));

            TrainingTacticalV3Config source = TacticalV3!;
            TrainingTacticalV3RewardConfig reward = TacticalV3Reward!;
            TacticalV2Config match = BuildTacticalMatch(source, reward.PointsWeight, 0f, 0f, 0f, 0f);
            var capacity = new TacticalV3CapacityProfile(
                source.Capacity.MaxCells, source.Capacity.MaxUnits, source.Capacity.MaxTemplates,
                source.Capacity.MaxCapabilityDefinitions, source.Capacity.MaxCapabilityAllocations,
                source.Capacity.MaxRules, source.Capacity.MaxMemoryRecords, source.Capacity.MaxRelations,
                source.Capacity.MaxCandidates);
            var runtimeReward = new TacticalV3RewardConfig(
                reward.TerminalWin, reward.TerminalNonWin, reward.MaterialAdjustmentBound,
                reward.TimePressureBound, reward.PointsWeight);
            TacticalV3ObjectiveConfig? objective = source.Objective == null
                ? null
                : new TacticalV3ObjectiveConfig(
                    source.Objective.Kind,
                    source.Objective.TargetPolicy,
                    source.Objective.Radius);
            return new TacticalV3Config(match, capacity, runtimeReward, objective);
        }

        private TacticalV2Config BuildTacticalMatch(
            TrainingTacticalV2Config source, float pointsWeight, float shapeScale, float stepPenalty,
            float closingWeight, float drawCreditWeight) =>
            BuildTacticalMatch(source.StartingUnitCount, source.MaxControllableUnits, source.PlacementPolicy,
                source.Templates, source.StartProfiles, source.StartDistribution, pointsWeight, shapeScale,
                stepPenalty, closingWeight, drawCreditWeight);

        private TacticalV2Config BuildTacticalMatch(
            TrainingTacticalV3Config source, float pointsWeight, float shapeScale, float stepPenalty,
            float closingWeight, float drawCreditWeight) =>
            BuildTacticalMatch(source.StartingUnitCount, source.MaxControllableUnits, source.PlacementPolicy,
                source.Templates, source.StartProfiles, source.StartDistribution, pointsWeight, shapeScale,
                stepPenalty, closingWeight, drawCreditWeight);

        private TacticalV2Config BuildTacticalMatch(
            int startingUnitCount, int maxControllableUnits, string placementPolicy,
            List<TrainingUnitTemplateConfig> sourceTemplates, List<TacticalV2StartProfile> startProfiles,
            List<TacticalV2StartWeight> startDistribution, float pointsWeight, float shapeScale,
            float stepPenalty, float closingWeight, float drawCreditWeight)
        {
            var templates = new List<TacticalV2Template>(sourceTemplates.Count);
            foreach (TrainingUnitTemplateConfig item in sourceTemplates)
            {
                var stats = new UnitStats(
                    item.Health, item.Damage, item.Defense,
                    item.Movement, item.VerticalMovement,
                    item.Range, item.RangeArc,
                    item.Vision, item.VisionArc);
                templates.Add(new TacticalV2Template(item.Id,
                    new UnitTemplate(UnitTemplate.Sanitize(item.Name), stats)));
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
                StartingUnitCount = startingUnitCount,
                MaxControllableUnits = maxControllableUnits,
                MaxSteps = Episode.MaxSteps,
                ShapeScale = shapeScale,
                StepPenalty = stepPenalty,
                ClosingWeight = closingWeight,
                DrawCreditWeight = drawCreditWeight,
                PointsWeight = pointsWeight,
                PlacementPolicy = placementPolicy,
                StartProfiles = startProfiles.AsReadOnly(),
                StartDistribution = new TacticalV2StartDistribution(startDistribution),
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
        private void ValidateTacticalV2(List<string> errors, TrainingTacticalV2Config tacticalV2) =>
            ValidateTacticalMatch(errors, "tactical-v2", tacticalV2.StartingUnitCount,
                tacticalV2.MaxControllableUnits, tacticalV2.PlacementPolicy, tacticalV2.Templates,
                tacticalV2.StartProfiles, tacticalV2.StartDistribution);

        private void ValidateTacticalV3(List<string> errors, TrainingTacticalV3Config tacticalV3)
        {
            ValidateTacticalMatch(errors, "tactical-v3", tacticalV3.StartingUnitCount,
                tacticalV3.MaxControllableUnits, tacticalV3.PlacementPolicy, tacticalV3.Templates,
                tacticalV3.StartProfiles, tacticalV3.StartDistribution, requirePositiveHealth: true);

            if (Rules != null && Rules.FogOfWar)
                errors.Add("tactical-v3 stage one requires fog_of_war=false");
            if (Rules != null && Episode != null && tacticalV3.StartingUnitCount > 0 &&
                Rules.RoundCap > 0)
            {
                long actionsPerRound = checked(2L * (tacticalV3.StartingUnitCount + 1L));
                long minimumMaxSteps = checked(actionsPerRound * Rules.RoundCap);
                if (Episode.MaxSteps < minimumMaxSteps)
                    errors.Add("tactical-v3 episode max steps are insufficient to reach the round cap");
            }
            if (tacticalV3.PlacementPolicy == "symmetric-random-v1" &&
                ((tacticalV3.StartProfiles != null && tacticalV3.StartProfiles.Count != 0) ||
                 (tacticalV3.StartDistribution != null && tacticalV3.StartDistribution.Count != 0)))
            {
                errors.Add(
                    "tactical-v3 symmetric-random-v1 must not declare start profiles or a start distribution");
            }

            TrainingTacticalV3ObjectiveConfig? objective = tacticalV3.Objective;
            if (objective != null)
            {
                if (objective.Kind != TacticalV3ObjectiveConfig.ReachCellKind)
                    errors.Add("tactical-v3 objective kind must be 'reach_cell'");
                if (objective.TargetPolicy !=
                    TacticalV3ObjectiveConfig.SeededFarthestReachableUnoccupiedPolicy)
                {
                    errors.Add(
                        "tactical-v3 reach-cell target policy must be '" +
                        TacticalV3ObjectiveConfig.SeededFarthestReachableUnoccupiedPolicy + "'");
                }
                if (objective.Radius != 0)
                    errors.Add("tactical-v3 reach-cell radius must be 0");
            }

            TrainingTacticalV3CapacityConfig? capacity = tacticalV3.Capacity;
            if (capacity == null)
            {
                errors.Add("tactical-v3 capacity section is required");
                return;
            }

            int[] values =
            {
                capacity.MaxCells, capacity.MaxUnits, capacity.MaxTemplates,
                capacity.MaxCapabilityDefinitions, capacity.MaxCapabilityAllocations,
                capacity.MaxRules, capacity.MaxMemoryRecords, capacity.MaxRelations,
                capacity.MaxCandidates,
            };
            if (values.Any(value => value <= 0))
                errors.Add("tactical-v3 capacity values must be positive");

            long cellCount;
            long maximumTotalUnits = checked((long)tacticalV3.StartingUnitCount * 2);
            if (tacticalV3.PlacementPolicy == "profiled-seeded-v1" &&
                tacticalV3.StartProfiles != null)
            {
                foreach (TacticalV2StartProfile profile in tacticalV3.StartProfiles)
                {
                    if (profile == null) continue;
                    long total = checked((long)profile.LearnerUnitCount + profile.OpponentUnitCount);
                    maximumTotalUnits = Math.Max(maximumTotalUnits, total);
                }
            }

            long templateCount = tacticalV3.Templates == null ? 0 : tacticalV3.Templates.Count;
            long templateRows;
            long allocations;
            long minimumRelations;
            long candidateCapacityRequirement;
            try
            {
                long width = Math.Max(0L, Board == null ? 0L : Board.Width);
                long height = Math.Max(0L, Board == null ? 0L : Board.Height);
                long declaredUnitCapacity = Math.Max(0L, capacity.MaxUnits);
                cellCount = checked(width * height);
                templateRows = checked(templateCount * 2);
                int definitions = TacticalV3Capabilities.All.Count;
                allocations = checked((declaredUnitCapacity + templateRows) * definitions);

                // An odd-q width-by-height rectangle has height-1 vertical edges per column and
                // 2*height-1 edges across each adjacent column pair. Relations store both directions.
                long directedAdjacency = width == 0 || height == 0 ? 0 : checked(2L * checked(
                    checked(width * (height - 1L)) +
                    checked((width - 1L) * checked(2L * height - 1L))));

                // Every live unit contributes one occupancy and nine capability-allocation relations;
                // both seat-relative template catalogs contribute exactly two rows per template.
                minimumRelations = checked(directedAdjacency + declaredUnitCapacity + allocations);

                // This is a safe structural upper-bound capacity requirement, not an exact LegalMoves
                // count: one end turn, every template/deployment-cell deploy, every unit/cell move,
                // and every ordered distinct unit pair as a possible active-seat attack.
                long deploymentCells = checked(height *
                    Math.Max(0L, Board == null ? 0L : Board.ZoneDepth));
                long orderedAttackPairs = checked(
                    declaredUnitCapacity * Math.Max(0L, declaredUnitCapacity - 1L));
                candidateCapacityRequirement = checked(
                    checked(templateCount * deploymentCells) +
                    checked(declaredUnitCapacity * cellCount) +
                    orderedAttackPairs + 1L);
            }
            catch (OverflowException)
            {
                errors.Add("tactical-v3 static capacity requirements overflow Int64");
                return;
            }

            if (capacity.MaxCells < cellCount)
                errors.Add("tactical-v3 max cells capacity is smaller than the board");
            if (capacity.MaxUnits < maximumTotalUnits)
                errors.Add("tactical-v3 max units capacity is smaller than a declared start profile");
            if (capacity.MaxTemplates < templateRows)
                errors.Add("tactical-v3 max templates capacity is smaller than both template catalogs");
            if (capacity.MaxCapabilityDefinitions < TacticalV3Capabilities.All.Count)
                errors.Add("tactical-v3 max capability definitions capacity is undersized");
            if (capacity.MaxCapabilityAllocations < allocations)
                errors.Add("tactical-v3 max capability allocations capacity is undersized");
            if (capacity.MaxRules < 15)
                errors.Add("tactical-v3 max rules capacity is undersized");
            if (capacity.MaxMemoryRecords < maximumTotalUnits)
                errors.Add("tactical-v3 max memory records capacity is undersized");
            if (capacity.MaxRelations < minimumRelations)
                errors.Add("tactical-v3 max relations capacity is undersized");
            if (capacity.MaxCandidates < candidateCapacityRequirement)
                errors.Add("tactical-v3 max candidates capacity is undersized");
        }

        private static void ValidateTacticalV3Reward(
            List<string> errors, TrainingTacticalV3RewardConfig reward)
        {
            if (reward.TerminalWin != 1f || reward.TerminalNonWin != -1f)
                errors.Add("tactical-v3 terminal rewards must be +1/-1");
            if (reward.MaterialAdjustmentBound != 0.2f || reward.TimePressureBound != 0.05f)
                errors.Add("tactical-v3 shaping bounds must be 0.20/0.05");
            if (reward.PointsWeight != 0.5f)
                errors.Add("tactical-v3 points weight must be 0.5");
        }

        private void ValidateTacticalMatch(
            List<string> errors, string version, int startingUnitCount, int maxControllableUnits,
            string placementPolicy, List<TrainingUnitTemplateConfig> templates,
            List<TacticalV2StartProfile> startProfiles, List<TacticalV2StartWeight> startDistribution,
            bool requirePositiveHealth = false)
        {
            if (startingUnitCount < 1 || startingUnitCount > 12)
                errors.Add(version + " starting unit count must be between 1 and 12");
            if (maxControllableUnits != startingUnitCount)
                errors.Add(version + " max controllable units must equal starting unit count");
            bool validProfileCollections = true;
            if (startProfiles == null)
            {
                errors.Add(version + " start profiles collection is required");
                validProfileCollections = false;
            }
            else if (startProfiles.Any(profile => profile == null))
            {
                errors.Add(version + " start profiles contain a null element");
                validProfileCollections = false;
            }
            if (startDistribution == null)
            {
                errors.Add(version + " start distribution collection is required");
                validProfileCollections = false;
            }
            else if (startDistribution.Any(weight => weight == null))
            {
                errors.Add(version + " start distribution contains a null element");
                validProfileCollections = false;
            }

            if (placementPolicy == "profiled-seeded-v1" && validProfileCollections)
            {
                TacticalV2Config profiled = TacticalV2Config.Default();
                profiled.StartingUnitCount = startingUnitCount;
                profiled.MaxControllableUnits = maxControllableUnits;
                profiled.PlacementPolicy = placementPolicy;
                profiled.StartProfiles = startProfiles!.AsReadOnly();
                profiled.StartDistribution = new TacticalV2StartDistribution(startDistribution!);
                foreach (string error in profiled.Validate())
                    if (error.Contains("profile") || error.Contains("start distribution") ||
                        error.Contains("starting unit count and max controllable units"))
                        errors.Add(version + " " + error);
            }
            else if (placementPolicy != "symmetric-random-v1")
                errors.Add(version + " placement policy must be 'symmetric-random-v1'");

            if (templates == null)
            {
                errors.Add(version + " template catalog is required");
            }
            else if (templates.Count == 0)
            {
                errors.Add(version + " template catalog must not be empty");
            }
            else
            {
                var seenIds = new HashSet<string>(StringComparer.Ordinal);
                foreach (TrainingUnitTemplateConfig template in templates)
                {
                    if (template == null)
                    {
                        errors.Add(version + " template catalog contains a null element");
                        continue;
                    }

                    if (string.IsNullOrEmpty(template.Id))
                        errors.Add(version + " template ids must not be empty");
                    else if (!seenIds.Add(template.Id))
                        errors.Add($"duplicate {version} template id '{template.Id}'");

                    if ((requirePositiveHealth ? template.Health < 1 : template.Health < 0) ||
                        template.Damage < 0 || template.Defense < 0
                        || template.Movement < 0 || template.VerticalMovement < 0
                        || template.Range < 0 || template.RangeArc < 0
                        || template.Vision < 0 || template.VisionArc < 0)
                        errors.Add($"{version} template '{template.Id}' has an invalid stat");
                }
            }

            if (Board != null && Board.Height > 0 && Board.ZoneDepth > 0)
            {
                long cellsPerSeat = (long)Board.Height * Board.ZoneDepth;
                if (cellsPerSeat < startingUnitCount)
                    errors.Add(version + " deployment cells must cover the starting unit count");
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
    public sealed class TrainingTacticalV3RewardConfig
    {
        public float TerminalWin = 1f;
        public float TerminalNonWin = -1f;
        public float MaterialAdjustmentBound = 0.2f;
        public float TimePressureBound = 0.05f;
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
    public sealed class TrainingTacticalV3Config
    {
        public int StartingUnitCount = 3;
        public List<TacticalV2StartProfile> StartProfiles = new List<TacticalV2StartProfile>();
        public List<TacticalV2StartWeight> StartDistribution = new List<TacticalV2StartWeight>();
        public int MaxControllableUnits = 3;
        public string PlacementPolicy = "symmetric-random-v1";
        public List<TrainingUnitTemplateConfig> Templates = new List<TrainingUnitTemplateConfig>();
        public TrainingTacticalV3CapacityConfig Capacity = new TrainingTacticalV3CapacityConfig();
        public TrainingTacticalV3ObjectiveConfig? Objective;
    }

    [Serializable]
    public sealed class TrainingTacticalV3ObjectiveConfig
    {
        public string Kind = TacticalV3ObjectiveConfig.ReachCellKind;
        public string TargetPolicy =
            TacticalV3ObjectiveConfig.SeededFarthestReachableUnoccupiedPolicy;
        public int Radius;
    }

    [Serializable]
    public sealed class TrainingTacticalV3CapacityConfig
    {
        public int MaxCells = 512;
        public int MaxUnits = 64;
        public int MaxTemplates = 32;
        public int MaxCapabilityDefinitions = 128;
        public int MaxCapabilityAllocations = 2048;
        public int MaxRules = 128;
        public int MaxMemoryRecords = 64;
        public int MaxRelations = 65536;
        public int MaxCandidates = 32768;
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

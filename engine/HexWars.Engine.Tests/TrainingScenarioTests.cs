using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TrainingScenarioTests
    {
        [Test]
        public void TacticalScenario_BuildsEveryConfigurableValue()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 24;
            scenario.Board.Height = 16;
            scenario.Board.MaxElevation = 6;
            scenario.Board.ZoneDepth = 4;
            scenario.Board.FlatChance = 0.75;
            scenario.Board.PlainsWeight = 50;
            scenario.Board.ForestWeight = 20;
            scenario.Board.RoughWeight = 20;
            scenario.Board.WaterWeight = 10;
            scenario.Rules.ActionsPerTurn = 7;
            scenario.Rules.RoundCap = 150;
            scenario.Rules.StartingPoints = 24;
            scenario.Rules.FogOfWar = true;
            scenario.Rules.BiomesEnabled = true;
            scenario.Rules.BountyRate = 0.75;
            scenario.Rules.DeployCostMultiplier = 1.25;
            scenario.Rules.GeneratorCost = 4;
            scenario.Rules.GeneratorOutput = 3;
            scenario.Rules.GeneratorHealth = 5;
            scenario.Episode.MaxSteps = 1200;
            scenario.TacticalReward.ShapeScale = 0.02f;
            scenario.TacticalReward.StepPenalty = 0.04f;
            scenario.TacticalReward.ClosingWeight = 0.05f;
            scenario.TacticalReward.DrawCreditWeight = 0.3f;
            scenario.TacticalReward.PointsWeight = 0.6f;

            EnvConfig config = scenario.BuildTactical();

            Assert.That(config.BoardGen.Width, Is.EqualTo(24));
            Assert.That(config.BoardGen.Height, Is.EqualTo(16));
            Assert.That(config.BoardGen.MaxElevation, Is.EqualTo(6));
            Assert.That(config.BoardGen.ZoneDepth, Is.EqualTo(4));
            Assert.That(config.BoardGen.FlatChance, Is.EqualTo(0.75));
            Assert.That(config.BoardGen.PlainsWeight, Is.EqualTo(50));
            Assert.That(config.BoardGen.ForestWeight, Is.EqualTo(20));
            Assert.That(config.BoardGen.RoughWeight, Is.EqualTo(20));
            Assert.That(config.BoardGen.WaterWeight, Is.EqualTo(10));
            Assert.That(config.Game.TurnPolicy.ActionsPerTurn, Is.EqualTo(7));
            Assert.That(config.Game.RoundCap, Is.EqualTo(150));
            Assert.That(config.Game.StartingPoints, Is.EqualTo(24));
            Assert.That(config.Game.FogOfWar, Is.True);
            Assert.That(config.Game.BiomesEnabled, Is.True);
            Assert.That(config.Game.BountyRate, Is.EqualTo(0.75));
            Assert.That(config.Game.DeployCostMultiplier, Is.EqualTo(1.25));
            Assert.That(config.Game.GeneratorCost, Is.EqualTo(4));
            Assert.That(config.Game.GeneratorOutput, Is.EqualTo(3));
            Assert.That(config.Game.GeneratorHealth, Is.EqualTo(5));
            Assert.That(config.MaxSteps, Is.EqualTo(1200));
            Assert.That(config.ShapeScale, Is.EqualTo(0.02f));
            Assert.That(config.StepPenalty, Is.EqualTo(0.04f));
            Assert.That(config.ClosingWeight, Is.EqualTo(0.05f));
            Assert.That(config.DrawCreditWeight, Is.EqualTo(0.3f));
            Assert.That(config.PointsWeight, Is.EqualTo(0.6f));
        }

        [Test]
        public void ZeroActionsPerTurn_BuildsAllUnitsPolicy()
        {
            EnvConfig config = TrainingScenario.CreateStandard("tactical-v1").BuildTactical();

            Assert.That(config.Game.TurnPolicy, Is.TypeOf<AllUnitsPolicy>());
            Assert.That(config.Game.TurnPolicy.ActionsPerTurn, Is.Null);
        }

        [Test]
        public void AdaptiveScenario_PreservesPinnedArchitecture()
        {
            var scenario = TrainingScenario.CreateStandard("adaptive-v1");
            scenario.Adaptive.StartingUnitCount = 7;
            scenario.Adaptive.StartingArmyBudget = 160;
            scenario.Adaptive.MaxDesignPointCost = 24;
            scenario.AdaptiveReward.IntermediateDecisionPenalty = 0.002f;
            scenario.AdaptiveReward.DeploymentCompletionBonus = 0.1f;

            AdaptiveEnvConfig config = scenario.BuildAdaptive();

            Assert.That(config.MaxControllableUnits, Is.EqualTo(24));
            Assert.That(config.Templates, Has.Count.EqualTo(9));
            Assert.That(config.FixedTemplateCount, Is.EqualTo(6));
            Assert.That(config.CustomTemplateCount, Is.EqualTo(3));
            Assert.That(config.StartingUnitCount, Is.EqualTo(7));
            Assert.That(config.StartingArmyBudget, Is.EqualTo(160));
            Assert.That(config.MaxDesignPointCost, Is.EqualTo(24));
            Assert.That(config.IntermediateDecisionPenalty, Is.EqualTo(0.002f));
            Assert.That(config.DeploymentCompletionBonus, Is.EqualTo(0.1f));
        }

        [TestCase(0, 9, 3, "width")]
        [TestCase(13, 0, 3, "height")]
        [TestCase(7, 9, 4, "deployment zones overlap")]
        public void Scenario_RejectsImpossibleGeometry(int width, int height, int zoneDepth, string message)
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = width;
            scenario.Board.Height = height;
            scenario.Board.ZoneDepth = zoneDepth;

            Assert.That(scenario.Validate(), Has.Some.Contains(message));
        }

        [Test]
        public void Scenario_AcceptsOddWidthWithDisjointDeploymentZones()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 7;
            scenario.Board.ZoneDepth = 3;

            Assert.That(scenario.Validate(), Has.None.Contains("deployment zones overlap"));
        }

        [Test]
        public void Scenario_AcceptsEvenWidthFullyPartitionedDeploymentZones()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 6;
            scenario.Board.ZoneDepth = 3;

            Assert.That(scenario.Validate(), Has.None.Contains("deployment zones overlap"));
        }

        [Test]
        public void Scenario_ReportsEachInvalidScalarField()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.MaxElevation = -1;
            scenario.Board.FlatChance = 1.1;
            scenario.Board.PlainsWeight = -1;
            scenario.Board.ForestWeight = 0;
            scenario.Board.RoughWeight = 0;
            scenario.Board.WaterWeight = 0;
            scenario.Rules.ActionsPerTurn = -1;
            scenario.Rules.RoundCap = 0;
            scenario.Episode.MaxSteps = 0;

            var errors = scenario.Validate();

            Assert.That(errors, Has.Some.Contains("max elevation"));
            Assert.That(errors, Has.Some.Contains("flat chance"));
            Assert.That(errors, Has.Some.Contains("plains weight"));
            Assert.That(errors, Has.Some.Contains("terrain weight sum"));
            Assert.That(errors, Has.Some.Contains("actions per turn"));
            Assert.That(errors, Has.Some.Contains("round cap"));
            Assert.That(errors, Has.Some.Contains("max steps"));
        }

        [Test]
        public void AdaptiveScenario_RejectsInsufficientDeploymentCells()
        {
            var scenario = TrainingScenario.CreateStandard("adaptive-v1");
            scenario.Board.Width = 2;
            scenario.Board.Height = 2;
            scenario.Board.ZoneDepth = 1;

            Assert.That(scenario.Validate(), Has.Some.Contains("deployment cells"));
        }

        [Test]
        public void AdaptiveScenario_UsesHeightForDeploymentCellCapacity()
        {
            var scenario = TrainingScenario.CreateStandard("adaptive-v1");
            scenario.Board.Width = 20;
            scenario.Board.Height = 2;
            scenario.Board.ZoneDepth = 2;

            Assert.That(scenario.Validate(), Has.Some.Contains("deployment cells"));
        }

        [Test]
        public void AdaptiveScenario_RejectsInsufficientBudget()
        {
            var scenario = TrainingScenario.CreateStandard("adaptive-v1");
            scenario.Adaptive.StartingArmyBudget = 1;

            Assert.That(scenario.Validate(), Has.Some.Contains("starting army budget"));
        }

        [Test]
        public void Scenario_RejectsEnvironmentSectionMismatch()
        {
            var tactical = TrainingScenario.CreateStandard("tactical-v1");
            tactical.Adaptive = new TrainingAdaptiveConfig();
            var adaptive = TrainingScenario.CreateStandard("adaptive-v1");
            adaptive.TacticalReward = new TacticalRewardConfig();

            Assert.That(tactical.Validate(), Has.Some.Contains("adaptive section"));
            Assert.That(adaptive.Validate(), Has.Some.Contains("tactical reward"));
        }

        [Test]
        public void Builders_ThrowAllValidationErrors()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 0;
            scenario.Rules.RoundCap = 0;

            var error = Assert.Throws<ArgumentException>(() => scenario.BuildTactical());

            Assert.That(error!.Message, Does.Contain("width").And.Contain("round cap"));
        }

        [Test]
        public void StandardScenarios_PreserveExistingDefaultContractIdentities()
        {
            MlContract tactical = MlContract.Create(
                TrainingScenario.CreateStandard("tactical-v1").BuildTactical());
            MlContract tacticalDefault = MlContract.Create(new EnvConfig());
            MlContract adaptive = MlContract.CreateAdaptive(
                TrainingScenario.CreateStandard("adaptive-v1").BuildAdaptive());
            MlContract adaptiveDefault = MlContract.CreateAdaptive(AdaptiveEnvConfig.Default());

            Assert.That(tactical.ContractHash, Is.EqualTo(tacticalDefault.ContractHash));
            Assert.That(tactical.EncodingHash, Is.EqualTo(tacticalDefault.EncodingHash));
            Assert.That(adaptive.ContractHash, Is.EqualTo(adaptiveDefault.ContractHash));
            Assert.That(adaptive.EncodingHash, Is.EqualTo(adaptiveDefault.EncodingHash));
        }

        [Test]
        public void RewardOrHorizonChange_ChangesContractButNotEncodingIdentity()
        {
            var first = TrainingScenario.CreateStandard("tactical-v1").BuildTactical();
            var changed = TrainingScenario.CreateStandard("tactical-v1").BuildTactical();
            changed.ShapeScale = 0.2f;
            changed.MaxSteps = 1200;

            MlContract a = MlContract.Create(first);
            MlContract b = MlContract.Create(changed);

            Assert.That(b.ContractHash, Is.Not.EqualTo(a.ContractHash));
            Assert.That(b.EncodingHash, Is.EqualTo(a.EncodingHash));
        }

        [Test]
        public void GeometryChange_ChangesEncodingIdentityAndDimensions()
        {
            var first = TrainingScenario.CreateStandard("adaptive-v1");
            var changed = TrainingScenario.CreateStandard("adaptive-v1");
            changed.Board.Width = 24;
            changed.Board.Height = 16;

            MlContract a = MlContract.CreateAdaptive(first.BuildAdaptive());
            MlContract b = MlContract.CreateAdaptive(changed.BuildAdaptive());

            Assert.That(b.EncodingHash, Is.Not.EqualTo(a.EncodingHash));
            Assert.That(b.ObservationSize, Is.Not.EqualTo(a.ObservationSize));
            Assert.That(b.ActionSize, Is.Not.EqualTo(a.ActionSize));
        }

        [Test]
        public void TacticalContract_RejectsUnsafeCellCountBeforeLayoutAllocation()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 257;
            scenario.Board.Height = 256;

            Assert.That(scenario.Validate(), Is.Empty);
            var error = Assert.Throws<ArgumentOutOfRangeException>(
                () => MlContract.Create(scenario.BuildTactical()));

            Assert.That(error!.Message, Does.Contain("cell count"));
        }

        [Test]
        public void TacticalContract_RejectsCellCountArithmeticOverflow()
        {
            var config = new EnvConfig
            {
                BoardGen = new BoardGenConfig(
                    width: int.MaxValue, height: 2),
                Roster = Array.Empty<UnitStats>(),
            };

            var error = Assert.Throws<ArgumentOutOfRangeException>(
                () => MlContract.Create(config));

            Assert.That(
                error!.Message,
                Does.StartWith(
                    "tactical cell count exceeds Int32 capacity"));
        }

        [Test]
        public void TacticalContract_RejectsActionSizeArithmeticOverflow()
        {
            var config = new EnvConfig
            {
                BoardGen = new BoardGenConfig(
                    width: 65_536, height: 1),
                Roster = new UnitStats[10_923],
            };

            var error = Assert.Throws<ArgumentOutOfRangeException>(
                () => MlContract.Create(config));

            Assert.That(
                error!.Message,
                Does.StartWith(
                    "tactical action size exceeds Int32 capacity"));
        }

        [Test]
        public void TacticalContract_RejectsObservationSizeArithmeticOverflow()
        {
            var config = new EnvConfig
            {
                BoardGen = new BoardGenConfig(
                    width: int.MaxValue, height: 1),
                Roster = Array.Empty<UnitStats>(),
            };

            var error = Assert.Throws<ArgumentOutOfRangeException>(
                () => MlContract.Create(config));

            Assert.That(
                error!.Message,
                Does.StartWith(
                    "tactical observation size exceeds Int32 capacity"));
        }

        [Test]
        public void TacticalV2Scenario_DefaultMaxStepsIsDerivedFromRoundCapAndStartingUnitCount()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");

            Assert.That(scenario.Episode.MaxSteps, Is.EqualTo(
                TacticalV2Config.DefaultMaxSteps(scenario.TacticalV2.StartingUnitCount, scenario.Rules.RoundCap)));
            Assert.That(scenario.Rules.RoundCap, Is.EqualTo(GameConfig.DefaultRoundCap));
        }

        [Test]
        public void Warnings_IsEmptyForTacticalV1AndAdaptive()
        {
            Assert.That(TrainingScenario.CreateStandard("tactical-v1").Warnings(), Is.Empty);
            Assert.That(TrainingScenario.CreateStandard("adaptive-v1").Warnings(), Is.Empty);
        }

        [Test]
        public void TacticalV2Scenario_WarnsButDoesNotFailValidationWhenMaxStepsCannotReachRoundCap()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.StartingUnitCount = 12;
            scenario.TacticalV2.MaxControllableUnits = 12;
            scenario.Episode.MaxSteps = 600; // the old, pre-fix magic constant — too small for 12 units

            Assert.That(scenario.Validate(), Is.Empty,
                "an old-style scenario.json with an undersized max_steps must still validate " +
                "(warning-only) so existing runs keep loading for resume/Arena");
            Assert.That(scenario.Warnings(), Has.Some.Contains("insufficient to reach the round cap"));
            Assert.DoesNotThrow(() => scenario.BuildTacticalV2());
        }

        [Test]
        public void TacticalV2Scenario_DoesNotWarnWhenMaxStepsIsSufficient()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");

            Assert.That(scenario.Warnings(), Is.Empty);
        }

        [Test]
        public void TacticalV2Scenario_BuildsSavedCatalogAndCount()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.StartingUnitCount = 6;
            scenario.TacticalV2.MaxControllableUnits = 6;
            scenario.TacticalV2.Templates.Add(CustomTemplate("custom-alpha", "Alpha"));

            TacticalV2Config config = scenario.BuildTacticalV2();

            Assert.That(config.StartingUnitCount, Is.EqualTo(6));
            Assert.That(config.Templates.Select(item => item.Id), Does.Contain("custom-alpha"));
        }

        [Test]
        public void TacticalV2Scenario_DefaultMatchesEngineDefaultCatalogIdentity()
        {
            TacticalV2Config fromScenario = TrainingScenario.CreateStandard("tactical-v2").BuildTacticalV2();
            TacticalV2Config engineDefault = TacticalV2Config.Default();

            MlContract fromScenarioContract = MlContract.CreateTacticalV2(fromScenario);
            MlContract engineDefaultContract = MlContract.CreateTacticalV2(engineDefault);

            Assert.That(fromScenarioContract.ContractHash, Is.EqualTo(engineDefaultContract.ContractHash));
            Assert.That(fromScenarioContract.EncodingHash, Is.EqualTo(engineDefaultContract.EncodingHash));
        }

        [Test]
        public void TacticalV2Scenario_ProfiledSeededFieldsBuildIntoExactEngineConfig()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.PlacementPolicy = "profiled-seeded-v1";
            scenario.TacticalV2.StartProfiles = TacticalV2StartCatalog.ProfiledSeededV1().ToList();
            scenario.TacticalV2.StartDistribution = TacticalV2StartCatalog.ProfiledSeededV1()
                .Select(profile => new TacticalV2StartWeight(
                    profile.Id, profile.Id == "conversion-2v1-medium" ? 10000 : 0))
                .ToList();

            TacticalV2Config config = scenario.BuildTacticalV2();

            Assert.That(config.Validate(), Is.Empty);
            Assert.That(config.StartProfiles.Select(profile => profile.Id), Is.EqualTo(
                TacticalV2StartCatalog.ProfiledSeededV1().Select(profile => profile.Id)));
            Assert.That(config.StartDistribution.Select(12345), Is.EqualTo("conversion-2v1-medium"));
        }

        [Test]
        public void TacticalV2Scenario_RejectsProfiledSeededCatalogThatIsNotVersionedExact()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.PlacementPolicy = "profiled-seeded-v1";
            scenario.TacticalV2.StartProfiles = new List<TacticalV2StartProfile>
            {
                new TacticalV2StartProfile("arbitrary", 3, 3, "legacy-mirrored"),
            };
            scenario.TacticalV2.StartDistribution = new List<TacticalV2StartWeight>
            {
                new TacticalV2StartWeight("arbitrary", 10000),
            };

            Assert.That(scenario.Validate(), Has.Some.Contains("exact versioned start profile catalog"));
        }
        [Test]
        public void TacticalV2Scenario_RejectsMismatchedCountAndCap()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.StartingUnitCount = 4;
            scenario.TacticalV2.MaxControllableUnits = 6;

            Assert.That(scenario.Validate(),
                Has.Some.Contains("max controllable units must equal starting unit count"));
        }

        [TestCase(0)]
        [TestCase(13)]
        public void TacticalV2Scenario_RejectsStartingCountsOutsideRegularGameLimit(int count)
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.StartingUnitCount = count;
            scenario.TacticalV2.MaxControllableUnits = count;

            Assert.That(scenario.Validate(), Has.Some.Contains("starting unit count must be between 1 and 12"));
        }

        [Test]
        public void TacticalV2Scenario_RejectsDuplicateTemplateIds()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v2");
            string existingId = scenario.TacticalV2.Templates[0].Id;
            scenario.TacticalV2.Templates.Add(CustomTemplate(existingId, "Duplicate"));

            Assert.That(scenario.Validate(), Has.Some.Contains($"duplicate tactical-v2 template id '{existingId}'"));
        }

        [Test]
        public void TacticalV2Scenario_RejectsInvalidStats()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.Templates[0].Health = -1;

            Assert.That(scenario.Validate(), Has.Some.Contains("invalid stat"));
        }

        [Test]
        public void TacticalV2Scenario_RejectsUnrecognizedPlacementPolicy()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.PlacementPolicy = "corner-stack-v1";

            Assert.That(scenario.Validate(), Has.Some.Contains("placement policy"));
        }

        [Test]
        public void TacticalV2Scenario_RejectsInsufficientDeploymentCells()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.Board.Width = 2;
            scenario.Board.Height = 2;
            scenario.Board.ZoneDepth = 1;

            Assert.That(scenario.Validate(), Has.Some.Contains("deployment cells"));
        }

        [Test]
        public void TacticalV2Scenario_RejectsEnvironmentSectionMismatch()
        {
            var tacticalV2 = TrainingScenario.CreateStandard("tactical-v2");
            tacticalV2.Adaptive = new TrainingAdaptiveConfig();
            var tactical = TrainingScenario.CreateStandard("tactical-v1");
            tactical.TacticalV2 = new TrainingTacticalV2Config();
            var adaptive = TrainingScenario.CreateStandard("adaptive-v1");
            adaptive.TacticalV2 = new TrainingTacticalV2Config();

            Assert.That(tacticalV2.Validate(), Has.Some.Contains("adaptive section"));
            Assert.That(tactical.Validate(), Has.Some.Contains("tactical-v2 section"));
            Assert.That(adaptive.Validate(), Has.Some.Contains("tactical-v2 section"));
        }

        [Test]
        public void TacticalV2Builder_ThrowsAllValidationErrors()
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.StartingUnitCount = 0;
            scenario.TacticalV2.MaxControllableUnits = 0;
            scenario.Rules.RoundCap = 0;

            var error = Assert.Throws<ArgumentException>(() => scenario.BuildTacticalV2());

            Assert.That(error!.Message, Does.Contain("starting unit count").And.Contain("round cap"));
        }

        private static TrainingUnitTemplateConfig CustomTemplate(string id, string name) => new TrainingUnitTemplateConfig
        {
            Id = id,
            Name = name,
            Health = 4,
            Damage = 2,
            Defense = 1,
            Movement = 3,
            VerticalMovement = 2,
            Range = 1,
            RangeArc = 1,
            Vision = 2,
            VisionArc = 1,
        };
    }
}

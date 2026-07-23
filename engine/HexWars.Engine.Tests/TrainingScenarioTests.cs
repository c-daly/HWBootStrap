using System;
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
        [TestCase(6, 9, 3, "deployment zones overlap")]
        public void Scenario_RejectsImpossibleGeometry(int width, int height, int zoneDepth, string message)
        {
            var scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = width;
            scenario.Board.Height = height;
            scenario.Board.ZoneDepth = zoneDepth;

            Assert.That(scenario.Validate(), Has.Some.Contains(message));
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
    }
}

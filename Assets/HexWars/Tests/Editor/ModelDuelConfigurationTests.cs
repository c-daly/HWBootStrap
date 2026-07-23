using System;
using System.IO;
using HexWars.Engine;
using HexWars.Engine.Rl;
using HexWars.Presentation;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public sealed class ModelDuelConfigurationTests
    {
        [TestCase(ModelControllerKind.Greedy, "", "greedy")]
        [TestCase(ModelControllerKind.Random, "", "random")]
        [TestCase(ModelControllerKind.FixedRun, "C:/runs/a", "run:C:/runs/a")]
        public void SeatSpec_BuildsExplicitControllerIdentity(
            ModelControllerKind kind, string path, string expected)
        {
            var seat = new ModelSeatConfiguration { Kind = kind, Path = path };

            Assert.That(seat.BuildSpec(), Is.EqualTo(expected));
        }

        [Test]
        public void LiveRunSpec_IsStructuredAndExplicitlyLive()
        {
            var seat = new ModelSeatConfiguration { Kind = ModelControllerKind.LiveRun, Path = "C:/runs/one" };

            string spec = seat.BuildSpec();

            Assert.That(spec, Does.Contain("\"kind\":\"run\""));
            Assert.That(spec, Does.Contain("\"mode\":\"live\""));
            Assert.That(spec, Does.Contain("C:/runs/one"));
        }

        [Test]
        public void LiveRunArenaDefaultsToDeterministicInference()
        {
            var seat = new ModelSeatConfiguration
            {
                Kind = ModelControllerKind.LiveRun,
                Path = "C:/runs/arena",
            };

            Assert.That(seat.BuildSpec(), Does.Contain("\"inference_mode\":\"deterministic\""));
        }

        [Test]
        public void LiveTrainingViewerExplicitlyRequestsStochasticInference()
        {
            string spec = HexWars.Presentation.EditorTools.ReplayViewerMenu
                .BuildLiveTrainingSpec("C:/runs/training");

            Assert.That(spec, Does.Contain("\"mode\":\"live\""));
            Assert.That(spec, Does.Contain("\"inference_mode\":\"stochastic\""));
        }

        [Test]
        public void Reload_IsAllowedOnlyAtGameBoundaryForLiveSeats()
        {
            var config = new ModelDuelConfiguration
            {
                P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.LiveRun, Path = "run-a" },
                P1 = new ModelSeatConfiguration { Kind = ModelControllerKind.Greedy },
            };

            Assert.That(config.ShouldReload(gameEnded: false), Is.False);
            Assert.That(config.ShouldReload(gameEnded: true), Is.True);
        }

        [Test]
        public void ControllerChoices_ExcludeManifestlessCheckpointPaths()
        {
            Assert.That(Enum.GetNames(typeof(ModelControllerKind)), Does.Not.Contain("FixedCheckpoint"));
            Assert.That(Enum.GetNames(typeof(ModelControllerKind)), Does.Not.Contain("Snapshot"),
                "metadata-backed snapshots are internal viewer specs, not player choices");
        }

        [Test]
        public void Validate_RejectsMissingModelPathsAndInvalidPacing()
        {
            var config = new ModelDuelConfiguration
            {
                P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.FixedRun },
                SecondsPerAction = 0,
            };

            Assert.That(config.Validate(), Has.Some.Contains("Seat 0"));
            Assert.That(config.Validate(), Has.Some.Contains("pacing"));
        }

        [Test]
        public void Defaults_SelectTacticalEnvironment()
        {
            var config = new ModelDuelConfiguration();

            Assert.That(config.Environment, Is.EqualTo(MlEnvironmentContract.TacticalV1));
            Assert.That(config.Observer, Is.EqualTo(ModelDuelObserverSeat.Player1));
            Assert.That(ModelDuelObserver.Resolve(config.Observer), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void FixedObserver_DoesNotFollowTheEnvironmentCurrentSeat()
        {
            var observer = ModelDuelObserverSeat.Player2;

            Assert.That(ModelDuelObserver.Resolve(observer), Is.EqualTo(PlayerId.Player1));
            foreach (int currentSeat in new[] { 0, 1 })
                Assert.That(ModelDuelObserver.Resolve(observer), Is.EqualTo(PlayerId.Player1),
                    "observer changed with current seat " + currentSeat);
        }

        [Test]
        public void ModelContracts_MustMatchSelectedEnvironmentWhileScriptedSeatsInheritIt()
        {
            var expected = ModelDuelEnvironmentFactory.ContractIdentity(MlEnvironmentContract.AdaptiveV1);
            var adaptive = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0,\"environment\":\"adaptive-v1\",\"contract_version\":\"adaptive-v1\",\"encoding_hash\":\"" + expected.EncodingHash + "\"}]}"
            ).Seats[0];
            var tactical = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0,\"environment\":\"tactical-v1\",\"contract_version\":\"tactical-v1\",\"encoding_hash\":\"" + expected.EncodingHash + "\"}]}"
            ).Seats[0];

            Assert.That(ModelDuelContractCompatibility.Validate(
                expected, true, adaptive, false, null), Is.Empty);
            Assert.That(ModelDuelContractCompatibility.Validate(
                expected, true, tactical, false, null),
                Has.Some.Contains("Seat 0").And.Contains("adaptive-v1"));
            Assert.That(ModelDuelContractCompatibility.Validate(
                expected, false, null, false, null), Is.Empty);
        }

        [Test]
        public void ModelContracts_RejectMissingVersionMetadata()
        {
            var missing = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":1}]}"
            ).Seats[0];

            Assert.That(ModelDuelContractCompatibility.Validate(
                ModelDuelEnvironmentFactory.ContractIdentity(MlEnvironmentContract.AdaptiveV1),
                false, null, true, missing),
                Has.Some.Contains("Seat 1").And.Contains("contract_version"));
        }

        [Test]
        public void ModelContracts_RejectMissingOrMismatchedEncodingHash()
        {
            var expected = ModelDuelEnvironmentFactory.ContractIdentity(MlEnvironmentContract.AdaptiveV1);
            var missing = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0,\"environment\":\"adaptive-v1\",\"contract_version\":\"adaptive-v1\"}]}"
            ).Seats[0];
            var mismatched = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0,\"environment\":\"adaptive-v1\",\"contract_version\":\"adaptive-v1\",\"encoding_hash\":\"" + new string('f', 64) + "\"}]}"
            ).Seats[0];

            Assert.That(ModelDuelContractCompatibility.Validate(expected, true, missing, false, null),
                Has.Some.Contains("encoding_hash"));
            Assert.That(ModelDuelContractCompatibility.Validate(expected, true, mismatched, false, null),
                Has.Some.Contains("encoding hash"));
        }

        [Test]
        public void EnvironmentFactory_DerivesExpectedEncodingIdentityFromEngineContract()
        {
            var tactical = ModelDuelEnvironmentFactory.ContractIdentity(MlEnvironmentContract.TacticalV1);
            var adaptive = ModelDuelEnvironmentFactory.ContractIdentity(MlEnvironmentContract.AdaptiveV1);

            Assert.That(tactical.Environment, Is.EqualTo("tactical-v1"));
            Assert.That(tactical.Version, Is.EqualTo("tactical-v1"));
            Assert.That(tactical.EncodingHash, Does.Match("^[0-9a-f]{64}$"));
            Assert.That(adaptive.Environment, Is.EqualTo("adaptive-v1"));
            Assert.That(adaptive.Version, Is.EqualTo("adaptive-v1"));
            Assert.That(adaptive.EncodingHash, Does.Match("^[0-9a-f]{64}$"));
        }

        [Test]
        public void PresentationState_RendersTacticalThroughoutAndAdaptiveOnlyAfterReveal()
        {
            var tactical = new ModelDuelPresentationState(MlEnvironmentContract.TacticalV1);
            var adaptive = new ModelDuelPresentationState(MlEnvironmentContract.AdaptiveV1);

            Assert.That(tactical.ShouldRender(deploymentComplete: false), Is.True);
            Assert.That(tactical.ShouldRender(deploymentComplete: true), Is.True);
            Assert.That(adaptive.ShouldRender(deploymentComplete: false), Is.False);
            Assert.That(adaptive.ShouldRender(deploymentComplete: true), Is.True);
        }

        [Test]
        public void AdaptivePresentation_InitializesExactlyOnceOnAtomicReveal()
        {
            var state = new ModelDuelPresentationState(MlEnvironmentContract.AdaptiveV1);

            Assert.That(state.Next(deploymentComplete: false), Is.EqualTo(ModelDuelRenderDirective.Suppress));
            Assert.That(state.Next(deploymentComplete: false), Is.EqualTo(ModelDuelRenderDirective.Suppress));
            Assert.That(state.Next(deploymentComplete: true), Is.EqualTo(ModelDuelRenderDirective.Initialize));
            Assert.That(state.Next(deploymentComplete: true), Is.EqualTo(ModelDuelRenderDirective.Update));
        }

        [Test]
        public void EnvironmentFactory_RoutesSelectedContractToMatchingEngineEnvironment()
        {
            Assert.That(ModelDuelEnvironmentFactory.Create(MlEnvironmentContract.TacticalV1).Environment,
                Is.EqualTo(MlEnvironmentContract.TacticalV1));
            Assert.That(ModelDuelEnvironmentFactory.Create(MlEnvironmentContract.AdaptiveV1).Environment,
                Is.EqualTo(MlEnvironmentContract.AdaptiveV1));
        }

        [Test]
        public void ScenarioFactory_UsesRecordedBoardAndEncoding()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 24;
            scenario.Board.Height = 16;

            IModelDuelEnvironment duel = ModelDuelEnvironmentFactory.Create(scenario);

            Assert.That(duel.Contract.Board["width"], Is.EqualTo(24));
            Assert.That(duel.Contract.Board["height"], Is.EqualTo(16));
            Assert.That(duel.Contract.EncodingHash,
                Is.EqualTo(ModelDuelEnvironmentFactory.ContractIdentity(scenario).EncodingHash));
        }

        [Test]
        public void ScenarioFactory_UsesAdaptiveBudgetAndEpisodeHorizon()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("adaptive-v1");
            scenario.Adaptive.StartingArmyBudget = 176;
            scenario.Episode.MaxSteps = 321;

            IModelDuelEnvironment duel = ModelDuelEnvironmentFactory.Create(scenario);

            Assert.That(duel.Contract.Semantics["starting_army_budget"], Is.EqualTo(176));
            Assert.That(duel.Contract.Board["max_steps"], Is.EqualTo(642),
                "duel contracts use the recorded per-seat horizon for both seats");
        }

        [Test]
        public void ScenarioFactory_RejectsInvalidScenario()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Episode.MaxSteps = 0;

            Assert.That(() => ModelDuelEnvironmentFactory.Create(scenario),
                Throws.ArgumentException.With.Message.Contains("max steps"));
        }

        [Test]
        public void ManualArena_LoadsSelectedRunScenarioWithoutChangingSeatsOrObserver()
        {
            string scratch = NewScratchDirectory();
            try
            {
                string scenarioRun = WriteScenarioRun(
                    scratch, "scenario-source", MlEnvironmentContract.TacticalV1,
                    scenario => scenario.Board.Width = 24);
                var config = new ModelDuelConfiguration
                {
                    Environment = MlEnvironmentContract.TacticalV1,
                    ScenarioRunPath = scenarioRun,
                    P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.Random },
                    P1 = new ModelSeatConfiguration { Kind = ModelControllerKind.Greedy },
                    Observer = ModelDuelObserverSeat.Player2,
                };

                MlArenaLaunchPlan plan = MlArenaLaunchPlan.Create(config);

                Assert.That(plan.Scenario.Board.Width, Is.EqualTo(24));
                Assert.That(plan.P0Spec, Is.EqualTo("random"));
                Assert.That(plan.P1Spec, Is.EqualTo("greedy"));
                Assert.That(plan.Observer, Is.EqualTo(ModelDuelObserverSeat.Player2));
            }
            finally
            {
                Directory.Delete(scratch, recursive: true);
            }
        }

        [Test]
        public void ManualArena_RejectsBothModelSeatsAgainstSelectedScenarioEncoding()
        {
            string scratch = NewScratchDirectory();
            try
            {
                string scenarioRun = WriteScenarioRun(
                    scratch, "large-board", MlEnvironmentContract.TacticalV1,
                    scenario =>
                    {
                        scenario.Board.Width = 24;
                        scenario.Board.Height = 16;
                    });
                ModelDuelContractIdentity standard =
                    ModelDuelEnvironmentFactory.ContractIdentity(MlEnvironmentContract.TacticalV1);
                string p0 = WriteModelRun(scratch, "p0", standard);
                string p1 = WriteModelRun(scratch, "p1", standard);
                var config = new ModelDuelConfiguration
                {
                    Environment = MlEnvironmentContract.TacticalV1,
                    ScenarioRunPath = scenarioRun,
                    P0 = new ModelSeatConfiguration
                        { Kind = ModelControllerKind.FixedRun, Path = p0 },
                    P1 = new ModelSeatConfiguration
                        { Kind = ModelControllerKind.LiveRun, Path = p1 },
                };
                string selectedHash = ModelDuelEnvironmentFactory.ContractIdentity(
                    MlArenaLaunchPlan.LoadScenario(config)).EncodingHash;

                InvalidOperationException error = Assert.Throws<InvalidOperationException>(
                    () => MlArenaLaunchPlan.Create(config));
                Assert.That(error.Message, Does.Contain("Seat 0"));
                Assert.That(error.Message, Does.Contain("Seat 1"));
                Assert.That(error.Message, Does.Contain(standard.EncodingHash));
                Assert.That(error.Message, Does.Contain(selectedHash));
            }
            finally
            {
                Directory.Delete(scratch, recursive: true);
            }
        }

        [Test]
        public void AdaptiveAdapter_RequestsExternalActionsWithoutExposingStateBeforeReveal()
        {
            var environment = ModelDuelEnvironmentFactory.Create(MlEnvironmentContract.AdaptiveV1);

            ModelDuelView view = environment.Reset(seed: 41, controller0: null, controller1: null);

            Assert.That(view.DeploymentComplete, Is.False);
            Assert.That(environment.CurrentState, Is.Null);
            int action = Array.FindIndex(view.ActionMask, legal => legal);
            Assert.That(action, Is.GreaterThanOrEqualTo(0));

            view = environment.Step(action);

            Assert.That(view.DeploymentComplete, Is.False);
            Assert.That(environment.CurrentState, Is.Null);
            Assert.That(view.ActionMask, Has.Some.True);
        }

        [Test]
        public void AdaptiveAdapter_ExposesRevealBeforeContinuingScriptedGameplay()
        {
            var first = new CountingEndTurnAgent();
            var second = new CountingEndTurnAgent();
            var environment = ModelDuelEnvironmentFactory.Create(MlEnvironmentContract.AdaptiveV1);

            ModelDuelView reveal = environment.Reset(seed: 43, controller0: first, controller1: second);

            Assert.That(reveal.DeploymentComplete, Is.True);
            Assert.That(reveal.Seat, Is.Zero);
            Assert.That(environment.RequiresContinuation, Is.True);
            Assert.That(first.Calls, Is.Zero);

            environment.Continue();

            Assert.That(environment.RequiresContinuation, Is.False);
            Assert.That(first.Calls, Is.GreaterThan(0));
        }

        [Test]
        public void TacticalAdapter_PreservesImmediateScriptedSeatAdvance()
        {
            var first = new CountingEndTurnAgent();
            var environment = ModelDuelEnvironmentFactory.Create(MlEnvironmentContract.TacticalV1);

            ModelDuelView view = environment.Reset(seed: 44, controller0: first, controller1: null);

            Assert.That(first.Calls, Is.EqualTo(1));
            Assert.That(view.Seat, Is.EqualTo(1));
            Assert.That(view.DeploymentComplete, Is.True);
            Assert.That(environment.RequiresContinuation, Is.False);
        }

        sealed class CountingEndTurnAgent : IAgent
        {
            public int Calls { get; private set; }

            public Command Decide(GameState state)
            {
                Calls++;
                return new EndTurn(state.ActivePlayer);
            }
        }

        static string NewScratchDirectory()
        {
            string path = Path.Combine(
                Path.GetTempPath(), "hexwars-arena-tests-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return path;
        }

        static string WriteScenarioRun(
            string scratch,
            string name,
            MlEnvironmentContract environment,
            Action<MlTrainingScenario> edit)
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string library = Path.Combine(
                projectRoot, "python", "config", "training-game-templates.json");
            MlTrainingScenario scenario = MlTrainingScenarioLibrary.Load(library)
                .Filter(environment)[0]
                .Clone();
            edit?.Invoke(scenario);
            string generated = MlTrainingScenarioStore.WriteSessionScenario(scratch, scenario);
            string run = Path.Combine(scratch, name);
            Directory.CreateDirectory(run);
            File.Copy(generated, Path.Combine(run, "scenario.json"));
            return run;
        }

        static string WriteModelRun(
            string scratch, string name, ModelDuelContractIdentity identity)
        {
            string run = Path.Combine(scratch, name);
            Directory.CreateDirectory(run);
            File.WriteAllText(Path.Combine(run, "run.json"),
                "{\"contract\":{\"environment\":\"" + identity.Environment +
                "\",\"version\":\"" + identity.Version +
                "\",\"encoding_hash\":\"" + identity.EncodingHash + "\"}}");
            return run;
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using HexWars.Engine;
using HexWars.Engine.Rl;
using HexWars.Presentation;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

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
        public void PresentationSchedule_CyclesGamesAndSurvivesUnitySerialization()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 24;
            var schedule = new MlPresentationSchedule
            {
                Games = new[]
                {
                    new MlPresentationGame(
                        "learner", "random", 0, ModelDuelObserverSeat.Player1,
                        "Random", scenario),
                    new MlPresentationGame(
                        "greedy", "learner", 1, ModelDuelObserverSeat.Player2,
                        "Greedy", scenario),
                },
            };
            var originalObject = new GameObject("original-driver");
            var restoredObject = new GameObject("restored-driver");
            try
            {
                var original = originalObject.AddComponent<ModelDuelDriver>();
                original.PresentationPlan = schedule;

                string json = EditorJsonUtility.ToJson(original);
                var restored = restoredObject.AddComponent<ModelDuelDriver>();
                EditorJsonUtility.FromJsonOverwrite(json, restored);

                Assert.That(restored.NextPresentationGame(0).LearnerSeat, Is.Zero);
                Assert.That(restored.NextPresentationGame(1).LearnerSeat, Is.EqualTo(1));
                Assert.That(restored.NextPresentationGame(2).LearnerSeat, Is.Zero);
                Assert.That(restored.NextPresentationGame(1).OpponentLabel, Is.EqualTo("Greedy"));
                Assert.That(restored.NextPresentationGame(1).Observer,
                    Is.EqualTo(ModelDuelObserverSeat.Player2));
                Assert.That(restored.NextPresentationGame(1).Scenario.Board.Width,
                    Is.EqualTo(24));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(originalObject);
                UnityEngine.Object.DestroyImmediate(restoredObject);
            }
        }

        [Test]
        public void ShouldReconfigure_OnlyChangesControllersAtACompletedGameBoundary()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v1");
            var game0 = new MlPresentationGame(
                "learner", "greedy", 0, ModelDuelObserverSeat.Player1,
                "Greedy", scenario);
            var game1 = new MlPresentationGame(
                "greedy", "learner", 1, ModelDuelObserverSeat.Player2,
                "Greedy", scenario);

            Assert.That(ModelDuelDriver.ShouldReconfigure(game0, game1, gameEnded: false),
                Is.False);
            Assert.That(ModelDuelDriver.ShouldReconfigure(game0, game1, gameEnded: true),
                Is.True);
            Assert.That(ModelDuelDriver.ShouldReconfigure(game0, game0, gameEnded: true),
                Is.False);
        }

        [Test]
        public void ApplyPresentationGame_DerivesObserverFromLearnerSeat()
        {
            var go = new GameObject("driver");
            try
            {
                var driver = go.AddComponent<ModelDuelDriver>();
                var game = new MlPresentationGame(
                    "greedy", "random", learnerSeat: 1,
                    observer: ModelDuelObserverSeat.Player1,
                    opponentLabel: "Greedy",
                    scenario: TrainingScenario.CreateStandard("tactical-v1"));

                InvokePrivate(driver, "ApplyPresentationGame", game);

                Assert.That(driver.Observer, Is.EqualTo(ModelDuelObserverSeat.Player2));
                Assert.That(driver.ObserverPlayer, Is.EqualTo(PlayerId.Player1));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void ApplyPresentationGame_MapsTacticalV2ScenarioEnvironment()
        {
            var go = new GameObject("driver-tactical-v2");
            try
            {
                var driver = go.AddComponent<ModelDuelDriver>();
                var game = new MlPresentationGame(
                    "greedy", "random", learnerSeat: 0,
                    observer: ModelDuelObserverSeat.Player1,
                    opponentLabel: "Greedy",
                    scenario: TrainingScenario.CreateStandard("tactical-v2"));

                InvokePrivate(driver, "ApplyPresentationGame", game);

                Assert.That(driver.Environment, Is.EqualTo(MlEnvironmentContract.TacticalV2));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void EmptyPresentationScheduleAtBoundaryFailsClosedAndClearsStarting()
        {
            var go = new GameObject("driver");
            try
            {
                var driver = go.AddComponent<ModelDuelDriver>();
                driver.P0Spec = "run:C:/runs/learner";
                driver.PresentationPlan = new MlPresentationSchedule();
                SetPrivate(driver, "_p0Model", true);

                LogAssert.Expect(
                    LogType.Error,
                    "ModelDuelDriver: game-boundary restart failed. " +
                    "presentation schedule must contain at least one game");
                InvokePrivate(driver, "AdvanceAtGameBoundary");

                Assert.That(driver.IsDone, Is.True);
                Assert.That(driver.IsStarting, Is.False);
                Assert.That(driver.P0ArenaStatus, Is.EqualTo("restart failed"));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
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
        public void Defaults_SelectTacticalV2Environment()
        {
            var config = new ModelDuelConfiguration();

            Assert.That(config.Environment, Is.EqualTo(MlEnvironmentContract.TacticalV2));
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
            var tacticalV2 = new ModelDuelPresentationState(MlEnvironmentContract.TacticalV2);
            var adaptive = new ModelDuelPresentationState(MlEnvironmentContract.AdaptiveV1);

            Assert.That(tactical.ShouldRender(deploymentComplete: false), Is.True);
            Assert.That(tactical.ShouldRender(deploymentComplete: true), Is.True);
            Assert.That(tacticalV2.ShouldRender(deploymentComplete: false), Is.True,
                "tactical-v2 has no hidden deployment phase and must render immediately, like tactical-v1");
            Assert.That(tacticalV2.ShouldRender(deploymentComplete: true), Is.True);
            Assert.That(adaptive.ShouldRender(deploymentComplete: false), Is.False);
            Assert.That(adaptive.ShouldRender(deploymentComplete: true), Is.True);
        }

        [Test]
        public void Factory_CreatesTacticalV2DuelWithMatchingIdentity()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
            IModelDuelEnvironment duel = ModelDuelEnvironmentFactory.Create(scenario);

            Assert.That(duel.Environment, Is.EqualTo(MlEnvironmentContract.TacticalV2));
            Assert.That(duel.Contract.Version, Is.EqualTo("tactical-v2"));
            Assert.That(ModelDuelEnvironmentFactory.ContractIdentity(scenario).EncodingHash,
                Is.EqualTo(duel.Contract.EncodingHash));
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
        public void DriverScenario_SurvivesUnitySerializationAndPreservesEncoding()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 24;
            scenario.Board.Height = 16;
            string expectedEncoding =
                ModelDuelEnvironmentFactory.ContractIdentity(scenario).EncodingHash;
            var originalObject = new GameObject("original-driver");
            var restoredObject = new GameObject("restored-driver");
            try
            {
                var original = originalObject.AddComponent<ModelDuelDriver>();
                original.Environment = MlEnvironmentContract.TacticalV1;
                original.Scenario = scenario;

                string serializedComponent = EditorJsonUtility.ToJson(original);
                var restored = restoredObject.AddComponent<ModelDuelDriver>();
                EditorJsonUtility.FromJsonOverwrite(serializedComponent, restored);
                TrainingScenario roundTripped = restored.ResolveScenario();

                Assert.That(roundTripped.Board.Width, Is.EqualTo(24));
                Assert.That(roundTripped.Board.Height, Is.EqualTo(16));
                Assert.That(
                    ModelDuelEnvironmentFactory.ContractIdentity(roundTripped).EncodingHash,
                    Is.EqualTo(expectedEncoding));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(originalObject);
                UnityEngine.Object.DestroyImmediate(restoredObject);
            }
        }

        [Test]
        public void DriverScenario_TacticalV2SurvivesUnitySerializationAndPreservesEncoding()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.StartingUnitCount = 5;
            scenario.TacticalV2.MaxControllableUnits = 5;
            string expectedEncoding =
                ModelDuelEnvironmentFactory.ContractIdentity(scenario).EncodingHash;
            var originalObject = new GameObject("original-driver-v2");
            var restoredObject = new GameObject("restored-driver-v2");
            try
            {
                var original = originalObject.AddComponent<ModelDuelDriver>();
                original.Environment = MlEnvironmentContract.TacticalV2;
                original.Scenario = scenario;

                string serializedComponent = EditorJsonUtility.ToJson(original);
                var restored = restoredObject.AddComponent<ModelDuelDriver>();
                EditorJsonUtility.FromJsonOverwrite(serializedComponent, restored);
                TrainingScenario roundTripped = restored.ResolveScenario();

                Assert.That(roundTripped.TacticalV2.StartingUnitCount, Is.EqualTo(5));
                Assert.That(
                    ModelDuelEnvironmentFactory.ContractIdentity(roundTripped).EncodingHash,
                    Is.EqualTo(expectedEncoding));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(originalObject);
                UnityEngine.Object.DestroyImmediate(restoredObject);
            }
        }

        [Test]
        public void DriverScenario_AdaptiveSurvivesUnitySerializationAndPreservesEncoding()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("adaptive-v1");
            scenario.Adaptive.StartingArmyBudget = 176;
            string expectedEncoding =
                ModelDuelEnvironmentFactory.ContractIdentity(scenario).EncodingHash;
            var originalObject = new GameObject("original-driver-adaptive");
            var restoredObject = new GameObject("restored-driver-adaptive");
            try
            {
                var original = originalObject.AddComponent<ModelDuelDriver>();
                original.Environment = MlEnvironmentContract.AdaptiveV1;
                original.Scenario = scenario;

                string serializedComponent = EditorJsonUtility.ToJson(original);
                var restored = restoredObject.AddComponent<ModelDuelDriver>();
                EditorJsonUtility.FromJsonOverwrite(serializedComponent, restored);
                TrainingScenario roundTripped = restored.ResolveScenario();

                Assert.That(roundTripped.Adaptive.StartingArmyBudget, Is.EqualTo(176));
                Assert.That(roundTripped.TacticalV2, Is.Null,
                    "Unity's by-value serializer materializes the absent TacticalV2 section; " +
                    "ResolveScenario must null it back out for adaptive-v1");
                Assert.That(
                    ModelDuelEnvironmentFactory.ContractIdentity(roundTripped).EncodingHash,
                    Is.EqualTo(expectedEncoding));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(originalObject);
                UnityEngine.Object.DestroyImmediate(restoredObject);
            }
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

        [Test]
        public void TacticalAdapter_CaptureTransitionsDefaultsToFalseAndDrainsScriptedCommandsWhenEnabled()
        {
            var first = new CountingEndTurnAgent();
            var environment = ModelDuelEnvironmentFactory.Create(MlEnvironmentContract.TacticalV1);
            Assert.That(environment.CaptureTransitions, Is.False,
                "opt-in capture defaults to false: headless training must never pay to retain states");

            environment.Reset(seed: 45, controller0: first, controller1: null);
            Assert.That(environment.DrainTransitions(), Is.Empty,
                "capture is still off: the scripted EndTurn above must not have been retained");

            environment.CaptureTransitions = true;
            environment.Reset(seed: 45, controller0: first, controller1: null);

            IReadOnlyList<DuelTransition> transitions = environment.DrainTransitions();
            Assert.That(transitions, Has.Count.EqualTo(1));
            Assert.That(transitions[0].Command, Is.InstanceOf<EndTurn>());
            Assert.That(environment.DrainTransitions(), Is.Empty, "draining clears the queue");
        }

        [Test]
        public void TacticalV2Adapter_DrainsScriptedTransitionsInOrderWhenCaptureIsEnabled()
        {
            var first = new CountingEndTurnAgent();
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
            var environment = ModelDuelEnvironmentFactory.Create(scenario);
            environment.CaptureTransitions = true;

            environment.Reset(seed: 46, controller0: first, controller1: null);

            IReadOnlyList<DuelTransition> transitions = environment.DrainTransitions();
            Assert.That(transitions, Is.Not.Empty);
            for (int i = 1; i < transitions.Count; i++)
                Assert.That(transitions[i].Previous, Is.SameAs(transitions[i - 1].Resulting),
                    "consecutive transitions must chain by reference");
        }

        [Test]
        public void AdaptiveAdapter_TransitionsStayEmptyPreRevealThenCapturePostRevealScriptedPlay()
        {
            var first = new CountingEndTurnAgent();
            var second = new CountingEndTurnAgent();
            var environment = ModelDuelEnvironmentFactory.Create(MlEnvironmentContract.AdaptiveV1);
            environment.CaptureTransitions = true;

            ModelDuelView reveal = environment.Reset(seed: 47, controller0: first, controller1: second);

            Assert.That(reveal.DeploymentComplete, Is.True);
            Assert.That(environment.DrainTransitions(), Is.Empty,
                "hidden pregame deployment placements never produce a transition");

            environment.Continue();

            Assert.That(environment.DrainTransitions(), Is.Not.Empty,
                "post-reveal scripted play must be captured once continuation resumes");
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

        static void InvokePrivate(object target, string method, params object[] arguments)
        {
            MethodInfo info = target.GetType().GetMethod(
                method, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(info, Is.Not.Null, method);
            info.Invoke(target, arguments);
        }

        static void SetPrivate(object target, string field, object value)
        {
            FieldInfo info = target.GetType().GetField(
                field, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(info, Is.Not.Null, field);
            info.SetValue(target, value);
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

        // ---- Viewer B: omniscient, paced, transition-driven playback ----

        [Test]
        public void ScriptedDuel_PresentsEveryAcceptedCommandInOrderAndAdvancesPresentedStatePerTransition()
        {
            var go = new GameObject("driver-order", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                var driver = StartScriptedDriver(go, MlEnvironmentContract.TacticalV2, "greedy", "random", seed: 5);
                var presenter = go.GetComponent<ActionPresenter>();
                var committed = new List<(GameState prev, Command cmd, GameState next)>();
                presenter.ItemCommitted += (prev, cmd, next) => committed.Add((prev, cmd, next));

                GameState anchor = driver.PresentedState;
                Assert.That(anchor, Is.Not.Null);
                Assert.That(presenter.IsBusy, Is.True,
                    "a full scripted-vs-scripted game resolves inside Reset and must already be queued " +
                    "for presentation");

                FastForwardIgnoringEditModeDestroyWarnings(presenter);

                Assert.That(committed, Is.Not.Empty);
                Assert.That(committed[0].prev, Is.SameAs(anchor),
                    "the first presented transition must start from the episode's presented anchor");
                for (int i = 1; i < committed.Count; i++)
                    Assert.That(committed[i].prev, Is.SameAs(committed[i - 1].next),
                        "presented transitions must play in the exact order the engine accepted them");
                Assert.That(driver.PresentedState, Is.SameAs(committed[committed.Count - 1].next),
                    "PresentedState must land on the Resulting state of the last fully-presented transition");
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        [Test]
        public void ScriptedDuel_AttackTransitionsReachTheAnimationQueue()
        {
            bool foundAttack = false;
            for (int seed = 0; seed < 20 && !foundAttack; seed++)
            {
                var go = new GameObject("driver-attack-" + seed, typeof(BoardRenderer), typeof(ModelDuelDriver));
                try
                {
                    var driver = StartScriptedDriver(go, MlEnvironmentContract.TacticalV2, "greedy", "greedy", seed);
                    var presenter = go.GetComponent<ActionPresenter>();
                    var committedCommands = new List<Command>();
                    presenter.ItemCommitted += (prev, cmd, next) => committedCommands.Add(cmd);

                    FastForwardIgnoringEditModeDestroyWarnings(presenter);

                    foundAttack = committedCommands.Any(cmd => cmd is AttackUnit);
                }
                finally { UnityEngine.Object.DestroyImmediate(go); }
            }

            Assert.That(foundAttack, Is.True,
                "expected at least one of the first 20 greedy-vs-greedy seeds to produce an attack " +
                "transition that reaches the animation queue");
        }

        [Test]
        public void Update_GatesTheNextAdvanceUntilPresentationCatchesUp()
        {
            var go = new GameObject("driver-pacing", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                var driver = AwakeDriverForTest(go);
                driver.P0Spec = "greedy";
                driver.P1Spec = "random";
                driver.Environment = MlEnvironmentContract.AdaptiveV1;
                driver.Loop = false;
                InvokePrivate(driver, "RefreshControllerFlags");
                SetPrivate(driver, "_activeScenario", driver.ResolveScenario());
                InvokePrivate(driver, "BeginGame");

                var duel = (IModelDuelEnvironment)GetPrivate(driver, "_duel");
                Assert.That(duel.RequiresContinuation, Is.True,
                    "the scripted first mover must await the atomic reveal before continuing");
                Assert.That(driver.PresentedState, Is.Not.Null,
                    "the reveal itself must already be presented before any post-reveal play");

                var presenter = go.GetComponent<ActionPresenter>();
                Assert.That(presenter.IsBusy, Is.False,
                    "nothing plays before the reveal: hidden deployment never produces a transition");

                InvokePrivate(driver, "Update"); // presenter idle -> allowed past the reveal
                Assert.That(duel.RequiresContinuation, Is.False);
                Assert.That(presenter.IsBusy, Is.True,
                    "post-reveal scripted play must have queued transitions for the viewer");

                int gamesBefore = driver.GamesPlayed;
                InvokePrivate(driver, "Update"); // presenter still busy -> must not progress further
                Assert.That(driver.GamesPlayed, Is.EqualTo(gamesBefore),
                    "Update must not advance game-boundary logic while presentation is still queued");

                FastForwardIgnoringEditModeDestroyWarnings(presenter);
                InvokePrivate(driver, "Update"); // presenter idle again -> game-boundary logic may proceed
                Assert.That(driver.GamesPlayed, Is.EqualTo(gamesBefore + 1));
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        [Test]
        public void FogOfWarScenario_RendersBothArmiesOmniscientlyRegardlessOfVisibility()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.Rules.FogOfWar = true;
            var go = new GameObject("driver-fog", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                var driver = AwakeDriverForTest(go);
                driver.P0Spec = "greedy";
                driver.P1Spec = "random";
                driver.Environment = MlEnvironmentContract.TacticalV2;
                driver.Scenario = scenario;
                driver.Seed = 9;
                InvokePrivate(driver, "RefreshControllerFlags");
                SetPrivate(driver, "_activeScenario", driver.ResolveScenario());
                InvokePrivate(driver, "BeginGame");

                var presenter = go.GetComponent<ActionPresenter>();
                FastForwardIgnoringEditModeDestroyWarnings(presenter);

                GameState presented = driver.PresentedState;
                Assert.That(presented.Config.FogOfWar, Is.True,
                    "the scenario under test must actually train with fog of war on");
                var tokens = go.GetComponent<TokenStore>();
                Assert.That(tokens, Is.Not.Null);

                int checkedUnits = 0;
                foreach (PlayerState player in presented.Players)
                    foreach (Unit unit in player.UnitsOnBoard)
                        if (unit.IsAlive)
                        {
                            Assert.That(tokens.UnitToken(unit.Id), Is.Not.Null,
                                $"unit {unit.Id} (player {player.Id}) must be rendered omnisciently " +
                                "even though the scenario trains with fog of war");
                            checkedUnits++;
                        }
                Assert.That(checkedUnits, Is.GreaterThan(0));
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        [Test]
        public void HandlePresentation_UnpresentableTransitionStopsWithAnExplicitStatus()
        {
            var go = new GameObject("driver-presentation-error", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                var driver = go.GetComponent<ModelDuelDriver>();
                InvokePrivate(driver, "Awake"); // EditMode tests never tick the player loop
                driver.Environment = MlEnvironmentContract.TacticalV1;
                SetPrivate(driver, "_duel", new ThrowingModelDuelEnvironment());
                SetPrivate(driver, "_presentation", new ModelDuelPresentationState(MlEnvironmentContract.TacticalV1));
                SetPrivate(driver, "_view", default(ModelDuelView));

                LogAssert.Expect(LogType.Error,
                    "ModelDuelDriver: presentation error: simulated unpresentable transition");
                InvokePrivate(driver, "HandlePresentation");

                Assert.That(driver.IsDone, Is.True);
                Assert.That(driver.P0ArenaStatus, Does.Contain("presentation error"));
                Assert.That(driver.P1ArenaStatus, Does.Contain("presentation error"));
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        static ModelDuelDriver StartScriptedDriver(
            GameObject go, MlEnvironmentContract environment, string p0Spec, string p1Spec, int seed)
        {
            var driver = AwakeDriverForTest(go);
            driver.P0Spec = p0Spec;
            driver.P1Spec = p1Spec;
            driver.Environment = environment;
            driver.Seed = seed;
            InvokePrivate(driver, "RefreshControllerFlags");
            SetPrivate(driver, "_activeScenario", driver.ResolveScenario());
            InvokePrivate(driver, "BeginGame");
            return driver;
        }

        /// <summary>EditMode batch tests never tick Unity's player loop, so components this driver
        /// wires up dynamically (<see cref="ActionPresenter"/>, and the <see cref="TokenStore"/>
        /// <see cref="BoardRenderer"/> auto-adds on first render) do not reliably run their own Awake()
        /// — the same reason <c>OnlineBarracksReconcileTests</c> pokes <c>TokenStore._board</c>
        /// directly rather than trusting Unity to call it. Drive every presentation component's Awake
        /// by hand so board/token wiring behaves exactly as it does at runtime.</summary>
        static ModelDuelDriver AwakeDriverForTest(GameObject go)
        {
            var driver = go.GetComponent<ModelDuelDriver>();
            InvokePrivate(driver, "Awake"); // resolves _board, lazily adds+wires _presenter
            var tokenStore = go.GetComponent<TokenStore>() ?? go.AddComponent<TokenStore>();
            SetPrivate(tokenStore, "_board", go.GetComponent<BoardRenderer>());
            var presenter = go.GetComponent<ActionPresenter>();
            if (presenter != null) InvokePrivate(presenter, "Awake");
            return driver;
        }

        /// <summary>Synchronously flushes the presenter's queue for the test. Its <see cref="TokenStore"/>
        /// prunes dead units through <c>UnityEngine.Object.Destroy</c> — correct at runtime, but Unity
        /// logs (and this harness fails on) an error when it's called outside Play mode. A real
        /// scripted-vs-scripted game run in an EditMode test always trips this on any unit death, so
        /// swallow just that expected noise around the flush rather than asserting on each seed's exact
        /// (non-deterministic) set of destroyed unit names.</summary>
        static void FastForwardIgnoringEditModeDestroyWarnings(ActionPresenter presenter)
        {
            LogAssert.ignoreFailingMessages = true;
            try { presenter.FastForward(); }
            finally { LogAssert.ignoreFailingMessages = false; }
        }

        static object GetPrivate(object target, string field)
        {
            FieldInfo info = target.GetType().GetField(
                field, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(info, Is.Not.Null, field);
            return info.GetValue(target);
        }

        sealed class ThrowingModelDuelEnvironment : IModelDuelEnvironment
        {
            public MlEnvironmentContract Environment => MlEnvironmentContract.TacticalV1;
            public MlContract Contract => null;
            public GameState CurrentState => null;
            public bool RequiresContinuation => false;
            public bool CaptureTransitions { get; set; }

            public ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1) => default;
            public ModelDuelView Step(int action) => default;
            public ModelDuelView Continue() => default;

            public IReadOnlyList<DuelTransition> DrainTransitions() =>
                throw new InvalidOperationException("simulated unpresentable transition");
        }
    }
}

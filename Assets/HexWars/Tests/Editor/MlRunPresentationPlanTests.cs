using System;
using System.IO;
using System.Linq;
using HexWars.Presentation;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlRunPresentationPlanTests
    {
        string _scratch;
        string _scenarioJson;

        [SetUp]
        public void SetUp()
        {
            _scratch = Path.Combine(
                Path.GetTempPath(), "hexwars-presentation-plan-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_scratch);
            MlTrainingScenario tactical = MlTrainingScenarioLibrary
                .Load(Path.Combine("python", "config", "training-game-templates.json"))
                .Filter(MlEnvironmentContract.TacticalV1)
                .First(item => item.Id == "tactical-standard");
            _scenarioJson = File.ReadAllText(
                MlTrainingScenarioStore.WriteSessionScenario(_scratch, tactical));
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_scratch)) Directory.Delete(_scratch, recursive: true);
        }

        [TestCase("0", 0, 0)]
        [TestCase("0", 1, 0)]
        [TestCase("0", 2, 0)]
        [TestCase("1", 0, 1)]
        [TestCase("1", 1, 1)]
        [TestCase("1", 2, 1)]
        [TestCase("alternating", 0, 0)]
        [TestCase("alternating", 1, 1)]
        [TestCase("alternating", 2, 0)]
        public void PlanGame_PlacesLearnerInRecordedSeat(
            string schedule, int game, int learnerSeat)
        {
            string run = WriteRun(
                schedule, "{\"kind\":\"scripted\",\"name\":\"random\"}");

            MlPresentationGame resolved = MlRunPresentationPlan.Load(run).PlanGame(game);

            Assert.That(resolved.LearnerSeat, Is.EqualTo(learnerSeat));
            Assert.That(resolved.Observer, Is.EqualTo(
                learnerSeat == 0
                    ? ModelDuelObserverSeat.Player1
                    : ModelDuelObserverSeat.Player2));
            Assert.That(
                learnerSeat == 0 ? resolved.P1Spec : resolved.P0Spec,
                Is.EqualTo("random"));
            Assert.That(
                learnerSeat == 0 ? resolved.P0Spec : resolved.P1Spec,
                Does.Contain("\"mode\":\"live\""));
        }

        [TestCase("greedy", "Greedy")]
        [TestCase("random", "Random")]
        public void ScriptedOpponent_RetainsRecordedIdentity(
            string name, string label)
        {
            string run = WriteRun(
                "0", "{\"kind\":\"scripted\",\"name\":\"" + name + "\"}");

            MlPresentationGame game = MlRunPresentationPlan.Load(run).PlanGame(0);

            Assert.That(game.P1Spec, Is.EqualTo(name));
            Assert.That(game.OpponentLabel, Is.EqualTo(label));
        }

        [Test]
        public void FixedSnapshot_RetainsExactRecordedCheckpoint()
        {
            string source = WriteSourceRun("source-fixed", "maskable_ppo", 10);
            string checkpoint = Path.Combine(
                source, "checkpoints", "step_000000010.zip");
            string opponent = "{\"kind\":\"snapshot\",\"path\":" + Json(checkpoint) +
                ",\"source_run\":" + Json(source) +
                ",\"algorithm\":\"maskable_ppo\",\"step\":10}";
            string run = WriteRun("0", opponent);

            MlPresentationGame game = MlRunPresentationPlan.Load(run).PlanGame(0);

            Assert.That(game.P1Spec, Does.Contain("\"kind\":\"snapshot\""));
            Assert.That(game.P1Spec, Does.Contain(JsonFragment(checkpoint)));
            Assert.That(game.P1Spec, Does.Contain("\"step\":10"));
            Assert.That(game.OpponentLabel, Does.Contain("step 10"));
        }

        [Test]
        public void SnapshotWithoutStepFailsInsteadOfDefaultingToZero()
        {
            string source = WriteSourceRun("source-no-step", "maskable_ppo", 0);
            string checkpoint = Path.Combine(
                source, "checkpoints", "step_000000000.zip");
            string opponent = "{\"kind\":\"snapshot\",\"path\":" + Json(checkpoint) +
                ",\"source_run\":" + Json(source) +
                ",\"algorithm\":\"maskable_ppo\"}";
            string run = WriteRun("0", opponent);

            Assert.That(
                () => MlRunPresentationPlan.Load(run),
                Throws.InvalidOperationException.With.Message.Contains(
                    "opponent_snapshot.step"));
        }

        [TestCase("\"0\"")]
        [TestCase("0.5")]
        [TestCase("true")]
        [TestCase("null")]
        public void SnapshotStepMustBeAnInteger(string stepJson)
        {
            string source = WriteSourceRun("source-step-type", "maskable_ppo", 0);
            string checkpoint = Path.Combine(
                source, "checkpoints", "step_000000000.zip");
            string opponent = "{\"kind\":\"snapshot\",\"path\":" + Json(checkpoint) +
                ",\"source_run\":" + Json(source) +
                ",\"algorithm\":\"maskable_ppo\",\"step\":" + stepJson + "}";
            string run = WriteRun("0", opponent);

            Assert.That(
                () => MlRunPresentationPlan.Load(run),
                Throws.InvalidOperationException.With.Message.Contains(
                    "opponent_snapshot.step"));
        }

        [Test]
        public void LiveRunOpponent_RetainsLiveMode()
        {
            string source = WriteSourceRun("source-live", "maskable_ppo", 10);
            string opponent = "{\"kind\":\"run\",\"path\":" + Json(source) +
                ",\"mode\":\"live\",\"inference_mode\":\"stochastic\"}";
            string run = WriteRun("0", opponent);

            MlPresentationGame game = MlRunPresentationPlan.Load(run).PlanGame(0);

            Assert.That(game.P1Spec, Does.Contain("\"kind\":\"run\""));
            Assert.That(game.P1Spec, Does.Contain("\"mode\":\"live\""));
            Assert.That(game.P1Spec, Does.Contain("\"inference_mode\":\"stochastic\""));
        }

        [Test]
        public void Pool_CyclesDeterministicallyInRecordedOrder()
        {
            string run = WriteRun(
                "0",
                "{\"kind\":\"pool\",\"controllers\":[" +
                "{\"kind\":\"scripted\",\"name\":\"greedy\"}," +
                "{\"kind\":\"scripted\",\"name\":\"random\"}]}");
            MlRunPresentationPlan plan = MlRunPresentationPlan.Load(run);

            Assert.That(
                Enumerable.Range(0, 5).Select(index => plan.PlanGame(index).P1Spec),
                Is.EqualTo(new[] { "greedy", "random", "greedy", "random", "greedy" }));
        }

        [Test]
        public void IncompatiblePoolMetadataFailsAtLoad()
        {
            string source = WriteSourceRun(
                "source-incompatible",
                "maskable_ppo",
                10,
                new string('c', 64));
            string checkpoint = Path.Combine(
                source, "checkpoints", "step_000000010.zip");
            string run = WriteRun(
                "0",
                "{\"kind\":\"pool\",\"controllers\":[" +
                "{\"kind\":\"scripted\",\"name\":\"greedy\"}," +
                "{\"kind\":\"snapshot\",\"path\":" + Json(checkpoint) +
                ",\"source_run\":" + Json(source) +
                ",\"algorithm\":\"maskable_ppo\",\"step\":10}]}");

            Assert.That(
                () => MlRunPresentationPlan.Load(run),
                Throws.InvalidOperationException.With.Message.Contains(
                    "incompatible"));
        }

        [Test]
        public void MissingOpponentFailsInsteadOfFallingBackToGreedy()
        {
            string run = WriteRun("0", opponentJson: null);

            Assert.That(
                () => MlRunPresentationPlan.Load(run),
                Throws.InvalidOperationException.With.Message.Contains("opponent"));
        }

        [TestCase("")]
        [TestCase("sideways")]
        public void MissingOrInvalidLearnerSeatFails(string schedule)
        {
            string run = WriteRun(
                schedule, "{\"kind\":\"scripted\",\"name\":\"greedy\"}");

            Assert.That(
                () => MlRunPresentationPlan.Load(run),
                Throws.InvalidOperationException.With.Message.Contains("learner_seat"));
        }

        [Test]
        public void RecordedScenarioPathMustExist()
        {
            string run = WriteRun(
                "0", "{\"kind\":\"scripted\",\"name\":\"greedy\"}",
                writeScenario: false);

            Assert.That(
                () => MlRunPresentationPlan.Load(run),
                Throws.InvalidOperationException.With.Message.Contains(
                    Path.Combine(run, "scenario.json")));
        }

        [Test]
        public void InvalidRecordedScenarioFailsWithItsPath()
        {
            string run = WriteRun(
                "0", "{\"kind\":\"scripted\",\"name\":\"greedy\"}",
                scenarioJson: "{}");

            Assert.That(
                () => MlRunPresentationPlan.Load(run),
                Throws.InvalidOperationException.With.Message.Contains(
                    Path.Combine(run, "scenario.json")));
        }

        [Test]
        public void LegacyRunWithoutScenarioMetadataUsesVisibleLegacyDefault()
        {
            string run = WriteRun(
                "0", "{\"kind\":\"scripted\",\"name\":\"greedy\"}",
                includeScenarioMetadata: false, writeScenario: false);

            MlPresentationGame game = MlRunPresentationPlan.Load(run).PlanGame(0);

            Assert.That(game.Scenario.Id, Is.EqualTo("legacy-default"));
        }

        [Test]
        public void NestedScenarioKeyDoesNotTurnLegacyRunIntoModernRun()
        {
            string run = WriteRun(
                "0", "{\"kind\":\"scripted\",\"name\":\"greedy\"}",
                includeScenarioMetadata: false, writeScenario: false);
            AppendTopLevelProperty(
                run,
                "\"metadata\":{\"scenario\":{\"path\":\"scenario.json\"}}");

            MlPresentationGame game = MlRunPresentationPlan.Load(run).PlanGame(0);

            Assert.That(game.Scenario.Id, Is.EqualTo("legacy-default"));
        }

        [TestCase("null")]
        [TestCase("\"scenario.json\"")]
        [TestCase("[]")]
        [TestCase("{}")]
        public void ModernScenarioMetadataMustBeANonNullValidObject(
            string scenarioJson)
        {
            string run = WriteRun(
                "0", "{\"kind\":\"scripted\",\"name\":\"greedy\"}",
                includeScenarioMetadata: false, writeScenario: false);
            AppendTopLevelProperty(run, "\"scenario\":" + scenarioJson);

            Assert.That(
                () => MlRunPresentationPlan.Load(run),
                Throws.InvalidOperationException.With.Message.Contains(
                    Path.Combine(run, "run.json")).And.Message.Contains("scenario"));
        }

        [Test]
        public void NegativeGameIndexIsRejected()
        {
            string run = WriteRun(
                "0", "{\"kind\":\"scripted\",\"name\":\"greedy\"}");

            Assert.That(
                () => MlRunPresentationPlan.Load(run).PlanGame(-1),
                Throws.InstanceOf<ArgumentOutOfRangeException>());
        }

        string WriteRun(
            string learnerSeat,
            string opponentJson,
            bool includeScenarioMetadata = true,
            bool writeScenario = true,
            string scenarioJson = null)
        {
            string run = Path.Combine(_scratch, "learner-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(run);
            if (writeScenario)
                File.WriteAllText(
                    Path.Combine(run, "scenario.json"), scenarioJson ?? _scenarioJson);
            string scenario = includeScenarioMetadata
                ? ",\"scenario\":{\"path\":\"scenario.json\"," +
                  "\"template_id\":\"tactical-standard\",\"schema_version\":1}"
                : string.Empty;
            string opponent = opponentJson == null
                ? string.Empty
                : ",\"opponent_snapshot\":" + opponentJson;
            File.WriteAllText(
                Path.Combine(run, "run.json"),
                "{\"schema_version\":1,\"config\":{\"algorithm\":\"maskable_ppo\"," +
                "\"environment\":\"tactical-v1\",\"learner_seat\":" +
                Json(learnerSeat) + "},\"contract\":{\"version\":\"tactical-v1\"," +
                "\"encoding_hash\":\"" + new string('b', 64) + "\"}" +
                scenario + opponent + "}");
            return run;
        }

        string WriteSourceRun(
            string name,
            string algorithm,
            int step,
            string encodingHash = null)
        {
            string run = Path.Combine(_scratch, name);
            string checkpoints = Path.Combine(run, "checkpoints");
            Directory.CreateDirectory(checkpoints);
            File.WriteAllText(
                Path.Combine(checkpoints, "step_" + step.ToString("D9") + ".zip"),
                "model");
            File.WriteAllText(
                Path.Combine(run, "run.json"),
                "{\"schema_version\":1,\"config\":{\"algorithm\":" + Json(algorithm) +
                "},\"contract\":{\"version\":\"tactical-v1\"," +
                "\"encoding_hash\":\"" +
                (encodingHash ?? new string('b', 64)) + "\"}}");
            return run;
        }

        static string Json(string value) =>
            "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

        static string JsonFragment(string value) =>
            value.Replace("\\", "\\\\").Replace("\"", "\\\"");

        static void AppendTopLevelProperty(string run, string property)
        {
            string path = Path.Combine(run, "run.json");
            string json = File.ReadAllText(path);
            File.WriteAllText(
                path, json.Substring(0, json.Length - 1) + "," + property + "}");
        }
    }
}

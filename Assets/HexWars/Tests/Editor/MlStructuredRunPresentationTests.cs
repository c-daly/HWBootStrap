using System;
using System.IO;
using HexWars.Engine.Rl;
using HexWars.Presentation.EditorTools;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlStructuredRunPresentationTests
    {
        string _scratch;

        [SetUp]
        public void SetUp()
        {
            _scratch = Path.Combine(
                Path.GetTempPath(),
                "hexwars-structured-presentation-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_scratch);
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_scratch))
                Directory.Delete(_scratch, recursive: true);
        }

        [Test]
        public void ActiveLifecycle_UsesSourceModelWithoutWaitingForLifecycleCheckpoint()
        {
            string source = WriteStructuredModel("source");
            string lifecycle = WriteLifecycle(
                publishedRun: null,
                sourceRun: source,
                opponentJson: Scripted("greedy"));

            string revision = MlWatchPresentationTarget.ResolveRevision(lifecycle);
            MlRunPresentationPlan plan = MlRunPresentationPlan.Load(lifecycle);
            MlPresentationGame first = plan.PlanGame(0);
            MlPresentationGame second = plan.PlanGame(1);

            Assert.That(revision, Does.Contain(source));
            Assert.That(plan.RunDirectory, Is.EqualTo(Path.GetFullPath(lifecycle)));
            Assert.That(plan.LearnerRunDirectory, Is.EqualTo(source));
            Assert.That(plan.Scenario.Environment, Is.EqualTo("tactical-v3"));
            Assert.That(first.P0Spec,
                Does.Contain(JsonFragment(source)).And.Contain("\"mode\":\"live\""));
            Assert.That(first.P1Spec, Is.EqualTo("greedy"));
            Assert.That(second.P0Spec, Is.EqualTo("greedy"));
            Assert.That(second.P1Spec, Does.Contain(JsonFragment(source)));
            Assert.That(first.OpponentLabel, Is.EqualTo("Greedy"));
        }

        [Test]
        public void CompletedLifecycle_PrefersValidPublishedModel()
        {
            string source = WriteStructuredModel("source");
            string published = WriteStructuredModel("published");
            string lifecycle = WriteLifecycle(
                published, source, Scripted("random"));

            MlRunPresentationPlan plan = MlRunPresentationPlan.Load(lifecycle);
            MlPresentationGame game = plan.PlanGame(0);

            Assert.That(plan.LearnerRunDirectory, Is.EqualTo(published));
            Assert.That(game.P0Spec, Does.Contain(JsonFragment(published)));
            Assert.That(game.P1Spec, Is.EqualTo("random"));
            Assert.That(game.OpponentLabel, Is.EqualTo("Random"));
        }

        [Test]
        public void DeclaredCorruptPublication_FailsClosedWithoutUsingSourceModel()
        {
            string source = WriteStructuredModel("source");
            string published = WriteStructuredModel("published");
            File.Delete(Path.Combine(published, "checkpoints", "best.pt"));
            string lifecycle = WriteLifecycle(
                published, source, Scripted("greedy"));

            Assert.That(
                () => MlRunPresentationPlan.Load(lifecycle),
                Throws.InvalidOperationException.With.Message.Contains(published));
            Assert.That(
                MlWatchPresentationTarget.ResolveRevision(lifecycle),
                Does.Contain(published).And.Not.Contain(source));
        }

        [TestCase("fixed_run", "fixed", "run:")]
        [TestCase("live_run", "live", "\"mode\":\"live\"")]
        public void Lifecycle_ModelOpponentUsesRecordedFixedOrLiveMode(
            string kind, string mode, string expectedSpec)
        {
            string learner = WriteStructuredModel("learner");
            string opponent = WriteStructuredModel("opponent");
            string opponentJson =
                "{\"kind\":" + Json(kind) +
                ",\"mode\":" + Json(mode) +
                ",\"source_run\":" + Json(opponent) +
                ",\"checkpoint\":" +
                Json(Path.Combine(opponent, "checkpoints", "best.pt")) +
                ",\"checkpoint_sha256\":\"" + new string('a', 64) + "\"" +
                ",\"step\":0,\"algorithm\":\"structured_imitation\"}";
            string lifecycle = WriteLifecycle(
                publishedRun: null,
                sourceRun: learner,
                opponentJson: opponentJson);

            MlPresentationGame game =
                MlRunPresentationPlan.Load(lifecycle).PlanGame(0);

            Assert.That(game.P1Spec, Does.Contain(expectedSpec));
            Assert.That(game.P1Spec, Does.Contain("opponent"));
            Assert.That(game.OpponentLabel,
                Does.Contain("opponent").And.Contain(mode));
        }

        [Test]
        public void FailedViewerAttempt_CanRetrySameRunningLifecycleAndSeeNewPublication()
        {
            string source = WriteStructuredModel("source");
            string lifecycle = WriteLifecycle(
                publishedRun: null,
                sourceRun: source,
                opponentJson: Scripted("greedy"));
            var watch = new MlStartAndWatchState();
            var ui = new MlLabWindowState();
            ui.MarkLaunched();
            watch.Begin(requested: true);
            string sourceRevision =
                MlWatchPresentationTarget.ResolveRevision(lifecycle);
            Assert.That(watch.TryQueue(sourceRevision), Is.True);
            watch.Apply(
                MlViewerLaunchResult.Failed("policy server did not start"), ui);

            string published = WriteStructuredModel("published");
            RewriteLifecyclePublication(lifecycle, published);
            watch.Retry();
            string publishedRevision =
                MlWatchPresentationTarget.ResolveRevision(lifecycle);

            Assert.That(publishedRevision, Is.Not.EqualTo(sourceRevision));
            Assert.That(publishedRevision, Does.Contain(published));
            Assert.That(watch.TryQueue(publishedRevision), Is.True);
            Assert.That(watch.LaunchPending, Is.True);
        }

        string WriteStructuredModel(string name)
        {
            string run = Path.Combine(_scratch, name);
            Directory.CreateDirectory(Path.Combine(run, "checkpoints"));
            File.Copy(
                Path.Combine("python", "config",
                    "annihilation-structured-imitation-v1.json"),
                Path.Combine(run, "scenario.json"));
            MlTrainingScenario scenario = MlTrainingScenarioFile.Load(
                Path.Combine(run, "scenario.json"));
            TrainingScenario engine = MlTrainingScenarioPreflight.ToEngine(scenario);
            TacticalV3Contract identity = TacticalV3Contract.Create(
                engine.BuildTacticalV3(), MlEnvironmentKind.Duel);
            File.Copy(
                Path.Combine("python", "tests", "fixtures", "tactical_v3",
                    "seed-41-duel-spaces.json"),
                Path.Combine(run, "policy-identity.json"));
            File.WriteAllText(Path.Combine(run, "checkpoints", "best.pt"), "checkpoint");
            File.WriteAllText(Path.Combine(run, "corpus-manifest.json"), "{}");
            File.WriteAllText(Path.Combine(run, "metrics.jsonl"), "{}\n");
            File.WriteAllText(Path.Combine(run, "inference-fixture.json"), "{}");
            File.WriteAllText(
                Path.Combine(run, "run.json"),
                "{\"schema_version\":2,\"state\":\"completed\"," +
                "\"evidence_status\":\"unsealed-experimental\"," +
                "\"config\":{\"algorithm\":\"structured_imitation\"}," +
                "\"contract\":{\"environment\":\"tactical-v3\"," +
                "\"version\":\"tactical-v3\",\"environment_kind\":\"duel\"," +
                "\"contract_hash\":" + Json(identity.ContractHash) + "," +
                "\"encoding_hash\":" + Json(identity.EncodingHash) + "," +
                "\"capacity_hash\":" + Json(identity.CapacityHash) + "}," +
                "\"policy_identity\":\"policy-identity.json\"," +
                "\"latest_checkpoint\":\"checkpoints/best.pt\"," +
                "\"latest_checkpoint_step\":0," +
                "\"dataset_manifest_sha256\":\"" + new string('a', 64) + "\"," +
                "\"best_epoch\":0,\"best_validation_policy_nll\":1.0}");
            return run;
        }

        string WriteLifecycle(
            string publishedRun,
            string sourceRun,
            string opponentJson)
        {
            string run = Path.Combine(
                _scratch, "lifecycle-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(run);
            File.Copy(
                Path.Combine("python", "config",
                    "annihilation-structured-imitation-v1.json"),
                Path.Combine(run, "scenario.json"));
            File.WriteAllText(
                Path.Combine(run, "run.json"),
                LifecycleJson(publishedRun, sourceRun, opponentJson));
            return run;
        }

        static string LifecycleJson(
            string publishedRun,
            string sourceRun,
            string opponentJson) =>
            "{\"schema_version\":1," +
            "\"state\":\"running\",\"latest_checkpoint\":null," +
            "\"config\":{\"algorithm\":\"structured_dagger\"," +
            "\"learner_seat\":\"alternating\"}," +
            "\"contract\":{\"environment\":\"tactical-v3\"," +
            "\"version\":\"tactical-v3\"}," +
            "\"scenario\":{\"path\":\"scenario.json\",\"schema_version\":1}," +
            "\"opponent_snapshot\":" + opponentJson + "," +
            "\"published_run\":" +
            (publishedRun == null ? "null" : Json(publishedRun)) + "," +
            "\"source_policy\":{\"run\":" + Json(sourceRun) + "}}";

        static void RewriteLifecyclePublication(string lifecycle, string published)
        {
            string path = Path.Combine(lifecycle, "run.json");
            string json = File.ReadAllText(path);
            string marker = "\"published_run\":null";
            Assert.That(json, Does.Contain(marker));
            File.WriteAllText(
                path,
                json.Replace(marker, "\"published_run\":" + Json(published)));
        }

        static string Scripted(string name) =>
            "{\"kind\":\"scripted\",\"name\":" + Json(name) + "}";

        static string Json(string value) =>
            "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

        static string JsonFragment(string value) =>
            value.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}

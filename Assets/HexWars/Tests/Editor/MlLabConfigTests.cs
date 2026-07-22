using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlLabConfigTests
    {
        [Test]
        public void Validate_RejectsUnsafeOrIncompleteTrainingConfiguration()
        {
            var config = MlLabConfig.Default();
            config.RunName = "../bad run";
            config.TotalTimesteps = 0;
            config.CheckpointInterval = 0;
            config.Workers = 0;

            var errors = config.Validate();

            Assert.That(errors, Has.Some.Contains("Run name"));
            Assert.That(errors, Has.Some.Contains("Timesteps"));
            Assert.That(errors, Has.Some.Contains("Checkpoint"));
            Assert.That(errors, Has.Some.Contains("Workers"));
        }

        [Test]
        public void BuildTrainArguments_SerializesAlgorithmOpponentSeatAndTrackers()
        {
            var config = MlLabConfig.Default();
            config.RunName = "counter run";
            config.Algorithm = MlAlgorithm.MaskablePpo;
            config.OpponentKind = MlOpponentKind.FixedRun;
            config.OpponentPath = @"C:\runs\frozen opponent";
            config.LearnerSeat = MlLearnerSeat.Alternating;
            config.Trackers.Add(new MlTrackerConfig("wandb"));
            config.Trackers.Add(new MlTrackerConfig("custom", "my_tracker:record"));
            config.WandbProject = "hex wars";

            string args = config.BuildTrainArguments();

            Assert.That(args, Does.Contain("train"));
            Assert.That(args, Does.Contain("--algorithm maskable_ppo"));
            Assert.That(args, Does.Contain("--run \"counter run\""));
            Assert.That(args, Does.Contain("--opponent \"run:C:\\runs\\frozen opponent\""));
            Assert.That(args, Does.Contain("--learner-seat alternating"));
            Assert.That(args, Does.Contain("--tracker wandb"));
            Assert.That(args, Does.Contain("--tracker custom=my_tracker:record"));
            Assert.That(args, Does.Contain("--wandb-project \"hex wars\""));
            Assert.That(args, Does.Contain("--no-console-output"));
        }

        [Test]
        public void BuildResumeArguments_UsesAuthoritativeSourceRun()
        {
            var config = MlLabConfig.Default();
            config.RunName = "continued";
            config.ResumeSource = @"C:\runs\source run";

            string args = config.BuildResumeArguments();

            Assert.That(args, Does.StartWith("resume \"C:\\runs\\source run\" --run continued"));
            Assert.That(args, Does.Contain("--no-console-output"));
        }

        [Test]
        public void BuildTrainArguments_LiveRunOpponentRequestsCheckpointReloading()
        {
            var config = MlLabConfig.Default();
            config.OpponentKind = MlOpponentKind.LiveRun;
            config.OpponentPath = @"C:\runs\live opponent";

            string args = config.BuildTrainArguments();

            Assert.That(args, Does.Contain("\\\"kind\\\":\\\"run\\\""));
            Assert.That(args, Does.Contain("\\\"mode\\\":\\\"live\\\""));
            Assert.That(args, Does.Not.Contain("--opponent \\\"run:"));
        }

        [Test]
        public void BuildTrainArguments_DoesNotLeakDisabledWandbOptions()
        {
            var config = MlLabConfig.Default();
            config.WandbProject = "remembered-project";

            Assert.That(config.BuildTrainArguments(), Does.Not.Contain("--wandb-project"));
        }

        [Test]
        public void Defaults_SelectTacticalV1AndEmitItExplicitly()
        {
            var config = MlLabConfig.Default();

            Assert.That(config.Environment, Is.EqualTo(MlEnvironmentContract.TacticalV1));
            Assert.That(config.BuildTrainArguments(), Does.Contain("--environment tactical-v1"));
        }

        [Test]
        public void OpponentChoices_ExcludeManifestlessCheckpointPaths()
        {
            Assert.That(System.Enum.GetNames(typeof(MlOpponentKind)), Does.Not.Contain("FixedCheckpoint"));
        }

        [Test]
        public void BuildTrainArguments_EmitsAdaptiveEnvironmentWhenSelected()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.AdaptiveV1;

            Assert.That(config.BuildTrainArguments(), Does.Contain("--environment adaptive-v1"));
        }

        [Test]
        public void AdaptivePreflight_UsesManifestSemanticsForExistingRun()
        {
            const string manifest = "{\"contract\":{\"version\":\"adaptive-v1\",\"semantics\":{" +
                "\"fixed_template_count\":5,\"custom_template_count\":4," +
                "\"max_controllable_units\":17,\"starting_unit_count\":4," +
                "\"starting_army_budget\":99,\"fog_rule\":\"manifest-hidden-rule\"," +
                "\"templates\":[{\"name\":\"Manifest Front\",\"fixed\":true}]}}}";

            var summary = MlEnvironmentSummary.FromRunManifest(manifest);

            Assert.That(summary.ContractVersion, Is.EqualTo("adaptive-v1"));
            Assert.That(summary.FixedTemplateCount, Is.EqualTo(5));
            Assert.That(summary.CustomTemplateCount, Is.EqualTo(4));
            Assert.That(summary.MaxControllableUnits, Is.EqualTo(17));
            Assert.That(summary.StartingUnitCount, Is.EqualTo(4));
            Assert.That(summary.StartingArmyBudget, Is.EqualTo(99));
            Assert.That(summary.FixedRoles, Is.EqualTo(new[] { "Manifest Front" }));
            Assert.That(summary.HiddenDeployment, Is.True);
        }

        [Test]
        public void AdaptivePreflight_ListsPinnedSelectionSemantics()
        {
            string text = MlEnvironmentSummary.ForSelection(MlEnvironmentContract.AdaptiveV1).DisplayText;

            Assert.That(text, Does.Contain("adaptive-v1"));
            Assert.That(text, Does.Contain("Frontline, Assault, Marksman, Artillery, Recon, Support"));
            Assert.That(text, Does.Contain("3 custom slots"));
            Assert.That(text, Does.Contain("24 maximum units"));
            Assert.That(text, Does.Contain("6 starting units"));
            Assert.That(text, Does.Contain("132 setup points"));
            Assert.That(text, Does.Contain("combined-arms scripted deployment"));
            Assert.That(text, Does.Contain("hidden deployment"));
        }
    }
}

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
            config.OpponentKind = MlOpponentKind.FixedCheckpoint;
            config.OpponentPath = @"C:\models\old model.zip";
            config.LearnerSeat = MlLearnerSeat.Alternating;
            config.Trackers.Add(new MlTrackerConfig("wandb"));
            config.Trackers.Add(new MlTrackerConfig("custom", "my_tracker:record"));
            config.WandbProject = "hex wars";

            string args = config.BuildTrainArguments();

            Assert.That(args, Does.Contain("train"));
            Assert.That(args, Does.Contain("--algorithm maskable_ppo"));
            Assert.That(args, Does.Contain("--run \"counter run\""));
            Assert.That(args, Does.Contain("--opponent \"ppo:C:\\models\\old model.zip\""));
            Assert.That(args, Does.Contain("--learner-seat alternating"));
            Assert.That(args, Does.Contain("--tracker wandb"));
            Assert.That(args, Does.Contain("--tracker custom=my_tracker:record"));
            Assert.That(args, Does.Contain("--wandb-project \"hex wars\""));
        }

        [Test]
        public void BuildResumeArguments_UsesAuthoritativeSourceRun()
        {
            var config = MlLabConfig.Default();
            config.RunName = "continued";
            config.ResumeSource = @"C:\runs\source run";

            Assert.That(config.BuildResumeArguments(),
                Does.StartWith("resume \"C:\\runs\\source run\" --run continued"));
        }

        [Test]
        public void BuildTrainArguments_DoesNotLeakDisabledWandbOptions()
        {
            var config = MlLabConfig.Default();
            config.WandbProject = "remembered-project";

            Assert.That(config.BuildTrainArguments(), Does.Not.Contain("--wandb-project"));
        }
    }
}

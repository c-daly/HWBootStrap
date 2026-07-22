using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlRunStatusTests
    {
        [Test]
        public void ParseJson_ReadsDurableRunTruthAndNestedArtifacts()
        {
            const string json = "{\"command\":\"status\",\"ok\":true,\"run\":{\"run_name\":\"ppo1\",\"state\":\"running\",\"pid\":42,\"step\":120,\"target_step\":1000,\"latest_checkpoint\":\"step_120.zip\",\"throughput\":81.5,\"tracker_degraded\":true}}";

            var status = MlRunStatus.Parse(json);

            Assert.That(status.Ok, Is.True);
            Assert.That(status.RunName, Is.EqualTo("ppo1"));
            Assert.That(status.State, Is.EqualTo(MlRunState.Running));
            Assert.That(status.Pid, Is.EqualTo(42));
            Assert.That(status.Step, Is.EqualTo(120));
            Assert.That(status.TargetStep, Is.EqualTo(1000));
            Assert.That(status.LatestCheckpoint, Is.EqualTo("step_120.zip"));
            Assert.That(status.Throughput, Is.EqualTo(81.5).Within(0.01));
            Assert.That(status.TrackerDegraded, Is.True);
        }

        [Test]
        public void ParseJson_RecognizesActualDegradedTrackerRecords()
        {
            const string json = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"running\",\"tracker_status\":[{\"name\":\"wandb:0\",\"status\":\"degraded\",\"message\":\"offline\"}]}}}";

            Assert.That(MlRunStatus.Parse(json).TrackerDegraded, Is.True);
        }

        [Test]
        public void LogBuffer_RetainsOnlyNewestBoundedLines()
        {
            var log = new MlLogBuffer(3);
            log.Add("one");
            log.Add("two");
            log.Add("three");
            log.Add("four");

            Assert.That(log.Lines, Is.EqualTo(new[] { "two", "three", "four" }));
        }

        [Test]
        public void ActiveRunAttachment_SurvivesOwnerDisposalWithoutSecrets()
        {
            MlRunAttachment.Forget();
            MlRunAttachment.Remember(@"C:\runs\active", 42);

            var restored = MlRunAttachment.Restore();

            Assert.That(restored.Exists, Is.True);
            Assert.That(restored.RunDirectory, Is.EqualTo(@"C:\runs\active"));
            Assert.That(restored.Pid, Is.EqualTo(42));
            MlRunAttachment.Forget();
            Assert.That(MlRunAttachment.Restore().Exists, Is.False);
        }

        [TestCase("created", MlRunState.Created)]
        [TestCase("stopping", MlRunState.Stopping)]
        [TestCase("completed", MlRunState.Completed)]
        [TestCase("failed", MlRunState.Failed)]
        [TestCase("unexpected", MlRunState.Unknown)]
        public void ParseState_IsStableForCliValues(string value, MlRunState expected) =>
            Assert.That(MlRunStatus.ParseState(value), Is.EqualTo(expected));
    }
}

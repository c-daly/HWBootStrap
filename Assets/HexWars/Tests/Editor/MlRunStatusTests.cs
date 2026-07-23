using System;
using System.IO;
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
        public void ParseJson_ReadsSeatAudit()
        {
            const string json = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"running\"," +
                "\"config\":{\"learner_seat\":\"alternating\"}}," +
                "\"seat_audit\":{\"seat_0_episodes\":12,\"seat_1_episodes\":11," +
                "\"readable\":true,\"balanced\":true,\"warning\":\"\"}}}";

            MlRunStatus status = MlRunStatus.Parse(json);

            Assert.That(status.LearnerSeat, Is.EqualTo("alternating"));
            Assert.That(status.SeatAuditBalanceApplicable, Is.True);
            Assert.That(status.Seat0Episodes, Is.EqualTo(12));
            Assert.That(status.Seat1Episodes, Is.EqualTo(11));
            Assert.That(status.SeatAuditReadable, Is.True);
            Assert.That(status.SeatAuditBalanced, Is.True);
        }

        [Test]
        public void SeatAuditDisplay_UsesInfoForUnreadableOrNonterminalImbalance()
        {
            const string unreadable = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"running\"}," +
                "\"seat_audit\":{\"readable\":false,\"balanced\":false,\"warning\":\"monitor.csv: missing header\"}}}";
            const string inProgress = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"running\"," +
                "\"config\":{\"learner_seat\":\"alternating\"}}," +
                "\"seat_audit\":{\"seat_0_episodes\":9,\"seat_1_episodes\":1,\"readable\":true," +
                "\"balanced\":false,\"warning\":\"\"}}}";

            Assert.That(MlRunStatus.Parse(unreadable).SeatAuditShowsInfo, Is.True);
            Assert.That(MlRunStatus.Parse(unreadable).SeatAuditShowsWarning, Is.False);
            Assert.That(MlRunStatus.Parse(inProgress).SeatAuditBalanceApplicable, Is.True);
            Assert.That(MlRunStatus.Parse(inProgress).SeatAuditShowsInfo, Is.True);
            Assert.That(MlRunStatus.Parse(inProgress).SeatAuditShowsWarning, Is.False);
        }

        [Test]
        public void SeatAuditDisplay_UsesWarningOnlyForTerminalMaterialImbalance()
        {
            const string running = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"running\"," +
                "\"config\":{\"learner_seat\":\"alternating\"}}," +
                "\"seat_audit\":{\"readable\":true,\"balanced\":false,\"warning\":\"imbalanced\"}}}";
            const string completed = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"completed\"," +
                "\"config\":{\"learner_seat\":\"alternating\"}}," +
                "\"seat_audit\":{\"readable\":true,\"balanced\":false,\"warning\":\"imbalanced\"}}}";
            const string fixedCompleted = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"completed\"," +
                "\"config\":{\"learner_seat\":\"0\"}}," +
                "\"seat_audit\":{\"readable\":true,\"balanced\":false,\"warning\":\"\"}}}";

            Assert.That(MlRunStatus.Parse(running).SeatAuditShowsWarning, Is.False);
            Assert.That(MlRunStatus.Parse(completed).SeatAuditShowsWarning, Is.True);
            Assert.That(MlRunStatus.Parse(completed).SeatAuditShowsInfo, Is.False);
            Assert.That(MlRunStatus.Parse(fixedCompleted).SeatAuditShowsWarning, Is.False);
            Assert.That(MlRunStatus.Parse(fixedCompleted).SeatAuditShowsInfo, Is.False);
        }

        [Test]
        public void SeatAuditDisplay_RunningFixedSeatShowsCountsWithoutBalanceBox()
        {
            const string json = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"running\"," +
                "\"config\":{\"learner_seat\":\"1\"}}," +
                "\"seat_audit\":{\"seat_0_episodes\":1,\"seat_1_episodes\":9,\"readable\":true," +
                "\"balanced\":false,\"warning\":\"\"}}}";

            MlRunStatus status = MlRunStatus.Parse(json);

            Assert.That(status.LearnerSeat, Is.EqualTo("1"));
            Assert.That(status.SeatAuditBalanceApplicable, Is.False);
            Assert.That(status.Seat0Episodes, Is.EqualTo(1));
            Assert.That(status.Seat1Episodes, Is.EqualTo(9));
            Assert.That(status.SeatAuditShowsInfo, Is.False);
            Assert.That(status.SeatAuditShowsWarning, Is.False);
        }

        [Test]
        public void SeatAuditDisplay_MissingLearnerSeatConfigDefaultsToNotApplicable()
        {
            const string json = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"running\"}," +
                "\"seat_audit\":{\"seat_0_episodes\":9,\"seat_1_episodes\":1,\"readable\":true," +
                "\"balanced\":false,\"warning\":\"\"}}}";

            MlRunStatus status = MlRunStatus.Parse(json);

            Assert.That(status.LearnerSeat, Is.Empty);
            Assert.That(status.SeatAuditBalanceApplicable, Is.False);
            Assert.That(status.SeatAuditShowsInfo, Is.False);
            Assert.That(status.SeatAuditShowsWarning, Is.False);
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
        public void ActiveRunAttachment_SurvivesOwnerDisposalWithoutProcessIdentity()
        {
            MlRunAttachment.Forget();
            MlRunAttachment.Remember(@"C:\runs\active");

            var restored = MlRunAttachment.Restore();

            Assert.That(restored.Exists, Is.True);
            Assert.That(restored.RunDirectory, Is.EqualTo(@"C:\runs\active"));
            MlRunAttachment.Forget();
            Assert.That(MlRunAttachment.Restore().Exists, Is.False);
        }

        [Test]
        public void RunLogTail_ReturnsOnlyNewestLinesFromDurableTrainLog()
        {
            string runDirectory = Path.Combine(Path.GetTempPath(), "hexwars-ml-log-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(runDirectory);
            try
            {
                File.WriteAllLines(Path.Combine(runDirectory, "train.log"),
                    new[] { "one", "two", "three", "four", "five" });

                Assert.That(MlRunLog.ReadTail(runDirectory, 3),
                    Is.EqualTo(new[] { "three", "four", "five" }));
            }
            finally
            {
                Directory.Delete(runDirectory, true);
            }
        }

        [Test]
        public void SharedFileSnapshot_ReadsWhileWriterRemainsOpenAndCanAppend()
        {
            string path = Path.Combine(Path.GetTempPath(), "hexwars-ml-shared-" + Guid.NewGuid().ToString("N") + ".log");
            try
            {
                using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write,
                           FileShare.ReadWrite | FileShare.Delete))
                using (var writer = new StreamWriter(stream) { AutoFlush = true })
                {
                    writer.WriteLine("one");
                    writer.WriteLine("two");

                    Assert.That(MlSharedFileSnapshot.ReadTail(path, 1), Is.EqualTo(new[] { "two" }));

                    writer.WriteLine("three");
                    Assert.That(MlSharedFileSnapshot.ReadTail(path, 2),
                        Is.EqualTo(new[] { "two", "three" }));
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Test]
        public void SharedFileSnapshot_LastNonEmptyLineIgnoresTrailingBlanksFromOpenWriter()
        {
            string path = Path.Combine(Path.GetTempPath(), "hexwars-ml-progress-" + Guid.NewGuid().ToString("N") + ".csv");
            try
            {
                using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write,
                           FileShare.ReadWrite | FileShare.Delete))
                using (var writer = new StreamWriter(stream) { AutoFlush = true })
                {
                    writer.WriteLine("timestamp,step,reward,loss,throughput");
                    writer.WriteLine("2026-07-22T13:39:00Z,64,1,0.5,81.25");
                    writer.WriteLine();

                    Assert.That(MlSharedFileSnapshot.ReadLastNonEmptyLine(path),
                        Is.EqualTo("2026-07-22T13:39:00Z,64,1,0.5,81.25"));
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Test]
        public void SharedFileSnapshot_TailReadIsByteBoundedAndReturnsMixedNewlineSuffix()
        {
            string path = Path.Combine(Path.GetTempPath(), "hexwars-ml-bounded-log-" + Guid.NewGuid().ToString("N") + ".log");
            try
            {
                using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write,
                           FileShare.ReadWrite | FileShare.Delete))
                using (var writer = new StreamWriter(stream) { AutoFlush = true })
                {
                    writer.Write(new string('x', MlSharedFileSnapshot.TailReadLimitBytes * 2));
                    writer.Write("\r\nold\r\none\r\ntwo\nthree\n");

                    string[] lines = MlSharedFileSnapshot.ReadTail(path, 3, out int bytesRead);

                    Assert.That(lines, Is.EqualTo(new[] { "one", "two", "three" }));
                    Assert.That(bytesRead, Is.LessThanOrEqualTo(MlSharedFileSnapshot.TailReadLimitBytes));

                    writer.Write("four\n");
                    Assert.That(MlSharedFileSnapshot.ReadTail(path, 2),
                        Is.EqualTo(new[] { "three", "four" }));
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Test]
        public void SharedFileSnapshot_LastCompleteMetricReadIsByteBounded()
        {
            string path = Path.Combine(Path.GetTempPath(), "hexwars-ml-bounded-progress-" + Guid.NewGuid().ToString("N") + ".csv");
            try
            {
                using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write,
                           FileShare.ReadWrite | FileShare.Delete))
                using (var writer = new StreamWriter(stream) { AutoFlush = true })
                {
                    writer.Write(new string('x', MlSharedFileSnapshot.LastLineReadLimitBytes * 2));
                    writer.Write("\ntimestamp,step,reward,loss,throughput\r\n");
                    writer.Write("2026-07-22T13:39:00Z,64,1,0.5,81.25\r\n\r\n");
                    writer.Write("2026-07-22T13:40:00Z,128,1,0.4,");

                    string line = MlSharedFileSnapshot.ReadLastNonEmptyLine(path, out int bytesRead);

                    Assert.That(line, Is.EqualTo("2026-07-22T13:39:00Z,64,1,0.5,81.25"));
                    Assert.That(bytesRead, Is.LessThanOrEqualTo(MlSharedFileSnapshot.LastLineReadLimitBytes));
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
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

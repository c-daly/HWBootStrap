using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlTrainerLivenessTests
    {
        // ---- Reattach decision (D1: "Lab stops lying about trainers") ----

        [Test]
        public void PersistedPidExistsAndMatches_IsTrackedAlive()
        {
            Assert.That(
                MlTrainerLivenessPolicy.Decide(
                    hasPersistedPid: true, processExists: true, processIdentityMatches: true,
                    progressFresh: false),
                Is.EqualTo(MlTrainerLivenessState.TrackedAlive),
                "a confirmed still-running process is alive regardless of progress.csv freshness");
        }

        [Test]
        public void PidGoneButProgressFresh_IsAliveByFiles()
        {
            Assert.That(
                MlTrainerLivenessPolicy.Decide(
                    hasPersistedPid: false, processExists: false, processIdentityMatches: false,
                    progressFresh: true),
                Is.EqualTo(MlTrainerLivenessState.AliveByFiles));
        }

        [Test]
        public void PidGoneAndProgressStale_IsDeadWithLastKnownStepLeftToTheCaller()
        {
            // The enum itself only says "dead"; MlTrainerStatusFormatter (below) is what carries the
            // last known step into the surfaced text once the caller supplies it.
            Assert.That(
                MlTrainerLivenessPolicy.Decide(
                    hasPersistedPid: false, processExists: false, processIdentityMatches: false,
                    progressFresh: false),
                Is.EqualTo(MlTrainerLivenessState.Dead));
        }

        [Test]
        public void PidReusedByAnUnrelatedProcess_IsNeverFalselyTrackedAlive()
        {
            // A PID being alive is not enough on its own -- the OS recycles PIDs, so if the process now
            // holding this PID isn't the interpreter we launched, that must never read as "alive."
            Assert.That(
                MlTrainerLivenessPolicy.Decide(
                    hasPersistedPid: true, processExists: true, processIdentityMatches: false,
                    progressFresh: true),
                Is.Not.EqualTo(MlTrainerLivenessState.TrackedAlive));
            Assert.That(
                MlTrainerLivenessPolicy.Decide(
                    hasPersistedPid: true, processExists: true, processIdentityMatches: false,
                    progressFresh: false),
                Is.EqualTo(MlTrainerLivenessState.Dead));
        }

        [TestCase(MlTrainerLivenessState.TrackedAlive, true)]
        [TestCase(MlTrainerLivenessState.AliveByFiles, true)]
        [TestCase(MlTrainerLivenessState.Dead, false)]
        public void IsAlive_TreatsOnlyDeadAsNotAlive(MlTrainerLivenessState state, bool expected) =>
            Assert.That(MlTrainerLivenessPolicy.IsAlive(state), Is.EqualTo(expected));

        // ---- Watch gate integration: the villVBONFab false positive ----

        [Test]
        public void WatchGate_FilesFreshDespiteLostProcessTracking_DoesNotFalselyGiveUp()
        {
            // Reproduces run villVBONFab: Start & Watch gave up around 14:30 reporting "training ended
            // before a validated checkpoint appeared" while the trainer kept writing progress.csv and
            // wrote checkpoint 20,480 at 14:35 -- the editor had simply lost its handle on the detached
            // process (no persisted/matching pid available here), even though the run's own files
            // proved it was still going.
            MlTrainerLivenessState liveness = MlTrainerLivenessPolicy.Decide(
                hasPersistedPid: false, processExists: false, processIdentityMatches: false,
                progressFresh: true);
            bool trainingAlive = MlTrainerLivenessPolicy.IsAlive(liveness);

            Assert.That(liveness, Is.EqualTo(MlTrainerLivenessState.AliveByFiles));
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: true,
                    checkpointReady: false,
                    trainingAlive: trainingAlive,
                    ceilingDeadlinePassed: false),
                Is.EqualTo(MlWatchStartDecision.WaitAndRetry),
                "fresh progress.csv must keep Start & Watch retrying instead of reporting training ended");
        }

        [Test]
        public void WatchGate_FilesTrulyStaleAndNoProcess_StillGivesUp()
        {
            // The flip side: once files really have gone stale with no process to back them up, the
            // gate must still be able to give up -- the fix must not make it wait forever.
            bool trainingAlive = MlTrainerLivenessPolicy.IsAlive(
                MlTrainerLivenessPolicy.Decide(
                    hasPersistedPid: false, processExists: false, processIdentityMatches: false,
                    progressFresh: false));

            Assert.That(trainingAlive, Is.False);
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: true,
                    checkpointReady: false,
                    trainingAlive: trainingAlive,
                    ceilingDeadlinePassed: false),
                Is.EqualTo(MlWatchStartDecision.GiveUp));
        }

        // ---- Progress freshness ----

        [TestCase(0.0, true)]
        [TestCase(4.99, true)]
        [TestCase(5.0, false)]
        [TestCase(30.0, false)]
        public void ProgressFreshness_ThresholdIsAFewMinutes(double minutesSinceLastWrite, bool expectedFresh) =>
            Assert.That(MlTrainerProgressFreshness.IsFresh(minutesSinceLastWrite), Is.EqualTo(expectedFresh));

        // ---- Status surfacing (D2) ----

        [Test]
        public void Describe_ConfirmedExit_ReportsLastKnownStep()
        {
            string status = MlTrainerStatusFormatter.Describe(
                confirmedExited: true, minutesSinceProgress: 1.0, step: 20480, targetStep: 200000);

            Assert.That(status, Is.EqualTo("trainer exited (step 20,480 of 200,000)"));
        }

        [Test]
        public void Describe_NoConfirmedExitButProgressStale_ReportsStalledWithMinutes()
        {
            string status = MlTrainerStatusFormatter.Describe(
                confirmedExited: false, minutesSinceProgress: 12.0, step: 1000, targetStep: 200000);

            Assert.That(status, Is.EqualTo("trainer stalled — no progress for 12 min"));
        }

        [Test]
        public void Describe_HealthyRun_IsBlank()
        {
            string status = MlTrainerStatusFormatter.Describe(
                confirmedExited: false, minutesSinceProgress: 0.2, step: 1000, targetStep: 200000);

            Assert.That(status, Is.Empty);
        }

        [Test]
        public void Describe_ConfirmedExitTakesPriorityOverStaleProgressWording()
        {
            string status = MlTrainerStatusFormatter.Describe(
                confirmedExited: true, minutesSinceProgress: 45.0, step: 5000, targetStep: 200000);

            Assert.That(status, Does.StartWith("trainer exited"));
        }
    }
}

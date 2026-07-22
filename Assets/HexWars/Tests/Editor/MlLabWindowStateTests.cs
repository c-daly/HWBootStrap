using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlLabWindowStateTests
    {
        [Test]
        public void LaunchLifecycle_TransitionsWithoutModalState()
        {
            var state = new MlLabWindowState();
            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Idle));

            state.BeginValidation();
            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Validating));
            state.MarkLaunched();
            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Running));
            Assert.That(state.LaunchedHere, Is.True);

            state.BeginStopping();
            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Stopping));
            state.Apply(MlRunState.Stopped, 0);
            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Completed));
        }

        [Test]
        public void RestoredRunningStatus_IsExternalAndTerminalStatusIsStable()
        {
            var state = new MlLabWindowState();

            state.Apply(MlRunState.Running, 99);
            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.ExternallyRunning));
            Assert.That(state.LaunchedHere, Is.False);

            state.Apply(MlRunState.Completed, 0);
            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Completed));
        }

        [Test]
        public void FailurePreservesInlineMessageAndCanReturnToIdle()
        {
            var state = new MlLabWindowState();
            state.Fail("Python was not found");

            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Failed));
            Assert.That(state.Error, Is.EqualTo("Python was not found"));

            state.Reset();
            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Idle));
            Assert.That(state.Error, Is.Empty);
        }

        [Test]
        public void LocallyLaunchedRun_RemainsRunningAcrossDifferentReportedPid()
        {
            var state = new MlLabWindowState();
            state.MarkLaunched();

            state.Apply(MlRunState.Running, 99);

            Assert.That(state.Phase, Is.EqualTo(MlLabUiPhase.Running));
        }

        [Test]
        public void DoctorReport_SurfacesRequiredFailuresAndOptionalCapabilities()
        {
            const string json = "{\"ok\":true,\"result\":{\"ok\":false,\"checks\":[{\"name\":\"gymserver_handshake\",\"ok\":false,\"required\":true,\"detail\":\"missing DLL\"},{\"name\":\"cuda\",\"ok\":false,\"required\":false,\"detail\":\"CPU only\"}]}}";

            var report = MlDoctorReport.Parse(json);

            Assert.That(report.Healthy, Is.False);
            Assert.That(report.Summary, Does.Contain("gymserver_handshake: missing DLL"));
            Assert.That(report.Summary, Does.Contain("Optional unavailable: cuda"));
        }

        [Test]
        public void DoctorReport_UnreadableResponseBecomesInlineFailure()
        {
            var report = MlDoctorReport.Parse(string.Empty);

            Assert.That(report.Healthy, Is.False);
            Assert.That(report.Summary, Does.Contain("unreadable"));
        }

        [TestCase("maskable_ppo", "ppo:")]
        [TestCase("masked_dqn", "dqn:")]
        [TestCase("other", null)]
        public void ModelAlgorithm_IsResolvedOnlyFromRunMetadata(string algorithm, string expected)
        {
            string json = "{\"config\":{\"algorithm\":\"" + algorithm + "\"}}";

            Assert.That(HexWars.Presentation.EditorTools.ReplayViewerMenu.AlgorithmPrefixFromManifest(json),
                Is.EqualTo(expected));
        }
    }
}

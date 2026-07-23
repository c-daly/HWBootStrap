using System;
using System.IO;
using System.Linq;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlLabWindowStateTests
    {
        static readonly string BuiltInLibraryPath =
            Path.Combine("python", "config", "training-game-templates.json");

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

        [Test]
        public void EnvironmentChange_SelectsThatEnvironmentsStandardTemplate()
        {
            var session = new MlTrainingScenarioSession(
                MlTrainingScenarioLibrary.Load(BuiltInLibraryPath));

            session.SelectEnvironment(MlEnvironmentContract.AdaptiveV1);

            Assert.That(session.WorkingCopy.Id, Is.EqualTo("adaptive-standard"));
            Assert.That(session.SelectedTemplateId, Is.EqualTo("adaptive-standard"));
        }

        [Test]
        public void EditIsSessionOnlyUntilSaveAndReloadDiscardsIt()
        {
            var session = new MlTrainingScenarioSession(
                MlTrainingScenarioLibrary.Load(BuiltInLibraryPath));
            session.SelectEnvironment(MlEnvironmentContract.AdaptiveV1);
            session.WorkingCopy.Board.Width = 64;

            session.Reload();

            Assert.That(session.WorkingCopy.Board.Width, Is.EqualTo(13));
        }

        [Test]
        public void TemplateSelection_IsFilteredToTheSelectedEnvironment()
        {
            var session = new MlTrainingScenarioSession(
                MlTrainingScenarioLibrary.Load(BuiltInLibraryPath));

            Assert.That(
                session.AvailableTemplates.Select(item => item.Id),
                Is.EqualTo(new[]
                {
                    "tactical-standard",
                    "tactical-long-battle",
                    "tactical-large-battle",
                }));
            Assert.Throws<ArgumentException>(
                () => session.SelectTemplate("adaptive-standard"));

            session.SelectEnvironment(MlEnvironmentContract.AdaptiveV1);

            Assert.That(
                session.AvailableTemplates.Select(item => item.Id),
                Is.EqualTo(new[]
                {
                    "adaptive-standard",
                    "adaptive-long-battle",
                    "adaptive-large-battle",
                }));
        }

        [Test]
        public void SaveAsUniqueId_PersistsAndSelectsTheNewTemplate()
        {
            string scratch = CreateLibraryCopy();
            try
            {
                var session = MlTrainingScenarioSession.Load(scratch);
                session.SelectEnvironment(MlEnvironmentContract.AdaptiveV1);
                session.SetSaveIdentity("Fast Skirmish", "adaptive-fast-skirmish");

                bool saved = session.SaveAsTemplate();

                Assert.That(saved, Is.True);
                Assert.That(session.OverwriteArmed, Is.False);
                Assert.That(session.SelectedTemplateId, Is.EqualTo("adaptive-fast-skirmish"));
                Assert.That(
                    MlTrainingScenarioLibrary.Load(scratch).Templates
                        .Single(item => item.Id == "adaptive-fast-skirmish").Name,
                    Is.EqualTo("Fast Skirmish"));
            }
            finally
            {
                if (File.Exists(scratch)) File.Delete(scratch);
            }
        }

        [Test]
        public void ExistingId_ArmsInlineOverwriteBeforeConfirming()
        {
            string scratch = CreateLibraryCopy();
            try
            {
                var session = MlTrainingScenarioSession.Load(scratch);
                session.SelectEnvironment(MlEnvironmentContract.AdaptiveV1);
                session.SetSaveIdentity("Changed Standard", "adaptive-standard");

                Assert.That(session.SaveAsTemplate(), Is.False);
                Assert.That(session.OverwriteArmed, Is.True);
                Assert.That(
                    MlTrainingScenarioLibrary.Load(scratch)
                        .Filter(MlEnvironmentContract.AdaptiveV1)
                        .Single(item => item.Id == "adaptive-standard").Name,
                    Is.EqualTo("Standard"));

                Assert.That(session.ConfirmOverwrite(), Is.True);
                Assert.That(session.OverwriteArmed, Is.False);
                Assert.That(
                    MlTrainingScenarioLibrary.Load(scratch)
                        .Filter(MlEnvironmentContract.AdaptiveV1)
                        .Single(item => item.Id == "adaptive-standard").Name,
                    Is.EqualTo("Changed Standard"));
            }
            finally
            {
                if (File.Exists(scratch)) File.Delete(scratch);
            }
        }

        [Test]
        public void LibraryLoadFailure_PreservesExactPathAndExceptionInline()
        {
            string path = Path.Combine(
                Path.GetTempPath(), "missing-ml-template-library-" +
                Guid.NewGuid().ToString("N") + ".json");

            var session = MlTrainingScenarioSession.Load(path);

            Assert.That(session.CanLaunch, Is.False);
            Assert.That(session.LibraryError, Does.Contain(path));
            Assert.That(session.LibraryError, Does.Contain("invalid scenario library"));
        }

        [Test]
        public void AuthoritativeContractFailure_DisablesLaunch()
        {
            var session = new MlTrainingScenarioSession(
                MlTrainingScenarioLibrary.Load(BuiltInLibraryPath));
            session.SelectEnvironment(MlEnvironmentContract.AdaptiveV1);
            session.WorkingCopy.Board.Width = 50000;
            session.WorkingCopy.Board.Height = 50000;

            Assert.That(session.WorkingCopy.Validate(), Is.Empty);
            Assert.That(session.CanLaunch, Is.False);
        }

        [Test]
        public void AuthoritativeTacticalContractFailure_DisablesLaunch()
        {
            var session = new MlTrainingScenarioSession(
                MlTrainingScenarioLibrary.Load(BuiltInLibraryPath));
            session.WorkingCopy.Board.Width = 257;
            session.WorkingCopy.Board.Height = 256;

            Assert.That(session.WorkingCopy.Validate(), Is.Empty);
            Assert.That(session.CanLaunch, Is.False);
        }

        [Test]
        public void InvalidMlForm_DisablesLaunchWithInlineErrors()
        {
            var session = new MlTrainingScenarioSession(
                MlTrainingScenarioLibrary.Load(BuiltInLibraryPath));
            var config = MlLabConfig.Default();
            config.RunName = "bad name";
            config.TotalTimesteps = 0;
            config.CheckpointInterval = 0;
            config.Workers = 0;
            config.Device = " ";
            config.OpponentKind = MlOpponentKind.FixedRun;
            config.OpponentPath = string.Empty;

            MlTrainingLaunchFormState state =
                MlTrainingLaunchFormState.Evaluate(
                    config, session, resume: false);

            Assert.That(state.CanLaunch, Is.False);
            Assert.That(state.Errors, Has.Some.Contains("Run name"));
            Assert.That(state.Errors, Has.Some.Contains("Timesteps"));
            Assert.That(state.Errors, Has.Some.Contains("Checkpoint interval"));
            Assert.That(state.Errors, Has.Some.Contains("Workers"));
            Assert.That(state.Errors, Has.Some.Contains("Device"));
            Assert.That(state.Errors, Has.Some.Contains("Opponent path"));
        }

        [Test]
        public void ValidMlForm_IsLaunchableWithoutDependencyChecks()
        {
            var session = new MlTrainingScenarioSession(
                MlTrainingScenarioLibrary.Load(BuiltInLibraryPath));

            MlTrainingLaunchFormState state =
                MlTrainingLaunchFormState.Evaluate(
                    MlLabConfig.Default(), session, resume: false);

            Assert.That(state.CanLaunch, Is.True);
            Assert.That(state.Errors, Is.Empty);
        }

        [TestCase("tactical-v1", MlEnvironmentContract.TacticalV1)]
        [TestCase("adaptive-v1", MlEnvironmentContract.AdaptiveV1)]
        public void ArenaEnvironment_IsResolvedFromRunMetadata(string contract, MlEnvironmentContract expected)
        {
            string json = "{\"contract\":{\"version\":\"" + contract + "\"}}";

            Assert.That(HexWars.Presentation.EditorTools.ReplayViewerMenu.EnvironmentFromRunManifest(json),
                Is.EqualTo(expected));
        }

        static string CreateLibraryCopy()
        {
            string path = Path.Combine(
                Path.GetTempPath(), "hexwars-session-library-" +
                Guid.NewGuid().ToString("N") + ".json");
            File.Copy(BuiltInLibraryPath, path);
            return path;
        }
    }
}

using System;
using System.IO;
using System.Linq;
using HexWars.Presentation.EditorTools;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;
using UnityEngine;

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
        public void StartAndWatch_WaitsForCheckpointAndQueuesOnlyOnce()
        {
            var watch = new MlStartAndWatchState();
            watch.Begin(requested: true);

            Assert.That(watch.TryQueue(string.Empty), Is.False);
            Assert.That(watch.TryQueue("  "), Is.False);
            Assert.That(watch.TryQueue("checkpoints/step_100.zip"), Is.True);
            Assert.That(watch.TryQueue("checkpoints/step_200.zip"), Is.False,
                "a second status poll must not schedule another viewer while launch is pending");
            Assert.That(watch.LaunchPending, Is.True);
            Assert.That(watch.Launched, Is.False);
        }

        [Test]
        public void StartAndWatch_SuccessRecordsTheExactPresentationIdentity()
        {
            var watch = new MlStartAndWatchState();
            var ui = new MlLabWindowState();
            ui.MarkLaunched();
            watch.Begin(requested: true);
            Assert.That(watch.TryQueue("checkpoints/step_100.zip"), Is.True);

            watch.Apply(
                MlViewerLaunchResult.Succeeded(
                    "adaptive-large-battle", learnerSeat: 0, opponent: "Random",
                    seatSchedule: "alternating"),
                ui);

            Assert.That(watch.Launched, Is.True);
            Assert.That(watch.LaunchPending, Is.False);
            Assert.That(watch.PresentationStatus,
                Does.Contain("adaptive-large-battle")
                    .And.Contain("schedule alternating")
                    .And.Contain("learner P1 (seat 0)")
                    .And.Contain("Random"));
            Assert.That(watch.TryQueue("checkpoints/step_200.zip"), Is.False);
            Assert.That(ui.Phase, Is.EqualTo(MlLabUiPhase.Running));
            Assert.That(ui.Error, Is.Empty);
        }

        [Test]
        public void StartAndWatch_StrictPlanFailureIsInlineAndRemainsRetryable()
        {
            var watch = new MlStartAndWatchState();
            var ui = new MlLabWindowState();
            ui.MarkLaunched();
            watch.Begin(requested: true);
            Assert.That(watch.TryQueue("checkpoints/step_100.zip"), Is.True);

            watch.Apply(
                MlViewerLaunchResult.Failed(
                    "run.json: scenario.path does not exist"),
                ui);

            Assert.That(watch.Launched, Is.False);
            Assert.That(watch.LaunchPending, Is.False);
            Assert.That(ui.Phase, Is.EqualTo(MlLabUiPhase.Failed));
            Assert.That(ui.Error,
                Is.EqualTo("run.json: scenario.path does not exist"));
            Assert.That(watch.CanRetry, Is.True);
            Assert.That(watch.TryQueue("checkpoints/step_100.zip"), Is.False,
                "a failed checkpoint must not auto-spam launch every status poll");
            watch.Retry();
            Assert.That(watch.TryQueue("checkpoints/step_100.zip"), Is.True,
                "Retry viewer must allow a repaired run to reuse the same checkpoint");
            ui.Apply(MlRunState.Running, 99);

            watch.Apply(
                MlViewerLaunchResult.Succeeded(
                    "adaptive-large-battle", learnerSeat: 1, opponent: "Random",
                    seatSchedule: "alternating"),
                ui);

            Assert.That(watch.Launched, Is.True);
            Assert.That(ui.Phase, Is.EqualTo(MlLabUiPhase.Running));
            Assert.That(ui.Error, Is.Empty,
                "a successful retry must remove the stale strict-plan error");
        }

        [Test]
        public void StartAndWatch_PresentationStatusSurvivesDomainReloadSerialization()
        {
            var watch = new MlStartAndWatchState();
            var ui = new MlLabWindowState();
            watch.Begin(requested: true);
            Assert.That(watch.TryQueue("checkpoints/step_100.zip"), Is.True);
            watch.Apply(
                MlViewerLaunchResult.Succeeded(
                    "adaptive-standard", learnerSeat: 1, opponent: "Pool B",
                    seatSchedule: "alternating"),
                ui);

            string json = JsonUtility.ToJson(watch);
            var restored = JsonUtility.FromJson<MlStartAndWatchState>(json);

            Assert.That(restored.Launched, Is.True);
            Assert.That(restored.PresentationStatus,
                Does.Contain("adaptive-standard")
                    .And.Contain("schedule alternating")
                    .And.Contain("learner P2 (seat 1)")
                    .And.Contain("Pool B"));
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

        [Test]
        public void LiveBlankCustomTracker_DisablesLaunchWithoutMutatingConfig()
        {
            var session = new MlTrainingScenarioSession(
                MlTrainingScenarioLibrary.Load(BuiltInLibraryPath));
            var config = MlLabConfig.Default();
            MlTrackerSelectionSnapshot trackers =
                MlTrackerSelectionSnapshot.Capture(
                    useTensorBoard: true,
                    useWandb: false,
                    useCustomTracker: true,
                    customTrackerAdapter: " ");

            MlTrainingLaunchFormState state =
                MlTrainingLaunchFormState.Evaluate(
                    config, session, resume: false, trackers);

            Assert.That(state.CanLaunch, Is.False);
            Assert.That(
                state.Errors,
                Has.Some.Contains(
                    "Custom tracker requires a module:function adapter."));
            Assert.That(config.Trackers, Has.Count.EqualTo(1));
            Assert.That(config.Trackers.Single().Kind, Is.EqualTo("local"));
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

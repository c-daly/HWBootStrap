using System;
using System.IO;
using System.Linq;
using HexWars.Engine;
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
        public void WatchStartPolicy_CheckpointNotReadyBeforeCeiling_WaitsAndRetriesWhileTrainingAlive()
        {
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: true,
                    checkpointReady: false,
                    trainingAlive: true,
                    ceilingDeadlinePassed: false),
                Is.EqualTo(MlWatchStartDecision.WaitAndRetry));
        }

        [Test]
        public void WatchStartPolicy_CheckpointReady_WatchesExactlyOnce()
        {
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: true,
                    checkpointReady: true,
                    trainingAlive: true,
                    ceilingDeadlinePassed: false),
                Is.EqualTo(MlWatchStartDecision.Watch));
        }

        [Test]
        public void WatchStartPolicy_CheckpointReadyRightAtCeiling_StillWatchesInsteadOfGivingUp()
        {
            // The checkpoint landing is what matters; a ceiling check that only just tripped must not
            // pre-empt a checkpoint that is already validated and ready.
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: true,
                    checkpointReady: true,
                    trainingAlive: true,
                    ceilingDeadlinePassed: true),
                Is.EqualTo(MlWatchStartDecision.Watch));
        }

        [Test]
        public void WatchStartPolicy_CheckpointReadyEvenAfterTrainingEnded_StillWatches()
        {
            // Training can legitimately finish in the same tick its final checkpoint becomes
            // validated; the checkpoint being ready must win over "training is no longer alive".
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: true,
                    checkpointReady: true,
                    trainingAlive: false,
                    ceilingDeadlinePassed: false),
                Is.EqualTo(MlWatchStartDecision.Watch));
        }

        [Test]
        public void WatchStartPolicy_CeilingPassesWithoutCheckpoint_GivesUpWithNoInfiniteSpin()
        {
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: true,
                    checkpointReady: false,
                    trainingAlive: true,
                    ceilingDeadlinePassed: true),
                Is.EqualTo(MlWatchStartDecision.GiveUp));
        }

        [Test]
        public void WatchStartPolicy_TrainingDeadBeforeCheckpointReady_GivesUpWithoutWaitingForCeiling()
        {
            // Training already stopped/completed/failed without ever publishing a validated
            // checkpoint; there is nothing left to wait for, so this must not spin until the ceiling.
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: true,
                    checkpointReady: false,
                    trainingAlive: false,
                    ceilingDeadlinePassed: false),
                Is.EqualTo(MlWatchStartDecision.GiveUp));
        }

        [Test]
        public void WatchStartPolicy_RunDirectoryChangedMidRetry_DropsTheStaleRetry()
        {
            // A second run started (or the selection changed) while the first was still waiting for
            // its checkpoint; the stale retry must never launch a viewer for the old run directory,
            // even if that old run's checkpoint becomes ready or its own ceiling has passed.
            Assert.That(
                MlWatchStartPolicy.Decide(
                    pendingRunDirectoryMatchesSelection: false,
                    checkpointReady: true,
                    trainingAlive: true,
                    ceilingDeadlinePassed: true),
                Is.EqualTo(MlWatchStartDecision.Stale));
        }

        [Test]
        public void WatchResumePolicy_PersistedTargetMatchesSelectionAndLaunchPending_Resumes()
        {
            Assert.That(
                MlWatchResumePolicy.Decide(
                    hasPersistedPendingRunDirectory: true,
                    persistedPendingRunDirectoryMatchesSelection: true,
                    launchPending: true),
                Is.EqualTo(MlWatchResumeDecision.Resume));
        }

        [Test]
        public void WatchResumePolicy_PersistedTargetDoesNotMatchSelection_ResetsToRetryable()
        {
            // The window's selection moved to a different run across the reload (or the persisted
            // record is simply gone stale); the interrupted retry must not resume against the wrong
            // run, but the user still needs a way forward instead of a silently dead watch.
            Assert.That(
                MlWatchResumePolicy.Decide(
                    hasPersistedPendingRunDirectory: true,
                    persistedPendingRunDirectoryMatchesSelection: false,
                    launchPending: true),
                Is.EqualTo(MlWatchResumeDecision.ResetToRetryable));
        }

        [Test]
        public void WatchResumePolicy_LaunchPendingWithNoPersistedTarget_ResetsToRetryable()
        {
            // MlStartAndWatchState.LaunchPending is [SerializeField] and survives a reload on its own;
            // if it is stuck true with nothing persisted to resume, that is exactly the silent-wedge
            // bug -- recover by resetting it so Retry viewer appears.
            Assert.That(
                MlWatchResumePolicy.Decide(
                    hasPersistedPendingRunDirectory: false,
                    persistedPendingRunDirectoryMatchesSelection: false,
                    launchPending: true),
                Is.EqualTo(MlWatchResumeDecision.ResetToRetryable));
        }

        [Test]
        public void WatchResumePolicy_NothingPendingAndNotLaunching_NoPendingWatch()
        {
            Assert.That(
                MlWatchResumePolicy.Decide(
                    hasPersistedPendingRunDirectory: false,
                    persistedPendingRunDirectoryMatchesSelection: false,
                    launchPending: false),
                Is.EqualTo(MlWatchResumeDecision.NoPendingWatch));
        }

        [Test]
        public void WatchResumePolicy_PersistedTargetButLaunchAlreadyResolved_NoPendingWatch()
        {
            // A stray persisted key with LaunchPending already false (the watch already succeeded,
            // failed, or was never armed) must not force a spurious resume.
            Assert.That(
                MlWatchResumePolicy.Decide(
                    hasPersistedPendingRunDirectory: true,
                    persistedPendingRunDirectoryMatchesSelection: true,
                    launchPending: false),
                Is.EqualTo(MlWatchResumeDecision.NoPendingWatch));
        }

        [Test]
        public void StartAndWatch_ResetStuckLaunch_WhenPendingClearsFlagAndArmsRetry()
        {
            var watch = new MlStartAndWatchState();
            watch.Begin(requested: true);
            Assert.That(watch.TryQueue("checkpoints/step_100.zip"), Is.True);
            Assert.That(watch.LaunchPending, Is.True);

            watch.ResetStuckLaunch();

            Assert.That(watch.LaunchPending, Is.False);
            Assert.That(watch.CanRetry, Is.True);
            Assert.That(watch.Launched, Is.False);
        }

        [Test]
        public void StartAndWatch_ResetStuckLaunch_WhenNotPendingIsANoOp()
        {
            var watch = new MlStartAndWatchState();
            watch.Begin(requested: true);

            watch.ResetStuckLaunch();

            Assert.That(watch.LaunchPending, Is.False);
            Assert.That(watch.CanRetry, Is.False,
                "must not fabricate a retry affordance when nothing was ever pending");
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
        public void TacticalV2RosterRefresh_UsesSelectedPlayerAndPreservesCount()
        {
            SessionBarracksCache.ResetForTests();
            SessionBarracksCache.ForLocalPlayer(1).Add(Custom("Player Two Custom"));
            MlTrainingScenarioSession session =
                MlTrainingScenarioSession.Load(BuiltInLibraryPath);
            session.SelectEnvironment(MlEnvironmentContract.TacticalV2);
            session.WorkingCopy.TacticalV2.StartingUnitCount = 8;

            session.RefreshTacticalRoster(1);

            Assert.That(session.WorkingCopy.TacticalV2.StartingUnitCount, Is.EqualTo(8));
            Assert.That(session.WorkingCopy.TacticalV2.Templates.Select(item => item.Name),
                Does.Contain("Player Two Custom"));
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
            session.SelectEnvironment(MlEnvironmentContract.TacticalV1);

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
            session.SelectEnvironment(MlEnvironmentContract.TacticalV1);
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
        public void TrainEnvironmentChoices_ExcludeOfflineTacticalV3()
        {
            Assert.That(MlLabWindow.TrainEnvironmentChoices,
                Has.None.EqualTo(MlEnvironmentContract.TacticalV3));
            Assert.That(MlLabWindow.TrainEnvironmentChoices,
                Is.EqualTo(new[] { MlEnvironmentContract.TacticalV1,
                    MlEnvironmentContract.AdaptiveV1,
                    MlEnvironmentContract.TacticalV2 }));
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

        [Test]
        public void StructuredArenaRun_LoadsThenRejectsBadEvidenceWithoutMutatingConfig()
        {
            string run = Path.Combine(Path.GetTempPath(), "hexwars-v3-arena-" + Guid.NewGuid().ToString("N"));
            try
            {
                Directory.CreateDirectory(Path.Combine(run, "checkpoints"));
                File.Copy(Path.Combine("python", "config",
                    "annihilation-structured-imitation-v1.json"), Path.Combine(run, "scenario.json"));
                var scenario = MlTrainingScenarioFile.Load(Path.Combine(run, "scenario.json"));
                var engine = MlTrainingScenarioPreflight.ToEngine(scenario);
                var identity = HexWars.Engine.Rl.TacticalV3Contract.Create(
                    engine.BuildTacticalV3(), HexWars.Engine.Rl.MlEnvironmentKind.Duel);
                File.Copy(Path.Combine("python", "tests", "fixtures",
                    "tactical_v3", "seed-41-duel-spaces.json"),
                    Path.Combine(run, "policy-identity.json"));
                File.WriteAllText(Path.Combine(run, "checkpoints", "best.pt"), "checkpoint");
                string manifest = $@"{{""schema_version"":2,""evidence_status"":""unsealed-experimental"",""config"":{{""algorithm"":""structured_imitation""}},""contract"":{{""environment"":""tactical-v3"",""version"":""tactical-v3"",""environment_kind"":""duel"",""contract_hash"":""{identity.ContractHash}"",""encoding_hash"":""{identity.EncodingHash}"",""capacity_hash"":""{identity.CapacityHash}""}},""policy_identity"":""policy-identity.json"",""latest_checkpoint"":""checkpoints/best.pt""}}";
                File.WriteAllText(Path.Combine(run, "run.json"), manifest);
                var config = new ModelDuelConfiguration { Environment = MlEnvironmentContract.TacticalV3,
                    ScenarioRunPath = run, P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.FixedRun, Path = run } };
                Assert.That(MlArenaLaunchPlan.Create(config).P0Spec, Is.EqualTo("run:" + run));
                File.WriteAllText(Path.Combine(run, "run.json"), manifest.Replace("unsealed-experimental", "sealed"));
                Assert.Throws<InvalidOperationException>(() => MlArenaLaunchPlan.Create(config));
                Assert.That(config.P0.Path, Is.EqualTo(run));
                Assert.That(config.ScenarioRunPath, Is.EqualTo(run));
            }
            finally { if (Directory.Exists(run)) Directory.Delete(run, true); }
        }

        [Test]
        public void StructuredArenaRun_AllowsDifferentPolicyMatchContract()
        {
            string run = Path.Combine(Path.GetTempPath(),
                "hexwars-v3-split-" + Guid.NewGuid().ToString("N"));
            try
            {
                Directory.CreateDirectory(Path.Combine(run, "checkpoints"));
                File.Copy(Path.Combine("python", "config",
                    "annihilation-structured-imitation-v1.json"),
                    Path.Combine(run, "scenario.json"));
                string arenaScenarioPath = Path.Combine(run, "scenario.json");
                string quote = ((char)34).ToString();
                string arenaScenario = File.ReadAllText(arenaScenarioPath)
                    .Replace(quote + "width" + quote + ": 13",
                        quote + "width" + quote + ": 24")
                    .Replace(quote + "height" + quote + ": 9",
                        quote + "height" + quote + ": 16");
                File.WriteAllText(arenaScenarioPath, arenaScenario);
                File.Copy(Path.Combine("python", "tests", "fixtures",
                    "tactical_v3", "seed-41-duel-spaces.json"),
                    Path.Combine(run, "policy-identity.json"));
                File.WriteAllText(
                    Path.Combine(run, "checkpoints", "best.pt"),
                    "checkpoint");
                string manifest = @"{
                    ""schema_version"":2,
                    ""evidence_status"":""unsealed-experimental"",
                    ""config"":{""algorithm"":""structured_imitation""},
                    ""contract"":{
                        ""environment"":""tactical-v3"",
                        ""version"":""tactical-v3"",
                        ""environment_kind"":""duel"",
                        ""contract_hash"":""bac4af4d4b8e68466ffaf37c2721f98129edc93b90f529999ba45463cd921437"",
                        ""encoding_hash"":""e7a62d698a5f516c72ca3d1269ebd4b1afc61e7950c8ff0aeb2716f80e45f4b6"",
                        ""capacity_hash"":""7aea1db4f008dc192e83811b2c13abd8ce2304d2a6a209f37f9847be5f367364""
                    },
                    ""policy_identity"":""policy-identity.json"",
                    ""latest_checkpoint"":""checkpoints/best.pt""
                }";
                File.WriteAllText(Path.Combine(run, "run.json"), manifest);
                var config = new ModelDuelConfiguration {
                    Environment = MlEnvironmentContract.TacticalV3,
                    ScenarioRunPath = run,
                    P0 = new ModelSeatConfiguration {
                        Kind = ModelControllerKind.FixedRun, Path = run } };

                Assert.That(MlArenaLaunchPlan.Create(config).P0Spec,
                    Is.EqualTo("run:" + run));
            }
            finally { if (Directory.Exists(run)) Directory.Delete(run, true); }
        }

        static (string Run, string Manifest, ModelDuelConfiguration Config)
            CreateStructuredArenaRun()
        {
            string run = Path.Combine(Path.GetTempPath(),
                "hexwars-v3-arena-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path.Combine(run, "checkpoints"));
            File.Copy(Path.Combine("python", "config",
                "annihilation-structured-imitation-v1.json"),
                Path.Combine(run, "scenario.json"));
            var scenario = MlTrainingScenarioFile.Load(Path.Combine(run, "scenario.json"));
            var engine = MlTrainingScenarioPreflight.ToEngine(scenario);
            var id = HexWars.Engine.Rl.TacticalV3Contract.Create(
                engine.BuildTacticalV3(), HexWars.Engine.Rl.MlEnvironmentKind.Duel);
            File.Copy(Path.Combine("python", "tests", "fixtures",
                "tactical_v3", "seed-41-duel-spaces.json"),
                Path.Combine(run, "policy-identity.json"));
            File.WriteAllText(Path.Combine(run, "checkpoints", "best.pt"), "checkpoint");
            string manifest = $@"{{""schema_version"":2,""evidence_status"":""unsealed-experimental"",""config"":{{""algorithm"":""structured_imitation""}},""contract"":{{""environment"":""tactical-v3"",""version"":""tactical-v3"",""environment_kind"":""duel"",""contract_hash"":""{id.ContractHash}"",""encoding_hash"":""{id.EncodingHash}"",""capacity_hash"":""{id.CapacityHash}""}},""policy_identity"":""policy-identity.json"",""latest_checkpoint"":""checkpoints/best.pt""}}";
            File.WriteAllText(Path.Combine(run, "run.json"), manifest);
            var config = new ModelDuelConfiguration {
                Environment = MlEnvironmentContract.TacticalV3, ScenarioRunPath = run,
                P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.FixedRun, Path = run } };
            return (run, manifest, config);
        }

        [TestCase("algorithm")]
        [TestCase("evidence_status")]
        [TestCase("environment")]
        [TestCase("version")]
        [TestCase("contract_hash")]
        [TestCase("encoding_hash")]
        [TestCase("capacity_hash")]
        public void StructuredArenaRun_RejectsManifestIdentityBeforeMutation(string field)
        {
            var fixture = CreateStructuredArenaRun();
            try
            {
                string marker = "\"" + field + "\":\"";
                string changed = fixture.Manifest.Replace(marker, marker + "bad-");
                Assert.That(changed, Is.Not.EqualTo(fixture.Manifest));
                File.WriteAllText(Path.Combine(fixture.Run, "run.json"), changed);
                Assert.Throws<InvalidOperationException>(() =>
                    MlArenaLaunchPlan.Create(fixture.Config));
                Assert.That(fixture.Config.P0.Path, Is.EqualTo(fixture.Run));
                Assert.That(fixture.Config.ScenarioRunPath, Is.EqualTo(fixture.Run));
            }
            finally { Directory.Delete(fixture.Run, true); }
        }

        [TestCase("\\u006fbservation_size")]
        [TestCase("acti\\u006fn_size")]
        public void StructuredArenaRun_RejectsEscapedFixedGeometryMember(string escapedName)
        {
            var fixture = CreateStructuredArenaRun();
            try
            {
                string changed = fixture.Manifest.Replace("\"contract\":{",
                    "\"contract\":{\"" + escapedName + "\":0,");
                File.WriteAllText(Path.Combine(fixture.Run, "run.json"), changed);
                Assert.Throws<InvalidOperationException>(() => MlArenaLaunchPlan.Create(fixture.Config));
            }
            finally { Directory.Delete(fixture.Run, true); }
        }

        [Test]
        public void StructuredArenaRun_AllowsFixedGeometryTokenInsideStringValue()
        {
            var fixture = CreateStructuredArenaRun();
            try
            {
                string changed = fixture.Manifest.Replace("\"config\":{",
                    "\"note\":\"observation_size and action_size are variable\",\"config\":{");
                File.WriteAllText(Path.Combine(fixture.Run, "run.json"), changed);
                Assert.That(MlArenaLaunchPlan.Create(fixture.Config).P0Spec,
                    Is.EqualTo("run:" + fixture.Run));
            }
            finally { Directory.Delete(fixture.Run, true); }
        }

        [Test]
        public void StructuredArenaRun_RejectsMissingScenarioBeforeMutation()
        {
            var fixture = CreateStructuredArenaRun();
            try
            {
                File.Delete(Path.Combine(fixture.Run, "scenario.json"));
                Assert.Throws<InvalidOperationException>(() =>
                    MlArenaLaunchPlan.Create(fixture.Config));
                Assert.That(fixture.Config.P0.Path, Is.EqualTo(fixture.Run));
                Assert.That(fixture.Config.ScenarioRunPath, Is.EqualTo(fixture.Run));
            }
            finally { Directory.Delete(fixture.Run, true); }
        }

        [Test]
        public void StructuredArenaRun_RejectsMissingPolicyIdentityBeforeMutation()
        {
            var fixture = CreateStructuredArenaRun();
            try
            {
                File.Delete(Path.Combine(fixture.Run, "policy-identity.json"));
                Assert.Throws<InvalidOperationException>(() =>
                    MlArenaLaunchPlan.Create(fixture.Config));
                Assert.That(fixture.Config.P0.Path, Is.EqualTo(fixture.Run));
                Assert.That(fixture.Config.ScenarioRunPath, Is.EqualTo(fixture.Run));
            }
            finally { Directory.Delete(fixture.Run, true); }
        }

        [TestCase(false)]
        [TestCase(true)]
        public void StructuredArenaRun_RejectsMissingCheckpointOrTraversalBeforeMutation(
            bool traversal)
        {
            var fixture = CreateStructuredArenaRun();
            try
            {
                if (traversal)
                    fixture.Config.ScenarioRunPath = Path.Combine(
                        fixture.Run, "checkpoints", "..");
                else
                    File.Delete(Path.Combine(fixture.Run, "checkpoints", "best.pt"));
                string selected = fixture.Config.ScenarioRunPath;
                Assert.Throws<InvalidOperationException>(() =>
                    MlArenaLaunchPlan.Create(fixture.Config));
                Assert.That(fixture.Config.P0.Path, Is.EqualTo(fixture.Run));
                Assert.That(fixture.Config.ScenarioRunPath, Is.EqualTo(selected));
            }
            finally { Directory.Delete(fixture.Run, true); }
        }

        [TestCase("observation_size")]
        [TestCase("action_size")]
        [TestCase("latest_checkpoint")]
        public void StructuredArenaRun_RejectsFixedGeometryOrWrongCheckpointBeforeMutation(
            string field)
        {
            var fixture = CreateStructuredArenaRun();
            try
            {
                string changed = field == "latest_checkpoint"
                    ? fixture.Manifest.Replace("checkpoints/best.pt", "checkpoints/other.pt")
                    : fixture.Manifest.Replace("\"contract\":{",
                        "\"contract\":{\"" + field + "\":0,");
                File.WriteAllText(Path.Combine(fixture.Run, "run.json"), changed);
                Assert.Throws<InvalidOperationException>(() =>
                    MlArenaLaunchPlan.Create(fixture.Config));
                Assert.That(fixture.Config.P0.Path, Is.EqualTo(fixture.Run));
                Assert.That(fixture.Config.ScenarioRunPath, Is.EqualTo(fixture.Run));
            }
            finally { Directory.Delete(fixture.Run, true); }
        }

        static string CreateLibraryCopy()
        {
            string path = Path.Combine(
                Path.GetTempPath(), "hexwars-session-library-" +
                Guid.NewGuid().ToString("N") + ".json");
            File.Copy(BuiltInLibraryPath, path);
            return path;
        }

        static UnitTemplate Custom(string name) =>
            new UnitTemplate(name, new UnitStats(4, 3, 1, 3, 2, 2, 1, 4, 1));

        [TearDown]
        public void TearDown() => SessionBarracksCache.ResetForTests();
    }
}

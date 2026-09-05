using System;
using System.IO;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlLabConfigTests
    {
        [Test]
        public void ResolvePythonExecutable_UsesCommonRepositoryEnvironmentForLinkedWorktree()
        {
            string scratch = Path.Combine(Path.GetTempPath(), "hexwars-ml-path-" + Guid.NewGuid().ToString("N"));
            string repository = Path.Combine(scratch, "repo");
            string worktree = Path.Combine(scratch, "worktree");
            string gitDirectory = Path.Combine(repository, ".git", "worktrees", "feature");
            string expected = Path.Combine(repository, "python", "winenv", "Scripts", "python.exe");
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(expected));
                File.WriteAllText(expected, string.Empty);
                Directory.CreateDirectory(gitDirectory);
                Directory.CreateDirectory(worktree);
                File.WriteAllText(Path.Combine(worktree, ".git"), "gitdir: " + gitDirectory);
                File.WriteAllText(Path.Combine(gitDirectory, "commondir"), "../..");

                Assert.That(MlLabPaths.ResolvePythonExecutable(worktree), Is.EqualTo(expected));
            }
            finally
            {
                if (Directory.Exists(scratch)) Directory.Delete(scratch, recursive: true);
            }
        }

        [Test]
        public void ResolvePythonExecutable_PrefersEnvironmentInsideCurrentProject()
        {
            string scratch = Path.Combine(Path.GetTempPath(), "hexwars-ml-local-" + Guid.NewGuid().ToString("N"));
            string expected = Path.Combine(scratch, "python", "winenv", "Scripts", "python.exe");
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(expected));
                File.WriteAllText(expected, string.Empty);

                Assert.That(MlLabPaths.ResolvePythonExecutable(scratch), Is.EqualTo(expected));
            }
            finally
            {
                if (Directory.Exists(scratch)) Directory.Delete(scratch, recursive: true);
            }
        }

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
        public void BuildTrainArguments_UsesResolvedScenarioFile()
        {
            var config = MlLabConfig.Default();

            string args = config.BuildTrainArguments(
                @"C:\project\Library\HexWars\MLLab\scenario.json");

            Assert.That(args, Does.Contain(
                "--scenario-file \"C:\\project\\Library\\HexWars\\MLLab\\scenario.json\""));
        }

        [Test]
        public void BuildTrainArguments_RejectsMissingResolvedScenarioPath()
        {
            var config = MlLabConfig.Default();

            Assert.Throws<ArgumentException>(() => config.BuildTrainArguments(" "));
        }

        [Test]
        public void BuildTrainArguments_AlwaysQuotesTrailingBackslashScenarioPath()
        {
            var config = MlLabConfig.Default();

            string args = config.BuildTrainArguments(@"C:\scenarios\");

            Assert.That(args, Does.Contain(
                "--scenario-file \"C:\\scenarios\\\\\""));
        }

        [Test]
        public void BuildTrainArguments_EscapesEmbeddedQuoteInScenarioPath()
        {
            var config = MlLabConfig.Default();

            string args = config.BuildTrainArguments("C:\\scenarios\\a\"b.json");

            Assert.That(args, Does.Contain(
                "--scenario-file \"C:\\scenarios\\a\\\"b.json\""));
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
        public void Defaults_SelectTacticalV2AndEmitItExplicitly()
        {
            var config = MlLabConfig.Default();

            Assert.That(config.Environment, Is.EqualTo(MlEnvironmentContract.TacticalV2));
            Assert.That(config.BuildTrainArguments(), Does.Contain("--environment tactical-v2"));
        }

        [Test]
        public void NewConfig_DefaultsToTacticalV2()
        {
            Assert.That(new MlLabConfig().Environment,
                Is.EqualTo(MlEnvironmentContract.TacticalV2));
            Assert.That(MlEnvironmentContracts.CliValue(MlEnvironmentContract.TacticalV2),
                Is.EqualTo("tactical-v2"));
        }

        [Test]
        public void OpponentChoices_IncludePassiveWithoutRemappingExistingSelections()
        {
            string[] choices = System.Enum.GetNames(typeof(MlOpponentKind));

            Assert.That(choices, Does.Contain(nameof(MlOpponentKind.Passive)));
            Assert.That(choices, Does.Not.Contain("FixedCheckpoint"));
            Assert.That((int)MlOpponentKind.Greedy, Is.EqualTo(0));
            Assert.That((int)MlOpponentKind.Random, Is.EqualTo(1));
            Assert.That((int)MlOpponentKind.FixedRun, Is.EqualTo(2));
            Assert.That((int)MlOpponentKind.LiveRun, Is.EqualTo(3));
        }

        [Test]
        public void BuildTrainArguments_EmitsPassiveOpponentWithoutModelPath()
        {
            var config = MlLabConfig.Default();
            config.OpponentKind = MlOpponentKind.Passive;
            config.OpponentPath = string.Empty;

            Assert.That(config.Validate(), Has.None.Contains("Opponent path"));
            Assert.That(config.BuildTrainArguments(),
                Does.Contain("--opponent passive"));
        }

        [Test]
        public void BuildTrainArguments_EmitsAdaptiveEnvironmentWhenSelected()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.AdaptiveV1;

            Assert.That(config.BuildTrainArguments(), Does.Contain("--environment adaptive-v1"));
        }

        [TestCase(MlEnvironmentContract.TacticalV1, "tactical-v1")]
        [TestCase(MlEnvironmentContract.AdaptiveV1, "adaptive-v1")]
        [TestCase(MlEnvironmentContract.TacticalV2, "tactical-v2")]
        public void BuildDoctorArguments_EmitsSelectedEnvironment(
            MlEnvironmentContract environment, string cliValue)
        {
            var config = MlLabConfig.Default();
            config.Environment = environment;

            string args = config.BuildDoctorArguments(@"C:\runs root", @"C:\server.dll");

            Assert.That(args, Does.StartWith("doctor --environment " + cliValue + " "));
            Assert.That(args, Does.Contain("--runs-root \"C:\\runs root\""));
            Assert.That(args, Does.Contain("--server C:\\server.dll"));
            Assert.That(args, Does.EndWith("--json"));
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

        [Test]
        public void TacticalV3Validation_RequiresSourceAndLabelsButIgnoresSb3OnlyFields()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.ResumeSource = string.Empty;
            config.TotalTimesteps = 0;
            config.CheckpointInterval = 0;
            config.Workers = 0;

            var errors = config.Validate();

            Assert.That(errors, Has.Some.Contains("DAgger train label target"));
            Assert.That(errors, Has.Some.Contains("source model"));
            Assert.That(errors, Has.None.Contains("Checkpoint interval"));
            Assert.That(errors, Has.None.Contains("Workers"));
            Assert.That(errors, Has.None.Contains("Timesteps"));

            config.TotalTimesteps = 1;
            config.Seed = 20001;
            errors = config.Validate();
            Assert.That(errors, Has.Some.Contains("at least two"));
            Assert.That(errors, Has.Some.Contains("0 through 20000"));

            errors = config.Validate(new[] { new MlTrackerConfig("wandb") });
            Assert.That(errors,
                Has.Some.Contains("local and TensorBoard trackers only"));
        }

        [Test]
        public void TacticalV3TrainingMode_DefaultsToSerializedDaggerValue()
        {
            var config = MlLabConfig.Default();

            Assert.That(config.TacticalV3TrainingMode,
                Is.EqualTo(MlTacticalV3TrainingMode.DaggerContinuation));
            Assert.That((int)MlTacticalV3TrainingMode.DaggerContinuation,
                Is.EqualTo(0));
            Assert.That((int)MlTacticalV3TrainingMode.OutcomeCandidate,
                Is.EqualTo(1));
        }

        [Test]
        public void OutcomeValidation_UsesDecisionTargetAndAllowsNoInitialModel()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.TacticalV3TrainingMode =
                MlTacticalV3TrainingMode.OutcomeCandidate;
            config.ResumeSource = string.Empty;
            config.TotalTimesteps = 0;
            config.CheckpointInterval = 0;
            config.Workers = 0;

            var errors = config.Validate();

            Assert.That(errors,
                Has.Some.Contains("Outcome learner decision target"));
            Assert.That(errors, Has.None.Contains("source model"));
            Assert.That(errors, Has.None.Contains("Checkpoint interval"));
            Assert.That(errors, Has.None.Contains("Workers"));

            config.TotalTimesteps = 2;
            Assert.That(config.Validate(), Is.Empty);

            config.ResumeSource = " ";
            Assert.That(config.Validate(),
                Has.Some.Contains("Initial model cannot be blank"));
        }

        [Test]
        public void BuildStructuredTrainArguments_EmitsInitializedContinuationContract()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.RunName = "dagger-continuation";
            config.ResumeSource = @"C:\runs\source run";
            config.OpponentKind = MlOpponentKind.Random;
            config.TotalTimesteps = 11;
            config.Seed = 17;
            config.Device = "cuda";
            config.LearnerSeat = MlLearnerSeat.Seat1;
            config.Trackers.Add(new MlTrackerConfig("tensorboard"));
            const string scenarioPath = @"C:\scenarios\new scenario.json";

            string args = config.BuildStructuredTrainArguments(scenarioPath);

            Assert.That(args, Does.StartWith(
                "train-structured --run dagger-continuation "));
            Assert.That(args, Does.Contain(
                "--source-run \"C:\\runs\\source run\""));
            Assert.That(args, Does.Contain(
                "--scenario-file \"C:\\scenarios\\new scenario.json\""));
            Assert.That(args, Does.Contain("--opponent random"));
            Assert.That(args, Does.Contain("--train-labels 11"));
            Assert.That(args, Does.Contain("--validation-labels 5"));
            Assert.That(args, Does.Contain("--seed 17"));
            Assert.That(args, Does.Contain("--device cuda"));
            Assert.That(args, Does.Contain("--learner-seat 1"));
            Assert.That(args, Does.Contain("--tracker local"));
            Assert.That(args, Does.Contain("--tracker tensorboard"));
            Assert.That(args, Does.EndWith("--no-console-output --json"));
            Assert.That(args, Does.Not.Contain("--environment"));
            Assert.That(args, Does.Not.Contain("--algorithm"));
            Assert.That(args, Does.Not.Contain("--timesteps"));
            Assert.That(args, Does.Not.Contain("--checkpoint-every"));
            Assert.That(args, Does.Not.Contain("--workers"));
        }

        [Test]
        public void BuildStructuredRetryArguments_UsesOnlySelectedCollectionAndFreshRun()
        {
            string args = MlLabConfig.BuildStructuredRetryArguments(
                @"C:\runs\stopped collection", "replayed-training");

            Assert.That(args, Does.StartWith(
                "retry-structured --collection-run \"C:\\runs\\stopped collection\" " +
                "--run replayed-training"));
            Assert.That(args, Does.EndWith("--no-console-output --json"));
            Assert.That(args, Does.Not.Contain("--source-run"));
            Assert.That(args, Does.Not.Contain("--scenario-file"));
            Assert.That(args, Does.Not.Contain("--opponent"));
            Assert.That(args, Does.Not.Contain("--device"));
            Assert.That(args, Does.Not.Contain("--tracker"));
        }

        [TestCase(null)]
        [TestCase("")]
        [TestCase(" ")]
        public void BuildStructuredRetryArguments_RequiresSelectedCollectionRun(
            string collectionRun)
        {
            ArgumentException error = Assert.Throws<ArgumentException>(() =>
                MlLabConfig.BuildStructuredRetryArguments(
                    collectionRun, "replayed-training"));

            Assert.That(error.ParamName, Is.EqualTo("collectionRun"));
        }

        [TestCase(null)]
        [TestCase("")]
        [TestCase("bad name")]
        [TestCase("../existing")]
        public void BuildStructuredRetryArguments_RequiresFreshSafeRunName(
            string runName)
        {
            ArgumentException error = Assert.Throws<ArgumentException>(() =>
                MlLabConfig.BuildStructuredRetryArguments(
                    @"C:\runs\stopped", runName));

            Assert.That(error.ParamName, Is.EqualTo("newRunName"));
            Assert.That(error.Message, Does.Contain("Fresh run name"));
        }

        [TestCase(MlOpponentKind.Greedy, "greedy")]
        [TestCase(MlOpponentKind.Random, "random")]
        [TestCase(MlOpponentKind.Passive, "passive")]
        [TestCase(MlOpponentKind.FixedRun, "run:C:\\runs\\opponent")]
        public void BuildStructuredTrainArguments_PreservesScriptedAndFixedOpponents(
            MlOpponentKind kind, string expected)
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.ResumeSource = @"C:\runs\source";
            config.OpponentKind = kind;
            config.OpponentPath = @"C:\runs\opponent";

            string args = config.BuildStructuredTrainArguments(
                @"C:\scenarios\target.json");

            Assert.That(args, Does.Contain("--opponent " + expected));
        }

        [Test]
        public void BuildStructuredTrainArguments_PreservesLiveRunOpponent()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.ResumeSource = @"C:\runs\source";
            config.OpponentKind = MlOpponentKind.LiveRun;
            config.OpponentPath = @"C:\runs\live opponent";

            string args = config.BuildStructuredTrainArguments(
                @"C:\scenarios\target.json");

            Assert.That(args, Does.Contain("\\\"kind\\\":\\\"run\\\""));
            Assert.That(args, Does.Contain("\\\"mode\\\":\\\"live\\\""));
            Assert.That(args, Does.Not.Contain("--opponent \\\"run:"));
        }

        [Test]
        public void BuildStructuredTrainArguments_RequiresSourceRun()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.ResumeSource = " ";

            Assert.Throws<InvalidOperationException>(
                () => config.BuildStructuredTrainArguments(
                    @"C:\scenarios\target.json"));
        }

        [TestCase(null)]
        [TestCase("")]
        [TestCase(" ")]
        public void BuildStructuredTrainArguments_RequiresResolvedScenarioPath(
            string scenarioPath)
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.ResumeSource = @"C:\runs\source";

            ArgumentException error = Assert.Throws<ArgumentException>(
                () => config.BuildStructuredTrainArguments(scenarioPath));

            Assert.That(error.ParamName, Is.EqualTo("scenarioPath"));
        }

        [Test]
        public void BuildStructuredTrainArguments_EscapesEmbeddedQuoteInScenarioPath()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.ResumeSource = @"C:\runs\source";

            string args = config.BuildStructuredTrainArguments(
                "C:\\scenarios\\a\"b.json");

            Assert.That(args, Does.Contain(
                "--scenario-file \"C:\\scenarios\\a\\\"b.json\""));
        }

        [Test]
        public void BuildStructuredTrainArguments_EscapesTrailingBackslashInScenarioPath()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.ResumeSource = @"C:\runs\source";

            string args = config.BuildStructuredTrainArguments(@"C:\scenarios\");

            Assert.That(args, Does.Contain(
                "--scenario-file \"C:\\scenarios\\\\\""));
        }

        [Test]
        public void BuildStructuredPreflightArguments_AuthenticatesSameInputsBeforeLaunch()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.ResumeSource = @"C:\runs\source model";
            config.OpponentKind = MlOpponentKind.LiveRun;
            config.OpponentPath = @"C:\runs\live opponent";
            config.Seed = 41;
            config.Device = "cuda:0";

            string args = config.BuildStructuredPreflightArguments(
                @"C:\scenarios\target scenario.json");

            Assert.That(args, Does.StartWith("preflight-structured "));
            Assert.That(args, Does.Contain(
                "--source-run \"C:\\runs\\source model\""));
            Assert.That(args, Does.Contain(
                "--scenario-file \"C:\\scenarios\\target scenario.json\""));
            Assert.That(args, Does.Contain("\\\"mode\\\":\\\"live\\\""));
            Assert.That(args, Does.Contain("--seed 41"));
            Assert.That(args, Does.Contain("--device cuda:0"));
            Assert.That(args, Does.EndWith("--json"));
            Assert.That(args, Does.Not.Contain("--run "));
            Assert.That(args, Does.Not.Contain("--train-labels"));
        }

        [Test]
        public void BuildOutcomeTrainArguments_EmitsExactFreshCandidateContract()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.TacticalV3TrainingMode =
                MlTacticalV3TrainingMode.OutcomeCandidate;
            config.RunName = "outcome-candidate";
            config.ResumeSource = string.Empty;
            config.OpponentKind = MlOpponentKind.Passive;
            config.TotalTimesteps = 1234;
            config.Seed = 29;
            config.Device = "cuda";
            config.LearnerSeat = MlLearnerSeat.Alternating;
            config.Trackers.Add(new MlTrackerConfig("tensorboard"));

            string args = config.BuildOutcomeTrainArguments(
                @"C:\scenarios\close static.json");

            Assert.That(args, Is.EqualTo(
                "train-outcome --run outcome-candidate " +
                "--scenario-file \"C:\\scenarios\\close static.json\" " +
                "--opponent passive --timesteps 1234 --seed 29 " +
                "--device cuda --learner-seat alternating --tracker local " +
                "--tracker tensorboard --no-console-output --json"));
        }

        [Test]
        public void BuildOutcomeTrainArguments_EmitsOptionalInitialModel()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.TacticalV3TrainingMode =
                MlTacticalV3TrainingMode.OutcomeCandidate;
            config.ResumeSource = @"C:\runs\initial model";

            string args = config.BuildOutcomeTrainArguments(
                @"C:\scenarios\target.json");

            Assert.That(args, Does.Contain(
                "--source-run \"C:\\runs\\initial model\""));
        }

        [Test]
        public void BuildOutcomePreflightArguments_EmitsExactOptionalSourceInputs()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.TacticalV3TrainingMode =
                MlTacticalV3TrainingMode.OutcomeCandidate;
            config.ResumeSource = @"C:\runs\initial model";
            config.OpponentKind = MlOpponentKind.Random;
            config.Seed = 41;
            config.Device = "cuda:0";
            config.LearnerSeat = MlLearnerSeat.Seat1;

            string args = config.BuildOutcomePreflightArguments(
                @"C:\scenarios\target scenario.json");

            Assert.That(args, Is.EqualTo(
                "preflight-outcome " +
                "--scenario-file \"C:\\scenarios\\target scenario.json\" " +
                "--opponent random --seed 41 --device cuda:0 " +
                "--learner-seat 1 " +
                "--source-run \"C:\\runs\\initial model\" --json"));
        }

        [TestCase(null)]
        [TestCase("")]
        [TestCase(" ")]
        public void BuildOutcomeArguments_RequireResolvedScenarioPath(
            string scenarioPath)
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;
            config.TacticalV3TrainingMode =
                MlTacticalV3TrainingMode.OutcomeCandidate;

            Assert.Throws<ArgumentException>(
                () => config.BuildOutcomeTrainArguments(scenarioPath));
            Assert.Throws<ArgumentException>(
                () => config.BuildOutcomePreflightArguments(scenarioPath));
        }

        [Test]
        public void TacticalV3GenericTrainBuilder_DirectsCallerToStructuredCommand()
        {
            var config = MlLabConfig.Default();
            config.Environment = MlEnvironmentContract.TacticalV3;

            InvalidOperationException error =
                Assert.Throws<InvalidOperationException>(
                    () => config.BuildTrainArguments(@"C:\scenario.json"));

            Assert.That(error.Message,
                Does.Contain("BuildStructuredTrainArguments"));
        }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class AdaptiveDuelEnvTests
    {
        [Test]
        public void TwoExternalPolicies_DeployFightDesignAndReconstructReplay()
        {
            var env = new AdaptiveDuelEnv();
            var view = env.Reset(33, null, null, null, null, PlayerId.Player0);
            view = DeployExternalSeat(env, view, useCustomTemplate: false);
            view = DeployExternalSeat(env, view, useCustomTemplate: true);

            Assert.That(env.DeploymentComplete, Is.True);
            Assert.That(env.Diagnostics.DistinctCustomTemplatesDeployed, Is.EqualTo(1));
            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.RedesignCustom], Is.True);
            view = env.Step((int)AdaptiveCommandChoice.RedesignCustom);
            view = env.Step(env.Layout.TemplateOffset + 6);
            view = env.Step(env.Layout.StatOffset + (int)AdaptiveStat.Health);
            view = env.Step(env.Layout.ValueOffset + 3); // health 4, the default value
            view = env.Step((int)AdaptiveCommandChoice.ConfirmDesign);
            Assert.That(env.Diagnostics.DesignCount, Is.EqualTo(1));

            int guard = 0;
            while (!view.Terminated && !view.Truncated && guard++ < 2000)
            {
                Assert.That(view.ActionMask.Length, Is.EqualTo(182));
                Assert.That(view.ActionMask.Any(value => value), Is.True);
                view = env.Step(ChooseProgressAction(view.ActionMask));
            }

            Assert.That(view.Terminated || view.Truncated, Is.True);
            var data = ReplayFile.Read(env.ToReplay());
            Assert.That(data.Commands.OfType<ReplaceTemplate>().Count(), Is.EqualTo(1));
            var replay = new Replay(data.Start, data.Commands);
            Assert.That(replay.Final.IsGameOver, Is.EqualTo(env.State.IsGameOver));
            Assert.That(replay.Final.Winner, Is.EqualTo(env.State.Winner));
            Assert.That(replay.Final.Player(PlayerId.Player0).Barracks[6].Stats,
                Is.EqualTo(env.State.Player(PlayerId.Player0).Barracks[6].Stats));
        }

        /// <summary>Capture smoke test (see TacticalV2DuelEnvTests for the thorough coverage): a fully
        /// scripted deployment + gameplay episode drains transitions that are exactly the accepted
        /// commands in order, matching the replay log, and chain by reference.</summary>
        [Test]
        public void DrainTransitions_MatchesReplayLog_ForFullyScriptedGame()
        {
            var env = new AdaptiveDuelEnv();
            env.CaptureTransitions = true;
            var view = env.Reset(71, new GreedyAgent(1), new GreedyAgent(2),
                new CombinedArmsDeploymentPolicy(1), new CombinedArmsDeploymentPolicy(2));

            if (env.AwaitingPostRevealAdvance) view = env.ContinueAfterReveal();

            Assert.That(env.DeploymentComplete, Is.True);

            var transitions = env.DrainTransitions();
            Assert.That(transitions, Is.Not.Empty);
            string replayText = env.ToReplay();
            var data = ReplayFile.Read(replayText);
            Assert.That(transitions.Count, Is.EqualTo(data.Commands.Count));
            for (int i = 0; i < transitions.Count; i++)
                Assert.That(transitions[i].Command, Is.EqualTo(data.Commands[i]));
            for (int i = 0; i < transitions.Count - 1; i++)
                Assert.That(transitions[i].Resulting, Is.SameAs(transitions[i + 1].Previous));
            Assert.That(transitions[transitions.Count - 1].Resulting, Is.SameAs(env.State));
            // transitions[0].Previous must be the episode's actual start state (the playback anchor):
            // GameState has no value equality, so re-serializing it with the replay's own commands and
            // checking it reproduces the replay text byte-for-byte is the cheapest structural proof.
            Assert.That(ReplayFile.Write(transitions[0].Previous, data.Commands), Is.EqualTo(replayText));

            Assert.That(env.DrainTransitions(), Is.Empty, "drain must empty the queue");
        }

        [Test]
        public void ScriptedAndExternalSeats_UseOnePrivateDeploymentCallAndExposeOnlyExternalSeat()
        {
            var policy = new RecordingDeploymentPolicy();
            var env = new AdaptiveDuelEnv();

            var view = env.Reset(44, new GreedyAgent(1), null, policy, null, PlayerId.Player1);

            Assert.That(policy.Calls, Is.EqualTo(1));
            Assert.That(policy.OpponentFactsObserved, Is.Zero);
            Assert.That(view.Seat, Is.EqualTo(1));
            Assert.That(env.DeploymentComplete, Is.False);

            view = DeployExternalSeat(env, view, useCustomTemplate: false);
            Assert.That(env.DeploymentComplete, Is.True);
            Assert.That(view.Seat, Is.EqualTo(0), "the atomic reveal must be observable before scripted gameplay");
            view = env.ContinueAfterReveal();
            Assert.That(view.Seat, Is.EqualTo(1), "continuation must guard past the internal gameplay seat");
        }

        [Test]
        public void Reveal_PausesBeforeScriptedFirstPlayerGameplayUntilExplicitContinuation()
        {
            var firstPlayer = new RecordingEndTurnAgent();
            var env = new AdaptiveDuelEnv();
            var view = env.Reset(45, firstPlayer, null,
                new CombinedArmsDeploymentPolicy(1), null, PlayerId.Player1);

            view = DeployExternalSeat(env, view, useCustomTemplate: false);

            Assert.That(env.DeploymentComplete, Is.True);
            Assert.That(env.AwaitingPostRevealAdvance, Is.True);
            Assert.That(firstPlayer.Calls, Is.Zero);
            Assert.That(env.State.ActivePlayer, Is.EqualTo(PlayerId.Player0));
            Assert.That(view.Seat, Is.Zero);

            view = env.ContinueAfterReveal();

            Assert.That(firstPlayer.Calls, Is.EqualTo(1));
            Assert.That(env.AwaitingPostRevealAdvance, Is.False);
            Assert.That(env.State.ActivePlayer, Is.EqualTo(PlayerId.Player1));
            Assert.That(view.Seat, Is.EqualTo(1));
            Assert.That(() => env.ContinueAfterReveal(), Throws.InvalidOperationException);
        }

        [Test]
        public void InvalidScriptedDeployment_FailsResetAtomically()
        {
            var env = new AdaptiveDuelEnv();
            var policy = new InvalidTrailingDeploymentPolicy();

            Assert.That(() => env.Reset(46, new GreedyAgent(1), null,
                policy, null, PlayerId.Player1), Throws.InvalidOperationException
                .With.Message.Contains("scripted deployment policy"));
            Assert.That(policy.Calls, Is.EqualTo(1));
        }

        [Test]
        public void GameplayIntermediateDecision_DoesNotAdvanceEngineTurn()
        {
            var env = new AdaptiveDuelEnv();
            var view = env.Reset(57, null, null,
                new CombinedArmsDeploymentPolicy(1), new CombinedArmsDeploymentPolicy(2));
            PlayerId active = env.State.ActivePlayer;

            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.ChooseUnit], Is.True);
            view = env.Step((int)AdaptiveCommandChoice.ChooseUnit);

            Assert.That(env.State.ActivePlayer, Is.EqualTo(active));
            Assert.That(view.Seat, Is.EqualTo((int)active));
            Assert.That(view.Reward, Is.EqualTo(-env.Config.IntermediateDecisionPenalty));
        }

        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void ExternalOpponentDeploymentMenu_DoesNotChargeLearnerShaping(PlayerId learner)
        {
            var env = new AdaptiveDuelEnv();
            IDeploymentPolicy? deployment0 = learner == PlayerId.Player0
                ? new CombinedArmsDeploymentPolicy(1)
                : null;
            IDeploymentPolicy? deployment1 = learner == PlayerId.Player1
                ? new CombinedArmsDeploymentPolicy(2)
                : null;
            var view = env.Reset(58, null, null, deployment0, deployment1, learner);

            Assert.That(view.Seat, Is.Not.EqualTo((int)learner));
            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.DeployStartingUnit], Is.True);

            view = env.Step((int)AdaptiveCommandChoice.DeployStartingUnit);

            Assert.That(view.Reward, Is.Zero,
                "an externally driven/model-like opponent menu decision is not learner shaping");
        }

        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void ExternalLearnerDeploymentMenu_ChargesLearnerShaping(PlayerId learner)
        {
            var env = new AdaptiveDuelEnv();
            IDeploymentPolicy? deployment0 = learner == PlayerId.Player0
                ? null
                : new CombinedArmsDeploymentPolicy(1);
            IDeploymentPolicy? deployment1 = learner == PlayerId.Player1
                ? null
                : new CombinedArmsDeploymentPolicy(2);
            var view = env.Reset(59, null, null, deployment0, deployment1, learner);

            Assert.That(view.Seat, Is.EqualTo((int)learner));
            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.DeployStartingUnit], Is.True);

            view = env.Step((int)AdaptiveCommandChoice.DeployStartingUnit);

            Assert.That(view.Reward, Is.EqualTo(-env.Config.IntermediateDecisionPenalty));
        }

        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void ExternalOpponentGameplayMenu_DoesNotChargeLearnerShaping(PlayerId learner)
        {
            var env = new AdaptiveDuelEnv();
            var view = env.Reset(60, null, null,
                new CombinedArmsDeploymentPolicy(1),
                new CombinedArmsDeploymentPolicy(2), learner);
            if (view.Seat == (int)learner)
            {
                Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.EndTurn], Is.True);
                view = env.Step((int)AdaptiveCommandChoice.EndTurn);
            }

            Assert.That(view.Seat, Is.Not.EqualTo((int)learner));
            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.ChooseUnit], Is.True);

            view = env.Step((int)AdaptiveCommandChoice.ChooseUnit);

            Assert.That(view.Reward, Is.Zero,
                "opponent gameplay hierarchy depth must not affect learner return");
        }

        [TestCase(PlayerId.Player0, 1f)]
        [TestCase(PlayerId.Player1, -1f)]
        public void TerminalReward_UsesLearnerPerspectiveWhenSeatZeroActionTerminates(
            PlayerId learner, float expectedReward)
        {
            var config = AdaptiveEnvConfig.Default();
            config.Game = GameConfig.Default(
                biomesEnabled: false,
                fogOfWar: true,
                winConditions: WinBy.Economy,
                economyWinThreshold: 0,
                maxDesignPointCost: config.MaxDesignPointCost,
                fixedTemplateCount: config.FixedTemplateCount,
                templateSlotCount: config.Templates.Count);
            var env = new AdaptiveDuelEnv(config);
            var view = env.Reset(61, null, null,
                new CombinedArmsDeploymentPolicy(1),
                new CombinedArmsDeploymentPolicy(2), learner);

            Assert.That(view.Seat, Is.Zero);
            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.EndTurn], Is.True);

            view = env.Step((int)AdaptiveCommandChoice.EndTurn);

            Assert.That(view.Terminated, Is.True);
            Assert.That(view.Winner, Is.Zero);
            Assert.That(view.Reward, Is.EqualTo(expectedReward));
        }

        [Test]
        public void InvalidGameplayAction_ReturnsLegalRootMaskWithoutApplyingFallback()
        {
            var env = new AdaptiveDuelEnv();
            var view = env.Reset(61, null, null,
                new CombinedArmsDeploymentPolicy(1), new CombinedArmsDeploymentPolicy(2));
            PlayerId active = env.State.ActivePlayer;
            int round = env.State.Round;

            view = env.Step(-1);

            Assert.That(env.Diagnostics.InvalidSequences, Is.EqualTo(1));
            Assert.That(env.State.ActivePlayer, Is.EqualTo(active));
            Assert.That(env.State.Round, Is.EqualTo(round));
            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.EndTurn], Is.True);
            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.Cancel], Is.False);
        }

        [Test]
        public void GymServer_DefaultAndAdaptiveProcessesKeepTheirContractsAndMasks()
        {
            using var legacy = new ServerProcess();
            using JsonDocument legacySpaces = legacy.Exchange(new { cmd = "spaces" });
            Assert.That(legacySpaces.RootElement.GetProperty("contract_version").GetString(), Is.EqualTo("tactical-v1"));
            Assert.That(legacySpaces.RootElement.TryGetProperty("adaptive", out _), Is.False);
            using var explicitLegacy = new ServerProcess("--environment", "tactical-v1");
            using JsonDocument explicitSpaces = explicitLegacy.Exchange(new { cmd = "spaces" });
            Assert.That(explicitSpaces.RootElement.GetRawText(), Is.EqualTo(legacySpaces.RootElement.GetRawText()));
            using JsonDocument legacyReset = legacy.Exchange(new { cmd = "reset", seed = 76 });
            using JsonDocument explicitReset = explicitLegacy.Exchange(new { cmd = "reset", seed = 76 });
            Assert.That(explicitReset.RootElement.GetRawText(), Is.EqualTo(legacyReset.RootElement.GetRawText()));
            bool[] legacyMask = legacyReset.RootElement.GetProperty("mask").EnumerateArray()
                .Select(x => x.GetBoolean()).ToArray();
            int legacyAction = Enumerable.Range(0, legacyMask.Length).First(i => legacyMask[i]);
            using JsonDocument legacyStep = legacy.Exchange(new { cmd = "step", action = legacyAction });
            using JsonDocument explicitStep = explicitLegacy.Exchange(new { cmd = "step", action = legacyAction });
            Assert.That(explicitStep.RootElement.GetRawText(), Is.EqualTo(legacyStep.RootElement.GetRawText()));

            using var adaptive = new ServerProcess("--environment", "adaptive-v1");
            using JsonDocument spaces = adaptive.Exchange(new { cmd = "spaces" });
            Assert.That(spaces.RootElement.GetProperty("contract_version").GetString(), Is.EqualTo("adaptive-v1"));
            string encodingHash = spaces.RootElement.GetProperty("encoding_hash").GetString();
            Assert.That(encodingHash, Does.Match("^[0-9a-f]{64}$"));
            Assert.That(spaces.RootElement.GetProperty("n_actions").GetInt32(), Is.EqualTo(182));
            Assert.That(spaces.RootElement.GetProperty("obs_len").GetInt32(), Is.EqualTo(5974));
            Assert.That(spaces.RootElement.GetProperty("adaptive").GetProperty("phases").GetArrayLength(), Is.EqualTo(14));
            using JsonDocument duelSpaces = adaptive.Exchange(new { cmd = "duel_spaces" });
            Assert.That(duelSpaces.RootElement.GetProperty("encoding_hash").GetString(), Is.EqualTo(encodingHash));
            Assert.That(duelSpaces.RootElement.GetProperty("contract_hash").GetString(),
                Is.Not.EqualTo(spaces.RootElement.GetProperty("contract_hash").GetString()));

            using JsonDocument reset = adaptive.Exchange(new { cmd = "reset", seed = 77 });
            JsonElement reply = reset.RootElement.Clone();
            var rng = new Random(77);
            for (int step = 0; step < 100; step++)
            {
                bool[] mask = reply.GetProperty("mask").EnumerateArray().Select(x => x.GetBoolean()).ToArray();
                Assert.That(mask.Length, Is.EqualTo(182));
                int[] legal = Enumerable.Range(0, mask.Length).Where(i => mask[i]).ToArray();
                Assert.That(legal, Is.Not.Empty);
                int action = legal[rng.Next(legal.Length)];
                using JsonDocument response = adaptive.Exchange(new { cmd = "step", action });
                reply = response.RootElement.Clone();
                Assert.That(reply.GetProperty("obs").GetArrayLength(), Is.EqualTo(5974));
                Assert.That(reply.GetProperty("diagnostics").GetProperty("invalid_sequences").GetInt32(), Is.Zero);
                if (reply.GetProperty("terminated").GetBoolean() || reply.GetProperty("truncated").GetBoolean())
                {
                    using JsonDocument again = adaptive.Exchange(new { cmd = "reset", seed = 78 + step });
                    reply = again.RootElement.Clone();
                }
            }
        }

        [TestCase("greedy", null, 1)]
        [TestCase(null, "greedy", 0)]
        public void AdaptiveDuelRpc_ConsumesRevealPauseForReciprocalExternalSeat(
            string? p0, string? p1, int externalSeat)
        {
            using var server = new ServerProcess("--environment", "adaptive-v1");
            using JsonDocument reset = server.Exchange(new { cmd = "duel_reset", seed = 91, p0, p1 });
            JsonElement reply = reset.RootElement.Clone();

            int guard = 0;
            while (!reply.GetProperty("deployment_complete").GetBoolean() && guard++ < 100)
            {
                Assert.That(reply.GetProperty("seat").GetInt32(), Is.EqualTo(externalSeat));
                bool[] mask = reply.GetProperty("mask").EnumerateArray().Select(x => x.GetBoolean()).ToArray();
                using JsonDocument stepped = server.Exchange(new
                {
                    cmd = "duel_step",
                    action = ChooseProgressAction(mask),
                });
                reply = stepped.RootElement.Clone();
            }

            Assert.That(reply.GetProperty("deployment_complete").GetBoolean(), Is.True);
            Assert.That(reply.GetProperty("terminated").GetBoolean()
                || reply.GetProperty("truncated").GetBoolean()
                || reply.GetProperty("seat").GetInt32() == externalSeat, Is.True,
                "headless reciprocal evaluation must receive the next external decision, not Unity's reveal pause");
        }

        [Test]
        public void AdaptiveDuelRpc_BothScriptedCompletesAndSavesTerminalReplay()
        {
            string replayPath = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "adaptive-scripted-" + Guid.NewGuid().ToString("N") + ".replay");
            try
            {
                using var server = new ServerProcess("--environment", "adaptive-v1");
                using JsonDocument reset = server.Exchange(new
                {
                    cmd = "duel_reset", seed = 92, p0 = "greedy", p1 = "greedy",
                });

                Assert.That(reset.RootElement.GetProperty("terminated").GetBoolean(), Is.True);
                using JsonDocument saved = server.Exchange(new { cmd = "duel_save", path = replayPath });
                Assert.That(saved.RootElement.GetProperty("saved").GetString(), Is.EqualTo(replayPath));
                ReplayData replay = ReplayFile.Read(File.ReadAllText(replayPath));
                Assert.That(new Replay(replay.Start, replay.Commands).Final.IsGameOver, Is.True);
            }
            finally
            {
                if (File.Exists(replayPath)) File.Delete(replayPath);
            }
        }

        [Test]
        public void GymServer_LoadsScenarioAndReportsResolvedContract()
        {
            string scenario = WriteScenario(environment: "adaptive-v1", width: 24, height: 16, maxSteps: 1800);
            try
            {
                using var server = new ServerProcess(
                    "--environment", "adaptive-v1", "--scenario-file", scenario);

                using JsonDocument spaces = server.Exchange(new { cmd = "spaces" });

                Assert.That(spaces.RootElement.GetProperty("scenario_id").GetString(), Is.EqualTo("test-large"));
                Assert.That(spaces.RootElement.GetProperty("scenario_schema_version").GetInt32(), Is.EqualTo(1));
                Assert.That(spaces.RootElement.GetProperty("board_w").GetInt32(), Is.EqualTo(24));
                Assert.That(spaces.RootElement.GetProperty("board_h").GetInt32(), Is.EqualTo(16));
                Assert.That(spaces.RootElement.GetProperty("max_steps").GetInt32(), Is.EqualTo(1800));

                using JsonDocument duelSpaces = server.Exchange(new { cmd = "duel_spaces" });
                Assert.That(duelSpaces.RootElement.GetProperty("scenario_id").GetString(), Is.EqualTo("test-large"));
                Assert.That(duelSpaces.RootElement.GetProperty("board_w").GetInt32(), Is.EqualTo(24));
                Assert.That(duelSpaces.RootElement.GetProperty("board_h").GetInt32(), Is.EqualTo(16));
                Assert.That(duelSpaces.RootElement.GetProperty("max_steps").GetInt32(), Is.EqualTo(3600));
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [TestCase(0)]
        [TestCase(1)]
        [TestCase(2)]
        [TestCase(3)]
        [TestCase(4)]
        [TestCase(5)]
        public void GymServer_AcceptsEveryCheckedInTrainingTemplate(int templateIndex)
        {
            string libraryPath = Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "..", "python", "config",
                "training-game-templates.json"));
            using JsonDocument library = JsonDocument.Parse(File.ReadAllText(libraryPath));
            JsonElement template = library.RootElement.GetProperty("templates")[templateIndex];
            string environment = template.GetProperty("environment").GetString()!;
            string scenario = WriteScenarioContent(template.GetRawText());
            try
            {
                using var server = new ServerProcess(
                    "--environment", environment, "--scenario-file", scenario);
                using JsonDocument spaces = server.Exchange(new { cmd = "spaces" });

                Assert.That(
                    spaces.RootElement.GetProperty("scenario_id").GetString(),
                    Is.EqualTo(template.GetProperty("id").GetString()));
                Assert.That(
                    spaces.RootElement.GetProperty("scenario_schema_version").GetInt32(),
                    Is.EqualTo(1));
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_RejectsMalformedScenarioBeforeCommandProcessing()
        {
            string scenario = WriteScenarioContent("{");
            try
            {
                AssertScenarioStartupFails("invalid scenario JSON", "--environment", "adaptive-v1", "--scenario-file", scenario);
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_RejectsUnsupportedScenarioSchemaBeforeCommandProcessing()
        {
            string scenario = WriteScenario(environment: "adaptive-v1", schemaVersion: 2);
            try
            {
                AssertScenarioStartupFails("schema version must be 1", "--environment", "adaptive-v1", "--scenario-file", scenario);
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_RejectsScenarioEnvironmentMismatchBeforeCommandProcessing()
        {
            string scenario = WriteScenario(environment: "adaptive-v1");
            try
            {
                AssertScenarioStartupFails("scenario environment does not match --environment",
                    "--environment", "tactical-v1", "--scenario-file", scenario);
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_RejectsScenarioMissingEnvironmentRewardBeforeCommandProcessing()
        {
            string scenario = WriteScenario(environment: "adaptive-v1", includeReward: false);
            try
            {
                AssertScenarioStartupFails("reward section is required", "--environment", "adaptive-v1", "--scenario-file", scenario);
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_RejectsScenarioWithImpossibleDeploymentBeforeCommandProcessing()
        {
            string scenario = WriteScenario(environment: "adaptive-v1", width: 2, height: 2, zoneDepth: 1);
            try
            {
                AssertScenarioStartupFails("adaptive deployment cells must cover the starting unit count",
                    "--environment", "adaptive-v1", "--scenario-file", scenario);
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_UnknownEnvironmentExitsTwoBeforeReadingInput()
        {
            string dll = ServerProcess.ServerDll;
            using var process = Process.Start(new ProcessStartInfo("dotnet", $"\"{dll}\" --environment future-v9")
            {
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            })!;

            Assert.That(process.WaitForExit(5000), Is.True);
            Assert.That(process.ExitCode, Is.EqualTo(2));
            Assert.That(process.StandardError.ReadToEnd(), Does.Contain("unsupported environment 'future-v9'"));
        }

        [TestCase(1)]
        [TestCase(12)]
        public void TacticalV2GymServer_SpacesResetStepAndDuelRoundTripForEveryCount(int unitCount)
        {
            string scenarioPath = WriteTacticalV2Scenario(unitCount);
            try
            {
                using var server = new ServerProcess("--environment", "tactical-v2", "--scenario-file", scenarioPath);

                using JsonDocument spaces = server.Exchange(new { cmd = "spaces" });
                Assert.That(spaces.RootElement.GetProperty("contract_version").GetString(), Is.EqualTo("tactical-v2"));
                Assert.That(spaces.RootElement.GetProperty("environment_kind").GetString(), Is.EqualTo("tactical"));
                Assert.That(
                    spaces.RootElement.GetProperty("tactical_v2").GetProperty("starting_unit_count").GetInt32(),
                    Is.EqualTo(unitCount));
                Assert.That(
                    spaces.RootElement.GetProperty("tactical_v2").GetProperty("max_controllable_units").GetInt32(),
                    Is.EqualTo(unitCount));
                Assert.That(spaces.RootElement.TryGetProperty("action_regions", out _), Is.True);
                Assert.That(spaces.RootElement.TryGetProperty("observation_channels", out _), Is.True);
                string encodingHash = spaces.RootElement.GetProperty("encoding_hash").GetString()!;
                Assert.That(encodingHash, Does.Match("^[0-9a-f]{64}$"));
                int obsLen = spaces.RootElement.GetProperty("obs_len").GetInt32();
                int nActions = spaces.RootElement.GetProperty("n_actions").GetInt32();

                using JsonDocument reset = server.Exchange(new { cmd = "reset", seed = 123 });
                Assert.That(reset.RootElement.GetProperty("obs").GetArrayLength(), Is.EqualTo(obsLen));
                bool[] mask = reset.RootElement.GetProperty("mask").EnumerateArray().Select(x => x.GetBoolean()).ToArray();
                Assert.That(mask.Length, Is.EqualTo(nActions));
                int action = Enumerable.Range(0, mask.Length).First(i => mask[i]);

                using JsonDocument step = server.Exchange(new { cmd = "step", action });
                Assert.That(step.RootElement.GetProperty("obs").GetArrayLength(), Is.EqualTo(obsLen));
                Assert.That(step.RootElement.GetProperty("mask").GetArrayLength(), Is.EqualTo(nActions));
                Assert.That(step.RootElement.TryGetProperty("terminated", out _), Is.True);
                Assert.That(step.RootElement.TryGetProperty("truncated", out _), Is.True);

                using JsonDocument duelSpaces = server.Exchange(new { cmd = "duel_spaces" });
                Assert.That(duelSpaces.RootElement.GetProperty("environment_kind").GetString(), Is.EqualTo("duel"));
                Assert.That(duelSpaces.RootElement.GetProperty("encoding_hash").GetString(), Is.EqualTo(encodingHash));
                Assert.That(duelSpaces.RootElement.GetProperty("contract_hash").GetString(),
                    Is.Not.EqualTo(spaces.RootElement.GetProperty("contract_hash").GetString()));

                using JsonDocument duelReset = server.Exchange(new { cmd = "duel_reset", seed = 5 });
                bool[] duelMask = duelReset.RootElement.GetProperty("mask").EnumerateArray()
                    .Select(x => x.GetBoolean()).ToArray();
                int duelAction = Enumerable.Range(0, duelMask.Length).First(i => duelMask[i]);
                using JsonDocument duelStep = server.Exchange(new { cmd = "duel_step", action = duelAction });
                Assert.That(duelStep.RootElement.TryGetProperty("seat", out _), Is.True);
                Assert.That(duelStep.RootElement.TryGetProperty("winner", out _), Is.True);

                string replayPath = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                    "tactical-v2-" + Guid.NewGuid().ToString("N") + ".replay");
                try
                {
                    using JsonDocument saved = server.Exchange(new { cmd = "duel_save", path = replayPath });
                    Assert.That(saved.RootElement.GetProperty("saved").GetString(), Is.EqualTo(replayPath));
                    ReplayData replay = ReplayFile.Read(File.ReadAllText(replayPath));
                    Assert.That(replay.Commands, Is.Not.Empty);
                }
                finally
                {
                    if (File.Exists(replayPath)) File.Delete(replayPath);
                }
            }
            finally
            {
                File.Delete(scenarioPath);
            }
        }

        [Test]
        public void TacticalV2GymServer_RoundCapDrawUsesIntegerMinusOneWinner()
        {
            string scenarioPath = WriteTacticalV2Scenario(
                startingUnitCount: 3, roundCap: 1);
            try
            {
                using var server = new ServerProcess(
                    "--environment", "tactical-v2", "--scenario-file", scenarioPath);

                using JsonDocument reset = server.Exchange(new
                {
                    cmd = "duel_reset", seed = 73,
                    p0 = "greedy", p1 = "random", learner = 0,
                });

                Assert.That(reset.RootElement.GetProperty("terminated").GetBoolean(), Is.True);
                Assert.That(reset.RootElement.GetProperty("truncated").GetBoolean(), Is.False);
                JsonElement winner = reset.RootElement.GetProperty("winner");
                Assert.That(winner.ValueKind, Is.EqualTo(JsonValueKind.Number));
                Assert.That(winner.GetInt32(), Is.EqualTo(-1));
            }
            finally
            {
                File.Delete(scenarioPath);
            }
        }

        [Test]
        public void TacticalV2GymServer_DefaultCatalogScenarioReportsInProcessEncodingHash()
        {
            // The checked-in "tactical-v2-standard" template is board/rules/reward/roster-identical
            // to TacticalV2Config.Default() (see TrainingScenario.CreateStandard's tactical-v2 branch,
            // which builds the same catalog verbatim). Pin the GymServer's reported encoding_hash for
            // that scenario to the in-process contract so a boundary bug in scenario parsing can't
            // silently drift the served identity away from what BuildTacticalV2 would compute.
            string libraryPath = Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "..", "python", "config",
                "training-game-templates.json"));
            using JsonDocument library = JsonDocument.Parse(File.ReadAllText(libraryPath));
            JsonElement template = library.RootElement.GetProperty("templates")
                .EnumerateArray()
                .Single(item => item.GetProperty("id").GetString() == "tactical-v2-standard");
            string scenario = WriteScenarioContent(template.GetRawText());
            try
            {
                using var server = new ServerProcess(
                    "--environment", "tactical-v2", "--scenario-file", scenario);
                using JsonDocument spaces = server.Exchange(new { cmd = "spaces" });

                string expectedEncodingHash = MlContract.CreateTacticalV2(TacticalV2Config.Default()).EncodingHash;
                Assert.That(spaces.RootElement.GetProperty("encoding_hash").GetString(),
                    Is.EqualTo(expectedEncodingHash));
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void TacticalV2GymServer_TraceIsOptInAndSeparateFromResetAndStepPayloads()
        {
            using var server = new ServerProcess("--environment", "tactical-v2");

            using JsonDocument enabled = server.Exchange(new { cmd = "duel_trace_enable", enabled = true });
            Assert.That(enabled.RootElement.GetProperty("enabled").GetBoolean(), Is.True);

            using JsonDocument spaces = server.Exchange(new { cmd = "spaces" });
            Assert.That(spaces.RootElement.TryGetProperty("trace", out _), Is.False);
            using JsonDocument reset = server.Exchange(new { cmd = "reset", seed = 41 });
            Assert.That(reset.RootElement.TryGetProperty("trace", out _), Is.False);
            int trainingAction = reset.RootElement.GetProperty("mask").EnumerateArray()
                .Select((item, index) => (item, index)).First(pair => pair.item.GetBoolean()).index;
            using JsonDocument step = server.Exchange(new { cmd = "step", action = trainingAction });
            Assert.That(step.RootElement.TryGetProperty("trace", out _), Is.False);

            using JsonDocument duelReset = server.Exchange(new { cmd = "duel_reset", seed = 42 });
            Assert.That(duelReset.RootElement.TryGetProperty("trace", out _), Is.False);
            int duelAction = duelReset.RootElement.GetProperty("mask").EnumerateArray()
                .Select((item, index) => (item, index)).First(pair => pair.item.GetBoolean()).index;
            using JsonDocument duelStep = server.Exchange(new { cmd = "duel_step", action = duelAction });
            Assert.That(duelStep.RootElement.TryGetProperty("trace", out _), Is.False);

            using JsonDocument resetClearsTrace = server.Exchange(new { cmd = "duel_reset", seed = 43 });
            Assert.That(resetClearsTrace.RootElement.TryGetProperty("trace", out _), Is.False);
            using JsonDocument emptyAfterReset = server.Exchange(new { cmd = "duel_trace_drain" });
            Assert.That(emptyAfterReset.RootElement.GetProperty("schema_version").GetInt32(), Is.EqualTo(1));
            Assert.That(emptyAfterReset.RootElement.GetProperty("transitions").GetArrayLength(), Is.Zero);

            using JsonDocument traceProducingReset = server.Exchange(new
            {
                cmd = "duel_reset", seed = 44, p0 = "greedy", p1 = "greedy", learner = 0,
            });
            Assert.That(traceProducingReset.RootElement.TryGetProperty("trace", out _), Is.False);
            using JsonDocument trace = server.Exchange(new { cmd = "duel_trace_drain" });
            Assert.That(trace.RootElement.GetProperty("schema_version").GetInt32(), Is.EqualTo(1));
            Assert.That(trace.RootElement.GetProperty("transitions").GetArrayLength(), Is.GreaterThan(0));
            using JsonDocument emptyAfterDrain = server.Exchange(new { cmd = "duel_trace_drain" });
            Assert.That(emptyAfterDrain.RootElement.GetProperty("transitions").GetArrayLength(), Is.Zero);

            using JsonDocument pending = server.Exchange(new
            {
                cmd = "duel_reset", seed = 45, p0 = "greedy", p1 = "greedy", learner = 0,
            });
            using JsonDocument disabled = server.Exchange(new { cmd = "duel_trace_enable", enabled = false });
            Assert.That(disabled.RootElement.GetProperty("enabled").GetBoolean(), Is.False);
            using JsonDocument emptyAfterDisable = server.Exchange(new { cmd = "duel_trace_drain" });
            Assert.That(emptyAfterDisable.RootElement.GetProperty("schema_version").GetInt32(), Is.EqualTo(1));
            Assert.That(emptyAfterDisable.RootElement.GetProperty("transitions").GetArrayLength(), Is.Zero);
        }

        [Test]
        public void TacticalV2GymServer_TraceCaptureIsBehaviorallyPassiveAndReplayMatchesFinalState()
        {
            string tracedReplayPath = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "tactical-v2-traced-" + Guid.NewGuid().ToString("N") + ".replay");
            string untracedReplayPath = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "tactical-v2-untraced-" + Guid.NewGuid().ToString("N") + ".replay");
            try
            {
                using var server = new ServerProcess("--environment", "tactical-v2");
                using JsonDocument enabled = server.Exchange(new
                {
                    cmd = "duel_trace_enable", enabled = true,
                });
                Assert.That(enabled.RootElement.GetProperty("enabled").GetBoolean(), Is.True);

                using JsonDocument tracedReset = server.Exchange(new
                {
                    cmd = "duel_reset", seed = 137, p0 = "greedy", p1 = "random", learner = 0,
                });
                Assert.That(tracedReset.RootElement.GetProperty("terminated").GetBoolean(), Is.True);
                using JsonDocument trace = server.Exchange(new { cmd = "duel_trace_drain" });
                int tracedTransitionCount = trace.RootElement.GetProperty("transitions").GetArrayLength();
                Assert.That(tracedTransitionCount, Is.GreaterThan(0));

                using JsonDocument tracedSaved = server.Exchange(new
                {
                    cmd = "duel_save", path = tracedReplayPath,
                });
                Assert.That(tracedSaved.RootElement.GetProperty("saved").GetString(),
                    Is.EqualTo(tracedReplayPath));
                ReplayData tracedData = ReplayFile.Read(File.ReadAllText(tracedReplayPath));
                var tracedReplay = new Replay(tracedData.Start, tracedData.Commands);
                Assert.That(tracedReplay.Final.IsGameOver, Is.True);
                Assert.That(tracedTransitionCount, Is.EqualTo(tracedData.Commands.Count));
                JsonElement tracedWinner = tracedReset.RootElement.GetProperty("winner");
                Assert.That(tracedWinner.ValueKind, Is.EqualTo(JsonValueKind.Number));
                Assert.That(tracedWinner.GetInt32(), Is.EqualTo(
                    tracedReplay.Final.Winner.HasValue
                        ? (int)tracedReplay.Final.Winner.Value
                        : -1));

                using JsonDocument disabled = server.Exchange(new
                {
                    cmd = "duel_trace_enable", enabled = false,
                });
                Assert.That(disabled.RootElement.GetProperty("enabled").GetBoolean(), Is.False);
                using JsonDocument untracedReset = server.Exchange(new
                {
                    cmd = "duel_reset", seed = 137, p0 = "greedy", p1 = "random", learner = 0,
                });
                Assert.That(untracedReset.RootElement.GetProperty("terminated").GetBoolean(), Is.True);
                using JsonDocument noTrace = server.Exchange(new { cmd = "duel_trace_drain" });
                Assert.That(noTrace.RootElement.GetProperty("transitions").GetArrayLength(), Is.Zero);

                using JsonDocument untracedSaved = server.Exchange(new
                {
                    cmd = "duel_save", path = untracedReplayPath,
                });
                Assert.That(untracedSaved.RootElement.GetProperty("saved").GetString(),
                    Is.EqualTo(untracedReplayPath));
                ReplayData untracedData = ReplayFile.Read(File.ReadAllText(untracedReplayPath));
                var untracedReplay = new Replay(untracedData.Start, untracedData.Commands);
                Assert.That(untracedReplay.Final.IsGameOver, Is.True);
                JsonElement untracedWinner = untracedReset.RootElement.GetProperty("winner");
                Assert.That(untracedWinner.ValueKind, Is.EqualTo(JsonValueKind.Number));
                Assert.That(untracedWinner.GetInt32(), Is.EqualTo(
                    untracedReplay.Final.Winner.HasValue
                        ? (int)untracedReplay.Final.Winner.Value
                        : -1));

                Assert.That(untracedData.Commands, Is.EqualTo(tracedData.Commands),
                    "trace capture must not change accepted commands");
                Assert.That(untracedReplay.Final.Winner, Is.EqualTo(tracedReplay.Final.Winner));
                Assert.That(
                    ReplayFile.Write(untracedReplay.Final, Array.Empty<Command>()),
                    Is.EqualTo(ReplayFile.Write(tracedReplay.Final, Array.Empty<Command>())),
                    "trace capture must not change the reconstructed final state");
            }
            finally
            {
                if (File.Exists(tracedReplayPath)) File.Delete(tracedReplayPath);
                if (File.Exists(untracedReplayPath)) File.Delete(untracedReplayPath);
            }
        }

        [TestCase("duel_trace_enable")]
        [TestCase("duel_trace_drain")]
        public void GymServer_RejectsTraceRpcOutsideTacticalV2(string command)
        {
            using var server = new ServerProcess("--environment", "adaptive-v1");

            string error = server.ExchangeFailure(command == "duel_trace_enable"
                ? new { cmd = command, enabled = true }
                : new { cmd = command });

            Assert.That(error, Does.Contain("duel trace is supported only for tactical-v2"));
        }

        [Test]
        public void GymServerScenarioJson_LoadsPythonProfiledTacticalV2SchemaIntoTrainingConfig()
        {
            string scenario = WriteScenarioContent("""
            {
              "schema_version": 1,
              "id": "python-profiled-tactical-v2",
              "name": "Python Profiled Tactical V2",
              "environment": "tactical-v2",
              "board": {"width": 13, "height": 9, "max_elevation": 4, "zone_depth": 3, "flat_chance": 0.6, "plains_weight": 70, "forest_weight": 15, "rough_weight": 10, "water_weight": 5},
              "rules": {"actions_per_turn": 0, "round_cap": 100, "starting_points": 12, "fog_of_war": false, "biomes_enabled": false, "bounty_rate": 0.5, "deploy_cost_multiplier": 1.0, "generator_cost": 2, "generator_output": 1, "generator_health": 3},
              "episode": {"max_steps": 600},
              "reward": {"shape_scale": 0.01, "step_penalty": 0.005, "closing_weight": 0.02, "draw_credit_weight": 0.25, "points_weight": 0.5},
              "tactical_v2": {
                "starting_unit_count": 3,
                "max_controllable_units": 3,
                "placement_policy": "profiled-seeded-v1",
                "start_profiles": [
                  {"id": "standard-3v3", "learner_units": 3, "opponent_units": 3, "separation": "legacy-mirrored"},
                  {"id": "conversion-3v1-near", "learner_units": 3, "opponent_units": 1, "separation": "near"},
                  {"id": "conversion-3v1-medium", "learner_units": 3, "opponent_units": 1, "separation": "medium"},
                  {"id": "conversion-3v1-far", "learner_units": 3, "opponent_units": 1, "separation": "far"},
                  {"id": "conversion-2v1-near", "learner_units": 2, "opponent_units": 1, "separation": "near"},
                  {"id": "conversion-2v1-medium", "learner_units": 2, "opponent_units": 1, "separation": "medium"},
                  {"id": "conversion-2v1-far", "learner_units": 2, "opponent_units": 1, "separation": "far"},
                  {"id": "conversion-1v1-near", "learner_units": 1, "opponent_units": 1, "separation": "near"},
                  {"id": "conversion-1v1-medium", "learner_units": 1, "opponent_units": 1, "separation": "medium"},
                  {"id": "conversion-1v1-far", "learner_units": 1, "opponent_units": 1, "separation": "far"}
                ],
                "start_distribution": [
                  {"profile_id": "standard-3v3", "basis_points": 10000},
                  {"profile_id": "conversion-3v1-near", "basis_points": 0},
                  {"profile_id": "conversion-3v1-medium", "basis_points": 0},
                  {"profile_id": "conversion-3v1-far", "basis_points": 0},
                  {"profile_id": "conversion-2v1-near", "basis_points": 0},
                  {"profile_id": "conversion-2v1-medium", "basis_points": 0},
                  {"profile_id": "conversion-2v1-far", "basis_points": 0},
                  {"profile_id": "conversion-1v1-near", "basis_points": 0},
                  {"profile_id": "conversion-1v1-medium", "basis_points": 0},
                  {"profile_id": "conversion-1v1-far", "basis_points": 0}
                ],
                "templates": [
                  {"id": "brute", "name": "Brute", "stats": {"health": 7, "damage": 2, "defense": 2, "movement": 3, "vertical_movement": 2, "range": 1, "range_arc": 1, "vision": 2, "vision_arc": 1}}
                ]
              }
            }
            """);
            try
            {
                var assembly = System.Reflection.Assembly.LoadFrom(ServerProcess.ServerDll);
                Type parserType = assembly.GetType("HexWars.GymServer.ScenarioJson", throwOnError: true)!;
                var load = parserType.GetMethod("Load", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
                var parsed = (TrainingScenario)load.Invoke(null, new object[] { scenario })!;
                TrainingTacticalV2Config tacticalV2 = parsed.TacticalV2;

                Assert.That(tacticalV2.PlacementPolicy, Is.EqualTo("profiled-seeded-v1"));
                Assert.That(tacticalV2.StartingUnitCount, Is.EqualTo(3));
                Assert.That(tacticalV2.MaxControllableUnits, Is.EqualTo(3));
                Assert.That(tacticalV2.StartProfiles.Select(profile =>
                    (profile.Id, profile.LearnerUnitCount, profile.OpponentUnitCount, profile.Separation)), Is.EqualTo(new[]
                {
                    ("standard-3v3", 3, 3, "legacy-mirrored"),
                    ("conversion-3v1-near", 3, 1, "near"), ("conversion-3v1-medium", 3, 1, "medium"), ("conversion-3v1-far", 3, 1, "far"),
                    ("conversion-2v1-near", 2, 1, "near"), ("conversion-2v1-medium", 2, 1, "medium"), ("conversion-2v1-far", 2, 1, "far"),
                    ("conversion-1v1-near", 1, 1, "near"), ("conversion-1v1-medium", 1, 1, "medium"), ("conversion-1v1-far", 1, 1, "far"),
                }));
                Assert.That(tacticalV2.StartDistribution.Select(weight =>
                    (weight.ProfileId, weight.BasisPoints)), Is.EqualTo(new[]
                {
                    ("standard-3v3", 10000), ("conversion-3v1-near", 0), ("conversion-3v1-medium", 0), ("conversion-3v1-far", 0),
                    ("conversion-2v1-near", 0), ("conversion-2v1-medium", 0), ("conversion-2v1-far", 0),
                    ("conversion-1v1-near", 0), ("conversion-1v1-medium", 0), ("conversion-1v1-far", 0),
                }));
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_RejectsTacticalV2ScenarioWithMismatchedCounts()
        {
            string scenario = WriteTacticalV2Scenario(startingUnitCount: 4, maxControllableUnits: 6);
            try
            {
                AssertScenarioStartupFails("tactical-v2 max controllable units must equal starting unit count",
                    "--environment", "tactical-v2", "--scenario-file", scenario);
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_RejectsTacticalV2ScenarioMissingSection()
        {
            string scenario = WriteTacticalV2Scenario(1, includeTacticalV2: false);
            try
            {
                AssertScenarioStartupFails("tactical_v2 section", "--environment", "tactical-v2", "--scenario-file", scenario);
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_RejectsTacticalV2ScenarioWithUnknownField()
        {
            string scenario = WriteTacticalV2Scenario(1, includeUnknownField: true);
            try
            {
                AssertScenarioStartupFails("invalid scenario JSON", "--environment", "tactical-v2", "--scenario-file", scenario);
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        private static string WriteTacticalV2Scenario(
            int startingUnitCount,
            int? maxControllableUnits = null,
            bool includeTacticalV2 = true,
            bool includeUnknownField = false,
            int width = 24,
            int height = 16,
            int zoneDepth = 4,
            int roundCap = 100)
        {
            var scenario = new Dictionary<string, object?>
            {
                ["schema_version"] = 1,
                ["id"] = "test-tactical-v2",
                ["name"] = "Test Tactical V2",
                ["environment"] = "tactical-v2",
                ["board"] = new
                {
                    width,
                    height,
                    max_elevation = 4,
                    zone_depth = zoneDepth,
                    flat_chance = 0.6,
                    plains_weight = 70,
                    forest_weight = 15,
                    rough_weight = 10,
                    water_weight = 5,
                },
                ["rules"] = new
                {
                    actions_per_turn = 0,
                    round_cap = roundCap,
                    starting_points = 12,
                    fog_of_war = false,
                    biomes_enabled = false,
                    bounty_rate = 0.5,
                    deploy_cost_multiplier = 1.0,
                    generator_cost = 2,
                    generator_output = 1,
                    generator_health = 3,
                },
                ["episode"] = new { max_steps = 600 },
                ["reward"] = new
                {
                    shape_scale = 0.01f,
                    step_penalty = 0.005f,
                    closing_weight = 0.02f,
                    draw_credit_weight = 0.25f,
                    points_weight = 0.5f,
                },
            };

            if (includeTacticalV2)
            {
                var tacticalV2 = new Dictionary<string, object?>
                {
                    ["starting_unit_count"] = startingUnitCount,
                    ["max_controllable_units"] = maxControllableUnits ?? startingUnitCount,
                    ["placement_policy"] = "symmetric-random-v1",
                    ["templates"] = new object[]
                    {
                        new
                        {
                            id = "brute", name = "Brute",
                            stats = new
                            {
                                health = 7, damage = 2, defense = 2, movement = 3, vertical_movement = 2,
                                range = 1, range_arc = 1, vision = 2, vision_arc = 1,
                            },
                        },
                        new
                        {
                            id = "scout", name = "Scout",
                            stats = new
                            {
                                health = 2, damage = 2, defense = 0, movement = 4, vertical_movement = 3,
                                range = 1, range_arc = 0, vision = 5, vision_arc = 2,
                            },
                        },
                    },
                };
                if (includeUnknownField) tacticalV2["bogus_field"] = true;
                scenario["tactical_v2"] = tacticalV2;
            }

            return WriteScenarioContent(JsonSerializer.Serialize(scenario));
        }

        private static AdaptiveDuelEnv.View DeployExternalSeat(
            AdaptiveDuelEnv env, AdaptiveDuelEnv.View view, bool useCustomTemplate)
        {
            int templateAction = env.Layout.TemplateOffset + (useCustomTemplate ? 6 : 0);
            for (int unit = 0; unit < 6; unit++)
            {
                Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.DeployStartingUnit], Is.True);
                view = env.Step((int)AdaptiveCommandChoice.DeployStartingUnit);
                Assert.That(view.ActionMask[templateAction], Is.True);
                view = env.Step(templateAction);
                int cell = Enumerable.Range(env.Layout.CellOffset, env.Layout.CellCount)
                    .First(i => view.ActionMask[i]);
                view = env.Step(cell);
            }
            Assert.That(view.ActionMask[(int)AdaptiveCommandChoice.ConfirmDeployment], Is.True);
            return env.Step((int)AdaptiveCommandChoice.ConfirmDeployment);
        }

        private static int ChooseProgressAction(bool[] mask)
        {
            if (mask[(int)AdaptiveCommandChoice.EndTurn]) return (int)AdaptiveCommandChoice.EndTurn;
            if (mask[(int)AdaptiveCommandChoice.ConfirmDeployment]) return (int)AdaptiveCommandChoice.ConfirmDeployment;
            if (mask[(int)AdaptiveCommandChoice.DeployStartingUnit]) return (int)AdaptiveCommandChoice.DeployStartingUnit;
            return Enumerable.Range(0, mask.Length).First(i => mask[i] && i != (int)AdaptiveCommandChoice.Cancel);
        }

        private static string WriteScenario(
            string environment,
            int schemaVersion = 1,
            int width = 13,
            int height = 9,
            int zoneDepth = 3,
            int maxSteps = 900,
            bool includeReward = true)
        {
            var scenario = new Dictionary<string, object?>
            {
                ["schema_version"] = schemaVersion,
                ["id"] = "test-large",
                ["name"] = "Test Large",
                ["environment"] = environment,
                ["board"] = new
                {
                    width,
                    height,
                    max_elevation = 4,
                    zone_depth = zoneDepth,
                    flat_chance = 0.6,
                    plains_weight = 70,
                    forest_weight = 15,
                    rough_weight = 10,
                    water_weight = 5,
                },
                ["rules"] = new
                {
                    actions_per_turn = 0,
                    round_cap = 100,
                    starting_points = 12,
                    fog_of_war = environment == "adaptive-v1",
                    biomes_enabled = false,
                    bounty_rate = 0.5,
                    deploy_cost_multiplier = 1.0,
                    generator_cost = 2,
                    generator_output = 1,
                    generator_health = 3,
                },
                ["episode"] = new { max_steps = maxSteps },
            };

            if (includeReward)
            {
                scenario["reward"] = environment == "adaptive-v1"
                    ? new { intermediate_decision_penalty = 0.001f, deployment_completion_bonus = 0.0f }
                    : new
                    {
                        shape_scale = 0.01f,
                        step_penalty = 0.005f,
                        closing_weight = 0.02f,
                        draw_credit_weight = 0.25f,
                        points_weight = 0.5f,
                    };
            }

            if (environment == "adaptive-v1")
            {
                scenario["adaptive"] = new
                {
                    starting_unit_count = 6,
                    starting_army_budget = 132,
                    max_design_point_cost = 24,
                };
            }

            return WriteScenarioContent(JsonSerializer.Serialize(scenario));
        }

        private static string WriteScenarioContent(string content)
        {
            string path = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "scenario-" + Guid.NewGuid().ToString("N") + ".json");
            File.WriteAllText(path, content);
            return path;
        }

        private static void AssertScenarioStartupFails(string expectedError, params string[] args)
        {
            string arguments = $"\"{ServerProcess.ServerDll}\" " + string.Join(" ", args.Select(x => $"\"{x}\""));
            using var process = Process.Start(new ProcessStartInfo("dotnet", arguments)
            {
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            })!;

            try
            {
                Assert.That(process.WaitForExit(5000), Is.True, "GymServer should reject the scenario before reading commands");
                string error = process.StandardError.ReadToEnd();
                Assert.That(process.ExitCode, Is.Not.EqualTo(0));
                Assert.That(error, Does.Contain(expectedError));
            }
            finally
            {
                if (!process.HasExited) process.Kill(entireProcessTree: true);
            }
        }

        private sealed class RecordingDeploymentPolicy : IDeploymentPolicy
        {
            public int Calls { get; private set; }
            public int OpponentFactsObserved { get; private set; }

            public IReadOnlyList<DeploymentPlacement> Choose(AdaptiveDeploymentView view)
            {
                Calls++;
                OpponentFactsObserved += view.OwnPlacements.Count;
                return new CombinedArmsDeploymentPolicy(7).Choose(view);
            }
        }

        private sealed class InvalidTrailingDeploymentPolicy : IDeploymentPolicy
        {
            public int Calls { get; private set; }

            public IReadOnlyList<DeploymentPlacement> Choose(AdaptiveDeploymentView view)
            {
                Calls++;
                var placements = new List<DeploymentPlacement>(
                    new CombinedArmsDeploymentPolicy(13).Choose(view));
                placements.Add(placements[0]);
                return placements;
            }
        }

        private sealed class RecordingEndTurnAgent : IAgent
        {
            public int Calls { get; private set; }

            public Command Decide(GameState state)
            {
                Calls++;
                return new EndTurn(state.ActivePlayer);
            }
        }

        private sealed class ServerProcess : IDisposable
        {
            private readonly Process _process;
            public static string ServerDll => Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "HexWars.GymServer", "bin", "Debug", "net8.0", "HexWars.GymServer.dll"));

            public ServerProcess(params string[] args)
            {
                Assert.That(File.Exists(ServerDll), Is.True, $"GymServer was not built at {ServerDll}");
                string arguments = $"\"{ServerDll}\" " + string.Join(" ", args.Select(x => $"\"{x}\""));
                _process = Process.Start(new ProcessStartInfo("dotnet", arguments)
                {
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                })!;
            }

            public JsonDocument Exchange(object request)
            {
                _process.StandardInput.WriteLine(JsonSerializer.Serialize(request));
                _process.StandardInput.Flush();
                string? line = _process.StandardOutput.ReadLine();
                if (line == null)
                    Assert.Fail($"GymServer exited without a reply: {_process.StandardError.ReadToEnd()}");
                return JsonDocument.Parse(line!);
            }

            public string ExchangeFailure(object request)
            {
                _process.StandardInput.WriteLine(JsonSerializer.Serialize(request));
                _process.StandardInput.Flush();
                Assert.That(_process.StandardOutput.ReadLine(), Is.Null);
                Assert.That(_process.WaitForExit(5000), Is.True, "GymServer should reject unsupported trace RPCs");
                return _process.StandardError.ReadToEnd();
            }

            public void Dispose()
            {
                if (_process.HasExited) { _process.Dispose(); return; }
                _process.StandardInput.WriteLine("{\"cmd\":\"close\"}");
                _process.StandardInput.Flush();
                if (!_process.WaitForExit(5000)) _process.Kill(entireProcessTree: true);
                _process.Dispose();
            }
        }
    }
}

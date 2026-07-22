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

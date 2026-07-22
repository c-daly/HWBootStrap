using System;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class AdaptiveTacticalEnvTests
    {
        [Test]
        public void TacticalAndDuelAdaptiveEnvironments_ReportIdenticalPinnedSpaces()
        {
            var tactical = new AdaptiveTacticalEnv(
                seed => new GreedyAgent(seed), seed => new CombinedArmsDeploymentPolicy(seed));
            var duel = new AdaptiveDuelEnv();

            Assert.That(tactical.ActionCount, Is.EqualTo(duel.ActionCount));
            Assert.That(tactical.ObservationLength, Is.EqualTo(duel.ObservationLength));
            Assert.That(tactical.ActionCount, Is.EqualTo(182));
            Assert.That(tactical.ObservationLength, Is.EqualTo(5974));
            Assert.That(tactical.Contract.Version, Is.EqualTo("adaptive-v1"));
            Assert.That(tactical.Contract.EnvironmentKind, Is.EqualTo("adaptive_tactical"));
            Assert.That(duel.Contract.EnvironmentKind, Is.EqualTo("adaptive_duel"));
        }

        [Test]
        public void Reset_InvokesScriptedDeploymentOnceAndKeepsItHidden()
        {
            var policy = new CountingDeploymentPolicy();
            var env = new AdaptiveTacticalEnv(seed => new GreedyAgent(seed), seed => policy);

            float[] observation = env.Reset(17);

            Assert.That(policy.Calls, Is.EqualTo(1));
            Assert.That(env.DeploymentComplete, Is.False);
            Assert.That(observation.Length, Is.EqualTo(5974));
            Assert.That(env.Diagnostics.PregameDecisions, Is.Zero);
            int boardValues = env.ObsChannels * env.Layout.CellCount;
            Assert.That(observation.Take(boardValues).Any(value => value < 0f || value > 1f), Is.False);
            for (int role = 0; role < env.Layout.TemplateCount; role++)
                Assert.That(observation.Skip(env.Layout.EnemyUnitPlane(role) * env.Layout.CellCount)
                    .Take(env.Layout.CellCount).Any(value => value != 0f), Is.False,
                    "scripted opponent placements must remain absent from the learner observation");
        }

        [Test]
        public void IntermediateDecision_OnlyPenalizesAndDoesNotCompleteDeployment()
        {
            var env = NewEnv();
            env.Reset(5);

            StepResult result = env.Step((int)AdaptiveCommandChoice.DeployStartingUnit);

            Assert.That(result.Reward, Is.EqualTo(-env.Config.IntermediateDecisionPenalty));
            Assert.That(env.DeploymentComplete, Is.False);
            Assert.That(env.Diagnostics.PregameDecisions, Is.EqualTo(1));
            Assert.That(result.ActionMask.Skip(env.Layout.TemplateOffset)
                .Take(env.Layout.TemplateCount).Any(value => value), Is.True);
        }

        [Test]
        public void InvalidSequence_ClearsToRootWithoutFallback()
        {
            var env = NewEnv();
            env.Reset(8);
            env.Step((int)AdaptiveCommandChoice.DeployStartingUnit);

            StepResult result = env.Step(env.ActionCount + 10);

            Assert.That(env.Diagnostics.InvalidSequences, Is.EqualTo(1));
            Assert.That(env.DeploymentComplete, Is.False);
            Assert.That(result.ActionMask[(int)AdaptiveCommandChoice.DeployStartingUnit], Is.True);
            Assert.That(result.ActionMask[(int)AdaptiveCommandChoice.Cancel], Is.False);
        }

        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void InvalidScriptedOpponentDeployment_FailsResetWithoutExposingOpponentSeat(PlayerId learner)
        {
            var policy = new InvalidTrailingDeploymentPolicy();
            var env = new AdaptiveTacticalEnv(
                seed => new GreedyAgent(seed), seed => policy, learner);

            Assert.That(() => env.Reset(18), Throws.InvalidOperationException
                .With.Message.Contains("scripted deployment policy"));
            Assert.That(policy.Calls, Is.EqualTo(1));
        }

        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void ValidScriptedOpponent_AlwaysReturnsLearnerActions(PlayerId learner)
        {
            var env = new AdaptiveTacticalEnv(
                seed => new GreedyAgent(seed), seed => new CombinedArmsDeploymentPolicy(seed), learner);

            env.Reset(19);

            Assert.That(env.LegalActionMask()[(int)AdaptiveCommandChoice.DeployStartingUnit], Is.True);
        }

        [Test]
        public void Player1Learner_ContinuesPastPausedScriptedFirstGameplaySeat()
        {
            var opponent = new CountingEndTurnAgent();
            var env = new AdaptiveTacticalEnv(
                seed => opponent, seed => new CombinedArmsDeploymentPolicy(seed), PlayerId.Player1);
            env.Reset(20);

            for (int template = 0; template < 6; template++) Place(env, template);
            StepResult reveal = env.Step((int)AdaptiveCommandChoice.ConfirmDeployment);

            Assert.That(env.DeploymentComplete, Is.True);
            Assert.That(opponent.Calls, Is.EqualTo(1));
            Assert.That(env.State.ActivePlayer, Is.EqualTo(PlayerId.Player1));
            Assert.That(reveal.ActionMask.Any(value => value), Is.True);
        }

        [Test]
        public void RevealContinuation_PreservesDeploymentAndTerminalRewardsExactlyOnce()
        {
            var config = AdaptiveEnvConfig.Default();
            config.Game = GameConfig.Default(
                biomesEnabled: false,
                winConditions: WinBy.Economy,
                economyWinThreshold: 0,
                fogOfWar: true,
                maxDesignPointCost: 24,
                fixedTemplateCount: 6,
                templateSlotCount: 9);
            config.DeploymentCompletionBonus = 0.25f;
            var opponent = new CountingEndTurnAgent();
            var env = new AdaptiveTacticalEnv(
                seed => opponent, seed => new CombinedArmsDeploymentPolicy(seed),
                PlayerId.Player1, config);
            env.Reset(21);

            for (int template = 0; template < 6; template++) Place(env, template);
            StepResult reveal = env.Step((int)AdaptiveCommandChoice.ConfirmDeployment);

            Assert.That(opponent.Calls, Is.EqualTo(1));
            Assert.That(reveal.Terminated, Is.True);
            Assert.That(env.State.Winner.HasValue, Is.True);
            float terminal = env.State.Winner == PlayerId.Player1 ? 1f : -1f;
            Assert.That(reveal.Reward, Is.EqualTo(
                config.DeploymentCompletionBonus - config.IntermediateDecisionPenalty + terminal));
        }

        [Test]
        public void LearnerSeatZero_TerminalEndTurnReturnsLearnerPerspective()
        {
            var config = AdaptiveEnvConfig.Default();
            config.Game = GameConfig.Default(
                biomesEnabled: false,
                winConditions: WinBy.Economy,
                economyWinThreshold: 0,
                fogOfWar: true,
                maxDesignPointCost: config.MaxDesignPointCost,
                fixedTemplateCount: config.FixedTemplateCount,
                templateSlotCount: config.Templates.Count);
            var env = new AdaptiveTacticalEnv(
                seed => new GreedyAgent(seed), seed => new CombinedArmsDeploymentPolicy(seed),
                PlayerId.Player0, config);
            env.Reset(22);
            for (int template = 0; template < 6; template++) Place(env, template);
            env.Step((int)AdaptiveCommandChoice.ConfirmDeployment);

            StepResult terminal = env.Step((int)AdaptiveCommandChoice.EndTurn);

            Assert.That(terminal.Terminated, Is.True);
            Assert.That(terminal.Reward, Is.EqualTo(1f));
            Assert.That(terminal.Observation.Length, Is.EqualTo(env.ObservationLength));
        }

        [Test]
        public void MaskedDeployment_IsDeterministicAndRevealsAtomically()
        {
            var a = NewEnv();
            var b = NewEnv();
            float[] obsA = a.Reset(29);
            float[] obsB = b.Reset(29);

            for (int template = 0; template < 6; template++)
            {
                obsA = Place(a, template).Observation;
                obsB = Place(b, template).Observation;
                Assert.That(obsB, Is.EqualTo(obsA));
            }
            var finalA = a.Step((int)AdaptiveCommandChoice.ConfirmDeployment);
            var finalB = b.Step((int)AdaptiveCommandChoice.ConfirmDeployment);

            Assert.That(a.DeploymentComplete, Is.True);
            Assert.That(b.DeploymentComplete, Is.True);
            Assert.That(finalB.Observation, Is.EqualTo(finalA.Observation));
            Assert.That(a.State.Players.SelectMany(player => player.UnitsOnBoard).Count(), Is.EqualTo(12));
            Assert.That(a.Diagnostics.PregameDecisions, Is.EqualTo(19));
            Assert.That(a.Diagnostics.DeploymentCompleted, Is.True);
        }

        private static AdaptiveTacticalEnv NewEnv() => new AdaptiveTacticalEnv(
            seed => new GreedyAgent(seed), seed => new CombinedArmsDeploymentPolicy(seed));

        private static StepResult Place(AdaptiveTacticalEnv env, int template)
        {
            env.Step((int)AdaptiveCommandChoice.DeployStartingUnit);
            env.Step(env.Layout.TemplateOffset + template);
            bool[] mask = env.LegalActionMask();
            int cellAction = Enumerable.Range(env.Layout.CellOffset, env.Layout.CellCount).First(i => mask[i]);
            return env.Step(cellAction);
        }

        private sealed class CountingDeploymentPolicy : IDeploymentPolicy
        {
            public int Calls { get; private set; }

            public System.Collections.Generic.IReadOnlyList<DeploymentPlacement> Choose(AdaptiveDeploymentView view)
            {
                Calls++;
                return new CombinedArmsDeploymentPolicy(91).Choose(view);
            }
        }

        private sealed class CountingEndTurnAgent : IAgent
        {
            public int Calls { get; private set; }

            public Command Decide(GameState state)
            {
                Calls++;
                return new EndTurn(state.ActivePlayer);
            }
        }

        private sealed class InvalidTrailingDeploymentPolicy : IDeploymentPolicy
        {
            public int Calls { get; private set; }

            public System.Collections.Generic.IReadOnlyList<DeploymentPlacement> Choose(AdaptiveDeploymentView view)
            {
                Calls++;
                var placements = new System.Collections.Generic.List<DeploymentPlacement>(
                    new CombinedArmsDeploymentPolicy(13).Choose(view));
                placements.Add(placements[0]);
                return placements;
            }
        }
    }
}

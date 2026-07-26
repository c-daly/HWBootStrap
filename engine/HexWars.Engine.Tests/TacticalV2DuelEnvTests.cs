using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV2DuelEnvTests
    {
        /// <summary>Regression: an internal scripted controller (Greedy) decides purely from raw engine
        /// legality — board cells and points, never the RL registry's synthetic per-seat capacity — so
        /// nothing stops it from proposing a DeployUnit once every registry slot already holds a living
        /// unit. Before the fix, TryApply forwarded that command straight to
        /// <see cref="TacticalV2UnitRegistry.RegisterDeployment"/>, which throws when the registry is
        /// full. Two fully internal Greedy controllers playing to completion, for every starting seed in
        /// this range, must never throw.</summary>
        [Test]
        public void TwoInternalGreedyControllers_PlayToEnd_NeverOverflowRegistryCapacity()
        {
            TacticalV2Config config = TacticalV2Config.Default();

            for (int seed = 0; seed < 40; seed++)
            {
                var env = new TacticalV2DuelEnv(config);
                var view = env.Reset(seed, new GreedyAgent(seed), new GreedyAgent(seed + 1));
                Assert.That(view.Terminated || view.Truncated, Is.True);
            }
        }

        private static int PickLegal(bool[] mask, Random rng)
        {
            var legal = new List<int>();
            for (int i = 0; i < mask.Length; i++) if (mask[i]) legal.Add(i);
            return legal[rng.Next(legal.Count)];
        }

        /// <summary>Asserts the drained transitions are exactly the accepted commands in order — same
        /// count and same commands (record equality) as the replay log — and that consecutive
        /// transitions chain by reference (transition[i].Resulting IS transition[i+1].Previous), so a
        /// consumer can play them back as one continuous sequence.</summary>
        private static void AssertTransitionsMatchReplay(TacticalV2DuelEnv env, IReadOnlyList<DuelTransition> transitions)
        {
            ReplayData data = ReplayFile.Read(env.ToReplay());
            Assert.That(transitions.Count, Is.EqualTo(data.Commands.Count));
            for (int i = 0; i < transitions.Count; i++)
                Assert.That(transitions[i].Command, Is.EqualTo(data.Commands[i]));
            for (int i = 0; i < transitions.Count - 1; i++)
                Assert.That(transitions[i].Resulting, Is.SameAs(transitions[i + 1].Previous));
            if (transitions.Count > 0)
                Assert.That(transitions[transitions.Count - 1].Resulting, Is.SameAs(env.State));
        }

        [Test]
        public void TwoInternalGreedyControllers_ProduceOneOrderedTransitionPerAcceptedCommand()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            var view = env.Reset(3, new GreedyAgent(3), new GreedyAgent(4));
            Assert.That(view.Terminated || view.Truncated, Is.True);

            IReadOnlyList<DuelTransition> transitions = env.DrainTransitions();
            Assert.That(transitions, Is.Not.Empty);
            AssertTransitionsMatchReplay(env, transitions);
        }

        [Test]
        public void ExternalAndInternalSteps_ProduceOneOrderedTransitionPerAcceptedCommand()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            var rng = new Random(11);
            var view = env.Reset(11, null, new GreedyAgent(3)); // seat0 external, seat1 internal

            int steps = 0;
            while (!view.Terminated && !view.Truncated && steps < 4000)
            {
                view = env.Step(PickLegal(view.ActionMask, rng));
                steps++;
            }
            Assert.That(view.Terminated || view.Truncated, Is.True);

            IReadOnlyList<DuelTransition> transitions = env.DrainTransitions();
            Assert.That(transitions, Is.Not.Empty);
            AssertTransitionsMatchReplay(env, transitions);
        }

        /// <summary>Across a scripted-vs-scripted episode segment, at least one accepted transition must
        /// be an AttackUnit — the viewer needs these specifically to drive projectile/impact/kill
        /// presentation instead of inferring an attack from a coarse before/after diff.</summary>
        [Test]
        public void ScriptedVsScripted_AttackCommandsAppearAsAttackUnitTransitions()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            bool foundAttack = false;

            for (int seed = 0; seed < 20 && !foundAttack; seed++)
            {
                var env = new TacticalV2DuelEnv(config);
                env.Reset(seed, new GreedyAgent(seed), new GreedyAgent(seed + 1));
                IReadOnlyList<DuelTransition> transitions = env.DrainTransitions();
                if (transitions.Any(t => t.Command is AttackUnit)) foundAttack = true;
            }

            Assert.That(foundAttack, Is.True, "expected at least one AttackUnit transition across seeds 0..19");
        }

        /// <summary>A masked-off move destination — decoded into a real MoveUnit for a slot that holds a
        /// living unit at the very start of the episode, so it can't degrade to the always-legal EndTurn
        /// fallback — is rejected by the engine and must not produce a transition or change state.</summary>
        [Test]
        public void RejectedExternalAction_ProducesNoTransition()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            var view = env.Reset(7, null, null); // both seats external; nothing auto-plays

            TacticalV2Layout layout = env.Layout;
            int n = layout.CellCount;
            int rejectedAction = -1;
            for (int i = layout.MoveOffset; i < layout.AttackOffset; i++)
            {
                if (!view.ActionMask[i]) { rejectedAction = i; break; }
            }
            Assert.That(rejectedAction, Is.GreaterThanOrEqualTo(0),
                "expected at least one masked-off move destination at the start of the episode");

            GameState before = env.State;
            env.Step(rejectedAction);

            Assert.That(env.State, Is.SameAs(before), "a rejected command must not change engine state");
            Assert.That(env.DrainTransitions(), Is.Empty);
        }

        [Test]
        public void Reset_ClearsPendingTransitions()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            env.Reset(3, new GreedyAgent(3), new GreedyAgent(4)); // plays to completion; NOT drained

            env.Reset(9, new GreedyAgent(9), new GreedyAgent(10)); // reset again without draining

            IReadOnlyList<DuelTransition> transitions = env.DrainTransitions();
            AssertTransitionsMatchReplay(env, transitions); // must reflect only the second game
        }

        [Test]
        public void DrainTransitions_EmptiesTheQueue()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            env.Reset(3, new GreedyAgent(3), new GreedyAgent(4));

            Assert.That(env.DrainTransitions(), Is.Not.Empty);
            Assert.That(env.DrainTransitions(), Is.Empty);
        }
    }
}

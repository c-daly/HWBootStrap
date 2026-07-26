using System;
using System.Collections.Generic;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class DuelEnvTests
    {
        private static int PickLegal(bool[] mask, Random rng)
        {
            var legal = new List<int>();
            for (int i = 0; i < mask.Length; i++) if (mask[i]) legal.Add(i);
            return legal[rng.Next(legal.Count)];
        }

        [Test]
        public void Duel_TwoExternalControllers_PlayToEnd_AndRecordReplays()
        {
            var env = new DuelEnv();
            var rng = new Random(5);

            var view = env.Reset(3, null, null); // both external
            Assert.That(view.Observation.Length, Is.EqualTo(env.ObservationLength));
            Assert.That(view.ActionMask.Length, Is.EqualTo(env.ActionCount));

            int steps = 0;
            while (!view.Terminated && !view.Truncated && steps < 4000)
            {
                view = env.Step(PickLegal(view.ActionMask, rng));
                steps++;
            }
            Assert.That(view.Terminated || view.Truncated, Is.True);

            // the recorded duel reconstructs to the same terminal state
            var data = ReplayFile.Read(env.ToReplay());
            var replay = new Replay(data.Start, data.Commands);
            Assert.That(replay.Final.IsGameOver, Is.EqualTo(env.State.IsGameOver));
            Assert.That(replay.Final.Winner, Is.EqualTo(env.State.Winner));
        }

        [Test]
        public void Duel_ExternalVsInternalAgent_OnlyExposesTheExternalSeat()
        {
            var env = new DuelEnv();
            var rng = new Random(9);

            // seat 0 external, seat 1 played internally by Greedy
            var view = env.Reset(4, null, new GreedyAgent(2));
            int steps = 0;
            while (!view.Terminated && !view.Truncated && steps < 4000)
            {
                Assert.That(view.Seat, Is.EqualTo(0), "only the external seat (0) should be exposed");
                view = env.Step(PickLegal(view.ActionMask, rng));
                steps++;
            }
            Assert.That(view.Terminated || view.Truncated, Is.True);
        }

        /// <summary>Capture smoke test (see TacticalV2DuelEnvTests for the thorough coverage): the
        /// drained transitions — from both the external Step path and the internal auto-play loop — are
        /// exactly the accepted commands in order, matching the replay log, and chain by reference.</summary>
        [Test]
        public void DrainTransitions_MatchesReplayLog_WithExternalAndInternalSeats()
        {
            var env = new DuelEnv();
            var rng = new Random(31);

            var view = env.Reset(31, null, new GreedyAgent(6)); // seat0 external, seat1 internal
            int steps = 0;
            while (!view.Terminated && !view.Truncated && steps < 4000)
            {
                view = env.Step(PickLegal(view.ActionMask, rng));
                steps++;
            }
            Assert.That(view.Terminated || view.Truncated, Is.True);

            var transitions = env.DrainTransitions();
            Assert.That(transitions, Is.Not.Empty);
            var data = ReplayFile.Read(env.ToReplay());
            Assert.That(transitions.Count, Is.EqualTo(data.Commands.Count));
            for (int i = 0; i < transitions.Count; i++)
                Assert.That(transitions[i].Command, Is.EqualTo(data.Commands[i]));
            for (int i = 0; i < transitions.Count - 1; i++)
                Assert.That(transitions[i].Resulting, Is.SameAs(transitions[i + 1].Previous));
            Assert.That(transitions[transitions.Count - 1].Resulting, Is.SameAs(env.State));

            Assert.That(env.DrainTransitions(), Is.Empty, "drain must empty the queue");
        }
    }
}

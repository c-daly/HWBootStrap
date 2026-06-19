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
    }
}

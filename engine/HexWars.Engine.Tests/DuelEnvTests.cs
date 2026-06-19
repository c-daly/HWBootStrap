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
        public void Duel_TwoControllers_PlayToEnd_AndRecordReplays()
        {
            var env = new DuelEnv();
            var rng = new Random(5);

            var view = env.Reset(3);
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
    }
}

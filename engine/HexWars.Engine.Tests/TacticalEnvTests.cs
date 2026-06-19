using System;
using System.Collections.Generic;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalEnvTests
    {
        private static int LastLegal(bool[] mask)
        {
            for (int i = mask.Length - 1; i >= 0; i--) if (mask[i]) return i;
            return 0;
        }

        [Test]
        public void Reset_GivesSizedObservationAndMask_WithEndTurnLegal()
        {
            var env = new TacticalEnv(s => new RandomAgent(s));
            var obs = env.Reset(1);
            var mask = env.LegalActionMask();

            Assert.That(obs.Length, Is.EqualTo(env.ObservationLength));
            Assert.That(mask.Length, Is.EqualTo(env.ActionCount));
            Assert.That(mask[0], Is.True, "EndTurn must always be legal");

            int legal = 0;
            foreach (var b in mask) if (b) legal++;
            Assert.That(legal, Is.GreaterThan(1), "units should have some legal moves at the start");
        }

        [Test]
        public void Mask_CountsEveryLegalMoveAndAttack_PlusEndTurn()
        {
            var env = new TacticalEnv(s => new RandomAgent(s));
            env.Reset(5);

            int moveAttack = 0;
            foreach (var c in LegalMoves.For(env.State))
                if (c is MoveUnit || c is AttackUnit) moveAttack++;

            int maskCount = 0;
            foreach (var b in env.LegalActionMask()) if (b) maskCount++;

            Assert.That(maskCount, Is.EqualTo(moveAttack + 1)); // every move/attack encodes uniquely, + EndTurn
        }

        [Test]
        public void Episode_PlayedWithMaskedActions_Ends()
        {
            var env = new TacticalEnv(s => new RandomAgent(s));
            env.Reset(2);

            var rng = new Random(123);
            bool done = false;
            StepResult sr = default;
            for (int t = 0; t < 3000 && !done; t++)
            {
                var mask = env.LegalActionMask();
                var legal = new List<int>();
                for (int i = 0; i < mask.Length; i++) if (mask[i]) legal.Add(i);
                sr = env.Step(legal[rng.Next(legal.Count)]);
                done = sr.Terminated || sr.Truncated;
            }

            Assert.That(done, Is.True);
            Assert.That(sr.Observation.Length, Is.EqualTo(env.ObservationLength));
        }

        [Test]
        public void SameSeedAndActions_AreReproducible()
        {
            float a = PlayDeterministic(9, out var obsA);
            float b = PlayDeterministic(9, out var obsB);
            Assert.That(b, Is.EqualTo(a).Within(1e-4));
            Assert.That(obsB, Is.EqualTo(obsA));
        }

        private static float PlayDeterministic(int seed, out float[] finalObs)
        {
            var env = new TacticalEnv(s => new RandomAgent(s));
            finalObs = env.Reset(seed);
            float total = 0f;
            for (int t = 0; t < 3000; t++)
            {
                var sr = env.Step(LastLegal(env.LegalActionMask()));
                total += sr.Reward;
                finalObs = sr.Observation;
                if (sr.Terminated || sr.Truncated) break;
            }
            return total;
        }
    }
}

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
    }
}

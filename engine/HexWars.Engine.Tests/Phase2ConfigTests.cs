using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class Phase2ConfigTests
    {
        [Test]
        public void Default_HasPhase2Factors()
        {
            var c = GameConfig.Default();
            Assert.That(c.UpkeepFactor, Is.EqualTo(0.25));
            Assert.That(c.CaptureFactor, Is.EqualTo(4.0));
            Assert.That(c.BuildFactor, Is.EqualTo(4.0));
        }

        [Test]
        public void Default_PassesThroughFactors()
        {
            var c = GameConfig.Default(upkeepFactor: 0.5, captureFactor: 6.0, buildFactor: 3.0);
            Assert.That(c.UpkeepFactor, Is.EqualTo(0.5));
            Assert.That(c.CaptureFactor, Is.EqualTo(6.0));
            Assert.That(c.BuildFactor, Is.EqualTo(3.0));
        }
    }
}

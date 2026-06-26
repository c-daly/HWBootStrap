using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TerritoryConfigTests
    {
        [Test]
        public void Default_IsAnnihilationOnly_WithControlDefaults()
        {
            var c = GameConfig.Default();
            Assert.That(c.WinConditions, Is.EqualTo(WinBy.Annihilation));
            Assert.That(c.CaptureCost, Is.EqualTo(3));
            Assert.That(c.EconomyWinThreshold, Is.EqualTo(200));
            Assert.That(c.ScoreKills, Is.EqualTo(1));
        }

        [Test]
        public void Default_PassesThroughWinConditions()
        {
            var c = GameConfig.Default(winConditions: WinBy.Economy | WinBy.Score, captureCost: 5);
            Assert.That(c.WinConditions, Is.EqualTo(WinBy.Economy | WinBy.Score));
            Assert.That(c.CaptureCost, Is.EqualTo(5));
        }
    }
}

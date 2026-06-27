using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class Phase3ConfigTests
    {
        [Test]
        public void Default_TerritoryDefaultsOff_ClaimEndsTurnOn()
        {
            var c = GameConfig.Default();
            Assert.That(c.TerritoryMode, Is.False);
            Assert.That(c.ClaimEndsTurn, Is.True);
        }

        [Test]
        public void Default_PassesThroughTerritoryFlagsAndStartingPoints()
        {
            var c = GameConfig.Default(startingPoints: 40, territoryMode: true, claimEndsTurn: false);
            Assert.That(c.StartingPoints, Is.EqualTo(40));
            Assert.That(c.TerritoryMode, Is.True);
            Assert.That(c.ClaimEndsTurn, Is.False);
        }
    }
}

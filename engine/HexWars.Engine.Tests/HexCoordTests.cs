using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class HexCoordTests
    {
        [Test]
        public void Distance_BetweenAdjacentHexes_IsOne()
        {
            var origin = new HexCoord(0, 0);
            var neighbor = new HexCoord(1, 0);

            Assert.That(HexCoord.Distance(origin, neighbor), Is.EqualTo(1));
        }
    }
}

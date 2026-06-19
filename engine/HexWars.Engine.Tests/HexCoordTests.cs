using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class HexCoordTests
    {
        [Test]
        public void Distance_BetweenAdjacentHexes_IsOne()
        {
            Assert.That(HexCoord.Distance(new HexCoord(0, 0), new HexCoord(1, 0)), Is.EqualTo(1));
        }

        [Test]
        public void Distance_AcrossCubeAxes_UsesCubeMetric()
        {
            // (0,0) -> (2,-1): cube delta (dq=2, dr=-1, ds=-1) -> (2+1+1)/2 = 2
            Assert.That(HexCoord.Distance(new HexCoord(0, 0), new HexCoord(2, -1)), Is.EqualTo(2));
        }

        [Test]
        public void Equality_SameQR_AreEqualAndShareHashCode()
        {
            var a = new HexCoord(2, -3);
            var b = new HexCoord(2, -3);
            Assert.That(a, Is.EqualTo(b));
            Assert.That(a == b, Is.True);
            Assert.That(a.GetHashCode(), Is.EqualTo(b.GetHashCode()));
        }

        [Test]
        public void Equality_DifferentQR_AreNotEqual()
        {
            Assert.That(new HexCoord(1, 0) != new HexCoord(0, 1), Is.True);
        }

        [Test]
        public void Neighbors_AreTheSixDistinctAdjacentCoords()
        {
            var origin = new HexCoord(0, 0);
            var neighbors = origin.Neighbors();

            Assert.That(neighbors.Count, Is.EqualTo(6));
            foreach (var n in neighbors)
                Assert.That(HexCoord.Distance(origin, n), Is.EqualTo(1));
            Assert.That(new HashSet<HexCoord>(neighbors).Count, Is.EqualTo(6));
        }
    }
}

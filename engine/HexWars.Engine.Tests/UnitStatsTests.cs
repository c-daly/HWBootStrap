using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class UnitStatsTests
    {
        [Test]
        public void PointCost_IsTheSumOfAllNineStats()
        {
            var stats = new UnitStats(
                health: 3, damage: 2, defense: 1,
                movement: 4, verticalMovement: 5,
                range: 2, rangeArc: 1,
                vision: 3, visionArc: 2);

            Assert.That(stats.PointCost, Is.EqualTo(23)); // 3+2+1+4+5+2+1+3+2
        }

        [Test]
        public void PointCost_OfAllZeroStats_IsZero()
        {
            var stats = new UnitStats(0, 0, 0, 0, 0, 0, 0, 0, 0);
            Assert.That(stats.PointCost, Is.EqualTo(0));
        }
    }
}

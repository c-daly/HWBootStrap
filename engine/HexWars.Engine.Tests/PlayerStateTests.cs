using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class PlayerStateTests
    {
        [Test]
        public void New_HasPointsAndEmptyCollections()
        {
            var p = new PlayerState(PlayerId.Player0, points: 12);

            Assert.That(p.Id, Is.EqualTo(PlayerId.Player0));
            Assert.That(p.Points, Is.EqualTo(12));
            Assert.That(p.Reserve, Is.Empty);
            Assert.That(p.UnitsOnBoard, Is.Empty);
            Assert.That(p.Generators, Is.Empty);
        }

        [Test]
        public void WithPoints_ReturnsCopy_OriginalUnchanged()
        {
            var p = new PlayerState(PlayerId.Player0, points: 12);
            var richer = p.WithPoints(20);

            Assert.That(richer.Points, Is.EqualTo(20));
            Assert.That(p.Points, Is.EqualTo(12));
        }
    }
}

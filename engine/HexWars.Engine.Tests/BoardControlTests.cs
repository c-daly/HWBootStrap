using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BoardControlTests
    {
        static Board OneTile() => new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });

        [Test]
        public void Hexes_StartNeutral()
        {
            var b = OneTile();
            Assert.That(b.Controller(new HexCoord(0, 0)), Is.Null);
            Assert.That(b.ControlledCount(PlayerId.Player0), Is.EqualTo(0));
        }

        [Test]
        public void WithControl_SetsController_AndIsImmutable()
        {
            var b = OneTile();
            var b2 = b.WithControl(new HexCoord(0, 0), PlayerId.Player1);

            Assert.That(b2.Controller(new HexCoord(0, 0)), Is.EqualTo(PlayerId.Player1));
            Assert.That(b2.ControlledCount(PlayerId.Player1), Is.EqualTo(1));
            Assert.That(b.Controller(new HexCoord(0, 0)), Is.Null, "original board must be unchanged");
        }
    }
}

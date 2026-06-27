using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BoardSeedControlTests
    {
        [Test]
        public void WithControl_Batch_SeedsAllHexes_LeavesOriginalUnchanged()
        {
            var a = new HexCoord(0, 0);
            var b = new HexCoord(1, 0);
            var board = new Board(new[] { new Tile(a, 0, TerrainType.Plains), new Tile(b, 0, TerrainType.Plains) });

            var seeded = board.WithControl(new List<HexCoord> { a, b }, PlayerId.Player1);

            Assert.That(seeded.Controller(a), Is.EqualTo(PlayerId.Player1));
            Assert.That(seeded.Controller(b), Is.EqualTo(PlayerId.Player1));
            Assert.That(seeded.ControlledCount(PlayerId.Player1), Is.EqualTo(2));
            Assert.That(board.Controller(a), Is.Null, "original board is unchanged");
        }
    }
}

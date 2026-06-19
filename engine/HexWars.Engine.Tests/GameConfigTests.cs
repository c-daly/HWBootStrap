using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameConfigTests
    {
        [Test]
        public void Default_Plains_IsCheapOpenGround()
        {
            var plains = GameConfig.Default().Terrain(TerrainType.Plains);
            Assert.That(plains.MoveCost, Is.EqualTo(1));
            Assert.That(plains.Concealment, Is.EqualTo(0));
            Assert.That(plains.Defense, Is.EqualTo(0));
            Assert.That(plains.Passable, Is.True);
        }

        [Test]
        public void Default_Forest_GivesCoverAndConcealment()
        {
            var forest = GameConfig.Default().Terrain(TerrainType.Forest);
            Assert.That(forest.MoveCost, Is.EqualTo(2));
            Assert.That(forest.Concealment, Is.EqualTo(2));
            Assert.That(forest.Defense, Is.EqualTo(1));
            Assert.That(forest.Passable, Is.True);
        }

        [Test]
        public void Default_Water_IsPassableButSlow()
        {
            var water = GameConfig.Default().Terrain(TerrainType.Water);
            Assert.That(water.MoveCost, Is.EqualTo(3));
            Assert.That(water.Passable, Is.True);
        }
    }
}

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

        [Test]
        public void Default_HasExpectedGameplayTunables()
        {
            var c = GameConfig.Default();
            Assert.That(c.StartingPoints, Is.EqualTo(12));
            Assert.That(c.BountyRate, Is.EqualTo(0.5));
            Assert.That(c.GeneratorCost, Is.EqualTo(2));
            Assert.That(c.GeneratorOutput, Is.EqualTo(1));
            Assert.That(c.GeneratorHealth, Is.EqualTo(3));
            Assert.That(c.DamageFloor, Is.EqualTo(0));
            Assert.That(c.DmgHighGroundBonus, Is.EqualTo(1));
            Assert.That(c.RangeHighGroundBonus, Is.EqualTo(1));
            Assert.That(c.RoundCap, Is.EqualTo(40));
            Assert.That(c.DesignFee, Is.EqualTo(0));          // design is free by default (configurable)
            Assert.That(c.DeployCostMultiplier, Is.EqualTo(1.0));
        }
    }
}

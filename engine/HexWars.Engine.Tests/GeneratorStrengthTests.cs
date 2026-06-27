using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GeneratorStrengthTests
    {
        static readonly HexCoord C = new HexCoord(0, 0);

        [Test]
        public void Strength_DefaultsToOne()
        {
            var g = new Generator(1, PlayerId.Player0, C, 0, 3);
            Assert.That(g.Strength, Is.EqualTo(1.0));
        }

        [Test]
        public void Strength_StoredAndPreservedAcrossDamageAndOwnerChange()
        {
            var g = new Generator(1, PlayerId.Player0, C, 0, 3, 0.5);
            Assert.That(g.Strength, Is.EqualTo(0.5));
            Assert.That(g.WithDamage(1).Strength, Is.EqualTo(0.5));
            var owned = g.WithOwner(PlayerId.Player1);
            Assert.That(owned.Owner, Is.EqualTo(PlayerId.Player1));
            Assert.That(owned.Strength, Is.EqualTo(0.5));
            Assert.That(owned.Cell, Is.EqualTo(C));
        }
    }
}

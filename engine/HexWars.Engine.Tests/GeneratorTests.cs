using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GeneratorTests
    {
        [Test]
        public void NewGenerator_StartsAtFullHealth_AndIsAlive()
        {
            var g = new Generator(id: 7, owner: PlayerId.Player1, cell: new HexCoord(0, 0),
                                  elevation: 1, maxHp: 3);

            Assert.That(g.CurrentHp, Is.EqualTo(3));
            Assert.That(g.IsAlive, Is.True);
            Assert.That(g.Owner, Is.EqualTo(PlayerId.Player1));
        }

        [Test]
        public void WithDamage_CanDestroyIt_AndDoesNotMutateOriginal()
        {
            var g = new Generator(7, PlayerId.Player1, new HexCoord(0, 0), 1, 3);

            var dead = g.WithDamage(5);
            Assert.That(dead.CurrentHp, Is.EqualTo(0));
            Assert.That(dead.IsAlive, Is.False);
            Assert.That(g.CurrentHp, Is.EqualTo(3));
        }
    }
}

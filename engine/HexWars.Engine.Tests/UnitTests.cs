using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class UnitTests
    {
        // Health-only stat line for terse tests.
        private static UnitStats Hp(int health) => new UnitStats(health, 0, 0, 0, 0, 0, 0, 0, 0);

        [Test]
        public void NewUnit_StartsAtFullHealth_AndIsAlive()
        {
            var u = new Unit(id: 1, owner: PlayerId.Player0, stats: Hp(5),
                             cell: new HexCoord(0, 0), elevation: 0);

            Assert.That(u.CurrentHp, Is.EqualTo(5));
            Assert.That(u.IsAlive, Is.True);
            Assert.That(u.Owner, Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void WithDamage_ReducesHp_ButNeverBelowZero()
        {
            var u = new Unit(1, PlayerId.Player0, Hp(5), new HexCoord(0, 0), 0);

            var hurt = u.WithDamage(3);
            Assert.That(hurt.CurrentHp, Is.EqualTo(2));
            Assert.That(hurt.IsAlive, Is.True);

            var dead = u.WithDamage(99);
            Assert.That(dead.CurrentHp, Is.EqualTo(0));
            Assert.That(dead.IsAlive, Is.False);

            // original is untouched (immutability)
            Assert.That(u.CurrentHp, Is.EqualTo(5));
        }

        [Test]
        public void WithCell_MovesTheUnit_KeepingHp()
        {
            var u = new Unit(1, PlayerId.Player0, Hp(5), new HexCoord(0, 0), 0).WithDamage(2);
            var moved = u.WithCell(new HexCoord(2, -1), elevation: 3);

            Assert.That(moved.Cell, Is.EqualTo(new HexCoord(2, -1)));
            Assert.That(moved.Elevation, Is.EqualTo(3));
            Assert.That(moved.CurrentHp, Is.EqualTo(3));
        }
    }
}

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

        [Test]
        public void DisplayName_ReturnsName_WhenSet()
        {
            var u = new Unit(1, PlayerId.Player0, Hp(5), new HexCoord(0, 0), 0, "Doom Turtle");
            Assert.That(u.DisplayName, Is.EqualTo("Doom Turtle"));
        }

        [Test]
        public void DisplayName_FallsBackToDominantRole_WhenNameEmpty()
        {
            var brute = new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1); // dominant stat: Health -> Brute
            var u = new Unit(1, PlayerId.Player0, brute, new HexCoord(0, 0), 0);
            Assert.That(u.Name, Is.EqualTo(""));
            Assert.That(u.DisplayName, Is.EqualTo("Brute"));
        }

        [Test]
        public void WithDamage_PreservesName()
        {
            var u = new Unit(1, PlayerId.Player0, Hp(5), new HexCoord(0, 0), 0, "Recon");
            Assert.That(u.WithDamage(2).Name, Is.EqualTo("Recon"));
        }

        [Test]
        public void WithCell_PreservesName()
        {
            var u = new Unit(1, PlayerId.Player0, Hp(5), new HexCoord(0, 0), 0, "Recon");
            Assert.That(u.WithCell(new HexCoord(1, 1), 0).Name, Is.EqualTo("Recon"));
        }
    }
}

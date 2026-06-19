using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class RolesTests
    {
        [Test]
        public void HighDamage_IsStriker() =>
            Assert.That(Roles.Dominant(TestStates.Stats(damage: 5)), Is.EqualTo(UnitRole.Striker));

        [Test]
        public void HighRange_IsSniper() =>
            Assert.That(Roles.Dominant(TestStates.Stats(range: 5)), Is.EqualTo(UnitRole.Sniper));

        [Test]
        public void HighVision_IsSpotter() =>
            Assert.That(Roles.Dominant(TestStates.Stats(vision: 5)), Is.EqualTo(UnitRole.Spotter));

        [Test]
        public void HighMovement_IsRunner() =>
            Assert.That(Roles.Dominant(TestStates.Stats(movement: 5)), Is.EqualTo(UnitRole.Runner));

        [Test]
        public void HighVerticalMovement_IsClimber() =>
            Assert.That(Roles.Dominant(TestStates.Stats(verticalMovement: 5)), Is.EqualTo(UnitRole.Climber));

        [Test]
        public void HighDefense_IsBulwark() =>
            Assert.That(Roles.Dominant(TestStates.Stats(defense: 5)), Is.EqualTo(UnitRole.Bulwark));

        [Test]
        public void OnlyHealth_IsBrute() =>
            Assert.That(Roles.Dominant(TestStates.Stats(health: 5)), Is.EqualTo(UnitRole.Brute));

        [Test]
        public void TieAtTop_IsGeneralist() =>
            Assert.That(Roles.Dominant(TestStates.Stats(damage: 5, range: 5)), Is.EqualTo(UnitRole.Generalist));

        [Test]
        public void BalancedBuild_IsGeneralist() =>
            Assert.That(Roles.Dominant(TestStates.Stats(damage: 2, defense: 2, movement: 2)),
                        Is.EqualTo(UnitRole.Generalist));
    }
}

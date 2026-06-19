using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class CombatResolverTests
    {
        private static GameConfig ConfigWithFloor(int floor) =>
            new GameConfig(new Dictionary<TerrainType, TerrainDef>
            {
                { TerrainType.Plains, new TerrainDef(1, 0, 0, true) },
            }, damageFloor: floor);

        [Test]
        public void ComputeDamage_IsDamageMinusDefense_OnLevelGround()
        {
            var cfg = GameConfig.Default(); // floor 0, high-ground bonus 1
            int dmg = CombatResolver.ComputeDamage(attackerDamage: 5, attackerElevation: 0,
                                                   targetElevation: 0, targetDefense: 2, config: cfg);
            Assert.That(dmg, Is.EqualTo(3));
        }

        [Test]
        public void ComputeDamage_NeverBelowDamageFloor()
        {
            Assert.That(CombatResolver.ComputeDamage(2, 0, 0, 10, ConfigWithFloor(0)), Is.EqualTo(0));
            Assert.That(CombatResolver.ComputeDamage(2, 0, 0, 10, ConfigWithFloor(1)), Is.EqualTo(1));
        }

        [Test]
        public void ComputeDamage_AddsHighGroundBonus_WhenAttackingDownhill()
        {
            var cfg = GameConfig.Default(); // bonus +1 per level
            // attacker elev 5, target elev 2 -> H = 3 -> +3 damage
            int dmg = CombatResolver.ComputeDamage(3, attackerElevation: 5, targetElevation: 2, targetDefense: 0, config: cfg);
            Assert.That(dmg, Is.EqualTo(6));
        }

        [Test]
        public void ComputeDamage_NoBonus_WhenAttackingUphill()
        {
            var cfg = GameConfig.Default();
            // attacker below target -> H = 0
            int dmg = CombatResolver.ComputeDamage(3, attackerElevation: 0, targetElevation: 3, targetDefense: 0, config: cfg);
            Assert.That(dmg, Is.EqualTo(3));
        }

        [Test]
        public void Bounty_IsFloorOfBuildCostTimesRate()
        {
            var cfg = GameConfig.Default(); // rate 0.5
            Assert.That(CombatResolver.Bounty(buildCost: 7, config: cfg), Is.EqualTo(3));  // floor(3.5)
            Assert.That(CombatResolver.Bounty(buildCost: 10, config: cfg), Is.EqualTo(5));
        }
    }
}

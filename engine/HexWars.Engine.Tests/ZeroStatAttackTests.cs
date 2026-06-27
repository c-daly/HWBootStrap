using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>A zero base stat means no capability: high-ground bonuses augment a real attack, they
    /// must not manufacture one. A Range-0 unit can't fire beyond its own hex even uphill, and a
    /// Damage-0 unit deals nothing even uphill.</summary>
    public class ZeroStatAttackTests
    {
        // attacker stats: H, D, Df, Mv, VMv, Range, RangeArc, Vision, VisionArc
        static readonly UnitStats ZeroRangeZeroDamage = new UnitStats(2, 0, 0, 4, 3, 0, 0, 5, 2);

        [Test]
        public void ZeroRange_CannotTargetAdjacent_EvenFromHighGround()
        {
            var attacker = new Unit(1, PlayerId.Player0, ZeroRangeZeroDamage, new HexCoord(0, 0), elevation: 1);
            // target one hex away, one level below (so high-ground bonus would otherwise apply)
            bool inRange = TargetingService.InRange(attacker, new HexCoord(1, 0), targetElevation: 0, GameConfig.Default());
            Assert.That(inRange, Is.False, "a Range-0 unit must not reach beyond its own hex, uphill or not");
        }

        [Test]
        public void ZeroDamage_DealsNothing_EvenFromHighGround()
        {
            int dmg = CombatResolver.ComputeDamage(attackerDamage: 0, attackerElevation: 1, targetElevation: 0,
                                                   targetDefense: 0, GameConfig.Default());
            Assert.That(dmg, Is.EqualTo(0), "a Damage-0 unit must deal 0 even with a high-ground bonus");
        }

        [Test]
        public void HighGround_StillAugmentsARealAttack()
        {
            // sanity: a Range-1 / Damage-1 unit keeps its high-ground bonuses
            var atk = new UnitStats(2, 1, 0, 4, 3, 1, 0, 5, 2);
            var unit = new Unit(1, PlayerId.Player0, atk, new HexCoord(0, 0), elevation: 1);
            Assert.That(TargetingService.InRange(unit, new HexCoord(2, 0), 0, GameConfig.Default()), Is.True,
                "Range 1 + high-ground bonus 1 reaches 2 hexes downhill");
            Assert.That(CombatResolver.ComputeDamage(1, 1, 0, 0, GameConfig.Default()), Is.EqualTo(2),
                "Damage 1 + high-ground bonus 1 = 2");
        }
    }
}

using System;

namespace HexWars.Engine
{
    /// <summary>
    /// Pure, deterministic combat math (no randomness). Damage =
    /// max(DamageFloor, attackerDamage + highGround*DmgHighGroundBonus - targetDefense), where
    /// highGround = max(0, attackerElevation - targetElevation) and targetDefense already folds in
    /// terrain defense. Bounty = floor(buildCost * BountyRate).
    /// </summary>
    public static class CombatResolver
    {
        public static int ComputeDamage(
            int attackerDamage,
            int attackerElevation,
            int targetElevation,
            int targetDefense,
            GameConfig config)
        {
            if (attackerDamage <= 0) return 0; // a non-combatant deals nothing, even from high ground

            int highGround = Math.Max(0, attackerElevation - targetElevation);
            int raw = attackerDamage + highGround * config.DmgHighGroundBonus - targetDefense;
            return Math.Max(config.DamageFloor, raw);
        }

        public static int Bounty(int buildCost, GameConfig config)
            => (int)Math.Floor(buildCost * config.BountyRate);
    }
}

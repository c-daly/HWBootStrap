namespace HexWars.Engine
{
    /// <summary>A unit's at-a-glance role, derived from its dominant stat (for icons / UI).</summary>
    public enum UnitRole
    {
        Brute,    // Health
        Striker,  // Damage
        Bulwark,  // Defense
        Runner,   // Movement
        Climber,  // Vertical Movement
        Sniper,   // Range (+ Range Arc)
        Spotter,  // Vision (+ Vision Arc)
    }

    /// <summary>Classifies a unit by its dominant stat. Pure — shared by the role icon and the
    /// hover tooltip so they never disagree.</summary>
    public static class Roles
    {
        public static UnitRole Dominant(UnitStats s)
        {
            // Ranked so earlier entries win ties (deterministic).
            (int value, UnitRole role)[] ranked =
            {
                (s.Damage, UnitRole.Striker),
                (s.Range + s.RangeArc, UnitRole.Sniper),
                (s.Vision + s.VisionArc, UnitRole.Spotter),
                (s.Movement, UnitRole.Runner),
                (s.VerticalMovement, UnitRole.Climber),
                (s.Defense, UnitRole.Bulwark),
                (s.Health, UnitRole.Brute),
            };

            var best = ranked[0];
            foreach (var entry in ranked)
                if (entry.value > best.value) best = entry;
            return best.role;
        }
    }
}

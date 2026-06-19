namespace HexWars.Engine
{
    /// <summary>A unit's at-a-glance role, derived from its dominant stat (for icons / UI).</summary>
    public enum UnitRole
    {
        Generalist, // no single dominant stat (balanced / tie)
        Brute,      // Health
        Striker,    // Damage
        Bulwark,    // Defense
        Runner,     // Movement
        Climber,    // Vertical Movement
        Sniper,     // Range (+ Range Arc)
        Spotter,    // Vision (+ Vision Arc)
    }

    /// <summary>Classifies a unit by its dominant stat. Pure — shared by the role icon and the
    /// hover tooltip so they never disagree.</summary>
    public static class Roles
    {
        public static UnitRole Dominant(UnitStats s)
        {
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

            int max = 0;
            foreach (var e in ranked)
                if (e.value > max) max = e.value;

            int atMax = 0;
            UnitRole top = UnitRole.Generalist;
            foreach (var e in ranked)
                if (e.value == max) { atMax++; top = e.role; }

            // A single clear leader gives that role; a tie at the top = no dominant stat = Generalist.
            return atMax == 1 ? top : UnitRole.Generalist;
        }
    }
}

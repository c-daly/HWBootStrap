namespace HexWars.Engine
{
    /// <summary>
    /// A unit's full stat line. Everything is bought from zero at a flat 1 point per step, on two
    /// axes for each spatial capability (horizontal + vertical). A unit with a stat at 0 simply
    /// cannot perform that capability — see the design's "you can only do what you bought" rule.
    /// </summary>
    public readonly struct UnitStats
    {
        public int Health { get; }
        public int Damage { get; }
        public int Defense { get; }

        /// <summary>Horizontal traversal budget per turn (hexes entered, paying terrain move cost).</summary>
        public int Movement { get; }

        /// <summary>Ascent budget per turn — levels it can climb (1 per level up). Descending is free.</summary>
        public int VerticalMovement { get; }

        /// <summary>Horizontal firing reach — max hex distance to a target.</summary>
        public int Range { get; }

        /// <summary>Vertical firing reach — levels above itself it can shoot. At/below its level is free.</summary>
        public int RangeArc { get; }

        /// <summary>Horizontal sight — max hex distance it can detect a target (shared army-wide).</summary>
        public int Vision { get; }

        /// <summary>Vertical sight — levels above itself it can see. At/below its level is free.</summary>
        public int VisionArc { get; }

        public UnitStats(
            int health, int damage, int defense,
            int movement, int verticalMovement,
            int range, int rangeArc,
            int vision, int visionArc)
        {
            Health = health;
            Damage = damage;
            Defense = defense;
            Movement = movement;
            VerticalMovement = verticalMovement;
            Range = range;
            RangeArc = rangeArc;
            Vision = vision;
            VisionArc = visionArc;
        }

        /// <summary>Flat point cost: the sum of every stat (1 point = 1 step).</summary>
        public int PointCost =>
            Health + Damage + Defense
            + Movement + VerticalMovement
            + Range + RangeArc
            + Vision + VisionArc;
    }
}

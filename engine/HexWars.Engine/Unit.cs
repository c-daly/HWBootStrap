namespace HexWars.Engine
{
    /// <summary>
    /// An on-board unit: its purchased <see cref="Stats"/>, owner, 3D position
    /// (<see cref="Cell"/> = q,r column + its own <see cref="Elevation"/>), and current health.
    /// Immutable — mutations return new copies, so <c>Apply</c> can fork state without side effects.
    /// </summary>
    public readonly struct Unit
    {
        public int Id { get; }
        public PlayerId Owner { get; }
        public UnitStats Stats { get; }
        public HexCoord Cell { get; }
        public int Elevation { get; }
        public int CurrentHp { get; }

        /// <summary>Create a fresh unit at full health.</summary>
        public Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation)
            : this(id, owner, stats, cell, elevation, stats.Health) { }

        private Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation, int currentHp)
        {
            Id = id;
            Owner = owner;
            Stats = stats;
            Cell = cell;
            Elevation = elevation;
            CurrentHp = currentHp;
        }

        public bool IsAlive => CurrentHp > 0;

        /// <summary>A copy with <paramref name="amount"/> damage applied (clamped at 0 HP).</summary>
        public Unit WithDamage(int amount)
        {
            int hp = CurrentHp - amount;
            if (hp < 0) hp = 0;
            return new Unit(Id, Owner, Stats, Cell, Elevation, hp);
        }

        /// <summary>A copy moved to a new 3D position, keeping current health.</summary>
        public Unit WithCell(HexCoord cell, int elevation) =>
            new Unit(Id, Owner, Stats, cell, elevation, CurrentHp);
    }
}

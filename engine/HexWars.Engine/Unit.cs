namespace HexWars.Engine
{
    /// <summary>
    /// An on-board unit: its purchased <see cref="Stats"/>, owner, 3D position
    /// (<see cref="Cell"/> = q,r column + its own <see cref="Elevation"/>), current health, and the
    /// <see cref="Name"/> copied from the barracks template it was deployed from (empty for units
    /// seeded directly, e.g. the starting army). Immutable — mutations return new copies, so
    /// <c>Apply</c> can fork state without side effects.
    /// </summary>
    public readonly struct Unit
    {
        public int Id { get; }
        public PlayerId Owner { get; }
        public UnitStats Stats { get; }
        public HexCoord Cell { get; }
        public int Elevation { get; }
        public int CurrentHp { get; }
        public string Name { get; }

        /// <summary>Create a fresh unit at full health. <paramref name="name"/> defaults to "" (no
        /// template name) — see <see cref="DisplayName"/> for the fallback shown to a player.</summary>
        public Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation, string name = "")
            : this(id, owner, stats, cell, elevation, stats.Health, name) { }

        private Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation, int currentHp, string name)
        {
            Id = id;
            Owner = owner;
            Stats = stats;
            Cell = cell;
            Elevation = elevation;
            CurrentHp = currentHp;
            Name = name;
        }

        public bool IsAlive => CurrentHp > 0;

        /// <summary>The name to show a player: the template Name if it has one, else the dominant-role
        /// label (so an unnamed unit still reads as something, never a blank).</summary>
        public string DisplayName => string.IsNullOrEmpty(Name) ? Roles.Dominant(Stats).ToString() : Name;

        /// <summary>A copy with <paramref name="amount"/> damage applied (clamped at 0 HP).</summary>
        public Unit WithDamage(int amount)
        {
            int hp = CurrentHp - amount;
            if (hp < 0) hp = 0;
            return new Unit(Id, Owner, Stats, Cell, Elevation, hp, Name);
        }

        /// <summary>A copy moved to a new 3D position, keeping current health.</summary>
        public Unit WithCell(HexCoord cell, int elevation) =>
            new Unit(Id, Owner, Stats, cell, elevation, CurrentHp, Name);
    }
}

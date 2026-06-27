namespace HexWars.Engine
{
    /// <summary>
    /// An on-board, attackable income structure. Sits on a column (3D position = <see cref="Cell"/>
    /// + <see cref="Elevation"/>) and has its own health, so the enemy can raid it. Per-turn income
    /// is a single configured value (see <c>GameConfig</c>), not stored per generator. Immutable.
    /// </summary>
    public readonly struct Generator
    {
        public int Id { get; }
        public PlayerId Owner { get; }
        public HexCoord Cell { get; }
        public int Elevation { get; }
        public int CurrentHp { get; }
        public double Strength { get; }

        public Generator(int id, PlayerId owner, HexCoord cell, int elevation, int maxHp, double strength = 1.0)
            : this(id, owner, cell, elevation, maxHp, maxHp, strength) { }

        private Generator(int id, PlayerId owner, HexCoord cell, int elevation, int maxHp, int currentHp, double strength)
        {
            Id = id;
            Owner = owner;
            Cell = cell;
            Elevation = elevation;
            CurrentHp = currentHp;
            Strength = strength;
        }

        public bool IsAlive => CurrentHp > 0;

        public Generator WithDamage(int amount)
        {
            int hp = CurrentHp - amount;
            if (hp < 0) hp = 0;
            return new Generator(Id, Owner, Cell, Elevation, hp, hp, Strength);
        }

        /// <summary>The same generator re-owned by <paramref name="owner"/> (used when a hex is captured).</summary>
        public Generator WithOwner(PlayerId owner) =>
            new Generator(Id, owner, Cell, Elevation, CurrentHp, CurrentHp, Strength);
    }
}

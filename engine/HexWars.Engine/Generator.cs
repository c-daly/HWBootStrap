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

        public Generator(int id, PlayerId owner, HexCoord cell, int elevation, int maxHp)
            : this(id, owner, cell, elevation, maxHp, maxHp) { }

        private Generator(int id, PlayerId owner, HexCoord cell, int elevation, int maxHp, int currentHp)
        {
            Id = id;
            Owner = owner;
            Cell = cell;
            Elevation = elevation;
            CurrentHp = currentHp;
        }

        public bool IsAlive => CurrentHp > 0;

        public Generator WithDamage(int amount)
        {
            int hp = CurrentHp - amount;
            if (hp < 0) hp = 0;
            return new Generator(Id, Owner, Cell, Elevation, hp, hp);
        }
    }
}

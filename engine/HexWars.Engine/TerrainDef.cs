namespace HexWars.Engine
{
    /// <summary>Data-driven modifiers for one <see cref="TerrainType"/>. Terrain affects movement,
    /// sight (concealment), and defense. Adding a terrain type is editing this table, not code.</summary>
    public readonly struct TerrainDef
    {
        /// <summary>Movement points to enter a hex of this terrain (drawn from a unit's Movement).</summary>
        public int MoveCost { get; }

        /// <summary>Added to the horizontal distance required to see a unit standing here.</summary>
        public int Concealment { get; }

        /// <summary>Added to a unit's Defense while it stands on this terrain.</summary>
        public int Defense { get; }

        /// <summary>Whether units may enter this terrain at all.</summary>
        public bool Passable { get; }

        public TerrainDef(int moveCost, int concealment, int defense, bool passable)
        {
            MoveCost = moveCost;
            Concealment = concealment;
            Defense = defense;
            Passable = passable;
        }
    }
}

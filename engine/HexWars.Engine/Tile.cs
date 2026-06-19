namespace HexWars.Engine
{
    /// <summary>One hex column: its <see cref="Coord"/> (q,r), its ground <see cref="Elevation"/>
    /// (the standable top of the solid pillar), and its <see cref="Terrain"/>.</summary>
    public readonly struct Tile
    {
        public HexCoord Coord { get; }
        public int Elevation { get; }
        public TerrainType Terrain { get; }

        public Tile(HexCoord coord, int elevation, TerrainType terrain)
        {
            Coord = coord;
            Elevation = elevation;
            Terrain = terrain;
        }
    }
}

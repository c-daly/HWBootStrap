using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// The hex grid: a set of <see cref="Tile"/> columns addressed by <see cref="HexCoord"/> (q,r).
    /// Deployment zones and adjacency are added as the rules that consume them are built (TDD).
    /// </summary>
    public sealed class Board
    {
        private readonly Dictionary<HexCoord, Tile> _tiles;

        public Board(IEnumerable<Tile> tiles)
        {
            _tiles = new Dictionary<HexCoord, Tile>();
            foreach (var tile in tiles)
                _tiles[tile.Coord] = tile;
        }

        public int TileCount => _tiles.Count;

        public bool Contains(HexCoord coord) => _tiles.ContainsKey(coord);

        public Tile TileAt(HexCoord coord) => _tiles[coord];
    }
}

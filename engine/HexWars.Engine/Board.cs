using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// The hex grid: a set of <see cref="Tile"/> columns addressed by <see cref="HexCoord"/> (q,r),
    /// plus each player's deployment zone (the columns where they may deploy reserved units).
    /// </summary>
    public sealed class Board
    {
        private static readonly HashSet<HexCoord> Empty = new HashSet<HexCoord>();

        private readonly Dictionary<HexCoord, Tile> _tiles;
        private readonly HashSet<HexCoord> _zone0;
        private readonly HashSet<HexCoord> _zone1;

        public Board(
            IEnumerable<Tile> tiles,
            IReadOnlyCollection<HexCoord>? zone0 = null,
            IReadOnlyCollection<HexCoord>? zone1 = null)
        {
            _tiles = new Dictionary<HexCoord, Tile>();
            foreach (var tile in tiles)
                _tiles[tile.Coord] = tile;

            _zone0 = zone0 != null ? new HashSet<HexCoord>(zone0) : Empty;
            _zone1 = zone1 != null ? new HashSet<HexCoord>(zone1) : Empty;
        }

        public int TileCount => _tiles.Count;

        public bool Contains(HexCoord coord) => _tiles.ContainsKey(coord);

        public Tile TileAt(HexCoord coord) => _tiles[coord];

        public IReadOnlyCollection<HexCoord> DeploymentZone(PlayerId player) =>
            player == PlayerId.Player0 ? _zone0 : _zone1;

        public bool IsInDeploymentZone(PlayerId player, HexCoord coord) =>
            (player == PlayerId.Player0 ? _zone0 : _zone1).Contains(coord);
    }
}

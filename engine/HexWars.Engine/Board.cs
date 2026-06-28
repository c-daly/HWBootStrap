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
        private readonly Dictionary<HexCoord, PlayerId> _control;

        public Board(
            IEnumerable<Tile> tiles,
            IReadOnlyCollection<HexCoord>? zone0 = null,
            IReadOnlyCollection<HexCoord>? zone1 = null,
            IReadOnlyDictionary<HexCoord, PlayerId>? control = null)
        {
            _tiles = new Dictionary<HexCoord, Tile>();
            foreach (var tile in tiles)
                _tiles[tile.Coord] = tile;

            _zone0 = zone0 != null ? new HashSet<HexCoord>(zone0) : Empty;
            _zone1 = zone1 != null ? new HashSet<HexCoord>(zone1) : Empty;
            _control = control != null
                ? new Dictionary<HexCoord, PlayerId>(control)
                : new Dictionary<HexCoord, PlayerId>();
        }

        public int TileCount => _tiles.Count;

        /// <summary>All tiles on the board (for renderers / iteration).</summary>
        public IReadOnlyCollection<Tile> Tiles => _tiles.Values;

        public bool Contains(HexCoord coord) => _tiles.ContainsKey(coord);

        public Tile TileAt(HexCoord coord) => _tiles[coord];

        public IReadOnlyCollection<HexCoord> DeploymentZone(PlayerId player) =>
            player == PlayerId.Player0 ? _zone0 : _zone1;

        public bool IsInDeploymentZone(PlayerId player, HexCoord coord) =>
            (player == PlayerId.Player0 ? _zone0 : _zone1).Contains(coord);

        /// <summary>Who currently controls this hex, or null if neutral.</summary>
        public PlayerId? Controller(HexCoord coord) =>
            _control.TryGetValue(coord, out var p) ? p : (PlayerId?)null;

        /// <summary>How many hexes the player controls.</summary>
        public int ControlledCount(PlayerId player)
        {
            int n = 0;
            foreach (var kv in _control) if (kv.Value == player) n++;
            return n;
        }

        /// <summary>A new board identical to this one but with <paramref name="coord"/> controlled by
        /// <paramref name="owner"/>. Immutable — this board is unchanged.</summary>
        public Board WithControl(HexCoord coord, PlayerId owner)
        {
            var control = new Dictionary<HexCoord, PlayerId>(_control) { [coord] = owner };
            return new Board(_tiles.Values, _zone0, _zone1, control);
        }

        /// <summary>A new board with every hex in <paramref name="coords"/> controlled by
        /// <paramref name="owner"/>. Immutable — this board is unchanged. (Used to seed home zones.)</summary>
        public Board WithControl(System.Collections.Generic.IEnumerable<HexCoord> coords, PlayerId owner)
        {
            var control = new Dictionary<HexCoord, PlayerId>(_control);
            foreach (var c in coords) control[c] = owner;
            return new Board(_tiles.Values, _zone0, _zone1, control);
        }
    }
}

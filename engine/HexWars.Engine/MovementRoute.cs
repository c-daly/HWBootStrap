using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>An immutable legal route and the movement budget it consumes and leaves.</summary>
    public sealed class MovementRoute
    {
        private readonly IReadOnlyList<HexCoord> _cells;
        public IReadOnlyList<HexCoord> Cells => _cells;
        public int HorizontalCost { get; }
        public int VerticalCost { get; }
        public int HorizontalRemaining { get; }
        public int VerticalRemaining { get; }

        public MovementRoute(IEnumerable<HexCoord> cells, int horizontalCost, int verticalCost,
                             int horizontalRemaining, int verticalRemaining)
        {
            if (cells == null) throw new ArgumentNullException(nameof(cells));
            var snapshot = new List<HexCoord>(cells).ToArray();
            _cells = Array.AsReadOnly(snapshot);
            if (_cells.Count < 2)
                throw new ArgumentException("A route requires an origin and destination.", nameof(cells));
            HorizontalCost = horizontalCost;
            VerticalCost = verticalCost;
            HorizontalRemaining = horizontalRemaining;
            VerticalRemaining = verticalRemaining;
        }
    }
}

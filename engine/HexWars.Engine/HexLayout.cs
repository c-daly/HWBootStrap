namespace HexWars.Engine
{
    /// <summary>
    /// Pure axial→world layout for flat-top hexes (Unity-agnostic: returns plain x/z doubles). The
    /// presentation layer maps a column's (q,r) to a world position with this, and stacks the hex
    /// prefab by elevation for the y axis. Kept here so it's deterministic and unit-tested.
    /// </summary>
    public static class HexLayout
    {
        /// <summary>World (x, z) of a column centre for a flat-top hex grid with the given size
        /// (centre-to-corner). Elevation maps to y separately (stacked columns).</summary>
        public static (double x, double z) ToWorld(HexCoord coord, double hexSize)
        {
            double x = hexSize * 1.5 * coord.Q;
            double z = hexSize * System.Math.Sqrt(3.0) * (coord.R + coord.Q / 2.0);
            return (x, z);
        }

        /// <summary>
        /// Convert a rectangular grid position (col, row) to its axial <see cref="HexCoord"/> for a
        /// flat-top "odd-q" offset layout. Used to generate a board with a rectangular footprint
        /// (alternate columns nudged by half) instead of a sheared parallelogram, while keeping all
        /// engine math (neighbours, distance) on the resulting axial coords.
        /// </summary>
        public static HexCoord OffsetToAxial(int col, int row)
        {
            int q = col;
            int r = row - (col - (col & 1)) / 2;
            return new HexCoord(q, r);
        }
    }
}

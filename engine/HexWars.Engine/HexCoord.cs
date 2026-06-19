namespace HexWars.Engine
{
    /// <summary>
    /// Axial hex coordinate (Q, R) plus an integer <see cref="Elevation"/>. The cube third axis
    /// is S = -Q - R. <see cref="Distance"/> is the horizontal hex distance and ignores elevation
    /// (vertical reach is bought separately — see the design's "uniform 3D pricing" principle).
    /// </summary>
    public readonly struct HexCoord
    {
        public int Q { get; }
        public int R { get; }
        public int Elevation { get; }

        /// <summary>Cube third axis, derived from Q and R.</summary>
        public int S => -Q - R;

        public HexCoord(int q, int r, int elevation = 0)
        {
            Q = q;
            R = r;
            Elevation = elevation;
        }

        /// <summary>Horizontal hex distance between two coordinates (ignores elevation).</summary>
        public static int Distance(HexCoord a, HexCoord b)
        {
            int dq = a.Q - b.Q;
            int dr = a.R - b.R;
            int ds = a.S - b.S;
            return (System.Math.Abs(dq) + System.Math.Abs(dr) + System.Math.Abs(ds)) / 2;
        }
    }
}

namespace HexWars.Engine
{
    /// <summary>
    /// Line-of-sight over the hex grid with elevation. The sightline runs from one column top to
    /// another; an intervening column blocks it if its top rises above the straight line at that
    /// point. Used for vision (a spotter can't see through a stack) and direct fire (a projectile
    /// can't pass through one). Indirect/arcing fire bypasses this — see <see cref="TargetingService"/>.
    /// </summary>
    public static class LineOfSight
    {
        public static bool IsClear(Board board, HexCoord from, int fromElev, HexCoord to, int toElev)
        {
            int n = HexCoord.Distance(from, to);
            if (n <= 1) return true; // adjacent or same column: nothing in between

            for (int i = 1; i < n; i++)
            {
                float t = (float)i / n;
                var cell = LerpRound(from, to, t);
                if (cell == from || cell == to || !board.Contains(cell)) continue;

                float lineHeight = fromElev + (toElev - fromElev) * t;
                if (board.TileAt(cell).Elevation > lineHeight + 1e-4f)
                    return false; // column top pokes above the sightline
            }
            return true;
        }

        private static HexCoord LerpRound(HexCoord a, HexCoord b, float t)
        {
            float q = a.Q + (b.Q - a.Q) * t;
            float r = a.R + (b.R - a.R) * t;
            float s = a.S + (b.S - a.S) * t;

            int rq = (int)System.Math.Round(q);
            int rr = (int)System.Math.Round(r);
            int rs = (int)System.Math.Round(s);
            double dq = System.Math.Abs(rq - q), dr = System.Math.Abs(rr - r), ds = System.Math.Abs(rs - s);
            if (dq > dr && dq > ds) rq = -rr - rs;
            else if (dr > ds) rr = -rq - rs;
            else rs = -rq - rr;
            return new HexCoord(rq, rr);
        }
    }
}

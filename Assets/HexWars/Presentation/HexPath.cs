using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>The straight hex-line between two cells (cube lerp + round), endpoints inclusive.
    /// Presentation-only: used to draw movement tweens and attack tracers. The engine's own
    /// line walk (LineOfSight.LerpRound) is private, so this mirrors it rather than changing
    /// the engine.</summary>
    public static class HexPath
    {
        public static List<HexCoord> Line(HexCoord a, HexCoord b)
        {
            int n = HexCoord.Distance(a, b);
            var cells = new List<HexCoord>(n + 1) { a };
            for (int i = 1; i <= n; i++)
            {
                var c = LerpRound(a, b, (float)i / n);
                if (c != cells[cells.Count - 1]) cells.Add(c);
            }
            if (cells[cells.Count - 1] != b) cells.Add(b);
            return cells;
        }

        static HexCoord LerpRound(HexCoord a, HexCoord b, float t)
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

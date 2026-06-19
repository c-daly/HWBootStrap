using System;

namespace HexWars.Engine
{
    /// <summary>
    /// Axial hex address (Q, R) — the column a tile or unit occupies. Cube third axis S = -Q - R.
    /// Elevation is deliberately NOT part of the coordinate: it is a property of the <c>Tile</c> at
    /// this address. A unit's full 3D position is therefore (Q, R, elevation-of-its-tile) — always
    /// derivable from the board, with one source of truth and no risk of drift.
    /// </summary>
    public readonly struct HexCoord : IEquatable<HexCoord>
    {
        public int Q { get; }
        public int R { get; }

        /// <summary>Cube third axis, derived from Q and R.</summary>
        public int S => -Q - R;

        public HexCoord(int q, int r)
        {
            Q = q;
            R = r;
        }

        /// <summary>Horizontal hex distance between two coordinates.</summary>
        public static int Distance(HexCoord a, HexCoord b)
        {
            int dq = a.Q - b.Q;
            int dr = a.R - b.R;
            int ds = a.S - b.S;
            return (Math.Abs(dq) + Math.Abs(dr) + Math.Abs(ds)) / 2;
        }

        public bool Equals(HexCoord other) => Q == other.Q && R == other.R;
        public override bool Equals(object? obj) => obj is HexCoord other && Equals(other);
        public override int GetHashCode() => unchecked((Q * 397) ^ R);
        public override string ToString() => $"({Q},{R})";

        public static bool operator ==(HexCoord a, HexCoord b) => a.Equals(b);
        public static bool operator !=(HexCoord a, HexCoord b) => !a.Equals(b);
    }
}

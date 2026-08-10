using System;

namespace HexWars.Engine.Rl
{
    public enum TacticalV3TableKind
    {
        Cells = 0,
        Units = 1,
        Templates = 2,
        CapabilityDefinitions = 3,
        CapabilityAllocations = 4,
        Rules = 5,
        MemoryRecords = 6,
        Relations = 7,
        Candidates = 8,
    }

    public readonly struct TacticalV3TokenRef : IEquatable<TacticalV3TokenRef>
    {
        public TacticalV3TokenRef(TacticalV3TableKind table, int row)
        {
            if (row < 0) throw new ArgumentOutOfRangeException(nameof(row), "row must not be negative");
            Table = table;
            Row = row;
        }

        public TacticalV3TableKind Table { get; }
        public int Row { get; }

        public bool Equals(TacticalV3TokenRef other) => Table == other.Table && Row == other.Row;
        public override bool Equals(object? obj) => obj is TacticalV3TokenRef other && Equals(other);
        public override int GetHashCode() => ((int)Table * 397) ^ Row;
    }
}

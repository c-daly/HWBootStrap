using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// One player's economy and forces: banked <see cref="Points"/>, designed-but-undeployed units
    /// (<see cref="Reserve"/>), on-board <see cref="UnitsOnBoard"/>, and <see cref="Generators"/>.
    /// Immutable — updates return copies (more <c>With…</c> helpers are added as <c>Apply</c> needs them).
    /// </summary>
    public sealed class PlayerState
    {
        private static readonly IReadOnlyList<UnitStats> NoReserve = new UnitStats[0];
        private static readonly IReadOnlyList<Unit> NoUnits = new Unit[0];
        private static readonly IReadOnlyList<Generator> NoGenerators = new Generator[0];

        public PlayerId Id { get; }
        public int Points { get; }
        public IReadOnlyList<UnitStats> Reserve { get; }
        public IReadOnlyList<Unit> UnitsOnBoard { get; }
        public IReadOnlyList<Generator> Generators { get; }

        public PlayerState(
            PlayerId id,
            int points,
            IReadOnlyList<UnitStats>? reserve = null,
            IReadOnlyList<Unit>? unitsOnBoard = null,
            IReadOnlyList<Generator>? generators = null)
        {
            Id = id;
            Points = points;
            Reserve = reserve ?? NoReserve;
            UnitsOnBoard = unitsOnBoard ?? NoUnits;
            Generators = generators ?? NoGenerators;
        }

        public PlayerState WithPoints(int points) =>
            new PlayerState(Id, points, Reserve, UnitsOnBoard, Generators);
    }
}

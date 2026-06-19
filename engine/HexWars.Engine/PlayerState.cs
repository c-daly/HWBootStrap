using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// One player's economy and forces: banked <see cref="Points"/>, designed unit templates in the
    /// <see cref="Barracks"/> (reusable blueprints — deploying a clone does NOT consume them),
    /// on-board <see cref="UnitsOnBoard"/>, and <see cref="Generators"/>. Immutable.
    /// </summary>
    public sealed class PlayerState
    {
        private static readonly IReadOnlyList<UnitStats> NoBarracks = new UnitStats[0];
        private static readonly IReadOnlyList<Unit> NoUnits = new Unit[0];
        private static readonly IReadOnlyList<Generator> NoGenerators = new Generator[0];

        public PlayerId Id { get; }
        public int Points { get; }
        public IReadOnlyList<UnitStats> Barracks { get; }
        public IReadOnlyList<Unit> UnitsOnBoard { get; }
        public IReadOnlyList<Generator> Generators { get; }

        public PlayerState(
            PlayerId id,
            int points,
            IReadOnlyList<UnitStats>? barracks = null,
            IReadOnlyList<Unit>? unitsOnBoard = null,
            IReadOnlyList<Generator>? generators = null)
        {
            Id = id;
            Points = points;
            Barracks = barracks ?? NoBarracks;
            UnitsOnBoard = unitsOnBoard ?? NoUnits;
            Generators = generators ?? NoGenerators;
        }

        public PlayerState WithPoints(int points) =>
            new PlayerState(Id, points, Barracks, UnitsOnBoard, Generators);
    }
}

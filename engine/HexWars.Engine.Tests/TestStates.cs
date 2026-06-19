using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    /// <summary>Shared builders for engine tests.</summary>
    internal static class TestStates
    {
        /// <summary>A stat line whose PointCost equals <paramref name="c"/> (all in health).</summary>
        public static UnitStats Cost(int c) => new UnitStats(c, 0, 0, 0, 0, 0, 0, 0, 0);

        public static UnitStats Stats(
            int health = 1, int damage = 0, int defense = 0,
            int movement = 0, int verticalMovement = 0,
            int range = 0, int rangeArc = 0, int vision = 0, int visionArc = 0) =>
            new UnitStats(health, damage, defense, movement, verticalMovement, range, rangeArc, vision, visionArc);

        /// <summary>
        /// Two plains columns (0,0) and (1,0); (0,0) is Player0's deployment zone, (1,0) is Player1's.
        /// Player0 to move, round 1, next entity id 1.
        /// </summary>
        public static GameState Fresh(int p0Points = 12, int p1Points = 12)
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) });

            var players = new[]
            {
                new PlayerState(PlayerId.Player0, p0Points),
                new PlayerState(PlayerId.Player1, p1Points),
            };
            return new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 1);
        }
    }
}

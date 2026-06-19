using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class LegalMovesTests
    {
        // Round 2 mid-game: P0 has a board unit (mover/attacker) at (1,0), one reserve unit, and points;
        // P1 has a target unit at (2,0) within reach. Zones: P0 -> (0,0), P1 -> (2,0).
        private static GameState MidGame()
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(2, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(2, 0) });

            var myUnit = new Unit(1, PlayerId.Player0,
                TestStates.Stats(health: 3, damage: 2, movement: 1, range: 2, vision: 3), new HexCoord(1, 0), 0);
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 3), new HexCoord(2, 0), 0);

            var p0 = new PlayerState(PlayerId.Player0, 5, barracks: new[] { TestStates.Cost(2) }, unitsOnBoard: new[] { myUnit });
            var p1 = new PlayerState(PlayerId.Player1, 5, unitsOnBoard: new[] { enemy });
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, round: 2, nextEntityId: 100);
        }

        [Test]
        public void For_AlwaysIncludesEndTurn() =>
            Assert.That(LegalMoves.For(MidGame()).OfType<EndTurn>().Any(), Is.True);

        [Test]
        public void For_IncludesAMoveForTheUnit() =>
            Assert.That(LegalMoves.For(MidGame()).OfType<MoveUnit>().Any(m => m.UnitId == 1), Is.True);

        [Test]
        public void For_IncludesAttackOnTargetableEnemy() =>
            Assert.That(LegalMoves.For(MidGame()).OfType<AttackUnit>().Any(a => a.TargetId == 2), Is.True);

        [Test]
        public void For_IncludesDeployForTemplateIntoEmptyZone() =>
            Assert.That(LegalMoves.For(MidGame()).OfType<DeployUnit>().Any(d => d.TemplateIndex == 0), Is.True);

        [Test]
        public void For_IsEmpty_WhenGameOver()
        {
            var s = MidGame();
            var over = new GameState(s.Board, s.Config, s.Players, s.ActivePlayer, s.Round, s.NextEntityId,
                isGameOver: true, winner: PlayerId.Player0);
            Assert.That(LegalMoves.For(over), Is.Empty);
        }

        [Test]
        public void For_EveryReturnedCommand_AppliesSuccessfully()
        {
            var s = MidGame();
            foreach (var cmd in LegalMoves.For(s))
                Assert.That(GameEngine.Apply(s, cmd).Success, Is.True, $"expected legal: {cmd}");
        }
    }
}

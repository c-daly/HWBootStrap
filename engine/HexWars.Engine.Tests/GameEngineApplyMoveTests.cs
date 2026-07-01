using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameEngineApplyMoveTests
    {
        // Three plains columns in a line; one Player0 unit at (0,0). Player0 to move.
        private static GameState MoveScene(UnitStats stats)
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(2, 0), 0, TerrainType.Plains),
            });
            var unit = new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0), 0);
            return new GameState(board, GameConfig.Default(),
                new[] { new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { unit }),
                        new PlayerState(PlayerId.Player1, 0) },
                PlayerId.Player0, round: 1, nextEntityId: 100);
        }

        private static UnitStats Mover(int movement) => TestStates.Stats(movement: movement);

        [Test]
        public void MoveUnit_MovesToReachableTile_AndMarksMoved()
        {
            var r = GameEngine.Apply(MoveScene(Mover(1)), new MoveUnit(PlayerId.Player0, 1, new HexCoord(1, 0)));

            Assert.That(r.Success, Is.True);
            var unit = r.NewState.Player(PlayerId.Player0).UnitsOnBoard.Single();
            Assert.That(unit.Cell, Is.EqualTo(new HexCoord(1, 0)));
            Assert.That(r.NewState.MovedUnitIds, Does.Contain(1));
        }

        [Test]
        public void MoveUnit_Rejects_OutOfMovementRange()
        {
            var r = GameEngine.Apply(MoveScene(Mover(1)), new MoveUnit(PlayerId.Player0, 1, new HexCoord(2, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.OutOfMovementRange));
        }

        [Test]
        public void MoveUnit_Rejects_WhenMovementBudgetIsSpent()
        {
            // movement is a per-turn budget spent across hops (see MultiHopMovementTests): with
            // Movement 2, hop + hop-back spends it all, and a third hop is the real rejection
            var first = GameEngine.Apply(MoveScene(Mover(2)), new MoveUnit(PlayerId.Player0, 1, new HexCoord(1, 0)));
            var second = GameEngine.Apply(first.NewState, new MoveUnit(PlayerId.Player0, 1, new HexCoord(0, 0)));
            Assert.That(second.Success, Is.True, "budget remains — hopping back is legal now");
            var third = GameEngine.Apply(second.NewState, new MoveUnit(PlayerId.Player0, 1, new HexCoord(1, 0)));
            Assert.That(third.Reason, Is.EqualTo(RejectionReason.UnitAlreadyMoved));
        }

        [Test]
        public void MoveUnit_Rejects_WhenUnitNotFound()
        {
            var r = GameEngine.Apply(MoveScene(Mover(1)), new MoveUnit(PlayerId.Player0, 99, new HexCoord(1, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.UnitNotFound));
        }
    }
}

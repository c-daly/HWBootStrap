using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class CaptureHexTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 has one unit on A with `points`; board is a 1x3 strip so A exists. Plains, biomes off.
        static GameState State(int points, PlayerId active = PlayerId.Player0, PlayerId? controlA = null)
        {
            var tiles = new[]
            {
                new Tile(A, 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(2, 0), 0, TerrainType.Plains),
            };
            Board board = new Board(tiles);
            if (controlA != null) board = board.WithControl(A, controlA.Value);
            var p0 = new PlayerState(PlayerId.Player0, points, null,
                new[] { new Unit(1, PlayerId.Player0, S, A, 0) });
            var p1 = new PlayerState(PlayerId.Player1, points);
            return new GameState(board, GameConfig.Default(biomesEnabled: false),
                new[] { p0, p1 }, active, round: 2, nextEntityId: 99);
        }

        [Test]
        public void Capture_FlipsControl_AndDeductsCost()
        {
            var r = GameEngine.Apply(State(10), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Board.Controller(A), Is.EqualTo(PlayerId.Player0));
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(10 - 3)); // default CaptureCost
        }

        [Test]
        public void Capture_RejectsWhenNoUnitOnHex()
        {
            var r = GameEngine.Apply(State(10), new CaptureHex(PlayerId.Player0, new HexCoord(1, 0)));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.NoUnitOnHex));
        }

        [Test]
        public void Capture_RejectsWhenAlreadyControlled()
        {
            var r = GameEngine.Apply(State(10, controlA: PlayerId.Player0), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.AlreadyControlled));
        }

        [Test]
        public void Capture_RejectsWhenBroke()
        {
            var r = GameEngine.Apply(State(2), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }

        [Test]
        public void Steal_RecapturesEnemyControlledHex()
        {
            var r = GameEngine.Apply(State(10, controlA: PlayerId.Player1), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Board.Controller(A), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void LegalMoves_IncludeCaptureOfGarrisonedUncontrolledHex()
        {
            var moves = LegalMoves.For(State(10));
            Assert.That(moves, Has.Some.Matches<Command>(m => m is CaptureHex ch && ch.Cell == A));
        }

        [Test]
        public void LegalMoves_ExcludeCaptureWhenAlreadyControlled()
        {
            var moves = LegalMoves.For(State(10, controlA: PlayerId.Player0));
            Assert.That(moves, Has.None.Matches<Command>(m => m is CaptureHex));
        }
    }
}

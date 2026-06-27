using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class ClaimTempoTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly HexCoord B = new HexCoord(1, 0);
        static readonly HexCoord C = new HexCoord(2, 0);
        static readonly UnitStats S = new UnitStats(8, 3, 2, 4, 2, 1, 1, 3, 1);

        // P0 has a unit on A (a neutral hex) and another on B; whole-army policy; territory mode, claim ends turn.
        static GameState State()
        {
            var board = new Board(new[]
            {
                new Tile(A, 0, TerrainType.Plains), new Tile(B, 0, TerrainType.Plains), new Tile(C, 0, TerrainType.Plains),
            });
            var p0 = new PlayerState(PlayerId.Player0, 50, null, new[]
            {
                new Unit(1, PlayerId.Player0, S, A, 0),
                new Unit(2, PlayerId.Player0, S, B, 0),
            });
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var cfg = GameConfig.Default(biomesEnabled: false, territoryMode: true, captureCost: 3);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 1, 9);
        }

        [Test]
        public void Claim_AsFirstAction_Succeeds_AndEndsTurn()
        {
            var r = GameEngine.Apply(State(), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Board.Controller(A), Is.EqualTo(PlayerId.Player0));
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player1), "claim ends the turn");
        }

        [Test]
        public void Claim_AfterMoving_Rejected()
        {
            var moved = GameEngine.Apply(State(), new MoveUnit(PlayerId.Player0, 2, C));
            Assert.That(moved.Success, Is.True);
            Assert.That(moved.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player0), "whole-army: still P0's turn");

            var r = GameEngine.Apply(moved.NewState, new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.MustClaimFirst));
        }

        [Test]
        public void LegalMoves_OmitsCapture_AfterAnArmyAction()
        {
            var moved = GameEngine.Apply(State(), new MoveUnit(PlayerId.Player0, 2, C));
            var moves = LegalMoves.For(moved.NewState);
            Assert.That(moves.Any(m => m is CaptureHex), Is.False);

            var fresh = LegalMoves.For(State());
            Assert.That(fresh.Any(m => m is CaptureHex c && c.Cell == A), Is.True);
        }
    }
}

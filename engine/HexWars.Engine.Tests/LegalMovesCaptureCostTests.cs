using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>LegalMoves must gate a CaptureHex on the same (possibly scaled) cost the handler charges,
    /// so an enumerated capture is always affordable — no enumerate-then-reject on a generator hex.</summary>
    public class LegalMovesCaptureCostTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 has a unit on A; P1 controls A with a strength-1 generator.
        // Output=10, CaptureFactor=4 -> scaled capture cost 40; flat CaptureCost=3.
        static GameState State(int p0Points)
        {
            var board = new Board(new[] { new Tile(A, 0, TerrainType.Plains) }).WithControl(A, PlayerId.Player1);
            var p0 = new PlayerState(PlayerId.Player0, p0Points, null, new[] { new Unit(1, PlayerId.Player0, S, A, 0) });
            var p1 = new PlayerState(PlayerId.Player1, 0, null, null,
                new[] { new Generator(7, PlayerId.Player1, A, 0, 3, 1.0) });
            var cfg = GameConfig.Default(biomesEnabled: false, generatorOutput: 10, captureFactor: 4.0, captureCost: 3);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 99);
        }

        [Test]
        public void CaptureHex_NotEnumerated_WhenOnlyFlatCostAffordable()
        {
            // 10 points affords the flat 3 but not the scaled 40 — must NOT be offered.
            var moves = LegalMoves.For(State(10));
            Assert.That(moves.Any(m => m is CaptureHex), Is.False);
        }

        [Test]
        public void CaptureHex_Enumerated_WhenScaledCostAffordable()
        {
            var moves = LegalMoves.For(State(100));
            Assert.That(moves.Any(m => m is CaptureHex c && c.Cell == A), Is.True);
        }
    }
}

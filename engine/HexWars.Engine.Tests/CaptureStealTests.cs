using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class CaptureStealTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 has a unit on A; P1 controls A and has a strength-1 generator there. Output=10, CaptureFactor=4, CaptureCost=3.
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
        public void Capture_OfGeneratorHex_ScalesCost_AndStealsGenerator()
        {
            var r = GameEngine.Apply(State(100), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            // cost = max(3, round(4 * (10*1.0))) = 40
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(100 - 40));
            // generator moved from P1 to P0, re-owned
            Assert.That(r.NewState.Player(PlayerId.Player1).Generators.Count(g => g.IsAlive), Is.EqualTo(0));
            var stolen = r.NewState.Player(PlayerId.Player0).Generators.Single();
            Assert.That(stolen.Id, Is.EqualTo(7));
            Assert.That(stolen.Owner, Is.EqualTo(PlayerId.Player0));
            Assert.That(r.NewState.Board.Controller(A), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void Capture_OfGeneratorHex_RejectsWhenCannotAffordScaledCost()
        {
            // can afford the flat CaptureCost (3) but not the scaled 40
            var r = GameEngine.Apply(State(10), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }
    }
}

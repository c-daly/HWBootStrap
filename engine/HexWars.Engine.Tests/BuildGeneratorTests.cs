using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BuildGeneratorTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 controls A and has a unit there, with `points`. GeneratorOutput=10, BuildFactor=4 -> build cost 40.
        static GameState State(int points, bool control = true, bool generatorAlready = false)
        {
            Board board = new Board(new[]
            {
                new Tile(A, 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            });
            if (control) board = board.WithControl(A, PlayerId.Player0);
            var gens = generatorAlready
                ? new[] { new Generator(5, PlayerId.Player0, A, 0, 3) }
                : null;
            var p0 = new PlayerState(PlayerId.Player0, points, null, new[] { new Unit(1, PlayerId.Player0, S, A, 0) }, gens);
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var cfg = GameConfig.Default(biomesEnabled: false, generatorOutput: 10, buildFactor: 4.0);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 99);
        }

        [Test]
        public void Build_PlacesGenerator_AndDeductsCost()
        {
            var r = GameEngine.Apply(State(50), new BuildGenerator(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            var gens = r.NewState.Player(PlayerId.Player0).Generators;
            Assert.That(gens.Count, Is.EqualTo(1));
            Assert.That(gens[0].Cell, Is.EqualTo(A));
            Assert.That(gens[0].Strength, Is.EqualTo(1.0));
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(50 - 40)); // BuildFactor*Output
        }

        [Test]
        public void Build_RejectsWhenHexNotControlled()
        {
            var r = GameEngine.Apply(State(50, control: false), new BuildGenerator(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.HexNotControlled));
        }

        [Test]
        public void Build_RejectsWhenGeneratorAlreadyThere()
        {
            var r = GameEngine.Apply(State(50, generatorAlready: true), new BuildGenerator(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TileOccupied));
        }

        [Test]
        public void Build_RejectsWhenBroke()
        {
            var r = GameEngine.Apply(State(10), new BuildGenerator(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }

        [Test]
        public void LegalMoves_IncludeBuildOnControlledGarrisonedHex()
        {
            var moves = LegalMoves.For(State(50));
            Assert.That(moves, Has.Some.Matches<Command>(m => m is BuildGenerator b && b.Cell == A));
        }
    }
}

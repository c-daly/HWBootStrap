using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TargetingServiceTests
    {
        private static UnitStats S(int range = 0, int rangeArc = 0, int vision = 0, int visionArc = 0) =>
            new UnitStats(health: 1, damage: 0, defense: 0, movement: 0, verticalMovement: 0,
                          range: range, rangeArc: rangeArc, vision: vision, visionArc: visionArc);

        private static Board Cells(params (int q, int r, int elev, TerrainType t)[] c) =>
            new Board(c.Select(x => new Tile(new HexCoord(x.q, x.r), x.elev, x.t)).ToArray());

        private static GameState State(Board board, Unit[] mine, Unit[] enemy) =>
            new GameState(board, GameConfig.Default(),
                new[] { new PlayerState(PlayerId.Player0, 0, unitsOnBoard: mine),
                        new PlayerState(PlayerId.Player1, 0, unitsOnBoard: enemy) },
                PlayerId.Player0, 1, 100);

        private static readonly (int, int, int, TerrainType)[] FlatLine =
        {
            (0, 0, 0, TerrainType.Plains), (1, 0, 0, TerrainType.Plains), (2, 0, 0, TerrainType.Plains),
        };

        [Test]
        public void CanTarget_WhenInRangeAndSeenByItself()
        {
            var board = Cells(FlatLine);
            var attacker = new Unit(1, PlayerId.Player0, S(range: 3, vision: 3), new HexCoord(0, 0), 0);
            var target = new Unit(2, PlayerId.Player1, S(), new HexCoord(2, 0), 0);
            var state = State(board, new[] { attacker }, new[] { target });

            Assert.That(TargetingService.CanTarget(state, attacker, target.Cell, target.Elevation), Is.True);
        }

        [Test]
        public void CannotTarget_WhenOutOfRange()
        {
            var board = Cells(FlatLine);
            var attacker = new Unit(1, PlayerId.Player0, S(range: 1, vision: 5), new HexCoord(0, 0), 0);
            var target = new Unit(2, PlayerId.Player1, S(), new HexCoord(2, 0), 0);
            var state = State(board, new[] { attacker }, new[] { target });

            Assert.That(TargetingService.CanTarget(state, attacker, target.Cell, target.Elevation), Is.False);
        }

        [Test]
        public void CannotTarget_WhenBlind_AndNoSpotter()
        {
            var board = Cells(FlatLine);
            var attacker = new Unit(1, PlayerId.Player0, S(range: 5, vision: 0), new HexCoord(0, 0), 0);
            var target = new Unit(2, PlayerId.Player1, S(), new HexCoord(2, 0), 0);
            var state = State(board, new[] { attacker }, new[] { target });

            Assert.That(TargetingService.CanTarget(state, attacker, target.Cell, target.Elevation), Is.False);
        }

        [Test]
        public void Spotter_GivesBlindAttacker_ArmyWideVision()
        {
            var board = Cells(FlatLine);
            var attacker = new Unit(1, PlayerId.Player0, S(range: 5, vision: 0), new HexCoord(0, 0), 0);
            var spotter = new Unit(3, PlayerId.Player0, S(vision: 5), new HexCoord(1, 0), 0); // 0 damage, all eyes
            var target = new Unit(2, PlayerId.Player1, S(), new HexCoord(2, 0), 0);
            var state = State(board, new[] { attacker, spotter }, new[] { target });

            Assert.That(TargetingService.CanTarget(state, attacker, target.Cell, target.Elevation), Is.True);
        }

        [Test]
        public void RangeArc_GatesFiringUpward()
        {
            var board = Cells((0, 0, 0, TerrainType.Plains), (1, 0, 3, TerrainType.Plains));
            var lowArc = new Unit(1, PlayerId.Player0, S(range: 5, rangeArc: 1, vision: 5, visionArc: 5), new HexCoord(0, 0), 0);
            var target = new Unit(2, PlayerId.Player1, S(), new HexCoord(1, 0), 3); // 3 levels up
            var state = State(board, new[] { lowArc }, new[] { target });
            Assert.That(TargetingService.CanTarget(state, lowArc, target.Cell, target.Elevation), Is.False);

            var highArc = new Unit(1, PlayerId.Player0, S(range: 5, rangeArc: 3, vision: 5, visionArc: 5), new HexCoord(0, 0), 0);
            var state2 = State(board, new[] { highArc }, new[] { target });
            Assert.That(TargetingService.CanTarget(state2, highArc, target.Cell, target.Elevation), Is.True);
        }

        [Test]
        public void VisionArc_GatesSightUpward()
        {
            var board = Cells((0, 0, 0, TerrainType.Plains), (1, 0, 3, TerrainType.Plains));
            var u = new Unit(1, PlayerId.Player0, S(range: 5, rangeArc: 5, vision: 5, visionArc: 1), new HexCoord(0, 0), 0);
            var target = new Unit(2, PlayerId.Player1, S(), new HexCoord(1, 0), 3);
            var state = State(board, new[] { u }, new[] { target });

            Assert.That(TargetingService.CanTarget(state, u, target.Cell, target.Elevation), Is.False);
        }

        [Test]
        public void HighGround_ExtendsHorizontalRange()
        {
            var board = Cells((0, 0, 5, TerrainType.Plains), (1, 0, 0, TerrainType.Plains), (2, 0, 0, TerrainType.Plains));
            // Range 1, but +1 per level of high ground (H=5) -> effective 6 >= dist 2
            var onPeak = new Unit(1, PlayerId.Player0, S(range: 1, vision: 5, visionArc: 5), new HexCoord(0, 0), 5);
            var target = new Unit(2, PlayerId.Player1, S(), new HexCoord(2, 0), 0);
            var state = State(board, new[] { onPeak }, new[] { target });

            Assert.That(TargetingService.CanTarget(state, onPeak, target.Cell, target.Elevation), Is.True);
        }

        [Test]
        public void Concealment_RaisesVisionNeeded()
        {
            var board = Cells((0, 0, 0, TerrainType.Plains), (1, 0, 0, TerrainType.Plains), (2, 0, 0, TerrainType.Forest));
            var target = new Unit(2, PlayerId.Player1, S(), new HexCoord(2, 0), 0); // in forest (concealment 2)

            var weakEyes = new Unit(1, PlayerId.Player0, S(range: 5, vision: 3), new HexCoord(0, 0), 0); // 2 + 2 = 4 > 3
            Assert.That(TargetingService.CanTarget(State(board, new[] { weakEyes }, new[] { target }),
                weakEyes, target.Cell, target.Elevation), Is.False);

            var sharpEyes = new Unit(1, PlayerId.Player0, S(range: 5, vision: 4), new HexCoord(0, 0), 0); // 2 + 2 = 4 <= 4
            Assert.That(TargetingService.CanTarget(State(board, new[] { sharpEyes }, new[] { target }),
                sharpEyes, target.Cell, target.Elevation), Is.True);
        }
    }
}

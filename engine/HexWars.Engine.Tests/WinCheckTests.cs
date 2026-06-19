using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class WinCheckTests
    {
        // A stat line whose PointCost equals `c` (all in health).
        private static UnitStats Cost(int c) => new UnitStats(c, 0, 0, 0, 0, 0, 0, 0, 0);
        private static Board Board1() => new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });

        private static GameState State(PlayerState p0, PlayerState p1, int round, GameConfig? cfg = null) =>
            new GameState(Board1(), cfg ?? GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, round, 100);

        [Test]
        public void Evaluate_SumsBankedUnitsAndGenerators_NotBarracks()
        {
            var unit = new Unit(1, PlayerId.Player0, Cost(4), new HexCoord(0, 0), 0);
            var gen = new Generator(2, PlayerId.Player0, new HexCoord(0, 0), 0, 3);
            var p0 = new PlayerState(PlayerId.Player0, 5,
                barracks: new[] { Cost(3) }, unitsOnBoard: new[] { unit }, generators: new[] { gen });

            // 5 banked + 4 unit + 2 generator(cost) = 11 (free barracks template adds nothing)
            Assert.That(WinCheck.Evaluate(State(p0, new PlayerState(PlayerId.Player1, 0), 2), PlayerId.Player0),
                        Is.EqualTo(11));
        }

        [Test]
        public void IsEliminated_True_WhenNoUnitsAndBroke()
        {
            var s = State(new PlayerState(PlayerId.Player0, 0), new PlayerState(PlayerId.Player1, 0), 2);
            Assert.That(WinCheck.IsEliminated(s, PlayerId.Player0), Is.True);
        }

        [Test]
        public void IsEliminated_False_WhenCanStillAffordACheapUnit()
        {
            var s = State(new PlayerState(PlayerId.Player0, 1), new PlayerState(PlayerId.Player1, 0), 2);
            Assert.That(WinCheck.IsEliminated(s, PlayerId.Player0), Is.False);
        }

        [Test]
        public void IsEliminated_False_WithBoardUnit()
        {
            var unit = new Unit(1, PlayerId.Player0, Cost(1), new HexCoord(0, 0), 0);
            var s = State(new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { unit }),
                          new PlayerState(PlayerId.Player1, 0), 2);
            Assert.That(WinCheck.IsEliminated(s, PlayerId.Player0), Is.False);
        }

        [Test]
        public void IsEliminated_True_WithBarracksTemplate_ButCannotAffordToDeployIt()
        {
            // a reusable template doesn't help if you can't pay the deploy cost
            var s = State(new PlayerState(PlayerId.Player0, 0, barracks: new[] { Cost(1) }),
                          new PlayerState(PlayerId.Player1, 0), 2);
            Assert.That(WinCheck.IsEliminated(s, PlayerId.Player0), Is.True);
        }

        [Test]
        public void Resolve_DeclaresWinner_WhenOpponentEliminated_AfterOpening()
        {
            var unit = new Unit(1, PlayerId.Player0, Cost(3), new HexCoord(0, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { unit });
            var p1 = new PlayerState(PlayerId.Player1, 0);
            Assert.That(WinCheck.Resolve(State(p0, p1, 2)), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void Resolve_NoWinner_DuringOpeningRound1()
        {
            var unit = new Unit(1, PlayerId.Player0, Cost(3), new HexCoord(0, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { unit });
            var p1 = new PlayerState(PlayerId.Player1, 0);
            Assert.That(WinCheck.Resolve(State(p0, p1, 1)), Is.Null);
        }

        [Test]
        public void Resolve_AtRoundCap_WinsByHigherTotalValue()
        {
            var u0 = new Unit(1, PlayerId.Player0, Cost(5), new HexCoord(0, 0), 0);
            var u1 = new Unit(2, PlayerId.Player1, Cost(2), new HexCoord(0, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { u0 });
            var p1 = new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { u1 });
            Assert.That(WinCheck.Resolve(State(p0, p1, 40)), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void IsTerminal_True_WhenAPlayerEliminated_AfterOpening()
        {
            var unit = new Unit(1, PlayerId.Player0, Cost(3), new HexCoord(0, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { unit });
            var p1 = new PlayerState(PlayerId.Player1, 0);
            Assert.That(WinCheck.IsTerminal(State(p0, p1, 2)), Is.True);
        }

        [Test]
        public void IsTerminal_False_DuringOpeningRound1()
        {
            var p0 = new PlayerState(PlayerId.Player0, 0);
            var p1 = new PlayerState(PlayerId.Player1, 0);
            Assert.That(WinCheck.IsTerminal(State(p0, p1, 1)), Is.False);
        }

        [Test]
        public void IsTerminal_True_AtRoundCap()
        {
            var u0 = new Unit(1, PlayerId.Player0, Cost(1), new HexCoord(0, 0), 0);
            var u1 = new Unit(2, PlayerId.Player1, Cost(1), new HexCoord(0, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { u0 });
            var p1 = new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { u1 });
            Assert.That(WinCheck.IsTerminal(State(p0, p1, 40)), Is.True);
        }
    }
}

using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// Incremental movement: a unit spends its per-turn movement budget across any number of hops
    /// instead of committing to one exact-distance move. All hops of one unit still count as a
    /// single "action" for turn pacing.
    /// </summary>
    public class MultiHopMovementTests
    {
        const PlayerId P0 = PlayerId.Player0;

        /// <summary>A flat 1×7 strip; P0's unit (Movement 3) at q=0, an enemy parked at q=6.</summary>
        static GameState Strip(ITurnPolicy? policy = null, int[]? elevations = null)
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 7; q++)
                tiles.Add(new Tile(new HexCoord(q, 0), elevations != null ? elevations[q] : 0, TerrainType.Plains));
            var board = new Board(tiles, new[] { new HexCoord(0, 0) }, new[] { new HexCoord(6, 0) });

            var stats = new UnitStats(5, 1, 0, 3, 1, 1, 0, 2, 1); // hp5 dmg1 def0 move3 vmove1
            var p0 = new PlayerState(P0, 10, null, new[] { new Unit(1, P0, stats, new HexCoord(0, 0), elevations != null ? elevations[0] : 0) }, null);
            var p1 = new PlayerState(PlayerId.Player1, 10, null, new[] { new Unit(2, PlayerId.Player1, stats, new HexCoord(6, 0), elevations != null ? elevations[6] : 0) }, null);
            return new GameState(board, GameConfig.Default(biomesEnabled: false, turnPolicy: policy),
                                 new[] { p0, p1 }, P0, 1, 3);
        }

        static GameState Move(GameState s, int q)
        {
            var r = GameEngine.Apply(s, new MoveUnit(P0, 1, new HexCoord(q, 0)));
            Assert.That(r.Success, Is.True, $"move to q={q} should be legal (got {r.Reason})");
            return r.NewState;
        }

        [Test]
        public void Unit_CanKeepHopping_UntilItsBudgetIsSpent()
        {
            var s = Strip();
            s = Move(s, 1);            // hop 1: budget 3 -> 2
            s = Move(s, 2);            // hop 2: -> 1
            s = Move(s, 3);            // hop 3: -> 0

            var r = GameEngine.Apply(s, new MoveUnit(P0, 1, new HexCoord(4, 0)));
            Assert.That(r.Success, Is.False, "a fourth hex exceeds the 3-movement budget");
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.UnitAlreadyMoved));
        }

        [Test]
        public void Budget_IsSharedAcrossHops_NotPerHop()
        {
            var s = Strip();
            s = Move(s, 3);            // one full-budget move
            var r = GameEngine.Apply(s, new MoveUnit(P0, 1, new HexCoord(4, 0)));
            Assert.That(r.Success, Is.False, "the budget was spent in one move; no hop remains");
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.UnitAlreadyMoved));
        }

        [Test]
        public void ReachableTiles_ShrinkWithSpentBudget()
        {
            var s = Strip();
            s = Move(s, 1);
            var unit = s.Player(P0).UnitsOnBoard[0];
            var reach = MovementService.ReachableTiles(s, unit);
            Assert.That(reach, Does.Contain(new HexCoord(3, 0)), "2 movement left reaches q=3");
            Assert.That(reach, Does.Not.Contain(new HexCoord(4, 0)), "q=4 needs 3 movement, only 2 remain");
        }

        [Test]
        public void VerticalBudget_IsAlsoSpentAcrossHops()
        {
            // elevations rise along the strip: climbing q1 (elev 1) uses the whole vmove-1 budget,
            // so the later climb to q2 (elev 2) must be rejected even though horizontal remains
            var s = Strip(elevations: new[] { 0, 1, 2, 2, 2, 2, 2 });
            s = Move(s, 1);
            var r = GameEngine.Apply(s, new MoveUnit(P0, 1, new HexCoord(2, 0)));
            Assert.That(r.Success, Is.False, "second climb exceeds the vertical budget");
        }

        [Test]
        public void MultiHop_CountsAsOneActionForPacing()
        {
            var s = Strip(policy: new KActionsPolicy(2));
            s = Move(s, 1);
            Assert.That(s.ActivePlayer, Is.EqualTo(P0), "1 action of 2 committed");
            s = Move(s, 2);
            Assert.That(s.ActivePlayer, Is.EqualTo(P0),
                "a second hop of the SAME unit is still the same committed action");
            Assert.That(s.MovedUnitIds.Count, Is.EqualTo(1));
        }

        [Test]
        public void EndTurn_RestoresTheFullBudget()
        {
            var s = Strip();
            s = Move(s, 3);
            s = GameEngine.Apply(s, new EndTurn(P0)).NewState;
            s = GameEngine.Apply(s, new EndTurn(PlayerId.Player1)).NewState;

            var r = GameEngine.Apply(s, new MoveUnit(P0, 1, new HexCoord(4, 0)));
            Assert.That(r.Success, Is.True, "a new turn brings a fresh movement budget");
        }
    }
}

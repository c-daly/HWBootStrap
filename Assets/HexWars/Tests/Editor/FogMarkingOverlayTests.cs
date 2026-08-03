using System.Collections.Generic;
using HexWars.Engine;
using HexWars.Presentation;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>
    /// Pure-computation tests for <see cref="FogMarkingOverlay"/> — spec §"Fog-of-War Indicator"
    /// (amended 2026-07-25): a marking over every cell outside the CURRENT ACTING PLAYER's visibility,
    /// computed straight from an already-produced <see cref="GameState"/> using the engine's
    /// authoritative <see cref="TargetingService.IsVisibleToArmy"/> rule. No rendering, no Unity
    /// objects — see <c>ModelDuelConfigurationTests</c> "Viewer C" section for the driver-integration
    /// (queue-advance, toggle, render) coverage.
    /// </summary>
    public sealed class FogMarkingOverlayTests
    {
        // A 5-wide single-row line board (q = 0..4, r = 0), flat plains (elevation 0, no concealment).
        // Distance((0,0),(q,0)) collapses to |q| on this line, which keeps expected visibility sets
        // easy to state by hand.
        static readonly UnitStats OneVisionUnit = new UnitStats(
            health: 1, damage: 0, defense: 0,
            movement: 0, verticalMovement: 0,
            range: 0, rangeArc: 0,
            vision: 1, visionArc: 0);

        static Board LineBoard(int length)
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < length; q++)
                tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));
            return new Board(tiles);
        }

        static GameState FogState(PlayerId activePlayer, HexCoord p0UnitCell, HexCoord p1UnitCell)
        {
            Board board = LineBoard(5);
            var p0Unit = new Unit(1, PlayerId.Player0, OneVisionUnit, p0UnitCell, 0);
            var p1Unit = new Unit(2, PlayerId.Player1, OneVisionUnit, p1UnitCell, 0);
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { p0Unit }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { p1Unit }),
            };
            return new GameState(
                board, GameConfig.Default(fogOfWar: true), players, activePlayer, round: 1, nextEntityId: 3);
        }

        static HashSet<HexCoord> ExpectedComplementOfArmyVisibility(GameState state, PlayerId acting)
        {
            var marked = new HashSet<HexCoord>();
            foreach (Tile tile in state.Board.Tiles)
                if (!TargetingService.IsVisibleToArmy(state, acting, tile.Coord, tile.Elevation))
                    marked.Add(tile.Coord);
            return marked;
        }

        [Test]
        public void MarkedCells_NullState_ReturnsEmpty()
        {
            Assert.That(FogMarkingOverlay.MarkedCells(null), Is.Empty);
        }

        [Test]
        public void MarkedCells_FogDisabledInConfig_ReturnsEmptyRegardlessOfVisibility()
        {
            Board board = LineBoard(5);
            var p0Unit = new Unit(1, PlayerId.Player0, OneVisionUnit, new HexCoord(0, 0), 0);
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { p0Unit }),
                new PlayerState(PlayerId.Player1, 0),
            };
            var state = new GameState(
                board, GameConfig.Default(fogOfWar: false), players, PlayerId.Player0, round: 1, nextEntityId: 2);

            Assert.That(FogMarkingOverlay.MarkedCells(state), Is.Empty,
                "spec: \"When fog of war is disabled, no marking is drawn.\"");
        }

        [Test]
        public void MarkedCells_FogEnabled_EqualsComplementOfArmyVisibilityForTheActingPlayer()
        {
            GameState state = FogState(PlayerId.Player0, new HexCoord(0, 0), new HexCoord(4, 0));

            IReadOnlyCollection<HexCoord> marked = FogMarkingOverlay.MarkedCells(state);
            HashSet<HexCoord> expected = ExpectedComplementOfArmyVisibility(state, PlayerId.Player0);

            Assert.That(marked, Is.Not.Empty, "the test board must actually exercise some hidden cells");
            Assert.That(new HashSet<HexCoord>(marked), Is.EquivalentTo(expected));
            // Pin the exact expected set by hand too, so a change to the engine's distance/vision
            // semantics that happened to keep ExpectedComplementOfArmyVisibility() in lockstep would
            // still be caught here.
            Assert.That(new HashSet<HexCoord>(marked), Is.EquivalentTo(new[]
            {
                new HexCoord(2, 0), new HexCoord(3, 0), new HexCoord(4, 0),
            }));
        }

        [Test]
        public void MarkedCells_FollowsTheActingPlayerAutomaticallyNotAFixedSeat()
        {
            GameState p0Active = FogState(PlayerId.Player0, new HexCoord(0, 0), new HexCoord(4, 0));
            GameState p1Active = FogState(PlayerId.Player1, new HexCoord(0, 0), new HexCoord(4, 0));

            HashSet<HexCoord> markedForP0 = new HashSet<HexCoord>(FogMarkingOverlay.MarkedCells(p0Active));
            HashSet<HexCoord> markedForP1 = new HashSet<HexCoord>(FogMarkingOverlay.MarkedCells(p1Active));

            Assert.That(markedForP0, Is.EquivalentTo(new[]
            {
                new HexCoord(2, 0), new HexCoord(3, 0), new HexCoord(4, 0),
            }), "acting P0 must be marked by P0's own army vision (its unit sits at q=0)");
            Assert.That(markedForP1, Is.EquivalentTo(new[]
            {
                new HexCoord(0, 0), new HexCoord(1, 0), new HexCoord(2, 0),
            }), "acting P1 must be marked by P1's own army vision (its unit sits at q=4), " +
                "not P0's — there is no fixed P1/P2 selector, only the current acting player");
            Assert.That(markedForP0, Is.Not.EquivalentTo(markedForP1));
        }

        [Test]
        public void UnitIdsToDim_NullStateOrEmptyMarkedCells_ReturnsEmpty()
        {
            GameState state = FogState(PlayerId.Player0, new HexCoord(0, 0), new HexCoord(4, 0));

            Assert.That(FogMarkingOverlay.UnitIdsToDim(null, new HashSet<HexCoord> { new HexCoord(0, 0) }),
                Is.Empty);
            Assert.That(FogMarkingOverlay.UnitIdsToDim(state, new HashSet<HexCoord>()), Is.Empty);
            Assert.That(FogMarkingOverlay.UnitIdsToDim(state, null), Is.Empty);
        }

        [Test]
        public void UnitIdsToDim_ReturnsOnlyLivingUnitsStandingInsideMarkedCells()
        {
            Board board = LineBoard(5);
            var p0Visible = new Unit(1, PlayerId.Player0, OneVisionUnit, new HexCoord(0, 0), 0);
            var p1Marked = new Unit(2, PlayerId.Player1, OneVisionUnit, new HexCoord(3, 0), 0);
            var p1Dead = new Unit(3, PlayerId.Player1, OneVisionUnit, new HexCoord(4, 0), 0).WithDamage(999);
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { p0Visible }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { p1Marked, p1Dead }),
            };
            var state = new GameState(
                board, GameConfig.Default(fogOfWar: true), players, PlayerId.Player0, round: 1, nextEntityId: 4);
            var markedCells = new HashSet<HexCoord> { new HexCoord(3, 0), new HexCoord(4, 0) };

            IReadOnlyCollection<int> dimmed = FogMarkingOverlay.UnitIdsToDim(state, markedCells);

            Assert.That(dimmed, Is.EquivalentTo(new[] { 2 }),
                "unit 1 sits outside the marked cells, unit 3 is dead (never rendered at all, so " +
                "nothing to dim) — only living unit 2, standing in a marked cell, should be flagged");
        }
    }
}

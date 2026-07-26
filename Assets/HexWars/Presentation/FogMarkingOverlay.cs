using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Pure computation for the spectator fog-marking overlay — spec §"Fog-of-War Indicator", amended
    /// 2026-07-25: a shaded marking over every cell outside the CURRENT ACTING PLAYER's visibility
    /// (<see cref="GameState.ActivePlayer"/>), computed from the omniscient
    /// <see cref="ModelDuelDriver.PresentedState"/> using the engine's authoritative visibility rule
    /// (<see cref="TargetingService.IsVisibleToArmy"/>). The marking follows turn order automatically —
    /// there is no P1/P2/learner selector.
    /// <para/>
    /// The viewer stays omniscient: every unit still renders, marked cell or not. <see cref="UnitIdsToDim"/>
    /// is a second pure function reporting which already-rendered units sit inside a marked cell, so the
    /// renderer can give them a distinct dimmed treatment instead of hiding them — "spectator can see
    /// this" vs. "the acting model can see this."
    /// <para/>
    /// Both functions only read an already-produced <see cref="GameState"/>; they never touch
    /// observations, masks, policy inputs, or simulation state, and have no rendering side effects —
    /// callers own turning the result into visuals.
    /// </summary>
    public static class FogMarkingOverlay
    {
        static readonly HashSet<HexCoord> EmptyCells = new HashSet<HexCoord>();
        static readonly HashSet<int> EmptyUnitIds = new HashSet<int>();

        /// <summary>Every board cell outside the acting player's army visibility. Empty when
        /// <paramref name="state"/> is null or the scenario's fog of war is off — spec: "When fog of war
        /// is disabled, no marking is drawn."</summary>
        public static IReadOnlyCollection<HexCoord> MarkedCells(GameState state)
        {
            if (state == null || !state.Config.FogOfWar) return EmptyCells;
            PlayerId acting = state.ActivePlayer;
            var marked = new HashSet<HexCoord>();
            foreach (Tile tile in state.Board.Tiles)
                if (!TargetingService.IsVisibleToArmy(state, acting, tile.Coord, tile.Elevation))
                    marked.Add(tile.Coord);
            return marked;
        }

        /// <summary>Ids of every living unit (either army) standing in a cell from
        /// <paramref name="markedCells"/> — still fully rendered at rest, but flagged so the renderer can
        /// apply the distinct dimmed/marked treatment.</summary>
        public static IReadOnlyCollection<int> UnitIdsToDim(
            GameState state, IReadOnlyCollection<HexCoord> markedCells)
        {
            if (state == null || markedCells == null || markedCells.Count == 0) return EmptyUnitIds;
            var cells = markedCells as HashSet<HexCoord> ?? new HashSet<HexCoord>(markedCells);
            var ids = new HashSet<int>();
            foreach (PlayerState player in state.Players)
                foreach (Unit unit in player.UnitsOnBoard)
                    if (unit.IsAlive && cells.Contains(unit.Cell))
                        ids.Add(unit.Id);
            return ids;
        }
    }
}

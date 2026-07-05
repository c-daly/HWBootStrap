using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// What the viewer is entitled to see of an action under fog. Pure functions over engine
    /// state — the presenter asks these before animating so the presentation never leaks more
    /// intel than the game state does. Cells are judged at their tile's ground elevation
    /// (ground-first M1: a unit's elevation is the tile it stands on).
    /// </summary>
    public static class FogPresentation
    {
        static bool Fogged(GameState state, PlayerId? viewer) => viewer.HasValue && state.Config.FogOfWar;

        static bool CellVisible(GameState state, PlayerId viewer, HexCoord cell) =>
            state.Board.Contains(cell)
            && TargetingService.IsVisibleToArmy(state, viewer, cell, state.Board.TileAt(cell).Elevation);

        /// <summary>Inclusive index span of <paramref name="path"/> cells the viewer can see;
        /// (-1,-1) when none. Full span when there is no viewer or fog is off.</summary>
        public static (int First, int Last) VisibleSpan(GameState state, PlayerId? viewer, IReadOnlyList<HexCoord> path)
        {
            if (path.Count == 0) return (-1, -1);
            if (!Fogged(state, viewer)) return (0, path.Count - 1);
            int first = -1, last = -1;
            for (int i = 0; i < path.Count; i++)
                if (CellVisible(state, viewer.Value, path[i]))
                {
                    if (first < 0) first = i;
                    last = i;
                }
            return (first, last);
        }

        /// <summary>Where a shot from <paramref name="from"/> toward <paramref name="to"/> may
        /// visually originate: the true muzzle when visible, else the first visible cell along
        /// the line (the fog boundary — real bearing, clamped origin). Null if the whole line is
        /// dark.</summary>
        public static HexCoord? TracerOrigin(GameState state, PlayerId? viewer, HexCoord from, HexCoord to)
        {
            if (!Fogged(state, viewer)) return from;
            foreach (var cell in HexPath.Line(from, to))
                if (CellVisible(state, viewer.Value, cell))
                    return cell;
            return null;
        }
    }
}

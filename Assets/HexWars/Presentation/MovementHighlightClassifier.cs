using HexWars.Engine;

namespace HexWars.Presentation
{
    public enum MovementHighlightKind { None, Reachable, Route, Expensive, Destination }

    public static class MovementHighlightClassifier
    {
        public static MovementHighlightKind Classify(
            GameState state, MovementRoute preview, HexCoord cell, bool isReachable)
        {
            for (int i = 0; i < preview.Cells.Count; i++)
            {
                if (preview.Cells[i] != cell) continue;
                if (i == preview.Cells.Count - 1)
                    return MovementHighlightKind.Destination;
                if (i == 0)
                    return MovementHighlightKind.Route;

                var previous = preview.Cells[i - 1];
                var tile = state.Board.TileAt(cell);
                var previousTile = state.Board.TileAt(previous);
                bool climbs = tile.Elevation > previousTile.Elevation;
                bool expensiveTerrain = state.Config.Terrain(tile.Terrain).MoveCost > 1;
                return climbs || expensiveTerrain
                    ? MovementHighlightKind.Expensive
                    : MovementHighlightKind.Route;
            }

            return isReachable
                ? MovementHighlightKind.Reachable
                : MovementHighlightKind.None;
        }
    }
}

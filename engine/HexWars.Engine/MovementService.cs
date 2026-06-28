using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// Pure movement rules. A move draws from TWO independent per-turn budgets: horizontal
    /// <see cref="UnitStats.Movement"/> (terrain move cost to enter a hex) and vertical
    /// <see cref="UnitStats.VerticalMovement"/> (ascent only; descending and level moves are free).
    /// Because the two budgets are separate constraints, reachability keeps a Pareto frontier of
    /// (horizontal, vertical) cost pairs per column. (Ground-first M1: a unit's elevation = the tile
    /// it stands on, so path costs use tile elevations.)
    /// </summary>
    public static class MovementService
    {
        public static IReadOnlyCollection<HexCoord> ReachableTiles(GameState state, Unit unit)
        {
            var board = state.Board;
            var config = state.Config;
            var occupied = OccupiedCells(state, unit);
            int maxH = unit.Stats.Movement;
            int maxV = unit.Stats.VerticalMovement;

            var reachable = new HashSet<HexCoord>();
            var frontier = new Dictionary<HexCoord, List<(int h, int v)>>();
            var queue = new Queue<(HexCoord coord, int h, int v)>();

            var start = unit.Cell;
            frontier[start] = new List<(int, int)> { (0, 0) };
            queue.Enqueue((start, 0, 0));

            while (queue.Count > 0)
            {
                var (coord, h, v) = queue.Dequeue();
                int fromElev = board.TileAt(coord).Elevation;

                foreach (var n in coord.Neighbors())
                {
                    if (!board.Contains(n) || occupied.Contains(n)) continue;

                    var tile = board.TileAt(n);
                    var terrain = config.Terrain(tile.Terrain);
                    if (!terrain.Passable) continue;

                    int nh = h + terrain.MoveCost;
                    int nv = v + Math.Max(0, tile.Elevation - fromElev);
                    if (nh > maxH || nv > maxV) continue;
                    if (IsDominated(frontier, n, nh, nv)) continue;

                    AddToFrontier(frontier, n, nh, nv);
                    reachable.Add(n);
                    queue.Enqueue((n, nh, nv));
                }
            }

            reachable.Remove(start);
            return reachable;
        }

        private static HashSet<HexCoord> OccupiedCells(GameState state, Unit mover)
        {
            var set = new HashSet<HexCoord>();
            foreach (var player in state.Players)
                foreach (var u in player.UnitsOnBoard)
                    if (u.IsAlive) set.Add(u.Cell);
            set.Remove(mover.Cell);
            return set;
        }

        private static bool IsDominated(Dictionary<HexCoord, List<(int h, int v)>> frontier,
                                        HexCoord coord, int h, int v)
        {
            if (!frontier.TryGetValue(coord, out var list)) return false;
            foreach (var (eh, ev) in list)
                if (eh <= h && ev <= v) return true;
            return false;
        }

        private static void AddToFrontier(Dictionary<HexCoord, List<(int h, int v)>> frontier,
                                          HexCoord coord, int h, int v)
        {
            if (!frontier.TryGetValue(coord, out var list))
            {
                list = new List<(int, int)>();
                frontier[coord] = list;
            }
            list.RemoveAll(p => h <= p.h && v <= p.v);
            list.Add((h, v));
        }
    }
}

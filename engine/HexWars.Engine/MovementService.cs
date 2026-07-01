using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// Pure movement rules. A move draws from TWO independent per-turn budgets: horizontal
    /// <see cref="UnitStats.Movement"/> (terrain move cost to enter a hex) and vertical
    /// <see cref="UnitStats.VerticalMovement"/> (ascent only; descending and level moves are free).
    /// Because the two budgets are separate constraints, reachability keeps a Pareto frontier of
    /// (horizontal, vertical) cost pairs per column. Budgets are per TURN, not per move: hops spend
    /// them incrementally (<see cref="GameState.MovementSpent"/>), so a unit can move, look, and move
    /// again until they run out. (Ground-first M1: a unit's elevation = the tile it stands on, so
    /// path costs use tile elevations.)
    /// </summary>
    public static class MovementService
    {
        public static IReadOnlyCollection<HexCoord> ReachableTiles(GameState state, Unit unit)
            => ReachableCosts(state, unit).Keys;

        /// <summary>Every hex the unit can still reach this turn, with the (horizontal, vertical)
        /// cost the hop would charge against its remaining budgets — the cheapest-horizontal
        /// (then cheapest-vertical) of the Pareto-optimal paths.</summary>
        public static Dictionary<HexCoord, (int H, int V)> ReachableCosts(GameState state, Unit unit)
        {
            var costs = new Dictionary<HexCoord, (int H, int V)>();
            var spent = state.MovementSpent.TryGetValue(unit.Id, out var sp) ? sp : (H: 0, V: 0);
            int maxH = unit.Stats.Movement - spent.H;
            int maxV = unit.Stats.VerticalMovement - spent.V;
            if (maxH <= 0) return costs; // horizontal budget gone — no hop can enter any hex

            var board = state.Board;
            var config = state.Config;
            var occupied = OccupiedCells(state, unit);

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
                    if (!costs.TryGetValue(n, out var best) || nh < best.H || (nh == best.H && nv < best.V))
                        costs[n] = (nh, nv);
                    queue.Enqueue((n, nh, nv));
                }
            }

            costs.Remove(start);
            return costs;
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

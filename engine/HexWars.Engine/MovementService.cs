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
        private sealed class RouteLabel
        {
            public HexCoord Coord { get; }
            public int H { get; }
            public int V { get; }
            public HexCoord[] Cells { get; }

            public RouteLabel(HexCoord coord, int h, int v, HexCoord[] cells)
            {
                Coord = coord;
                H = h;
                V = v;
                Cells = cells;
            }

            public RouteLabel Extend(HexCoord next, int h, int v)
            {
                var cells = new HexCoord[Cells.Length + 1];
                Array.Copy(Cells, cells, Cells.Length);
                cells[cells.Length - 1] = next;
                return new RouteLabel(next, h, v, cells);
            }
        }

        public static Dictionary<HexCoord, MovementRoute> Routes(GameState state, Unit unit)
        {
            var routes = new Dictionary<HexCoord, MovementRoute>();
            var spent = state.MovementSpent.TryGetValue(unit.Id, out var sp) ? sp : (H: 0, V: 0);
            int maxH = unit.Stats.Movement - spent.H;
            int maxV = unit.Stats.VerticalMovement - spent.V;
            if (maxH <= 0) return routes;

            var board = state.Board;
            var occupied = OccupiedCells(state, unit);
            var frontier = new Dictionary<HexCoord, List<RouteLabel>>();
            var queue = new Queue<RouteLabel>();
            var start = new RouteLabel(unit.Cell, 0, 0, new[] { unit.Cell });
            frontier[start.Coord] = new List<RouteLabel> { start };
            queue.Enqueue(start);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (!frontier.TryGetValue(current.Coord, out var active) || !active.Contains(current))
                    continue;

                int fromElevation = board.TileAt(current.Coord).Elevation;
                var neighbors = new List<HexCoord>(current.Coord.Neighbors());
                neighbors.Sort(CompareRouteCoords);
                foreach (var next in neighbors)
                {
                    if (!board.Contains(next) || occupied.Contains(next)) continue;
                    var tile = board.TileAt(next);
                    var terrain = state.Config.Terrain(tile.Terrain);
                    if (!terrain.Passable) continue;

                    int h = current.H + terrain.MoveCost;
                    int v = current.V + Math.Max(0, tile.Elevation - fromElevation);
                    if (h > maxH || v > maxV) continue;

                    var candidate = current.Extend(next, h, v);
                    if (!TryAddRouteLabel(frontier, candidate)) continue;
                    queue.Enqueue(candidate);
                }
            }

            foreach (var pair in frontier)
            {
                if (pair.Key == unit.Cell) continue;
                var best = pair.Value[0];
                for (int i = 1; i < pair.Value.Count; i++)
                    if (CompareRouteLabels(pair.Value[i], best) < 0) best = pair.Value[i];
                routes[pair.Key] = new MovementRoute(best.Cells, best.H, best.V,
                    maxH - best.H, maxV - best.V);
            }
            return routes;
        }

        private static bool TryAddRouteLabel(
            Dictionary<HexCoord, List<RouteLabel>> frontier, RouteLabel candidate)
        {
            if (!frontier.TryGetValue(candidate.Coord, out var labels))
            {
                frontier[candidate.Coord] = new List<RouteLabel> { candidate };
                return true;
            }

            foreach (var existing in labels)
            {
                if (existing.H > candidate.H || existing.V > candidate.V) continue;
                if (existing.H < candidate.H || existing.V < candidate.V
                    || CompareRouteLabels(existing, candidate) <= 0)
                    return false;
            }

            labels.RemoveAll(existing =>
                candidate.H <= existing.H && candidate.V <= existing.V
                && (candidate.H < existing.H || candidate.V < existing.V
                    || CompareRouteLabels(candidate, existing) < 0));
            labels.Add(candidate);
            return true;
        }

        private static int CompareRouteLabels(RouteLabel a, RouteLabel b)
        {
            int comparison = a.H.CompareTo(b.H);
            if (comparison != 0) return comparison;
            comparison = a.V.CompareTo(b.V);
            if (comparison != 0) return comparison;
            comparison = a.Cells.Length.CompareTo(b.Cells.Length);
            if (comparison != 0) return comparison;
            for (int i = 0; i < a.Cells.Length; i++)
            {
                comparison = CompareRouteCoords(a.Cells[i], b.Cells[i]);
                if (comparison != 0) return comparison;
            }
            return 0;
        }

        private static int CompareRouteCoords(HexCoord a, HexCoord b)
        {
            int comparison = a.Q.CompareTo(b.Q);
            return comparison != 0 ? comparison : a.R.CompareTo(b.R);
        }

        public static IReadOnlyCollection<HexCoord> ReachableTiles(GameState state, Unit unit)
            => Routes(state, unit).Keys;

        /// <summary>Compatibility projection for callers that need only movement costs.</summary>
        public static Dictionary<HexCoord, (int H, int V)> ReachableCosts(GameState state, Unit unit)
        {
            var costs = new Dictionary<HexCoord, (int H, int V)>();
            foreach (var pair in Routes(state, unit))
                costs[pair.Key] = (pair.Value.HorizontalCost, pair.Value.VerticalCost);
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
    }
}

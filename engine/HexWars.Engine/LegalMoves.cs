using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine
{
    /// <summary>
    /// Enumerates the discrete legal commands for the active player — the action space for the AI /
    /// RandomAgent and self-play. (CreateUnit is intentionally excluded: its design space is
    /// unbounded, so agents generate unit designs themselves.)
    /// </summary>
    public static class LegalMoves
    {
        public static IReadOnlyList<Command> For(GameState state)
        {
            var moves = new List<Command>();
            if (state.IsGameOver) return moves;

            var me = state.ActivePlayer;
            var player = state.Player(me);
            var enemy = state.Opponent(me);
            var board = state.Board;

            var emptyZone = board.DeploymentZone(me)
                .Where(coord => board.Contains(coord)
                                && state.Config.Terrain(board.TileAt(coord).Terrain).Passable
                                && !IsOccupied(state, coord))
                .ToList();

            for (int i = 0; i < player.Barracks.Count; i++)
            {
                if (player.Points < Economy.DeployCost(player.Barracks[i], state.Config)) continue;
                foreach (var coord in emptyZone)
                    moves.Add(new DeployUnit(me, i, coord));
            }

            // generators removed from the game model: the only income is bounty from kills

            foreach (var unit in player.UnitsOnBoard)
            {
                if (!unit.IsAlive) continue;

                if (!state.MovedUnitIds.Contains(unit.Id))
                    foreach (var dest in MovementService.ReachableTiles(state, unit))
                        moves.Add(new MoveUnit(me, unit.Id, dest));

                if (!state.AttackedUnitIds.Contains(unit.Id))
                {
                    foreach (var t in enemy.UnitsOnBoard)
                        if (t.IsAlive && TargetingService.CanTarget(state, unit, t.Cell, t.Elevation))
                            moves.Add(new AttackUnit(me, unit.Id, t.Id));
                    foreach (var g in enemy.Generators)
                        if (g.IsAlive && TargetingService.CanTarget(state, unit, g.Cell, g.Elevation))
                            moves.Add(new AttackUnit(me, unit.Id, g.Id));
                }
            }

            moves.Add(new EndTurn(me));
            return moves;
        }

        private static bool IsOccupied(GameState state, HexCoord coord)
        {
            foreach (var p in state.Players)
            {
                foreach (var u in p.UnitsOnBoard) if (u.IsAlive && u.Cell == coord) return true;
                foreach (var g in p.Generators) if (g.IsAlive && g.Cell == coord) return true;
            }
            return false;
        }
    }
}

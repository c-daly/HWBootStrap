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

            bool claimLegal = !(state.Config.TerritoryMode && state.Config.ClaimEndsTurn)
                || (state.MovedUnitIds.Count == 0 && state.AttackedUnitIds.Count == 0);

            foreach (var unit in player.UnitsOnBoard)
            {
                if (!unit.IsAlive) continue;

                // ReachableTiles honours the budget already spent by earlier hops this turn
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

                if (claimLegal
                    && board.Controller(unit.Cell) != me
                    && player.Points >= CaptureCostFor(state, unit.Cell))
                    moves.Add(new CaptureHex(me, unit.Cell));

                if (state.Config.GeneratorsEnabled
                    && board.Controller(unit.Cell) == me
                    && !HasGeneratorAt(state, unit.Cell)
                    && player.Points >= BuildCostFor(state.Config))
                    moves.Add(new BuildGenerator(me, unit.Cell));
            }

            // build-anywhere: a generator may go on any empty hex you control, not just under a unit
            if (state.Config.GeneratorsEnabled && state.Config.TerritoryMode && state.Config.BuildAnywhere
                && player.Points >= BuildCostFor(state.Config))
                foreach (var t in board.Tiles)
                    if (board.Controller(t.Coord) == me && !IsOccupied(state, t.Coord))
                        moves.Add(new BuildGenerator(me, t.Coord));

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

        private static bool HasGeneratorAt(GameState state, HexCoord coord)
        {
            foreach (var p in state.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == coord) return true;
            return false;
        }

        private static int BuildCostFor(GameConfig cfg) =>
            (int)System.Math.Round(cfg.BuildFactor * cfg.GeneratorOutput, System.MidpointRounding.AwayFromZero);

        // Mirrors GameEngine.CaptureCostFor so an enumerated CaptureHex is always affordable by the handler.
        private static int CaptureCostFor(GameState state, HexCoord cell)
        {
            int flat = state.Config.CaptureCost;
            foreach (var p in state.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == cell)
                    {
                        int income = (int)System.Math.Round(state.Config.GeneratorOutput * g.Strength,
                                                            System.MidpointRounding.AwayFromZero);
                        int scaled = (int)System.Math.Round(state.Config.CaptureFactor * income,
                                                            System.MidpointRounding.AwayFromZero);
                        return System.Math.Max(flat, scaled);
                    }
            return flat;
        }
    }
}

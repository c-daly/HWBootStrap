using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine
{
    /// <summary>
    /// The single, non-mutating mutation path: <see cref="Apply"/> validates a command against the
    /// state and returns a NEW <see cref="GameState"/> (or a rejection with the input unchanged).
    /// Handlers are added one command at a time (TDD).
    /// </summary>
    public static class GameEngine
    {
        public static Result Apply(GameState state, Command command)
        {
            if (state.IsGameOver) return Result.Reject(state, RejectionReason.GameAlreadyOver);
            if (command.Issuer != state.ActivePlayer) return Result.Reject(state, RejectionReason.NotYourTurn);

            switch (command)
            {
                case CreateUnit c: return ApplyCreateUnit(state, c);
                case DeployGenerator c: return ApplyDeployGenerator(state, c);
                case DeployUnit c: return ApplyDeployUnit(state, c);
                case MoveUnit c: return ApplyMoveUnit(state, c);
                default: return Result.Reject(state, RejectionReason.None);
            }
        }

        private static Result ApplyCreateUnit(GameState state, CreateUnit c)
        {
            if (c.Stats.Health < 1) return Result.Reject(state, RejectionReason.InvalidStats);

            var player = state.Player(c.Issuer);
            int cost = c.Stats.PointCost;
            if (player.Points < cost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var reserve = new List<UnitStats>(player.Reserve) { c.Stats };
            var updated = new PlayerState(player.Id, player.Points - cost, reserve,
                                          player.UnitsOnBoard, player.Generators);
            return Result.Ok(WithPlayer(state, updated));
        }

        private static Result ApplyDeployGenerator(GameState state, DeployGenerator c)
        {
            var board = state.Board;
            if (!board.Contains(c.Cell)) return Result.Reject(state, RejectionReason.TileNotFound);
            if (!board.IsInDeploymentZone(c.Issuer, c.Cell)) return Result.Reject(state, RejectionReason.OutsideDeploymentZone);

            var tile = board.TileAt(c.Cell);
            if (!state.Config.Terrain(tile.Terrain).Passable) return Result.Reject(state, RejectionReason.TileImpassable);
            if (IsOccupied(state, c.Cell)) return Result.Reject(state, RejectionReason.TileOccupied);

            var player = state.Player(c.Issuer);
            if (player.Points < state.Config.GeneratorCost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var gen = new Generator(state.NextEntityId, c.Issuer, c.Cell, tile.Elevation, state.Config.GeneratorHealth);
            var generators = new List<Generator>(player.Generators) { gen };
            var updated = new PlayerState(player.Id, player.Points - state.Config.GeneratorCost,
                                          player.Reserve, player.UnitsOnBoard, generators);
            return Result.Ok(WithPlayer(state, updated, state.NextEntityId + 1));
        }

        private static Result ApplyDeployUnit(GameState state, DeployUnit c)
        {
            var player = state.Player(c.Issuer);
            if (c.ReserveIndex < 0 || c.ReserveIndex >= player.Reserve.Count)
                return Result.Reject(state, RejectionReason.ReserveUnitNotFound);

            var board = state.Board;
            if (!board.Contains(c.Cell)) return Result.Reject(state, RejectionReason.TileNotFound);
            if (!board.IsInDeploymentZone(c.Issuer, c.Cell)) return Result.Reject(state, RejectionReason.OutsideDeploymentZone);

            var tile = board.TileAt(c.Cell);
            if (!state.Config.Terrain(tile.Terrain).Passable) return Result.Reject(state, RejectionReason.TileImpassable);
            if (IsOccupied(state, c.Cell)) return Result.Reject(state, RejectionReason.TileOccupied);

            var stats = player.Reserve[c.ReserveIndex];
            var unit = new Unit(state.NextEntityId, c.Issuer, stats, c.Cell, tile.Elevation);

            var reserve = new List<UnitStats>(player.Reserve);
            reserve.RemoveAt(c.ReserveIndex);
            var units = new List<Unit>(player.UnitsOnBoard) { unit };

            var updated = new PlayerState(player.Id, player.Points, reserve, units, player.Generators);
            return Result.Ok(WithPlayer(state, updated, state.NextEntityId + 1));
        }

        private static Result ApplyMoveUnit(GameState state, MoveUnit c)
        {
            var player = state.Player(c.Issuer);
            int idx = IndexOfLivingUnit(player, c.UnitId);
            if (idx < 0) return Result.Reject(state, RejectionReason.UnitNotFound);
            if (state.MovedUnitIds.Contains(c.UnitId)) return Result.Reject(state, RejectionReason.UnitAlreadyMoved);

            var unit = player.UnitsOnBoard[idx];
            var reachable = MovementService.ReachableTiles(state, unit);
            if (!reachable.Contains(c.Dest)) return Result.Reject(state, RejectionReason.OutOfMovementRange);

            var moved = unit.WithCell(c.Dest, state.Board.TileAt(c.Dest).Elevation);
            var units = new List<Unit>(player.UnitsOnBoard);
            units[idx] = moved;
            var updated = new PlayerState(player.Id, player.Points, player.Reserve, units, player.Generators);

            var movedIds = new HashSet<int>(state.MovedUnitIds) { c.UnitId };
            return Result.Ok(WithPlayer(state, updated, movedUnitIds: movedIds));
        }

        private static int IndexOfLivingUnit(PlayerState player, int unitId)
        {
            for (int i = 0; i < player.UnitsOnBoard.Count; i++)
                if (player.UnitsOnBoard[i].Id == unitId && player.UnitsOnBoard[i].IsAlive) return i;
            return -1;
        }

        /// <summary>True if any living unit or generator (either player) stands on the column.</summary>
        private static bool IsOccupied(GameState state, HexCoord coord)
        {
            foreach (var p in state.Players)
            {
                foreach (var u in p.UnitsOnBoard) if (u.IsAlive && u.Cell == coord) return true;
                foreach (var g in p.Generators) if (g.IsAlive && g.Cell == coord) return true;
            }
            return false;
        }

        /// <summary>Rebuild the state with one player replaced (optionally a new entity-id counter or
        /// acted-tracking sets); all other fields, including the per-turn acted sets, are preserved.</summary>
        private static GameState WithPlayer(GameState state, PlayerState updated, int? nextEntityId = null,
            IReadOnlyCollection<int>? movedUnitIds = null, IReadOnlyCollection<int>? attackedUnitIds = null)
        {
            var players = state.Players.ToArray();
            players[(int)updated.Id] = updated;
            return new GameState(state.Board, state.Config, players, state.ActivePlayer,
                                 state.Round, nextEntityId ?? state.NextEntityId, state.IsGameOver, state.Winner,
                                 movedUnitIds ?? state.MovedUnitIds, attackedUnitIds ?? state.AttackedUnitIds);
        }
    }
}

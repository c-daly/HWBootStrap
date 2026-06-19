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

        /// <summary>Rebuild the state with one player replaced (and optionally a new entity-id counter).</summary>
        private static GameState WithPlayer(GameState state, PlayerState updated, int? nextEntityId = null)
        {
            var players = state.Players.ToArray();
            players[(int)updated.Id] = updated;
            return new GameState(state.Board, state.Config, players, state.ActivePlayer,
                                 state.Round, nextEntityId ?? state.NextEntityId, state.IsGameOver, state.Winner);
        }
    }
}

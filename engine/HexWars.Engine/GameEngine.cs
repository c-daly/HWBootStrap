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

        /// <summary>Rebuild the state with one player replaced (everything else unchanged).</summary>
        private static GameState WithPlayer(GameState state, PlayerState updated)
        {
            var players = state.Players.ToArray();
            players[(int)updated.Id] = updated;
            return new GameState(state.Board, state.Config, players, state.ActivePlayer,
                                 state.Round, state.NextEntityId, state.IsGameOver, state.Winner);
        }
    }
}

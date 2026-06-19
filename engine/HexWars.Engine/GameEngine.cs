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

            var result = Dispatch(state, command);
            if (!result.Success) return result;

            var newState = Finalize(result.NewState);

            // One-action turn policies auto-end the turn after a single non-EndTurn action.
            if (!newState.IsGameOver && !(command is EndTurn)
                && newState.Config.TurnPolicy.AutoEndTurnAfter(command))
            {
                newState = Finalize(ApplyEndTurn(newState, new EndTurn(command.Issuer)).NewState);
            }

            return Result.Ok(newState);
        }

        private static Result Dispatch(GameState state, Command command)
        {
            switch (command)
            {
                case CreateUnit c: return ApplyCreateUnit(state, c);
                case DeployGenerator c: return ApplyDeployGenerator(state, c);
                case DeployUnit c: return ApplyDeployUnit(state, c);
                case MoveUnit c: return ApplyMoveUnit(state, c);
                case AttackUnit c: return ApplyAttackUnit(state, c);
                case EndTurn c: return ApplyEndTurn(state, c);
                default: return Result.Reject(state, RejectionReason.None);
            }
        }

        /// <summary>Win check after every successful command: stamps IsGameOver/Winner if terminal.</summary>
        private static GameState Finalize(GameState s)
        {
            if (s.IsGameOver) return s;
            if (WinCheck.IsTerminal(s)) return WithGameOver(s, WinCheck.Resolve(s));
            return s;
        }

        private static GameState WithGameOver(GameState s, PlayerId? winner) =>
            new GameState(s.Board, s.Config, s.Players, s.ActivePlayer, s.Round, s.NextEntityId,
                          isGameOver: true, winner: winner,
                          movedUnitIds: s.MovedUnitIds, attackedUnitIds: s.AttackedUnitIds);

        private static Result ApplyEndTurn(GameState state, EndTurn c)
        {
            var next = state.ActivePlayer == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            int round = state.Round + (next == PlayerId.Player0 ? 1 : 0);

            int income = Economy.Income(state, next);
            var players = state.Players.ToArray();
            var np = players[(int)next];
            players[(int)next] = np.WithPoints(np.Points + income);

            return Result.Ok(new GameState(state.Board, state.Config, players, next, round,
                state.NextEntityId, state.IsGameOver, state.Winner,
                movedUnitIds: System.Array.Empty<int>(), attackedUnitIds: System.Array.Empty<int>()));
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

        private static int IndexOfLivingGenerator(PlayerState player, int generatorId)
        {
            for (int i = 0; i < player.Generators.Count; i++)
                if (player.Generators[i].Id == generatorId && player.Generators[i].IsAlive) return i;
            return -1;
        }

        private static Result ApplyAttackUnit(GameState state, AttackUnit c)
        {
            var player = state.Player(c.Issuer);
            int aIdx = IndexOfLivingUnit(player, c.AttackerId);
            if (aIdx < 0) return Result.Reject(state, RejectionReason.UnitNotFound);
            if (state.AttackedUnitIds.Contains(c.AttackerId)) return Result.Reject(state, RejectionReason.UnitAlreadyAttacked);

            var attacker = player.UnitsOnBoard[aIdx];
            var enemy = state.Opponent(c.Issuer);

            int tUnitIdx = IndexOfLivingUnit(enemy, c.TargetId);
            int tGenIdx = tUnitIdx < 0 ? IndexOfLivingGenerator(enemy, c.TargetId) : -1;
            if (tUnitIdx < 0 && tGenIdx < 0)
                return Result.Reject(state, RejectionReason.TargetNotEnemy); // unknown or friendly id

            HexCoord targetCell;
            int targetElevation, targetDefense, buildCost;
            if (tUnitIdx >= 0)
            {
                var t = enemy.UnitsOnBoard[tUnitIdx];
                targetCell = t.Cell;
                targetElevation = t.Elevation;
                targetDefense = t.Stats.Defense + state.Config.Terrain(state.Board.TileAt(t.Cell).Terrain).Defense;
                buildCost = t.Stats.PointCost;
            }
            else
            {
                var g = enemy.Generators[tGenIdx];
                targetCell = g.Cell;
                targetElevation = g.Elevation;
                targetDefense = state.Config.Terrain(state.Board.TileAt(g.Cell).Terrain).Defense;
                buildCost = state.Config.GeneratorCost;
            }

            if (!TargetingService.InRange(attacker, targetCell, targetElevation, state.Config))
                return Result.Reject(state, RejectionReason.TargetNotInRange);
            if (!TargetingService.IsVisibleToArmy(state, c.Issuer, targetCell, targetElevation))
                return Result.Reject(state, RejectionReason.TargetNotVisible);

            int damage = CombatResolver.ComputeDamage(attacker.Stats.Damage, attacker.Elevation,
                                                      targetElevation, targetDefense, state.Config);

            PlayerState newEnemy;
            bool killed;
            if (tUnitIdx >= 0)
            {
                var hurt = enemy.UnitsOnBoard[tUnitIdx].WithDamage(damage);
                killed = !hurt.IsAlive;
                var units = new List<Unit>(enemy.UnitsOnBoard);
                if (killed) units.RemoveAt(tUnitIdx); else units[tUnitIdx] = hurt;
                newEnemy = new PlayerState(enemy.Id, enemy.Points, enemy.Reserve, units, enemy.Generators);
            }
            else
            {
                var hurt = enemy.Generators[tGenIdx].WithDamage(damage);
                killed = !hurt.IsAlive;
                var gens = new List<Generator>(enemy.Generators);
                if (killed) gens.RemoveAt(tGenIdx); else gens[tGenIdx] = hurt;
                newEnemy = new PlayerState(enemy.Id, enemy.Points, enemy.Reserve, enemy.UnitsOnBoard, gens);
            }

            var newPlayer = killed
                ? player.WithPoints(player.Points + CombatResolver.Bounty(buildCost, state.Config))
                : player;

            var players = state.Players.ToArray();
            players[(int)newPlayer.Id] = newPlayer;
            players[(int)newEnemy.Id] = newEnemy;
            var attackedIds = new HashSet<int>(state.AttackedUnitIds) { c.AttackerId };

            return Result.Ok(new GameState(state.Board, state.Config, players, state.ActivePlayer,
                state.Round, state.NextEntityId, state.IsGameOver, state.Winner,
                state.MovedUnitIds, attackedIds));
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

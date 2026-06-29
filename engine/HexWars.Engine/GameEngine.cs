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
                && (newState.Config.TurnPolicy.AutoEndTurnAfter(command, newState)
                    || (newState.Config.TerritoryMode && newState.Config.ClaimEndsTurn && command is CaptureHex)))
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
                case CaptureHex c: return ApplyCaptureHex(state, c);
                case BuildGenerator c: return ApplyBuildGenerator(state, c);
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

            int net = Economy.Income(state, next) - Economy.Upkeep(state, next);
            var players = state.Players.ToArray();
            var np = players[(int)next];
            int pts = System.Math.Max(0, np.Points + net);
            // use-it-or-lose-it: a fraction of banked points decays each turn (big hoards bleed fastest)
            if (state.Config.PointDecay > 0)
                pts -= (int)System.Math.Round(pts * state.Config.PointDecay, System.MidpointRounding.AwayFromZero);
            players[(int)next] = np.WithPoints(pts);

            return Result.Ok(new GameState(state.Board, state.Config, players, next, round,
                state.NextEntityId, state.IsGameOver, state.Winner,
                movedUnitIds: System.Array.Empty<int>(), attackedUnitIds: System.Array.Empty<int>()));
        }

        private static Result ApplyCreateUnit(GameState state, CreateUnit c)
        {
            if (c.Stats.Health < 1) return Result.Reject(state, RejectionReason.InvalidStats);

            var player = state.Player(c.Issuer);
            int fee = state.Config.DesignFee;
            if (player.Points < fee) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var barracks = new List<UnitStats>(player.Barracks) { c.Stats }; // reusable template
            var updated = new PlayerState(player.Id, player.Points - fee, barracks,
                                          player.UnitsOnBoard, player.Generators, player.DestroyedValue);
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
                                          player.Barracks, player.UnitsOnBoard, generators, player.DestroyedValue);
            return Result.Ok(WithPlayer(state, updated, state.NextEntityId + 1));
        }

        private static Result ApplyDeployUnit(GameState state, DeployUnit c)
        {
            var player = state.Player(c.Issuer);
            if (c.TemplateIndex < 0 || c.TemplateIndex >= player.Barracks.Count)
                return Result.Reject(state, RejectionReason.TemplateNotFound);

            var board = state.Board;
            if (!board.Contains(c.Cell)) return Result.Reject(state, RejectionReason.TileNotFound);
            if (state.Config.TerritoryMode)
            {
                if (board.Controller(c.Cell) != c.Issuer) return Result.Reject(state, RejectionReason.HexNotControlled);
            }
            else
            {
                if (!board.IsInDeploymentZone(c.Issuer, c.Cell)) return Result.Reject(state, RejectionReason.OutsideDeploymentZone);
            }

            var tile = board.TileAt(c.Cell);
            if (!state.Config.Terrain(tile.Terrain).Passable) return Result.Reject(state, RejectionReason.TileImpassable);
            if (IsOccupied(state, c.Cell)) return Result.Reject(state, RejectionReason.TileOccupied);

            var stats = player.Barracks[c.TemplateIndex];
            int cost = Economy.DeployCost(stats, state.Config);
            if (player.Points < cost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var unit = new Unit(state.NextEntityId, c.Issuer, stats, c.Cell, tile.Elevation);
            var units = new List<Unit>(player.UnitsOnBoard) { unit };

            // barracks is unchanged — the template is reusable
            var updated = new PlayerState(player.Id, player.Points - cost, player.Barracks, units, player.Generators, player.DestroyedValue);
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
            var updated = new PlayerState(player.Id, player.Points, player.Barracks, units, player.Generators, player.DestroyedValue);

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
            if (!TargetingService.HasShot(state, attacker, targetCell, targetElevation))
                return Result.Reject(state, RejectionReason.LineOfSightBlocked); // blocked by a stack & no arc

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
                newEnemy = new PlayerState(enemy.Id, enemy.Points, enemy.Barracks, units, enemy.Generators, enemy.DestroyedValue);
            }
            else
            {
                var hurt = enemy.Generators[tGenIdx].WithDamage(damage);
                killed = !hurt.IsAlive;
                var gens = new List<Generator>(enemy.Generators);
                if (killed) gens.RemoveAt(tGenIdx); else gens[tGenIdx] = hurt;
                newEnemy = new PlayerState(enemy.Id, enemy.Points, enemy.Barracks, enemy.UnitsOnBoard, gens, enemy.DestroyedValue);
            }

            var newPlayer = killed
                ? player.WithPoints(player.Points + CombatResolver.Bounty(buildCost, state.Config))
                       .WithDestroyed(buildCost)
                : player;

            var players = state.Players.ToArray();
            players[(int)newPlayer.Id] = newPlayer;
            players[(int)newEnemy.Id] = newEnemy;
            var attackedIds = new HashSet<int>(state.AttackedUnitIds) { c.AttackerId };

            return Result.Ok(new GameState(state.Board, state.Config, players, state.ActivePlayer,
                state.Round, state.NextEntityId, state.IsGameOver, state.Winner,
                state.MovedUnitIds, attackedIds));
        }

        private static Result ApplyCaptureHex(GameState state, CaptureHex c)
        {
            if (!state.Board.Contains(c.Cell)) return Result.Reject(state, RejectionReason.TileNotFound);

            var player = state.Player(c.Issuer);
            bool hasUnit = false;
            foreach (var u in player.UnitsOnBoard)
                if (u.IsAlive && u.Cell == c.Cell) { hasUnit = true; break; }
            if (!hasUnit) return Result.Reject(state, RejectionReason.NoUnitOnHex);

            if (state.Board.Controller(c.Cell) == c.Issuer)
                return Result.Reject(state, RejectionReason.AlreadyControlled);

            if (state.Config.TerritoryMode && state.Config.ClaimEndsTurn
                && (state.MovedUnitIds.Count > 0 || state.AttackedUnitIds.Count > 0))
                return Result.Reject(state, RejectionReason.MustClaimFirst);

            int cost = CaptureCostFor(state, c.Cell);
            if (player.Points < cost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var players = state.Players.ToArray();

            // steal: transfer an enemy generator on the captured hex to the capturer
            var enemy = state.Opponent(c.Issuer);
            int gi = IndexOfGeneratorAt(enemy, c.Cell);
            if (gi >= 0)
            {
                var stolen = enemy.Generators[gi].WithOwner(c.Issuer);
                var enemyGens = new List<Generator>(enemy.Generators);
                enemyGens.RemoveAt(gi);
                players[(int)enemy.Id] = new PlayerState(enemy.Id, enemy.Points, enemy.Barracks,
                                                         enemy.UnitsOnBoard, enemyGens, enemy.DestroyedValue);
                var myGens = new List<Generator>(player.Generators) { stolen };
                player = new PlayerState(player.Id, player.Points, player.Barracks,
                                         player.UnitsOnBoard, myGens, player.DestroyedValue);
            }

            players[(int)c.Issuer] = player.WithPoints(player.Points - cost);
            var newBoard = state.Board.WithControl(c.Cell, c.Issuer);

            return Result.Ok(new GameState(newBoard, state.Config, players, state.ActivePlayer,
                state.Round, state.NextEntityId, state.IsGameOver, state.Winner,
                state.MovedUnitIds, state.AttackedUnitIds));
        }

        /// <summary>Capture cost for a hex: flat CaptureCost, or — if a generator sits here — scaled by its
        /// income: max(CaptureCost, round(CaptureFactor × round(GeneratorOutput × strength))).</summary>
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

        private static int IndexOfGeneratorAt(PlayerState p, HexCoord cell)
        {
            for (int i = 0; i < p.Generators.Count; i++)
                if (p.Generators[i].IsAlive && p.Generators[i].Cell == cell) return i;
            return -1;
        }

        private static Result ApplyBuildGenerator(GameState state, BuildGenerator c)
        {
            if (!state.Board.Contains(c.Cell)) return Result.Reject(state, RejectionReason.TileNotFound);
            if (state.Board.Controller(c.Cell) != c.Issuer)
                return Result.Reject(state, RejectionReason.HexNotControlled);
            if (HasGeneratorAt(state, c.Cell)) return Result.Reject(state, RejectionReason.TileOccupied);

            var player = state.Player(c.Issuer);
            int cost = BuildCost(state.Config);
            if (player.Points < cost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var tile = state.Board.TileAt(c.Cell);
            var gen = new Generator(state.NextEntityId, c.Issuer, c.Cell, tile.Elevation, state.Config.GeneratorHealth, 1.0);
            var generators = new List<Generator>(player.Generators) { gen };
            var updated = new PlayerState(player.Id, player.Points - cost, player.Barracks,
                                          player.UnitsOnBoard, generators, player.DestroyedValue);
            return Result.Ok(WithPlayer(state, updated, state.NextEntityId + 1));
        }

        /// <summary>Build cost of a full-strength generator = round(BuildFactor × GeneratorOutput).</summary>
        private static int BuildCost(GameConfig cfg) =>
            (int)System.Math.Round(cfg.BuildFactor * cfg.GeneratorOutput, System.MidpointRounding.AwayFromZero);

        private static bool HasGeneratorAt(GameState state, HexCoord coord)
        {
            foreach (var p in state.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == coord) return true;
            return false;
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

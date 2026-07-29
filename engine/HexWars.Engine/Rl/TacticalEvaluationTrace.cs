using System;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>Pure, JSON-serializable diagnostic projection of an accepted tactical transition.</summary>
    public static class TacticalEvaluationTrace
    {
        public static TacticalTraceTransition Project(DuelTransition transition) => new TacticalTraceTransition
        {
            Before = ProjectState(transition.Previous),
            Command = ProjectCommand(transition.Command),
            After = ProjectState(transition.Resulting),
        };

        private static TacticalTraceState ProjectState(GameState state) => new TacticalTraceState
        {
            Round = state.Round,
            ActiveSeat = (int)state.ActivePlayer,
            IsGameOver = state.IsGameOver,
            Winner = state.Winner == null ? (int?)null : (int)state.Winner.Value,
            ProductiveLegalActions = LegalMoves.For(state).Count(command => !(command is EndTurn)),
            Seats = new[] { ProjectSeat(state, PlayerId.Player0), ProjectSeat(state, PlayerId.Player1) },
            ControlledHexes = state.Board.Tiles
                .Select(tile => new { Cell = tile.Coord, Owner = state.Board.Controller(tile.Coord) })
                .Where(item => item.Owner.HasValue)
                .OrderBy(item => item.Cell.Q)
                .ThenBy(item => item.Cell.R)
                .Select(item => new TacticalTraceControl
                {
                    Q = item.Cell.Q, R = item.Cell.R, Controller = (int)item.Owner!.Value,
                })
                .ToArray(),
        };

        private static TacticalTraceSeat ProjectSeat(GameState state, PlayerId seat)
        {
            PlayerState player = state.Player(seat);
            PlayerState enemy = state.Opponent(seat);
            Unit[] units = player.UnitsOnBoard.OrderBy(unit => unit.Id).ToArray();
            bool enemyHasLivingUnit = enemy.UnitsOnBoard.Any(unit => unit.IsAlive);
            double healthAdjustedMaterial = 0;
            foreach (Unit unit in units)
                healthAdjustedMaterial += unit.Stats.PointCost * unit.CurrentHp
                    / (double)Math.Max(1, unit.Stats.Health);

            return new TacticalTraceSeat
            {
                Seat = (int)seat,
                Points = player.Points,
                DestroyedValue = player.DestroyedValue,
                AliveUnits = units.Count(unit => unit.IsAlive),
                CurrentHitPoints = units.Sum(unit => unit.CurrentHp),
                MaximumHitPoints = units.Sum(unit => unit.Stats.Health),
                HealthAdjustedMaterial = healthAdjustedMaterial,
                CanDamageEnemy = enemyHasLivingUnit
                    && units.Any(unit => unit.IsAlive && unit.Stats.Damage > 0),
                CanCurrentlyAttackEnemy = units.Any(attacker => attacker.IsAlive
                    && enemy.UnitsOnBoard.Any(target => target.IsAlive
                        && TargetingService.CanTarget(state, attacker, target.Cell, target.Elevation))),
                CanMove = units.Any(unit => unit.IsAlive
                    && (unit.Stats.Movement > 0 || unit.Stats.VerticalMovement > 0)),
                Units = units.Select(unit => ProjectUnit(state, unit)).ToArray(),
            };
        }

        private static TacticalTraceUnit ProjectUnit(GameState state, Unit unit)
        {
            (int H, int V) spent = state.MovementSpent.TryGetValue(unit.Id, out var value)
                ? value
                : (0, 0);
            return new TacticalTraceUnit
            {
                Id = unit.Id,
                Q = unit.Cell.Q,
                R = unit.Cell.R,
                CurrentHp = unit.CurrentHp,
                MaximumHp = unit.Stats.Health,
                PointCost = unit.Stats.PointCost,
                Damage = unit.Stats.Damage,
                Defense = unit.Stats.Defense,
                Movement = unit.Stats.Movement,
                VerticalMovement = unit.Stats.VerticalMovement,
                Range = unit.Stats.Range,
                Moved = state.MovedUnitIds.Contains(unit.Id),
                Attacked = state.AttackedUnitIds.Contains(unit.Id),
                MovementSpentH = spent.H,
                MovementSpentV = spent.V,
            };
        }

        private static TacticalTraceCommand ProjectCommand(Command command)
        {
            var trace = new TacticalTraceCommand { Issuer = (int)command.Issuer };
            switch (command)
            {
                case EndTurn:
                    trace.Kind = "end_turn";
                    break;
                case MoveUnit move:
                    trace.Kind = "move";
                    trace.ActorId = move.UnitId;
                    trace.Q = move.Dest.Q;
                    trace.R = move.Dest.R;
                    break;
                case AttackUnit attack:
                    trace.Kind = "attack";
                    trace.ActorId = attack.AttackerId;
                    trace.TargetId = attack.TargetId;
                    break;
                case DeployUnit deploy:
                    trace.Kind = "deploy";
                    trace.Q = deploy.Cell.Q;
                    trace.R = deploy.Cell.R;
                    break;
                case CaptureHex capture:
                    trace.Kind = "capture";
                    trace.Q = capture.Cell.Q;
                    trace.R = capture.Cell.R;
                    break;
                case BuildGenerator build:
                    trace.Kind = "build_generator";
                    trace.Q = build.Cell.Q;
                    trace.R = build.Cell.R;
                    break;
                default:
                    trace.Kind = command.GetType().Name;
                    break;
            }
            return trace;
        }
    }

    public sealed class TacticalTraceTransition
    {
        public TacticalTraceState Before { get; set; } = null!;
        public TacticalTraceCommand Command { get; set; } = null!;
        public TacticalTraceState After { get; set; } = null!;
    }

    public sealed class TacticalTraceState
    {
        public int Round { get; set; }
        public int ActiveSeat { get; set; }
        public bool IsGameOver { get; set; }
        public int? Winner { get; set; }
        public int ProductiveLegalActions { get; set; }
        public TacticalTraceSeat[] Seats { get; set; } = Array.Empty<TacticalTraceSeat>();
        public TacticalTraceControl[] ControlledHexes { get; set; } = Array.Empty<TacticalTraceControl>();
    }

    public sealed class TacticalTraceControl
    {
        public int Q { get; set; }
        public int R { get; set; }
        public int Controller { get; set; }
    }


    public sealed class TacticalTraceSeat
    {
        public int Seat { get; set; }
        public int Points { get; set; }
        public int DestroyedValue { get; set; }
        public int AliveUnits { get; set; }
        public int CurrentHitPoints { get; set; }
        public int MaximumHitPoints { get; set; }
        public double HealthAdjustedMaterial { get; set; }
        public bool CanDamageEnemy { get; set; }
        public bool CanCurrentlyAttackEnemy { get; set; }
        public bool CanMove { get; set; }
        public TacticalTraceUnit[] Units { get; set; } = Array.Empty<TacticalTraceUnit>();
    }

    public sealed class TacticalTraceUnit
    {
        public int Id { get; set; }
        public int Q { get; set; }
        public int R { get; set; }
        public int CurrentHp { get; set; }
        public int MaximumHp { get; set; }
        public int PointCost { get; set; }
        public int Damage { get; set; }
        public int Defense { get; set; }
        public int Movement { get; set; }
        public int VerticalMovement { get; set; }
        public int Range { get; set; }
        public bool Moved { get; set; }
        public bool Attacked { get; set; }
        public int MovementSpentH { get; set; }
        public int MovementSpentV { get; set; }
    }

    public sealed class TacticalTraceCommand
    {
        public string Kind { get; set; } = "";
        public int Issuer { get; set; }
        public int? ActorId { get; set; }
        public int? TargetId { get; set; }
        public int? Q { get; set; }
        public int? R { get; set; }
    }
}

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using HexWars.Engine;

namespace HexWars.Engine.Rl
{
    public enum TacticalV3CandidateKind
    {
        Attack = 0,
        Move = 1,
        Deploy = 2,
        EndTurn = 3,
    }

    public sealed class TacticalV3ProjectedDelta
    {
        public TacticalV3ProjectedDelta(
            TacticalV3TokenRef? sourceCell,
            TacticalV3TokenRef? destinationCell,
            TacticalV3TokenRef? template,
            TacticalV3TokenRef? target,
            int horizontalMovementSpent,
            int verticalMovementSpent,
            int targetHpDelta,
            int damage,
            bool isLethal,
            int bountyDelta,
            int pointsDelta,
            int roundDelta,
            bool isTerminal)
        {
            SourceCell = sourceCell;
            DestinationCell = destinationCell;
            Template = template;
            Target = target;
            HorizontalMovementSpent = horizontalMovementSpent;
            VerticalMovementSpent = verticalMovementSpent;
            TargetHpDelta = targetHpDelta;
            Damage = damage;
            IsLethal = isLethal;
            BountyDelta = bountyDelta;
            PointsDelta = pointsDelta;
            RoundDelta = roundDelta;
            IsTerminal = isTerminal;
        }

        public TacticalV3TokenRef? SourceCell { get; }
        public TacticalV3TokenRef? DestinationCell { get; }
        public TacticalV3TokenRef? Template { get; }
        public TacticalV3TokenRef? Target { get; }
        public int HorizontalMovementSpent { get; }
        public int VerticalMovementSpent { get; }
        public int TargetHpDelta { get; }
        public int Damage { get; }
        public bool IsLethal { get; }
        public int BountyDelta { get; }
        public int PointsDelta { get; }
        public int RoundDelta { get; }
        public bool IsTerminal { get; }
    }

    public sealed class TacticalV3Candidate
    {
        public TacticalV3Candidate(
            int candidateId,
            long decisionId,
            TacticalV3CandidateKind kind,
            TacticalV3TokenRef? actor,
            TacticalV3TokenRef? target,
            TacticalV3TokenRef? template,
            TacticalV3TokenRef? cell,
            TacticalV3ProjectedDelta projection)
        {
            CandidateId = candidateId;
            DecisionId = decisionId;
            Kind = kind;
            Actor = actor;
            Target = target;
            Template = template;
            Cell = cell;
            Projection = projection ?? throw new ArgumentNullException(nameof(projection));
        }

        public int CandidateId { get; }
        public long DecisionId { get; }
        public TacticalV3CandidateKind Kind { get; }
        public TacticalV3TokenRef? Actor { get; }
        public TacticalV3TokenRef? Target { get; }
        public TacticalV3TokenRef? Template { get; }
        public TacticalV3TokenRef? Cell { get; }
        public TacticalV3ProjectedDelta Projection { get; }
    }

    public interface ICandidateProjector
    {
        TacticalV3ProjectedDelta Project(
            GameState state, PlayerId seat, Command command, TacticalV3Observation observation);
    }

    public interface ILegalCandidateSource
    {
        TacticalV3DecisionFrame CreateFrame(
            GameState state, PlayerId seat, IObservationMemory memory, long decisionId);
    }

    public interface IActionResolver
    {
        Command Resolve(
            TacticalV3DecisionFrame frame,
            long decisionId,
            int candidateId,
            GameState currentState);
    }

    public sealed class TacticalV3DecisionFrame
    {
        private readonly GameState _sourceState;
        private readonly IReadOnlyList<Command> _commands;

        internal TacticalV3DecisionFrame(
            GameState sourceState,
            long decisionId,
            PlayerId seat,
            TacticalV3Observation observation,
            IReadOnlyList<TacticalV3Candidate> candidates,
            IReadOnlyList<Command> commands)
        {
            _sourceState = sourceState ?? throw new ArgumentNullException(nameof(sourceState));
            DecisionId = decisionId;
            Seat = seat;
            Observation = observation ?? throw new ArgumentNullException(nameof(observation));
            Candidates = new ReadOnlyCollection<TacticalV3Candidate>(
                (candidates ?? throw new ArgumentNullException(nameof(candidates))).ToArray());
            _commands = new ReadOnlyCollection<Command>(
                (commands ?? throw new ArgumentNullException(nameof(commands))).ToArray());
            if (Candidates.Count != _commands.Count)
                throw new ArgumentException("candidate and command rows must have equal counts");
        }

        public long DecisionId { get; }
        public PlayerId Seat { get; }
        public TacticalV3Observation Observation { get; }
        public IReadOnlyList<TacticalV3Candidate> Candidates { get; }

        internal bool IsFor(GameState state) => ReferenceEquals(_sourceState, state);
        internal Command CommandAt(int candidateId) => _commands[candidateId];

        internal int RequireUniqueCandidateId(Command selected)
        {
            if (selected == null) throw new ArgumentNullException(nameof(selected));

            int candidateId = -1;
            int matchCount = 0;
            for (int index = 0; index < _commands.Count; index++)
            {
                if (!_commands[index].Equals(selected)) continue;
                candidateId = Candidates[index].CandidateId;
                matchCount++;
            }

            if (matchCount != 1)
                throw new InvalidOperationException(
                    "teacher command must match exactly one tactical-v3 candidate");
            return candidateId;
        }
    }

    public sealed class TacticalV3CandidateProjector : ICandidateProjector
    {
        private readonly Func<GameState, PlayerId, Command, bool>? _episodeTerminal;

        public TacticalV3CandidateProjector()
        {
        }

        internal TacticalV3CandidateProjector(
            Func<GameState, PlayerId, Command, bool> episodeTerminal)
        {
            _episodeTerminal = episodeTerminal ??
                throw new ArgumentNullException(nameof(episodeTerminal));
        }

        public TacticalV3ProjectedDelta Project(
            GameState state, PlayerId seat, Command command, TacticalV3Observation observation)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (command == null) throw new ArgumentNullException(nameof(command));
            if (observation == null) throw new ArgumentNullException(nameof(observation));
            if (seat != PlayerId.Player0 && seat != PlayerId.Player1)
                throw new ArgumentOutOfRangeException(nameof(seat));

            CandidateReferences references = CandidateReferences.For(state, seat, observation);
            Result applied = GameEngine.Apply(state, command);
            if (!applied.Success)
                throw new InvalidOperationException(
                    "tactical-v3 projection command was rejected: " + applied.Reason);

            GameState after = applied.NewState;
            TacticalV3TokenRef? sourceCell = null;
            TacticalV3TokenRef? destinationCell = null;
            TacticalV3TokenRef? template = null;
            TacticalV3TokenRef? target = null;
            int horizontalMovementSpent = 0;
            int verticalMovementSpent = 0;
            int targetHpDelta = 0;
            int damage = 0;
            bool isLethal = false;
            int bountyDelta = 0;

            if (command is MoveUnit move)
            {
                Unit unit = FindLivingUnit(state.Player(seat), move.UnitId);
                sourceCell = references.Cell(unit.Cell);
                destinationCell = references.Cell(move.Dest);
                (int beforeHorizontal, int beforeVertical) = MovementSpend(state, move.UnitId);
                (int afterHorizontal, int afterVertical) = MovementSpend(after, move.UnitId);
                horizontalMovementSpent = afterHorizontal - beforeHorizontal;
                verticalMovementSpent = afterVertical - beforeVertical;
            }
            else if (command is AttackUnit attack)
            {
                Unit attacker = FindLivingUnit(state.Player(seat), attack.AttackerId);
                Unit victim = FindLivingUnit(state.Opponent(seat), attack.TargetId);
                target = references.Unit(attack.TargetId);
                sourceCell = references.Cell(attacker.Cell);
                int afterHp = FindUnitOrZero(after.Opponent(seat), attack.TargetId);
                targetHpDelta = afterHp - victim.CurrentHp;
                damage = victim.CurrentHp - afterHp;
                isLethal = afterHp == 0;
                if (isLethal)
                    bountyDelta = after.Player(seat).Points - state.Player(seat).Points;
            }
            else if (command is DeployUnit deploy)
            {
                template = references.Template(deploy.TemplateIndex);
                destinationCell = references.Cell(deploy.Cell);
            }
            else if (!(command is EndTurn))
            {
                throw new NotSupportedException(
                    "tactical-v3 does not support legal command type " + command.GetType().Name);
            }

            return new TacticalV3ProjectedDelta(
                sourceCell, destinationCell, template, target,
                horizontalMovementSpent, verticalMovementSpent,
                targetHpDelta, damage, isLethal, bountyDelta,
                after.Player(seat).Points - state.Player(seat).Points,
                after.Round - state.Round,
                after.IsGameOver || (_episodeTerminal?.Invoke(after, seat, command) ?? false));
        }

        private static (int Horizontal, int Vertical) MovementSpend(GameState state, int unitId) =>
            state.MovementSpent.TryGetValue(unitId, out var spent) ? (spent.H, spent.V) : (0, 0);

        private static Unit FindLivingUnit(PlayerState player, int unitId)
        {
            foreach (Unit unit in player.UnitsOnBoard)
                if (unit.Id == unitId && unit.IsAlive) return unit;
            throw new InvalidOperationException("legal command referenced a missing living unit");
        }

        private static int FindUnitOrZero(PlayerState player, int unitId)
        {
            foreach (Unit unit in player.UnitsOnBoard)
                if (unit.Id == unitId && unit.IsAlive) return unit.CurrentHp;
            return 0;
        }
    }

    public sealed class TacticalV3LegalCandidateSource : ILegalCandidateSource
    {
        private readonly ISeatObservationSource _observations;
        private readonly ICandidateProjector _projector;
        private readonly TacticalV3CapacityProfile _capacity;
        private readonly Func<GameState, PlayerId, HexCoord?>? _moveTarget;

        public TacticalV3LegalCandidateSource(
            ISeatObservationSource observations,
            TacticalV3CapacityProfile capacity)
            : this(observations, new TacticalV3CandidateProjector(), capacity, null)
        {
        }

        public TacticalV3LegalCandidateSource(
            ISeatObservationSource observations,
            ICandidateProjector projector,
            TacticalV3CapacityProfile capacity)
            : this(observations, projector, capacity, null)
        {
        }

        internal TacticalV3LegalCandidateSource(
            ISeatObservationSource observations,
            TacticalV3CapacityProfile capacity,
            Func<GameState, PlayerId, HexCoord?> moveTarget,
            Func<GameState, PlayerId, Command, bool> episodeTerminal)
            : this(
                observations,
                new TacticalV3CandidateProjector(episodeTerminal),
                capacity,
                moveTarget)
        {
        }

        private TacticalV3LegalCandidateSource(
            ISeatObservationSource observations,
            ICandidateProjector projector,
            TacticalV3CapacityProfile capacity,
            Func<GameState, PlayerId, HexCoord?>? moveTarget)
        {
            _observations = observations ?? throw new ArgumentNullException(nameof(observations));
            _projector = projector ?? throw new ArgumentNullException(nameof(projector));
            _capacity = capacity ?? throw new ArgumentNullException(nameof(capacity));
            _moveTarget = moveTarget;
        }

        public TacticalV3DecisionFrame CreateFrame(
            GameState state, PlayerId seat, IObservationMemory memory, long decisionId)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (memory == null) throw new ArgumentNullException(nameof(memory));
            if (seat != PlayerId.Player0 && seat != PlayerId.Player1)
                throw new ArgumentOutOfRangeException(nameof(seat));
            if (decisionId < 0) throw new ArgumentOutOfRangeException(nameof(decisionId));
            if (state.ActivePlayer != seat)
                throw new InvalidOperationException("tactical-v3 candidates require the active seat");

            TacticalV3Observation observation = _observations.Observe(state, seat, memory);
            CandidateReferences references = CandidateReferences.For(state, seat, observation);
            HexCoord? targetCell = _moveTarget?.Invoke(state, seat);
            TacticalV3TokenRef? moveTarget = targetCell.HasValue
                ? references.Cell(targetCell.Value)
                : (TacticalV3TokenRef?)null;
            List<Command> commands = new List<Command>(LegalMoves.For(state));
            foreach (Command command in commands)
                RequireSupported(command);
            commands.Sort((left, right) => CompareCommands(left, right, references));

            if (commands.Count > _capacity.MaxCandidates)
                throw new InvalidOperationException(
                    "tactical-v3 candidate capacity exceeded: " + commands.Count +
                    " > " + _capacity.MaxCandidates);

            var candidates = new List<TacticalV3Candidate>(commands.Count);
            for (int index = 0; index < commands.Count; index++)
            {
                Command command = commands[index];
                candidates.Add(ToCandidate(
                    index, decisionId, command, references, moveTarget,
                    _projector.Project(state, seat, command, observation)));
            }

            return new TacticalV3DecisionFrame(
                state, decisionId, seat, observation, candidates, commands);
        }

        private static TacticalV3Candidate ToCandidate(
            int candidateId,
            long decisionId,
            Command command,
            CandidateReferences references,
            TacticalV3TokenRef? moveTarget,
            TacticalV3ProjectedDelta projection)
        {
            if (command is AttackUnit attack)
                return new TacticalV3Candidate(
                    candidateId, decisionId, TacticalV3CandidateKind.Attack,
                    references.Unit(attack.AttackerId), references.Unit(attack.TargetId),
                    null, null, projection);
            if (command is MoveUnit move)
                return new TacticalV3Candidate(
                    candidateId, decisionId, TacticalV3CandidateKind.Move,
                    references.Unit(move.UnitId), moveTarget, null,
                    references.Cell(move.Dest), projection);
            if (command is DeployUnit deploy)
                return new TacticalV3Candidate(
                    candidateId, decisionId, TacticalV3CandidateKind.Deploy,
                    null, null, references.Template(deploy.TemplateIndex), references.Cell(deploy.Cell), projection);
            if (command is EndTurn)
                return new TacticalV3Candidate(
                    candidateId, decisionId, TacticalV3CandidateKind.EndTurn,
                    null, null, null, null, projection);
            throw new NotSupportedException(
                "tactical-v3 does not support legal command type " + command.GetType().Name);
        }

        private static void RequireSupported(Command command)
        {
            if (!(command is AttackUnit) &&
                !(command is MoveUnit) &&
                !(command is DeployUnit) &&
                !(command is EndTurn))
            {
                throw new NotSupportedException(
                    "tactical-v3 does not support legal command type " + command.GetType().Name);
            }
        }

        private static int CompareCommands(Command left, Command right, CandidateReferences references)
        {
            int byKind = Kind(left).CompareTo(Kind(right));
            if (byKind != 0) return byKind;

            if (left is AttackUnit leftAttack && right is AttackUnit rightAttack)
            {
                int byActor = references.Unit(leftAttack.AttackerId).Row.CompareTo(
                    references.Unit(rightAttack.AttackerId).Row);
                return byActor != 0 ? byActor : references.Unit(leftAttack.TargetId).Row.CompareTo(
                    references.Unit(rightAttack.TargetId).Row);
            }
            if (left is MoveUnit leftMove && right is MoveUnit rightMove)
            {
                int byActor = references.Unit(leftMove.UnitId).Row.CompareTo(
                    references.Unit(rightMove.UnitId).Row);
                return byActor != 0 ? byActor : references.Cell(leftMove.Dest).Row.CompareTo(
                    references.Cell(rightMove.Dest).Row);
            }
            if (left is DeployUnit leftDeploy && right is DeployUnit rightDeploy)
            {
                int byTemplate = references.Template(leftDeploy.TemplateIndex).Row.CompareTo(
                    references.Template(rightDeploy.TemplateIndex).Row);
                return byTemplate != 0 ? byTemplate : references.Cell(leftDeploy.Cell).Row.CompareTo(
                    references.Cell(rightDeploy.Cell).Row);
            }
            return 0;
        }

        private static TacticalV3CandidateKind Kind(Command command)
        {
            if (command is AttackUnit) return TacticalV3CandidateKind.Attack;
            if (command is MoveUnit) return TacticalV3CandidateKind.Move;
            if (command is DeployUnit) return TacticalV3CandidateKind.Deploy;
            if (command is EndTurn) return TacticalV3CandidateKind.EndTurn;
            throw new NotSupportedException(
                "tactical-v3 does not support legal command type " + command.GetType().Name);
        }

    }

    public sealed class TacticalV3ActionResolver : IActionResolver
    {
        public Command Resolve(
            TacticalV3DecisionFrame frame,
            long decisionId,
            int candidateId,
            GameState currentState)
        {
            if (frame == null) throw new ArgumentNullException(nameof(frame));
            if (currentState == null) throw new ArgumentNullException(nameof(currentState));
            if (!frame.IsFor(currentState))
                throw new InvalidOperationException("tactical-v3 candidate frame is stale");
            if (candidateId < 0 || candidateId >= frame.Candidates.Count)
                throw new ArgumentOutOfRangeException(nameof(candidateId));
            if (decisionId != frame.DecisionId)
                throw new InvalidOperationException("tactical-v3 decision id is stale");

            TacticalV3Candidate candidate = frame.Candidates[candidateId];
            if (candidate.CandidateId != candidateId || candidate.DecisionId != frame.DecisionId)
                throw new InvalidOperationException("tactical-v3 candidate decision id is stale");

            Command command = frame.CommandAt(candidateId);
            if (!LegalMoves.For(currentState).Contains(command))
                throw new InvalidOperationException("tactical-v3 candidate no longer round-trips");
            return command;
        }
    }

    internal sealed class CandidateReferences
    {
        private readonly IReadOnlyDictionary<int, TacticalV3TokenRef> _units;
        private readonly IReadOnlyDictionary<int, TacticalV3TokenRef> _templates;
        private readonly IReadOnlyDictionary<HexCoord, TacticalV3TokenRef> _cells;

        private CandidateReferences(
            IReadOnlyDictionary<int, TacticalV3TokenRef> units,
            IReadOnlyDictionary<int, TacticalV3TokenRef> templates,
            IReadOnlyDictionary<HexCoord, TacticalV3TokenRef> cells)
        {
            _units = units;
            _templates = templates;
            _cells = cells;
        }

        public static CandidateReferences For(
            GameState state, PlayerId seat, TacticalV3Observation observation)
        {
            if (observation.Cells.Count != state.Board.TileCount)
                throw new InvalidOperationException("tactical-v3 observation cell rows do not match the board");

            var units = new Dictionary<int, TacticalV3TokenRef>();
            int unitRow = 0;
            AddUnits(state.Player(seat), units, ref unitRow);
            AddUnits(state.Opponent(seat), units, ref unitRow);
            if (unitRow != observation.Units.Count)
                throw new InvalidOperationException("tactical-v3 observation unit rows do not match the state");

            var templates = new Dictionary<int, TacticalV3TokenRef>();
            PlayerState player = state.Player(seat);
            for (int index = 0; index < player.Barracks.Count; index++)
                templates.Add(index, new TacticalV3TokenRef(TacticalV3TableKind.Templates, index));
            if (player.Barracks.Count > observation.Templates.Count)
                throw new InvalidOperationException("tactical-v3 observation template rows do not match the state");

            var cells = new Dictionary<HexCoord, TacticalV3TokenRef>();
            foreach (Tile tile in state.Board.Tiles)
                cells.Add(tile.Coord, observation.CellReference(tile.Coord));
            return new CandidateReferences(units, templates, cells);
        }

        public TacticalV3TokenRef Unit(int unitId) =>
            _units.TryGetValue(unitId, out TacticalV3TokenRef value)
                ? value
                : throw new InvalidOperationException("legal command referenced a unit outside the observation");

        public TacticalV3TokenRef Template(int templateIndex) =>
            _templates.TryGetValue(templateIndex, out TacticalV3TokenRef value)
                ? value
                : throw new InvalidOperationException("legal command referenced a template outside the observation");

        public TacticalV3TokenRef Cell(HexCoord cell) =>
            _cells.TryGetValue(cell, out TacticalV3TokenRef value)
                ? value
                : throw new InvalidOperationException("legal command referenced a cell outside the observation");

        private static void AddUnits(
            PlayerState player, IDictionary<int, TacticalV3TokenRef> rows, ref int row)
        {
            foreach (Unit unit in player.UnitsOnBoard.Where(unit => unit.IsAlive).OrderBy(unit => unit.Id))
                rows.Add(unit.Id, new TacticalV3TokenRef(TacticalV3TableKind.Units, row++));
        }
    }
}

using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace HexWars.Engine.Rl
{
    public interface IOraclePreflightBenchmarkSink
    {
        void Reset();
        void Accepted(OraclePreflightBenchmarkRecord record);
    }

    public sealed class BufferedOraclePreflightBenchmarkSink : IOraclePreflightBenchmarkSink
    {
        private readonly List<OraclePreflightBenchmarkRecord> _items =
            new List<OraclePreflightBenchmarkRecord>();

        public void Reset() => _items.Clear();

        public void Accepted(OraclePreflightBenchmarkRecord record)
        {
            if (record == null) throw new ArgumentNullException(nameof(record));
            _items.Add(record);
        }

        public IReadOnlyList<OraclePreflightBenchmarkRecord> Drain()
        {
            var result = new List<OraclePreflightBenchmarkRecord>(_items);
            _items.Clear();
            return result;
        }
    }

    public sealed class OraclePreflightBenchmarkRecord
    {
        private readonly float[] _observation;
        private readonly bool[] _legalMask;
        private readonly TacticalTraceState _state;
        private readonly TacticalV2OracleDecision _first;
        private readonly TacticalV2OracleDecision _second;

        public OraclePreflightBenchmarkRecord(string stateHash, int decisionIndex,
            float[] observation, bool[] legalMask, TacticalTraceState state,
            TacticalV2OracleDecision first, TacticalV2OracleDecision second,
            long firstElapsedTicks, long secondElapsedTicks, long clockFrequency)
        {
            if (string.IsNullOrWhiteSpace(stateHash))
                throw new ArgumentException("state hash must not be empty", nameof(stateHash));
            if (decisionIndex < 0) throw new ArgumentOutOfRangeException(nameof(decisionIndex));
            if (observation == null) throw new ArgumentNullException(nameof(observation));
            if (legalMask == null) throw new ArgumentNullException(nameof(legalMask));
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (first == null) throw new ArgumentNullException(nameof(first));
            if (second == null) throw new ArgumentNullException(nameof(second));
            if (firstElapsedTicks < 0) throw new ArgumentOutOfRangeException(nameof(firstElapsedTicks));
            if (secondElapsedTicks < 0) throw new ArgumentOutOfRangeException(nameof(secondElapsedTicks));
            if (clockFrequency < 1) throw new ArgumentOutOfRangeException(nameof(clockFrequency));
            if (!SameSemantics(first, second))
                throw new ArgumentException("repeated oracle decisions must be identical", nameof(second));
            if (first.Action >= legalMask.Length || !legalMask[first.Action])
                throw new ArgumentException("oracle action is masked", nameof(first));

            StateHash = stateHash;
            DecisionIndex = decisionIndex;
            _observation = (float[])observation.Clone();
            _legalMask = (bool[])legalMask.Clone();
            _state = CopyState(state);
            _first = CopyDecision(first);
            _second = CopyDecision(second);
            FirstElapsedTicks = firstElapsedTicks;
            SecondElapsedTicks = secondElapsedTicks;
            ClockFrequency = clockFrequency;
        }

        public string StateHash { get; }
        public int DecisionIndex { get; }
        public float[] Observation => (float[])_observation.Clone();
        public bool[] LegalMask => (bool[])_legalMask.Clone();
        public TacticalTraceState State => CopyState(_state);
        public TacticalV2OracleDecision First => CopyDecision(_first);
        public TacticalV2OracleDecision Second => CopyDecision(_second);
        public long FirstElapsedTicks { get; }
        public long SecondElapsedTicks { get; }
        public long ClockFrequency { get; }

        internal static bool SameSemantics(TacticalV2OracleDecision first,
            TacticalV2OracleDecision second) =>
            first.Action == second.Action && first.Command.Equals(second.Command) &&
            first.Depth == second.Depth && first.ExpansionBudget == second.ExpansionBudget &&
            string.Equals(first.HeuristicIdentity, second.HeuristicIdentity,
                StringComparison.Ordinal) &&
            first.ActualExpansionCount == second.ActualExpansionCount;

        internal static TacticalV2OracleDecision CopyDecision(TacticalV2OracleDecision source) =>
            new TacticalV2OracleDecision(source.Action, CopyCommand(source.Command), source.Depth,
                source.ExpansionBudget, source.HeuristicIdentity, source.ActualExpansionCount);

        internal static Command CopyCommand(Command source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            return source switch
            {
                CreateUnit create => new CreateUnit(create.Issuer, create.Stats, create.Name),
                ReplaceTemplate replace => new ReplaceTemplate(replace.Issuer, replace.TemplateIndex,
                    replace.Stats, replace.Name),
                DeleteTemplate delete => new DeleteTemplate(delete.Issuer, delete.TemplateIndex),
                DeployGenerator generator => new DeployGenerator(generator.Issuer, generator.Cell),
                DeployUnit deploy => new DeployUnit(deploy.Issuer, deploy.TemplateIndex, deploy.Cell),
                MoveUnit move => new MoveUnit(move.Issuer, move.UnitId, move.Dest),
                AttackUnit attack => new AttackUnit(attack.Issuer, attack.AttackerId, attack.TargetId),
                CaptureHex capture => new CaptureHex(capture.Issuer, capture.Cell),
                BuildGenerator build => new BuildGenerator(build.Issuer, build.Cell),
                EndTurn end => new EndTurn(end.Issuer),
                _ => throw new ArgumentException("unsupported command type", nameof(source)),
            };
        }

        internal static TacticalTraceState CopyState(TacticalTraceState source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (source.Seats == null || source.ControlledHexes == null)
                throw new ArgumentException("state projection collections must not be null", nameof(source));
            return new TacticalTraceState
            {
                Round = source.Round,
                ActiveSeat = source.ActiveSeat,
                IsGameOver = source.IsGameOver,
                Winner = source.Winner,
                ProductiveLegalActions = source.ProductiveLegalActions,
                Seats = CopySeats(source.Seats),
                ControlledHexes = CopyControls(source.ControlledHexes),
            };
        }

        private static TacticalTraceSeat[] CopySeats(TacticalTraceSeat[] source)
        {
            var result = new TacticalTraceSeat[source.Length];
            for (int index = 0; index < source.Length; index++)
            {
                TacticalTraceSeat seat = source[index] ?? throw new ArgumentException(
                    "state projection seat must not be null", nameof(source));
                if (seat.Units == null) throw new ArgumentException(
                    "state projection units must not be null", nameof(source));
                result[index] = new TacticalTraceSeat
                {
                    Seat = seat.Seat,
                    Points = seat.Points,
                    DestroyedValue = seat.DestroyedValue,
                    AliveUnits = seat.AliveUnits,
                    CurrentHitPoints = seat.CurrentHitPoints,
                    MaximumHitPoints = seat.MaximumHitPoints,
                    HealthAdjustedMaterial = seat.HealthAdjustedMaterial,
                    CanDamageEnemy = seat.CanDamageEnemy,
                    CanCurrentlyAttackEnemy = seat.CanCurrentlyAttackEnemy,
                    CanMove = seat.CanMove,
                    Units = CopyUnits(seat.Units),
                };
            }
            return result;
        }

        private static TacticalTraceUnit[] CopyUnits(TacticalTraceUnit[] source)
        {
            var result = new TacticalTraceUnit[source.Length];
            for (int index = 0; index < source.Length; index++)
            {
                TacticalTraceUnit unit = source[index] ?? throw new ArgumentException(
                    "state projection unit must not be null", nameof(source));
                result[index] = new TacticalTraceUnit
                {
                    Id = unit.Id, Q = unit.Q, R = unit.R, CurrentHp = unit.CurrentHp,
                    MaximumHp = unit.MaximumHp, PointCost = unit.PointCost, Damage = unit.Damage,
                    Defense = unit.Defense, Movement = unit.Movement,
                    VerticalMovement = unit.VerticalMovement, Range = unit.Range, Moved = unit.Moved,
                    Attacked = unit.Attacked, MovementSpentH = unit.MovementSpentH,
                    MovementSpentV = unit.MovementSpentV,
                };
            }
            return result;
        }

        private static TacticalTraceControl[] CopyControls(TacticalTraceControl[] source)
        {
            var result = new TacticalTraceControl[source.Length];
            for (int index = 0; index < source.Length; index++)
            {
                TacticalTraceControl control = source[index] ?? throw new ArgumentException(
                    "state projection control must not be null", nameof(source));
                result[index] = new TacticalTraceControl
                {
                    Q = control.Q, R = control.R, Controller = control.Controller,
                };
            }
            return result;
        }
    }

    public sealed class OraclePreflightActionOracle : IActionOracle
    {
        private readonly IActionOracle _inner;
        private readonly IOraclePreflightBenchmarkSink _sink;
        private readonly Func<long> _timestamp;
        private readonly long _clockFrequency;

        public OraclePreflightActionOracle(IActionOracle inner,
            IOraclePreflightBenchmarkSink sink, Func<long> timestamp, long clockFrequency)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            _sink = sink ?? throw new ArgumentNullException(nameof(sink));
            _timestamp = timestamp ?? throw new ArgumentNullException(nameof(timestamp));
            if (clockFrequency < 1) throw new ArgumentOutOfRangeException(nameof(clockFrequency));
            _clockFrequency = clockFrequency;
        }

        public TacticalV2OracleDecision Decide(TacticalV2DecisionContext context)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));
            Snapshot before = Snapshot.Capture(context);
            (TacticalV2OracleDecision first, long firstTicks) = TimedDecision(context);
            AssertUnchanged(context, before, "first oracle decision");
            (TacticalV2OracleDecision second, long secondTicks) = TimedDecision(context);
            AssertUnchanged(context, before, "second oracle decision");
            if (!OraclePreflightBenchmarkRecord.SameSemantics(first, second))
                throw new InvalidOperationException("repeated oracle decisions differ");

            Revalidate(context, before, first, "first oracle decision");
            Revalidate(context, before, second, "second oracle decision");
            _sink.Accepted(new OraclePreflightBenchmarkRecord(before.StateHash,
                context.DecisionIndex, before.Observation, before.LegalMask, before.State,
                first, second, firstTicks, secondTicks, _clockFrequency));
            return OraclePreflightBenchmarkRecord.CopyDecision(first);
        }

        private (TacticalV2OracleDecision Decision, long ElapsedTicks) TimedDecision(
            TacticalV2DecisionContext context)
        {
            long started = _timestamp();
            TacticalV2OracleDecision decision = _inner.Decide(context) ??
                throw new InvalidOperationException("oracle returned null decision");
            long ended = _timestamp();
            if (ended < started)
                throw new InvalidOperationException("oracle timestamp moved backwards");
            return (decision, ended - started);
        }

        private static void AssertUnchanged(TacticalV2DecisionContext context,
            Snapshot expected, string stage)
        {
            Snapshot actual = Snapshot.Capture(context);
            if (!string.Equals(actual.Fingerprint, expected.Fingerprint, StringComparison.Ordinal))
                throw new InvalidOperationException(stage + " mutated the decision context");
        }

        private static void Revalidate(TacticalV2DecisionContext context, Snapshot snapshot,
            TacticalV2OracleDecision decision, string owner)
        {
            if (decision.Action < 0 || decision.Action >= snapshot.LegalMask.Length ||
                !snapshot.LegalMask[decision.Action])
                throw new InvalidOperationException(owner + " action is masked");

            GameState state = context.State.Clone();
            TacticalV2UnitRegistry registry = context.OwnRegistry;
            if (!TacticalV2Coding.TryEncode(decision.Command, state, context.Layout, registry,
                out int encoded) || encoded != decision.Action)
                throw new InvalidOperationException(owner + " command does not encode to its action");
            Command decoded = TacticalV2Coding.Decode(decision.Action, state, context.Seat,
                context.Layout, registry);
            if (!decoded.Equals(decision.Command))
                throw new InvalidOperationException(owner + " command failed codec round-trip");
            if (!GameEngine.Apply(state, decision.Command).Success)
                throw new InvalidOperationException(owner + " command failed authoritative legality");
        }

        private sealed class Snapshot
        {
            private Snapshot(string stateHash, string fingerprint, float[] observation,
                bool[] legalMask, TacticalTraceState state)
            {
                StateHash = stateHash;
                Fingerprint = fingerprint;
                Observation = observation;
                LegalMask = legalMask;
                State = state;
            }

            public string StateHash { get; }
            public string Fingerprint { get; }
            public float[] Observation { get; }
            public bool[] LegalMask { get; }
            public TacticalTraceState State { get; }

            public static Snapshot Capture(TacticalV2DecisionContext context)
            {
                float[] observation = context.Observation;
                bool[] legalMask = context.LegalMask;
                TacticalTraceState state = TacticalEvaluationTrace.ProjectState(context.State);
                string stateHash = Hash(StateKey(state));
                string fingerprint = Hash(stateHash + "|" + ObservationKey(observation) + "|" +
                    MaskKey(legalMask));
                return new Snapshot(stateHash, fingerprint, observation, legalMask, state);
            }

            private static string ObservationKey(float[] observation)
            {
                var builder = new StringBuilder(observation.Length * 9);
                foreach (float value in observation)
                    builder.Append(BitConverter.SingleToInt32Bits(value)).Append(',');
                return builder.ToString();
            }

            private static string MaskKey(bool[] mask)
            {
                var builder = new StringBuilder(mask.Length);
                foreach (bool value in mask) builder.Append(value ? '1' : '0');
                return builder.ToString();
            }

            private static string StateKey(TacticalTraceState state)
            {
                var builder = new StringBuilder();
                builder.Append(state.Round).Append('|').Append(state.ActiveSeat).Append('|')
                    .Append(state.IsGameOver ? 1 : 0).Append('|').Append(state.Winner)
                    .Append('|').Append(state.ProductiveLegalActions);
                foreach (TacticalTraceSeat seat in state.Seats)
                {
                    builder.Append("|S,").Append(seat.Seat).Append(',').Append(seat.Points)
                        .Append(',').Append(seat.DestroyedValue).Append(',').Append(seat.AliveUnits)
                        .Append(',').Append(seat.CurrentHitPoints).Append(',')
                        .Append(seat.MaximumHitPoints).Append(',')
                        .Append(BitConverter.DoubleToInt64Bits(seat.HealthAdjustedMaterial))
                        .Append(',').Append(seat.CanDamageEnemy ? 1 : 0).Append(',')
                        .Append(seat.CanCurrentlyAttackEnemy ? 1 : 0).Append(',')
                        .Append(seat.CanMove ? 1 : 0);
                    foreach (TacticalTraceUnit unit in seat.Units)
                        builder.Append("|U,").Append(unit.Id).Append(',').Append(unit.Q)
                            .Append(',').Append(unit.R).Append(',').Append(unit.CurrentHp)
                            .Append(',').Append(unit.MaximumHp).Append(',').Append(unit.PointCost)
                            .Append(',').Append(unit.Damage).Append(',').Append(unit.Defense)
                            .Append(',').Append(unit.Movement).Append(',')
                            .Append(unit.VerticalMovement).Append(',').Append(unit.Range)
                            .Append(',').Append(unit.Moved ? 1 : 0).Append(',')
                            .Append(unit.Attacked ? 1 : 0).Append(',')
                            .Append(unit.MovementSpentH).Append(',').Append(unit.MovementSpentV);
                }
                foreach (TacticalTraceControl control in state.ControlledHexes)
                    builder.Append("|C,").Append(control.Q).Append(',').Append(control.R)
                        .Append(',').Append(control.Controller);
                return builder.ToString();
            }

            private static string Hash(string value)
            {
                byte[] bytes = Encoding.UTF8.GetBytes(value);
                using SHA256 sha = SHA256.Create();
                byte[] hash = sha.ComputeHash(bytes);
                return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
            }
        }
    }
}

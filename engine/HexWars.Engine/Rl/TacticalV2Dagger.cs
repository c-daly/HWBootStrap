using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace HexWars.Engine.Rl
{
    [Flags]
    public enum DaggerEligibilityReason
    {
        None = 0,
        Conversion = 1,
        Favorable = 2,
        CycleWarning = 4,
        WastedEndTurn = 8,
    }

    public interface IActionOracle
    {
        TacticalV2OracleDecision Decide(TacticalV2DecisionContext context);
    }

    public sealed class TacticalV2OracleDecision
    {
        public TacticalV2OracleDecision(int action, Command command, int depth,
            int expansionBudget, string heuristicIdentity, int actualExpansionCount)
        {
            if (action < 0) throw new ArgumentOutOfRangeException(nameof(action));
            Command = command ?? throw new ArgumentNullException(nameof(command));
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth));
            if (expansionBudget < 1)
                throw new ArgumentOutOfRangeException(nameof(expansionBudget));
            if (string.IsNullOrEmpty(heuristicIdentity))
                throw new ArgumentException("heuristic identity must not be empty",
                    nameof(heuristicIdentity));
            if (actualExpansionCount < 0 || actualExpansionCount > expansionBudget)
                throw new ArgumentOutOfRangeException(nameof(actualExpansionCount));
            Action = action;
            Depth = depth;
            ExpansionBudget = expansionBudget;
            HeuristicIdentity = heuristicIdentity;
            ActualExpansionCount = actualExpansionCount;
        }

        public int Action { get; }
        public Command Command { get; }
        public int Depth { get; }
        public int ExpansionBudget { get; }
        public string HeuristicIdentity { get; }
        public int ActualExpansionCount { get; }
    }

    public sealed class BoundedSearchActionOracle : IActionOracle
    {
        private readonly BoundedSearchAgent _search;

        public BoundedSearchActionOracle(GameConfig gameConfig, int expansionBudget,
            int depth, string heuristicIdentity)
        {
            if (gameConfig == null) throw new ArgumentNullException(nameof(gameConfig));
            if (gameConfig.FogOfWar)
                throw new ArgumentException(
                    "bounded-search oracle requires fog_of_war=false", nameof(gameConfig));
            if (!string.Equals(heuristicIdentity, BoundedSearchAgent.HeuristicIdentity,
                StringComparison.Ordinal))
                throw new ArgumentException(
                    "unrecognized bounded-search heuristic identity",
                    nameof(heuristicIdentity));
            _search = new BoundedSearchAgent(expansionBudget, depth, useHeuristic: true);
            HeuristicIdentity = heuristicIdentity;
        }

        public int ExpansionBudget => _search.ExpansionBudget;
        public int Depth => _search.Depth;
        public string HeuristicIdentity { get; }

        public TacticalV2OracleDecision Decide(TacticalV2DecisionContext context)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (context.State.Config.FogOfWar)
                throw new InvalidOperationException(
                    "bounded-search oracle requires a fully observed decision state");
            if (context.State.ActivePlayer != context.Seat)
                throw new InvalidOperationException(
                    "oracle decision seat must be the active player");

            Command command = _search.Decide(context.State);
            TacticalV2UnitRegistry registry = context.OwnRegistry;
            if (!TacticalV2Coding.TryEncode(
                command, context.State, context.Layout, registry, out int action))
                throw new InvalidOperationException(
                    "bounded-search command is not legal and encodable in the decision snapshot");

            bool[] mask = context.LegalMask;
            if (action < 0 || action >= mask.Length || !mask[action])
                throw new InvalidOperationException(
                    "bounded-search action is masked in the recorded decision snapshot");
            Command decoded = TacticalV2Coding.Decode(
                action, context.State, context.Seat, context.Layout, registry);
            if (!decoded.Equals(command))
                throw new InvalidOperationException(
                    "bounded-search command failed the recorded codec round-trip");
            if (!GameEngine.Apply(context.State, command).Success)
                throw new InvalidOperationException(
                    "bounded-search command failed authoritative legality validation");

            return new TacticalV2OracleDecision(action, command, _search.Depth,
                _search.ExpansionBudget, HeuristicIdentity, _search.LastExpansionCount);
        }
    }

    public sealed class TacticalV2DaggerDecision
    {
        private readonly float[] _observation;
        private readonly bool[] _legalMask;
        private readonly TacticalTraceCommand _learnerCommand;
        private readonly TacticalTraceCommand _teacherCommand;

        public TacticalV2DaggerDecision(
            float[] observation,
            bool[] legalMask,
            int learnerAction,
            TacticalTraceCommand learnerCommand,
            int teacherAction,
            TacticalTraceCommand teacherCommand,
            DaggerEligibilityReason reasons,
            string stateHash,
            double normalizedAdvantage,
            int opponentLivingUnitCount,
            int productiveLegalActionCount,
            int seat,
            int round,
            int decisionIndex,
            bool disagreement,
            int oracleDepth,
            int oracleExpansionBudget,
            string oracleHeuristicIdentity,
            int oracleActualExpansionCount)
        {
            if (observation == null) throw new ArgumentNullException(nameof(observation));
            if (legalMask == null) throw new ArgumentNullException(nameof(legalMask));
            if (learnerCommand == null) throw new ArgumentNullException(nameof(learnerCommand));
            if (teacherCommand == null) throw new ArgumentNullException(nameof(teacherCommand));
            if (reasons == DaggerEligibilityReason.None)
                throw new ArgumentOutOfRangeException(nameof(reasons));
            if (string.IsNullOrEmpty(stateHash))
                throw new ArgumentException("state hash must not be empty", nameof(stateHash));
            if (double.IsNaN(normalizedAdvantage) ||
                double.IsInfinity(normalizedAdvantage))
                throw new ArgumentOutOfRangeException(nameof(normalizedAdvantage));
            if (opponentLivingUnitCount < 0)
                throw new ArgumentOutOfRangeException(nameof(opponentLivingUnitCount));
            if (productiveLegalActionCount < 0)
                throw new ArgumentOutOfRangeException(nameof(productiveLegalActionCount));
            if (decisionIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(decisionIndex));
            if (oracleDepth < 1) throw new ArgumentOutOfRangeException(nameof(oracleDepth));
            if (oracleExpansionBudget < 1)
                throw new ArgumentOutOfRangeException(nameof(oracleExpansionBudget));
            if (string.IsNullOrEmpty(oracleHeuristicIdentity))
                throw new ArgumentException("oracle heuristic identity must not be empty",
                    nameof(oracleHeuristicIdentity));
            if (oracleActualExpansionCount < 0 ||
                oracleActualExpansionCount > oracleExpansionBudget)
                throw new ArgumentOutOfRangeException(nameof(oracleActualExpansionCount));

            _observation = (float[])observation.Clone();
            _legalMask = (bool[])legalMask.Clone();
            _learnerCommand = Copy(learnerCommand);
            _teacherCommand = Copy(teacherCommand);
            LearnerAction = learnerAction;
            TeacherAction = teacherAction;
            Reasons = reasons;
            StateHash = stateHash;
            NormalizedAdvantage = normalizedAdvantage;
            OpponentLivingUnitCount = opponentLivingUnitCount;
            ProductiveLegalActionCount = productiveLegalActionCount;
            Seat = seat;
            Round = round;
            DecisionIndex = decisionIndex;
            Disagreement = disagreement;
            OracleDepth = oracleDepth;
            OracleExpansionBudget = oracleExpansionBudget;
            OracleHeuristicIdentity = oracleHeuristicIdentity;
            OracleActualExpansionCount = oracleActualExpansionCount;
        }

        public float[] Observation => (float[])_observation.Clone();
        public bool[] LegalMask => (bool[])_legalMask.Clone();
        public int LearnerAction { get; }
        public TacticalTraceCommand LearnerCommand => Copy(_learnerCommand);
        public int TeacherAction { get; }
        public TacticalTraceCommand TeacherCommand => Copy(_teacherCommand);
        public DaggerEligibilityReason Reasons { get; }
        public string StateHash { get; }
        public double NormalizedAdvantage { get; }
        public int OpponentLivingUnitCount { get; }
        public int ProductiveLegalActionCount { get; }
        public int Seat { get; }
        public int Round { get; }
        public int DecisionIndex { get; }
        public bool Disagreement { get; }
        public int OracleDepth { get; }
        public int OracleExpansionBudget { get; }
        public string OracleHeuristicIdentity { get; }
        public int OracleActualExpansionCount { get; }

        private static TacticalTraceCommand Copy(TacticalTraceCommand source) =>
            new TacticalTraceCommand
            {
                Kind = source.Kind,
                Issuer = source.Issuer,
                ActorId = source.ActorId,
                TargetId = source.TargetId,
                Q = source.Q,
                R = source.R,
            };
    }

    public interface ITacticalV2DaggerSink
    {
        bool Enabled { get; }
        void Reset();
        void Accepted(TacticalV2DaggerDecision decision);
    }

    public sealed class BufferedTacticalV2DaggerSink : ITacticalV2DaggerSink
    {
        private readonly List<TacticalV2DaggerDecision> _items =
            new List<TacticalV2DaggerDecision>();

        public bool Enabled { get; set; }

        public void Reset() => _items.Clear();

        public void Accepted(TacticalV2DaggerDecision decision)
        {
            if (decision == null) throw new ArgumentNullException(nameof(decision));
            if (Enabled) _items.Add(decision);
        }

        public IReadOnlyList<TacticalV2DaggerDecision> Drain()
        {
            var result = new List<TacticalV2DaggerDecision>(_items);
            _items.Clear();
            return result;
        }
    }

    public sealed class SelectiveDaggerObserver : ITacticalV2DecisionObserver
    {
        private readonly IActionOracle _oracle;
        private readonly ITacticalV2DaggerSink _sink;
        private readonly Dictionary<string, int> _occurrences =
            new Dictionary<string, int>(StringComparer.Ordinal);
        private readonly HashSet<string> _emittedStateHashes =
            new HashSet<string>(StringComparer.Ordinal);
        private TacticalV2EpisodeContext? _episode;
        private double _initialMaterial;

        public SelectiveDaggerObserver(IActionOracle oracle, ITacticalV2DaggerSink sink)
        {
            _oracle = oracle ?? throw new ArgumentNullException(nameof(oracle));
            _sink = sink ?? throw new ArgumentNullException(nameof(sink));
        }

        public void Reset(TacticalV2EpisodeContext episode)
        {
            _episode = episode ?? throw new ArgumentNullException(nameof(episode));
            _occurrences.Clear();
            _emittedStateHashes.Clear();
            _initialMaterial =
                Material(episode.InitialState, PlayerId.Player0, episode.PointsWeight) +
                Material(episode.InitialState, PlayerId.Player1, episode.PointsWeight);
            _sink.Reset();
        }

        public void Observe(TacticalV2DecisionContext decision)
        {
            if (decision == null) throw new ArgumentNullException(nameof(decision));
            TacticalV2EpisodeContext episode = _episode ??
                throw new InvalidOperationException(
                    "selective DAgger observer must be reset before observing decisions");
            if (decision.Seat != episode.LearnerSeat)
                throw new InvalidOperationException(
                    "decision seat does not match the episode learner seat");

            TacticalTraceState projection = TacticalEvaluationTrace.ProjectState(decision.State);
            string key = CanonicalStateKey(projection);
            int occurrence = _occurrences.TryGetValue(key, out int seen) ? seen + 1 : 1;
            _occurrences[key] = occurrence;

            int opponentSeat = 1 - (int)decision.Seat;
            int opponentLiving = projection.Seats[opponentSeat].AliveUnits;
            double learnerMaterial =
                Material(decision.State, decision.Seat, episode.PointsWeight);
            double opponentMaterial =
                Material(decision.State, decision.State.Opponent(decision.Seat).Id,
                    episode.PointsWeight);
            double normalizedAdvantage =
                (learnerMaterial - opponentMaterial) / Math.Max(1d, _initialMaterial);

            DaggerEligibilityReason reasons = DaggerEligibilityReason.None;
            if (opponentLiving <= 1) reasons |= DaggerEligibilityReason.Conversion;
            if (normalizedAdvantage > 0d) reasons |= DaggerEligibilityReason.Favorable;
            if (occurrence == 2) reasons |= DaggerEligibilityReason.CycleWarning;
            if (decision.LearnerCommand is EndTurn &&
                projection.ProductiveLegalActions > 0)
                reasons |= DaggerEligibilityReason.WastedEndTurn;

            if (reasons == DaggerEligibilityReason.None) return;
            string stateHash = Hash(key);
            if (_emittedStateHashes.Contains(stateHash) || !_sink.Enabled) return;

            TacticalV2OracleDecision teacher = _oracle.Decide(decision);
            Revalidate(decision, decision.LearnerAction, decision.LearnerCommand, "learner");
            Revalidate(decision, teacher.Action, teacher.Command, "teacher");

            var row = new TacticalV2DaggerDecision(
                decision.Observation,
                decision.LegalMask,
                decision.LearnerAction,
                TacticalEvaluationTrace.ProjectCommand(decision.LearnerCommand),
                teacher.Action,
                TacticalEvaluationTrace.ProjectCommand(teacher.Command),
                reasons,
                stateHash,
                normalizedAdvantage,
                opponentLiving,
                projection.ProductiveLegalActions,
                (int)decision.Seat,
                decision.State.Round,
                decision.DecisionIndex,
                decision.LearnerAction != teacher.Action,
                teacher.Depth,
                teacher.ExpansionBudget,
                teacher.HeuristicIdentity,
                teacher.ActualExpansionCount);
            _sink.Accepted(row);
            _emittedStateHashes.Add(stateHash);
        }

        private static void Revalidate(TacticalV2DecisionContext decision,
            int action, Command command, string owner)
        {
            bool[] mask = decision.LegalMask;
            if (action < 0 || action >= mask.Length || !mask[action])
                throw new InvalidOperationException(
                    owner + " action is masked in the recorded decision snapshot");
            TacticalV2UnitRegistry registry = decision.OwnRegistry;
            if (!TacticalV2Coding.TryEncode(
                command, decision.State, decision.Layout, registry, out int encoded) ||
                encoded != action)
                throw new InvalidOperationException(
                    owner + " command does not encode to its recorded action");
            Command decoded = TacticalV2Coding.Decode(
                action, decision.State, decision.Seat, decision.Layout, registry);
            if (!decoded.Equals(command))
                throw new InvalidOperationException(
                    owner + " action failed the recorded codec round-trip");
        }

        private static double Material(GameState state, PlayerId seat, float pointsWeight)
        {
            PlayerState player = state.Player(seat);
            double material = pointsWeight * player.Points;
            foreach (Unit unit in player.UnitsOnBoard)
                if (unit.IsAlive)
                    material += unit.Stats.PointCost * (double)unit.CurrentHp /
                        unit.Stats.Health;
            return material;
        }

        public static string CanonicalStateKey(GameState state)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            return CanonicalStateKey(TacticalEvaluationTrace.ProjectState(state));
        }

        private static string CanonicalStateKey(TacticalTraceState projection)
        {
            var key = new StringBuilder();
            Append(key, projection.ActiveSeat);
            key.Append("|[");
            for (int seat = 0; seat < projection.Seats.Length; seat++)
            {
                if (seat > 0) key.Append(',');
                key.Append('(');
                Append(key, projection.Seats[seat].Points);
                key.Append(',');
                Append(key, projection.Seats[seat].DestroyedValue);
                key.Append(')');
            }
            key.Append("]|[");
            for (int index = 0; index < projection.ControlledHexes.Length; index++)
            {
                if (index > 0) key.Append(',');
                TacticalTraceControl control = projection.ControlledHexes[index];
                key.Append('(');
                Append(key, control.Q);
                key.Append(',');
                Append(key, control.R);
                key.Append(',');
                Append(key, control.Controller);
                key.Append(')');
            }
            key.Append("]|[");
            bool first = true;
            for (int seat = 0; seat < projection.Seats.Length; seat++)
            {
                foreach (TacticalTraceUnit unit in projection.Seats[seat].Units)
                {
                    if (unit.CurrentHp <= 0) continue;
                    if (!first) key.Append(',');
                    first = false;
                    key.Append('(');
                    Append(key, projection.Seats[seat].Seat);
                    key.Append(',');
                    Append(key, unit.Id);
                    key.Append(',');
                    Append(key, unit.Q);
                    key.Append(',');
                    Append(key, unit.R);
                    key.Append(',');
                    Append(key, unit.CurrentHp);
                    key.Append(',');
                    Append(key, unit.Moved ? 1 : 0);
                    key.Append(',');
                    Append(key, unit.Attacked ? 1 : 0);
                    key.Append(',');
                    Append(key, unit.MovementSpentH);
                    key.Append(',');
                    Append(key, unit.MovementSpentV);
                    key.Append(')');
                }
            }
            key.Append(']');
            return key.ToString();
        }

        private static void Append(StringBuilder builder, int value) =>
            builder.Append(value.ToString(CultureInfo.InvariantCulture));

        private static string Hash(string canonicalKey)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(canonicalKey);
            byte[] digest;
            using (SHA256 sha = SHA256.Create()) digest = sha.ComputeHash(bytes);
            var text = new StringBuilder(digest.Length * 2);
            foreach (byte value in digest)
                text.Append(value.ToString("x2", CultureInfo.InvariantCulture));
            return text.ToString();
        }
    }
}

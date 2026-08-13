using System;

namespace HexWars.Engine.Rl
{
    /// <summary>Passive, opt-in instrumentation for externally supplied tactical-v2 learner decisions.</summary>
    public interface ITacticalV2DecisionObserver
    {
        void Reset(TacticalV2EpisodeContext episode);
        void Observe(TacticalV2DecisionContext decision);
    }

    /// <summary>Immutable metadata and initial state for one tactical-v2 episode.</summary>
    public sealed class TacticalV2EpisodeContext
    {
        public TacticalV2EpisodeContext(GameState initialState, string selectedStartProfileId,
            PlayerId referenceSeat, PlayerId learnerSeat, float pointsWeight)
        {
            InitialState = initialState ?? throw new ArgumentNullException(nameof(initialState));
            if (string.IsNullOrEmpty(selectedStartProfileId))
                throw new ArgumentException("selected start profile id must not be empty", nameof(selectedStartProfileId));
            if (float.IsNaN(pointsWeight) || float.IsInfinity(pointsWeight))
                throw new ArgumentOutOfRangeException(nameof(pointsWeight));
            SelectedStartProfileId = string.Copy(selectedStartProfileId);
            ReferenceSeat = referenceSeat;
            LearnerSeat = learnerSeat;
            PointsWeight = pointsWeight;
        }

        public GameState InitialState { get; }
        public string SelectedStartProfileId { get; }
        public PlayerId ReferenceSeat { get; }
        public PlayerId LearnerSeat { get; }
        public float PointsWeight { get; }
    }

    /// <summary>Immutable pre-action data for one externally supplied learner decision.</summary>
    public sealed class TacticalV2DecisionContext
    {
        private readonly float[] _observation;
        private readonly bool[] _legalMask;
        private readonly TacticalV2UnitRegistry _ownRegistry;
        private readonly TacticalV2UnitRegistry _foeRegistry;

        public TacticalV2DecisionContext(GameState state, PlayerId seat, int decisionIndex,
            float[] observation, bool[] legalMask, int learnerAction, Command learnerCommand,
            TacticalV2UnitRegistry ownRegistry, TacticalV2UnitRegistry foeRegistry, TacticalV2Layout layout)
        {
            State = state ?? throw new ArgumentNullException(nameof(state));
            if (decisionIndex < 0) throw new ArgumentOutOfRangeException(nameof(decisionIndex));
            _observation = observation == null ? throw new ArgumentNullException(nameof(observation))
                : (float[])observation.Clone();
            _legalMask = legalMask == null ? throw new ArgumentNullException(nameof(legalMask))
                : (bool[])legalMask.Clone();
            if (learnerAction < 0 || learnerAction >= _legalMask.Length || !_legalMask[learnerAction])
                throw new ArgumentOutOfRangeException(nameof(learnerAction));
            LearnerCommand = learnerCommand ?? throw new ArgumentNullException(nameof(learnerCommand));
            _ownRegistry = (ownRegistry ?? throw new ArgumentNullException(nameof(ownRegistry))).Snapshot();
            _foeRegistry = (foeRegistry ?? throw new ArgumentNullException(nameof(foeRegistry))).Snapshot();
            Layout = layout ?? throw new ArgumentNullException(nameof(layout));
            Seat = seat;
            DecisionIndex = decisionIndex;
            LearnerAction = learnerAction;
        }

        public GameState State { get; }
        public PlayerId Seat { get; }
        public int DecisionIndex { get; }
        public float[] Observation => (float[])_observation.Clone();
        public bool[] LegalMask => (bool[])_legalMask.Clone();
        public int LearnerAction { get; }
        public Command LearnerCommand { get; }
        public TacticalV2UnitRegistry OwnRegistry => _ownRegistry.Snapshot();
        public TacticalV2UnitRegistry FoeRegistry => _foeRegistry.Snapshot();
        public TacticalV2Layout Layout { get; }
    }
}

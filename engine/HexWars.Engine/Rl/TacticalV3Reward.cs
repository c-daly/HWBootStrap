using System;

namespace HexWars.Engine.Rl
{
    public interface IRewardContract
    {
        void Reset(GameState initialState, PlayerId learnerSeat);
        TacticalV3RewardBreakdown Evaluate(
            GameState state, bool terminated, bool truncated);
    }

    public sealed class TacticalV3RewardBreakdown
    {
        public TacticalV3RewardBreakdown(
            float terminalOutcome,
            float knownHealthAdjustedMaterialProgress,
            float publicResourceProgress,
            float timePressure,
            float total,
            bool finalized)
        {
            TerminalOutcome = terminalOutcome;
            KnownHealthAdjustedMaterialProgress = knownHealthAdjustedMaterialProgress;
            PublicResourceProgress = publicResourceProgress;
            TimePressure = timePressure;
            Total = total;
            Finalized = finalized;
        }

        public float TerminalOutcome { get; }
        public float KnownHealthAdjustedMaterialProgress { get; }
        public float PublicResourceProgress { get; }
        public float TimePressure { get; }
        public float Total { get; }
        public bool Finalized { get; }
    }

    public sealed class TacticalV3Reward : IRewardContract
    {
        private readonly TacticalV3RewardConfig _config;
        private PlayerId _learnerSeat;
        private float _initialAdvantage;
        private float _initialTotalValue;
        private TacticalV3RewardBreakdown? _finalBreakdown;
        private bool _hasReset;

        public TacticalV3Reward(TacticalV3RewardConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        public void Reset(GameState initialState, PlayerId learnerSeat)
        {
            if (initialState == null) throw new ArgumentNullException(nameof(initialState));

            _learnerSeat = learnerSeat;
            _initialAdvantage = Advantage(initialState, learnerSeat);
            _initialTotalValue = TotalValue(initialState);
            _finalBreakdown = null;
            _hasReset = true;
        }

        public TacticalV3RewardBreakdown Evaluate(GameState state, bool terminated, bool truncated)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (!_hasReset) throw new InvalidOperationException("reward contract must be reset before evaluation");
            if (_finalBreakdown != null) return _finalBreakdown;
            if (!terminated && !truncated) return Zero();

            float material = Clamp(
                (Advantage(state, _learnerSeat) - _initialAdvantage) /
                Math.Max(1f, _initialTotalValue),
                -_config.MaterialAdjustmentBound,
                _config.MaterialAdjustmentBound);
            float time = -_config.TimePressureBound * Clamp(
                (state.Round - 1f) / state.Config.RoundCap, 0f, 1f);
            float terminal = terminated && state.Winner == _learnerSeat
                ? _config.TerminalWin
                : _config.TerminalNonWin;
            float total = terminal + material + time;

            _finalBreakdown = new TacticalV3RewardBreakdown(
                terminal, material, 0f, time, total, finalized: true);
            return _finalBreakdown;
        }

        private static TacticalV3RewardBreakdown Zero() =>
            new TacticalV3RewardBreakdown(0f, 0f, 0f, 0f, 0f, finalized: false);

        private float Advantage(GameState state, PlayerId learnerSeat) =>
            PlayerValue(state.Player(learnerSeat)) -
            PlayerValue(state.Opponent(learnerSeat));

        private float TotalValue(GameState state) =>
            PlayerValue(state.Player(PlayerId.Player0)) +
            PlayerValue(state.Player(PlayerId.Player1));

        private float PlayerValue(PlayerState player) {
            float value = _config.PointsWeight * player.Points;
            foreach (Unit unit in player.UnitsOnBoard)
            {
                int health = Math.Max(1, unit.Stats.Health);
                value += unit.Stats.PointCost * unit.CurrentHp / (float)health;
            }
            return value;
        }

        private static float Clamp(float value, float minimum, float maximum)
        {
            if (float.IsNaN(value)) return 0f;
            return value < minimum ? minimum : value > maximum ? maximum : value;
        }
    }
}

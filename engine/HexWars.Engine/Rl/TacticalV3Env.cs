using System;

namespace HexWars.Engine.Rl
{
    public sealed class TacticalV3Env
    {
        private readonly Func<int, IAgent> _opponentFactory;
        private readonly PlayerId _learnerSeat;
        private readonly TacticalV3DuelEnv _duel;

        public TacticalV3Env(
            Func<int, IAgent> opponentFactory,
            PlayerId learnerSeat,
            TacticalV3Config config)
        {
            _opponentFactory = opponentFactory ??
                throw new ArgumentNullException(nameof(opponentFactory));
            if (learnerSeat != PlayerId.Player0 && learnerSeat != PlayerId.Player1)
                throw new ArgumentOutOfRangeException(nameof(learnerSeat));
            _learnerSeat = learnerSeat;
            _duel = new TacticalV3DuelEnv(
                config ?? throw new ArgumentNullException(nameof(config)));
        }

        public TacticalV3View Reset(int seed)
        {
            IAgent opponent = _opponentFactory(seed) ??
                throw new InvalidOperationException("tactical-v3 opponent factory returned null");
            IAgent? controller0 = _learnerSeat == PlayerId.Player0 ? null : opponent;
            IAgent? controller1 = _learnerSeat == PlayerId.Player1 ? null : opponent;

            return _duel.ResetSelectedProfile(
                seed, controller0, controller1, _learnerSeat);
        }

        public TacticalV3View Step(long decisionId, int candidateId) =>
            _duel.Step(decisionId, candidateId);
    }
}

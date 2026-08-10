using System;
using System.Linq;

namespace HexWars.Engine.Rl
{
    public sealed class TacticalV3Env
    {
        private readonly Func<int, IAgent> _opponentFactory;
        private readonly PlayerId _learnerSeat;
        private readonly TacticalV3Config _config;
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
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _duel = new TacticalV3DuelEnv(_config);
        }

        public TacticalV3View Reset(int seed)
        {
            IAgent opponent = _opponentFactory(seed) ??
                throw new InvalidOperationException("tactical-v3 opponent factory returned null");
            IAgent? controller0 = _learnerSeat == PlayerId.Player0 ? null : opponent;
            IAgent? controller1 = _learnerSeat == PlayerId.Player1 ? null : opponent;

            if (_config.Match.PlacementPolicy == "profiled-seeded-v1")
            {
                if (_config.Match.StartDistribution == null)
                    throw new InvalidOperationException(
                        "profiled tactical-v3 reset requires a start distribution");
                string profileId = _config.Match.StartDistribution.Select(seed);
                if (!_config.Match.StartProfiles.Any(profile => profile.Id == profileId))
                    throw new InvalidOperationException(
                        "selected start profile '" + profileId +
                        "' is not declared by the tactical-v3 configuration");
                return _duel.Reset(
                    seed,
                    controller0,
                    controller1,
                    profileId,
                    _learnerSeat,
                    _learnerSeat);
            }

            return _duel.Reset(
                seed,
                controller0,
                controller1,
                _learnerSeat);
        }

        public TacticalV3View Step(long decisionId, int candidateId) =>
            _duel.Step(decisionId, candidateId);
    }
}

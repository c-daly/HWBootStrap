using System;

namespace HexWars.Engine.Rl
{
    /// <summary>Single-learner adaptive-v1 facade over the same duel orchestrator used by self-play.</summary>
    public sealed class AdaptiveTacticalEnv
    {
        private readonly Func<int, IAgent> _opponentFactory;
        private readonly Func<int, IDeploymentPolicy> _deploymentFactory;
        private readonly PlayerId _seat;
        private readonly AdaptiveEnvConfig _cfg;
        private readonly AdaptiveDuelEnv _duel;
        private readonly MlContract _contract;
        private AdaptiveDuelEnv.View _view;
        private int _steps;

        public AdaptiveTacticalEnv(Func<int, IAgent> opponentFactory,
            Func<int, IDeploymentPolicy> deploymentFactory,
            PlayerId learningSeat = PlayerId.Player0,
            AdaptiveEnvConfig? config = null)
        {
            _opponentFactory = opponentFactory ?? (seed => new RandomAgent(seed));
            _deploymentFactory = deploymentFactory ?? (seed => new CombinedArmsDeploymentPolicy(seed));
            _seat = learningSeat;
            _cfg = config ?? AdaptiveEnvConfig.Default();
            _duel = new AdaptiveDuelEnv(_cfg);
            _contract = MlContract.CreateAdaptive(_cfg, MlEnvironmentKind.AdaptiveTactical);
        }

        public int ActionCount => _duel.ActionCount;
        public int ObservationLength => _duel.ObservationLength;
        public int ObsChannels => _duel.ObsChannels;
        public int BoardH => _duel.BoardH;
        public int BoardW => _duel.BoardW;
        public AdaptiveEnvConfig Config => _cfg;
        public AdaptiveLayout Layout => _duel.Layout;
        public MlContract Contract => _contract;
        public bool DeploymentComplete => _duel.DeploymentComplete;
        public GameState State => _duel.State;
        public AdaptiveDiagnostics Diagnostics => _duel.Diagnostics;

        public float[] Reset(int seed)
        {
            IAgent opponent = _opponentFactory(seed);
            IDeploymentPolicy deployment = _deploymentFactory(seed);
            _view = _seat == PlayerId.Player0
                ? _duel.Reset(seed, null, opponent, null, deployment, _seat)
                : _duel.Reset(seed, opponent, null, deployment, null, _seat);
            EnsureLearnerView();
            _steps = 0;
            return _view.Observation;
        }

        public StepResult Step(int action)
        {
            _view = _duel.Step(action);
            EnsureLearnerView();
            _steps++;
            bool truncated = !_view.Terminated && (_view.Truncated || _steps >= _cfg.MaxSteps);
            return new StepResult(_view.Observation, _view.Reward,
                _view.Terminated, truncated, _view.ActionMask);
        }

        public bool[] LegalActionMask() => _view.ActionMask;

        private void EnsureLearnerView()
        {
            if (_view.Seat != (int)_seat)
                throw new InvalidOperationException(
                    $"adaptive tactical environment exposed seat {_view.Seat} to learner seat {(int)_seat}");
        }
    }
}

using System;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// Single-agent RL environment over the tactical-v2 foundation (deterministic catalog, stable
    /// per-slot unit identity, symmetric seeded starts). The learning agent controls one seat; the
    /// other seat is a configurable opponent the env plays automatically, so the agent always acts on
    /// its own turn. Mirrors <see cref="TacticalEnv"/>'s opponent scheduling and reward shaping
    /// exactly, but routes reset/observe/mask/decode through the tactical-v2 layout, coding, and
    /// per-seat <see cref="TacticalV2UnitRegistry"/> slots — kept in sync with the game state after
    /// every accepted command (learner or opponent alike), including registering a freshly deployed
    /// unit's template identity, so a trained model always sees a registry that matches the board.
    /// </summary>
    public sealed class TacticalV2Env
    {
        private readonly TacticalV2Config _cfg;
        private readonly TacticalV2Layout _layout;
        private readonly Func<int, IAgent> _opponentFactory;
        private readonly PlayerId _seat;
        private readonly PlayerId _foe;

        private GameState _state = null!;
        private IAgent _opponent = null!;
        private TacticalV2UnitRegistry _slots0 = null!;
        private TacticalV2UnitRegistry _slots1 = null!;
        private TacticalV2UnitRegistry _own = null!;
        private int _steps;
        private float _prevAdvantage;
        private float _armyValue;

        public TacticalV2Env(Func<int, IAgent> opponentFactory, PlayerId seat, TacticalV2Config config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            _cfg = config;
            _layout = new TacticalV2Layout(_cfg);
            _opponentFactory = opponentFactory ?? (s => new RandomAgent(s));
            _seat = seat;
            _foe = seat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
        }

        public TacticalV2Config Config => _cfg;
        public TacticalV2Layout Layout => _layout;
        public GameState State => _state;
        public TacticalV2UnitRegistry Slots0 => _slots0;
        public TacticalV2UnitRegistry Slots1 => _slots1;
        public string SelectedStartProfileId { get; private set; } = "standard-3v3";

        public float[] Reset(int seed)
        {
            TacticalV2Start start;
            if (_cfg.PlacementPolicy == "profiled-seeded-v1")
            {
                if (_cfg.StartDistribution == null)
                    throw new InvalidOperationException("profiled tactical-v2 reset requires a start distribution");
                string profileId = _cfg.StartDistribution.Select(seed);
                TacticalV2StartProfile? profile = _cfg.StartProfiles?
                    .SingleOrDefault(item => item.Id == profileId);
                if (profile == null)
                    throw new InvalidOperationException(
                        $"selected start profile '{profileId}' is not declared by the tactical-v2 config");
                start = _layout.NewGame(seed, profile, _seat);
            }
            else
            {
                start = _layout.NewGame(seed);
            }
            SelectedStartProfileId = start.ProfileId;
            _state = start.State;
            _slots0 = start.Slots0;
            _slots1 = start.Slots1;
            _own = _seat == PlayerId.Player0 ? _slots0 : _slots1;
            _opponent = _opponentFactory(seed);
            _steps = 0;
            AdvanceToSeat();
            _prevAdvantage = Advantage();
            _armyValue = RewardShaping.PositionValue(_state, _seat, _cfg.PointsWeight);
            return TacticalV2Coding.Observe(_state, _seat, _layout, _own, FoeRegistry);
        }

        public StepResult Step(int action)
        {
            Command cmd = TacticalV2Coding.Decode(action, _state, _seat, _layout, _own);
            var mv = cmd as MoveUnit;
            int gapBefore = mv != null ? RewardShaping.GapOfUnit(_state, mv.UnitId, _seat, _foe) : -1;

            float closing = 0f;
            if (TryApply(cmd) && mv != null && gapBefore >= 0)
            {
                int gapAfter = RewardShaping.GapOfUnit(_state, mv.UnitId, _seat, _foe);
                if (gapAfter >= 0) closing = _cfg.ClosingWeight * (gapBefore - gapAfter);
            }
            AdvanceToSeat();
            _steps++;

            float reward = Reward(closing);
            bool terminated = _state.IsGameOver;
            bool truncated = !terminated && _steps >= _cfg.MaxSteps;
            return new StepResult(TacticalV2Coding.Observe(_state, _seat, _layout, _own, FoeRegistry),
                reward, terminated, truncated, LegalActionMask());
        }

        public bool[] LegalActionMask() => TacticalV2Coding.Mask(_state, _seat, _layout, _own);

        private TacticalV2UnitRegistry FoeRegistry => _seat == PlayerId.Player0 ? _slots1 : _slots0;

        /// <summary>Applies one command and, on success, keeps both slot registries synchronized with the
        /// resulting state: releases any slot whose unit just died, then — if the command was a
        /// DeployUnit — claims the newly deployed unit's slot under its template index. Shared by the
        /// learner's own commands and the scripted opponent's, so Slots0/Slots1 are always consistent
        /// with <see cref="State"/> regardless of who acted.</summary>
        private bool TryApply(Command cmd)
        {
            // A scripted opponent decides from raw engine legality (board cells and points), never the
            // RL registry's synthetic per-seat capacity, so it can propose a DeployUnit with nowhere to
            // land once every registry slot already holds a living unit. Reject it exactly like any
            // other illegal move — before touching state — rather than letting RegisterDeployment throw.
            if (cmd is DeployUnit pendingDeploy)
            {
                TacticalV2UnitRegistry pendingRegistry =
                    pendingDeploy.Issuer == PlayerId.Player0 ? _slots0 : _slots1;
                if (!pendingRegistry.HasFreeSlot) return false;
            }

            GameState before = _state;
            var r = GameEngine.Apply(_state, cmd);
            if (!r.Success) return false;

            _state = r.NewState;
            _slots0.ReleaseDead(_state, PlayerId.Player0);
            _slots1.ReleaseDead(_state, PlayerId.Player1);
            if (cmd is DeployUnit deploy)
            {
                TacticalV2UnitRegistry registry = deploy.Issuer == PlayerId.Player0 ? _slots0 : _slots1;
                registry.RegisterDeployment(before, _state, deploy.Issuer, deploy.TemplateIndex);
            }
            return true;
        }

        private void AdvanceToSeat()
        {
            int guard = 0;
            while (!_state.IsGameOver && _state.ActivePlayer != _seat && guard++ < 4000)
            {
                if (TryApply(_opponent.Decide(_state))) continue;
                if (TryApply(new EndTurn(_state.ActivePlayer))) continue; // unstick an illegal pick
                break;
            }
        }

        private float Advantage() => RewardShaping.Advantage(_state, _seat, _foe, _cfg.PointsWeight);

        private float Reward(float closing)
        {
            float adv = Advantage();
            float shaped = _cfg.ShapeScale * (adv - _prevAdvantage) + closing - _cfg.StepPenalty;
            _prevAdvantage = adv;

            if (!_state.IsGameOver) return shaped;
            if (_state.Winner == _seat) return shaped + 1f;
            if (_state.Winner == _foe) return shaped - 1f;
            return shaped + RewardShaping.DrawCredit(_state, _seat, _foe, _armyValue, _cfg.DrawCreditWeight, _cfg.PointsWeight); // cap draw
        }
    }
}

using System;
using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// A tactical-v2 duel: a single game whose two seats can each be driven externally (a policy
    /// supplies actions via <see cref="Step"/>) or internally (a scripted <see cref="IAgent"/> the env
    /// auto-plays) — a null controller means that seat is external. Mirrors <see cref="DuelEnv"/>'s
    /// controller rotation and reward shaping, but routes reset/observe/mask/decode through the
    /// tactical-v2 layout, coding, and per-seat <see cref="TacticalV2UnitRegistry"/> slots. Every
    /// scripted and external command — whichever seat issued it — passes through the same registry
    /// update path, so Slots0/Slots1 always match <see cref="State"/> before the next <see cref="View"/>
    /// is built. Records the game for replay.
    /// </summary>
    public sealed class TacticalV2DuelEnv
    {
        private readonly TacticalV2Config _cfg;
        private readonly TacticalV2Layout _layout;
        private readonly List<Command> _log = new List<Command>();
        private readonly List<DuelTransition> _transitions = new List<DuelTransition>();

        private GameState _start = null!;
        private GameState _state = null!;
        private TacticalV2UnitRegistry _slots0 = null!;
        private TacticalV2UnitRegistry _slots1 = null!;
        private IAgent? _ctrl0;
        private IAgent? _ctrl1;
        private PlayerId _learner;
        private float _prevAdv;
        private float _armyValue;
        private int _steps;

        public TacticalV2DuelEnv(TacticalV2Config config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            _cfg = config;
            _layout = new TacticalV2Layout(_cfg);
        }

        public TacticalV2Config Config => _cfg;
        public TacticalV2Layout Layout => _layout;
        public GameState State => _state;

        /// <summary>Opt-in: when true, every accepted command captures a <see cref="DuelTransition"/>
        /// for <see cref="DrainTransitions"/>. Defaults to false so headless training (which never
        /// drains) doesn't pay the memory cost of retaining every intermediate <see cref="GameState"/>
        /// in a vectorized run until the next Reset.</summary>
        public bool CaptureTransitions { get; set; }

        /// <summary>Start a duel. A null controller = that seat is external (caller supplies its actions);
        /// non-null = the env auto-plays it. <paramref name="learnerSeat"/> sets whose perspective the
        /// per-step reward is from (for self-play training).</summary>
        public View Reset(int seed, IAgent? controller0, IAgent? controller1, PlayerId learnerSeat = PlayerId.Player0)
        {
            TacticalV2Start start = _layout.NewGame(seed);
            _start = start.State;
            _state = start.State;
            _slots0 = start.Slots0;
            _slots1 = start.Slots1;
            _ctrl0 = controller0;
            _ctrl1 = controller1;
            _learner = learnerSeat;
            _steps = 0;
            _log.Clear();
            _transitions.Clear();
            AdvancePastInternal();
            _prevAdv = Advantage();
            _armyValue = RewardShaping.PositionValue(_state, _learner, _cfg.PointsWeight);
            return MakeView(0f);
        }

        /// <summary>Apply one action for the current (external) seat, auto-play any internal seats, and
        /// return the learner-perspective reward for the transition.</summary>
        public View Step(int action)
        {
            PlayerId seat = _state.ActivePlayer;
            Command cmd = TacticalV2Coding.Decode(action, _state, seat, _layout, Registry(seat));

            // active-piece closing, credited only when the LEARNER moves one of its own units
            var mv = seat == _learner ? cmd as MoveUnit : null;
            int gapBefore = mv != null ? RewardShaping.GapOfUnit(_state, mv.UnitId, _learner, Foe) : -1;

            float closing = 0f;
            if (TryApply(cmd) && mv != null && gapBefore >= 0)
            {
                int gapAfter = RewardShaping.GapOfUnit(_state, mv.UnitId, _learner, Foe);
                if (gapAfter >= 0) closing = _cfg.ClosingWeight * (gapBefore - gapAfter);
            }
            _steps++;
            AdvancePastInternal();
            return MakeView(ComputeReward(closing));
        }

        /// <summary>The recorded duel as a portable replay (start + commands), for Unity playback.</summary>
        public string ToReplay() => ReplayFile.Write(_start, _log);

        /// <summary>Every accepted-command transition since the last drain (or Reset), in order, then
        /// clears the queue. See <see cref="DuelTransition"/>: covers the external step path, internal
        /// scripted controllers, and the unstick EndTurn fallback alike — anywhere <see cref="TryApply"/>
        /// accepted a command.</summary>
        public IReadOnlyList<DuelTransition> DrainTransitions()
        {
            var drained = new List<DuelTransition>(_transitions);
            _transitions.Clear();
            return drained;
        }

        private IAgent? Controller(PlayerId seat) => seat == PlayerId.Player0 ? _ctrl0 : _ctrl1;

        private TacticalV2UnitRegistry Registry(PlayerId seat) => seat == PlayerId.Player0 ? _slots0 : _slots1;

        /// <summary>Applies one command and, on success, keeps both slot registries synchronized with the
        /// resulting state: releases any slot whose unit just died, then — if the command was a
        /// DeployUnit — claims the newly deployed unit's slot under its template index. Shared by the
        /// external step path and the internal controller loop, so Slots0/Slots1 always match
        /// <see cref="State"/> regardless of which seat, or which controller, acted.</summary>
        private bool TryApply(Command cmd)
        {
            // An internal scripted controller decides from raw engine legality (board cells and
            // points), never the RL registry's synthetic per-seat capacity, so it can propose a
            // DeployUnit with nowhere to land once every registry slot already holds a living unit.
            // Reject it exactly like any other illegal move — before touching state — rather than
            // letting RegisterDeployment throw.
            if (cmd is DeployUnit pendingDeploy && !Registry(pendingDeploy.Issuer).HasFreeSlot)
                return false;

            GameState before = _state;
            var r = GameEngine.Apply(_state, cmd);
            if (!r.Success) return false;

            _state = r.NewState;
            _log.Add(cmd);
            if (CaptureTransitions) _transitions.Add(new DuelTransition(before, cmd, _state));
            _slots0.ReleaseDead(_state, PlayerId.Player0);
            _slots1.ReleaseDead(_state, PlayerId.Player1);
            if (cmd is DeployUnit deploy)
            {
                TacticalV2UnitRegistry registry = Registry(deploy.Issuer);
                registry.RegisterDeployment(before, _state, deploy.Issuer, deploy.TemplateIndex);
            }
            return true;
        }

        private void AdvancePastInternal()
        {
            int guard = 0;
            while (!_state.IsGameOver && Controller(_state.ActivePlayer) != null && guard++ < 8000)
            {
                PlayerId seat = _state.ActivePlayer;
                Command cmd = Controller(seat)!.Decide(_state);
                if (TryApply(cmd)) continue;
                if (TryApply(new EndTurn(seat))) continue; // unstick an illegal pick
                break;
            }
        }

        private PlayerId Foe => _learner == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;

        private float Advantage() => RewardShaping.Advantage(_state, _learner, Foe, _cfg.PointsWeight);

        private float ComputeReward(float closing)
        {
            float adv = Advantage();
            float shaped = _cfg.ShapeScale * (adv - _prevAdv) + closing - _cfg.StepPenalty;
            _prevAdv = adv;
            if (!_state.IsGameOver) return shaped;
            if (_state.Winner == _learner) return shaped + 1f;
            if (_state.Winner != null) return shaped - 1f;
            return shaped + RewardShaping.DrawCredit(_state, _learner, Foe, _armyValue, _cfg.DrawCreditWeight, _cfg.PointsWeight); // cap draw
        }

        private View MakeView(float reward)
        {
            PlayerId seat = _state.ActivePlayer;
            TacticalV2UnitRegistry own = Registry(seat);
            TacticalV2UnitRegistry foe = Registry(seat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0);
            bool terminated = _state.IsGameOver;
            bool truncated = !terminated && _steps >= _cfg.MaxSteps * 2;
            return new View
            {
                Observation = TacticalV2Coding.Observe(_state, seat, _layout, own, foe),
                ActionMask = TacticalV2Coding.Mask(_state, seat, _layout, own),
                Seat = seat,
                Reward = reward,
                Winner = terminated ? _state.Winner : null,
                Terminated = terminated,
                Truncated = truncated,
            };
        }

        /// <summary>Per-step result: observation + mask are from <see cref="Seat"/>'s point of view;
        /// <see cref="Reward"/> is from the learner seat's perspective; <see cref="Winner"/> is set only
        /// at a terminal state (null = draw/none).</summary>
        public sealed class View
        {
            public float[] Observation = null!;
            public bool[] ActionMask = null!;
            public PlayerId Seat;
            public float Reward;
            public PlayerId? Winner;
            public bool Terminated;
            public bool Truncated;
        }
    }
}

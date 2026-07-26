using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// A single game whose two seats can each be driven externally (a policy supplies actions via
    /// <see cref="Step"/>) or internally (a scripted <see cref="IAgent"/> the env auto-plays). Enables
    /// any matchup — model vs model, model vs greedy — and, with both seats external plus a frozen
    /// opponent on the Python side, self-play training. Each <see cref="View"/> carries a reward from
    /// the learner seat's perspective (shaped value change + terminal ±1), so a Python wrapper can sum
    /// it over the learner's turn + the opponent's reply. Records the game for replay; shares
    /// <see cref="TacticalCoding"/> with training so policies see a matching observation.
    /// </summary>
    public sealed class DuelEnv
    {
        private readonly EnvConfig _cfg;
        private readonly TacticalLayout _layout;
        private readonly List<Command> _log = new List<Command>();
        private readonly List<DuelTransition> _transitions = new List<DuelTransition>();

        private GameState _start = null!;
        private GameState _state = null!;
        private int[] _slot0 = System.Array.Empty<int>();
        private int[] _slot1 = System.Array.Empty<int>();
        private IAgent? _ctrl0;
        private IAgent? _ctrl1;
        private PlayerId _learner;
        private float _prevAdv;
        private float _armyValue;
        private int _steps;

        public DuelEnv(EnvConfig? cfg = null)
        {
            _cfg = cfg ?? new EnvConfig();
            _layout = new TacticalLayout(_cfg);
        }

        public int ActionCount => _layout.ActionCount;
        public int ObservationLength => _layout.ObservationLength;
        public int ObsChannels => _layout.ObsChannels;
        public int BoardH => _layout.BoardH;
        public int BoardW => _layout.BoardW;
        public EnvConfig Config => _cfg;
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
            var (state, s0, s1) = _layout.NewGame(seed);
            _start = state;
            _state = state;
            _slot0 = s0;
            _slot1 = s1;
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
            var seat = _state.ActivePlayer;
            var slot = seat == PlayerId.Player0 ? _slot0 : _slot1;
            var cmd = TacticalCoding.Decode(action, _state, seat, slot, _layout);
            float closing = 0f;
            if (cmd != null)
            {
                // active-piece closing, credited only when the LEARNER moves one of its own units
                var mv = seat == _learner ? cmd as MoveUnit : null;
                int gapBefore = mv != null ? RewardShaping.GapOfUnit(_state, mv.UnitId, _learner, Foe) : -1;
                GameState before = _state;
                var r = GameEngine.Apply(_state, cmd);
                if (r.Success)
                {
                    _state = r.NewState; _log.Add(cmd);
                    if (CaptureTransitions) _transitions.Add(new DuelTransition(before, cmd, _state));
                    if (mv != null && gapBefore >= 0)
                    {
                        int gapAfter = RewardShaping.GapOfUnit(_state, mv.UnitId, _learner, Foe);
                        if (gapAfter >= 0) closing = _cfg.ClosingWeight * (gapBefore - gapAfter);
                    }
                }
            }
            _steps++;
            AdvancePastInternal();
            return MakeView(ComputeReward(closing));
        }

        /// <summary>The recorded duel as a portable replay (start + commands), for Unity playback.</summary>
        public string ToReplay() => ReplayFile.Write(_start, _log);

        /// <summary>Every accepted-command transition since the last drain (or Reset), in order, then
        /// clears the queue. See <see cref="DuelTransition"/>: covers both the external step path and
        /// the internal scripted-controller loop (including its unstick EndTurn fallback).</summary>
        public IReadOnlyList<DuelTransition> DrainTransitions()
        {
            var drained = new List<DuelTransition>(_transitions);
            _transitions.Clear();
            return drained;
        }

        private IAgent? Controller(PlayerId seat) => seat == PlayerId.Player0 ? _ctrl0 : _ctrl1;

        private void AdvancePastInternal()
        {
            int guard = 0;
            while (!_state.IsGameOver && Controller(_state.ActivePlayer) != null && guard++ < 8000)
            {
                var seat = _state.ActivePlayer;
                var cmd = Controller(seat)!.Decide(_state);
                GameState before = _state;
                var r = GameEngine.Apply(_state, cmd);
                if (r.Success)
                {
                    _state = r.NewState; _log.Add(cmd);
                    if (CaptureTransitions) _transitions.Add(new DuelTransition(before, cmd, _state));
                    continue;
                }
                var endCmd = new EndTurn(seat);
                var end = GameEngine.Apply(_state, endCmd);
                if (end.Success)
                {
                    _state = end.NewState; _log.Add(endCmd);
                    if (CaptureTransitions) _transitions.Add(new DuelTransition(before, endCmd, _state));
                }
                else break;
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
            var seat = _state.ActivePlayer;
            var slot = seat == PlayerId.Player0 ? _slot0 : _slot1;
            bool terminated = _state.IsGameOver;
            bool truncated = !terminated && _steps >= _cfg.MaxSteps * 2;
            int winner = terminated && _state.Winner != null ? (int)_state.Winner.Value : -1; // -1 = draw/none
            return new View(
                TacticalCoding.Observe(_state, seat, _layout),
                TacticalCoding.Mask(_state, seat, slot, _layout),
                (int)seat, reward, winner, terminated, truncated);
        }

        /// <summary>Per-step result: observation + mask are from <see cref="Seat"/>'s point of view;
        /// <see cref="Reward"/> is from the learner seat's perspective; <see cref="Winner"/> is 0/1 at a
        /// terminal state, else -1.</summary>
        public readonly struct View
        {
            public readonly float[] Observation;
            public readonly bool[] ActionMask;
            public readonly int Seat;
            public readonly float Reward;
            public readonly int Winner;
            public readonly bool Terminated;
            public readonly bool Truncated;

            public View(float[] obs, bool[] mask, int seat, float reward, int winner, bool terminated, bool truncated)
            {
                Observation = obs;
                ActionMask = mask;
                Seat = seat;
                Reward = reward;
                Winner = winner;
                Terminated = terminated;
                Truncated = truncated;
            }
        }
    }
}

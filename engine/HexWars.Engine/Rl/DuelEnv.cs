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
        public GameState State => _state;

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
            AdvancePastInternal();
            _prevAdv = Advantage();
            _armyValue = WinCheck.Evaluate(_state, _learner);
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
                var r = GameEngine.Apply(_state, cmd);
                if (r.Success)
                {
                    _state = r.NewState; _log.Add(cmd);
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

        private IAgent? Controller(PlayerId seat) => seat == PlayerId.Player0 ? _ctrl0 : _ctrl1;

        private void AdvancePastInternal()
        {
            int guard = 0;
            while (!_state.IsGameOver && Controller(_state.ActivePlayer) != null && guard++ < 8000)
            {
                var seat = _state.ActivePlayer;
                var cmd = Controller(seat)!.Decide(_state);
                var r = GameEngine.Apply(_state, cmd);
                if (r.Success) { _state = r.NewState; _log.Add(cmd); continue; }
                var end = GameEngine.Apply(_state, new EndTurn(seat));
                if (end.Success) { _state = end.NewState; _log.Add(new EndTurn(seat)); } else break;
            }
        }

        private PlayerId Foe => _learner == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;

        private float Advantage() => WinCheck.Evaluate(_state, _learner) - WinCheck.Evaluate(_state, Foe);

        private float ComputeReward(float closing)
        {
            float adv = Advantage();
            float shaped = _cfg.ShapeScale * (adv - _prevAdv) + closing - _cfg.StepPenalty;
            _prevAdv = adv;
            if (!_state.IsGameOver) return shaped;
            if (_state.Winner == _learner) return shaped + 1f;
            if (_state.Winner != null) return shaped - 1f;
            return shaped + RewardShaping.DrawCredit(_state, _learner, Foe, _armyValue, _cfg.DrawCreditWeight); // cap draw
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

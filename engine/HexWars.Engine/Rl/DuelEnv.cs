using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// A single game whose two seats can each be driven either EXTERNALLY (a trained policy supplies the
    /// action via <see cref="Step"/>) or INTERNALLY (a scripted <see cref="IAgent"/> — greedy/random —
    /// the env plays automatically). So any matchup works: model vs model, model vs greedy, greedy vs
    /// random, etc. After each external action the env auto-plays any internal seats, so the caller only
    /// ever supplies actions for external seats. Records the game for replay; shares <see
    /// cref="TacticalCoding"/> with training so policies see a matching observation.
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
        private int _steps;

        public DuelEnv(EnvConfig? cfg = null)
        {
            _cfg = cfg ?? new EnvConfig();
            _layout = new TacticalLayout(_cfg);
        }

        public int ActionCount => _layout.ActionCount;
        public int ObservationLength => _layout.ObservationLength;
        public GameState State => _state;

        /// <summary>Start a duel. A null controller = that seat is external (caller supplies its actions
        /// via <see cref="Step"/>); a non-null controller = the env plays that seat automatically.</summary>
        public View Reset(int seed, IAgent? controller0, IAgent? controller1)
        {
            var (state, s0, s1) = _layout.NewGame(seed);
            _start = state;
            _state = state;
            _slot0 = s0;
            _slot1 = s1;
            _ctrl0 = controller0;
            _ctrl1 = controller1;
            _steps = 0;
            _log.Clear();
            AdvancePastInternal();
            return CurrentView();
        }

        /// <summary>Apply one action for the current (external) seat, then auto-play any internal seats.</summary>
        public View Step(int action)
        {
            var seat = _state.ActivePlayer;
            var slot = seat == PlayerId.Player0 ? _slot0 : _slot1;
            var cmd = TacticalCoding.Decode(action, _state, seat, slot, _layout);
            if (cmd != null)
            {
                var r = GameEngine.Apply(_state, cmd);
                if (r.Success) { _state = r.NewState; _log.Add(cmd); }
            }
            _steps++;
            AdvancePastInternal();
            return CurrentView();
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
                var end = GameEngine.Apply(_state, new EndTurn(seat)); // unstick an illegal pick
                if (end.Success) { _state = end.NewState; _log.Add(new EndTurn(seat)); } else break;
            }
        }

        private View CurrentView()
        {
            var seat = _state.ActivePlayer;
            var slot = seat == PlayerId.Player0 ? _slot0 : _slot1;
            bool terminated = _state.IsGameOver;
            bool truncated = !terminated && _steps >= _cfg.MaxSteps * 2;
            return new View(
                TacticalCoding.Observe(_state, seat, _layout),
                TacticalCoding.Mask(_state, seat, slot, _layout),
                (int)seat, terminated, truncated);
        }

        /// <summary>Per-step result: observation + mask are from <see cref="Seat"/>'s point of view.</summary>
        public readonly struct View
        {
            public readonly float[] Observation;
            public readonly bool[] ActionMask;
            public readonly int Seat;
            public readonly bool Terminated;
            public readonly bool Truncated;

            public View(float[] obs, bool[] mask, int seat, bool terminated, bool truncated)
            {
                Observation = obs;
                ActionMask = mask;
                Seat = seat;
                Terminated = terminated;
                Truncated = truncated;
            }
        }
    }
}

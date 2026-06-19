using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// Two-controller environment: both seats are driven externally (e.g. two trained policies), so two
    /// learned agents can play a single game head-to-head. Each <see cref="Step"/> applies one command
    /// for the current seat and returns the view from the NEXT seat to act — the caller picks the action
    /// with whichever model owns that seat. Records the game so it can be saved as a replay and watched
    /// in Unity. Uses the same <see cref="TacticalCoding"/> as training, so policies see a matching obs.
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
        private int _steps;

        public DuelEnv(EnvConfig? cfg = null)
        {
            _cfg = cfg ?? new EnvConfig();
            _layout = new TacticalLayout(_cfg);
        }

        public int ActionCount => _layout.ActionCount;
        public int ObservationLength => _layout.ObservationLength;
        public GameState State => _state;

        public View Reset(int seed)
        {
            var (state, s0, s1) = _layout.NewGame(seed);
            _start = state;
            _state = state;
            _slot0 = s0;
            _slot1 = s1;
            _steps = 0;
            _log.Clear();
            return CurrentView(false, false);
        }

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

            bool terminated = _state.IsGameOver;
            bool truncated = !terminated && _steps >= _cfg.MaxSteps * 2; // two seats share the budget
            return CurrentView(terminated, truncated);
        }

        /// <summary>The recorded duel as a portable replay (start + commands), for Unity playback.</summary>
        public string ToReplay() => ReplayFile.Write(_start, _log);

        private View CurrentView(bool terminated, bool truncated)
        {
            var seat = _state.ActivePlayer;
            var slot = seat == PlayerId.Player0 ? _slot0 : _slot1;
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

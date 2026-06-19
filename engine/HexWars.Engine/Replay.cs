using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// Reconstructs every game state from a <see cref="MatchRecord"/> by re-applying its commands from
    /// the start. Frames are precomputed for O(1) random access, so a viewer can play them back at any
    /// speed, pause, step, or scrub backwards. Frame 0 is the start; the last frame is terminal.
    /// </summary>
    public sealed class Replay
    {
        private readonly List<GameState> _frames;

        public Replay(MatchRecord record)
        {
            _frames = new List<GameState> { record.Start };
            var state = record.Start;
            foreach (var command in record.Commands)
            {
                var result = GameEngine.Apply(state, command);
                if (result.Success) state = result.NewState;
                _frames.Add(state);
            }
        }

        public int FrameCount => _frames.Count;
        public GameState Final => _frames[_frames.Count - 1];

        /// <summary>The state at a frame index (clamped to range), for playback or scrubbing.</summary>
        public GameState Frame(int index)
        {
            if (index < 0) index = 0;
            if (index >= _frames.Count) index = _frames.Count - 1;
            return _frames[index];
        }
    }
}

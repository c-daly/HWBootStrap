using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// A recorded match: the start state plus the ordered commands that were actually applied, and the
    /// final <see cref="MatchResult"/>. Because the engine is deterministic, this is a complete,
    /// compact recording — feed it to <see cref="Replay"/> to reconstruct every frame.
    /// </summary>
    public sealed class MatchRecord
    {
        public GameState Start { get; }
        public IReadOnlyList<Command> Commands { get; }
        public MatchResult Result { get; }

        public MatchRecord(GameState start, IReadOnlyList<Command> commands, MatchResult result)
        {
            Start = start;
            Commands = commands;
            Result = result;
        }
    }
}

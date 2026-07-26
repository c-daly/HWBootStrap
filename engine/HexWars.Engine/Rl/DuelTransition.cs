namespace HexWars.Engine.Rl
{
    /// <summary>One accepted command applied during a duel: the immutable game state immediately before
    /// it, the command itself, and the immutable game state immediately after. Captured at every
    /// command-accept point in <see cref="DuelEnv"/>, <see cref="TacticalV2DuelEnv"/>, and
    /// <see cref="AdaptiveDuelEnv"/> — including commands auto-played by internal (scripted) controllers
    /// and the unstick <see cref="EndTurn"/> fallback — so a spectator (the Unity viewer) can play back
    /// every move/attack/deploy/etc. exactly as decided, never inferring it from a coarse before/after
    /// state diff. Rejected or invalid actions never produce a transition. <see cref="Previous"/> and
    /// <see cref="Resulting"/> are the SAME <see cref="GameState"/> instances the env already holds — no
    /// copies — so consecutive transitions in a drained batch chain by reference:
    /// <c>transitions[i].Resulting == transitions[i + 1].Previous</c>.</summary>
    public readonly struct DuelTransition
    {
        public GameState Previous { get; }
        public Command Command { get; }
        public GameState Resulting { get; }

        public DuelTransition(GameState previous, Command command, GameState resulting)
        {
            Previous = previous;
            Command = command;
            Resulting = resulting;
        }
    }
}

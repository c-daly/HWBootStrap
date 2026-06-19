namespace HexWars.Engine
{
    /// <summary>
    /// Headless game loop: drives two <see cref="IAgent"/>s through the engine with no UI, until the
    /// game ends or a command budget is hit. The foundation for self-play, AI evaluation, and RL.
    /// A match is fully determined by (start state + agents), so episodes are reproducible.
    /// </summary>
    public static class Match
    {
        /// <summary>Plays to a terminal state (or the command budget) and returns the final state.</summary>
        public static GameState Play(GameState start, IAgent agent0, IAgent agent1, int maxCommands)
            => Run(start, agent0, agent1, maxCommands).Final;

        /// <summary>Plays to a terminal state (or the command budget) and returns a <see cref="MatchResult"/>
        /// with the outcome and length — the unit a self-play/balance batch aggregates over.</summary>
        public static MatchResult Run(GameState start, IAgent agent0, IAgent agent1, int maxCommands)
        {
            var state = start;
            int commands = 0;
            for (; commands < maxCommands && !state.IsGameOver; commands++)
            {
                var agent = state.ActivePlayer == PlayerId.Player0 ? agent0 : agent1;
                var result = GameEngine.Apply(state, agent.Decide(state));
                if (result.Success)
                    state = result.NewState; // illegal choices are simply ignored; agents pick legal commands
            }

            return new MatchResult(
                state.Winner, state.Round, commands, timedOut: !state.IsGameOver,
                WinCheck.Evaluate(state, PlayerId.Player0),
                WinCheck.Evaluate(state, PlayerId.Player1),
                state);
        }
    }
}

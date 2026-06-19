namespace HexWars.Engine
{
    /// <summary>
    /// Headless game loop: drives two <see cref="IAgent"/>s through the engine with no UI, until the
    /// game ends or a command budget is hit. The foundation for self-play, AI evaluation, and RL.
    /// A match is fully determined by (start state + agents), so episodes are reproducible.
    /// </summary>
    public static class Match
    {
        public static GameState Play(GameState start, IAgent agent0, IAgent agent1, int maxCommands)
        {
            var state = start;
            for (int i = 0; i < maxCommands && !state.IsGameOver; i++)
            {
                var agent = state.ActivePlayer == PlayerId.Player0 ? agent0 : agent1;
                var result = GameEngine.Apply(state, agent.Decide(state));
                if (result.Success)
                    state = result.NewState; // illegal choices are simply ignored; agents pick legal commands
            }
            return state;
        }
    }
}

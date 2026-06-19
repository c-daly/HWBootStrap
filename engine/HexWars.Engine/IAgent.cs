namespace HexWars.Engine
{
    /// <summary>
    /// A player driver. Given the current state (the agent is <c>state.ActivePlayer</c>), it returns
    /// the command to play. The human UI, the built-in AI, and external/RL agents are all just
    /// implementations of this over the same <c>Apply</c> command API.
    /// </summary>
    public interface IAgent
    {
        Command Decide(GameState state);
    }
}

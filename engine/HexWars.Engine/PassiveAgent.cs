namespace HexWars.Engine
{
    /// <summary>A scripted controller that takes no action and immediately ends each turn.</summary>
    public sealed class PassiveAgent : IAgent
    {
        public Command Decide(GameState state) => new EndTurn(state.ActivePlayer);
    }
}

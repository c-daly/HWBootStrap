namespace HexWars.Engine
{
    /// <summary>
    /// The swappable turn-structure rule (the only "framework" rule beyond what you buy). Decides
    /// whether the turn auto-ends after a given command — letting the same engine support both the
    /// snappy "act with everything, then End Turn" mode and the chess-like "one action per turn" mode.
    /// </summary>
    public interface ITurnPolicy
    {
        bool AutoEndTurnAfter(Command command);
    }

    /// <summary>Default M1 mode: the turn never auto-ends; the player acts freely then ends it.</summary>
    public sealed class AllUnitsPolicy : ITurnPolicy
    {
        public bool AutoEndTurnAfter(Command command) => false;
    }

    /// <summary>Chess-like mode: the turn auto-ends after any single non-EndTurn action.</summary>
    public sealed class OneActionPolicy : ITurnPolicy
    {
        public bool AutoEndTurnAfter(Command command) => !(command is EndTurn);
    }
}

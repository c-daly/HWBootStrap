namespace HexWars.Engine
{
    /// <summary>
    /// The swappable turn-structure rule (the only "framework" rule beyond what you buy). Decides
    /// whether the turn auto-ends after a given command — letting the same engine support the snappy
    /// "act with everything, then End Turn" mode, the chess-like "one action per turn" mode, and the
    /// tunable "K actions then pass" middle ground. The state after the command is provided so a policy
    /// can count actions taken this turn.
    /// </summary>
    public interface ITurnPolicy
    {
        bool AutoEndTurnAfter(Command command, GameState stateAfter);
    }

    /// <summary>Default M1 mode: the turn never auto-ends; the player acts freely then ends it.</summary>
    public sealed class AllUnitsPolicy : ITurnPolicy
    {
        public bool AutoEndTurnAfter(Command command, GameState stateAfter) => false;
    }

    /// <summary>Chess-like mode: the turn auto-ends after any single non-EndTurn action.</summary>
    public sealed class OneActionPolicy : ITurnPolicy
    {
        public bool AutoEndTurnAfter(Command command, GameState stateAfter) => !(command is EndTurn);
    }

    /// <summary>
    /// Middle ground: the turn auto-ends once the active player has taken <c>K</c> combat actions (moves +
    /// attacks) this turn, so a player commits at most K actions before the opponent can respond. K=1 is
    /// near one-action; a large K is effectively whole-army. The lever for the second-mover advantage —
    /// committing your whole army before the opponent acts is what hands them a free full-army counter.
    /// </summary>
    public sealed class KActionsPolicy : ITurnPolicy
    {
        private readonly int _k;
        public KActionsPolicy(int k) => _k = k < 1 ? 1 : k;

        public bool AutoEndTurnAfter(Command command, GameState stateAfter)
            => !(command is EndTurn)
               && (stateAfter.MovedUnitIds.Count + stateAfter.AttackedUnitIds.Count) >= _k;
    }
}

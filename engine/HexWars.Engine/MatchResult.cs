namespace HexWars.Engine
{
    /// <summary>
    /// Outcome of one headless match: who won (null = draw), how long it ran, and the final position
    /// values. <see cref="TimedOut"/> is true when the command budget was hit before a terminal state —
    /// those games never resolved and should usually be excluded from balance statistics.
    /// </summary>
    public readonly struct MatchResult
    {
        public PlayerId? Winner { get; }
        public int Rounds { get; }
        public int Commands { get; }
        public bool TimedOut { get; }
        public int Value0 { get; }
        public int Value1 { get; }
        public GameState Final { get; }

        public MatchResult(PlayerId? winner, int rounds, int commands, bool timedOut,
                           int value0, int value1, GameState final)
        {
            Winner = winner;
            Rounds = rounds;
            Commands = commands;
            TimedOut = timedOut;
            Value0 = value0;
            Value1 = value1;
            Final = final;
        }
    }
}

namespace HexWars.Engine
{
    /// <summary>Aggregate outcome of a self-play batch. Win-rate skew between the two slots (same agent
    /// on both sides) is the first-move-advantage signal; <see cref="TimeoutRate"/> and
    /// <see cref="DrawRate"/> flag stalemate problems; <see cref="AvgRounds"/> is pace.</summary>
    public readonly struct BatchReport
    {
        public int Games { get; }
        public int Player0Wins { get; }
        public int Player1Wins { get; }
        public int Draws { get; }
        public int TimedOut { get; }
        public double AvgRounds { get; }
        public double AvgCommands { get; }

        public BatchReport(int games, int player0Wins, int player1Wins, int draws, int timedOut,
                           double avgRounds, double avgCommands)
        {
            Games = games;
            Player0Wins = player0Wins;
            Player1Wins = player1Wins;
            Draws = draws;
            TimedOut = timedOut;
            AvgRounds = avgRounds;
            AvgCommands = avgCommands;
        }

        public double Player0WinRate => Games == 0 ? 0 : (double)Player0Wins / Games;
        public double Player1WinRate => Games == 0 ? 0 : (double)Player1Wins / Games;
        public double DrawRate => Games == 0 ? 0 : (double)Draws / Games;
        public double TimeoutRate => Games == 0 ? 0 : (double)TimedOut / Games;
    }
}

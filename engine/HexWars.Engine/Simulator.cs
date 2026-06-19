using System;

namespace HexWars.Engine
{
    /// <summary>
    /// Runs a batch of headless self-play matches and aggregates the outcomes into a
    /// <see cref="BatchReport"/>. Each game gets its own start state (so boards can vary by seed) and
    /// fresh, seeded agents, so the whole batch is reproducible. The substrate for balance testing.
    /// </summary>
    public static class Simulator
    {
        /// <param name="startFactory">Builds the start state for game i (ignore i for a fixed board).</param>
        /// <param name="agentFactory">Builds a seeded agent; each game seeds its two agents distinctly.</param>
        public static BatchReport RunBatch(Func<int, GameState> startFactory, Func<int, IAgent> agentFactory,
                                           int games, int maxCommands)
            => RunBatch(startFactory, agentFactory, agentFactory, games, maxCommands);

        /// <summary>Asymmetric batch — different agent per slot (e.g. greedy vs random). Player-0 stats
        /// are the first factory's; a win-rate skew is that matchup's edge plus any first-move bias.</summary>
        public static BatchReport RunBatch(Func<int, GameState> startFactory,
                                           Func<int, IAgent> agent0Factory, Func<int, IAgent> agent1Factory,
                                           int games, int maxCommands)
        {
            int p0 = 0, p1 = 0, draws = 0, timedOut = 0;
            long rounds = 0, commands = 0;

            for (int i = 0; i < games; i++)
            {
                var agent0 = agent0Factory(2 * i + 1);
                var agent1 = agent1Factory(2 * i + 2);
                var r = Match.Run(startFactory(i), agent0, agent1, maxCommands);

                if (r.TimedOut) timedOut++;
                if (r.Winner == PlayerId.Player0) p0++;
                else if (r.Winner == PlayerId.Player1) p1++;
                else draws++;

                rounds += r.Rounds;
                commands += r.Commands;
            }

            double avgRounds = games == 0 ? 0 : (double)rounds / games;
            double avgCommands = games == 0 ? 0 : (double)commands / games;
            return new BatchReport(games, p0, p1, draws, timedOut, avgRounds, avgCommands);
        }
    }
}

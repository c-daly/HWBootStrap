using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GreedyAgentTests
    {
        private static GameState NewGame()
        {
            var board = new RandomBoardGenerator(BoardGenConfig.Default()).Generate(98765);
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 30),
                new PlayerState(PlayerId.Player1, 30),
            };
            return new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 1);
        }

        // A competent heuristic must beat random play decisively; playing both slots removes any
        // first-move bias so the margin reflects skill, not seating.
        [Test]
        public void Greedy_BeatsRandom_OverABalancedBatch()
        {
            int games = 0, greedyWins = 0, decisive = 0;
            for (int i = 0; i < 15; i++)
            {
                var asP0 = Match.Run(NewGame(), new GreedyAgent(2 * i + 1), new RandomAgent(2 * i + 2), 2000);
                games++;
                if (asP0.Winner != null) decisive++;
                if (asP0.Winner == PlayerId.Player0) greedyWins++;

                var asP1 = Match.Run(NewGame(), new RandomAgent(2 * i + 1), new GreedyAgent(2 * i + 2), 2000);
                games++;
                if (asP1.Winner != null) decisive++;
                if (asP1.Winner == PlayerId.Player1) greedyWins++;
            }

            Assert.That(decisive, Is.GreaterThan(games / 2), "most games should resolve, not draw out");
            Assert.That(greedyWins, Is.GreaterThan(games * 0.6), $"greedy won only {greedyWins}/{games}");
        }

        [Test]
        public void RunBatch_WithTwoAgentTypes_TalliesAndFavorsGreedy()
        {
            var rep = Simulator.RunBatch(
                _ => NewGame(),
                seed => new GreedyAgent(seed),   // Player 0
                seed => new RandomAgent(seed),   // Player 1
                games: 12, maxCommands: 2000);

            Assert.That(rep.Player0Wins + rep.Player1Wins + rep.Draws, Is.EqualTo(12));
            Assert.That(rep.Player0Wins, Is.GreaterThan(rep.Player1Wins));
        }
    }
}

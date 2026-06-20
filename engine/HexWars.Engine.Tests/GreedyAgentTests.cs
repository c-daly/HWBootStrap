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

        // With win-by-annihilation only, economy games (endless reinforcement) mostly draw, so we
        // measure competence by FORCE: greedy should end with far more position value than random.
        // Playing both slots removes first-move bias.
        [Test]
        public void Greedy_OutvaluesRandom_OverABalancedBatch()
        {
            long greedyValue = 0, randomValue = 0;
            for (int i = 0; i < 15; i++)
            {
                var asP0 = Match.Run(NewGame(), new GreedyAgent(2 * i + 1), new RandomAgent(2 * i + 2), 3000);
                greedyValue += WinCheck.Evaluate(asP0.Final, PlayerId.Player0);
                randomValue += WinCheck.Evaluate(asP0.Final, PlayerId.Player1);

                var asP1 = Match.Run(NewGame(), new RandomAgent(2 * i + 1), new GreedyAgent(2 * i + 2), 3000);
                greedyValue += WinCheck.Evaluate(asP1.Final, PlayerId.Player1);
                randomValue += WinCheck.Evaluate(asP1.Final, PlayerId.Player0);
            }

            Assert.That(greedyValue, Is.GreaterThan(randomValue), $"greedy {greedyValue} vs random {randomValue}");
        }

        [Test]
        public void RunBatch_WithTwoAgentTypes_TalliesEveryGame()
        {
            var rep = Simulator.RunBatch(
                _ => NewGame(),
                seed => new GreedyAgent(seed),   // Player 0
                seed => new RandomAgent(seed),   // Player 1
                games: 12, maxCommands: 3000);

            Assert.That(rep.Player0Wins + rep.Player1Wins + rep.Draws, Is.EqualTo(12));
            Assert.That(rep.AvgRounds, Is.GreaterThan(0));
        }
    }
}

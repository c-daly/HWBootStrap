using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class SelfPlayTests
    {
        private static GameState NewGame(int startingPoints)
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 5; q++)
                tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));

            var board = new Board(tiles, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(4, 0) });
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, startingPoints),
                new PlayerState(PlayerId.Player1, startingPoints),
            };
            return new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 1);
        }

        [Test]
        public void RandomSelfPlay_Terminates_WithinCommandBudget()
        {
            var final = Match.Play(NewGame(10), new RandomAgent(1), new RandomAgent(2), maxCommands: 5000);
            Assert.That(final.IsGameOver, Is.True);
        }

        [Test]
        public void RandomSelfPlay_IsReproducible_ForTheSameSeeds()
        {
            var a = Match.Play(NewGame(10), new RandomAgent(7), new RandomAgent(8), maxCommands: 5000);
            var b = Match.Play(NewGame(10), new RandomAgent(7), new RandomAgent(8), maxCommands: 5000);

            Assert.That(b.Round, Is.EqualTo(a.Round));
            Assert.That(b.Winner, Is.EqualTo(a.Winner));
        }
    }
}

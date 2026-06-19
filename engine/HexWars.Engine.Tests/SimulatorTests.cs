using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class SimulatorTests
    {
        private static GameState NewGame()
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 5; q++)
                tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));

            var board = new Board(tiles, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(4, 0) });
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 10),
                new PlayerState(PlayerId.Player1, 10),
            };
            return new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 1);
        }

        [Test]
        public void Run_ReportsCleanTermination_AndMatchesFinalState()
        {
            var r = Match.Run(NewGame(), new RandomAgent(1), new RandomAgent(2), maxCommands: 5000);
            Assert.That(r.TimedOut, Is.False);                  // ends within the command budget
            Assert.That(r.Winner, Is.EqualTo(r.Final.Winner));
            Assert.That(r.Rounds, Is.EqualTo(r.Final.Round));
            Assert.That(r.Commands, Is.GreaterThan(0));
        }

        [Test]
        public void Run_IsReproducible_ForTheSameSeeds()
        {
            var a = Match.Run(NewGame(), new RandomAgent(7), new RandomAgent(8), 5000);
            var b = Match.Run(NewGame(), new RandomAgent(7), new RandomAgent(8), 5000);
            Assert.That(b.Winner, Is.EqualTo(a.Winner));
            Assert.That(b.Rounds, Is.EqualTo(a.Rounds));
            Assert.That(b.Commands, Is.EqualTo(a.Commands));
        }

        [Test]
        public void RunBatch_TalliesEveryGame()
        {
            var rep = Simulator.RunBatch(_ => NewGame(), seed => new RandomAgent(seed), games: 20, maxCommands: 5000);
            Assert.That(rep.Games, Is.EqualTo(20));
            Assert.That(rep.Player0Wins + rep.Player1Wins + rep.Draws, Is.EqualTo(20));
            Assert.That(rep.TimedOut, Is.LessThanOrEqualTo(20));
            Assert.That(rep.AvgRounds, Is.GreaterThan(0));
        }

        [Test]
        public void RunBatch_IsReproducible()
        {
            var a = Simulator.RunBatch(_ => NewGame(), seed => new RandomAgent(seed), 12, 5000);
            var b = Simulator.RunBatch(_ => NewGame(), seed => new RandomAgent(seed), 12, 5000);
            Assert.That(b.Player0Wins, Is.EqualTo(a.Player0Wins));
            Assert.That(b.Player1Wins, Is.EqualTo(a.Player1Wins));
            Assert.That(b.Draws, Is.EqualTo(a.Draws));
        }
    }
}

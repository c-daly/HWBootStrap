using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class ReplayTests
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
        public void Record_ThenReplay_ReconstructsTheFinalState()
        {
            var rec = Match.Record(NewGame(), new RandomAgent(3), new RandomAgent(4), maxCommands: 5000);
            var replay = new Replay(rec);

            Assert.That(replay.FrameCount, Is.EqualTo(rec.Commands.Count + 1)); // start + one per applied command
            Assert.That(replay.Final.Round, Is.EqualTo(rec.Result.Rounds));
            Assert.That(replay.Final.Winner, Is.EqualTo(rec.Result.Winner));
            Assert.That(replay.Final.IsGameOver, Is.EqualTo(rec.Result.Final.IsGameOver));
        }

        [Test]
        public void Replay_FramesRunFromStartToFinish()
        {
            var rec = Match.Record(NewGame(), new RandomAgent(1), new RandomAgent(2), maxCommands: 5000);
            var replay = new Replay(rec);

            Assert.That(replay.Frame(0).Round, Is.EqualTo(1));                         // first frame = start
            Assert.That(replay.Frame(replay.FrameCount - 1).IsGameOver, Is.True);      // last frame = terminal
        }

        [Test]
        public void Replay_IsReproducible_ForTheSameSeeds()
        {
            var a = new Replay(Match.Record(NewGame(), new RandomAgent(7), new RandomAgent(8), 5000));
            var b = new Replay(Match.Record(NewGame(), new RandomAgent(7), new RandomAgent(8), 5000));

            Assert.That(b.FrameCount, Is.EqualTo(a.FrameCount));
            Assert.That(b.Final.Winner, Is.EqualTo(a.Final.Winner));
        }
    }
}

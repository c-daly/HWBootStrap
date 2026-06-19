using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class ReplayFileTests
    {
        private const PlayerId P0 = PlayerId.Player0;
        private const PlayerId P1 = PlayerId.Player1;

        private static GameState AgentGame()
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 5; q++)
                tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));
            var board = new Board(tiles, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(4, 0) });
            var players = new[] { new PlayerState(P0, 10), new PlayerState(P1, 10) };
            return new GameState(board, GameConfig.Default(), players, P0, 1, 1);
        }

        [Test]
        public void WriteThenRead_ReplaysToTheSameFinalState()
        {
            var rec = Match.Record(AgentGame(), new RandomAgent(3), new RandomAgent(4), maxCommands: 5000);
            var data = ReplayFile.Read(ReplayFile.Write(rec));
            var replay = new Replay(data.Start, data.Commands);

            Assert.That(data.Commands.Count, Is.EqualTo(rec.Commands.Count));
            Assert.That(replay.Final.Round, Is.EqualTo(rec.Result.Rounds));
            Assert.That(replay.Final.Winner, Is.EqualTo(rec.Result.Winner));
            Assert.That(replay.Final.IsGameOver, Is.EqualTo(rec.Result.Final.IsGameOver));
        }

        [Test]
        public void RichStartState_RoundTrips()
        {
            var board = new RandomBoardGenerator(BoardGenConfig.Default()).Generate(7);
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var z0 = new List<HexCoord>(board.DeploymentZone(P0));
            var z1 = new List<HexCoord>(board.DeploymentZone(P1));

            var u0 = new Unit(1, P0, stats, z0[0], board.TileAt(z0[0]).Elevation);
            var g0 = new Generator(2, P0, z0[1], board.TileAt(z0[1]).Elevation, 10);
            var p0 = new PlayerState(P0, 15, new[] { stats }, new[] { u0 }, new[] { g0 });
            var u1 = new Unit(3, P1, stats, z1[0], board.TileAt(z1[0]).Elevation);
            var p1 = new PlayerState(P1, 15, null, new[] { u1 }, null);
            var start = new GameState(board, GameConfig.Default(), new[] { p0, p1 }, P0, 1, 4);

            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;

            foreach (var t in board.Tiles)
            {
                var rt = s.Board.TileAt(t.Coord);
                Assert.That(rt.Elevation, Is.EqualTo(t.Elevation));
                Assert.That(rt.Terrain, Is.EqualTo(t.Terrain));
            }
            Assert.That(s.Board.DeploymentZone(P0).Count, Is.EqualTo(board.DeploymentZone(P0).Count));
            Assert.That(s.NextEntityId, Is.EqualTo(4));

            var rp0 = s.Player(P0);
            Assert.That(rp0.Points, Is.EqualTo(15));
            Assert.That(rp0.UnitsOnBoard.Count, Is.EqualTo(1));
            Assert.That(rp0.UnitsOnBoard[0].Cell, Is.EqualTo(u0.Cell));
            Assert.That(rp0.UnitsOnBoard[0].Stats.Damage, Is.EqualTo(3));
            Assert.That(rp0.Generators.Count, Is.EqualTo(1));
            Assert.That(rp0.Generators[0].CurrentHp, Is.EqualTo(10));
            Assert.That(rp0.Barracks.Count, Is.EqualTo(1));
            Assert.That(s.Player(P1).UnitsOnBoard.Count, Is.EqualTo(1));
        }
    }
}

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
        public void TurnActionBudget_RoundTrips()
        {
            // The START message online is a ReplayFile dump — the pace (K actions per turn) must
            // survive it, or the client falls back to whole-army and the HUD can't show the budget.
            var tiles = new List<Tile>();
            for (int q = 0; q < 5; q++)
                tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));
            var board = new Board(tiles, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(4, 0) });
            var players = new[] { new PlayerState(P0, 10), new PlayerState(P1, 10) };
            var start = new GameState(board, GameConfig.Default(turnPolicy: new KActionsPolicy(3)),
                                      players, P0, 1, 1);

            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;
            Assert.That(s.Config.TurnPolicy.ActionsPerTurn, Is.EqualTo(3));

            var unlimited = ReplayFile.Read(ReplayFile.Write(AgentGame(), new List<Command>())).Start;
            Assert.That(unlimited.Config.TurnPolicy.ActionsPerTurn, Is.Null,
                "a whole-army game must not grow a budget in the round trip");
        }

        [Test]
        public void EffectiveConfig_RoundTrips_ThroughTheWire()
        {
            // The START message online must carry the rules: otherwise the client re-simulates the
            // game under different config than the server (damageFloor 0 => "0 damage" attacks,
            // territoryMode false => dead claims, wrong win conditions => wrong winner).
            var start = GameFactory.Build(new GameSetup(GameMode.Territory, 11, 9, 40, 7, turnActions: 3));
            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;

            Assert.That(s.Config.TerritoryMode, Is.True, "territory mode");
            Assert.That(s.Config.DamageFloor, Is.EqualTo(start.Config.DamageFloor), "damage floor");
            Assert.That(s.Config.WinConditions, Is.EqualTo(start.Config.WinConditions), "win conditions");
            Assert.That(s.Config.StartingPoints, Is.EqualTo(start.Config.StartingPoints), "starting points");
            Assert.That(s.Config.ClaimEndsTurn, Is.EqualTo(start.Config.ClaimEndsTurn), "claim ends turn");
            Assert.That(s.Config.CaptureCost, Is.EqualTo(start.Config.CaptureCost), "capture cost");
            Assert.That(s.Config.TerritoryIncome, Is.EqualTo(start.Config.TerritoryIncome), "territory income");
            Assert.That(s.Config.GeneratorsEnabled, Is.EqualTo(start.Config.GeneratorsEnabled), "generators");
            Assert.That(s.Config.UpkeepFactor, Is.EqualTo(start.Config.UpkeepFactor), "upkeep factor");
            Assert.That(s.Config.PointDecay, Is.EqualTo(start.Config.PointDecay), "point decay");
            Assert.That(s.Config.TurnPolicy.ActionsPerTurn, Is.EqualTo(3), "pace");
        }

        [Test]
        public void TerritoryControl_RoundTrips_ThroughTheWire()
        {
            var start = GameFactory.Build(new GameSetup(GameMode.Territory, 11, 9, 40, 7));
            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;

            Assert.That(s.Board.ControlledCount(P0), Is.GreaterThan(0), "P0 home zone control survives");
            foreach (var t in start.Board.Tiles)
                Assert.That(s.Board.Controller(t.Coord), Is.EqualTo(start.Board.Controller(t.Coord)),
                    $"control of {t.Coord} must survive the wire");
        }

        [Test]
        public void ClientLockstep_ReplayingServerCommands_ReachesTheSameState()
        {
            // Online, the client applies every server-echoed command to its own copy of the state.
            // Same engine + same config + same start => it must never reject one, and must land on
            // the exact same final state as the server.
            var server = GameFactory.Build(new GameSetup(GameMode.Territory, 11, 9, 40, 7, turnActions: 3));
            var rec = Match.Record(server, new RandomAgent(3), new RandomAgent(4), maxCommands: 2000);

            var client = ReplayFile.Read(ReplayFile.Write(server, new List<Command>())).Start;
            foreach (var c in rec.Commands)
            {
                var r = GameEngine.Apply(client, c);
                Assert.That(r.Success, Is.True,
                    $"client must accept the server-validated {c.GetType().Name} (got {r.Reason})");
                client = r.NewState;
            }

            var final = rec.Result.Final;
            Assert.That(client.Round, Is.EqualTo(final.Round), "round");
            Assert.That(client.Winner, Is.EqualTo(final.Winner), "winner");
            Assert.That(client.IsGameOver, Is.EqualTo(final.IsGameOver), "game over");
            foreach (var p in final.Players)
            {
                Assert.That(client.Player(p.Id).Points, Is.EqualTo(p.Points), $"{p.Id} points");
                Assert.That(client.Player(p.Id).UnitsOnBoard.Count, Is.EqualTo(p.UnitsOnBoard.Count), $"{p.Id} units");
            }
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

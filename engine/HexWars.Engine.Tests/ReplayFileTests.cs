using System;
using System.Collections.Generic;
using System.Linq;
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
        public void OneActionTurnPolicy_RoundTripsWithoutBecomingKActionsPolicy()
        {
            GameState baseState = AgentGame();
            var start = new GameState(
                baseState.Board,
                GameConfig.Default(turnPolicy: new OneActionPolicy()),
                baseState.Players,
                baseState.ActivePlayer,
                baseState.Round,
                baseState.NextEntityId);

            GameConfig actual = ReplayFile.Read(
                ReplayFile.Write(start, new List<Command>())).Start.Config;

            Assert.That(actual.TurnPolicy, Is.TypeOf<OneActionPolicy>());
            Assert.That(actual.TurnPolicy.ActionsPerTurn, Is.EqualTo(1));
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
        public void SparseTerrainConfig_RoundTripsPresentTerrainWithoutRequiringUnusedDefinitions()
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>
            {
                { TerrainType.Plains, new TerrainDef(3, 2, 1, true) },
            };
            GameState baseState = AgentGame();
            var start = new GameState(
                baseState.Board,
                new GameConfig(terrain),
                baseState.Players,
                baseState.ActivePlayer,
                baseState.Round,
                baseState.NextEntityId);

            GameState actual = ReplayFile.Read(
                ReplayFile.Write(start, new List<Command>())).Start;

            Assert.That(actual.Config.Terrain(TerrainType.Plains).MoveCost, Is.EqualTo(3));
        }

        [Test]
        public void AllAuthoritativeGameplayRules_RoundTrip()
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>
            {
                { TerrainType.Plains, new TerrainDef(2, 1, 3, true) },
                { TerrainType.Forest, new TerrainDef(4, 5, 6, true) },
                { TerrainType.Rough, new TerrainDef(7, 8, 9, false) },
                { TerrainType.Water, new TerrainDef(10, 11, 12, true) },
            };
            var expected = new GameConfig(
                terrain,
                startingPoints: 23,
                bountyRate: 0.75,
                generatorCost: 7,
                generatorOutput: 5,
                generatorHealth: 13,
                damageFloor: 2,
                dmgHighGroundBonus: 4,
                rangeHighGroundBonus: 3,
                roundCap: 17,
                designFee: 6,
                deployCostMultiplier: 0.25,
                turnPolicy: new KActionsPolicy(3),
                biomesEnabled: true,
                winConditions: WinBy.Annihilation | WinBy.Economy | WinBy.Score,
                captureCost: 9,
                economyWinThreshold: 123,
                scoreKills: 2,
                scorePoints: 3,
                scoreArmy: 4,
                scoreTerritory: 5,
                upkeepFactor: 0.33,
                captureFactor: 2.5,
                buildFactor: 7.25,
                territoryMode: true,
                claimEndsTurn: false,
                buildAnywhere: true,
                territoryIncome: 8,
                generatorsEnabled: false,
                pointDecay: 0.125,
                fogOfWar: true,
                maxDesignPointCost: 41,
                fixedTemplateCount: 2,
                templateSlotCount: 7);
            GameState baseState = AgentGame();
            var start = new GameState(baseState.Board, expected, baseState.Players,
                baseState.ActivePlayer, baseState.Round, baseState.NextEntityId);

            GameConfig actual = ReplayFile.Read(
                ReplayFile.Write(start, new List<Command>())).Start.Config;

            Assert.Multiple(() =>
            {
                Assert.That(actual.StartingPoints, Is.EqualTo(expected.StartingPoints));
                Assert.That(actual.BountyRate, Is.EqualTo(expected.BountyRate));
                Assert.That(actual.GeneratorCost, Is.EqualTo(expected.GeneratorCost));
                Assert.That(actual.GeneratorOutput, Is.EqualTo(expected.GeneratorOutput));
                Assert.That(actual.GeneratorHealth, Is.EqualTo(expected.GeneratorHealth));
                Assert.That(actual.DamageFloor, Is.EqualTo(expected.DamageFloor));
                Assert.That(actual.DmgHighGroundBonus, Is.EqualTo(expected.DmgHighGroundBonus));
                Assert.That(actual.RangeHighGroundBonus, Is.EqualTo(expected.RangeHighGroundBonus));
                Assert.That(actual.RoundCap, Is.EqualTo(expected.RoundCap));
                Assert.That(actual.DesignFee, Is.EqualTo(expected.DesignFee));
                Assert.That(actual.DeployCostMultiplier, Is.EqualTo(expected.DeployCostMultiplier));
                Assert.That(actual.TurnPolicy.ActionsPerTurn, Is.EqualTo(3));
                Assert.That(actual.BiomesEnabled, Is.EqualTo(expected.BiomesEnabled));
                Assert.That(actual.WinConditions, Is.EqualTo(expected.WinConditions));
                Assert.That(actual.CaptureCost, Is.EqualTo(expected.CaptureCost));
                Assert.That(actual.EconomyWinThreshold, Is.EqualTo(expected.EconomyWinThreshold));
                Assert.That(actual.ScoreKills, Is.EqualTo(expected.ScoreKills));
                Assert.That(actual.ScorePoints, Is.EqualTo(expected.ScorePoints));
                Assert.That(actual.ScoreArmy, Is.EqualTo(expected.ScoreArmy));
                Assert.That(actual.ScoreTerritory, Is.EqualTo(expected.ScoreTerritory));
                Assert.That(actual.UpkeepFactor, Is.EqualTo(expected.UpkeepFactor));
                Assert.That(actual.CaptureFactor, Is.EqualTo(expected.CaptureFactor));
                Assert.That(actual.BuildFactor, Is.EqualTo(expected.BuildFactor));
                Assert.That(actual.TerritoryMode, Is.EqualTo(expected.TerritoryMode));
                Assert.That(actual.ClaimEndsTurn, Is.EqualTo(expected.ClaimEndsTurn));
                Assert.That(actual.BuildAnywhere, Is.EqualTo(expected.BuildAnywhere));
                Assert.That(actual.TerritoryIncome, Is.EqualTo(expected.TerritoryIncome));
                Assert.That(actual.GeneratorsEnabled, Is.EqualTo(expected.GeneratorsEnabled));
                Assert.That(actual.PointDecay, Is.EqualTo(expected.PointDecay));
                Assert.That(actual.FogOfWar, Is.EqualTo(expected.FogOfWar));
                Assert.That(actual.MaxDesignPointCost, Is.EqualTo(expected.MaxDesignPointCost));
                Assert.That(actual.FixedTemplateCount, Is.EqualTo(expected.FixedTemplateCount));
                Assert.That(actual.TemplateSlotCount, Is.EqualTo(expected.TemplateSlotCount));
            });
            foreach (TerrainType terrainType in Enum.GetValues(typeof(TerrainType)))
            {
                TerrainDef expectedTerrain = expected.Terrain(terrainType);
                TerrainDef actualTerrain = actual.Terrain(terrainType);
                Assert.Multiple(() =>
                {
                    Assert.That(actualTerrain.MoveCost, Is.EqualTo(expectedTerrain.MoveCost), terrainType.ToString());
                    Assert.That(actualTerrain.Concealment, Is.EqualTo(expectedTerrain.Concealment), terrainType.ToString());
                    Assert.That(actualTerrain.Defense, Is.EqualTo(expectedTerrain.Defense), terrainType.ToString());
                    Assert.That(actualTerrain.Passable, Is.EqualTo(expectedTerrain.Passable), terrainType.ToString());
                });
            }
        }


        [Test]
        public void OldConfigWithoutNewRuleKeys_UsesBackwardCompatibleDefaults()
        {
            string modern = ReplayFile.Write(AgentGame(), new List<Command>());
            string old = modern
                .Replace(" designFee=0", "")
                .Replace(" maxDesignCost=0", "")
                .Replace(" fixedTemplates=0", "")
                .Replace(" templateSlots=0", "");
            old = string.Join("\n", old.Replace("\r\n", "\n").Split('\n').Select(line =>
                line.StartsWith("CONFIG", StringComparison.Ordinal)
                    ? string.Join(" ", line.Split(' ').Where(token =>
                        !token.StartsWith("bounty=", StringComparison.Ordinal) &&
                        !token.StartsWith("genCost=", StringComparison.Ordinal) &&
                        !token.StartsWith("genHealth=", StringComparison.Ordinal) &&
                        !token.StartsWith("dmgHigh=", StringComparison.Ordinal) &&
                        !token.StartsWith("rangeHigh=", StringComparison.Ordinal) &&
                        !token.StartsWith("roundCap=", StringComparison.Ordinal) &&
                        !token.StartsWith("deployMultiplier=", StringComparison.Ordinal) &&
                        !token.StartsWith("terrain", StringComparison.Ordinal))) : line));

            var s = ReplayFile.Read(old).Start;

            Assert.That(s.Config.DesignFee, Is.EqualTo(0));
            Assert.That(s.Config.MaxDesignPointCost, Is.EqualTo(0));
            Assert.That(s.Config.FixedTemplateCount, Is.EqualTo(0));
            Assert.That(s.Config.BountyRate, Is.EqualTo(0.5));
            Assert.That(s.Config.GeneratorCost, Is.EqualTo(2));
            Assert.That(s.Config.GeneratorHealth, Is.EqualTo(3));
            Assert.That(s.Config.DmgHighGroundBonus, Is.EqualTo(1));
            Assert.That(s.Config.RangeHighGroundBonus, Is.EqualTo(1));
            Assert.That(s.Config.RoundCap, Is.EqualTo(GameConfig.DefaultRoundCap));
            Assert.That(s.Config.DeployCostMultiplier, Is.EqualTo(1.0));
            Assert.That(s.Config.Terrain(TerrainType.Forest).MoveCost, Is.EqualTo(2));
            Assert.That(s.Config.TemplateSlotCount, Is.EqualTo(0));
        }

        [Test]
        public void LegacyReplayWithoutConfigLine_UsesHistoricalDefaults()
        {
            const string legacy =
                "HEXWARS-REPLAY 1\n" +
                "META 1 0 1\n" +
                "TILES 1\n" +
                "0 0 0 0\n" +
                "ZONE0 1 0 0\n" +
                "ZONE1 0\n" +
                "PLAYER 0 10 0 0 0\n" +
                "PLAYER 1 10 0 0 0\n" +
                "CMDS 0\n";

            GameState state = ReplayFile.Read(legacy).Start;

            Assert.Multiple(() =>
            {
                Assert.That(state.Config.StartingPoints, Is.EqualTo(12));
                Assert.That(state.Config.BountyRate, Is.EqualTo(0.5));
                Assert.That(state.Config.RoundCap, Is.EqualTo(GameConfig.DefaultRoundCap));
                Assert.That(state.Config.TurnPolicy, Is.TypeOf<AllUnitsPolicy>());
            });
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
            var p0 = new PlayerState(P0, 15, new[] { new UnitTemplate("Vanguard", stats) }, new[] { u0 }, new[] { g0 });
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
            Assert.That(rp0.Barracks[0].Name, Is.EqualTo("Vanguard"));
            Assert.That(s.Player(P1).UnitsOnBoard.Count, Is.EqualTo(1));
        }

        [Test]
        public void SeededStartState_RoundTrips_WithNames()
        {
            var start = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7));
            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;

            var expected = new[] { "Brute", "Striker", "Sniper", "Artillery", "Scout" };
            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                var barracks = s.Player(pid).Barracks;
                Assert.That(barracks.Count, Is.EqualTo(5));
                for (int i = 0; i < 5; i++)
                    Assert.That(barracks[i].Name, Is.EqualTo(expected[i]), $"{pid} slot {i}");
            }
        }

        [Test]
        public void UnitAndBarracks_Name_RoundTrip()
        {
            var board = new RandomBoardGenerator(BoardGenConfig.Default()).Generate(7);
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var z0 = new List<HexCoord>(board.DeploymentZone(P0));

            var u0 = new Unit(1, P0, stats, z0[0], board.TileAt(z0[0]).Elevation, "Doom Turtle");
            var p0 = new PlayerState(P0, 10, new[] { new UnitTemplate("Longshot", stats) }, new[] { u0 });
            var p1 = new PlayerState(P1, 10);
            var start = new GameState(board, GameConfig.Default(), new[] { p0, p1 }, P0, 1, 2);

            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;
            var rp0 = s.Player(P0);

            Assert.That(rp0.UnitsOnBoard[0].Name, Is.EqualTo("Doom Turtle"));
            Assert.That(rp0.Barracks[0].Name, Is.EqualTo("Longshot"));
        }

        [Test]
        public void OldFormatReplay_MissingNameTokens_DefaultsToEmptyNames()
        {
            var board = new RandomBoardGenerator(BoardGenConfig.Default()).Generate(7);
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var z0 = new List<HexCoord>(board.DeploymentZone(P0));

            var u0 = new Unit(1, P0, stats, z0[0], board.TileAt(z0[0]).Elevation, "Recon");
            var p0 = new PlayerState(P0, 10, new[] { new UnitTemplate("Vanguard", stats) }, new[] { u0 });
            var p1 = new PlayerState(P1, 10);
            var start = new GameState(board, GameConfig.Default(), new[] { p0, p1 }, P0, 1, 2);

            string modern = ReplayFile.Write(start, new List<Command>());

            // Simulate a pre-name-feature payload by dropping the trailing name token from each U/B
            // line — exactly what a file written before this feature shipped looks like.
            var oldLines = modern.Replace("\r\n", "\n").Split('\n');
            for (int i = 0; i < oldLines.Length; i++)
            {
                if (oldLines[i].StartsWith("U ", StringComparison.Ordinal) || oldLines[i].StartsWith("B ", StringComparison.Ordinal))
                {
                    int lastSpace = oldLines[i].LastIndexOf(' ');
                    oldLines[i] = oldLines[i].Substring(0, lastSpace);
                }
            }
            string old = string.Join("\n", oldLines);

            var s = ReplayFile.Read(old).Start;
            var rp0 = s.Player(P0);
            Assert.That(rp0.UnitsOnBoard[0].Name, Is.EqualTo(""), "old payloads with no trailing unit-name token default to \"\"");
            Assert.That(rp0.Barracks[0].Name, Is.EqualTo(""), "old payloads with no trailing barracks-name token default to \"\"");
            Assert.That(rp0.UnitsOnBoard[0].Stats.Damage, Is.EqualTo(3), "everything else still parses");
        }
    }
}

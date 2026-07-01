using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace HexWars.Engine
{
    /// <summary>Start state + commands parsed from a replay file.</summary>
    public readonly struct ReplayData
    {
        public readonly GameState Start;
        public readonly IReadOnlyList<Command> Commands;
        public ReplayData(GameState start, IReadOnlyList<Command> commands) { Start = start; Commands = commands; }
    }

    /// <summary>
    /// Portable, dependency-free text serialization of a match (start state + applied commands). The
    /// SAME engine code reads and writes it, so a file written headless in WSL2 reconstructs identically
    /// in Unity — no format drift. Loads GameConfig.Default(), preserving the recorded biomes-on/off flag
    /// (META) so terrain rules replay faithfully; start-state units/generators are assumed full health.
    /// Feed <see cref="ReplayData"/> to a Replay.
    /// </summary>
    public static class ReplayFile
    {
        private const string Header = "HEXWARS-REPLAY 1";

        public static string Write(MatchRecord record) => Write(record.Start, record.Commands);

        public static string Write(GameState s, IReadOnlyList<Command> commands)
        {
            var sb = new StringBuilder();
            sb.Append(Header).Append('\n');
            sb.Append("META ").Append(s.NextEntityId).Append(' ').Append((int)s.ActivePlayer).Append(' ').Append(s.Round)
              .Append(' ').Append(s.Config.BiomesEnabled ? 1 : 0)
              .Append(' ').Append(s.Config.TurnPolicy.ActionsPerTurn ?? 0).Append('\n');

            var tiles = new List<Tile>(s.Board.Tiles);
            sb.Append("TILES ").Append(tiles.Count).Append('\n');
            foreach (var t in tiles)
                sb.Append(t.Coord.Q).Append(' ').Append(t.Coord.R).Append(' ').Append(t.Elevation).Append(' ').Append((int)t.Terrain).Append('\n');

            WriteZone(sb, "ZONE0", s.Board.DeploymentZone(PlayerId.Player0));
            WriteZone(sb, "ZONE1", s.Board.DeploymentZone(PlayerId.Player1));

            WritePlayer(sb, s.Player(PlayerId.Player0));
            WritePlayer(sb, s.Player(PlayerId.Player1));

            sb.Append("CMDS ").Append(commands.Count).Append('\n');
            foreach (var c in commands) sb.Append(WriteCommand(c)).Append('\n');
            return sb.ToString();
        }

        public static ReplayData Read(string text)
        {
            var lines = text.Replace("\r\n", "\n").Split('\n');
            int li = 0;
            string Next() { while (li < lines.Length && lines[li].Length == 0) li++; return lines[li++]; }

            if (Next() != Header) throw new FormatException("not a HexWars replay");

            var meta = Next().Split(' ');           // META nextId active round [biomes] [turnActions]
            int nextId = int.Parse(meta[1], CultureInfo.InvariantCulture);
            var active = (PlayerId)int.Parse(meta[2], CultureInfo.InvariantCulture);
            int round = int.Parse(meta[3], CultureInfo.InvariantCulture);
            bool biomes = meta.Length <= 4 || int.Parse(meta[4], CultureInfo.InvariantCulture) != 0; // old replays: biomes on
            int turnActions = meta.Length > 5 ? int.Parse(meta[5], CultureInfo.InvariantCulture) : 0; // old replays: whole army

            int tileCount = int.Parse(Next().Split(' ')[1], CultureInfo.InvariantCulture);
            var tiles = new List<Tile>(tileCount);
            for (int i = 0; i < tileCount; i++)
            {
                var p = Next().Split(' ');
                tiles.Add(new Tile(new HexCoord(I(p[0]), I(p[1])), I(p[2]), (TerrainType)I(p[3])));
            }
            var zone0 = ReadZone(Next());
            var zone1 = ReadZone(Next());
            var board = new Board(tiles, zone0, zone1);

            var p0 = ReadPlayer(Next, PlayerId.Player0);
            var p1 = ReadPlayer(Next, PlayerId.Player1);
            var start = new GameState(board,
                GameConfig.Default(biomesEnabled: biomes,
                                   turnPolicy: turnActions > 0 ? new KActionsPolicy(turnActions) : null),
                new[] { p0, p1 }, active, round, nextId);

            int cmdCount = int.Parse(Next().Split(' ')[1], CultureInfo.InvariantCulture);
            var commands = new List<Command>(cmdCount);
            for (int i = 0; i < cmdCount; i++) commands.Add(ReadCommand(Next()));

            return new ReplayData(start, commands);
        }

        // ---- helpers ----

        private static int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);

        private static void WriteZone(StringBuilder sb, string tag, IReadOnlyCollection<HexCoord> zone)
        {
            sb.Append(tag).Append(' ').Append(zone.Count);
            foreach (var c in zone) sb.Append(' ').Append(c.Q).Append(' ').Append(c.R);
            sb.Append('\n');
        }

        private static List<HexCoord> ReadZone(string line)
        {
            var p = line.Split(' ');
            int n = I(p[1]);
            var zone = new List<HexCoord>(n);
            for (int i = 0; i < n; i++) zone.Add(new HexCoord(I(p[2 + i * 2]), I(p[3 + i * 2])));
            return zone;
        }

        private static void WritePlayer(StringBuilder sb, PlayerState p)
        {
            sb.Append("PLAYER ").Append((int)p.Id).Append(' ').Append(p.Points)
              .Append(' ').Append(p.UnitsOnBoard.Count).Append(' ').Append(p.Generators.Count)
              .Append(' ').Append(p.Barracks.Count).Append('\n');

            foreach (var u in p.UnitsOnBoard)
            {
                sb.Append("U ").Append(u.Id).Append(' ').Append((int)u.Owner).Append(' ');
                AppendStats(sb, u.Stats);
                sb.Append(' ').Append(u.Cell.Q).Append(' ').Append(u.Cell.R).Append(' ').Append(u.Elevation).Append('\n');
            }
            foreach (var g in p.Generators)
                sb.Append("G ").Append(g.Id).Append(' ').Append((int)g.Owner).Append(' ')
                  .Append(g.Cell.Q).Append(' ').Append(g.Cell.R).Append(' ').Append(g.Elevation).Append(' ').Append(g.CurrentHp).Append('\n');
            foreach (var b in p.Barracks)
            {
                sb.Append("B ");
                AppendStats(sb, b);
                sb.Append('\n');
            }
        }

        private static PlayerState ReadPlayer(Func<string> next, PlayerId expected)
        {
            var head = next().Split(' ');           // PLAYER pid points unitCount genCount barracksCount
            int points = I(head[2]);
            int units = I(head[3]), gens = I(head[4]), barr = I(head[5]);

            var unitList = new List<Unit>(units);
            var genList = new List<Generator>(gens);
            var barracks = new List<UnitStats>(barr);

            for (int i = 0; i < units; i++)
            {
                var p = next().Split(' ');           // U id owner <9 stats> q r elev
                int id = I(p[1]); var owner = (PlayerId)I(p[2]);
                var stats = ReadStats(p, 3);
                unitList.Add(new Unit(id, owner, stats, new HexCoord(I(p[12]), I(p[13])), I(p[14])));
            }
            for (int i = 0; i < gens; i++)
            {
                var p = next().Split(' ');           // G id owner q r elev hp
                genList.Add(new Generator(I(p[1]), (PlayerId)I(p[2]), new HexCoord(I(p[3]), I(p[4])), I(p[5]), I(p[6])));
            }
            for (int i = 0; i < barr; i++)
                barracks.Add(ReadStats(next().Split(' '), 1)); // B <9 stats>

            return new PlayerState(expected, points, barracks, unitList, genList);
        }

        // Stats + command encoding live in CommandWire (the networking wire-format), shared verbatim so
        // replays and the multiplayer relay never drift.
        private static void AppendStats(StringBuilder sb, UnitStats s) => sb.Append(CommandWire.WriteStats(s));

        private static UnitStats ReadStats(string[] p, int o) => CommandWire.ReadStats(p, o);

        private static string WriteCommand(Command c) => CommandWire.Write(c);

        private static Command ReadCommand(string line) => CommandWire.Read(line);
    }
}

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
            WriteConfig(sb, s.Config);

            var tiles = new List<Tile>(s.Board.Tiles);
            sb.Append("TILES ").Append(tiles.Count).Append('\n');
            foreach (var t in tiles)
                sb.Append(t.Coord.Q).Append(' ').Append(t.Coord.R).Append(' ').Append(t.Elevation).Append(' ').Append((int)t.Terrain).Append('\n');

            WriteZone(sb, "ZONE0", s.Board.DeploymentZone(PlayerId.Player0));
            WriteZone(sb, "ZONE1", s.Board.DeploymentZone(PlayerId.Player1));
            WriteControl(sb, "CONTROL0", s.Board, PlayerId.Player0);
            WriteControl(sb, "CONTROL1", s.Board, PlayerId.Player1);

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
            string Peek() { while (li < lines.Length && lines[li].Length == 0) li++; return li < lines.Length ? lines[li] : ""; }

            if (Next() != Header) throw new FormatException("not a HexWars replay");

            var meta = Next().Split(' ');           // META nextId active round [biomes] [turnActions]
            int nextId = int.Parse(meta[1], CultureInfo.InvariantCulture);
            var active = (PlayerId)int.Parse(meta[2], CultureInfo.InvariantCulture);
            int round = int.Parse(meta[3], CultureInfo.InvariantCulture);
            bool biomes = meta.Length <= 4 || int.Parse(meta[4], CultureInfo.InvariantCulture) != 0; // old replays: biomes on
            int turnActions = meta.Length > 5 ? int.Parse(meta[5], CultureInfo.InvariantCulture) : 0; // old replays: whole army

            // CONFIG is optional (old payloads lack it) — without it, defaults reproduce the old behaviour
            var cfgKv = new Dictionary<string, string>();
            if (Peek().StartsWith("CONFIG", StringComparison.Ordinal)) ParseConfig(Next(), cfgKv);

            int tileCount = int.Parse(Next().Split(' ')[1], CultureInfo.InvariantCulture);
            var tiles = new List<Tile>(tileCount);
            for (int i = 0; i < tileCount; i++)
            {
                var p = Next().Split(' ');
                tiles.Add(new Tile(new HexCoord(I(p[0]), I(p[1])), I(p[2]), (TerrainType)I(p[3])));
            }
            var zone0 = ReadZone(Next());
            var zone1 = ReadZone(Next());

            // CONTROL is optional too (old payloads: no hex is controlled at the start)
            var control = new Dictionary<HexCoord, PlayerId>();
            if (Peek().StartsWith("CONTROL0", StringComparison.Ordinal))
                foreach (var c in ReadZone(Next())) control[c] = PlayerId.Player0;
            if (Peek().StartsWith("CONTROL1", StringComparison.Ordinal))
                foreach (var c in ReadZone(Next())) control[c] = PlayerId.Player1;

            var board = new Board(tiles, zone0, zone1, control);

            var p0 = ReadPlayer(Next, PlayerId.Player0);
            var p1 = ReadPlayer(Next, PlayerId.Player1);
            var start = new GameState(board, BuildConfig(cfgKv, biomes, turnActions),
                new[] { p0, p1 }, active, round, nextId);

            int cmdCount = int.Parse(Next().Split(' ')[1], CultureInfo.InvariantCulture);
            var commands = new List<Command>(cmdCount);
            for (int i = 0; i < cmdCount; i++) commands.Add(ReadCommand(Next()));

            return new ReplayData(start, commands);
        }

        // ---- helpers ----

        private static int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);

        // The effective ruleset, as key=value pairs mapping onto GameConfig.Default's parameters —
        // every game-construction path (GameFactory, GameBootstrap, tests) builds through Default, so
        // this covers the whole varied space. Omitting a key on read falls back to Default's default,
        // which keeps old payloads parseable and lets new knobs be added without a format break.
        private static void WriteConfig(StringBuilder sb, GameConfig c)
        {
            var inv = CultureInfo.InvariantCulture;
            sb.Append("CONFIG");
            sb.Append(" win=").Append((int)c.WinConditions);
            sb.Append(" captureCost=").Append(c.CaptureCost);
            sb.Append(" economyWin=").Append(c.EconomyWinThreshold);
            sb.Append(" scoreKills=").Append(c.ScoreKills);
            sb.Append(" scorePoints=").Append(c.ScorePoints);
            sb.Append(" scoreArmy=").Append(c.ScoreArmy);
            sb.Append(" scoreTerritory=").Append(c.ScoreTerritory);
            sb.Append(" upkeep=").Append(c.UpkeepFactor.ToString("R", inv));
            sb.Append(" captureFactor=").Append(c.CaptureFactor.ToString("R", inv));
            sb.Append(" buildFactor=").Append(c.BuildFactor.ToString("R", inv));
            sb.Append(" genOutput=").Append(c.GeneratorOutput);
            sb.Append(" startingPoints=").Append(c.StartingPoints);
            sb.Append(" damageFloor=").Append(c.DamageFloor);
            sb.Append(" territory=").Append(c.TerritoryMode ? 1 : 0);
            sb.Append(" claimEndsTurn=").Append(c.ClaimEndsTurn ? 1 : 0);
            sb.Append(" buildAnywhere=").Append(c.BuildAnywhere ? 1 : 0);
            sb.Append(" territoryIncome=").Append(c.TerritoryIncome);
            sb.Append(" generators=").Append(c.GeneratorsEnabled ? 1 : 0);
            sb.Append(" pointDecay=").Append(c.PointDecay.ToString("R", inv));
            sb.Append('\n');
        }

        private static void ParseConfig(string line, Dictionary<string, string> kv)
        {
            var parts = line.Split(' ');
            for (int i = 1; i < parts.Length; i++)     // parts[0] == "CONFIG"
            {
                int eq = parts[i].IndexOf('=');
                if (eq > 0) kv[parts[i].Substring(0, eq)] = parts[i].Substring(eq + 1);
            }
        }

        private static GameConfig BuildConfig(Dictionary<string, string> kv, bool biomes, int turnActions)
        {
            int Gi(string k, int def) => kv.TryGetValue(k, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : def;
            double Gd(string k, double def) => kv.TryGetValue(k, out var v) ? double.Parse(v, CultureInfo.InvariantCulture) : def;
            bool Gb(string k, bool def) => kv.TryGetValue(k, out var v) ? v != "0" : def;

            return GameConfig.Default(
                biomesEnabled: biomes,
                turnPolicy: turnActions > 0 ? new KActionsPolicy(turnActions) : null,
                winConditions: (WinBy)Gi("win", (int)WinBy.Annihilation),
                captureCost: Gi("captureCost", 3),
                economyWinThreshold: Gi("economyWin", 200),
                scoreKills: Gi("scoreKills", 1),
                scorePoints: Gi("scorePoints", 1),
                scoreArmy: Gi("scoreArmy", 1),
                scoreTerritory: Gi("scoreTerritory", 1),
                upkeepFactor: Gd("upkeep", 0.25),
                captureFactor: Gd("captureFactor", 4.0),
                buildFactor: Gd("buildFactor", 4.0),
                generatorOutput: Gi("genOutput", 1),
                startingPoints: Gi("startingPoints", 12),
                damageFloor: Gi("damageFloor", 0),
                territoryMode: Gb("territory", false),
                claimEndsTurn: Gb("claimEndsTurn", true),
                buildAnywhere: Gb("buildAnywhere", false),
                territoryIncome: Gi("territoryIncome", 0),
                generatorsEnabled: Gb("generators", true),
                pointDecay: Gd("pointDecay", 0.0));
        }

        private static void WriteControl(StringBuilder sb, string tag, Board board, PlayerId owner)
        {
            var owned = new List<HexCoord>();
            foreach (var t in board.Tiles)
                if (board.Controller(t.Coord) == owner) owned.Add(t.Coord);
            WriteZone(sb, tag, owned);
        }

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

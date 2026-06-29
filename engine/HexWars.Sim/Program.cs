using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using HexWars.Engine;
using HexWars.Engine.Rl;

// Two modes:
//   Batch:  dotnet run --project engine/HexWars.Sim -- [matchup] [games] [maxCommands] [points]
//   Record: dotnet run --project engine/HexWars.Sim -- record [matchup] <outfile> [seed] [points]
// matchup = two chars, g=Greedy r=Random, for Player 1 / Player 2 (e.g. gg, gr, rr).

// Audit toggles: HEXWARS_BIOMES=off (terrain inert, matches the base game) and HEXWARS_TURN=one
// (one action per turn, vs the default whole-army turn). Default = biomes on, whole-army.
bool biomesOff = string.Equals(Environment.GetEnvironmentVariable("HEXWARS_BIOMES"), "off", StringComparison.OrdinalIgnoreCase);
bool oneAction = string.Equals(Environment.GetEnvironmentVariable("HEXWARS_TURN"), "one", StringComparison.OrdinalIgnoreCase);
var cfg = GameConfig.Default(biomesEnabled: !biomesOff, turnPolicy: oneAction ? new OneActionPolicy() : null);
var gen = new RandomBoardGenerator(BoardGenConfig.Default());

// Inspect: reconstruct replay file(s) and print the real final outcome.
//   dotnet run --project engine/HexWars.Sim -- inspect <file.replay> [more...]
if (args.Length >= 2 && args[0] == "inspect")
{
    for (int i = 1; i < args.Length; i++)
    {
        var data = ReplayFile.Read(File.ReadAllText(args[i]));
        var replay = new Replay(data.Start, data.Commands);
        var f = replay.Final;
        string w = f.Winner == null ? "DRAW" : (f.Winner == PlayerId.Player0 ? "P0" : "P1");
        Console.WriteLine($"{System.IO.Path.GetFileName(args[i])}: round={f.Round} winner={w}");
        foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
        {
            var p = f.Player(pid);
            int units = 0; foreach (var u in p.UnitsOnBoard) if (u.IsAlive) units++;
            int gens = 0; foreach (var g in p.Generators) if (g.IsAlive) gens++;
            Console.WriteLine($"    {pid}: bankedPoints={p.Points}  unitsOnBoard={units}  generators={gens}  totalValue={WinCheck.Evaluate(f, pid)}");
        }
    }
    return;
}

// Grid: greedy-vs-greedy TACTICAL batches over board × combat-lethality, to find params where decisive
// play is achievable (low draw rate) before committing to RL. Fast proxy: if competent agents can't make
// decisive games under a config, RL won't either.
//   dotnet run --project engine/HexWars.Sim -- grid [games]
if (args.Length >= 1 && args[0] == "grid")
{
    int games = args.Length >= 2 ? int.Parse(args[1]) : 50;
    var boards = new (string name, BoardGenConfig bg)[]
    {
        ("13x9", new BoardGenConfig(13, 9, 4, 3)),
        ("11x8", new BoardGenConfig(11, 8, 4, 3)),
        ("9x7",  new BoardGenConfig(9, 7, 4, 2)),
    };
    var rosters = new (string name, IReadOnlyList<UnitStats> roster)[]
    {
        ("std",   GridRoster("std")),
        ("glass", GridRoster("glass")),   // ~half HP, no defense -> attacks kill fast
        ("hidmg", GridRoster("hidmg")),   // +3 damage -> attacks kill fast
    };

    Console.WriteLine($"greedy-vs-greedy tactical, {games} games/config (lower draw% = decisive play achievable):");
    Console.WriteLine("board  roster | decisive%  draw%  avgRounds  avgUnitsLeft");
    foreach (var (bn, bg) in boards)
        foreach (var (rn, roster) in rosters)
        {
            var env = new EnvConfig { BoardGen = bg, Roster = roster };
            var layout = new TacticalLayout(env);
            int dec = 0, draw = 0; long rounds = 0, unitsLeft = 0;
            for (int s = 0; s < games; s++)
            {
                var (state, _, _) = layout.NewGame(s);
                var res = Match.Run(state, new GreedyAgent(2 * s + 1), new GreedyAgent(2 * s + 2), 4000);
                if (res.Winner != null) dec++; else draw++;
                rounds += res.Rounds;
                unitsLeft += AliveCount(res.Final, PlayerId.Player0) + AliveCount(res.Final, PlayerId.Player1);
            }
            Console.WriteLine($"{bn,-6} {rn,-6} | dec={100.0 * dec / games,5:F0}%  draw={100.0 * draw / games,4:F0}%  rounds={rounds / (double)games,5:F0}  units={unitsLeft / (double)games,4:F1}");
        }
    return;

    static IReadOnlyList<UnitStats> GridRoster(string kind)
    {
        var d = EnvConfig.DefaultRoster();
        if (kind == "std") return d;
        var list = new List<UnitStats>();
        foreach (var u in d)
            list.Add(kind == "glass"
                ? new UnitStats(Math.Max(1, u.Health / 2), u.Damage, 0, u.Movement, u.VerticalMovement, u.Range, u.RangeArc, u.Vision, u.VisionArc)
                : new UnitStats(u.Health, u.Damage + 3, u.Defense, u.Movement, u.VerticalMovement, u.Range, u.RangeArc, u.Vision, u.VisionArc));
        return list;
    }

    static int AliveCount(GameState s, PlayerId p)
    {
        int c = 0;
        foreach (var u in s.Player(p).UnitsOnBoard) if (u.IsAlive) c++;
        return c;
    }
}

// Territory balance: greedy-mirror batches across a matrix of territory rulesets (build placement ×
// income model × capture cost × starting points), reporting who wins, how (economy vs annihilation),
// decisiveness, length, and end-state economy.
//   dotnet run --project engine/HexWars.Sim -- territory [games]
if (args.Length >= 1 && args[0] == "territory")
{
    int games = args.Length >= 2 ? int.Parse(args[1]) : 100;
    int width = 11, height = 9, maxCmds = 4000;
    double Pc(int x) => 100.0 * x / games;

    // combat-centric territory ruleset (no economy win; gentle decay) — the lever under test is PointDecay
    GameConfig Terr(double decay) => GameConfig.Default(
        biomesEnabled: false, territoryMode: true, claimEndsTurn: true,
        winConditions: WinBy.Annihilation | WinBy.Score,
        generatorOutput: 1, startingPoints: 40, damageFloor: 1, pointDecay: decay);

    // A) Does decay keep games decisive, and does it push points into army instead of the bank?
    Console.WriteLine($"A) Point-decay sweep — Greedy mirror, {games} games, {width}x{height}, annihilation win");
    Console.WriteLine("decay |  P1%  P2% draw%  to% | rounds | armyEnd maxBank");
    foreach (var decay in new[] { 0.0, 0.1, 0.2, 0.3 })
    {
        var c = Terr(decay);
        int p0 = 0, p1 = 0, dr = 0, to = 0; long rounds = 0, army = 0, bank = 0;
        for (int s = 0; s < games; s++)
        {
            var r = Match.Run(GameFactory.BuildTerritory(c, width, height, s), new GreedyAgent(2 * s + 1), new GreedyAgent(2 * s + 2), maxCmds);
            rounds += r.Rounds;
            army += Units(r.Final, PlayerId.Player0) + Units(r.Final, PlayerId.Player1);
            bank += System.Math.Max(r.Final.Player(PlayerId.Player0).Points, r.Final.Player(PlayerId.Player1).Points);
            if (r.TimedOut) to++; else if (r.Winner == null) dr++; else if (r.Winner == PlayerId.Player0) p0++; else p1++;
        }
        Console.WriteLine($"{decay,5:F2} | {Pc(p0),4:F0} {Pc(p1),4:F0} {Pc(dr),4:F0} {Pc(to),4:F0} | {rounds / (double)games,6:F1} | {army / (double)games,6:F1} {bank / (double)games,6:F0}");
    }

    // B) Does decay punish a hoarder (banks points, never spends)? Run both seatings.
    Console.WriteLine();
    Console.WriteLine($"B) Greedy vs Hoarder — {2 * games} games, annihilation win");
    Console.WriteLine("decay | greedyWin% | hoarderEndBank");
    foreach (var decay in new[] { 0.0, 0.15 })
    {
        var c = Terr(decay);
        int gWin = 0; long bank = 0; int n = 2 * games;
        for (int s = 0; s < games; s++)
        {
            var ra = Match.Run(GameFactory.BuildTerritory(c, width, height, s), new GreedyAgent(2 * s + 1), new HoarderAgent(2 * s + 2), maxCmds);
            if (ra.Winner == PlayerId.Player0) gWin++;
            bank += ra.Final.Player(PlayerId.Player1).Points;
            var rb = Match.Run(GameFactory.BuildTerritory(c, width, height, s + 100000), new HoarderAgent(2 * s + 1), new GreedyAgent(2 * s + 2), maxCmds);
            if (rb.Winner == PlayerId.Player1) gWin++;
            bank += rb.Final.Player(PlayerId.Player0).Points;
        }
        Console.WriteLine($"{decay,5:F2} | {100.0 * gWin / n,9:F0} | {bank / (double)n,14:F0}");
    }

    // C) Where does the second-player edge come from? Compare turn structures. If P2's edge shrinks under
    // one-action-per-turn, the cause is the second-mover counter to a whole-army commitment.
    Console.WriteLine();
    Console.WriteLine($"C) Second-player advantage by turn structure — Greedy mirror, {games} games");
    Console.WriteLine("turn structure | P1(first)% P2(second)% | rounds");
    foreach (var (pname, policy) in new (string, ITurnPolicy)[] { ("whole-army", new AllUnitsPolicy()), ("one-action", new OneActionPolicy()) })
    {
        var c = GameConfig.Default(biomesEnabled: false, territoryMode: true, claimEndsTurn: true,
            winConditions: WinBy.Annihilation | WinBy.Score, generatorOutput: 1, startingPoints: 40,
            damageFloor: 1, turnPolicy: policy);
        int p0 = 0, p1 = 0; long rounds = 0;
        for (int s = 0; s < games; s++)
        {
            var r = Match.Run(GameFactory.BuildTerritory(c, width, height, s), new GreedyAgent(2 * s + 1), new GreedyAgent(2 * s + 2), maxCmds);
            rounds += r.Rounds;
            if (r.Winner == PlayerId.Player0) p0++; else if (r.Winner == PlayerId.Player1) p1++;
        }
        Console.WriteLine($"{pname,-14} | {Pc(p0),9:F0} {Pc(p1),10:F0} | {rounds / (double)games,6:F1}");
    }
    return;

    static int Units(GameState s, PlayerId p) { int n = 0; foreach (var u in s.Player(p).UnitsOnBoard) if (u.IsAlive) n++; return n; }
}

GameState NewGame(int seed, int points)
{
    var board = gen.Generate(seed + 1); // +1 so seed 0 isn't the trivial board
    var players = new[]
    {
        new PlayerState(PlayerId.Player0, points),
        new PlayerState(PlayerId.Player1, points),
    };
    return new GameState(board, cfg, players, PlayerId.Player0, 1, 1);
}

Func<int, IAgent> Make(char c) => c == 'g' ? (s => new GreedyAgent(s)) : (Func<int, IAgent>)(s => new RandomAgent(s));
string Name(char c) => c == 'g' ? "Greedy" : "Random";
string Norm(string s)
{
    s = s.ToLowerInvariant();
    return (s.Length == 2 && "gr".IndexOf(s[0]) >= 0 && "gr".IndexOf(s[1]) >= 0) ? s : "gg";
}

if (args.Length >= 3 && args[0] == "record")
{
    string mu = Norm(args[1]);
    string outfile = args[2];
    int seed = args.Length > 3 && int.TryParse(args[3], out var sd) ? sd : 0;
    int points = args.Length > 4 && int.TryParse(args[4], out var pt) ? pt : 30;

    var rec = Match.Record(NewGame(seed, points), Make(mu[0])(seed * 2 + 1), Make(mu[1])(seed * 2 + 2), 5000);
    File.WriteAllText(outfile, ReplayFile.Write(rec));
    string winner = rec.Result.Winner == null ? "draw"
        : (rec.Result.Winner == PlayerId.Player0 ? $"Player 1 ({Name(mu[0])})" : $"Player 2 ({Name(mu[1])})");
    Console.WriteLine($"recorded {Name(mu[0])}(P1) vs {Name(mu[1])}(P2) -> {outfile}");
    Console.WriteLine($"  {rec.Commands.Count} commands, {rec.Result.Rounds} rounds, winner: {winner}");
    return;
}

// ---- batch mode ----
string matchup = args.Length > 0 ? Norm(args[0]) : "gg";
int batchGames = args.Length > 1 && int.TryParse(args[1], out var gv) ? gv : 200;
int maxCommands = args.Length > 2 && int.TryParse(args[2], out var mc) ? mc : 4000;
int startingPoints = args.Length > 3 && int.TryParse(args[3], out var sp) ? sp : 30;

var sw = Stopwatch.StartNew();
var report = Simulator.RunBatch(s => NewGame(s, startingPoints), Make(matchup[0]), Make(matchup[1]), batchGames, maxCommands);
sw.Stop();

Console.WriteLine($"HexWars self-play — {report.Games} games, {Name(matchup[0])}(P1) vs {Name(matchup[1])}(P2), {startingPoints} pts  ({sw.ElapsedMilliseconds} ms)");
Console.WriteLine($"  Player 1 ({Name(matchup[0])}) wins : {report.Player0Wins,6}   {report.Player0WinRate,7:P1}");
Console.WriteLine($"  Player 2 ({Name(matchup[1])}) wins : {report.Player1Wins,6}   {report.Player1WinRate,7:P1}");
Console.WriteLine($"  Draws                  : {report.Draws,6}   {report.DrawRate,7:P1}");
Console.WriteLine($"  Timed out (no result)  : {report.TimedOut,6}   {report.TimeoutRate,7:P1}");
Console.WriteLine($"  Avg length             : {report.AvgRounds:F1} rounds, {report.AvgCommands:F0} commands/game");

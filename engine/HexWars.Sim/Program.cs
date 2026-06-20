using System;
using System.Diagnostics;
using System.IO;
using HexWars.Engine;

// Two modes:
//   Batch:  dotnet run --project engine/HexWars.Sim -- [matchup] [games] [maxCommands] [points]
//   Record: dotnet run --project engine/HexWars.Sim -- record [matchup] <outfile> [seed] [points]
// matchup = two chars, g=Greedy r=Random, for Player 1 / Player 2 (e.g. gg, gr, rr).

var cfg = GameConfig.Default();
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

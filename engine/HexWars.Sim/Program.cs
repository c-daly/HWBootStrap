using System;
using System.Diagnostics;
using HexWars.Engine;

// Usage: dotnet run --project engine/HexWars.Sim -- [matchup] [games] [maxCommands] [points]
//   matchup: two chars, g=Greedy r=Random for Player 1 / Player 2 (e.g. gg, gr, rr). Default gg.
string matchup = args.Length > 0 ? args[0].ToLowerInvariant() : "gg";
if (matchup.Length != 2 || "gr".IndexOf(matchup[0]) < 0 || "gr".IndexOf(matchup[1]) < 0)
    matchup = "gg";
int games = args.Length > 1 && int.TryParse(args[1], out var gv) ? gv : 200;
int maxCommands = args.Length > 2 && int.TryParse(args[2], out var mc) ? mc : 4000;
int startingPoints = args.Length > 3 && int.TryParse(args[3], out var sp) ? sp : 30;

var cfg = GameConfig.Default();
var gen = new RandomBoardGenerator(BoardGenConfig.Default());

GameState NewGame(int seed)
{
    var board = gen.Generate(seed + 1); // +1 so game 0 isn't the trivial seed-0 board
    var players = new[]
    {
        new PlayerState(PlayerId.Player0, startingPoints),
        new PlayerState(PlayerId.Player1, startingPoints),
    };
    return new GameState(board, cfg, players, PlayerId.Player0, 1, 1);
}

static Func<int, IAgent> Make(char c)
{
    if (c == 'g') return seed => new GreedyAgent(seed);
    return seed => new RandomAgent(seed);
}

static string Name(char c) => c == 'g' ? "Greedy" : "Random";

var sw = Stopwatch.StartNew();
var report = Simulator.RunBatch(NewGame, Make(matchup[0]), Make(matchup[1]), games, maxCommands);
sw.Stop();

Console.WriteLine($"HexWars self-play — {report.Games} games, {Name(matchup[0])}(P1) vs {Name(matchup[1])}(P2), {startingPoints} pts  ({sw.ElapsedMilliseconds} ms)");
Console.WriteLine($"  Player 1 ({Name(matchup[0])}) wins : {report.Player0Wins,6}   {report.Player0WinRate,7:P1}");
Console.WriteLine($"  Player 2 ({Name(matchup[1])}) wins : {report.Player1Wins,6}   {report.Player1WinRate,7:P1}");
Console.WriteLine($"  Draws                  : {report.Draws,6}   {report.DrawRate,7:P1}");
Console.WriteLine($"  Timed out (no result)  : {report.TimedOut,6}   {report.TimeoutRate,7:P1}");
Console.WriteLine($"  Avg length             : {report.AvgRounds:F1} rounds, {report.AvgCommands:F0} commands/game");

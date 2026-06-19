using System;
using System.Diagnostics;
using HexWars.Engine;

// Usage: dotnet run --project engine/HexWars.Sim -- [games] [maxCommands] [startingPoints]
int games = args.Length > 0 && int.TryParse(args[0], out var g) ? g : 200;
int maxCommands = args.Length > 1 && int.TryParse(args[1], out var m) ? m : 5000;
int startingPoints = args.Length > 2 && int.TryParse(args[2], out var p) ? p : 20;

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

var sw = Stopwatch.StartNew();
var report = Simulator.RunBatch(NewGame, seed => new RandomAgent(seed), games, maxCommands);
sw.Stop();

Console.WriteLine($"HexWars self-play — {report.Games} games, RandomAgent vs RandomAgent  ({sw.ElapsedMilliseconds} ms)");
Console.WriteLine($"  Player 1 (first)  wins : {report.Player0Wins,6}   {report.Player0WinRate,7:P1}");
Console.WriteLine($"  Player 2 (second) wins : {report.Player1Wins,6}   {report.Player1WinRate,7:P1}");
Console.WriteLine($"  Draws                  : {report.Draws,6}   {report.DrawRate,7:P1}");
Console.WriteLine($"  Timed out (no result)  : {report.TimedOut,6}   {report.TimeoutRate,7:P1}");
Console.WriteLine($"  Avg length             : {report.AvgRounds:F1} rounds, {report.AvgCommands:F0} commands/game");

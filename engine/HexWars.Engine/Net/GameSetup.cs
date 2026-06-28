using System.Collections.Generic;
using System.Globalization;

namespace HexWars.Engine
{
    /// <summary>Which ruleset a match uses.</summary>
    public enum GameMode { Annihilation, Territory }

    /// <summary>
    /// The host-chosen options for a match (the lobby's output). Carries to the server in one wire line
    /// so the room is built from the host's picks instead of a fixed factory.
    /// </summary>
    public readonly struct GameSetup
    {
        public readonly GameMode Mode;
        public readonly int Width;
        public readonly int Height;
        public readonly int StartingPoints;
        public readonly int Seed;

        public GameSetup(GameMode mode, int width, int height, int startingPoints, int seed)
        {
            Mode = mode; Width = width; Height = height; StartingPoints = startingPoints; Seed = seed;
        }

        public static GameSetup Default => new GameSetup(GameMode.Annihilation, 9, 7, 0, 7);

        public string ToWire() => $"{(int)Mode} {Width} {Height} {StartingPoints} {Seed}";

        public static GameSetup Parse(string wire)
        {
            var p = wire.Split(' ');
            return new GameSetup((GameMode)I(p[0]), I(p[1]), I(p[2]), I(p[3]), I(p[4]));
        }

        static int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);
    }

    /// <summary>Builds a fresh <see cref="GameState"/> from a <see cref="GameSetup"/> — the one place that
    /// turns lobby picks into a playable start state (board, ruleset, seeded armies, territory control).</summary>
    public static class GameFactory
    {
        static readonly UnitStats[] Roster =
        {
            new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1), // Brute
            new UnitStats(2, 6, 0, 3, 2, 2, 1, 3, 1), // Striker
            new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1), // Sniper
        };

        public static GameState Build(GameSetup setup)
        {
            // maxElevation 2 keeps climbs within unit vertical budgets (no unclimbable cliffs)
            var board = new RandomBoardGenerator(new BoardGenConfig(setup.Width, setup.Height, maxElevation: 2)).Generate(setup.Seed);

            bool territory = setup.Mode == GameMode.Territory;
            // biomesEnabled false: terrain is inert (no impassable/expensive tiles that box units in).
            // damageFloor 1: a landed attack by a real combatant always deals at least 1 (no 0-damage hits).
            var config = territory
                ? GameConfig.Default(biomesEnabled: false, winConditions: WinBy.Economy | WinBy.Annihilation,
                                     startingPoints: setup.StartingPoints, territoryMode: true, damageFloor: 1)
                : GameConfig.Default(biomesEnabled: false, startingPoints: setup.StartingPoints, damageFloor: 1);

            if (territory)
            {
                board = board.WithControl(board.DeploymentZone(PlayerId.Player0), PlayerId.Player0);
                board = board.WithControl(board.DeploymentZone(PlayerId.Player1), PlayerId.Player1);
            }

            int nextId = 1;
            var p0 = SeedPlayer(board, PlayerId.Player0, setup.StartingPoints, ref nextId);
            var p1 = SeedPlayer(board, PlayerId.Player1, setup.StartingPoints, ref nextId);
            return new GameState(board, config, new[] { p0, p1 }, PlayerId.Player0, 1, nextId);
        }

        static PlayerState SeedPlayer(Board board, PlayerId id, int startingPoints, ref int nextId)
        {
            var flat = new List<HexCoord>();
            foreach (var c in board.DeploymentZone(id))
                if (board.TileAt(c).Elevation == 0) flat.Add(c);
            flat.Sort((x, y) => x.Q != y.Q ? x.Q - y.Q : x.R - y.R);

            var units = new List<Unit>();
            for (int i = 0; i < Roster.Length && i < flat.Count; i++)
                units.Add(new Unit(nextId++, id, Roster[i], flat[i], 0));
            return new PlayerState(id, startingPoints, null, units, null);
        }
    }
}

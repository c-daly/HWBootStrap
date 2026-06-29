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
        public readonly int ArmySize;   // total units per side
        public readonly int Brutes;     // requested counts of each role; the rest of ArmySize is filled
        public readonly int Strikers;   // with random roles (so all-zero = a fully random army)
        public readonly int Snipers;

        public GameSetup(GameMode mode, int width, int height, int startingPoints, int seed,
                         int armySize = 3, int brutes = 1, int strikers = 1, int snipers = 1)
        {
            Mode = mode; Width = width; Height = height; StartingPoints = startingPoints; Seed = seed;
            ArmySize = armySize; Brutes = brutes; Strikers = strikers; Snipers = snipers;
        }

        public static GameSetup Default => new GameSetup(GameMode.Annihilation, 9, 7, 0, 7);

        public string ToWire() => $"{(int)Mode} {Width} {Height} {StartingPoints} {Seed} {ArmySize} {Brutes} {Strikers} {Snipers}";

        public static GameSetup Parse(string wire)
        {
            var p = wire.Split(' ');
            int G(int i, int def) => i < p.Length
                && int.TryParse(p[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : def;
            return new GameSetup((GameMode)G(0, 0), G(1, 9), G(2, 7), G(3, 0), G(4, 7),
                                 G(5, 3), G(6, 1), G(7, 1), G(8, 1));
        }
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
            // Territory is combat-centric: economy funds the army, it doesn't win on its own. So no Economy
            // win — Annihilation decides, Score breaks a round-cap tie. (Sim validation showed dropping the
            // economy win already makes hoarding a ~96% loss; point decay on top just taxes the active spender,
            // so it's left off — the PointDecay knob stays available for experiments.)
            var config = territory
                ? GameConfig.Default(biomesEnabled: false, winConditions: WinBy.Annihilation | WinBy.Score,
                                     startingPoints: setup.StartingPoints, territoryMode: true, damageFloor: 1)
                : GameConfig.Default(biomesEnabled: false, startingPoints: setup.StartingPoints, damageFloor: 1);

            if (territory)
            {
                board = board.WithControl(board.DeploymentZone(PlayerId.Player0), PlayerId.Player0);
                board = board.WithControl(board.DeploymentZone(PlayerId.Player1), PlayerId.Player1);
            }

            var army = BuildArmy(setup);
            int nextId = 1;
            var p0 = SeedPlayer(board, PlayerId.Player0, setup.StartingPoints, army, ref nextId);
            var p1 = SeedPlayer(board, PlayerId.Player1, setup.StartingPoints, army, ref nextId);
            return new GameState(board, config, new[] { p0, p1 }, PlayerId.Player0, 1, nextId);
        }

        /// <summary>Build a territory game with an explicit ruleset (for balance experiments): same board and
        /// army seeding as <see cref="Build"/>, but the caller supplies the <see cref="GameConfig"/> to vary.</summary>
        public static GameState BuildTerritory(GameConfig config, int width, int height, int seed)
        {
            var board = new RandomBoardGenerator(new BoardGenConfig(width, height, maxElevation: 2)).Generate(seed);
            board = board.WithControl(board.DeploymentZone(PlayerId.Player0), PlayerId.Player0);
            board = board.WithControl(board.DeploymentZone(PlayerId.Player1), PlayerId.Player1);
            var army = BuildArmy(new GameSetup(GameMode.Territory, width, height, config.StartingPoints, seed));
            int nextId = 1;
            var p0 = SeedPlayer(board, PlayerId.Player0, config.StartingPoints, army, ref nextId);
            var p1 = SeedPlayer(board, PlayerId.Player1, config.StartingPoints, army, ref nextId);
            return new GameState(board, config, new[] { p0, p1 }, PlayerId.Player0, 1, nextId);
        }

        // Both sides share one composition (symmetric/fair). Requested role counts come first; any remaining
        // slots up to ArmySize are filled with random roles from a seed-derived RNG — so all-zero counts give
        // a fully random army, partial counts fill the rest randomly.
        static UnitStats[] BuildArmy(GameSetup setup)
        {
            int size = System.Math.Max(1, setup.ArmySize);
            var list = new List<UnitStats>();
            void AddN(UnitStats s, int n) { for (int i = 0; i < n && list.Count < size; i++) list.Add(s); }
            AddN(Roster[0], System.Math.Max(0, setup.Brutes));
            AddN(Roster[1], System.Math.Max(0, setup.Strikers));
            AddN(Roster[2], System.Math.Max(0, setup.Snipers));
            var rng = new System.Random(setup.Seed ^ 0x5A17);
            while (list.Count < size) list.Add(Roster[rng.Next(Roster.Length)]);
            return list.ToArray();
        }

        static PlayerState SeedPlayer(Board board, PlayerId id, int startingPoints, UnitStats[] army, ref int nextId)
        {
            var flat = new List<HexCoord>();
            foreach (var c in board.DeploymentZone(id))
                if (board.TileAt(c).Elevation == 0) flat.Add(c);
            flat.Sort((x, y) => x.Q != y.Q ? x.Q - y.Q : x.R - y.R);

            var units = new List<Unit>();
            for (int i = 0; i < army.Length && i < flat.Count; i++)
                units.Add(new Unit(nextId++, id, army[i], flat[i], 0));
            return new PlayerState(id, startingPoints, null, units, null);
        }
    }
}

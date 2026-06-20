using System;
using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// Board-derived constants and start-state construction shared by the single-agent training env and
    /// the two-agent duel env, so their observation/action encodings can never drift apart (a trained
    /// model must see at duel time exactly what it saw at training time). Cell order is fixed by the
    /// board dimensions, so action/observation sizes are stable across episodes.
    /// </summary>
    public sealed class TacticalLayout
    {
        public readonly List<HexCoord> Cells = new List<HexCoord>();
        public readonly Dictionary<HexCoord, int> CellIndex = new Dictionary<HexCoord, int>();
        public readonly int CellCount;
        public readonly int Roster;
        public readonly BoardGenConfig BoardGen;
        public readonly GameConfig Game;
        public readonly IReadOnlyList<UnitStats> RosterStats;

        public TacticalLayout(EnvConfig cfg)
        {
            BoardGen = cfg.BoardGen;
            Game = cfg.Game;
            RosterStats = cfg.Roster;
            Roster = cfg.Roster.Count;

            for (int row = 0; row < BoardGen.Height; row++)
                for (int col = 0; col < BoardGen.Width; col++)
                {
                    var c = HexLayout.OffsetToAxial(col, row);
                    if (CellIndex.ContainsKey(c)) continue;
                    CellIndex[c] = Cells.Count;
                    Cells.Add(c);
                }
            CellCount = Cells.Count;
        }

        public int ActionCount => 1 + 3 * Roster * CellCount; // EndTurn + (move | attack | deploy) × slot/template × cell
        public int ObservationLength => TacticalCoding.PerCell * CellCount + TacticalCoding.Globals;

        /// <summary>Builds the start state (rosters placed in deployment zones) and each seat's stable
        /// slot→unitId map for the action codec.</summary>
        public (GameState state, int[] slot0, int[] slot1) NewGame(int seed)
        {
            var board = new RandomBoardGenerator(BoardGen).Generate(seed);
            int nextId = 1;
            var u0 = BuildRoster(board, PlayerId.Player0, ref nextId);
            var u1 = BuildRoster(board, PlayerId.Player1, ref nextId);
            // seed each side's barracks with the roster types so they can DEPLOY reinforcements from bounty
            var templates = new List<UnitStats>(RosterStats);
            var p0 = new PlayerState(PlayerId.Player0, 0, templates, u0, null);
            var p1 = new PlayerState(PlayerId.Player1, 0, templates, u1, null);
            var state = new GameState(board, Game, new[] { p0, p1 }, PlayerId.Player0, 1, nextId);
            return (state, SlotMap(u0), SlotMap(u1));
        }

        private int[] SlotMap(IReadOnlyList<Unit> units)
        {
            var map = new int[Roster];
            for (int i = 0; i < Roster; i++) map[i] = i < units.Count ? units[i].Id : -1;
            return map;
        }

        private IReadOnlyList<Unit> BuildRoster(Board board, PlayerId player, ref int nextId)
        {
            var zone = new List<HexCoord>(board.DeploymentZone(player));
            zone.Sort((x, y) => x.Q != y.Q ? x.Q.CompareTo(y.Q) : x.R.CompareTo(y.R));

            var units = new List<Unit>();
            int count = Math.Min(Roster, zone.Count);
            for (int i = 0; i < count; i++)
            {
                var c = zone[i];
                units.Add(new Unit(nextId++, player, RosterStats[i], c, board.TileAt(c).Elevation));
            }
            return units;
        }
    }
}

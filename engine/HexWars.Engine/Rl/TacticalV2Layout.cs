using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>The output of <see cref="TacticalV2Layout.NewGame"/>: the constructed start state, the
    /// one sampled template composition used by both seats (so a caller can label planes/log without
    /// re-deriving it), and each seat's freshly initialized stable-slot registry.</summary>
    public sealed class TacticalV2Start
    {
        public GameState State { get; }
        public int[] TemplateIndices0 { get; }
        public int[] TemplateIndices1 { get; }
        public TacticalV2UnitRegistry Slots0 { get; }
        public TacticalV2UnitRegistry Slots1 { get; }

        public TacticalV2Start(GameState state, int[] templateIndices0, int[] templateIndices1,
            TacticalV2UnitRegistry slots0, TacticalV2UnitRegistry slots1)
        {
            State = state;
            TemplateIndices0 = templateIndices0;
            TemplateIndices1 = templateIndices1;
            Slots0 = slots0;
            Slots1 = slots1;
        }
    }

    /// <summary>
    /// Tactical-v2 board-derived constants and start-state construction. Separates the ordered
    /// TEMPLATE catalog (roles: what a unit can be — addressed by catalog index, used for deploy
    /// actions and observation planes) from the stable UNIT SLOT space (identity: which controllable
    /// unit — assigned by <see cref="TacticalV2UnitRegistry"/>, used for move/attack actions). A unit
    /// never changes slot for its lifetime, so a per-slot policy head tracks the same unit across an
    /// episode even as it takes damage, moves, or is eventually replaced (in its freed slot) by a
    /// freshly deployed reinforcement of a different role.
    /// </summary>
    public sealed class TacticalV2Layout
    {
        private readonly TacticalV2Config _config;

        public BoardGenConfig BoardGen { get; }
        public GameConfig Game { get; }
        public IReadOnlyList<TacticalV2Template> Templates { get; }
        public IReadOnlyList<HexCoord> Cells { get; }
        public IReadOnlyDictionary<HexCoord, int> CellIndex { get; }

        public int TemplateCount { get; }
        public int UnitSlotCount { get; }
        public int CellCount => Cells.Count;

        // Action layout: 1 (EndTurn) + move-by-slot + attack-by-slot + deploy-by-template, each a
        // slot/template-major, cell-minor block, so move/attack address stable unit identity while
        // deploy addresses the (not-yet-instantiated-until-deployed) template catalog.
        public int MoveOffset => 1;
        public int AttackOffset => MoveOffset + UnitSlotCount * CellCount;
        public int DeployOffset => AttackOffset + UnitSlotCount * CellCount;
        public int ActionCount => DeployOffset + TemplateCount * CellCount;

        // Spatial observation: my-template + enemy-template HP planes (2×TemplateCount) plus one
        // elevation plane, channel-major over row-major cells, then a handful of scalar globals.
        public int ObservationChannels => 2 * TemplateCount + 1;
        public int ObservationGlobals => TacticalV2Coding.Globals;
        public int ObservationLength => ObservationChannels * CellCount + ObservationGlobals;

        public TacticalV2Layout(TacticalV2Config config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            if (config.BoardGen == null)
                throw new ArgumentException("tactical-v2 board configuration is required", nameof(config));
            if (config.Templates == null || config.Templates.Count == 0)
                throw new ArgumentException("tactical-v2 template catalog must not be empty", nameof(config));
            if (config.BoardGen.Width <= 0 || config.BoardGen.Height <= 0)
                throw new ArgumentOutOfRangeException(nameof(config), "tactical-v2 board dimensions must be positive");
            if (config.MaxControllableUnits <= 0)
                throw new ArgumentOutOfRangeException(nameof(config), "tactical-v2 max controllable units must be positive");

            _config = config;
            BoardGen = config.BoardGen;
            Game = config.Game;
            Templates = Array.AsReadOnly(config.Templates.ToArray());
            TemplateCount = Templates.Count;
            UnitSlotCount = config.MaxControllableUnits;

            var cells = new List<HexCoord>();
            var cellIndex = new Dictionary<HexCoord, int>();
            for (int row = 0; row < BoardGen.Height; row++)
                for (int col = 0; col < BoardGen.Width; col++)
                {
                    var cell = HexLayout.OffsetToAxial(col, row);
                    if (cellIndex.ContainsKey(cell)) continue;
                    cellIndex[cell] = cells.Count;
                    cells.Add(cell);
                }
            Cells = cells.AsReadOnly();
            CellIndex = new ReadOnlyDictionary<HexCoord, int>(cellIndex);
        }

        /// <summary>The column 180 degrees around the board's center from <paramref name="cell"/> — the
        /// same reflection <see cref="RandomBoardGenerator"/> already uses to keep generated terrain and
        /// elevation symmetric, expressed in offset (col,row) space as (col,row) -> (Width-1-col,
        /// Height-1-row). <paramref name="cell"/> is converted back to (col,row) via the inverse of
        /// <see cref="HexLayout.OffsetToAxial"/> before reflecting.</summary>
        public HexCoord MirrorCell(HexCoord cell)
        {
            int col = cell.Q;
            int row = cell.R + (col - (col & 1)) / 2; // inverse of HexLayout.OffsetToAxial
            return HexLayout.OffsetToAxial(BoardGen.Width - 1 - col, BoardGen.Height - 1 - row);
        }

        /// <summary>Builds a fresh, seeded, symmetric start state: one template composition sampled by
        /// <see cref="TacticalV2Config.SampleStartingArmy"/> and used for both seats, Player 0 placed on
        /// its deterministically sorted deployment zone, and every Player 1 cell derived by mirroring
        /// the matching Player 0 cell — never independently sampled — so the two starting positions are
        /// provably geometric mirrors of one another, not merely distributionally similar.</summary>
        public TacticalV2Start NewGame(int seed)
        {
            var board = new RandomBoardGenerator(BoardGen).Generate(seed);

            var zone0 = SortedZone(board, PlayerId.Player0);
            var zone1 = new HashSet<HexCoord>(board.DeploymentZone(PlayerId.Player1));
            int smallerZone = Math.Min(zone0.Count, zone1.Count);
            if (smallerZone < UnitSlotCount)
                throw new InvalidOperationException(
                    $"tactical-v2 deployment zone too small: {smallerZone} cells available but " +
                    $"{UnitSlotCount} starting units are required");

            IReadOnlyList<TacticalV2Template> sampled = _config.SampleStartingArmy(seed);
            var templateIndices = new int[UnitSlotCount];
            for (int i = 0; i < UnitSlotCount; i++) templateIndices[i] = IndexOfTemplate(sampled[i]);

            var barracks = Templates.Select(template => template.Template).ToList();
            int nextId = 1;
            var units0 = new List<Unit>(UnitSlotCount);
            var units1 = new List<Unit>(UnitSlotCount);
            var usedMirrors = new HashSet<HexCoord>();
            for (int i = 0; i < UnitSlotCount; i++)
            {
                HexCoord cell0 = zone0[i];
                HexCoord cell1 = MirrorCell(cell0);
                if (!zone1.Contains(cell1))
                    throw new InvalidOperationException(
                        $"tactical-v2 mirrored cell {cell1} does not belong to Player 1's deployment zone");
                if (!usedMirrors.Add(cell1))
                    throw new InvalidOperationException(
                        $"tactical-v2 mirrored cell {cell1} collides with another starting unit's cell");

                UnitTemplate template = Templates[templateIndices[i]].Template;
                units0.Add(new Unit(nextId++, PlayerId.Player0, template.Stats, cell0,
                    board.TileAt(cell0).Elevation, template.Name));
                units1.Add(new Unit(nextId++, PlayerId.Player1, template.Stats, cell1,
                    board.TileAt(cell1).Elevation, template.Name));
            }

            var p0 = new PlayerState(PlayerId.Player0, 0, barracks, units0, null);
            var p1 = new PlayerState(PlayerId.Player1, 0, barracks, units1, null);
            var state = new GameState(board, Game, new PlayerState[] { p0, p1 }, PlayerId.Player0, 1, nextId);

            var slots0 = new TacticalV2UnitRegistry(UnitSlotCount);
            slots0.Initialize(units0, templateIndices);
            var slots1 = new TacticalV2UnitRegistry(UnitSlotCount);
            slots1.Initialize(units1, templateIndices);

            return new TacticalV2Start(state, templateIndices, (int[])templateIndices.Clone(), slots0, slots1);
        }

        private int IndexOfTemplate(TacticalV2Template template)
        {
            for (int i = 0; i < Templates.Count; i++)
                if (string.Equals(Templates[i].Id, template.Id, StringComparison.Ordinal)) return i;
            throw new InvalidOperationException(
                $"sampled template '{template.Id}' is not present in the layout's catalog");
        }

        private static List<HexCoord> SortedZone(Board board, PlayerId player)
        {
            var zone = new List<HexCoord>(board.DeploymentZone(player));
            zone.Sort((x, y) => x.Q != y.Q ? x.Q.CompareTo(y.Q) : x.R.CompareTo(y.R));
            return zone;
        }
    }
}

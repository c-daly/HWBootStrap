using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>One rejected deterministic conversion-start construction attempt. Template IDs are
    /// retained so feasibility filtering remains auditable rather than silently preferring easier
    /// compositions.</summary>
    public sealed class TacticalV2ConstructionRejection
    {
        public TacticalV2ConstructionRejection(
            int attempt,
            string reason,
            string[] learnerTemplateIds,
            string[] opponentTemplateIds)
        {
            Attempt = attempt;
            Reason = reason;
            LearnerTemplateIds = learnerTemplateIds;
            OpponentTemplateIds = opponentTemplateIds;
        }

        public int Attempt { get; }
        public string Reason { get; }
        public string[] LearnerTemplateIds { get; }
        public string[] OpponentTemplateIds { get; }
    }

    /// <summary>The output of <see cref="TacticalV2Layout.NewGame"/>: the constructed start state, the
    /// sampled template compositions, each seat's freshly initialized stable-slot registry, and
    /// profiled-start construction diagnostics.</summary>
    public sealed class TacticalV2Start
    {
        public GameState State { get; }
        public int[] TemplateIndices0 { get; }
        public int[] TemplateIndices1 { get; }
        public string[] TemplateIds0 { get; }
        public string[] TemplateIds1 { get; }
        public TacticalV2UnitRegistry Slots0 { get; }
        public TacticalV2UnitRegistry Slots1 { get; }
        public string ProfileId { get; }
        public PlayerId ReferenceSeat { get; }
        public int ConstructionAttempts { get; }
        public IReadOnlyList<TacticalV2ConstructionRejection> ConstructionRejections { get; }

        public TacticalV2Start(GameState state, int[] templateIndices0, int[] templateIndices1,
            TacticalV2UnitRegistry slots0, TacticalV2UnitRegistry slots1)
            : this(state, templateIndices0, templateIndices1,
                Array.Empty<string>(), Array.Empty<string>(), slots0, slots1,
                "standard-3v3", PlayerId.Player0, 1,
                Array.Empty<TacticalV2ConstructionRejection>())
        {
        }

        public TacticalV2Start(
            GameState state,
            int[] templateIndices0,
            int[] templateIndices1,
            string[] templateIds0,
            string[] templateIds1,
            TacticalV2UnitRegistry slots0,
            TacticalV2UnitRegistry slots1,
            string profileId,
            PlayerId referenceSeat,
            int constructionAttempts,
            IReadOnlyList<TacticalV2ConstructionRejection> constructionRejections)
        {
            State = state;
            TemplateIndices0 = templateIndices0;
            TemplateIndices1 = templateIndices1;
            TemplateIds0 = templateIds0;
            TemplateIds1 = templateIds1;
            Slots0 = slots0;
            Slots1 = slots1;
            ProfileId = profileId;
            ReferenceSeat = referenceSeat;
            ConstructionAttempts = constructionAttempts;
            ConstructionRejections = constructionRejections;
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
        private const int MaxConstructionAttempts = 512;
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

            string[] templateIds = templateIndices.Select(index => Templates[index].Id).ToArray();
            return new TacticalV2Start(
                state,
                templateIndices,
                (int[])templateIndices.Clone(),
                templateIds,
                (string[])templateIds.Clone(),
                slots0,
                slots1,
                "standard-3v3",
                PlayerId.Player0,
                1,
                Array.Empty<TacticalV2ConstructionRejection>());
        }

        /// <summary>Build a declared learner-relative profile. The standard profile delegates to the
        /// legacy constructor; conversion profiles independently sample both compositions and place
        /// them through bounded, deterministic feasibility rejection.</summary>
        public TacticalV2Start NewGame(
            int seed,
            TacticalV2StartProfile profile,
            PlayerId learnerSeat)
        {
            if (profile == null) throw new ArgumentNullException(nameof(profile));
            if (learnerSeat != PlayerId.Player0 && learnerSeat != PlayerId.Player1)
                throw new ArgumentOutOfRangeException(nameof(learnerSeat));
            if (!IsDeclaredProfile(profile))
                throw new ArgumentException(
                    $"start profile '{profile.Id}' is not declared by this tactical-v2 layout",
                    nameof(profile));

            if (profile.Id == "standard-3v3")
            {
                TacticalV2Start legacy = NewGame(seed);
                return new TacticalV2Start(
                    legacy.State,
                    legacy.TemplateIndices0,
                    legacy.TemplateIndices1,
                    legacy.TemplateIds0,
                    legacy.TemplateIds1,
                    legacy.Slots0,
                    legacy.Slots1,
                    profile.Id,
                    learnerSeat,
                    1,
                    Array.Empty<TacticalV2ConstructionRejection>());
            }

            var board = new RandomBoardGenerator(BoardGen).Generate(seed);
            var candidates = Cells
                .Where(cell => Game.Terrain(board.TileAt(cell).Terrain).Passable)
                .OrderBy(cell => cell.Q)
                .ThenBy(cell => cell.R)
                .ToArray();
            int requiredCells = profile.LearnerUnitCount + profile.OpponentUnitCount;
            if (candidates.Length < requiredCells)
            {
                throw new InvalidOperationException(
                    $"profile '{profile.Id}' seed {seed} requires {requiredCells} passable cells but " +
                    $"the board has only {candidates.Length}");
            }

            var learnerTemplateRng = new Random(DomainSeed(seed, 0x4C54504Cu));
            var opponentTemplateRng = new Random(DomainSeed(seed, 0x4F54504Cu));
            var placementRng = new Random(DomainSeed(seed, 0x504C4143u));
            var rejections = new List<TacticalV2ConstructionRejection>();

            for (int attempt = 1; attempt <= MaxConstructionAttempts; attempt++)
            {
                int[] learnerIndices = SampleTemplateIndices(profile.LearnerUnitCount, learnerTemplateRng);
                int[] opponentIndices = SampleTemplateIndices(profile.OpponentUnitCount, opponentTemplateRng);
                string[] learnerTemplateIds = learnerIndices.Select(index => Templates[index].Id).ToArray();
                string[] opponentTemplateIds = opponentIndices.Select(index => Templates[index].Id).ToArray();

                HexCoord[] shuffled = (HexCoord[])candidates.Clone();
                Shuffle(shuffled, placementRng);
                HexCoord[] learnerCells = shuffled.Take(profile.LearnerUnitCount).ToArray();
                HexCoord[] opponentCells = shuffled
                    .Skip(profile.LearnerUnitCount)
                    .Take(profile.OpponentUnitCount)
                    .ToArray();

                int closest = ClosestDistance(learnerCells, opponentCells);
                if (!MatchesSeparation(profile.Separation, closest))
                {
                    rejections.Add(new TacticalV2ConstructionRejection(
                        attempt,
                        $"closest opposing distance {closest} is outside '{profile.Separation}'",
                        learnerTemplateIds,
                        opponentTemplateIds));
                    continue;
                }

                TacticalV2Start start = BuildConversionStart(
                    board,
                    profile,
                    learnerSeat,
                    learnerIndices,
                    opponentIndices,
                    learnerCells,
                    opponentCells,
                    learnerTemplateIds,
                    opponentTemplateIds,
                    attempt,
                    rejections);
                if (!HasPlausibleDamagePath(start.State, learnerSeat))
                {
                    rejections.Add(new TacticalV2ConstructionRejection(
                        attempt,
                        "no learner unit has an authoritative path to a positive-damage attack",
                        learnerTemplateIds,
                        opponentTemplateIds));
                    continue;
                }

                return start;
            }

            string summary = string.Join(", ", rejections
                .GroupBy(rejection => rejection.Reason)
                .OrderByDescending(group => group.Count())
                .ThenBy(group => group.Key, StringComparer.Ordinal)
                .Take(5)
                .Select(group => $"{group.Count()}x {group.Key}"));
            throw new InvalidOperationException(
                $"profile '{profile.Id}' seed {seed} exhausted {MaxConstructionAttempts} deterministic " +
                $"construction attempts ({summary})");
        }

        private TacticalV2Start BuildConversionStart(
            Board board,
            TacticalV2StartProfile profile,
            PlayerId learnerSeat,
            int[] learnerIndices,
            int[] opponentIndices,
            HexCoord[] learnerCells,
            HexCoord[] opponentCells,
            string[] learnerTemplateIds,
            string[] opponentTemplateIds,
            int constructionAttempts,
            IReadOnlyList<TacticalV2ConstructionRejection> rejections)
        {
            var barracks = Templates.Select(template => template.Template).ToList();
            int nextId = 1;
            List<Unit> learnerUnits = CreateUnits(
                board, learnerSeat, learnerIndices, learnerCells, ref nextId);
            PlayerId opponentSeat = learnerSeat == PlayerId.Player0
                ? PlayerId.Player1
                : PlayerId.Player0;
            List<Unit> opponentUnits = CreateUnits(
                board, opponentSeat, opponentIndices, opponentCells, ref nextId);

            IReadOnlyList<Unit> units0 = learnerSeat == PlayerId.Player0 ? learnerUnits : opponentUnits;
            IReadOnlyList<Unit> units1 = learnerSeat == PlayerId.Player1 ? learnerUnits : opponentUnits;
            int[] indices0 = learnerSeat == PlayerId.Player0 ? learnerIndices : opponentIndices;
            int[] indices1 = learnerSeat == PlayerId.Player1 ? learnerIndices : opponentIndices;
            string[] ids0 = learnerSeat == PlayerId.Player0 ? learnerTemplateIds : opponentTemplateIds;
            string[] ids1 = learnerSeat == PlayerId.Player1 ? learnerTemplateIds : opponentTemplateIds;

            var p0 = new PlayerState(PlayerId.Player0, 0, barracks, units0, null);
            var p1 = new PlayerState(PlayerId.Player1, 0, barracks, units1, null);
            var state = new GameState(
                board, Game, new PlayerState[] { p0, p1 }, PlayerId.Player0, 1, nextId);

            var slots0 = new TacticalV2UnitRegistry(UnitSlotCount);
            slots0.Initialize(units0, indices0);
            var slots1 = new TacticalV2UnitRegistry(UnitSlotCount);
            slots1.Initialize(units1, indices1);

            return new TacticalV2Start(
                state,
                (int[])indices0.Clone(),
                (int[])indices1.Clone(),
                (string[])ids0.Clone(),
                (string[])ids1.Clone(),
                slots0,
                slots1,
                profile.Id,
                learnerSeat,
                constructionAttempts,
                Array.AsReadOnly(rejections.ToArray()));
        }

        private List<Unit> CreateUnits(
            Board board,
            PlayerId owner,
            IReadOnlyList<int> templateIndices,
            IReadOnlyList<HexCoord> cells,
            ref int nextId)
        {
            var units = new List<Unit>(templateIndices.Count);
            for (int index = 0; index < templateIndices.Count; index++)
            {
                UnitTemplate template = Templates[templateIndices[index]].Template;
                HexCoord cell = cells[index];
                units.Add(new Unit(
                    nextId++, owner, template.Stats, cell,
                    board.TileAt(cell).Elevation, template.Name));
            }
            return units;
        }

        /// <summary>Conservative multi-round reachability using authoritative terrain, targeting,
        /// visibility, line-of-sight, and damage calculations. It rejects only starts where no
        /// learner unit can ever move to a cell from which a positive-damage attack is legal.</summary>
        internal static bool HasPlausibleDamagePath(GameState state, PlayerId learnerSeat)
        {
            IReadOnlyList<Unit> enemies = state.Opponent(learnerSeat).UnitsOnBoard;
            if (enemies.Count == 0) return false;

            foreach (Unit attacker in state.Player(learnerSeat).UnitsOnBoard)
            {
                if (!attacker.IsAlive || attacker.Stats.Damage <= 0) continue;

                var visited = new HashSet<HexCoord> { attacker.Cell };
                var queue = new Queue<HexCoord>();
                queue.Enqueue(attacker.Cell);
                while (queue.Count > 0)
                {
                    HexCoord cell = queue.Dequeue();
                    Unit movedAttacker = attacker.WithCell(cell, state.Board.TileAt(cell).Elevation);
                    GameState movedState = WithMovedUnit(state, learnerSeat, movedAttacker);
                    foreach (Unit target in enemies)
                    {
                        if (!target.IsAlive ||
                            !TargetingService.CanTarget(
                                movedState, movedAttacker, target.Cell, target.Elevation))
                        {
                            continue;
                        }

                        int defense = target.Stats.Defense +
                            state.Config.Terrain(state.Board.TileAt(target.Cell).Terrain).Defense;
                        if (CombatResolver.ComputeDamage(
                                movedAttacker.Stats.Damage,
                                movedAttacker.Elevation,
                                target.Elevation,
                                defense,
                                state.Config) > 0)
                        {
                            return true;
                        }
                    }

                    if (attacker.Stats.Movement <= 0) continue;
                    var neighbors = cell.Neighbors().OrderBy(value => value.Q).ThenBy(value => value.R);
                    foreach (HexCoord next in neighbors)
                    {
                        if (visited.Contains(next) || !state.Board.Contains(next)) continue;
                        TerrainDef terrain = state.Config.Terrain(state.Board.TileAt(next).Terrain);
                        if (!terrain.Passable || terrain.MoveCost > attacker.Stats.Movement) continue;
                        int climb = Math.Max(
                            0,
                            state.Board.TileAt(next).Elevation - state.Board.TileAt(cell).Elevation);
                        if (climb > attacker.Stats.VerticalMovement) continue;
                        if (IsOccupiedByOtherUnit(state, next, attacker.Id)) continue;
                        visited.Add(next);
                        queue.Enqueue(next);
                    }
                }
            }

            return false;
        }

        private static GameState WithMovedUnit(
            GameState state,
            PlayerId learnerSeat,
            Unit movedAttacker)
        {
            PlayerState learner = state.Player(learnerSeat);
            Unit[] movedUnits = learner.UnitsOnBoard
                .Select(unit => unit.Id == movedAttacker.Id ? movedAttacker : unit)
                .ToArray();
            var movedPlayer = new PlayerState(
                learner.Id,
                learner.Points,
                learner.Barracks,
                movedUnits,
                learner.Generators,
                learner.DestroyedValue);
            PlayerState[] players = state.Players.ToArray();
            players[(int)learnerSeat] = movedPlayer;
            return new GameState(
                state.Board,
                state.Config,
                players,
                state.ActivePlayer,
                state.Round,
                state.NextEntityId,
                state.IsGameOver,
                state.Winner,
                state.MovedUnitIds,
                state.AttackedUnitIds,
                state.MovementSpent);
        }

        private static bool IsOccupiedByOtherUnit(GameState state, HexCoord cell, int movingUnitId)
        {
            foreach (PlayerState player in state.Players)
                foreach (Unit unit in player.UnitsOnBoard)
                    if (unit.IsAlive && unit.Id != movingUnitId && unit.Cell == cell) return true;
            return false;
        }

        private int[] SampleTemplateIndices(int count, Random rng)
        {
            var result = new int[count];
            for (int index = 0; index < count; index++) result[index] = rng.Next(Templates.Count);
            return result;
        }

        private bool IsDeclaredProfile(TacticalV2StartProfile requested)
        {
            if (_config.StartProfiles == null) return false;
            return _config.StartProfiles.Any(profile =>
                profile.Id == requested.Id &&
                profile.LearnerUnitCount == requested.LearnerUnitCount &&
                profile.OpponentUnitCount == requested.OpponentUnitCount &&
                profile.Separation == requested.Separation);
        }

        private static void Shuffle<T>(T[] values, Random rng)
        {
            for (int index = values.Length - 1; index > 0; index--)
            {
                int swapIndex = rng.Next(index + 1);
                T value = values[index];
                values[index] = values[swapIndex];
                values[swapIndex] = value;
            }
        }

        private static int ClosestDistance(
            IReadOnlyList<HexCoord> learnerCells,
            IReadOnlyList<HexCoord> opponentCells)
        {
            int closest = int.MaxValue;
            foreach (HexCoord learnerCell in learnerCells)
                foreach (HexCoord opponentCell in opponentCells)
                    closest = Math.Min(closest, HexCoord.Distance(learnerCell, opponentCell));
            return closest;
        }

        private static bool MatchesSeparation(string separation, int closest) => separation switch
        {
            TacticalV2StartSeparations.Near => closest >= 2 && closest <= 3,
            TacticalV2StartSeparations.Medium => closest >= 4 && closest <= 6,
            TacticalV2StartSeparations.Far => closest >= 7,
            _ => false,
        };

        private static int DomainSeed(int seed, uint domain)
        {
            uint value = unchecked((uint)seed) ^ domain;
            value += 0x9E3779B9u;
            value = (value ^ (value >> 16)) * 0x85EBCA6Bu;
            value = (value ^ (value >> 13)) * 0xC2B2AE35u;
            return unchecked((int)(value ^ (value >> 16)));
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

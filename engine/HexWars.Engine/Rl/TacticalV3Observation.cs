using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;

namespace HexWars.Engine.Rl
{
    public enum TacticalV3RelativeOwner
    {
        Self = 0,
        Opponent = 1,
    }

    public enum TacticalV3RuleKind
    {
        WinConditions = 0,
        Round = 1,
        RoundCap = 2,
        ActionsPerTurn = 3,
        StartingPoints = 4,
        SelfPoints = 5,
        OpponentPoints = 6,
        DamageFloor = 7,
        DamageHighGroundBonus = 8,
        RangeHighGroundBonus = 9,
        BountyRate = 10,
        DeployCostMultiplier = 11,
        FogOfWar = 12,
        MaxDesignPointCost = 13,
        DesignFee = 14,
    }

    public enum TacticalV3RelationKind
    {
        Neighbor = 0,
        Occupies = 1,
        HasCapability = 2,
    }

    public sealed class TacticalV3CellToken
    {
        public TacticalV3CellToken(int q, int r, TerrainType terrain, int elevation,
            bool selfDeploymentZone, bool opponentDeploymentZone, TacticalV3RelativeOwner? controller,
            bool isBoundary, bool currentlyVisible, bool previouslyObserved)
        {
            Q = q;
            R = r;
            Terrain = terrain;
            Elevation = elevation;
            SelfDeploymentZone = selfDeploymentZone;
            OpponentDeploymentZone = opponentDeploymentZone;
            Controller = controller;
            IsBoundary = isBoundary;
            CurrentlyVisible = currentlyVisible;
            PreviouslyObserved = previouslyObserved;
        }

        public int Q { get; }
        public int R { get; }
        public TerrainType Terrain { get; }
        public int Elevation { get; }
        public bool SelfDeploymentZone { get; }
        public bool OpponentDeploymentZone { get; }
        public TacticalV3RelativeOwner? Controller { get; }
        public bool IsBoundary { get; }
        public bool CurrentlyVisible { get; }
        public bool PreviouslyObserved { get; }
    }

    public sealed class TacticalV3UnitToken
    {
        public TacticalV3UnitToken(TacticalV3RelativeOwner owner, int currentHp, int maxHp,
            TacticalV3TokenRef cell, int elevation, bool moved, bool attacked,
            int horizontalMovementSpent, int verticalMovementSpent, int pointCost, int deployCost,
            bool currentlyVisible)
        {
            Owner = owner;
            CurrentHp = currentHp;
            MaxHp = maxHp;
            Cell = cell;
            Elevation = elevation;
            Moved = moved;
            Attacked = attacked;
            HorizontalMovementSpent = horizontalMovementSpent;
            VerticalMovementSpent = verticalMovementSpent;
            PointCost = pointCost;
            DeployCost = deployCost;
            CurrentlyVisible = currentlyVisible;
        }

        public TacticalV3RelativeOwner Owner { get; }
        public int CurrentHp { get; }
        public int MaxHp { get; }
        public TacticalV3TokenRef Cell { get; }
        public int Elevation { get; }
        public bool Moved { get; }
        public bool Attacked { get; }
        public int HorizontalMovementSpent { get; }
        public int VerticalMovementSpent { get; }
        public int PointCost { get; }
        public int DeployCost { get; }
        public bool CurrentlyVisible { get; }
    }

    public sealed class TacticalV3TemplateToken
    {
        public TacticalV3TemplateToken(TacticalV3RelativeOwner owner, int pointCost, int deployCost,
            bool isFixed, bool isDeployable)
        {
            Owner = owner;
            PointCost = pointCost;
            DeployCost = deployCost;
            IsFixed = isFixed;
            IsDeployable = isDeployable;
        }

        public TacticalV3RelativeOwner Owner { get; }
        public int PointCost { get; }
        public int DeployCost { get; }
        public bool IsFixed { get; }
        public bool IsDeployable { get; }
    }

    public sealed class TacticalV3CapabilityAllocationToken
    {
        public TacticalV3CapabilityAllocationToken(TacticalV3TokenRef owner,
            TacticalV3TokenRef definition, TacticalV3CapabilityKind capability,
            int purchasedLevel, int effectiveValue)
        {
            Owner = owner;
            Definition = definition;
            Capability = capability;
            PurchasedLevel = purchasedLevel;
            EffectiveValue = effectiveValue;
        }

        public TacticalV3TokenRef Owner { get; }
        public TacticalV3TokenRef Definition { get; }
        public TacticalV3CapabilityKind Capability { get; }
        public int PurchasedLevel { get; }
        public int EffectiveValue { get; }
    }

    public sealed class TacticalV3RuleToken
    {
        public TacticalV3RuleToken(TacticalV3RuleKind kind, int intValue = 0,
            float floatValue = 0f, bool boolValue = false)
        {
            Kind = kind;
            IntValue = intValue;
            FloatValue = floatValue;
            BoolValue = boolValue;
        }

        public TacticalV3RuleKind Kind { get; }
        public int IntValue { get; }
        public float FloatValue { get; }
        public bool BoolValue { get; }
    }

    public sealed class TacticalV3MemoryToken
    {
        public TacticalV3MemoryToken(TacticalV3TokenRef cell, int lastSeenRound, int observationAge,
            int lastKnownCurrentHp, bool currentlyVisible)
        {
            Cell = cell;
            LastSeenRound = lastSeenRound;
            ObservationAge = observationAge;
            LastKnownCurrentHp = lastKnownCurrentHp;
            CurrentlyVisible = currentlyVisible;
        }

        public TacticalV3TokenRef Cell { get; }
        public int LastSeenRound { get; }
        public int ObservationAge { get; }
        public int LastKnownCurrentHp { get; }
        public bool CurrentlyVisible { get; }
    }

    public sealed class TacticalV3RelationToken
    {
        public TacticalV3RelationToken(TacticalV3RelationKind kind, TacticalV3TokenRef source,
            TacticalV3TokenRef target, int intFeature = 0, float floatFeature = 0f,
            bool boolFeature = false)
        {
            Kind = kind;
            Source = source;
            Target = target;
            IntFeature = intFeature;
            FloatFeature = floatFeature;
            BoolFeature = boolFeature;
        }

        public TacticalV3RelationKind Kind { get; }
        public TacticalV3TokenRef Source { get; }
        public TacticalV3TokenRef Target { get; }
        public int IntFeature { get; }
        public float FloatFeature { get; }
        public bool BoolFeature { get; }
    }

    public sealed class TacticalV3Observation
    {
        public TacticalV3Observation(IEnumerable<TacticalV3CellToken> cells,
            IEnumerable<TacticalV3UnitToken> units, IEnumerable<TacticalV3TemplateToken> templates,
            IEnumerable<TacticalV3CapabilityDefinition> capabilityDefinitions,
            IEnumerable<TacticalV3CapabilityAllocationToken> capabilityAllocations,
            IEnumerable<TacticalV3RuleToken> rules, IEnumerable<TacticalV3MemoryToken> memory,
            IEnumerable<TacticalV3RelationToken> relations)
        {
            Cells = Snapshot(cells);
            Units = Snapshot(units);
            Templates = Snapshot(templates);
            CapabilityDefinitions = Snapshot(capabilityDefinitions);
            CapabilityAllocations = Snapshot(capabilityAllocations);
            Rules = Snapshot(rules);
            Memory = Snapshot(memory);
            Relations = Snapshot(relations);
        }

        public IReadOnlyList<TacticalV3CellToken> Cells { get; }
        public IReadOnlyList<TacticalV3UnitToken> Units { get; }
        public IReadOnlyList<TacticalV3TemplateToken> Templates { get; }
        public IReadOnlyList<TacticalV3CapabilityDefinition> CapabilityDefinitions { get; }
        public IReadOnlyList<TacticalV3CapabilityAllocationToken> CapabilityAllocations { get; }
        public IReadOnlyList<TacticalV3RuleToken> Rules { get; }
        public IReadOnlyList<TacticalV3MemoryToken> Memory { get; }
        public IReadOnlyList<TacticalV3RelationToken> Relations { get; }

        private static IReadOnlyList<T> Snapshot<T>(IEnumerable<T> source) =>
            Array.AsReadOnly((source ?? throw new ArgumentNullException(nameof(source))).ToArray());
    }

    public interface IObservationMemory
    {
        IReadOnlyList<TacticalV3MemoryToken> Snapshot(PlayerId seat);
    }

    public sealed class EmptyObservationMemory : IObservationMemory
    {
        private static readonly IReadOnlyList<TacticalV3MemoryToken> Empty =
            Array.AsReadOnly(Array.Empty<TacticalV3MemoryToken>());

        private EmptyObservationMemory() { }

        public static EmptyObservationMemory Instance { get; } = new EmptyObservationMemory();

        public IReadOnlyList<TacticalV3MemoryToken> Snapshot(PlayerId seat) => Empty;
    }

    public interface ISeatObservationSource
    {
        TacticalV3Observation Observe(GameState state, PlayerId seat, IObservationMemory memory);
    }

    public sealed class TacticalV3SeatObservationSource : ISeatObservationSource
    {
        private readonly TacticalV3Config _config;
        private readonly TacticalV2Layout _layout;

        public TacticalV3SeatObservationSource(TacticalV3Config config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            IReadOnlyList<string> errors = _config.Validate();
            if (errors.Count != 0)
                throw new ArgumentException("invalid tactical-v3 configuration: " + string.Join("; ", errors),
                    nameof(config));
            _layout = new TacticalV2Layout(_config.Match);
        }

        public TacticalV3Observation Observe(GameState state, PlayerId seat, IObservationMemory memory)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (memory == null) throw new ArgumentNullException(nameof(memory));
            if (seat != PlayerId.Player0 && seat != PlayerId.Player1)
                throw new ArgumentOutOfRangeException(nameof(seat));

            PlayerId opponent = state.Opponent(seat).Id;
            var cellRows = new List<TacticalV3CellToken>(_layout.Cells.Count);
            var cellIndex = new Dictionary<HexCoord, int>();
            foreach (HexCoord cell in _layout.Cells)
            {
                Tile tile = state.Board.TileAt(cell);
                cellIndex.Add(cell, cellRows.Count);
                PlayerId? controller = state.Board.Controller(cell);
                cellRows.Add(new TacticalV3CellToken(cell.Q, cell.R, tile.Terrain, tile.Elevation,
                    state.Board.IsInDeploymentZone(seat, cell), state.Board.IsInDeploymentZone(opponent, cell),
                    controller.HasValue ? Relative(controller.Value, seat) : (TacticalV3RelativeOwner?)null,
                    cell.Neighbors().Any(neighbor => !state.Board.Contains(neighbor)), true, false));
            }

            var unitRows = new List<TacticalV3UnitToken>();
            var unitStats = new List<UnitStats>();
            AddUnits(state.Player(seat), seat, cellIndex, state, unitRows, unitStats);
            AddUnits(state.Player(opponent), seat, cellIndex, state, unitRows, unitStats);

            var templateRows = new List<TacticalV3TemplateToken>();
            var templateStats = new List<UnitStats>();
            AddTemplates(state.Player(seat), seat, state.Config, templateRows, templateStats);
            AddTemplates(state.Player(opponent), seat, state.Config, templateRows, templateStats);

            IReadOnlyList<TacticalV3CapabilityDefinition> definitions = TacticalV3Capabilities.All;
            var allocations = new List<TacticalV3CapabilityAllocationToken>();
            for (int row = 0; row < unitRows.Count; row++)
                AddAllocations(new TacticalV3TokenRef(TacticalV3TableKind.Units, row), unitStats[row], allocations);
            for (int row = 0; row < templateRows.Count; row++)
                AddAllocations(new TacticalV3TokenRef(TacticalV3TableKind.Templates, row), templateStats[row], allocations);

            var rules = Rules(state, seat);
            IReadOnlyList<TacticalV3MemoryToken> memoryRows = memory.Snapshot(seat) ??
                throw new InvalidOperationException("observation memory returned null");
            var relations = Relations(cellIndex, unitRows, allocations);
            EnsureCapacity(cellRows, unitRows, templateRows, definitions, allocations, rules, memoryRows, relations);
            return new TacticalV3Observation(cellRows, unitRows, templateRows, definitions, allocations, rules,
                memoryRows, relations);
        }

        private static void AddUnits(PlayerState player, PlayerId seat, IReadOnlyDictionary<HexCoord, int> cellIndex,
            GameState state, ICollection<TacticalV3UnitToken> rows, ICollection<UnitStats> stats)
        {
            foreach (Unit unit in player.UnitsOnBoard.Where(unit => unit.IsAlive).OrderBy(unit => unit.Id))
            {
                (int horizontal, int vertical) spent = state.MovementSpent.TryGetValue(unit.Id, out var value)
                    ? value : (0, 0);
                rows.Add(new TacticalV3UnitToken(Relative(unit.Owner, seat), unit.CurrentHp, unit.Stats.Health,
                    new TacticalV3TokenRef(TacticalV3TableKind.Cells, cellIndex[unit.Cell]), unit.Elevation,
                    state.MovedUnitIds.Contains(unit.Id), state.AttackedUnitIds.Contains(unit.Id), spent.horizontal,
                    spent.vertical, unit.Stats.PointCost, Economy.DeployCost(unit.Stats, state.Config), true));
                stats.Add(unit.Stats);
            }
        }

        private static void AddTemplates(PlayerState player, PlayerId seat, GameConfig game,
            ICollection<TacticalV3TemplateToken> rows, ICollection<UnitStats> stats)
        {
            for (int index = 0; index < player.Barracks.Count; index++)
            {
                UnitTemplate template = player.Barracks[index];
                int deployCost = Economy.DeployCost(template.Stats, game);
                rows.Add(new TacticalV3TemplateToken(Relative(player.Id, seat), template.Stats.PointCost,
                    deployCost, index < game.FixedTemplateCount, player.Points >= deployCost));
                stats.Add(template.Stats);
            }
        }

        private static void AddAllocations(TacticalV3TokenRef owner, UnitStats stats,
            ICollection<TacticalV3CapabilityAllocationToken> rows)
        {
            foreach (TacticalV3CapabilityDefinition definition in TacticalV3Capabilities.All)
            {
                int value = CapabilityValue(definition.Kind, stats);
                rows.Add(new TacticalV3CapabilityAllocationToken(owner,
                    new TacticalV3TokenRef(TacticalV3TableKind.CapabilityDefinitions, (int)definition.Kind),
                    definition.Kind, value, value));
            }
        }

        private static List<TacticalV3RuleToken> Rules(GameState state, PlayerId seat)
        {
            GameConfig game = state.Config;
            return new List<TacticalV3RuleToken>
            {
                new TacticalV3RuleToken(TacticalV3RuleKind.WinConditions, (int)game.WinConditions),
                new TacticalV3RuleToken(TacticalV3RuleKind.Round, state.Round),
                new TacticalV3RuleToken(TacticalV3RuleKind.RoundCap, game.RoundCap),
                new TacticalV3RuleToken(TacticalV3RuleKind.ActionsPerTurn, game.TurnPolicy.ActionsPerTurn ?? -1),
                new TacticalV3RuleToken(TacticalV3RuleKind.StartingPoints, game.StartingPoints),
                new TacticalV3RuleToken(TacticalV3RuleKind.SelfPoints, state.Player(seat).Points),
                new TacticalV3RuleToken(TacticalV3RuleKind.OpponentPoints, state.Opponent(seat).Points),
                new TacticalV3RuleToken(TacticalV3RuleKind.DamageFloor, game.DamageFloor),
                new TacticalV3RuleToken(TacticalV3RuleKind.DamageHighGroundBonus, game.DmgHighGroundBonus),
                new TacticalV3RuleToken(TacticalV3RuleKind.RangeHighGroundBonus, game.RangeHighGroundBonus),
                new TacticalV3RuleToken(TacticalV3RuleKind.BountyRate, floatValue: (float)game.BountyRate),
                new TacticalV3RuleToken(TacticalV3RuleKind.DeployCostMultiplier,
                    floatValue: (float)game.DeployCostMultiplier),
                new TacticalV3RuleToken(TacticalV3RuleKind.FogOfWar, boolValue: game.FogOfWar),
                new TacticalV3RuleToken(TacticalV3RuleKind.MaxDesignPointCost, game.MaxDesignPointCost),
                new TacticalV3RuleToken(TacticalV3RuleKind.DesignFee, game.DesignFee),
            };
        }

        private List<TacticalV3RelationToken> Relations(IReadOnlyDictionary<HexCoord, int> cellIndex,
            IReadOnlyList<TacticalV3UnitToken> units, IReadOnlyList<TacticalV3CapabilityAllocationToken> allocations)
        {
            var rows = new List<TacticalV3RelationToken>();
            foreach (KeyValuePair<HexCoord, int> item in cellIndex)
                foreach (HexCoord neighbor in item.Key.Neighbors())
                    if (cellIndex.TryGetValue(neighbor, out int target))
                        rows.Add(new TacticalV3RelationToken(TacticalV3RelationKind.Neighbor,
                            new TacticalV3TokenRef(TacticalV3TableKind.Cells, item.Value),
                            new TacticalV3TokenRef(TacticalV3TableKind.Cells, target)));
            for (int row = 0; row < units.Count; row++)
                rows.Add(new TacticalV3RelationToken(TacticalV3RelationKind.Occupies,
                    new TacticalV3TokenRef(TacticalV3TableKind.Units, row), units[row].Cell));
            foreach (TacticalV3CapabilityAllocationToken allocation in allocations)
                rows.Add(new TacticalV3RelationToken(TacticalV3RelationKind.HasCapability, allocation.Owner,
                    allocation.Definition, allocation.EffectiveValue));
            rows.Sort(CompareRelations);
            return rows;
        }

        private void EnsureCapacity<TCells, TUnits, TTemplates, TDefinitions, TAllocations, TRules, TMemory, TRelations>(
            IReadOnlyCollection<TCells> cells, IReadOnlyCollection<TUnits> units, IReadOnlyCollection<TTemplates> templates,
            IReadOnlyCollection<TDefinitions> definitions, IReadOnlyCollection<TAllocations> allocations,
            IReadOnlyCollection<TRules> rules, IReadOnlyCollection<TMemory> memory, IReadOnlyCollection<TRelations> relations)
        {
            TacticalV3CapacityProfile capacity = _config.Capacity;
            if (cells.Count > capacity.MaxCells || units.Count > capacity.MaxUnits ||
                templates.Count > capacity.MaxTemplates || definitions.Count > capacity.MaxCapabilityDefinitions ||
                allocations.Count > capacity.MaxCapabilityAllocations || rules.Count > capacity.MaxRules ||
                memory.Count > capacity.MaxMemoryRecords || relations.Count > capacity.MaxRelations)
            {
                throw new InvalidOperationException("tactical-v3 observation exceeds configured capacity");
            }
        }

        private static int CapabilityValue(TacticalV3CapabilityKind kind, UnitStats stats) => kind switch
        {
            TacticalV3CapabilityKind.Health => stats.Health,
            TacticalV3CapabilityKind.Damage => stats.Damage,
            TacticalV3CapabilityKind.Defense => stats.Defense,
            TacticalV3CapabilityKind.Movement => stats.Movement,
            TacticalV3CapabilityKind.VerticalMovement => stats.VerticalMovement,
            TacticalV3CapabilityKind.Range => stats.Range,
            TacticalV3CapabilityKind.RangeArc => stats.RangeArc,
            TacticalV3CapabilityKind.Vision => stats.Vision,
            TacticalV3CapabilityKind.VisionArc => stats.VisionArc,
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static TacticalV3RelativeOwner Relative(PlayerId owner, PlayerId seat) =>
            owner == seat ? TacticalV3RelativeOwner.Self : TacticalV3RelativeOwner.Opponent;

        private static int CompareRelations(TacticalV3RelationToken left, TacticalV3RelationToken right)
        {
            int result = left.Kind.CompareTo(right.Kind);
            if (result != 0) return result;
            result = left.Source.Table.CompareTo(right.Source.Table);
            if (result != 0) return result;
            result = left.Source.Row.CompareTo(right.Source.Row);
            if (result != 0) return result;
            result = left.Target.Table.CompareTo(right.Target.Table);
            if (result != 0) return result;
            result = left.Target.Row.CompareTo(right.Target.Row);
            if (result != 0) return result;
            result = left.IntFeature.CompareTo(right.IntFeature);
            if (result != 0) return result;
            result = left.FloatFeature.CompareTo(right.FloatFeature);
            return result != 0 ? result : left.BoolFeature.CompareTo(right.BoolFeature);
        }
    }
}

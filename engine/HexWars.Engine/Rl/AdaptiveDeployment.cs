using System;
using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine.Rl
{
    public readonly struct DeploymentPlacement : IEquatable<DeploymentPlacement>
    {
        public int Slot { get; }
        public int TemplateIndex { get; }
        public HexCoord Cell { get; }

        public DeploymentPlacement(int slot, int templateIndex, HexCoord cell)
        {
            Slot = slot;
            TemplateIndex = templateIndex;
            Cell = cell;
        }

        public bool Equals(DeploymentPlacement other) =>
            Slot == other.Slot && TemplateIndex == other.TemplateIndex && Cell == other.Cell;

        public override bool Equals(object? obj) => obj is DeploymentPlacement other && Equals(other);
        public override int GetHashCode() => unchecked(((Slot * 397) ^ TemplateIndex) * 397 ^ Cell.GetHashCode());
        public static bool operator ==(DeploymentPlacement left, DeploymentPlacement right) => left.Equals(right);
        public static bool operator !=(DeploymentPlacement left, DeploymentPlacement right) => !left.Equals(right);
    }

    public interface IDeploymentPolicy
    {
        IReadOnlyList<DeploymentPlacement> Choose(AdaptiveDeploymentView view);
    }

    /// <summary>
    /// A snapshot containing only public setup data and one seat's private ledger. There is deliberately
    /// no route from a view to the opponent ledger.
    /// </summary>
    public sealed class AdaptiveDeploymentView
    {
        public PlayerId Seat { get; }
        public Board Board { get; }
        public GameConfig Game { get; }
        public IReadOnlyList<UnitTemplate> Templates { get; }
        public IReadOnlyList<DeploymentPlacement> OwnPlacements { get; }
        public int RemainingBudget { get; }
        public int RequiredUnits { get; }

        internal AdaptiveDeploymentView(PlayerId seat, Board board, GameConfig game,
            IReadOnlyList<UnitTemplate> templates, IReadOnlyList<DeploymentPlacement> ownPlacements,
            int remainingBudget, int requiredUnits)
        {
            Seat = seat;
            Board = AdaptiveDeployment.CopyBoard(board);
            Game = game;
            Templates = templates.ToArray();
            OwnPlacements = ownPlacements.OrderBy(p => p.Slot).ToArray();
            RemainingBudget = remainingBudget;
            RequiredUnits = requiredUnits;
        }
    }

    /// <summary>
    /// Owns two isolated pregame ledgers. Nothing enters GameState until both seats confirm and Reveal
    /// constructs the complete round-one state in one operation.
    /// </summary>
    public sealed class AdaptiveDeployment
    {
        private readonly Board _board;
        private readonly GameConfig _game;
        private readonly int _requiredUnits;
        private readonly int _startingBudget;
        private readonly List<DeploymentPlacement>[] _placements;
        private readonly List<UnitTemplate>[] _barracks;
        private readonly bool[] _confirmed = new bool[2];

        public Board Board => CopyBoard(_board);
        public GameConfig Game => _game;
        public bool IsRevealed => _confirmed[0] && _confirmed[1];

        public AdaptiveDeployment(Board board, AdaptiveEnvConfig? config = null)
        {
            if (board == null) throw new ArgumentNullException(nameof(board));
            var source = config ?? AdaptiveEnvConfig.Default();
            _board = CopyBoard(board);
            _game = CopyGameConfig(source.Game
                ?? throw new ArgumentException("adaptive deployment requires game rules", nameof(config)));
            _requiredUnits = source.StartingUnitCount;
            _startingBudget = source.StartingArmyBudget;
            var templates = source.Templates?.ToArray()
                ?? throw new ArgumentException("adaptive deployment requires templates", nameof(config));
            ValidateConfiguration(templates, source.FixedTemplateCount, source.CustomTemplateCount);

            _placements = new[] { new List<DeploymentPlacement>(), new List<DeploymentPlacement>() };
            _barracks = new[]
            {
                new List<UnitTemplate>(templates),
                new List<UnitTemplate>(templates),
            };
        }

        public IReadOnlyList<DeploymentPlacement> Placements(PlayerId seat)
        {
            int index = SeatIndex(seat);
            return index < 0
                ? Array.Empty<DeploymentPlacement>()
                : _placements[index].OrderBy(p => p.Slot).ToArray();
        }

        public AdaptiveDeploymentView View(PlayerId seat)
        {
            int index = RequiredSeatIndex(seat);
            return new AdaptiveDeploymentView(seat, _board, _game, _barracks[index], _placements[index],
                RemainingBudget(index), _requiredUnits);
        }

        public bool TryPlace(PlayerId seat, int templateIndex, HexCoord cell)
        {
            int index = SeatIndex(seat);
            if (index < 0 || _confirmed[index] || !ValidTemplate(index, templateIndex)) return false;
            if (_placements[index].Count >= _requiredUnits || !ValidCell(seat, index, cell, -1))
                return false;

            int cost = _barracks[index][templateIndex].Stats.PointCost;
            if (cost > RemainingBudget(index)) return false;

            int slot = LowestFreeSlot(index);
            if (slot < 0) return false;
            _placements[index].Add(new DeploymentPlacement(slot, templateIndex, cell));
            return true;
        }

        public bool TryMove(PlayerId seat, int placementSlot, HexCoord cell)
        {
            int index = SeatIndex(seat);
            if (index < 0 || _confirmed[index]) return false;
            int at = _placements[index].FindIndex(p => p.Slot == placementSlot);
            if (at < 0 || !ValidCell(seat, index, cell, placementSlot)) return false;

            var old = _placements[index][at];
            _placements[index][at] = new DeploymentPlacement(old.Slot, old.TemplateIndex, cell);
            return true;
        }

        public bool TryRemove(PlayerId seat, int placementSlot)
        {
            int index = SeatIndex(seat);
            if (index < 0 || _confirmed[index]) return false;
            int at = _placements[index].FindIndex(p => p.Slot == placementSlot);
            if (at < 0) return false;
            _placements[index].RemoveAt(at);
            return true;
        }

        public bool CanConfirm(PlayerId seat)
        {
            int index = SeatIndex(seat);
            if (index < 0 || _confirmed[index] || _placements[index].Count != _requiredUnits)
                return false;
            if (RemainingBudget(index) < 0) return false;

            var slots = new HashSet<int>();
            var cells = new HashSet<HexCoord>();
            foreach (var placement in _placements[index])
            {
                if (!slots.Add(placement.Slot) || !cells.Add(placement.Cell)) return false;
                if (placement.Slot < 0 || placement.Slot >= _requiredUnits) return false;
                if (!ValidTemplate(index, placement.TemplateIndex)) return false;
                if (!ValidCell(seat, index, placement.Cell, placement.Slot)) return false;
            }
            return true;
        }

        public bool TryConfirm(PlayerId seat)
        {
            int index = SeatIndex(seat);
            if (index < 0 || !CanConfirm(seat)) return false;
            _confirmed[index] = true;
            return true;
        }

        public bool Confirmed(PlayerId seat)
        {
            int index = SeatIndex(seat);
            return index >= 0 && _confirmed[index];
        }

        public GameState Reveal(PlayerId firstPlayer)
        {
            RequiredSeatIndex(firstPlayer);
            if (!IsRevealed)
                throw new InvalidOperationException("both seats must confirm before deployment can be revealed");

            int nextEntityId = 1;
            var units0 = BuildUnits(PlayerId.Player0, 0, ref nextEntityId);
            var units1 = BuildUnits(PlayerId.Player1, 1, ref nextEntityId);
            var player0 = new PlayerState(PlayerId.Player0, Game.StartingPoints,
                new List<UnitTemplate>(_barracks[0]), units0, null);
            var player1 = new PlayerState(PlayerId.Player1, Game.StartingPoints,
                new List<UnitTemplate>(_barracks[1]), units1, null);
            return new GameState(CopyBoard(_board), _game, new[] { player0, player1 }, firstPlayer, round: 1,
                nextEntityId: nextEntityId);
        }

        private IReadOnlyList<Unit> BuildUnits(PlayerId seat, int index, ref int nextEntityId)
        {
            var units = new List<Unit>(_requiredUnits);
            foreach (var placement in _placements[index].OrderBy(p => p.Slot))
            {
                var template = _barracks[index][placement.TemplateIndex];
                var tile = _board.TileAt(placement.Cell);
                units.Add(new Unit(nextEntityId++, seat, template.Stats, placement.Cell, tile.Elevation,
                    template.Name));
            }
            return units;
        }

        private bool ValidTemplate(int index, int templateIndex) =>
            templateIndex >= 0 && templateIndex < _barracks[index].Count;

        private bool ValidCell(PlayerId seat, int index, HexCoord cell, int ignoredSlot)
        {
            if (!_board.Contains(cell) || !_board.IsInDeploymentZone(seat, cell)) return false;
            if (!_game.Terrain(_board.TileAt(cell).Terrain).Passable) return false;
            return !_placements[index].Any(p => p.Slot != ignoredSlot && p.Cell == cell);
        }

        private int RemainingBudget(int index) => _startingBudget
            - _placements[index].Sum(p => _barracks[index][p.TemplateIndex].Stats.PointCost);

        private int LowestFreeSlot(int index)
        {
            for (int slot = 0; slot < _requiredUnits; slot++)
                if (_placements[index].All(p => p.Slot != slot)) return slot;
            return -1;
        }

        private void ValidateConfiguration(IReadOnlyList<UnitTemplate> templates, int fixedTemplateCount,
            int customTemplateCount)
        {
            var errors = new List<string>();
            if (fixedTemplateCount != 6) errors.Add("adaptive deployment requires exactly 6 fixed templates");
            if (customTemplateCount != 3) errors.Add("adaptive deployment requires exactly 3 custom templates");
            if (templates.Count != 9) errors.Add("adaptive deployment requires exactly 9 templates");
            if (_requiredUnits <= 0) errors.Add("starting deployment must require at least one unit");
            if (_startingBudget < 0) errors.Add("starting deployment budget cannot be negative");

            int cheapest = templates.Count == 0 ? 0 : templates.Min(t => t.Stats.PointCost);
            if (_startingBudget < cheapest * _requiredUnits)
                errors.Add($"starting deployment requires at least {cheapest * _requiredUnits} points but only {_startingBudget} are available");

            int pinnedCount = Math.Min(templates.Count, AdaptiveContractData.Templates.Count);
            for (int slot = 0; slot < pinnedCount; slot++)
            {
                var actual = templates[slot];
                var pinned = AdaptiveContractData.Templates[slot];
                if (!string.Equals(actual.Name, pinned.Name, StringComparison.Ordinal))
                    errors.Add($"template slot {slot} must remain {pinned.Name}");
                if (slot < 6 && !actual.Stats.Equals(pinned.Stats))
                    errors.Add($"fixed template slot {slot} must retain its pinned stat line");
            }

            foreach (var seat in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                int legal = 0;
                foreach (var cell in _board.DeploymentZone(seat))
                    if (_board.Contains(cell) && _game.Terrain(_board.TileAt(cell).Terrain).Passable) legal++;
                if (legal < _requiredUnits)
                    errors.Add($"starting deployment requires {_requiredUnits} passable cells for {seat} but only {legal} are available");
            }

            if (_board.DeploymentZone(PlayerId.Player0)
                .Any(cell => _board.IsInDeploymentZone(PlayerId.Player1, cell)))
                errors.Add("adaptive deployment zones must not overlap");

            if (errors.Count > 0) throw new ArgumentException(string.Join("; ", errors), "config");
        }

        internal static Board CopyBoard(Board source)
        {
            var control = new Dictionary<HexCoord, PlayerId>();
            foreach (var tile in source.Tiles)
            {
                var owner = source.Controller(tile.Coord);
                if (owner.HasValue) control[tile.Coord] = owner.Value;
            }
            return new Board(source.Tiles.ToArray(),
                source.DeploymentZone(PlayerId.Player0).ToArray(),
                source.DeploymentZone(PlayerId.Player1).ToArray(),
                control);
        }

        private static GameConfig CopyGameConfig(GameConfig source)
        {
            var terrain = Enum.GetValues(typeof(TerrainType)).Cast<TerrainType>()
                .ToDictionary(type => type, type => source.Terrain(type));
            return new GameConfig(
                terrain,
                startingPoints: source.StartingPoints,
                bountyRate: source.BountyRate,
                generatorCost: source.GeneratorCost,
                generatorOutput: source.GeneratorOutput,
                generatorHealth: source.GeneratorHealth,
                damageFloor: source.DamageFloor,
                dmgHighGroundBonus: source.DmgHighGroundBonus,
                rangeHighGroundBonus: source.RangeHighGroundBonus,
                roundCap: source.RoundCap,
                designFee: source.DesignFee,
                deployCostMultiplier: source.DeployCostMultiplier,
                turnPolicy: source.TurnPolicy,
                biomesEnabled: source.BiomesEnabled,
                winConditions: source.WinConditions,
                captureCost: source.CaptureCost,
                economyWinThreshold: source.EconomyWinThreshold,
                scoreKills: source.ScoreKills,
                scorePoints: source.ScorePoints,
                scoreArmy: source.ScoreArmy,
                scoreTerritory: source.ScoreTerritory,
                upkeepFactor: source.UpkeepFactor,
                captureFactor: source.CaptureFactor,
                buildFactor: source.BuildFactor,
                territoryMode: source.TerritoryMode,
                claimEndsTurn: source.ClaimEndsTurn,
                buildAnywhere: source.BuildAnywhere,
                territoryIncome: source.TerritoryIncome,
                generatorsEnabled: source.GeneratorsEnabled,
                pointDecay: source.PointDecay,
                fogOfWar: source.FogOfWar);
        }

        private static int RequiredSeatIndex(PlayerId seat)
        {
            int index = SeatIndex(seat);
            if (index < 0) throw new ArgumentOutOfRangeException(nameof(seat));
            return index;
        }

        private static int SeatIndex(PlayerId seat)
        {
            int index = (int)seat;
            return index == 0 || index == 1 ? index : -1;
        }
    }

    public sealed class RandomDeploymentPolicy : IDeploymentPolicy
    {
        private readonly int _seed;

        public RandomDeploymentPolicy(int seed) { _seed = seed; }

        public IReadOnlyList<DeploymentPlacement> Choose(AdaptiveDeploymentView view)
        {
            if (view == null) throw new ArgumentNullException(nameof(view));
            var rng = new Random(_seed);
            var cells = DeploymentPolicySupport.LegalUnusedCells(view);
            DeploymentPolicySupport.Shuffle(cells, rng);
            var slots = DeploymentPolicySupport.FreeSlots(view);
            int needed = Math.Min(slots.Count, cells.Count);
            int remainingBudget = view.RemainingBudget;
            int minimumCost = view.Templates.Min(t => t.Stats.PointCost);
            var result = new List<DeploymentPlacement>(needed);

            for (int i = 0; i < needed; i++)
            {
                int leftAfterThis = needed - i - 1;
                var affordable = Enumerable.Range(0, view.Templates.Count)
                    .Where(t => view.Templates[t].Stats.PointCost + minimumCost * leftAfterThis <= remainingBudget)
                    .ToList();
                if (affordable.Count == 0) break;
                DeploymentPolicySupport.Shuffle(affordable, rng);
                int templateIndex = affordable[0];
                result.Add(new DeploymentPlacement(slots[i], templateIndex, cells[i]));
                remainingBudget -= view.Templates[templateIndex].Stats.PointCost;
            }
            return result;
        }
    }

    public sealed class CombinedArmsDeploymentPolicy : IDeploymentPolicy
    {
        private readonly int _seed;

        public CombinedArmsDeploymentPolicy(int seed) { _seed = seed; }

        public IReadOnlyList<DeploymentPlacement> Choose(AdaptiveDeploymentView view)
        {
            if (view == null) throw new ArgumentNullException(nameof(view));
            var rng = new Random(_seed);
            var cells = DeploymentPolicySupport.LegalUnusedCells(view);
            var slots = DeploymentPolicySupport.FreeSlots(view);
            int needed = Math.Min(slots.Count, cells.Count);
            int remainingBudget = view.RemainingBudget;
            var occupied = view.OwnPlacements.Select(p => p.Cell).ToList();
            var selectedFixedTemplates = new HashSet<int>(view.OwnPlacements
                .Where(p => p.TemplateIndex >= 0 && p.TemplateIndex < 6)
                .Select(p => p.TemplateIndex));
            var result = new List<DeploymentPlacement>(needed);

            for (int i = 0; i < needed; i++)
            {
                int desired = -1;
                for (int template = 0; template < 6; template++)
                    if (!selectedFixedTemplates.Contains(template))
                    {
                        desired = template;
                        break;
                    }
                int leftAfterThis = needed - i - 1;
                int templateIndex = AffordableTemplate(view, desired, remainingBudget, leftAfterThis);
                if (templateIndex < 0) break;

                var ranked = cells.Select(cell => new RankedCell(
                        cell,
                        ScoreCell(view, templateIndex, cell, occupied),
                        rng.Next()))
                    .OrderByDescending(x => x.Score)
                    .ThenByDescending(x => x.Tie)
                    .ThenBy(x => x.Cell.Q)
                    .ThenBy(x => x.Cell.R)
                    .ToArray();
                if (ranked.Length == 0) break;

                var chosen = ranked[0].Cell;
                result.Add(new DeploymentPlacement(slots[i], templateIndex, chosen));
                occupied.Add(chosen);
                if (templateIndex < 6) selectedFixedTemplates.Add(templateIndex);
                cells.Remove(chosen);
                remainingBudget -= view.Templates[templateIndex].Stats.PointCost;
            }
            return result;
        }

        private static int AffordableTemplate(AdaptiveDeploymentView view, int desired, int remainingBudget,
            int remainingUnits)
        {
            int minimumCost = view.Templates.Min(t => t.Stats.PointCost);
            if (desired >= 0 && desired < view.Templates.Count
                && view.Templates[desired].Stats.PointCost + minimumCost * remainingUnits <= remainingBudget)
                return desired;

            return Enumerable.Range(0, view.Templates.Count)
                .Where(t => view.Templates[t].Stats.PointCost + minimumCost * remainingUnits <= remainingBudget)
                .OrderBy(t => view.Templates[t].Stats.PointCost)
                .ThenBy(t => t)
                .DefaultIfEmpty(-1)
                .First();
        }

        private static long ScoreCell(AdaptiveDeploymentView view, int templateIndex, HexCoord cell,
            IReadOnlyList<HexCoord> friendly)
        {
            var ownZone = view.Board.DeploymentZone(view.Seat);
            var opponent = view.Seat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            double ownAverage = ownZone.Average(c => c.Q);
            double opponentAverage = view.Board.DeploymentZone(opponent).Average(c => c.Q);
            bool forwardIsPositive = ownAverage <= opponentAverage;
            int minQ = ownZone.Min(c => c.Q);
            int maxQ = ownZone.Max(c => c.Q);
            int forwardDepth = forwardIsPositive ? cell.Q - minQ : maxQ - cell.Q;
            int rearDepth = maxQ - minQ - forwardDepth;
            int span = Math.Max(1, maxQ - minQ + 1);

            if (templateIndex == 0 || templateIndex == 1)
                return forwardDepth;
            if (templateIndex == 2 || templateIndex == 3)
                return (long)view.Board.TileAt(cell).Elevation * span + rearDepth;
            if (templateIndex == 4)
            {
                int spacing = friendly.Count == 0 ? 0 : friendly.Min(other => HexCoord.Distance(cell, other));
                return (long)forwardDepth * (span + 1) + spacing;
            }
            if (templateIndex == 5)
                return -friendly.Sum(other => HexCoord.Distance(cell, other));
            return 0;
        }

        private readonly struct RankedCell
        {
            public HexCoord Cell { get; }
            public long Score { get; }
            public int Tie { get; }

            public RankedCell(HexCoord cell, long score, int tie)
            {
                Cell = cell;
                Score = score;
                Tie = tie;
            }
        }
    }

    internal static class DeploymentPolicySupport
    {
        public static List<HexCoord> LegalUnusedCells(AdaptiveDeploymentView view)
        {
            var occupied = new HashSet<HexCoord>(view.OwnPlacements.Select(p => p.Cell));
            return view.Board.DeploymentZone(view.Seat)
                .Where(cell => view.Board.Contains(cell)
                    && view.Game.Terrain(view.Board.TileAt(cell).Terrain).Passable
                    && !occupied.Contains(cell))
                .OrderBy(cell => cell.Q)
                .ThenBy(cell => cell.R)
                .ToList();
        }

        public static List<int> FreeSlots(AdaptiveDeploymentView view)
        {
            var occupied = new HashSet<int>(view.OwnPlacements.Select(p => p.Slot));
            return Enumerable.Range(0, view.RequiredUnits).Where(slot => !occupied.Contains(slot)).ToList();
        }

        public static void Shuffle<T>(IList<T> items, Random rng)
        {
            for (int i = items.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                T swap = items[i];
                items[i] = items[j];
                items[j] = swap;
            }
        }
    }
}

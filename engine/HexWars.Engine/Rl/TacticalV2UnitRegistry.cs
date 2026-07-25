using System;
using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// Stable per-seat unit-slot identity for tactical-v2: a fixed-capacity slot -> (unit id, template
    /// index) map. A slot is assigned once — at <see cref="Initialize"/> for the starting army, or by
    /// <see cref="RegisterDeployment"/> for a reinforcement — and is never reassigned to a different
    /// living unit; only <see cref="ReleaseDead"/> frees it, once its unit has died. Template identity
    /// is recorded explicitly per slot, never inferred by comparing <see cref="UnitStats"/> (two
    /// catalog templates can share an identical stat line and still be distinct roles).
    /// </summary>
    public sealed class TacticalV2UnitRegistry
    {
        private readonly int[] _unitIds;
        private readonly int[] _templateIndices;

        public TacticalV2UnitRegistry(int capacity)
        {
            if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            _unitIds = Enumerable.Repeat(-1, capacity).ToArray();
            _templateIndices = Enumerable.Repeat(-1, capacity).ToArray();
        }

        public int Capacity => _unitIds.Length;

        /// <summary>True while at least one slot has no living unit tracked in it — i.e. a deploy can
        /// still claim a slot via <see cref="RegisterDeployment"/> without throwing.</summary>
        public bool HasFreeSlot => Array.IndexOf(_unitIds, -1) >= 0;

        public int UnitIdAt(int slot) => slot >= 0 && slot < Capacity ? _unitIds[slot] : -1;
        public int TemplateIndexAt(int slot) => slot >= 0 && slot < Capacity ? _templateIndices[slot] : -1;
        public int SlotOf(int unitId) => Array.IndexOf(_unitIds, unitId);

        /// <summary>Seeds the registry from the starting army: <paramref name="units"/>[i] takes slot i
        /// and is tagged with <paramref name="templateIndices"/>[i]. Overwrites any prior contents; any
        /// slot beyond <paramref name="units"/>'s length is left free.</summary>
        public void Initialize(IReadOnlyList<Unit> units, IReadOnlyList<int> templateIndices)
        {
            if (units == null) throw new ArgumentNullException(nameof(units));
            if (templateIndices == null) throw new ArgumentNullException(nameof(templateIndices));
            if (units.Count != templateIndices.Count)
                throw new ArgumentException(
                    "units and templateIndices must be the same length", nameof(templateIndices));
            if (units.Count > Capacity)
                throw new ArgumentException(
                    $"tactical-v2 unit registry capacity {Capacity} exceeded by {units.Count} starting units",
                    nameof(units));

            for (int slot = 0; slot < Capacity; slot++)
            {
                _unitIds[slot] = slot < units.Count ? units[slot].Id : -1;
                _templateIndices[slot] = slot < units.Count ? templateIndices[slot] : -1;
            }
        }

        /// <summary>Frees the slot of any tracked unit no longer alive on <paramref name="seat"/>'s
        /// board. Never assigns a slot to a different unit — only <see cref="RegisterDeployment"/> does
        /// that — so this method only ever clears entries, it never moves them.</summary>
        public void ReleaseDead(GameState state, PlayerId seat)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));

            var livingIds = new HashSet<int>();
            foreach (var unit in state.Player(seat).UnitsOnBoard)
                if (unit.IsAlive) livingIds.Add(unit.Id);

            for (int slot = 0; slot < Capacity; slot++)
            {
                if (_unitIds[slot] == -1 || livingIds.Contains(_unitIds[slot])) continue;
                _unitIds[slot] = -1;
                _templateIndices[slot] = -1;
            }
        }

        /// <summary>Identifies the single living unit id present in <paramref name="after"/> but absent
        /// from <paramref name="before"/> for <paramref name="seat"/> — the unit a just-applied
        /// DeployUnit created — and claims the lowest free slot for it, tagging that slot with
        /// <paramref name="templateIndex"/>. Throws if the two states don't differ by exactly one new
        /// living unit for this seat, or if every slot is already taken.</summary>
        public void RegisterDeployment(GameState before, GameState after, PlayerId seat, int templateIndex)
        {
            if (before == null) throw new ArgumentNullException(nameof(before));
            if (after == null) throw new ArgumentNullException(nameof(after));

            var beforeIds = new HashSet<int>();
            foreach (var unit in before.Player(seat).UnitsOnBoard) beforeIds.Add(unit.Id);

            int newUnitId = -1;
            int newUnitCount = 0;
            foreach (var unit in after.Player(seat).UnitsOnBoard)
            {
                if (!unit.IsAlive || beforeIds.Contains(unit.Id)) continue;
                newUnitId = unit.Id;
                newUnitCount++;
            }
            if (newUnitCount != 1)
                throw new InvalidOperationException(
                    $"expected exactly one newly deployed unit for {seat}, found {newUnitCount}");

            int freeSlot = -1;
            for (int slot = 0; slot < Capacity; slot++)
                if (_unitIds[slot] == -1) { freeSlot = slot; break; }
            if (freeSlot < 0)
                throw new InvalidOperationException(
                    $"tactical-v2 unit registry capacity {Capacity} exceeded");

            _unitIds[freeSlot] = newUnitId;
            _templateIndices[freeSlot] = templateIndex;
        }
    }
}

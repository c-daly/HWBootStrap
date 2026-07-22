using System;
using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>Stable policy-slot mapping for a bounded set of living units.</summary>
    public sealed class AdaptiveUnitSlots
    {
        private readonly int[] _unitIds;

        public AdaptiveUnitSlots(int capacity)
        {
            if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            _unitIds = Enumerable.Repeat(-1, capacity).ToArray();
        }

        public int Capacity => _unitIds.Length;
        public int UnitIdAt(int slot) => slot >= 0 && slot < Capacity ? _unitIds[slot] : -1;
        public int SlotOf(int unitId) => Array.IndexOf(_unitIds, unitId);
        public bool HasFreeSlot => Array.IndexOf(_unitIds, -1) >= 0;
        public bool HasOverflow => OverflowCount > 0;
        public int OverflowCount { get; private set; }

        public void Sync(GameState state, PlayerId seat)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));

            var livingIds = new HashSet<int>(state.Player(seat).UnitsOnBoard
                .Where(unit => unit.IsAlive)
                .Select(unit => unit.Id));

            for (int slot = 0; slot < _unitIds.Length; slot++)
                if (!livingIds.Contains(_unitIds[slot])) _unitIds[slot] = -1;

            foreach (int unitId in livingIds.OrderBy(id => id))
            {
                if (SlotOf(unitId) >= 0) continue;
                int free = Array.IndexOf(_unitIds, -1);
                if (free < 0) break;
                _unitIds[free] = unitId;
            }

            int tracked = 0;
            foreach (int unitId in _unitIds)
                if (livingIds.Contains(unitId)) tracked++;
            OverflowCount = livingIds.Count - tracked;
        }
    }
}

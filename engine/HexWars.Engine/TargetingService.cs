using System;

namespace HexWars.Engine
{
    /// <summary>
    /// Pure targeting rules. A target may be attacked when it is in the attacker's per-unit reach
    /// (horizontal <see cref="UnitStats.Range"/> + high-ground bonus, vertical <see cref="UnitStats.RangeArc"/>)
    /// AND it is visible to the attacker's ARMY — sight is shared, so any living friendly unit whose
    /// horizontal <see cref="UnitStats.Vision"/> (after the target's terrain concealment) and vertical
    /// <see cref="UnitStats.VisionArc"/> cover the target makes it targetable for the whole team.
    /// </summary>
    public static class TargetingService
    {
        public static bool CanTarget(GameState state, Unit attacker, HexCoord targetCell, int targetElevation)
            => InRange(attacker, targetCell, targetElevation, state.Config)
               && IsVisibleToArmy(state, attacker.Owner, targetCell, targetElevation);

        /// <summary>Per-unit reach: horizontal Range (+ high-ground bonus) and vertical RangeArc (firing up).</summary>
        public static bool InRange(Unit attacker, HexCoord targetCell, int targetElevation, GameConfig config)
        {
            int hd = HexCoord.Distance(attacker.Cell, targetCell);
            int up = Math.Max(0, targetElevation - attacker.Elevation);
            int highGround = Math.Max(0, attacker.Elevation - targetElevation);

            bool horizontal = hd <= attacker.Stats.Range + highGround * config.RangeHighGroundBonus;
            bool vertical = up <= attacker.Stats.RangeArc;
            return horizontal && vertical;
        }

        /// <summary>Army-wide shared vision: true if ANY living friendly unit can see the target.</summary>
        public static bool IsVisibleToArmy(GameState state, PlayerId army, HexCoord targetCell, int targetElevation)
        {
            int concealment = state.Config.Terrain(state.Board.TileAt(targetCell).Terrain).Concealment;

            foreach (var spotter in state.Player(army).UnitsOnBoard)
            {
                if (!spotter.IsAlive) continue;
                int hd = HexCoord.Distance(spotter.Cell, targetCell);
                int up = Math.Max(0, targetElevation - spotter.Elevation);
                if (hd + concealment <= spotter.Stats.Vision && up <= spotter.Stats.VisionArc)
                    return true;
            }
            return false;
        }
    }
}

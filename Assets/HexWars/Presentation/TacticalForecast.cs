using System;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>Read-only presentation of an engine-validated attack. Never predicts hidden units.</summary>
    public readonly struct TacticalForecast
    {
        public Unit Target { get; }
        public int Weapon { get; }
        public int HeightBonus { get; }
        public int Armor { get; }
        public int Cover { get; }
        public int Damage { get; }
        public int RemainingHealth => Math.Max(0, Target.CurrentHp - Damage);

        TacticalForecast(GameState state, Unit attacker, Unit target)
        {
            Target = target;
            Weapon = attacker.Stats.Damage;
            HeightBonus = Weapon > 0 ? Math.Max(0, attacker.Elevation - target.Elevation) * state.Config.DmgHighGroundBonus : 0;
            Armor = target.Stats.Defense;
            Cover = state.Config.Terrain(state.Board.TileAt(target.Cell).Terrain).Defense;
            Damage = CombatResolver.ComputeDamage(Weapon, attacker.Elevation, target.Elevation, Armor + Cover, state.Config);
        }

        public static Unit? FindVisible(GameState state, int id, PlayerId viewer)
        {
            if (state == null) return null;
            foreach (var player in state.Players)
                foreach (var unit in player.UnitsOnBoard)
                    if (unit.Id == id && unit.IsAlive)
                    {
                        if (unit.Owner != viewer && state.Config.FogOfWar &&
                            !TargetingService.IsVisibleToArmy(state, viewer, unit.Cell, unit.Elevation)) return null;
                        return unit;
                    }
            return null;
        }

        public static bool TryCreate(GameState state, PlayerId viewer, int attackerId, int targetId, out TacticalForecast forecast)
        {
            forecast = default;
            var attacker = FindVisible(state, attackerId, viewer);
            var target = FindVisible(state, targetId, viewer);
            if (!attacker.HasValue || !target.HasValue || attacker.Value.Owner != viewer) return false;
            // Apply is pure. Asking the engine also covers pacing, ended games and already-spent attacks.
            var command = new AttackUnit(viewer, attackerId, targetId);
            if (!GameEngine.Apply(state, command).Success) return false;
            forecast = new TacticalForecast(state, attacker.Value, target.Value);
            return true;
        }

        public static bool HasAttacked(GameState state, int id)
        {
            foreach (int used in state.AttackedUnitIds) if (used == id) return true;
            return false;
        }
    }
}

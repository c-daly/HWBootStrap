using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    public readonly struct AttackPreviewTarget
    {
        public int UnitId { get; }
        public HexCoord Cell { get; }
        public int Elevation { get; }

        public AttackPreviewTarget(int unitId, HexCoord cell, int elevation)
        {
            UnitId = unitId;
            Cell = cell;
            Elevation = elevation;
        }
    }

    /// <summary>Finds enemy units the selected unit may attack without exposing hidden information.</summary>
    public static class AttackPreviewTargets
    {
        public static IReadOnlyList<AttackPreviewTarget> Resolve(
            GameState state,
            Unit attacker,
            HexCoord? previewDestination,
            PlayerId viewer)
        {
            foreach (var unitId in state.AttackedUnitIds)
                if (unitId == attacker.Id) return new AttackPreviewTarget[0];

            if (previewDestination.HasValue && !state.Board.Contains(previewDestination.Value))
                return new AttackPreviewTarget[0];

            var observedState = state;
            if (previewDestination.HasValue)
                state = WithAttackerAt(state, attacker, previewDestination.Value, out attacker);

            var targets = new List<AttackPreviewTarget>();
            foreach (var enemy in state.Opponent(attacker.Owner).UnitsOnBoard)
            {
                if (!enemy.IsAlive) continue;
                if (observedState.Config.FogOfWar
                    && !TargetingService.IsVisibleToArmy(
                        observedState, viewer, enemy.Cell, enemy.Elevation))
                    continue;
                if (!TargetingService.CanTarget(state, attacker, enemy.Cell, enemy.Elevation)) continue;
                targets.Add(new AttackPreviewTarget(enemy.Id, enemy.Cell, enemy.Elevation));
            }
            return targets;
        }

        static GameState WithAttackerAt(
            GameState state,
            Unit attacker,
            HexCoord destination,
            out Unit movedAttacker)
        {
            movedAttacker = attacker.WithCell(
                destination, state.Board.TileAt(destination).Elevation);

            var units = new List<Unit>(state.Player(attacker.Owner).UnitsOnBoard.Count);
            foreach (var unit in state.Player(attacker.Owner).UnitsOnBoard)
                units.Add(unit.Id == attacker.Id ? movedAttacker : unit);

            var players = new List<PlayerState>(state.Players);
            var owner = state.Player(attacker.Owner);
            players[(int)attacker.Owner] = new PlayerState(
                owner.Id, owner.Points, owner.Barracks, units, owner.Generators, owner.DestroyedValue);

            return new GameState(
                state.Board, state.Config, players, state.ActivePlayer, state.Round, state.NextEntityId,
                state.IsGameOver, state.Winner, state.MovedUnitIds, state.AttackedUnitIds,
                state.MovementSpent);
        }
    }
}

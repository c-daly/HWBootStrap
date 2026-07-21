using System;
using HexWars.Engine;

namespace HexWars.Presentation
{
    public static class MovementTooltipFormatter
    {
        public static string FormatRoute(MovementRoute route) =>
            $"Route: {route.HorizontalCost} move · {route.VerticalCost} climb · leaves " +
            $"{route.HorizontalRemaining} move / {route.VerticalRemaining} climb";

        public static string FormatMovementStatus(Unit unit, GameState state)
        {
            foreach (var attackedId in state.AttackedUnitIds)
                if (attackedId == unit.Id) return "Movement ended by attack";

            var spent = state.MovementSpent.TryGetValue(unit.Id, out var value)
                ? value
                : (H: 0, V: 0);
            int moveLeft = Math.Max(0, unit.Stats.Movement - spent.H);
            int climbLeft = Math.Max(0, unit.Stats.VerticalMovement - spent.V);
            return $"Move {moveLeft}/{unit.Stats.Movement}   " +
                   $"Climb {climbLeft}/{unit.Stats.VerticalMovement}";
        }
    }
}

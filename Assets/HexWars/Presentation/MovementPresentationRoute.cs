using System;
using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>Resolves presentation playback from the same route map that accepted the move.</summary>
    public static class MovementPresentationRoute
    {
        public static IReadOnlyList<HexCoord> Resolve(GameState previous, MoveUnit command)
        {
            if (previous == null || command == null) return Array.Empty<HexCoord>();

            foreach (var unit in previous.Player(command.Issuer).UnitsOnBoard)
            {
                if (unit.Id != command.UnitId || !unit.IsAlive) continue;
                var routes = MovementService.Routes(previous, unit);
                return routes.TryGetValue(command.Dest, out var route)
                    ? route.Cells
                    : Array.Empty<HexCoord>();
            }

            return Array.Empty<HexCoord>();
        }
    }
}

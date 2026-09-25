using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine
{
    /// <summary>Free, sequential army arrangement. It spends no points, movement, actions or rounds.</summary>
    public static class StartingPlacement
    {
        public static IReadOnlyList<HexCoord> Cells(GameState state, Unit unit)
        {
            var cells = new List<HexCoord>();
            if (!state.PlacingStartingUnits || unit.Owner != state.ActivePlayer || !unit.IsAlive) return cells;
            foreach (var cell in state.Board.DeploymentZone(unit.Owner))
                if (ValidateCell(state, unit.Owner, cell) == RejectionReason.None) cells.Add(cell);
            cells.Sort((a, b) => a.Q == b.Q ? a.R.CompareTo(b.R) : a.Q.CompareTo(b.Q));
            return cells;
        }

        static RejectionReason ValidateCell(GameState state, PlayerId owner, HexCoord cell)
        {
            if (!state.Board.Contains(cell)) return RejectionReason.TileNotFound;
            if (!state.Board.IsInDeploymentZone(owner, cell)) return RejectionReason.OutsideDeploymentZone;
            if (!state.Config.Terrain(state.Board.TileAt(cell).Terrain).Passable) return RejectionReason.TileImpassable;
            foreach (var player in state.Players)
            {
                foreach (var unit in player.UnitsOnBoard)
                    if (unit.IsAlive && unit.Cell == cell) return RejectionReason.TileOccupied;
                foreach (var generator in player.Generators)
                    if (generator.IsAlive && generator.Cell == cell) return RejectionReason.TileOccupied;
            }
            return RejectionReason.None;
        }

        internal static Result Apply(GameState state, Command command)
        {
            if (command is FinishPlacement)
                return Result.Ok(new GameState(state.Board, state.Config, state.Players,
                    state.ActivePlayer == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0,
                    state.Round, state.NextEntityId,
                    placingStartingUnits: state.ActivePlayer == PlayerId.Player0));
            if (!(command is PlaceStartingUnit place)) return Result.Reject(state, RejectionReason.PlacementOnly);
            var owner = state.Player(place.Issuer);
            var units = owner.UnitsOnBoard.ToArray();
            int index = System.Array.FindIndex(units, u => u.Id == place.UnitId && u.IsAlive);
            if (index < 0) return Result.Reject(state, RejectionReason.UnitNotFound);
            var reason = ValidateCell(state, place.Issuer, place.Cell);
            if (reason != RejectionReason.None) return Result.Reject(state, reason);
            units[index] = units[index].WithCell(place.Cell, state.Board.TileAt(place.Cell).Elevation);
            var players = state.Players.ToArray();
            players[(int)place.Issuer] = new PlayerState(owner.Id, owner.Points, owner.Barracks, units, owner.Generators, owner.DestroyedValue);
            return Result.Ok(new GameState(state.Board, state.Config, players, state.ActivePlayer,
                state.Round, state.NextEntityId, placingStartingUnits: true));
        }
    }
}

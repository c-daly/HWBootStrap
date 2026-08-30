using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Presentation
{
    /// <summary>
    /// Adapts the live, ordinary playable GameState to tactical-v3's ragged observation and legal-
    /// candidate contract.  The game remains authoritative; this class only asks a policy to choose
    /// one command and round-trips that choice back to the same immutable state instance.
    /// </summary>
    public sealed class PlayableModelAdapter
    {
        readonly TacticalV3LegalCandidateSource _candidates;
        readonly TacticalV3ActionResolver _resolver = new TacticalV3ActionResolver();

        public PlayableModelAdapter(GameState state, PlayerId modelSeat)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            RequireSupportedState(state);
            (int width, int height, int maxElevation) = BoardShape(state.Board);

            int startingUnits = Math.Max(1, Math.Max(
                state.Player(PlayerId.Player0).UnitsOnBoard.Count,
                state.Player(PlayerId.Player1).UnitsOnBoard.Count));
            if (startingUnits > 12)
                throw new InvalidOperationException(
                    "trained model supports at most 12 starting units per player");

            TacticalV2Config match = TacticalV2Config.Default();
            match.BoardGen = new BoardGenConfig(
                width, height, maxElevation: Math.Max(1, maxElevation));
            match.Game = state.Config;
            match.StartingUnitCount = startingUnits;
            match.MaxControllableUnits = startingUnits;
            match.MaxSteps = TacticalV2Config.DefaultMaxSteps(
                startingUnits, state.Config.RoundCap);

            Config = new TacticalV3Config(
                match,
                TacticalV3CapacityProfile.ExperimentalDefault(),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));
            var observations = new TacticalV3SeatObservationSource(Config);
            _candidates = new TacticalV3LegalCandidateSource(
                observations, Config.Capacity);

            // Observe immediately so board shape, roster size, template count, and relation capacity
            // fail before a human can spend time playing toward the model's first turn.
            observations.Observe(state, modelSeat, EmptyObservationMemory.Instance);
            TacticalV3Contract contract = TacticalV3Contract.Create(
                Config, MlEnvironmentKind.Duel);
            ContractIdentity = new ModelDuelContractIdentity(
                contract.Version, contract.Version,
                contract.EncodingHash, contract.CapacityHash);
        }

        public TacticalV3Config Config { get; }
        public ModelDuelContractIdentity ContractIdentity { get; }

        public TacticalV3DecisionFrame CreateFrame(
            GameState state, PlayerId seat, long decisionId) =>
            _candidates.CreateFrame(
                state, seat, EmptyObservationMemory.Instance, decisionId);

        public Command Resolve(
            TacticalV3DecisionFrame frame,
            PolicyCandidateResult selected,
            GameState currentState)
        {
            if (selected == null) throw new ArgumentNullException(nameof(selected));
            return _resolver.Resolve(
                frame, selected.DecisionId, selected.CandidateId, currentState);
        }

        public static bool Supports(GameSetup setup, out string reason)
        {
            GameSetup sanitized = setup.Sanitized();
            if (sanitized.Mode != GameMode.Annihilation)
            {
                reason = "The trained model currently supports Annihilation only.";
                return false;
            }
            if (sanitized.Fog)
            {
                reason = "Turn off fog to play against the trained model.";
                return false;
            }
            long cells = (long)sanitized.Width * sanitized.Height;
            if (cells > TacticalV3CapacityProfile.ExperimentalDefault().MaxCells)
            {
                reason = "That map is too large for the trained model (512 hexes maximum).";
                return false;
            }
            reason = string.Empty;
            return true;
        }

        /// <summary>
        /// Keeps a live normal match inside the published policy's ragged-table capacities. Creation
        /// and deployment are ordinary human-facing commands rather than tactical-v3 candidates, so
        /// they must be stopped before they make the next model observation unrepresentable.
        /// </summary>
        public static bool PreservesCapacity(
            GameState state, Command command, out string reason)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (command == null) throw new ArgumentNullException(nameof(command));
            TacticalV3CapacityProfile capacity =
                TacticalV3CapacityProfile.ExperimentalDefault();

            if (command is CreateUnit)
            {
                int templates = state.Players.Sum(player => player.Barracks.Count);
                if (templates >= capacity.MaxTemplates)
                {
                    reason = "The trained-model match has reached its 32-template limit.";
                    return false;
                }
            }
            if (command is DeployUnit)
            {
                int units = state.Players.Sum(player => player.UnitsOnBoard.Count);
                if (units >= capacity.MaxUnits)
                {
                    reason = "The trained-model match has reached its 64-unit limit.";
                    return false;
                }
            }

            reason = string.Empty;
            return true;
        }

        static void RequireSupportedState(GameState state)
        {
            GameConfig game = state.Config;
            if (game.FogOfWar || game.TerritoryMode ||
                game.CaptureCost != int.MaxValue || game.GeneratorsEnabled ||
                game.WinConditions != WinBy.Annihilation)
                throw new InvalidOperationException(
                    "trained model requires an annihilation match without fog, capture, or generators");
        }

        static (int Width, int Height, int MaxElevation) BoardShape(Board board)
        {
            if (board == null) throw new ArgumentNullException(nameof(board));
            if (board.TileCount == 0)
                throw new InvalidOperationException("trained model requires a non-empty board");

            int minColumn = int.MaxValue;
            int maxColumn = int.MinValue;
            int minRow = int.MaxValue;
            int maxRow = int.MinValue;
            int maxElevation = 0;
            var occupied = new HashSet<(int Column, int Row)>();
            foreach (Tile tile in board.Tiles)
            {
                int column = tile.Coord.Q;
                int row = tile.Coord.R + (column - (column & 1)) / 2;
                occupied.Add((column, row));
                minColumn = Math.Min(minColumn, column);
                maxColumn = Math.Max(maxColumn, column);
                minRow = Math.Min(minRow, row);
                maxRow = Math.Max(maxRow, row);
                maxElevation = Math.Max(maxElevation, tile.Elevation);
            }

            int width = checked(maxColumn - minColumn + 1);
            int height = checked(maxRow - minRow + 1);
            if (minColumn != 0 || minRow != 0 ||
                occupied.Count != checked(width * height) ||
                occupied.Any(cell => cell.Column < 0 || cell.Row < 0))
                throw new InvalidOperationException(
                    "trained model requires a complete odd-q rectangular board rooted at (0,0)");
            if (occupied.Count > TacticalV3CapacityProfile.ExperimentalDefault().MaxCells)
                throw new InvalidOperationException(
                    "trained model board exceeds its 512-cell capacity");
            return (width, height, maxElevation);
        }
    }
}

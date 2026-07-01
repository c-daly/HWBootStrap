using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// The complete, immutable game state. The single <c>Apply</c> mutation path returns a NEW
    /// GameState, so the engine is search-able (an AI can fork and score freely).
    /// <see cref="MovedUnitIds"/> / <see cref="AttackedUnitIds"/> track which units have used their
    /// move / attack this turn; <see cref="MovementSpent"/> tracks how much of a unit's per-turn
    /// movement budgets (horizontal, vertical) its hops have consumed, so a unit may move in several
    /// steps. All reset on EndTurn.
    /// </summary>
    public sealed class GameState
    {
        private static readonly IReadOnlyCollection<int> NoIds = new int[0];
        private static readonly IReadOnlyDictionary<int, (int H, int V)> NoSpent =
            new Dictionary<int, (int H, int V)>();

        public Board Board { get; }
        public GameConfig Config { get; }

        /// <summary>Length-2, indexed by (int)PlayerId.</summary>
        public IReadOnlyList<PlayerState> Players { get; }

        public PlayerId ActivePlayer { get; }
        public int Round { get; }
        public int NextEntityId { get; }
        public bool IsGameOver { get; }
        public PlayerId? Winner { get; }
        public IReadOnlyCollection<int> MovedUnitIds { get; }
        public IReadOnlyCollection<int> AttackedUnitIds { get; }

        /// <summary>Per-unit (horizontal, vertical) movement points consumed this turn by hops.</summary>
        public IReadOnlyDictionary<int, (int H, int V)> MovementSpent { get; }

        public GameState(
            Board board,
            GameConfig config,
            IReadOnlyList<PlayerState> players,
            PlayerId activePlayer,
            int round,
            int nextEntityId,
            bool isGameOver = false,
            PlayerId? winner = null,
            IReadOnlyCollection<int>? movedUnitIds = null,
            IReadOnlyCollection<int>? attackedUnitIds = null,
            IReadOnlyDictionary<int, (int H, int V)>? movementSpent = null)
        {
            Board = board;
            Config = config;
            Players = players;
            ActivePlayer = activePlayer;
            Round = round;
            NextEntityId = nextEntityId;
            IsGameOver = isGameOver;
            Winner = winner;
            MovedUnitIds = movedUnitIds ?? NoIds;
            AttackedUnitIds = attackedUnitIds ?? NoIds;
            MovementSpent = movementSpent ?? NoSpent;
        }

        public PlayerState Player(PlayerId id) => Players[(int)id];
        public PlayerState Opponent(PlayerId id) => Players[1 - (int)id];

        /// <summary>A distinct GameState with the same (immutable) contents.</summary>
        public GameState Clone() =>
            new GameState(Board, Config, Players, ActivePlayer, Round, NextEntityId,
                          IsGameOver, Winner, MovedUnitIds, AttackedUnitIds, MovementSpent);
    }
}

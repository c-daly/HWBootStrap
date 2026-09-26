#nullable enable
using System;

namespace HexWars.Presentation
{
    /// <summary>Where the Steam matchmaking flow currently stands.</summary>
    public enum SteamLobbyPhase
    {
        /// <summary>Nothing in flight.</summary>
        Idle,

        /// <summary>Steam is not initialised or the user is not logged on.</summary>
        SteamUnavailable,

        /// <summary>Looking for an open Quick Match lobby, or joining an invited one.</summary>
        Searching,

        /// <summary>Creating a lobby of our own.</summary>
        CreatingLobby,

        /// <summary>In a lobby, alone.</summary>
        WaitingForPlayer,

        /// <summary>Both players are in the lobby; at least one has not readied up.</summary>
        WaitingForReady,

        /// <summary>Asking Steam for a Web API auth ticket.</summary>
        RequestingTicket,

        /// <summary>The lobby owner is asking the match service to allocate a match.</summary>
        AllocatingMatch,

        /// <summary>A guest is joining the match the owner allocated.</summary>
        JoiningMatch,

        /// <summary>A match ticket has been handed to the caller.</summary>
        MatchReady,

        /// <summary>Rejoining a match that was already allocated.</summary>
        Reconnecting,

        /// <summary>This client build or protocol version cannot play here.</summary>
        VersionMismatch,

        /// <summary>The match service could not be reached, or asked us to try later.</summary>
        BackendUnavailable,

        /// <summary>A terminal failure that is not one of the specific cases above.</summary>
        Failed,

        /// <summary>The player cancelled. The next Tick returns to <see cref="Idle"/>.</summary>
        Cancelled,
    }

    /// <summary>Immutable snapshot of the matchmaking flow, published on every change.</summary>
    public sealed class SteamLobbyStatus
    {
        public SteamLobbyStatus(
            SteamLobbyPhase phase,
            string? message,
            string? lobbyId,
            string? matchId,
            bool isOwner,
            bool localReady,
            bool remoteReady,
            string? opponentName,
            bool canCancel,
            bool canRetry,
            bool canReady)
        {
            Phase = phase;
            Message = message ?? string.Empty;
            LobbyId = lobbyId;
            MatchId = matchId;
            IsOwner = isOwner;
            LocalReady = localReady;
            RemoteReady = remoteReady;
            OpponentName = opponentName;
            CanCancel = canCancel;
            CanRetry = canRetry;
            CanReady = canReady;
        }

        public SteamLobbyPhase Phase { get; }

        /// <summary>The player-visible line for this phase, or the server message when there is one.</summary>
        public string Message { get; }

        public string? LobbyId { get; }

        public string? MatchId { get; }

        /// <summary>True when the local player owns the lobby, and therefore takes seat 0.</summary>
        public bool IsOwner { get; }

        public bool LocalReady { get; }

        public bool RemoteReady { get; }

        /// <summary>Display name of the other lobby member, or null when nobody else is present.</summary>
        public string? OpponentName { get; }

        public bool CanCancel { get; }

        public bool CanRetry { get; }

        public bool CanReady { get; }

        /// <summary>True when every field matches, used to suppress duplicate publications.</summary>
        public bool Matches(SteamLobbyStatus? other)
        {
            return other != null
                && Phase == other.Phase
                && string.Equals(Message, other.Message, StringComparison.Ordinal)
                && string.Equals(LobbyId, other.LobbyId, StringComparison.Ordinal)
                && string.Equals(MatchId, other.MatchId, StringComparison.Ordinal)
                && IsOwner == other.IsOwner
                && LocalReady == other.LocalReady
                && RemoteReady == other.RemoteReady
                && string.Equals(OpponentName, other.OpponentName, StringComparison.Ordinal)
                && CanCancel == other.CanCancel
                && CanRetry == other.CanRetry
                && CanReady == other.CanReady;
        }
    }

    /// <summary>The exact player-visible line for each phase.</summary>
    public static class SteamLobbyMessages
    {
        public const string Idle = "";
        public const string Searching = "Searching for a match\u2026";
        public const string CreatingLobby = "Creating lobby\u2026";
        public const string WaitingForPlayer = "Waiting for a player\u2026";
        public const string WaitingForReady = "Waiting for both players to ready up";
        public const string RequestingTicket = "Signing in with Steam\u2026";
        public const string AllocatingMatch = "Allocating server match\u2026";
        public const string JoiningMatch = "Joining match\u2026";
        public const string MatchReady = "Connecting\u2026";
        public const string Reconnecting = "Reconnecting\u2026";
        public const string SteamUnavailable = "Steam is unavailable \u2014 start the game through Steam.";
        public const string BackendUnavailable = "The match service is unavailable \u2014 try again.";
        public const string VersionMismatch = "Your game version is out of date \u2014 update HexWars in Steam.";
        public const string Failed = "Could not start the match.";

        /// <summary>Steam refused a lobby metadata write, so nobody else could ever see the lobby.</summary>
        public const string PublishFailed = "Could not publish the lobby.";
        public const string Cancelled = "Cancelled.";

        public static string For(SteamLobbyPhase phase)
        {
            switch (phase)
            {
                case SteamLobbyPhase.Searching: return Searching;
                case SteamLobbyPhase.CreatingLobby: return CreatingLobby;
                case SteamLobbyPhase.WaitingForPlayer: return WaitingForPlayer;
                case SteamLobbyPhase.WaitingForReady: return WaitingForReady;
                case SteamLobbyPhase.RequestingTicket: return RequestingTicket;
                case SteamLobbyPhase.AllocatingMatch: return AllocatingMatch;
                case SteamLobbyPhase.JoiningMatch: return JoiningMatch;
                case SteamLobbyPhase.MatchReady: return MatchReady;
                case SteamLobbyPhase.Reconnecting: return Reconnecting;
                case SteamLobbyPhase.SteamUnavailable: return SteamUnavailable;
                case SteamLobbyPhase.BackendUnavailable: return BackendUnavailable;
                case SteamLobbyPhase.VersionMismatch: return VersionMismatch;
                case SteamLobbyPhase.Failed: return Failed;
                case SteamLobbyPhase.Cancelled: return Cancelled;
                default: return Idle;
            }
        }
    }
}

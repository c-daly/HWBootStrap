#nullable enable
using System;
using System.Collections.Generic;

namespace HexWars.Presentation
{
    /// <summary>Who can see and join a lobby. Maps onto the Steam ELobbyType values.</summary>
    public enum SteamLobbyVisibility
    {
        Public,
        FriendsOnly,
        Private,
    }

    /// <summary>
    /// The lobby surface HexWars needs from Steam, expressed without any Steamworks or Unity type
    /// so it can be faked and unit tested off-platform. Every asynchronous result is delivered on
    /// the main thread from inside <see cref="Pump"/>, never synchronously from the call itself.
    /// </summary>
    public interface ISteamLobbyClient : IDisposable
    {
        /// <summary>True when Steam is initialised and the local user is logged on.</summary>
        bool IsAvailable { get; }

        /// <summary>Canonical decimal SteamID64 of the local user, or an empty string when unavailable.</summary>
        string LocalSteamId { get; }

        /// <summary>Steam persona name of the local user, or an empty string when unavailable.</summary>
        string LocalDisplayName { get; }

        /// <summary>The Steam App ID this client runs under, or 0 when unavailable.</summary>
        uint AppId { get; }

        /// <summary>Creates a lobby. <paramref name="onDone"/> receives the lobby id, or null on failure.</summary>
        void CreateLobby(SteamLobbyVisibility visibility, int maxMembers, Action<string?> onDone);

        /// <summary>Searches for lobbies whose metadata matches every entry of <paramref name="requiredMetadata"/>.</summary>
        void RequestLobbyList(IReadOnlyDictionary<string, string> requiredMetadata, Action<IReadOnlyList<SteamLobbySearchResult>> onDone);

        /// <summary>Joins a lobby. <paramref name="onDone"/> reports whether the join succeeded.</summary>
        void JoinLobby(string lobbyId, Action<bool> onDone);

        void LeaveLobby(string lobbyId);

        /// <summary>Writes lobby-level metadata. Owner only; returns false when the local user is not the owner.</summary>
        bool SetLobbyData(string lobbyId, string key, string value);

        /// <summary>Writes the local user own member data (for example <c>hw_ready</c>).</summary>
        void SetMemberData(string lobbyId, string key, string value);

        /// <summary>
        /// Current view of a lobby, or null when it is unknown. Member data only carries
        /// <c>hw_ready</c>, because Steam cannot enumerate member keys.
        /// </summary>
        SteamLobbySnapshot? GetLobby(string lobbyId);

        /// <summary>Opens the Steam overlay invite dialog for a lobby.</summary>
        void OpenInviteOverlay(string lobbyId);

        /// <summary>
        /// Requests a Web API auth ticket for identity <c>hexwars-match</c>. <paramref name="onDone"/>
        /// receives the uppercase hex ticket, or null on failure.
        /// </summary>
        void RequestAuthTicket(Action<string?> onDone);

        /// <summary>
        /// Cancels the outstanding auth ticket, if any, and drops its pending callback. A Web API
        /// ticket is good for one exchange, so the caller cancels it as soon as that exchange ends.
        /// </summary>
        void CancelAuthTicket();

        /// <summary>
        /// Handle of the auth ticket this client is currently waiting on, or 0 when there is none.
        /// Responses carrying any other handle belong to an abandoned request and are dropped.
        /// </summary>
        uint CurrentAuthTicketHandle { get; }

        /// <summary>Lobby metadata (or a member data entry) changed. The argument is the lobby id.</summary>
        event Action<string> LobbyDataChanged;

        /// <summary>A member entered a lobby. Arguments are the lobby id and the member SteamID64.</summary>
        event Action<string, string> MemberJoined;

        /// <summary>A member left, disconnected from, or was removed from a lobby.</summary>
        event Action<string, string> MemberLeft;

        /// <summary>The player accepted an invite (overlay join or +connect_lobby). The argument is the lobby id.</summary>
        event Action<string> InviteAccepted;

        /// <summary>Dispatches pending callbacks. Called once per frame on the main thread.</summary>
        void Pump();
    }
}

#nullable enable
using System;
using System.Collections.Generic;

namespace HexWars.Presentation
{
    /// <summary>Immutable view of one member of a Steam lobby.</summary>
    public sealed class SteamLobbyMemberSnapshot
    {
        public SteamLobbyMemberSnapshot(string? steamId, string? displayName, IReadOnlyDictionary<string, string>? data = null)
        {
            SteamId = steamId ?? string.Empty;
            DisplayName = displayName ?? string.Empty;
            Data = data ?? SteamLobbyCollections.EmptyData;
        }

        /// <summary>Canonical decimal SteamID64 of the member.</summary>
        public string SteamId { get; }

        /// <summary>Steam persona name, or an empty string when Steam has not cached it locally.</summary>
        public string DisplayName { get; }

        /// <summary>
        /// Per-member lobby data. Steam cannot enumerate member keys, so this only ever carries
        /// the keys HexWars asks for by name (today just <c>hw_ready</c>).
        /// </summary>
        public IReadOnlyDictionary<string, string> Data { get; }
    }

    /// <summary>Immutable view of a Steam lobby the local player is currently in.</summary>
    public sealed class SteamLobbySnapshot
    {
        public SteamLobbySnapshot(
            string? lobbyId,
            string? ownerSteamId,
            IReadOnlyList<SteamLobbyMemberSnapshot>? members = null,
            IReadOnlyDictionary<string, string>? metadata = null)
        {
            LobbyId = lobbyId ?? string.Empty;
            OwnerSteamId = ownerSteamId ?? string.Empty;
            Members = members ?? SteamLobbyCollections.EmptyMembers;
            Metadata = metadata ?? SteamLobbyCollections.EmptyData;
        }

        /// <summary>Canonical decimal SteamID64 of the lobby.</summary>
        public string LobbyId { get; }

        /// <summary>Canonical decimal SteamID64 of the lobby owner. The owner takes seat 0.</summary>
        public string OwnerSteamId { get; }

        public IReadOnlyList<SteamLobbyMemberSnapshot> Members { get; }

        /// <summary>Lobby-level metadata (the <c>hw_*</c> keys).</summary>
        public IReadOnlyDictionary<string, string> Metadata { get; }
    }

    /// <summary>One row of a lobby-list search result.</summary>
    public sealed class SteamLobbySearchResult
    {
        public SteamLobbySearchResult(string? lobbyId, IReadOnlyDictionary<string, string>? metadata = null, int memberCount = 0)
        {
            LobbyId = lobbyId ?? string.Empty;
            Metadata = metadata ?? SteamLobbyCollections.EmptyData;
            MemberCount = memberCount;
        }

        public string LobbyId { get; }

        public IReadOnlyDictionary<string, string> Metadata { get; }

        public int MemberCount { get; }
    }

    /// <summary>Shared empty collections so a snapshot never hands back null.</summary>
    internal static class SteamLobbyCollections
    {
        internal static readonly IReadOnlyDictionary<string, string> EmptyData =
            new Dictionary<string, string>(StringComparer.Ordinal);

        internal static readonly IReadOnlyList<SteamLobbyMemberSnapshot> EmptyMembers =
            Array.Empty<SteamLobbyMemberSnapshot>();
    }
}

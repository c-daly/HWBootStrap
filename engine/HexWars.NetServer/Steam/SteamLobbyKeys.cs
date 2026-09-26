namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// The lobby and member data keys HexWars writes into a Steam lobby. Both the Unity client and this
    /// server read them, so the names live in one place and are never spelled inline: a typo on either
    /// side would not fail a build, it would silently make every lobby look misconfigured.
    /// </summary>
    public static class SteamLobbyKeys
    {
        /// <summary>Lobby data: the App ID in decimal, so a lobby from another product is rejected.</summary>
        public const string App = "hw_app";

        /// <summary>Lobby data: the wire protocol version the host client speaks.</summary>
        public const string Protocol = "hw_protocol";

        /// <summary>Lobby data: the client build identifier (Application.version).</summary>
        public const string Build = "hw_build";

        /// <summary>Lobby data: quick-v1 or custom. See <see cref="SteamLobbyRules"/>.</summary>
        public const string Ruleset = "hw_ruleset";

        /// <summary>Lobby data: GameSetup.ToWire() as chosen by the lobby owner.</summary>
        public const string Setup = "hw_setup";

        /// <summary>Lobby data: the allocated match id, written by the owner after the server responds.</summary>
        public const string Match = "hw_match";

        /// <summary>Lobby data: the owner display name, for the client lobby list only.</summary>
        public const string Name = "hw_name";

        /// <summary>Member data: 1 when that member has pressed ready.</summary>
        public const string MemberReady = "hw_ready";

        /// <summary>The only value of <see cref="MemberReady"/> that counts as ready.</summary>
        public const string ReadyTrue = "1";
    }
}

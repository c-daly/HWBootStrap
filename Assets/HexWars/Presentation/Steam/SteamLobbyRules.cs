#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// The lobby metadata keys HexWars writes into Steam. The server validator reads exactly these
    /// names, so the two lists must stay identical.
    /// </summary>
    public static class SteamLobbyKeys
    {
        /// <summary>Lobby data: Steam App ID as a decimal string.</summary>
        public const string App = "hw_app";

        /// <summary>Lobby data: match protocol version as a decimal string.</summary>
        public const string Protocol = "hw_protocol";

        /// <summary>Lobby data: the owner client build (Application.version).</summary>
        public const string Build = "hw_build";

        /// <summary>Lobby data: <see cref="SteamLobbyRules.QuickRuleset"/> or <see cref="SteamLobbyRules.CustomRuleset"/>.</summary>
        public const string Ruleset = "hw_ruleset";

        /// <summary>Lobby data: the requested <see cref="GameSetup"/> wire line.</summary>
        public const string Setup = "hw_setup";

        /// <summary>Lobby data: the allocated server match id. Written by the owner after allocation.</summary>
        public const string Match = "hw_match";

        /// <summary>Lobby data: the owner display name, so a search result can be labelled.</summary>
        public const string Name = "hw_name";

        /// <summary>Member data: whether that member has readied up.</summary>
        public const string MemberReady = "hw_ready";

        /// <summary>The <see cref="MemberReady"/> value that counts as ready.</summary>
        public const string ReadyTrue = "1";

        /// <summary>The <see cref="MemberReady"/> value that counts as not ready.</summary>
        public const string ReadyFalse = "0";
    }

    /// <summary>
    /// Client twin of the server-side lobby rules: which ruleset Quick Match plays, what a compatible
    /// lobby looks like, and which metadata a lobby search must require.
    /// </summary>
    public static class SteamLobbyRules
    {
        /// <summary>The fixed Quick Match ruleset name.</summary>
        public const string QuickRuleset = "quick-v1";

        /// <summary>The ruleset name for a host-configured lobby.</summary>
        public const string CustomRuleset = "custom";

        /// <summary>Lowest board seed a lobby may advertise.</summary>
        public const int MinSeed = 1;

        /// <summary>Highest board seed a lobby may advertise.</summary>
        public const int MaxSeed = 9999;

        /// <summary>
        /// The <c>quick-v1</c> setup: 9x7 annihilation, no starting points, a three unit army of one of
        /// each role, three actions per turn, no fog. Only the seed varies, and it is clamped into
        /// <see cref="MinSeed"/>..<see cref="MaxSeed"/> so every advertised lobby is in range.
        /// </summary>
        public static GameSetup QuickMatchSetup(int seed)
        {
            return new GameSetup(GameMode.Annihilation, 9, 7, 0, ClampSeed(seed), 3, 1, 1, 1, 3, false);
        }

        /// <summary>Clamps a rolled seed into the advertised range.</summary>
        public static int ClampSeed(int seed)
        {
            if (seed < MinSeed) return MinSeed;
            if (seed > MaxSeed) return MaxSeed;
            return seed;
        }

        /// <summary>True when two setups describe the same match once both are sanitized.</summary>
        public static bool SetupEquals(GameSetup a, GameSetup b)
        {
            return string.Equals(a.Sanitized().ToWire(), b.Sanitized().ToWire(), StringComparison.Ordinal);
        }

        /// <summary>
        /// True when the lobby advertises this App ID and protocol version. The build id is deliberately
        /// not compared: the server decides which client builds interoperate.
        /// </summary>
        public static bool IsCompatible(IReadOnlyDictionary<string, string>? metadata, uint appId, int protocol)
        {
            if (metadata == null) return false;

            string? advertisedApp;
            if (!metadata.TryGetValue(SteamLobbyKeys.App, out advertisedApp)) return false;
            if (!string.Equals(advertisedApp, Decimal(appId), StringComparison.Ordinal)) return false;

            string? advertisedProtocol;
            if (!metadata.TryGetValue(SteamLobbyKeys.Protocol, out advertisedProtocol)) return false;
            return string.Equals(advertisedProtocol, Decimal(protocol), StringComparison.Ordinal);
        }

        /// <summary>The metadata every lobby-list search must require, so Steam filters server side.</summary>
        public static IReadOnlyDictionary<string, string> RequiredSearchMetadata(uint appId, int protocol, string ruleset)
        {
            return new Dictionary<string, string>(StringComparer.Ordinal)
            {
                { SteamLobbyKeys.App, Decimal(appId) },
                { SteamLobbyKeys.Protocol, Decimal(protocol) },
                { SteamLobbyKeys.Ruleset, ruleset ?? QuickRuleset },
            };
        }

        /// <summary>True when the lobby already carries an allocated server match.</summary>
        public static bool HasMatch(IReadOnlyDictionary<string, string>? metadata)
        {
            string? matchId;
            return metadata != null
                && metadata.TryGetValue(SteamLobbyKeys.Match, out matchId)
                && !string.IsNullOrEmpty(matchId);
        }

        internal static string Decimal(uint value)
        {
            return value.ToString(CultureInfo.InvariantCulture);
        }

        internal static string Decimal(int value)
        {
            return value.ToString(CultureInfo.InvariantCulture);
        }
    }
}

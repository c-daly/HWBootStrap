using HexWars.Engine;

namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// The two rulesets a Steam lobby may advertise and what quick-v1 means numerically. Quick Match is
    /// pinned here rather than trusted from the lobby: the client and the server must agree on the exact
    /// board, army and pace, so the only thing the owner really chooses in quick-v1 is the seed.
    /// </summary>
    public static class SteamLobbyRules
    {
        /// <summary>The fixed Quick Match ruleset: everything but the seed is server-defined.</summary>
        public const string QuickRuleset = "quick-v1";

        /// <summary>A lobby-authored setup, accepted after sanitizing rather than matched field by field.</summary>
        public const string CustomRuleset = "custom";

        public const int MinSeed = 1;

        /// <summary>Quick Match seeds are the 1..9999 the client picks, a narrower band than the engine clamp.</summary>
        public const int MaxSeed = 9999;

        /// <summary>The one setup a quick-v1 lobby is allowed to carry, for a given seed.</summary>
        public static GameSetup QuickMatchSetup(int seed) =>
            new GameSetup(GameMode.Annihilation, 9, 7, 0, seed, 3, 1, 1, 1, 3, false);

        /// <summary>
        /// Value equality over the sanitized wire form, so two setups that build the same match compare
        /// equal even when one of them arrived with out-of-range fields.
        /// </summary>
        public static bool SetupEquals(GameSetup a, GameSetup b) =>
            string.Equals(a.Sanitized().ToWire(), b.Sanitized().ToWire(), StringComparison.Ordinal);

        /// <summary>
        /// True when <paramref name="setup"/> is the quick-v1 setup for its own seed. The seed is read as
        /// supplied, NOT after Sanitized(): the engine clamp would turn a seed of 0 into 1 and quietly make
        /// an out-of-band setup look legitimate, and a quick-v1 lobby that did not come from our own client
        /// is exactly the thing this rule exists to catch.
        /// </summary>
        public static bool IsQuickMatchSetup(GameSetup setup) =>
            setup.Seed is >= MinSeed and <= MaxSeed && SetupEquals(setup, QuickMatchSetup(setup.Seed));

        /// <summary>True for the two ruleset names a lobby may advertise.</summary>
        public static bool IsKnownRuleset(string? ruleset) =>
            string.Equals(ruleset, QuickRuleset, StringComparison.Ordinal) ||
            string.Equals(ruleset, CustomRuleset, StringComparison.Ordinal);
    }
}

using System.Globalization;
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

        /// <summary>The exact number of fields GameSetup.ToWire writes.</summary>
        const int SetupFieldCount = 11;

        /// <summary>Index of the two enumerated fields, which have a range rather than a clamp.</summary>
        const int ModeIndex = 0;
        const int FogIndex = 10;

        /// <summary>
        /// Reads an hw_setup the way a lobby is allowed to write one: exactly the eleven fields ToWire
        /// emits, every one an invariant integer, with the two enumerated fields inside their range. On
        /// success <paramref name="setup"/> is already Sanitized and can go straight to GameFactory.
        ///
        /// This exists because GameSetup.Parse is deliberately lenient - it substitutes a default for
        /// every field it cannot read, which is right for a local lobby form and wrong for a value that
        /// arrived from the network. Parsing hw_setup directly would turn a garbage string into the
        /// default board and start a match neither player agreed to play.
        /// </summary>
        public static bool TryParseSetupStrict(string? wire, out GameSetup setup)
        {
            setup = default;
            if (string.IsNullOrEmpty(wire)) return false;

            var tokens = wire.Split(SetupSeparator, StringSplitOptions.None);
            if (tokens.Length != SetupFieldCount) return false;

            for (var i = 0; i < tokens.Length; i++)
            {
                if (!int.TryParse(
                        tokens[i], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var value))
                {
                    return false;
                }

                // Mode and fog are enumerations, not clamped numbers: the engine clamp would silently
                // turn a mode of 2 into Territory, which is not the game the lobby advertised.
                if ((i == ModeIndex || i == FogIndex) && value is not (0 or 1)) return false;
            }

            setup = GameSetup.Parse(wire).Sanitized();
            return true;
        }

        /// <summary>The single space GameSetup.ToWire puts between fields. Never a run of spaces: two in
        /// a row would produce an empty token, and an empty token is not an integer.</summary>
        const string SetupSeparator = " ";

        /// <summary>True for the two ruleset names a lobby may advertise.</summary>
        public static bool IsKnownRuleset(string? ruleset) =>
            string.Equals(ruleset, QuickRuleset, StringComparison.Ordinal) ||
            string.Equals(ruleset, CustomRuleset, StringComparison.Ordinal);
    }
}

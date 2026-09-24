using HexWars.NetServer.Configuration;
using HexWars.NetServer.Steam;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// The accounts this server will not serve, read from MATCH_BLOCKED_STEAM_IDS.
    ///
    /// It exists as a service rather than as a helper over the options object so there is exactly one
    /// canonicalisation rule. An id can be written with padding, with leading zeros, or copied out of a
    /// support ticket with a stray newline, and every one of those has to name the same account as the
    /// ticket the endpoint just authenticated. A comparison done at each call site would eventually be done
    /// two different ways, and the failure mode of that is a blocked player who is not actually blocked.
    ///
    /// The list is operator input and changes by redeploy. Render restarts the service when an environment
    /// variable changes, and for the Playtest that restart IS the admin path: a block takes effect within a
    /// deploy rather than immediately, and no in-flight match is interrupted by one. IOptionsMonitor is used
    /// so a host that does reload configuration picks the change up without a restart, but nothing in this
    /// deployment relies on that.
    /// </summary>
    public sealed class PlayerBlockList(IOptionsMonitor<MatchHostingOptions> options)
    {
        /// <summary>True for an account named on the block list, compared canonically.</summary>
        public bool IsBlocked(string? steamId)
        {
            string[] blocked = options.CurrentValue.BlockedSteamIds;
            if (blocked.Length == 0 || string.IsNullOrWhiteSpace(steamId)) return false;

            string canonical = Canonical(steamId);
            foreach (string entry in blocked)
            {
                if (string.Equals(Canonical(entry), canonical, StringComparison.Ordinal)) return true;
            }

            return false;
        }

        /// <summary>How many accounts are currently blocked. For the startup report and for a test that wants
        /// to know the list was read at all.</summary>
        public int Count => options.CurrentValue.BlockedSteamIds.Length;

        /// <summary>
        /// The canonical decimal SteamID64, or the trimmed input when it is not one.
        ///
        /// The fallback matters: an operator who mistypes an id must still get a list that refuses exactly
        /// what it says, rather than one entry silently matching everything or nothing in particular.
        /// </summary>
        static string Canonical(string steamId) =>
            SteamId64.TryNormalize(steamId, out string canonical) ? canonical : steamId.Trim();
    }
}

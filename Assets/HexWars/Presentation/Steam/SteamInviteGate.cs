#nullable enable

namespace HexWars.Presentation
{
    /// <summary>
    /// The one rule that decides whether an accepted Steam invite may open a lobby.
    /// <para>
    /// It is a pure function because getting it wrong is expensive and invisible: honouring an invite
    /// while a match socket is still up tears down a live game, and honouring one while a lobby screen
    /// is already open strands the coordinator that screen owns, leaving an empty Steam lobby and a
    /// live auth ticket behind. Neither shows up in a play session until a friend clicks Join at the
    /// wrong moment.
    /// </para>
    /// </summary>
    public static class SteamInviteGate
    {
        /// <summary>
        /// True only when the title screen really is the front door: no real game on screen (the demo
        /// does not count as one), no match socket, and no lobby screen already running a flow.
        /// </summary>
        /// <param name="hasState">A game state exists on the bootstrap.</param>
        /// <param name="demoMode">That state is the attract-mode demo, not a real match.</param>
        /// <param name="hasConnection">A match socket component is attached.</param>
        /// <param name="hasScreen">A lobby screen component is attached.</param>
        public static bool CanAccept(bool hasState, bool demoMode, bool hasConnection, bool hasScreen)
        {
            if (hasState && !demoMode) return false;
            if (hasConnection) return false;
            if (hasScreen) return false;
            return true;
        }
    }
}

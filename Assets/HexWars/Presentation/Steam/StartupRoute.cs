#nullable enable

namespace HexWars.Presentation
{
    /// <summary>
    /// Which front door a launched client opens.
    /// <para>
    /// This used to be an inline <c>#if UNITY_WEBGL</c> in <see cref=\"GameBootstrap\"/> that only ever
    /// forced the networked title for browser builds. A native Steam player therefore fell through to
    /// the editor hotseat game, because the checked-in scene has the <c>Networked</c> checkbox off, and
    /// neither the title screen nor <c>SteamRuntime</c> was ever created. The rule lives here, in plain
    /// C#, so that it is covered by a test rather than by a scene checkbox.
    /// </para>
    /// </summary>
    public static class StartupRoute
    {
        /// <summary>The three entry points a launched client can take.</summary>
        public enum Route
        {
            /// <summary>Build the local hotseat game immediately (the editor default).</summary>
            LocalGame,

            /// <summary>Demo plus title screen over the legacy room-code server flow.</summary>
            TitleWithLegacyNetwork,

            /// <summary>Demo plus title screen with Steam matchmaking behind it.</summary>
            TitleWithSteam,
        }

        /// <summary>
        /// Picks the entry point. A Steam build always reaches the title, whatever the scene says, and
        /// whatever a stray launch argument says, because Steam owns matchmaking there. Browser builds
        /// keep the legacy networked title (including its <c>?room=</c> auto-join), and so does any
        /// build whose scene asks for it or that was handed a room to join. Everything else is local.
        /// </summary>
        public static Route Decide(bool isWebGl, bool isSteamBuild, bool sceneNetworkedFlag, bool hasRoomQuery)
        {
            if (isSteamBuild) return Route.TitleWithSteam;
            if (isWebGl || sceneNetworkedFlag || hasRoomQuery) return Route.TitleWithLegacyNetwork;
            return Route.LocalGame;
        }

        /// <summary>
        /// True when the client should join the room from the page URL straight away instead of
        /// showing the title menu. Only the legacy network route ever does this.
        /// </summary>
        public static bool AutoJoinsRoom(Route route, bool hasRoomQuery)
        {
            return route == Route.TitleWithLegacyNetwork && hasRoomQuery;
        }
    }
}

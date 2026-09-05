#nullable enable
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Owns the process-wide <see cref="ISteamLobbyClient"/> and pumps its callbacks once per
    /// frame. This is the only MonoBehaviour in the Steam lobby layer, which keeps every other
    /// piece of it plain C# and unit testable.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SteamRuntime : MonoBehaviour
    {
#if HEXWARS_STEAM && !UNITY_WEBGL && !DISABLESTEAMWORKS
        /// <summary>True when Steamworks is compiled into this build.</summary>
        public static bool IsSteamBuild { get { return true; } }
#else
        /// <summary>True when Steamworks is compiled into this build.</summary>
        public static bool IsSteamBuild { get { return false; } }
#endif

        static SteamRuntime? _instance;
        static ISteamLobbyClient? _client;
        static bool _clientOverridden;
        static bool _quitting;

        /// <summary>The shared lobby client. Created on first use.</summary>
        public static ISteamLobbyClient Client
        {
            get
            {
                EnsureCreated();
                return _client!;
            }
        }

        /// <summary>
        /// The shared lobby client if one exists, without creating it. Teardown paths use this: an
        /// OnDestroy that runs during shutdown must not initialise Steam all over again just to
        /// unsubscribe from an event.
        /// </summary>
        public static ISteamLobbyClient? ClientIfCreated { get { return _client; } }

        /// <summary>
        /// Creates the lobby client, plus the per-frame pump object once the game is playing.
        /// Safe to call repeatedly.
        /// </summary>
        public static void EnsureCreated()
        {
            if (_quitting) return;   // the process is going away; never re-initialise Steam into that
            if (_client == null)
            {
                _client = new SteamLobbyClient(message => Debug.Log("[Steam] " + message));
                Debug.Log("[Steam] Lobby client created (steamBuild=" + IsSteamBuild + ", available=" + _client.IsAvailable + ").");
            }

            if (_clientOverridden || !Application.isPlaying) return;
            if (_instance != null) return;

            var host = new GameObject("HexWars.SteamRuntime");
            DontDestroyOnLoad(host);
            _instance = host.AddComponent<SteamRuntime>();
        }

        /// <summary>
        /// Installs a test double. A client this class created itself is disposed first; the
        /// injected one is left to the test to dispose.
        /// </summary>
        internal static void OverrideClientForTests(ISteamLobbyClient client)
        {
            if (!_clientOverridden && _client != null) _client.Dispose();
            _client = client;
            _clientOverridden = true;
            _quitting = false;   // a previous play session quitting must not poison the next one
        }

        void Update()
        {
            if (_client != null) _client.Pump();
        }

        void OnApplicationQuit()
        {
            _quitting = true;
            ReleaseClient();
        }

        void OnDestroy()
        {
            if (_instance == this) _instance = null;
            ReleaseClient();
        }

        static void ReleaseClient()
        {
            if (_clientOverridden || _client == null) return;
            _client.Dispose();
            _client = null;
            Debug.Log("[Steam] Lobby client disposed.");
        }
    }
}

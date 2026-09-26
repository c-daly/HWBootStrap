using System.Runtime.CompilerServices;

// The PlayMode suite drives real MonoBehaviours (SteamLobbyScreen, SteamRuntime) and needs their test
// seams. The EditMode suite reaches internals by reflection because it predates this file; new PlayMode
// tests call them directly, which keeps a rename a compile error instead of a runtime surprise.
[assembly: InternalsVisibleTo("HexWars.Presentation.PlayModeTests")]

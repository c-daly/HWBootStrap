# Steam client configuration

How the Unity client finds Steam and the match service, and how to run the two test suites that
cover the Steam lobby flow. Server-side deployment lives with the match service; everything here is
about the game client.

## The `HEXWARS_STEAM` define

`Assets/HexWars/Presentation/HexWars.Presentation.asmdef` declares a `versionDefines` entry that sets
`HEXWARS_STEAM` whenever the `com.rlabrecque.steamworks.net` package is present. Nothing sets it by
hand. `SteamRuntime.IsSteamBuild` is that define (minus WebGL and `DISABLESTEAMWORKS`), and it is the
single switch that decides which front door the player sees:

- **Steam builds** get Quick Match, Invite Friend and Host Game on the title screen. There is no room
  code row and no Browse Games button, because Steam owns matchmaking.
- **Every other build** keeps the existing server flow unchanged: Browse Games, Host Game, and
  join-by-room-code.

Removing the Steamworks.NET package therefore reverts the client to the server flow with no code
change.

## `steam_appid.txt` for editor runs

Outside Steam, `SteamAPI.Init` needs the App ID in a file next to the executable. For Play mode in
the editor that means the **project root** (the directory that holds `Assets/` and `ProjectSettings/`):

```
steam_appid.txt      # one line, the decimal App ID, no trailing newline required
```

The file is a local development aid and must not be committed. A shipped build takes its App ID from
Steam itself and ignores the file. Without it, `SteamRuntime.Client.IsAvailable` is false and the
lobby screen reports `Steam is unavailable - start the game through Steam.`

## Where the match-service URL comes from

`SteamMatchConfig.Resolve()` takes the first configured source, in this order:

1. **Command line** - `-hexwars-match-url https://match.example.com`. Passed to a built player or to
   the editor. Unavailable on WebGL, which is why the lookup is guarded.
2. **Environment** - `HEXWARS_MATCH_URL=https://match.example.com`. Convenient for CI and for local
   runs launched from a shell.
3. **Shipped asset** - `Assets/Resources/HexWarsSteamConfig.json`:

   ```json
   {
     "matchBaseUrl": "https://match.example.com",
     "protocolVersion": 2
   }
   ```

The asset ships with the literal placeholder `OWNER-INPUT`, which resolves to *not configured* on
purpose. An unconfigured build does not fail at the first request: the lobby screen opens, says
`Match service not configured`, and offers only Back. Set the real URL in the asset before cutting a
release build, or override it per run with the argument or the environment variable.

The URL must be an absolute `http`/`https` base with no trailing slash. `protocolVersion` is the
wire version the client speaks (2 today); it is advertised in lobby metadata so incompatible clients
never match each other.

`SteamMatchConfig.Resolve()` caches its answer for the process. Call `SteamMatchConfig.Invalidate()`
if you change the source at runtime, which is what the tests do.

## Running the suites

Both suites live in the Unity editor under **Window > General > Test Runner**.

- **EditMode** (`HexWars.Presentation.Tests`, `Assets/HexWars/Tests/Editor/`) covers the pure C#:
  `SteamLobbyCoordinatorTests`, `SteamMatchApiContractsTests`, `SteamMatchProtocolTests`,
  `FakeSteamLobbyClientTests`. Most of these files are also linked into
  `engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj`, so `dotnet test
  engine/HexWars.Engine.Tests` runs the same assertions without opening Unity - that is the fast loop.
- **PlayMode** (`HexWars.Presentation.PlayModeTests`, `Assets/HexWars/Tests/PlayMode/`) covers what
  only a running scene can: `SteamLobbyFlowSmokeTests` drives `SteamLobbyScreen` as a live
  MonoBehaviour through quick match, cancel, a match-service outage and a version mismatch. It needs
  the editor and cannot run headless from `dotnet`.

Neither suite touches Steam or the network. `SteamRuntime.OverrideClientForTests` injects
`FakeSteamLobbyClient` and `SteamLobbyScreen.ApiOverrideForTests` injects `FakeSteamMatchApi`, so no
App ID, `steam_appid.txt` or match URL is required to run tests.

The fakes are duplicated: `Assets/HexWars/Tests/Editor/Fakes/` for EditMode and
`Assets/HexWars/Tests/PlayMode/Fakes/` for PlayMode. Unity test assemblies cannot reference one
another, and only the EditMode copy is linked into the dotnet project. Keep the two in step when you
change either.

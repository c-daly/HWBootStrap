# Local multiplayer validation

No registered Steam App ID is needed for these checks. The durable verifier supplies the
development placeholder App ID `480000`, two synthetic accounts and a scripted Steam API.
The Unity EditMode and PlayMode suites likewise inject their existing fake Steam client.
Leave `Assets/Resources/HexWarsSteamConfig.json` at `OWNER-INPUT` until a real match endpoint
exists. No `steam_appid.txt`, publisher key or Steam login is required for this workflow.

## Run the server proof

Provide .NET 8 (the deployment runtime), a .NET SDK, and a **disposable PostgreSQL 16 database**.
The verifier drops and recreates that database's `public` schema. Its database-name guard rejects
unmarked targets; use a name such as `hexwars_multiplayer_test`. Bind a local database to loopback.

```sh
export HEXWARS_TEST_DATABASE_URL='postgres://test_user:test_password@127.0.0.1:55432/hexwars_multiplayer_test'
bash scripts/verify-multiplayer.sh
```

`HEXWARS_DOTNET_SDK` and `HEXWARS_DOTNET_RUNTIME` can select separate SDK/runtime executables.
The script builds Release, checks legacy WebSocket play/reconnection, then runs two durable games:

1. Two real HTTP/WebSocket clients allocate seats, submit barracks, move, attack and end a turn.
2. The child server exits gracefully (first run) or is forcibly terminated (second run).
3. A new process recovers the same PostgreSQL journal. Both clients reauthenticate and verify
   their seats, complete command history and position against an independent engine replay.
4. The clients finish the game. Every command must reach both sockets and occur exactly once,
   in sequence, in PostgreSQL. The stored completion status and winner must match.
5. Another process restart confirms both players can reconnect to the same terminal result.

Each child reports its PID and recovery count. A nonzero exit fails verification. The local server
binds only `127.0.0.1:5235`; the legacy check uses `127.0.0.1:5234`. Run sequentially against one
database. The internal `selftest-durable-host` command is controlled by its parent's stdin and
exits on EOF; it refuses unmarked databases and interactive invocation.

## Broader regression checks

```sh
dotnet test engine/HexWars.NetServer.Tests/HexWars.NetServer.Tests.csproj
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
```

Keep `HEXWARS_TEST_DATABASE_URL` set for the first command. These tests reset the schema too,
so run them after the verifier, using the same disposable database. The current server test project
targets .NET 10; the process verifier separately tests the shipped server on .NET 8.

In Unity 6000.5.0f1, run `HexWars.Presentation.Tests` in EditMode and
`HexWars.Presentation.PlayModeTests` in PlayMode. Both use placeholder identities. See
[client configuration](steam-client-configuration.md) for the test seams.

## What remains for Steam Playtest

This proves local server transport, persistence, recovery and scripted client flows. Actual Steam
ticket authentication, lobby discovery/invites, two-account interaction and deployed Render behavior
still require registered App IDs, configured secrets and live client validation. Local placeholders
do not establish those results. The production configuration continues to require real values.

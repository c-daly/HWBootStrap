# Steam Lobby, Render, and Durable Match Hosting Implementation Plan

**Date:** 2026-09-03  
**Status:** Proposed  
**Target:** Closed Steam Playtest  
**Production authority:** Revalidate the current `web-main` deployment lineage before implementation or deployment.

**Goal:** Replace the prototype room-discovery and client-minted identity path for the Steam build with Steam-hosted lobbies and authenticated players, keep HexWars authoritative gameplay on one paid Render instance, and make every started match recoverable from Postgres after a process restart or deploy.

**Architecture:** Steam remains authoritative for the transient waiting room: discovery, invitations, lobby membership, owner, ready state, and searchable lobby metadata. `HexWars.NetServer` verifies Steam identity and lobby membership through the Steamworks Web APIs, creates a durable match, runs the deterministic engine, and commits every accepted command to Postgres before broadcasting it. Live game state remains an in-memory projection of the durable start state plus command log. The first release intentionally uses one match-server instance; horizontal scaling is gated on externalized room routing and measured demand.

**Tech stack:** Unity 6000.5, Steamworks client SDK behind a local adapter, ASP.NET Core/.NET 8, PostgreSQL, Npgsql, Docker, paid Render Web Service, Render Postgres, Steamworks Web API.

## Authority references

- [Steam Matchmaking and Lobbies](https://partner.steamgames.com/doc/features/multiplayer/matchmaking)
- [Steam server-side Lobby Web API](https://partner.steamgames.com/doc/webapi/ILobbyMatchmakingService)
- [Steam user authentication and ownership](https://partner.steamgames.com/doc/features/auth?l=english)
- [Render WebSocket lifecycle](https://render.com/docs/websocket)
- [Render free-tier limitations](https://render.com/docs/free)
- [Render compute plans](https://render.com/docs/compute-plans)

## Decisions fixed by this plan

1. Steam Lobbies, not HexWars or Postgres, own pre-match discovery and membership.
2. Steam does not write to Postgres. The secure HexWars server calls `GetLobbyData`, verifies the returned owner/members/metadata, and persists only the match-boundary facts it needs.
3. Postgres becomes authoritative for started matches, seats, accepted command order, recovery, results, and server-issued join credentials.
4. The deterministic C# engine remains the only gameplay-rules authority.
5. One always-on paid Render instance hosts all matches for the initial Playtest.
6. WebGL retains the existing legacy lobby path during migration. Shared Steam/WebGL matchmaking is a later product decision.
7. Steam Community handles public discussion. General-purpose in-game chat, a custom forum, ranked matchmaking, and a game-server fleet are not part of this tranche.

## Current-state evidence

- `engine/HexWars.NetServer/Program.cs` serves both WebGL assets and `/ws`, keeps one static `MatchHub`, serializes it with one global lock, exposes a literal `/healthz`, and creates unbounded per-connection queues.
- `engine/HexWars.Engine/Net/MatchHub.cs` keeps rooms, token-to-seat assignments, start state, and accepted commands only in memory. Empty started rooms expire after ten minutes; any process replacement loses them immediately.
- `Assets/HexWars/Presentation/NetClient.cs` mints a 16-character PlayerPrefs seat token and places it in the WebSocket query string. Native builds fall back to localhost when `Application.absoluteURL` is absent.
- `Dockerfile` packages the WebGL files and match server into the same image.
- The current checkout contains no Steamworks integration, Npgsql dependency, Postgres schema, migration runner, or infrastructure-as-code definition for Render.
- The existing reconnect design already reconstructs state from a start state plus accepted command log. That is the persistence seam to preserve.

## Data authority matrix

| Data | Authority | Durable? | Notes |
|---|---|---:|---|
| Lobby ID, owner, membership and ready state | Steam | No | Fetch from Steam at match creation; do not continuously mirror it |
| Searchable game settings | Steam lobby metadata | No | Treat as requested settings; server sanitizes before use |
| Authenticated identity and ownership | Steam ticket verification | No | Exchange once for a HexWars join credential; never store the raw ticket |
| Match ID, seats, engine/build version and status | Postgres | Yes | Created transactionally from a verified lobby snapshot |
| Initial state and accepted commands | Postgres | Yes | Sufficient to reconstruct the deterministic game exactly |
| Live projected `GameState` and socket membership | Match-server memory | No | Rebuild from Postgres after restart |
| Saved armies/preferences | Local save/Steam Cloud initially | Yes | Separate from match hosting; not lobby metadata |
| Community posts and reports | Steam Community | Yes, by Steam | Do not copy into the game database |

## Non-negotiable invariants

- The server never trusts a client-supplied Steam ID, lobby member list, seat assignment, ready state, or unsanitized setup.
- The Steam publisher Web API key exists only in the Render secret store and server process.
- Raw Steam authentication tickets and join credentials never appear in URLs, logs, lobby metadata, analytics, crash reports, or Postgres.
- A command is not acknowledged or broadcast as accepted until its database append commits.
- `(match_id, sequence)` is unique; retries cannot apply a command twice.
- A fresh process can reproduce the exact current `GameState` from Postgres and the versioned engine contract.
- Steam lobby metadata contains identifiers and non-secret display/search data only.
- New match creation fails closed when Steam verification or Postgres is unavailable. Existing matches never silently advance without durable storage.
- Only one production match-server instance runs until room ownership/routing is externalized and tested.
- The existing WebGL deployment continues to function throughout migration unless a separately approved cutover removes it.

## Target flow

```text
Host Unity client       Steam             HexWars server            Postgres
       |                  |                       |                      |
       | CreateLobby      |                       |                      |
       |----------------->|                       |                      |
       | lobby callbacks  |                       |                      |
       |<-----------------|                       |                      |
       |                  |<--- second player --->|                      |
       |                  |                       |                      |
       | POST create-match(lobby ID, Steam ticket, requested setup)     |
       |------------------------------------------>|                      |
       |                  |  verify ticket         |                      |
       |                  |<---------------------->|                      |
       |                  |  GetLobbyData          |                      |
       |                  |<---------------------->|                      |
       |                  |                       | create match+seats   |
       |                  |                       |--------------------->|
       | match ID + opaque join credential        |                      |
       |<------------------------------------------|                      |
       | SetLobbyData(match ID, protocol)          |                      |
       |----------------->|                       |                      |
       | WSS connect, then AUTH as first frame     |                      |
       |------------------------------------------>|                      |
       | CATALOG                                  | persist catalog      |
       |------------------------------------------>|--------------------->|
       |                  after both catalogs: persist and send START    |
       |<------------------------------------------|--------------------->|
       | command                                  | append command      |
       |------------------------------------------>|--------------------->|
       |                  accepted broadcast only after commit           |
       |<------------------------------------------|                      |
```

---

### Task 0: Freeze prerequisites and environment authority

**Purpose:** Prevent implementation against the wrong deployment branch, wrong Steam App ID, or an unreviewed third-party wrapper.

**Files:**

- Create: `docs/operations/steam-render-environments.md`
- Create: `engine/HexWars.NetServer/Configuration/SteamOptions.cs`
- Create: `engine/HexWars.NetServer/Configuration/MatchHostingOptions.cs`
- Create: `engine/HexWars.NetServer.Tests/HexWars.NetServer.Tests.csproj`
- Test: `engine/HexWars.NetServer.Tests/ConfigurationTests.cs`

**External inputs requiring owner action:**

- Steamworks base-game App ID and Playtest App ID.
- Steamworks publisher Web API key with only the required product permissions.
- Render staging and production service identifiers.
- Render Postgres staging and production connection strings.
- Approved Steamworks Unity wrapper and its license/provenance record.
- Initial Playtest region and invited-player/concurrent-match target.

**Environment variables:**

```text
STEAM_APP_ID
STEAM_PUBLISHER_WEB_API_KEY
DATABASE_URL
MATCH_PUBLIC_BASE_URL
MATCH_JOIN_TOKEN_TTL_SECONDS
MATCH_BUILD_ID
MATCH_PROTOCOL_VERSION
ALLOWED_WEB_ORIGINS
LOBBY_PROVIDER=Legacy|Steam
```

- [ ] Fetch and inspect the latest `origin/web-main`; record the deployed commit and Docker image identity before branching.
- [ ] Confirm PR #10's immutable-container configuration-watcher fix is present in the actual production lineage.
- [ ] Record development, staging, Playtest, and production App IDs/endpoints without committing secrets.
- [ ] Choose the Steamworks Unity wrapper after checking license, current maintenance, Unity 6 compatibility, Windows x64 packaging, callback pumping, and headless-test seams.
- [ ] Create an explicit net8 `HexWars.NetServer.Tests` project with `IsTestProject=true`, pinned NUnit/test SDK dependencies, a reference to `HexWars.NetServer`, and a documented disposable-Postgres test strategy. Do not rely on solution-wide discovery of a generated Unity solution.
- [ ] Wrap configuration in validated options; production startup must fail when required Steam/Postgres values are absent or placeholder values are supplied.
- [ ] Ensure local and automated tests use fake Steam endpoints and disposable Postgres credentials.
- [ ] Commit boundary: environment contract and configuration validation only.

**Exit gate:** A clean server process can report its environment, App ID, build ID, protocol version, lobby provider, and database target without revealing a secret.

---

### Task 1: Add an append-only Postgres match store

**Purpose:** Make match creation, accepted commands, completion, and recovery durable without coupling the pure engine to Postgres.

**Files:**

- Modify: `engine/HexWars.NetServer/HexWars.NetServer.csproj` — add pinned Npgsql dependency.
- Create: `engine/HexWars.NetServer/Persistence/IMatchStore.cs`
- Create: `engine/HexWars.NetServer/Persistence/PostgresMatchStore.cs`
- Create: `engine/HexWars.NetServer/Persistence/MatchRecord.cs`
- Create: `engine/HexWars.NetServer/Persistence/Migrations/001_match_journal.sql`
- Create: `engine/HexWars.NetServer/Persistence/MigrationRunner.cs`
- Create: `engine/HexWars.NetServer.Tests/PostgresMatchStoreTests.cs`

**Minimum schema:**

```text
matches
  match_id UUID primary key
  steam_lobby_id TEXT null
  status TEXT
  setup_wire TEXT
  start_replay TEXT null
  engine_version TEXT
  protocol_version INTEGER
  build_id TEXT
  created_at/started_at/completed_at/last_activity_at TIMESTAMPTZ
  winner_seat INTEGER null

match_players
  match_id UUID
  steam_id TEXT
  seat INTEGER
  catalog_wire TEXT null
  joined_at/last_seen_at TIMESTAMPTZ
  primary key(match_id, steam_id)
  unique(match_id, seat)

match_commands
  match_id UUID
  sequence INTEGER
  command_wire TEXT
  accepted_at TIMESTAMPTZ
  issuer_steam_id TEXT
  primary key(match_id, sequence)

match_join_credentials
  credential_hash BYTEA primary key
  match_id UUID
  steam_id TEXT
  expires_at TIMESTAMPTZ
  revoked_at TIMESTAMPTZ null

partial unique index
  steam_lobby_id where status in ('waiting', 'active')
```

- [ ] Write integration tests against disposable Postgres for migration idempotence and constraints.
- [ ] Store Steam IDs as canonical decimal strings to avoid unsigned-64-bit conversion mistakes across C#, SQL, JavaScript, and JSON.
- [ ] Implement idempotent `CreateMatchForLobby`: the same Steam lobby can never create two waiting/active matches, while a completed match does not permanently prevent an explicitly requested rematch.
- [ ] Implement atomic append with the next sequence number and a unique constraint protecting retries.
- [ ] Persist each seat's normalized `CATALOG` payload before constructing the start state; a restart while waiting for the other player's catalog must not lose the first submission.
- [ ] Implement load of start state plus ordered accepted commands.
- [ ] Implement status transitions with explicit allowed edges: `waiting -> active -> completed|expired|abandoned`.
- [ ] Add a retention decision before launch: active/unfinished retention, completed replay retention, deletion/export policy, and backup retention.
- [ ] Never store raw Steam tickets or plaintext join credentials.
- [ ] Commit boundary: schema, store, and Postgres integration tests.

**Exit gate:** A match written by process A can be loaded by a fresh process B, replayed through the authoritative engine, and produce an exactly equivalent state and command sequence.

---

### Task 2: Implement secure Steam server APIs

**Purpose:** Allow the backend—not the client—to verify Steam identity, ownership, lobby owner, membership, ready state, and metadata.

**Files:**

- Create: `engine/HexWars.NetServer/Steam/ISteamWebApiClient.cs`
- Create: `engine/HexWars.NetServer/Steam/SteamWebApiClient.cs`
- Create: `engine/HexWars.NetServer/Steam/SteamLobbySnapshot.cs`
- Create: `engine/HexWars.NetServer/Steam/SteamIdentity.cs`
- Create: `engine/HexWars.NetServer/Steam/SteamApiException.cs`
- Create: `engine/HexWars.NetServer.Tests/SteamWebApiClientTests.cs`
- Create: `engine/HexWars.NetServer.Tests/SteamLobbyValidationTests.cs`

**Required Steam calls:**

- `ISteamUserAuth/AuthenticateUserTicket` — exchange a client ticket for a verified Steam ID and ownership result.
- `ILobbyMatchmakingService/GetLobbyData` — fetch the lobby owner, members, lobby metadata, and member metadata.
- `ILobbyMatchmakingService/CreateLobby` and `RemoveUserFromLobby` remain optional; use them only if a server-owned/private-unique lobby is deliberately chosen later.

- [ ] Use typed `HttpClient` instances with bounded connect/request timeouts and cancellation.
- [ ] Parse 64-bit identifiers as strings at the API boundary.
- [ ] Reject wrong App ID, invalid/expired ticket, missing ownership, nonmember, nonowner host request, wrong member count, missing ready state, and incompatible protocol/build metadata.
- [ ] Sanitize all lobby-selected settings through `GameSetup.Sanitized()` before persistence or engine construction.
- [ ] Add bounded retry with jitter only for safe Steam reads; do not retry a state-changing operation blindly.
- [ ] Redact query strings, publisher keys, Steam tickets, and credential bodies from logs.
- [ ] Classify failures into player-safe results: authentication failed, lobby changed, service temporarily unavailable, incompatible version.
- [ ] Use a fake `HttpMessageHandler` in tests; no automated test calls Valve's live API.
- [ ] Commit boundary: Steam API adapter and validation only.

**Exit gate:** Given a lobby ID and requester ticket, the backend can prove the requester's Steam ID, ownership, lobby membership, owner status where required, both players' readiness, and sanitized match settings without trusting a client-supplied roster.

---

### Task 3: Add lobby-bound match creation and join credentials

**Purpose:** Convert a verified transient Steam lobby into one durable HexWars match and two authenticated seats.

**Files:**

- Create: `engine/HexWars.NetServer/Endpoints/SteamMatchEndpoints.cs`
- Create: `engine/HexWars.NetServer/Auth/IMatchCredentialService.cs`
- Create: `engine/HexWars.NetServer/Auth/MatchCredentialService.cs`
- Create: `engine/HexWars.NetServer/Contracts/CreateSteamMatchRequest.cs`
- Create: `engine/HexWars.NetServer/Contracts/CreateSteamMatchResponse.cs`
- Create: `engine/HexWars.NetServer/Contracts/JoinSteamMatchRequest.cs`
- Create: `engine/HexWars.NetServer/Contracts/JoinSteamMatchResponse.cs`
- Modify: `engine/HexWars.NetServer/Program.cs`
- Create: `engine/HexWars.NetServer.Tests/SteamMatchEndpointTests.cs`

**API surface:**

```text
POST /api/v1/steam/matches
  input: steamLobbyId, requester Steam ticket, requested setup
  requires: requester is verified lobby owner; exactly two verified ready members
  output: matchId, protocolVersion, websocketUrl, requester's opaque join credential

POST /api/v1/steam/matches/{matchId}/join
  input: requester Steam ticket
  requires: verified Steam ID occupies a persisted seat
  output: matchId, websocketUrl, opaque join credential
```

- [ ] Make create idempotent by Steam lobby ID; retries return the same durable match rather than creating another.
- [ ] Assign seats deterministically and persist both verified Steam IDs in the same transaction as the match.
- [ ] Generate at least 256 bits of random credential material, store only its SHA-256 hash, bind it to match and Steam ID, and give it an explicit expiry/revocation policy.
- [ ] Rate-limit ticket verification, match creation, join exchange, and failed authentication by IP plus stable verified identity where available.
- [ ] Return the credential only over HTTPS; never put it in Steam lobby metadata.
- [ ] Put only `match_id`, `protocol_version`, and non-secret presentation/search values in lobby metadata.
- [ ] Preserve the existing `/games` and legacy query-token `/ws` route for WebGL behind `LOBBY_PROVIDER=Legacy` during migration.
- [ ] Commit boundary: HTTP match allocation and credential exchange.

**Exit gate:** A lobby owner can create exactly one match; either verified member can obtain only their own seat credential; a nonmember, replayed invalid ticket, wrong App ID, or forged roster cannot create or join it.

---

### Task 4: Make accepted gameplay durable and restartable

**Purpose:** Ensure a deploy, crash, or host replacement cannot erase or fork a started match.

**Files:**

- Create: `engine/HexWars.NetServer/Runtime/DurableMatchCoordinator.cs`
- Create: `engine/HexWars.NetServer/Runtime/LiveMatch.cs`
- Create: `engine/HexWars.NetServer/Runtime/MatchRecoveryService.cs`
- Modify: `engine/HexWars.NetServer/Program.cs`
- Modify as needed: `engine/HexWars.Engine/Net/GameSession.cs`
- Modify as needed: `engine/HexWars.Engine/Net/MatchHub.cs`
- Extend: `engine/HexWars.NetServer/SelfTest.cs`
- Create: `engine/HexWars.NetServer.Tests/DurableMatchCoordinatorTests.cs`
- Create: `engine/HexWars.NetServer.Tests/MatchRestartRecoveryTests.cs`

**Protocol-v2 connection:**

- Connect to `wss://<host>/ws/v2` without credentials in the URL.
- Send `AUTH <match-id> <opaque-credential>` as the first application frame.
- Do not seat the socket or accept gameplay messages until authentication succeeds.
- Preserve protocol v1 only for the legacy WebGL path during migration.

- [ ] Replace the single global gameplay lock for protocol v2 with per-match serialization.
- [ ] Separate “evaluate command” from “publish accepted command” so the database commit occurs before any accepted acknowledgement/broadcast.
- [ ] On database failure, do not broadcast; discard or rebuild the uncommitted in-memory projection from the persisted log and return a temporary failure.
- [ ] Record server-assigned command sequence numbers and make retries idempotent.
- [ ] Persist each authenticated seat's normalized barracks catalog. After both are present, build the authoritative start state once and atomically persist `start_replay` plus the `waiting -> active` transition before sending `START`.
- [ ] On startup or first connection, load the match's start replay and ordered commands, replay them through the exact compatible engine, and verify the resulting sequence/state.
- [ ] Refuse recovery when the stored engine/protocol contract is not supported; surface an explicit maintenance/recovery error instead of silently using new semantics.
- [ ] Add bounded per-connection outbound queues with a documented slow-client policy.
- [ ] Add server heartbeat and stale-connection cleanup.
- [ ] Complete and persist a match before broadcasting its final accepted command/result.
- [ ] Extend self-test to start a match, accept commands, stop the first server, start a fresh server against the same database, reconnect both seats, and prove exact continuation.
- [ ] Commit boundary: durable gameplay, protocol v2, and restart recovery.

**Exit gate:** Killing the server at every command boundary and starting a fresh process cannot lose, duplicate, reorder, or alter an acknowledged action.

---

### Task 5: Add the Unity Steam lobby adapter and player flow

**Purpose:** Give the native Steam build Quick Match, Host Private, Invite Friend, ready-state, cancellation, match allocation, and authenticated reconnection without coupling presentation code directly to one Steam wrapper.

**Files:**

- Create: `Assets/HexWars/Presentation/Steam/ISteamLobbyClient.cs`
- Create: `Assets/HexWars/Presentation/Steam/SteamLobbyClient.cs`
- Create: `Assets/HexWars/Presentation/Steam/SteamLobbyCoordinator.cs`
- Create: `Assets/HexWars/Presentation/Steam/SteamLobbySnapshot.cs`
- Create: `Assets/HexWars/Presentation/Steam/SteamMatchApiClient.cs`
- Create: `Assets/HexWars/Presentation/Steam/SteamMatchConnection.cs`
- Modify: `Assets/HexWars/Presentation/TitleScreen.cs`
- Modify: `Assets/HexWars/Presentation/GameBrowser.cs`
- Modify: `Assets/HexWars/Presentation/NetClient.cs`
- Modify: `Assets/HexWars/Presentation/SetupForm.cs`
- Create: `Assets/HexWars/Tests/Editor/SteamLobbyCoordinatorTests.cs`
- Create: `Assets/HexWars/Tests/Editor/SteamMatchApiClientTests.cs`
- Create: `Assets/HexWars/Tests/PlayMode/HexWars.Presentation.PlayModeTests.asmdef`
- Create: `Assets/HexWars/Tests/PlayMode/SteamLobbyFlowSmokeTests.cs`

**Player-visible paths:**

- `Quick Match` — search one canonical two-player rules queue, join a compatible lobby, or create one when none exists.
- `Invite Friend` — create a private/friends lobby and open the Steam invitation overlay.
- `Host Game` — create a lobby with sanitized searchable setup metadata.
- `Cancel` — leave the lobby and cancel any outstanding API/matchmaking operation cleanly.
- `Ready` — publish member readiness; the verified owner requests match creation only after both members are ready.
- `Reconnect` — exchange a fresh Steam ticket for the persisted seat and reconnect to protocol v2.

- [ ] Put all Steam SDK calls and callbacks behind `ISteamLobbyClient`; EditMode tests use a deterministic fake.
- [ ] Pump Steam callbacks on the Unity main thread and make scene changes/disposal unregister every callback.
- [ ] Define one initial Quick Match ruleset to avoid splitting a small player population across map/mode/fog/pace queues.
- [ ] Store lobby metadata keys for App ID/build/protocol/ruleset/setup/match ID; reject incompatible lobbies before joining.
- [ ] Use Steam lobby member callbacks for responsive UI, but let the backend re-fetch `GetLobbyData` before allocating the match.
- [ ] Exchange Steam tickets via HTTPS and keep the resulting match credential only in memory unless an explicitly encrypted reconnect cache is approved.
- [ ] Show specific states: searching, lobby found, waiting for player, waiting for ready, allocating server match, connecting, reconnecting, Steam unavailable, backend unavailable, version mismatch.
- [ ] Preserve the current WebGL title/browser path through a platform/lobby-provider switch; never attempt to initialize Steamworks in WebGL.
- [ ] Verify keyboard/mouse navigation, cancellation at every wait state, Steam overlay return, and duplicate callback resistance.
- [ ] Commit boundary: native Steam lobby and connection UX.

**Exit gate:** Two Steam clients can discover/invite, ready, receive verified seats, complete a match, disconnect/reconnect, rematch, and return to the title without using `/games` or client-minted seat tokens.

---

### Task 6: Productionize the Render deployment

**Purpose:** Remove free-tier cold starts, separate static delivery from match hosting, make service health meaningful, and ensure deploys drain or restore active games safely.

**Files:**

- Create: `render.yaml`
- Modify: `Dockerfile`
- Create: `engine/HexWars.NetServer/Operations/ServiceReadiness.cs`
- Create: `engine/HexWars.NetServer/Operations/GracefulShutdownService.cs`
- Create: `engine/HexWars.NetServer/Operations/MatchMetrics.cs`
- Modify: `engine/HexWars.NetServer/Program.cs`
- Create: `docs/operations/render-steam-playtest-runbook.md`
- Create: `docs/operations/match-recovery-runbook.md`

**Initial Render topology:**

```text
Static Site/CDN: www + optional WebGL build
Paid Web Service: 0.5 CPU / 512 MB starting point, exactly one instance
Managed Postgres: same region, private connection, backups enabled
Staging Web Service + staging database: separate secrets and Steam App ID
```

- [ ] Upgrade the match service from free to paid compute; do not rely on synthetic keepalive traffic to avoid free-tier sleep.
- [ ] Keep exactly one instance until the horizontal-scaling gate is complete.
- [ ] Split static WebGL output from the server image, or document a deliberately deferred split with measured bandwidth/deploy impact.
- [ ] Add `/health/live` for process liveness and `/health/ready` for database/schema/startup/recovery readiness.
- [ ] Mark readiness false before shutdown, stop new match creation, finish in-flight durable commits, persist active projections, notify clients, and close WebSockets with a retryable service-restart reason.
- [ ] Preserve `DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE=false` in the immutable production image.
- [ ] Configure custom TLS domains, WSS endpoint, allowed WebGL origins, service region, deployment branch, spend alerts, and log retention.
- [ ] Record schema migration and application rollback procedures. Application rollback must remain compatible with the current database schema.
- [ ] Add an external uptime check for readiness and alerts for process restart, database errors, Steam API errors, reconnect spikes, and match-recovery failures.
- [ ] Commit boundary: Render topology, health, shutdown, metrics, and runbooks.

**Exit gate:** A staging deployment can be replaced while matches are active; every client reconnects to the restored match, and no acknowledged command disappears or applies twice.

---

### Task 7: Security, privacy, and operational controls

**Purpose:** Establish the minimum safe public-service boundary before strangers can create traffic and persistent player records.

**Files:**

- Create: `docs/operations/multiplayer-data-inventory.md`
- Create: `docs/operations/multiplayer-threat-model.md`
- Create: `docs/operations/multiplayer-incident-response.md`
- Modify: server logging/configuration files identified during implementation.

- [ ] Inventory every persisted field, purpose, retention period, access path, deletion path, and backup copy.
- [ ] Document that Steam lobby/chat/community data is not copied into Postgres except for match-bound lobby and player identifiers.
- [ ] Use least-privilege database credentials and a separate migration credential if Render permits it.
- [ ] Rotate the Steam publisher key and match credential material in staging before launch; record the production rotation procedure.
- [ ] Add request/body size limits, connection limits, match-creation limits, and bounded queues.
- [ ] Verify origin/CORS policy for WebGL while recognizing that native clients do not provide browser-origin protection.
- [ ] Ensure logs use match/lobby correlation IDs and redacted or hashed player identifiers where full Steam IDs are unnecessary.
- [ ] Define ban/block/report administration for the game service while leaving discussion moderation to Steam Community.
- [ ] Document database outage, Steam outage, Render outage, compromised-key, abusive-client, and bad-deploy response procedures.
- [ ] Commit boundary: public-service security and operations baseline.

**Exit gate:** A reviewer can trace what player data exists, why it exists, how it is protected/deleted, and how an operator responds to the expected failure classes.

---

### Task 8: Verification and Playtest promotion gate

**Purpose:** Prove the full system, not merely individual adapters, before changing the Steam store feature claims or inviting unrestricted testers.

**Automated gates:**

- [ ] Run the full engine suite with explicit test discovery; a zero-test success is a failure.
- [ ] Run NetServer unit and disposable-Postgres integration suites.
- [ ] Run Steam API tests against fakes covering success, invalid ticket, wrong App ID, nonmember, owner change, lobby change during allocation, rate limit, timeout, malformed response, and Valve outage.
- [ ] Run Unity full EditMode and new PlayMode suites.
- [ ] Run the server self-test with process restart and Postgres recovery.
- [ ] Build the engine DLL and Unity player in one job; verify hashes before packaging.
- [ ] Build and install Windows x64 through a private Steam branch on a clean machine.

**Staging hands-on gates:**

- [ ] Two separate Steam accounts complete Quick Match and Invite Friend flows.
- [ ] A third/nonmember account cannot obtain a seat.
- [ ] Both players reconnect after client termination, server restart, Render deploy, and brief network loss.
- [ ] An existing match continues when Steam lobby lookup is temporarily unavailable after allocation.
- [ ] New allocation fails clearly and safely while Steam or Postgres is unavailable.
- [ ] A rollback does not corrupt or orphan matches created by the immediately newer deployment.
- [ ] WebGL legacy hosting still launches, creates/joins, and completes a match if it remains an advertised surface.
- [ ] No URL or captured log contains Steam tickets, publisher keys, raw join credentials, or legacy production seat tokens.

**Provisional capacity gate for the first Playtest:**

- [ ] Agree the invite cap before running the test. Initial test target: 100 concurrent matches / 200 WebSockets on the chosen paid instance.
- [ ] Sustain the target for 30 minutes with realistic command pacing plus reconnect churn.
- [ ] Demonstrate zero lost/duplicate accepted commands and zero unrecoverable matches.
- [ ] Record CPU, working set, database pool use, command commit latency, accept-to-broadcast latency, outbound queue depth, reconnect rate, and bandwidth.
- [ ] Use evidence to stay on the current compute plan or resize it; do not infer capacity from plan specifications alone.

**Promotion gate:**

- [ ] Product owner signs off on the Steam Playtest invite cap, service region, monthly spend ceiling, retention policy, support coverage, maintenance messaging, and rollback owner.
- [ ] Publish the exact server build ID, Unity build ID, engine hash, schema version, protocol version, and Steam branch used for the Playtest.
- [ ] Keep the public store claims limited to the multiplayer paths demonstrated by this candidate.

## Horizontal-scaling gate — deliberately deferred

Do not add a second match-server instance merely because Render exposes a scaling control. Render can assign reconnecting WebSockets to a different instance, while the current service has process-local rooms.

Horizontal scaling becomes a separate plan only when measured demand exceeds vertical capacity. Its prerequisites are:

- external room-to-instance ownership with leases and fencing;
- a distributed lobby/match directory;
- an explicit routing strategy for new and reconnecting sockets;
- bounded cross-instance messaging or deterministic recovery from Postgres;
- no global lock spanning unrelated matches;
- failure tests for split brain, lease expiry, instance death, and rolling deployment; and
- a capacity/cost comparison showing that a second instance is preferable to a larger single instance.

Redis or another coordination service is optional until that gate is opened. Postgres alone is sufficient for the one-instance Playtest design.

## Implementation order and dependencies

```text
Task 0: authority/configuration
        |
Task 1: Postgres journal
        |
Task 2: Steam verification
        |
Task 3: lobby -> durable match allocation
        |
Task 4: durable command/restart path
        |
Task 5: Unity Steam flow
        |
Task 6: paid Render staging/production
        |
Task 7: security and operations
        |
Task 8: complete-system promotion evidence
```

No production provider change, paid-plan purchase, Steam secret creation, deployment, branch merge, or store-feature change is implied by this plan document. Those external actions require explicit owner approval at their respective gates.

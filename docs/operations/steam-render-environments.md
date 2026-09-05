# Steam / Render environment authority

## 1. Purpose and status

This document is the environment contract for the HexWars Steam Playtest match service. It names every
environment the match server runs in, every configuration value it reads, which of those values are secret,
and which values only the project owner can supply. Where any other document disagrees about an environment
name, a variable name, a default, or a secret classification, this document wins. It is the reference that
the server configuration binding, the startup validation, and the deployment runbook are all written
against. **Status: Proposed.** **Date: 2026-09-04.** Implementation plan:
[Steam Lobby, Render, and Durable Match Hosting](../superpowers/plans/2026-09-03-steam-lobby-render-postgres-production.md).

## 2. Deployment lineage (verified 2026-09-04)

The repository has two long-lived branches that reach production, and they do not hold the same code. This
effort branches from `main`, but the live WebGL service deploys from `web-main`. Confusing the two is the
most likely way to break a working deployment during Steam cutover.

| Fact | Value |
|---|---|
| `origin/main` tip | `8e245f3` \u2014 "Merge pull request #11 from c-daly/codex/tactical-v3-safe-stop-retry" |
| `origin/web-main` tip | `9535800` \u2014 "Merge pull request #10 from c-daly/codex/fix-render-inotify-reload" (merged 2026-09-01T21:35:02Z) |
| `web-main` Dockerfile | Contains `ENV DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE=false`, the PR #10 immutable-container configuration-watcher fix |
| `main` Dockerfile | Does **not** contain that line |
| Other `web-main` vs `main` differences | Only the committed WebGL build artefacts under `engine/HexWars.NetServer/wwwroot/`, and `web-main` lacking the newer Python ML-lab work |
| Deployed Docker image identity / Render deploy id | `OWNER-INPUT` |

**Consequences.**

- This effort branches from `main`; the integration branch is `steam-hosting/integration`.
- The `main` Dockerfile must carry the `DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE=false` line forward. That
  is done in the Render productionisation task, not here.
- Production cutover must not regress the WebGL service, which deploys from `web-main`. Any change to the
  Render service that serves WebGL is a separate, explicitly approved step.
- The deployed image identity is not derivable from the repository. Record the Render deploy commit SHA and
  image digest in the table above **before** the first Steam deployment, so a regression can be bisected
  against a known-good artefact.

## 3. Environments

- **Development** \u2014 a developer machine. Steam calls are pointed at a fake endpoint; Postgres runs locally in
  Docker. Never uses Render credentials.
- **Automated test** \u2014 `dotnet test`. Steam is a fake `HttpMessageHandler`; Postgres is disposable
  (Testcontainers `postgres:16-alpine`, or `HEXWARS_TEST_DATABASE_URL` where Docker is unavailable). Tests
  never call the live Valve API and never use Render credentials.
- **Staging** \u2014 a Render web service plus Render Postgres, using the Steam **Playtest** App ID.
- **Production** \u2014 a Render web service plus Render Postgres, using the **base-game** App ID.

| Environment | Steam App ID | Render service id | Postgres | Public base URL | `LOBBY_PROVIDER` | Secrets location |
|---|---|---|---|---|---|---|
| Development | fake, `480000`-style | none (developer machine) | local Docker `postgres:16-alpine` | `http://localhost:5000` | `Legacy`, or `Legacy,Steam` with a fake Steam endpoint | developer shell / user secrets; never committed |
| Automated test | fake, `480000`-style | none (in-process test host) | disposable Testcontainers `postgres:16-alpine`, or `HEXWARS_TEST_DATABASE_URL` | in-process test host | set per test | none; fakes only |
| Staging | `OWNER-INPUT` (Playtest App ID) | `OWNER-INPUT` | `OWNER-INPUT` (connection target, never the connection string) | `OWNER-INPUT` | `Legacy,Steam` | Render environment group / secret store only |
| Production | `OWNER-INPUT` (base-game App ID) | `OWNER-INPUT` | `OWNER-INPUT` (connection target, never the connection string) | `OWNER-INPUT` | `Legacy,Steam` | Render environment group / secret store only |

Connection strings are never written into this repository, a plan document, a ticket, or a chat message.
They exist only in the Render secret store and in the server process that reads them.

## 4. Environment variable contract

The types and defaults below are binding for the server implementation. Names are flat environment variable
names, bound by the server configuration layer.

| Variable | Type | Default | Required | Secret | Semantics |
|---|---|---|---|---|---|
| `STEAM_APP_ID` | uint | none | when Steam is enabled, or in Production | no | The Steamworks App ID that tickets and lobby metadata are validated against. |
| `STEAM_PUBLISHER_WEB_API_KEY` | string | none | when Steam is enabled, or in Production | **YES** | Publisher Web API key. Exists only in the Render secret store and the server process. Never logged, never returned by any endpoint. |
| `STEAM_WEB_API_BASE_URL` | URL | `https://partner.steam-api.com` | no | no | Base URL for Steamworks Web API calls. Tests point this at a fake endpoint. |
| `DATABASE_URL` | `postgres://` URI, or an Npgsql keyword/value string | none | when Steam is enabled, or in Production | **YES** | Postgres connection target. Only `host:port/db` may ever be echoed back; credentials never are. |
| `MATCH_PUBLIC_BASE_URL` | absolute URL, https in Production, no credentials, query, or fragment | none | when Steam is enabled, or in Production | no | The externally reachable base URL. The server derives the websocket URL from it by mapping `http` to `ws` and `https` to `wss`, then appending `/ws/v2`. Startup rejects a value carrying userinfo, a query string, or a fragment, because this value is echoed into the environment report and logged; the report renders scheme, authority and path only. |
| `MATCH_JOIN_TOKEN_TTL_SECONDS` | int | `900` | no | no | Lifetime of an issued join credential. Valid range 60..86400. |
| `MATCH_BUILD_ID` | string | none | yes | no | Identifies the running build. On Render, set it from `RENDER_GIT_COMMIT`. |
| `MATCH_PROTOCOL_VERSION` | int | `2` | no | no | Wire protocol version advertised to clients and stored on each match. |
| `MATCH_HEARTBEAT_SECONDS` | int | `20` | no | no | How often `/ws/v2` sends `PING` on every authenticated socket. Valid range 1..300. It is the only traffic on an idle match, so it is also the only thing keeping an intermediary from dropping a socket both ends still believe in. |
| `MATCH_STALE_CONNECTION_SECONDS` | int | `60` | no | no | Silence after which an authenticated `/ws/v2` socket is closed with 1001. Valid range 2..900, and it must be greater than `MATCH_HEARTBEAT_SECONDS`: a window no longer than the ping cadence judges silence over an interval the client was never given a chance to answer in. Startup fails if it is not. |
| `MATCH_OUTBOUND_QUEUE_CAPACITY` | int | `256` | no | no | Frames one `/ws/v2` connection may have waiting before it is closed with 1008 `slow client`. Valid range 16..4096. The bound is what stops a client that has stopped reading from being paid for by every other match on the host; see `docs/operations/protocol-v2.md` §5. |
| `MATCH_AUTH_TIMEOUT_SECONDS` | int | `10` | no | no | How long a freshly accepted `/ws/v2` socket has to send its `AUTH` frame before it is closed with 1008. Valid range 1..120. |
| `ALLOWED_WEB_ORIGINS` | comma list | empty | no | no | Browser origins permitted on the legacy WebGL routes. |
| `LOBBY_PROVIDER` | comma list of `Legacy`, `Steam` | `Legacy` | no | no | Which lobby surfaces are mapped. `Legacy` maps `/games` and `/ws`; `Steam` maps `/api/v1/steam/*` and `/ws/v2`. |
| `MATCH_COMPATIBLE_CLIENT_BUILDS` | comma list | empty | no | no | Accepted client build strings. Empty means any client build is accepted; the protocol version must still match. |
| `MATCH_TRUST_FORWARDED_HEADERS` | bool | `false` | no | no | Honour forwarded-for headers. Set it `true` on Render so rate limiting sees the real client IP rather than the proxy. Read the note below before setting it. |
| `MATCH_TRUSTED_PROXY_CIDRS` | comma list of IP addresses or CIDR ranges | empty | no | no | Whose forwarded-for header this server believes. Only consulted when `MATCH_TRUST_FORWARDED_HEADERS` is `true`. Each entry must be an IPv4 or IPv6 address, optionally with a prefix length; an entry that does not parse fails startup. |
| `MATCH_TRUST_ALL_PROXIES` | bool | `false` | when `MATCH_TRUST_FORWARDED_HEADERS` is `true` and `MATCH_TRUSTED_PROXY_CIDRS` is empty | no | Confirms that trusting every peer to name the client is deliberate. Render does not publish its proxy addresses, so `render.yaml` sets this `true` alongside `MATCH_TRUST_FORWARDED_HEADERS`. Startup fails if forwarded headers are trusted with neither a proxy list nor this acknowledgement. |
| `MATCH_BLOCKED_STEAM_IDS` | comma list of SteamID64 | empty | no | no | Accounts refused at match create and join. |
| `MATCH_METRICS_TOKEN` | string | unset | no | **YES** | When set, `GET /api/v1/metrics` requires the header `X-Metrics-Token` carrying this value. |
| `MATCH_LOG_PSEUDONYM_KEY` | string, at least 16 characters | unset | no | **YES** | Key behind the `sid:` pseudonyms that stand in for Steam account ids in logs. Steam ids are an enumerable namespace, so the handle is an HMAC rather than a plain digest and the key is what stops a log reader precomputing it. Unset means a random key is generated per process, so handles correlate only within one process lifetime and never across a restart or between instances. Production should set it from the Render secret store so handles stay comparable across restarts and across instances. |
| `DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE` | bool | `false` in the image | yes, in the container | no | Disables the .NET configuration file watcher. Required in the immutable container; see the lineage section. |
| `PORT` | int | injected by Render | n/a on Render | no | The port the web service must listen on. Render injects it; do not hard-code it. |

### Trusting forwarded headers

The rate limiter and the auth-failure throttle both partition on the client address. With
`MATCH_TRUST_FORWARDED_HEADERS` off, that address is the peer this process sees, which behind a proxy
is the proxy: one abusive client then spends the budget for everyone behind it. With it on, the server
reads the last `X-Forwarded-For` entry instead, and a caller who can reach this process directly can
pick their own partition by writing that header, which makes both limits decorative.

`MATCH_TRUSTED_PROXY_CIDRS` closes that gap: the header is believed only when the peer is one of the
listed addresses or ranges. Set it whenever the proxy addresses are known.

Render does not publish the addresses of its proxy fleet, so on Render the list stays empty. That
configuration means **every peer is trusted to name the client**, and it is safe only because nothing
can reach the service except through the platform proxy. Because that is also what a half-finished
configuration looks like, it has to be said rather than arrived at: startup fails unless
`MATCH_TRUST_ALL_PROXIES=true` is set as well, and `render.yaml` sets it. The server still logs a
Warning at startup naming the assumption.

If the service is ever given a second way in - a direct port, a private network peer, a sidecar -
either populate `MATCH_TRUSTED_PROXY_CIDRS` or turn `MATCH_TRUST_FORWARDED_HEADERS` back off.

### Placeholder values are rejected

Startup validation treats a required value as missing when it contains any of these tokens, matched
case-insensitively as a substring:

`changeme`, `placeholder`, `your-`, `xxx`, `todo`, `example`

This catches the common failure where a copied template reaches Render with its scaffolding text intact. It
also means a legitimate value must not embed one of these tokens. If one ever must, the validation rule
changes deliberately rather than being worked around in the deployment.

## 5. Startup validation and the environment report

Production startup **fails closed**. When the environment is Production, or when `LOBBY_PROVIDER` includes
`Steam`, and any of `STEAM_APP_ID`, `STEAM_PUBLISHER_WEB_API_KEY`, `DATABASE_URL`, `MATCH_PUBLIC_BASE_URL`,
or `MATCH_BUILD_ID` is missing or holds a placeholder, the process exits non-zero before serving traffic.
The failure message names the offending **keys** only. It never prints a value, because the fastest way to
leak a publisher key is to log it while complaining that it looks wrong.

To inspect a configured process without starting it, run:

```bash
dotnet HexWars.NetServer.dll describe-environment
```

It prints a JSON report and exits. The report answers the question "what is this process actually configured
as" without revealing anything an operator could not already read from the secret store. It contains exactly
`environment`, `steamAppId`, `buildId`, `protocolVersion`, `lobbyProvider`, `engineVersion`,
`engineAssemblyHash`, `databaseTarget` (`host:port/db` only), and `publicBaseUrl`.

It never contains the publisher Web API key, the database password or full connection string, a Steam
authentication ticket, a join credential, or the metrics token.

Sample output, with fabricated values for illustration only:

```json
{
  "environment": "Staging",
  "steamAppId": 480000,
  "buildId": "0f1e2d3c4b5a69788796a5b4c3d2e1f001234567",
  "protocolVersion": 2,
  "lobbyProvider": "Legacy,Steam",
  "engineVersion": "hexwars-engine/1",
  "engineAssemblyHash": "sha256:3f7a1c9e5d2b8046a1c3e5f7092b4d6e8a0c2e4f6180a2c4e6f80123456789ab",
  "databaseTarget": "db.internal.invalid:5432/hexwars_staging",
  "publicBaseUrl": "https://match-staging.invalid"
}
```

`engineVersion` is the engine contract string `hexwars-engine/1`. `engineAssemblyHash` is the full SHA-256
digest of the loaded engine assembly, always printed as `sha256:` followed by 64 lowercase hexadecimal
characters, exactly as in the sample above (or the literal `unavailable` when the assembly has no on-disk
location). It identifies the exact engine assembly, so a match journal replayed on a different build can be
proven to have been replayed against identical rules; the digest is never truncated, because a prefix is not
enough to identify a replay build long after it shipped.

## 6. Steamworks Unity wrapper decision

**Decision: Steamworks.NET** \u2014 <https://github.com/rlabrecque/Steamworks.NET>, MIT licensed.

Rationale:

- MIT licensed, which imposes no obligation on a shipped commercial build.
- Actively maintained, with releases that track the Steamworks SDK.
- Installs as a UPM git package pinned to a release tag:
  `https://github.com/rlabrecque/Steamworks.NET.git?path=/com.rlabrecque.steamworks.net#<tag>`. Pinning to a
  tag rather than a branch keeps the client build reproducible.
- Unity 6 compatible.
- Ships `steam_api64.dll` for Windows x64, the target platform for the Playtest.
- Callbacks are pumped explicitly with `SteamAPI.RunCallbacks()` from a main-thread `MonoBehaviour.Update`,
  so callback timing stays ours to control rather than a hidden background thread.
- Every call sits behind `ISteamLobbyClient`, so EditMode tests run against a deterministic fake and no test
  requires a running Steam client.
- `DISABLESTEAMWORKS` and platform guards keep it out of the WebGL build entirely.

Alternative considered: **Facepunch.Steamworks** (also MIT, a more idiomatic C# surface, but less active
maintenance). The maintenance signal decided it. A wrapper that lags the Steamworks SDK becomes a release
blocker at exactly the wrong moment.

### Provenance record

Fill this in when the Unity install is verified. Until then the wrapper is chosen but not pinned.

| Field | Value |
|---|---|
| Repository | `https://github.com/rlabrecque/Steamworks.NET` |
| Pinned tag | `OWNER-INPUT` (recommend the newest release tag, at least `2024.8.0`) |
| Commit SHA | `OWNER-INPUT` |
| License file path inside the package | `com.rlabrecque.steamworks.net/LICENSE.txt` |
| Steamworks SDK version bundled | `OWNER-INPUT` |
| Verified on Unity version | `OWNER-INPUT` (target 6000.5) |

## 7. Disposable Postgres test strategy

Postgres-backed tests run against a real Postgres, never a mock and never a shared database.

- `Testcontainers.PostgreSql` starts image `postgres:16-alpine`.
- One container per test run, shared by the tests in that run.
- Schema is created by the migration runner that ships with the server, so the tests exercise the same
  migration path production uses. A migration that only works by hand is a migration that has not been
  tested.
- The database is dropped with the container. There is no cleanup step to forget.
- `HEXWARS_TEST_DATABASE_URL`, when set, overrides the container and points the fixture at an existing
  database. This is for CI runners and hosts without a Docker daemon.
- Whatever it points at is checked before a single connection is opened, because every fixture starts by
  dropping and recreating the public schema. The run proceeds only if the database name contains `test`,
  or if `HEXWARS_TEST_DATABASE_DISPOSABLE` names that exact database. Anything else fails with a message
  naming the database and how to confirm it. The container the fixture starts uses `hexwars_test`, so it
  passes the same rule rather than being exempt from it.
- Automated tests never use Render credentials, and never reach the live Valve API.

### The test project does not run on the production runtime

`engine/HexWars.NetServer.Tests` targets **`net10.0`**, because only the .NET 10 runtime is installed in the
development environment and ASP.NET Core 8 TestHost cannot serve the System.Text.Json 10 responses it would
produce. The server itself targets `net8.0` and production runs it on the `aspnet:8.0` image. A green
`dotnet test` therefore proves the behaviour, not the production runtime: the **Docker image smoke test in
the deployment verification script is the runtime gate**, and it is the step that catches an API that exists
on .NET 10 but not on .NET 8. Anything that must be observed on the deployed image, such as serving a build
with no `wwwroot`, belongs in that smoke test rather than in the test project.

## 8. External inputs requiring owner action

None of the following can be derived from the repository. Each one blocks the deployment step that needs it.

- [ ] Steamworks base-game App ID and Playtest App ID
- [ ] Publisher Web API key, scoped to only the required permissions
- [ ] Render staging and production service identifiers
- [ ] Render Postgres staging and production connection strings, stored only in Render
- [ ] Approval of the Steamworks.NET wrapper, and the filled-in provenance record above
- [ ] Initial Playtest region, and the invited-player / concurrent-match target
- [ ] Record of the deployed `web-main` image identity

## 9. Exit gate

This foundation is complete when a clean server process reports its environment, App ID, build ID, protocol
version, lobby provider, engine version, and database target without revealing a secret.

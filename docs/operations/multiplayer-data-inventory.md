# Multiplayer data inventory

**Status: current for the first Playtest. Every row is grounded in the code on this branch.**

What the HexWars match service holds, where it holds it, who can read it, and what removes it. The
retention windows themselves are decided in [match-data-retention.md](./match-data-retention.md); this
document is the field-level companion to that decision. Items marked `OWNER-INPUT` are the only parts
still open.

The schema described here is
[`001_match_journal.sql`](../../engine/HexWars.NetServer/Persistence/Migrations/001_match_journal.sql).
The only code that reads or writes it is
[`PostgresMatchStore`](../../engine/HexWars.NetServer/Persistence/PostgresMatchStore.cs) behind
[`IMatchStore`](../../engine/HexWars.NetServer/Persistence/IMatchStore.cs).

## 1. Sensitivity classes

| Class | Meaning |
| --- | --- |
| **identifier** | Names a person or an account. A SteamID64 is a public, stable, enumerable identifier: it is not a secret, but it is personal data, and it is what a deletion request is about. |
| **gameplay** | Describes a game. Reveals nothing about a person except that they played. |
| **secret-derived** | Computed from a secret. Not the secret itself, and not reversible into it. |
| **operational** | Timestamps and version strings the service needs to run and to recover. |

## 2. `matches`

One row per match. This table is the retention anchor: the other three cascade from it.

| Column | Purpose | Sensitivity | Deletion path |
| --- | --- | --- | --- |
| `match_id` | Primary key, and the id the client carries in `AUTH` and in every API path. | operational | 90-day purge |
| `steam_lobby_id` | The Steam lobby the match was allocated from. Backs the partial unique index that makes allocation idempotent per lobby. A lobby id is not an account id. | operational | 90-day purge |
| `status` | `waiting`, `active`, `completed`, `expired` or `abandoned`. Transitions are enforced by the `matches_enforce_status_transition` trigger, not only by the store. | gameplay | 90-day purge |
| `setup_wire` | The `GameSetup` the match is played under. | gameplay | 90-day purge |
| `start_replay` | The dealt start state, in `ReplayFile` form. The largest column, 10-30 KB. | gameplay | 90-day purge |
| `engine_version` | The engine contract the journal was written under. A build that cannot replay it refuses the match rather than replaying it differently. | operational | 90-day purge |
| `protocol_version` | The wire protocol the match was allocated under. | operational | 90-day purge |
| `build_id` | The server build that allocated it. | operational | 90-day purge |
| `created_at` | Allocation time. Drives the 30-minute waiting expiry. | operational | 90-day purge |
| `started_at` | When both catalogs landed and the journal opened. | operational | 90-day purge |
| `completed_at` | When the match reached a terminal status. Starts the 90-day clock. | operational | 90-day purge |
| `last_activity_at` | Last write of any kind. Drives the 7-day abandonment rule. Monotonic: every write takes `GREATEST`. | operational | 90-day purge |
| `winner_seat` | 0, 1, or null for a draw or a match that ended without a result. | gameplay | 90-day purge |

## 3. `match_players`

One row per seat. **This is the table a deletion request is about.**

| Column | Purpose | Sensitivity | Deletion path |
| --- | --- | --- | --- |
| `match_id` | The match. Cascades on delete. | operational | cascade from `matches` |
| `steam_id` | The SteamID64 of the player in this seat, canonical decimal, digits only by CHECK constraint. | **identifier** | cascade from `matches`; pseudonymisation on request, once the follow-up migration ships |
| `seat` | 0 (lobby owner) or 1. | gameplay | cascade |
| `catalog_wire` | The normalised barracks this seat chose. Null until they send one. | gameplay | cascade |
| `joined_at` | When the seat was allocated. | operational | cascade |
| `last_seen_at` | Last liveness stamp for this seat. Monotonic. | operational | cascade |

A seat is never deleted on its own. `match_commands` holds a `RESTRICT` foreign key to it, so deleting a
player who issued any command raises SQLSTATE 23503; an operator who needs a player gone deletes the
match. See section 3 of the retention decision for the pseudonymisation form and the migration it waits
on.

## 4. `match_commands`

The accepted command journal. Append-only in practice: there is no update path in `IMatchStore`.

| Column | Purpose | Sensitivity | Deletion path |
| --- | --- | --- | --- |
| `match_id` | The match. | operational | cascade from `matches` |
| `sequence` | Server-assigned, starting at 1, no gaps. With `match_id` it is the primary key, which is what makes append idempotent under retry. | operational | cascade |
| `command_wire` | The move, in `CommandWire` form. | gameplay | cascade |
| `accepted_at` | When the server committed it. | operational | cascade |
| `issuer_steam_id` | Which seat issued it, as a composite foreign key to `match_players`. | **identifier** | cascade; pseudonymisation on request |

## 5. `match_join_credentials`

| Column | Purpose | Sensitivity | Deletion path |
| --- | --- | --- | --- |
| `credential_hash` | `SHA-256` of the 32 random bytes handed to the client. Primary key, CHECKed at exactly 32 bytes. **The credential itself is never stored** - see [`MatchCredentialService`](../../engine/HexWars.NetServer/Auth/MatchCredentialService.cs). | **secret-derived** | 24 hours past `expires_at`; cascade from `matches` |
| `match_id` | The match this credential opens. | operational | as above |
| `steam_id` | The seat it was issued to. Composite foreign key to `match_players`. | **identifier** | as above |
| `expires_at` | Issue time plus `MATCH_JOIN_TOKEN_TTL_SECONDS`, capped at the terminal reconnect window when the match has already ended. | operational | as above |
| `revoked_at` | Set when the same seat is issued a newer credential. A revoked row is kept until the purge so an audit can see the replacement happened. | operational | as above |

## 6. Access paths

| Path | Who | What they see | Control |
| --- | --- | --- | --- |
| The server process, through `IMatchStore` | The running service | Everything, one match at a time | The only path the code has. No raw SQL outside `PostgresMatchStore` and the migration runner. |
| Render Postgres console or `psql` | Whoever holds the Render dashboard login | Everything | `OWNER-INPUT`: confirm who has Render dashboard access and enable two-factor on every such account. |
| Render automatic backups | Same | Everything, as of the backup | Section 4 of the retention decision. Backups outlive the retention windows by design. |
| Logs | Whoever can read the Render log stream | Only what section 8 lists | The service never logs a raw Steam id. |

There is no analytics export, no third-party processor, and no copy of this data outside Render.

## 7. In-memory state

None of this is persisted, and all of it is lost on restart. That is deliberate: **no IP address is ever
written to the database.**

| State | Where | Keyed by | Lifetime |
| --- | --- | --- | --- |
| `LiveMatch` projections | [`DurableMatchCoordinator`](../../engine/HexWars.NetServer/Runtime/DurableMatchCoordinator.cs) | match id | Released 10 minutes after the last socket leaves (`IdleEvictionWindow`), or on eviction |
| Live sockets and their queues | [`V2ConnectionRegistry`](../../engine/HexWars.NetServer/Hosting/V2ConnectionRegistry.cs), [`V2Connection`](../../engine/HexWars.NetServer/Hosting/V2Connection.cs) | connection id; the remote address is held for the per-address cap | The socket |
| Credential hash and expiry per socket | `V2Connection` | connection id | The socket. Re-checked against the store every `MATCH_CREDENTIAL_RECHECK_SECONDS`. |
| Request rate-limit budgets | ASP.NET rate limiter, configured in [`ServerComposition`](../../engine/HexWars.NetServer/Hosting/ServerComposition.cs) | **client IP address** | One-minute fixed window |
| Auth-failure lockouts | [`AuthFailureThrottle`](../../engine/HexWars.NetServer/Auth/AuthFailureThrottle.cs) | **client IP address** | Five-minute window, at most 10,000 tracked callers, swept once a minute |
| Open-match quota | [`OpenMatchQuota`](../../engine/HexWars.NetServer/Operations/OpenMatchQuota.cs) | **client IP address** | Ten-minute window, at most 10,000 tracked callers |
| Per-address socket count | `V2ConnectionRegistry.TryReserve` | **client IP address** | While the sockets are open |
| Unrecoverable-match cache | `DurableMatchCoordinator` | match id | 60 seconds (`UnrecoverableRetryWindow`) |
| Steam log pseudonym key | [`SteamLogRedaction`](../../engine/HexWars.NetServer/Steam/SteamLogRedaction.cs) | process | The process, unless `MATCH_LOG_PSEUDONYM_KEY` is set |

The client IP is the address the connection arrives from, or the single forwarded address when
`MATCH_TRUST_FORWARDED_HEADERS` is on. It is used to partition counters and is never logged as a value,
never stored, and never sent anywhere.

## 8. Logs

**Logged.** Match ids as their first eight characters and Steam lobby ids in full, both as logging scopes
([`LogScopes`](../../engine/HexWars.NetServer/Operations/LogScopes.cs)); Steam account ids only as
`sid:` plus 16 hex characters of an HMAC-SHA256 handle (`SteamLogRedaction.HashSteamId`); seat numbers,
statuses, close codes, frame types and lengths, retention row counts, and the startup environment report,
which is built to carry no secret
([`EnvironmentReport`](../../engine/HexWars.NetServer/Configuration/EnvironmentReport.cs)).

**Never logged.** Steam auth tickets; join credentials, in any form, at any level, including the fact
that a particular string was offered; the publisher Web API key; `DATABASE_URL` or any part of it; query
strings, which are stripped wholesale by `SteamLogRedaction.Redact`; client IP addresses; raw
SteamID64s. The whole `System.Net.Http.HttpClient.SteamWebApi.` logging category is filtered to `None`,
because those loggers write the full request URI and for that one client the URI is the key and the
ticket. This is asserted end-to-end by `AFullMatchFlow_PutsNoSecretInAnyLogLine` in
[`SecurityControlsTests`](../../engine/HexWars.NetServer.Tests/SecurityControlsTests.cs), which plays a
whole match and then searches every captured line **and logging scope** for either ticket, either
credential, the publisher key and the database password.

**Log retention:** `OWNER-INPUT`. Confirm how long the Render log stream is kept and whether any log
drain forwards it anywhere.

## 9. Data that stays with Steam

The service reads these through the Steam Web API and does not copy them into Postgres. Lobby data is
read once, at allocation, by
[`SteamLobbyValidator.ValidateForMatchCreation`](../../engine/HexWars.NetServer/Steam/SteamLobbyValidator.cs);
what it hands back is a `VerifiedLobby` of Steam ids, seats, setup and ruleset, and that record is all
[`SteamMatchEndpoints`](../../engine/HexWars.NetServer/Endpoints/SteamMatchEndpoints.cs) passes to
`CreateMatchForLobbyAsync`.

- **Persona and display names.** The owner name lives in lobby metadata as `hw_name`
  ([`SteamLobbyKeys`](../../engine/HexWars.NetServer/Steam/SteamLobbyKeys.cs)). The validator never reads
  that key and no column exists for it.
- **Ready state.** `hw_ready` is read to decide whether a match may start and is not stored.
- **Lobby membership.** Read at allocation to settle the roster. After that, membership on join comes
  from the persisted roster, never from a re-read of the lobby.
- **Lobby chat, invites, friends lists, Community posts, avatars, ownership and ban status.** Ownership
  and publisher-ban status are checked per request and discarded; nothing else is ever requested.

## 10. Secrets

No secret value appears in this repository, in a log, or in an error body.

| Secret | Where it lives | Who can read it | Rotation |
| --- | --- | --- | --- |
| `STEAM_PUBLISHER_WEB_API_KEY` | Render environment, secret group | Render dashboard holders | Owner: `OWNER-INPUT`. Cadence: `OWNER-INPUT`, and immediately on suspected compromise. Procedure in [multiplayer-incident-response.md](./multiplayer-incident-response.md). |
| `DATABASE_URL` | Render, injected from the managed Postgres | Render dashboard holders | Owner: `OWNER-INPUT`. Rotating it means rotating the Postgres password and redeploying. |
| `MATCH_METRICS_TOKEN` | Render environment | Render dashboard holders | Owner: `OWNER-INPUT`. Bound by configuration on this branch; the endpoint that consumes it arrives with the health and metrics work (see `docs/operations/health-and-shutdown.md`). |
| `MATCH_LOG_PSEUDONYM_KEY` | Render environment, optional | Render dashboard holders | Optional. Unset means a per-process random key, which costs only cross-restart correlation of log handles. |
| Join credentials | Transient. 32 bytes from `RandomNumberGenerator`, handed to one client over TLS, stored only as SHA-256. | Nobody but the holder | Re-issued on every create or join, which revokes the previous one for that seat. |

**Before launch:** rehearse a full rotation of the publisher key against a staging deployment and record
the elapsed time in the incident-response document. A rotation procedure that has never been run is a
plan, not a control.

## 11. Least privilege on the database

The intended setup is two roles:

- `hexwars_app` - `SELECT, INSERT, UPDATE, DELETE` on the four tables and nothing else. No `CREATE`, no
  `DROP`, no ownership. This is what `DATABASE_URL` would carry.
- `hexwars_migrate` - DDL, used only by the startup migration runner
  ([`MigrationRunner`](../../engine/HexWars.NetServer/Persistence/MigrationRunner.cs)), which takes
  advisory lock 7331001 so two instances cannot migrate at once.

Render managed Postgres exposes a single owner user by default, and the service currently runs as that
one role for both jobs.

`OWNER-INPUT: confirm whether Render allows a second role on the chosen plan. If it does, create the two
roles above and point DATABASE_URL at hexwars_app. If it does not, the single owner role is accepted for
the Playtest and is recorded as a residual risk in`
[multiplayer-threat-model.md](./multiplayer-threat-model.md).

## 12. Open items

1. Log retention and any log drain (section 8).
2. Render dashboard access list and two-factor enforcement (section 6).
3. Rotation owner and cadence for each secret (section 10).
4. The second database role (section 11).
5. The pseudonymisation migration `002_pseudonymised_players.sql`, without which a deletion request
   cannot be serviced by the automated path (retention decision, section 3).

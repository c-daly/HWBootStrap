# Multiplayer threat model

**Status: current for the first Playtest. Revisit at the promotion gate.**

What an attacker could want from the HexWars match service, what stops them, and what is knowingly left
open. Every mitigation names the file that implements it and the test that proves it, so a row that stops
being true fails a build rather than quietly ageing into fiction.

## 1. Assets

| Asset | Why it matters |
| --- | --- |
| The command journal | It IS the game. A journal that can be altered, reordered or given a hole replays into a different match than the one that was played. |
| Seat integrity | A seat is a person. Anything that lets one account act as another turns every result into a claim. |
| Join credentials | A bearer token for a seat. Whoever holds one is that seat until it expires. |
| The publisher Web API key | Authenticates this service to Valve. Its holder can verify tickets and read lobbies as HexWars. |
| `DATABASE_URL` | Total read and write access to every match ever played. |
| Player identifiers | SteamID64s in three tables. Public, but personal, and enumerable. |
| Availability | A Playtest that is down is a Playtest that did not happen. |

## 2. Actors

| Actor | Trusted for | Never trusted for |
| --- | --- | --- |
| **Player** (authenticated, holds a seat) | Issuing commands for their own seat | Their own seat number, the roster, the setup, sequence numbers, or anything in a request body |
| **Lobby owner** | Choosing the setup and starting the match | Naming who else is in the lobby - that comes from Valve |
| **Non-member** (authenticated, no seat) | Nothing | Anything about a match they hold no seat in |
| **Anonymous caller** | Nothing at all | Anything. Reaches only the rate limiter, the origin check and the body cap. |
| **Valve** | Ticket verification, ownership, lobby membership, ban status | Nothing about game state |
| **Render** | Running the process, terminating TLS, storing the database and its backups | - (a full-trust dependency; see residual risks) |
| **Operator** | Configuration, deploys, the block list, direct database access | - (a full-trust role; there is no operator audit trail) |

## 3. Trust boundaries

```
   Steam client (Unity)                       WebGL client (browser)
   untrusted                                  untrusted, plus a browser Origin
        |                                          |
        |  https  POST /api/v1/steam/matches       |  wss /ws  (v1 legacy)
        |  wss    /ws/v2                           |  https GET /games, /healthz
        v                                          v
  ==========================  TLS terminates at the Render proxy  ==========================
   Render edge   -- X-Forwarded-For honoured only when MATCH_TRUST_FORWARDED_HEADERS is on
        |
        v
  +--------------------------------------------------------------------------------+
  | HexWars.NetServer  (single instance)                                            |
  |                                                                                 |
  |  request limits -> CORS -> rate limiter -> endpoints                            |
  |  origin policy  -> per-IP socket cap  -> /ws/v2 handshake                       |
  |                                                                                 |
  |  identity comes ONLY from a Steam ticket, never from a request body             |
  |  the roster comes ONLY from GetLobbyData, never from a request body             |
  |  sequence numbers come ONLY from the store, never from a client                 |
  +--------------------------------------------------------------------------------+
        |                                     |
        | https, publisher key                | TLS, DATABASE_URL
        v                                     v
   Valve partner API                    Render managed Postgres  --> automatic backups
   (external trust)                     (four tables; see the data inventory)
```

The three lines inside the box are the whole design. Everything a client sends is a claim; the ticket,
the lobby read and the store are the evidence.

## 4. Threats

| Threat | Impact | Mitigation | Residual risk | Verification |
| --- | --- | --- | --- | --- |
| **Forged roster or seat.** A caller claims a seat, an opponent, or a setup the lobby does not have. | A match played by the wrong people, or a result against someone who never joined. | The roster is built from a server-side `GetLobbyData` read and validated by [`SteamLobbyValidator`](../../engine/HexWars.NetServer/Steam/SteamLobbyValidator.cs), which is handed the **authenticated identity**, never a body field. Owner takes seat 0, both members must be ready, duplicate members are refused, and the ruleset and setup must match what the lobby advertises. On a repeat create the stored roster and setup are compared and a changed lobby is a 409. | A lobby owner can still choose who they play with, which is the point of a lobby. | `ValidQuickLobby_SeatsTheOwnerFirstAndReturnsTheQuickSetup`, `GuestRequestingTheMatch_IsNotLobbyOwner`, `RequesterOutsideTheLobby_IsNotLobbyMember`, `AnExistingAllocationWhoseRosterHasChanged_IsAConflict` |
| **Replayed or stolen Steam ticket.** | Impersonation of a real account. | Every ticket is verified against Valve with the identity string `hexwars-match`, per request, with no retry on failure; the ticket is discarded immediately and never stored or logged. A failure is counted against the caller by [`AuthFailureThrottle`](../../engine/HexWars.NetServer/Auth/AuthFailureThrottle.cs). Requests must be https in Production. | A ticket stolen from a compromised machine works until Valve stops honouring it. Nothing on this server can tell that apart from the real player. | `SteamBeingDownIsAServiceOutageRatherThanAPlayerError`, `AnEleventhBadTicketIsRefusedWithoutAskingSteam`, `InProductionAProxiedHttpsRequestIsAllowed` |
| **Credential theft or guessing.** | Someone else plays your seat. | 32 bytes from `RandomNumberGenerator`, transported once over TLS as 43 characters of base64url and stored **only** as SHA-256 ([`MatchCredentialService`](../../engine/HexWars.NetServer/Auth/MatchCredentialService.cs)). Bound to one match and one seat, expires after `MATCH_JOIN_TOKEN_TTL_SECONDS`, revoked the moment that seat is issued another. Never in a URL, a query string or a log line. Live sockets have the credential re-checked against the store every `MATCH_CREDENTIAL_RECHECK_SECONDS` and are closed 1008 when it expires or is revoked. Malformed strings are refused without a store lookup. | The Unity client caches its credential in memory only; a client that writes one to disk would weaken this. Not encrypted at rest on the client. | `Issuing_StoresTheHashAndNeverTheCredential`, `TwoCredentialsForTheSameSeat_AreNeverTheSameString`, `IssuingAgain_RevokesWhatTheSeatAlreadyHeld`, `AMalformedCredential_IsRefusedWithoutReachingTheStore`, `AFullMatchFlow_PutsNoSecretInAnyLogLine` |
| **Impersonating the other seat.** A seated player issues a command whose issuer is their opponent. | A move nobody made. | The coordinator compares the command issuer against the seat the credential proved and answers `REJECT WrongSeat` ([`DurableMatchCoordinator`](../../engine/HexWars.NetServer/Runtime/DurableMatchCoordinator.cs)). The journal enforces the same thing at rest: `issuer_steam_id` is a composite foreign key to `match_players`. | None known. | `OneSeatCannotIssueACommandForTheOther` |
| **Command replay, duplication or reordering.** | Two moves where one was played, or a journal that replays differently. | Sequence numbers are assigned by the server, never accepted from a client. `AppendCommandAsync` requires exactly the next sequence and the `(match_id, sequence)` primary key makes a retry of the identical command `AlreadyApplied` rather than a second row. One `SemaphoreSlim` per match serialises commits. The durable rule is append-then-advance: nothing is broadcast until the row is committed. An ambiguous commit that cannot be verified closes the socket 1011 `resync required` rather than inviting a retry. | A client that loses its connection mid-command learns the outcome only by reconnecting and reading `START`. | `AppendCommand_ReplayedExactly_IsAlreadyApplied`, `AppendCommand_AheadOfTheNextSequence_Conflicts`, `AnAmbiguousAppendThatCannotBeVerified_ClosesTheIssuerRatherThanInvitingARetry`, the six `MatchRestartRecoveryTests` scenarios |
| **Resource exhaustion / DoS.** | The Playtest is down for everyone. | Layered, and every layer is before the expensive work. 16 KB request bodies, 2000 connections, 1000 upgraded, 10 s header deadline, 5 s undeclared-body read deadline, 120 s keep-alive ([`RequestLimits`](../../engine/HexWars.NetServer/Hosting/RequestLimits.cs)); 5 creates and 20 joins per minute per address; 10 auth failures per 5 minutes per address; 3 allocations per 10 minutes per address ([`OpenMatchQuota`](../../engine/HexWars.NetServer/Operations/OpenMatchQuota.cs)); 8 sockets per address, reserved atomically before the upgrade; 64 KB frames; bounded outbound queues by both frame count and bytes, with a slow client closed 1008; at most 64 handshakes validated at once; a 10 s deadline to send `AUTH`. | A distributed attack from many addresses is not stopped by per-address limits. Render is the only thing in front of this process. | `TheSixthCreateInAWindowIsRateLimited`, `TheFourthMatchOneAddressAllocates_IsRefused`, `AHundredUpgradesAtOnceFromOneAddress_LeaveExactlyTheCapAccepted`, `AFrameOverTheSizeCap_ClosesTheSocketAsTooBig`, `AClientThatStopsReading_DoesNotStallTheOtherSeat`, `ABodyOverTheRequestCap_IsRefusedWithoutBeingRead` |
| **Database outage.** | Nothing can be allocated or committed. | Fail closed. Allocation and join answer 503 `service_unavailable` with a fixed body that quotes no exception. A failed append advances nothing, broadcasts nothing and answers the issuer `REJECT TemporaryFailure`; the projection is marked stale and rebuilt from the journal before the next operation. | Matches in progress cannot advance for the duration. | `WithTheDatabaseUnreachable_CreateIsAServiceOutageAndSaysNothingElse`, `AnAppendThatNeverLanded_IsStillATemporaryFailure`, `AStoreFailureIsAServiceOutageAndLeavesNothingBehind` |
| **Steam Web API outage.** | No new matches. | Allocation and join fail with the player-safe "temporarily unavailable" message. Matches already allocated keep playing: the socket handshake needs only a credential, and membership on join comes from the persisted roster rather than a fresh lobby read. | A player who has not yet joined cannot get in until Valve returns. | `SteamBeingDownIsAServiceOutageRatherThanAPlayerError` |
| **Publisher key compromise.** | An attacker can verify tickets and read lobbies as HexWars. | The key exists only in the Render secret group and never in the repository, a log, an error body or the startup environment report. The whole `HttpClient.SteamWebApi` logging category is filtered off, because those loggers write the request URI and the URI carries the key. Rotation procedure in [multiplayer-incident-response.md](./multiplayer-incident-response.md). | Rotation is manual and has not been rehearsed. Recorded as an accepted gap. | `SteamHttpClientLoggingTests`, `AFullMatchFlow_PutsNoSecretInAnyLogLine` |
| **Abusive player.** | Harassment, or one account monopolising the service. | `MATCH_BLOCKED_STEAM_IDS` refuses an account at create and at join, compared canonically by [`PlayerBlockList`](../../engine/HexWars.NetServer/Operations/PlayerBlockList.cs). Publisher bans from Valve are refused outright. Community behaviour - chat, names, harassment - happens in Steam and is reported through Steam. | Updating the block list needs a redeploy, so a block lands within a deploy rather than immediately. A VAC ban is deliberately **not** a refusal: VAC is a signal for the games that consume it and HexWars does not. | `ABlockedAccountCannotStartAMatch`, `APublisherBannedOwnerCannotStartAMatch`, `AVacBannedPlayerIsStillAllowedToPlay`, `ABlockedAccount_MatchesHoweverItWasWritten` |
| **Bad deploy.** | Matches in progress become unplayable. | Options are validated at startup and a half-configured deployment refuses to serve rather than serving wrongly. Startup recovery replays every open match before the first player asks, and readiness stays false until it has ([`MatchRecoveryService`](../../engine/HexWars.NetServer/Runtime/MatchRecoveryService.cs)). A journal written under an engine contract or protocol this build cannot replay is refused by name rather than replayed differently. Migrations are additive and take an advisory lock. Rollback per the Render runbook. | A rollback past an additive migration is untested. | `AJournalWrittenUnderAnEngineContractThisBuildCannotReplay_IsRefused`, `AJournalFromAProtocolThisBuildDoesNotSpeak_IsRefused`, `TheStartupService_RecordsAStoreFailureWithoutClaimingItFinished` |
| **Horizontal scaling split-brain.** Two instances hosting one match. | Divergent projections, and a journal written from two places. | **The service runs as a single instance.** The store is safe for it - allocation is idempotent on a partial unique index, append takes `SELECT ... FOR UPDATE` on the match row, status edges live in a trigger - but the socket layer is not: seat supersede, the per-address caps and the live projections are all per-process. | Scaling out needs sticky routing by match id or a shared coordinator first. Not attempted for the Playtest. | The store half: `PostgresMatchStoreConcurrencyTests` |
| **Cross-site websocket hijack from a browser.** | A page on another site drives a logged-in WebGL session. | [`OriginPolicy`](../../engine/HexWars.NetServer/Hosting/OriginPolicy.cs) is applied **before** the upgrade on both socket routes: same-host or allow-listed passes, anything else is 403, and an `Origin` that is present but unreadable - the literal `null` a sandboxed frame sends - is refused rather than waved through. `/games` and `/healthz` carry a GET-only CORS policy over the same list; `/api/v1/steam/*` and both socket routes emit no CORS headers at all. | An absent `Origin` still passes, because that is every native client. For those, the credential is the real control - this is a browser rule and nothing else. | `AnOriginFromAnotherSite_IsRefusedBeforeTheUpgrade`, `UnparseableOrigin_IsRefused`, `Games_TellsAnyOtherOriginNothing`, `TheSteamApi_NeverEmitsCorsHeaders` |
| **Log-based disclosure.** | Tickets or credentials leak to whoever reads the logs. | Nothing secret is ever passed to a logger; identifiers travel as scopes carrying a shortened match id, a lobby id, or an HMAC `sid:` handle. Query strings are stripped wholesale before any Steam-derived string reaches a sink. | Whoever can read the Render log stream can still correlate matches and handles. | `AFullMatchFlow_PutsNoSecretInAnyLogLine`, and its companion `TheLeakAuditWouldNoticeASecret`, which proves the audit can see both message text and scope values |

## 5. Residual risks accepted for the Playtest

Each of these is a decision, not an oversight. Each is revisited at the promotion gate.

1. **One database role.** Render managed Postgres exposes a single owner user by default, so the service
   runs with DDL rights it only needs at startup. See section 11 of
   [multiplayer-data-inventory.md](./multiplayer-data-inventory.md).
2. **No per-account cap on match creation.** The open-match quota is per address, so one account behind
   several addresses is not bounded by it. The rate limiter and the Steam ownership check are what stand
   in the way.
3. **No encrypted reconnect cache on the client.** The Unity client holds its credential in memory for
   the life of the process. A machine already compromised enough to read that memory has bigger problems.
4. **The block list requires a restart.** Changing `MATCH_BLOCKED_STEAM_IDS` on Render triggers a
   redeploy, and that is the accepted admin path for the Playtest.
5. **No automated key rotation.** Rotating the publisher key is a manual procedure, and it has not been
   rehearsed. Rehearsing it against staging is a launch prerequisite.
6. **Single instance.** Availability depends on one process and on Render. There is no failover.
7. **Full trust in Render and in the operator.** There is no operator audit trail and no separation
   between the person who deploys and the person who can read the database.
8. **Backups outlive retention.** A match deleted at 90 days can still exist in a backup until that
   backup ages out, and a deletion request is satisfied in the live database first.
9. **Deletion requests are not yet automated.** The pseudonymisation path waits on migration
   `002_pseudonymised_players.sql`.

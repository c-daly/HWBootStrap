# Multiplayer incident response

**Status: current for the first Playtest.**

What to do when the HexWars match service is broken, abused, or leaking. Each procedure runs detect,
contain, recover, verify, communicate, in that order, because containing before you understand is how a
small incident becomes a data-loss incident.

Related: [threat model](./multiplayer-threat-model.md), [data inventory](./multiplayer-data-inventory.md), [retention](./match-data-retention.md), [environments](./steam-render-environments.md).

## 1. Severity

| Severity | Meaning | Response | Paged |
| --- | --- | --- | --- |
| **SEV-1** | Data loss, a confirmed secret disclosure, or down for everyone. | Immediately, any hour. | `OWNER-INPUT` |
| **SEV-2** | Matches cannot advance or cannot be allocated, but nothing is lost. | Within the hour, waking hours. | `OWNER-INPUT` |
| **SEV-3** | Degraded: elevated errors, one player affected, an abuse report. | Same day. | `OWNER-INPUT` |
| **SEV-4** | Cosmetic or informational. | Next working session. | Nobody. |

`OWNER-INPUT: name a primary and a secondary contact for SEV-1 and SEV-2, with a contact method that
works when Render is down, and state whether there is any out-of-hours expectation for a Playtest.`

## 2. Signals

| Signal | Source | Watch for |
| --- | --- | --- |
| Uptime check | `OWNER-INPUT: configure an external check against the liveness endpoint.` | Any failure |
| Readiness | `/health/ready` (health and shutdown work; see `docs/operations/health-and-shutdown.md`) | 503 naming the failing check |
| Render service events | Render dashboard | Deploy failures, restarts, out-of-memory kills |
| Render Postgres metrics | Render dashboard | Connections, disk, CPU |
| Startup environment report | `Environment report {...}`, one line at startup | Wrong build id, protocol, or database target |
| Startup recovery | `Startup recovery verified N open match(es), healed N and refused N` | Any refusal; it names the match and the failure |
| Retention sweep | `Retention swept ... expired, ... abandoned, ... credentials deleted, ... matches deleted` | A sudden large abandon count means players are being dropped |
| Refusals | 429 and 403 responses and the warnings behind them | A sustained rise from one address |
| Metrics | `/api/v1/metrics`, gated by `MATCH_METRICS_TOKEN` (health and shutdown work) | Per the metrics documentation |

The environment report, the recovery line and the retention line are the three worth pinning to a dashboard; none carries a secret or a raw account id.

## 3. First five minutes

1. Note the UTC time and open a record from the template in section 6.
2. Read the last environment report line: build id and database target. A surprise here explains most
   incidents.
3. Check readiness and the Render service events before changing anything.
4. **Do not restart to see if it helps.** A restart drops every live socket and every in-memory
   rate-limit and quota counter. The journal survives; the evidence does not.
5. Set the severity and page accordingly.

## 4. Procedures

### 4.1 Database outage

**Detect.** Allocation and join answer 503 `service_unavailable`; commands answer
`REJECT TemporaryFailure`; readiness reports the database check failing. **Contain.** Nothing to do. The
service fails closed: a failed append advances nothing, broadcasts nothing and writes nothing, so there
is no divergent state. Do not restart; it recovers on its own. **Recover.** Follow Render Postgres
recovery. If the database was restored from a backup, matches committed after the backup point are gone:
that is SEV-1, and section 4.10 applies to the logs as well. **Verify.** Readiness 200; allocate a match,
join it, play one command end to end; startup recovery reports zero refusals. **Communicate.**
Maintenance template - players saw "temporarily unavailable", so tell them it is over.

### 4.2 Steam Web API outage

**Detect.** Create and join return 503 with the temporarily-unavailable message while everything else is
healthy; Valve status confirms it. **Contain.** Nothing. Matches already allocated keep playing: the
handshake needs only a credential and membership on join comes from the persisted roster, not a fresh
lobby read. **Recover.** Wait; there is no action on this side. **Verify.** Allocate a new match from a
real lobby. **Communicate.** Say plainly that new matches cannot start and games in progress are fine.

### 4.3 Render outage or region incident

**Detect.** Uptime check fails; the Render dashboard shows the incident. **Contain.** Nothing from here.
**Recover.** Follow Render status; on recovery the startup pass replays every open match before traffic
is accepted. **Verify.** Readiness 200, zero recovery refusals, one match played end to end.
**Communicate.** Maintenance template, naming the platform.

### 4.4 Compromised publisher Web API key - SEV-1

**Detect.** Unexplained usage on the Steamworks partner site; the key appearing anywhere it should not
(a screenshot, a paste, a repository, a log drain); a report from Valve.

**Contain.** Rotate first, investigate second. A key you are still thinking about is a key still in use.

**Recover.**

1. Generate a new publisher Web API key on the Steamworks partner site and revoke the old one there.
2. Update `STEAM_PUBLISHER_WEB_API_KEY` in the Render secret group. Never paste it into a ticket, a chat
   message, or this repository.
3. Redeploy. The service reads configuration at startup only.
4. Revoke every outstanding join credential, because you cannot yet know what was done with the key:
   `UPDATE match_join_credentials SET revoked_at = now() WHERE revoked_at IS NULL;` Live sockets close
   1008 within `MATCH_CREDENTIAL_RECHECK_SECONDS` and clients reconnect through `/join`, which issues
   fresh credentials. Matches in progress survive: the journal is untouched.
5. Review the Steamworks API usage log for the window the old key was live.

**Verify.** Sign in from a real Steam client, allocate a match, play one command; the old key no longer
authenticates; no log line contains either key value. **Communicate.** Credential-reset template.
Players must be told they will be signed out of matches in progress and why, without being told anything
that helps an attacker.

**Prerequisite:** rehearse this whole procedure against staging before launch.
`OWNER-INPUT: rotation rehearsal date and elapsed time.`

### 4.5 Compromised or abused metrics token

**Detect.** Unexplained requests to the metrics endpoint, or the token somewhere it should not be.
**Contain.** Rotate `MATCH_METRICS_TOKEN` in Render and redeploy; the metrics surface exposes counts and
no player data, so this is SEV-3 unless something else was exposed with it. **Recover.** Distribute the
new token out of band. **Verify.** The old token is refused, the new one works. **Communicate.** Internal
only.

### 4.6 Abusive client

**Detect.** A player report, or a sustained pattern of 429s, failed handshakes, or match creation from
one address.

**Contain.** Capture evidence **first**, before anything restarts: the relevant log lines, which carry
`sid:` handles, match ids and counts but never raw account ids, plus the affected match ids. Resolve the
`sid:` handle to an account - it is an HMAC, so this needs `MATCH_LOG_PSEUDONYM_KEY` and the candidate
ids, and with the key unset the handles are per-process and usable only within one process lifetime.
Then add the SteamID64 to `MATCH_BLOCKED_STEAM_IDS` in Render and redeploy; padding, leading zeros and
surrounding whitespace are all tolerated.

**Recover.** No data action; the block takes effect on the redeploy and matches in progress are not
interrupted by it. **Verify.** A create from that account returns 403 `blocked`. **Communicate.** Report
community behaviour - harassment, names, chat - to Valve through the Steam reporting flow; that content
lives in Steam and this service never sees it. Do not name the account publicly.

### 4.7 Bad deploy

**Detect.** Readiness stays 503 after a deploy; the startup pass reports refusals; the environment report
shows the wrong build or protocol; errors jump immediately after a release.

**Contain.** Roll back to the previous Render deploy. Migrations are additive and take an advisory lock,
so the previous build reads the newer schema; a rollback past a migration is untested and is an
escalation, not a routine step.

**Recover.** Confirm the rolled-back build starts, passes recovery and reports ready. **Verify.**
Environment report shows the expected build id; play one match end to end.
**Communicate.** Maintenance template if players saw anything.

### 4.8 Match recovery failures

**Detect.** `Match {MatchId} cannot be recovered: {Failure} {Detail} - maintenance required` at startup,
or a handshake answering `AUTH FAIL unavailable` for one particular match.

**Contain.** The failure is per match: every other match is served normally, and the refusal is cached
for 60 seconds and then retried. **Do not delete the journal.**

**Recover.** The failure name says what is wrong - `UnsupportedEngineContract`, `UnsupportedProtocol`,
`CorruptStartState`, `CorruptCommand`, `CommandReplayFailed`, `SequenceGap`, `IssuerSeatMismatch`. A
contract or protocol refusal usually means the wrong build is deployed: check the environment report
before touching any row. Then follow `docs/operations/match-recovery-runbook.md`, still to be written.

**Verify.** A restart reports zero refusals. **Communicate.** Tell the affected players directly; their
match may not be resumable.

### 4.9 Reconnect storm

**Detect.** A spike in `/join` and `/ws/v2` after a restart or network event; 429s on join; sockets
refused at the per-address cap. **Contain.** Usually nothing - the limits are the containment: 20 joins
per minute per address, 8 sockets per address, at most 64 handshakes validated at once with a two-second
admission window, and clients back off on their own. **Do not raise a limit during the storm**; a limit
raised under load is a limit that was never tested. **Recover.** Let it drain; if it does not, look
upstream for a restart loop. **Verify.** Socket count and refusal rate return to baseline.
**Communicate.** Only if players were kept out for more than a few minutes.

### 4.10 Suspected secret in the logs - SEV-1

**Detect.** A ticket, credential, key or connection string spotted in a log line, or a code change that
could have introduced one.

**Contain.** Treat every secret that could be in that log as compromised: rotate the publisher key (4.4),
the metrics token (4.5), and the database password if `DATABASE_URL` could be involved. Revoke all
outstanding join credentials with the SQL in 4.4. Establish who can read the log stream and how long it
is retained (`OWNER-INPUT`, section 8 of the data inventory), and purge or shorten that retention.

**Recover.** Find the line that wrote it and fix it. Add a case to
`AFullMatchFlow_PutsNoSecretInAnyLogLine` in
[`SecurityControlsTests`](../../engine/HexWars.NetServer.Tests/SecurityControlsTests.cs) that would have
caught it, and see it fail before the fix. **Verify.** The leak audit passes with the new case; a search
of the retained logs for the value finds nothing. **Communicate.** Credential-reset template. If player
identifiers were exposed beyond what Steam already makes public, say so plainly and say what was done.

## 5. Communication templates

Fill the brackets. Never include a match id, an account id, a stack trace, or anything about the
cause that would help someone repeat it.

**Steam Community maintenance post**

> **HexWars multiplayer - [service disruption / maintenance]**
>
> Between [start] and [end] UTC, [online matches could not be started / matches in progress could not
> advance]. [Cause in one plain sentence, no internals.]
>
> This is now resolved. [Matches in progress were preserved and can be rejoined from the main menu. /
> Matches in progress were lost and will need to be started again - we are sorry.]
>
> If you still see problems, restart the game and try again. Thank you for your patience during the
> Playtest.

**Credential reset**

> **HexWars multiplayer - you may need to rejoin your match**
>
> As a precaution we have reset the session tokens that connect players to matches. If you were playing,
> your game will have disconnected and you can rejoin it from the main menu; your match and its history
> are intact.
>
> No action is needed on your Steam account, and no Steam password or payment information is involved.

**In-client message.** The client already has a path for this: the server broadcasts `SERVER RESTART`
and closes 1012 on a graceful shutdown, and the client reconnects with backoff and re-authenticates
([protocol v2](./protocol-v2.md); the shutdown sequence itself is described in
`docs/operations/health-and-shutdown.md`). Use it for anything planned - players see a reconnect rather
than an error. For an unplanned outage the client shows the standard "The match service is temporarily
unavailable - try again shortly."

## 6. Post-incident record

One file per incident under `docs/operations/incidents/YYYY-MM-DD-short-name.md`, within two working
days while the detail is still recoverable.

```markdown
# Incident YYYY-MM-DD: [short name]

**Severity:** SEV-[n]  **Detected:** [UTC]  **Resolved:** [UTC]  **Author:** [name]

## Impact
Who was affected, how many, what they experienced. Matches lost, if any. Data exposed, if any.

## Timeline
| Time (UTC) | Event |
| --- | --- |
| | First signal, and what produced it |
| | Who was paged, and when |
| | Each action taken, and its effect |
| | Service confirmed healthy |

## Root cause
What happened, not who. If it is still unknown, say so and say what would establish it.

## What worked
The controls that did their job. They are the argument for keeping them.

## What did not
Where detection was late, the runbook was wrong, or a control was missing.

## Actions
| Action | Owner | Due | Done |
| --- | --- | --- | --- |

At least one action should be a test that would have caught this, or a signal that would have caught it
sooner.
```

## 7. Open items

1. Contacts and paging expectations per severity (section 1).
2. An external uptime check (section 2).
3. The publisher-key rotation rehearsal against staging (section 4.4).
4. The match recovery runbook (section 4.8), and log retention and readership, which 4.10 depends on.

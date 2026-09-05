# Health, metrics and graceful shutdown

What the platform asks the match server, what it does when it is told to go away, and what an operator can
read while it is running. Where any other document disagrees about a route, a status code or a close code,
this one wins for the operations surface. **Status: Implemented.** **Date: 2026-09-05.** Environment
variables: [Steam / Render environment authority](steam-render-environments.md). Socket behaviour:
[Protocol v2](protocol-v2.md).

## 1. The two probes are different questions

| Route | Answers | Configure as |
|---|---|---|
| `GET /health/live` | 200 for as long as the process is running, whatever state it is in. | Liveness probe |
| `GET /healthz` | Identical body and status. Kept as an alias for deployments that already point at it. | Liveness probe |
| `GET /health/ready` | 200 only when this host can actually serve a match. | Readiness probe |

Pointing a liveness probe at `/health/ready` is the one configuration mistake that matters here. A host
whose database is unreachable is **alive** and reports so: it holds no traffic, names the problem on
readiness, and starts serving the moment the database returns. A liveness probe on the readiness route
turns that into a restart loop against the same database, which holds no traffic either and also loses
every in-memory projection on the way past.

Liveness body:

```json
{"status": "live", "buildId": "<MATCH_BUILD_ID>"}
```

## 2. Readiness

`GET /health/ready` runs every check tagged `ready` and answers **200** for `Healthy` and `Degraded`, **503**
for `Unhealthy`. The body names each check, so a 503 says which one:

```json
{
  "status": "Unhealthy",
  "checks": [
    {"name": "database", "status": "Healthy", "description": null},
    {"name": "schema", "status": "Healthy", "description": null},
    {"name": "recovery", "status": "Unhealthy", "description": "the startup recovery pass has not finished"},
    {"name": "shutdown", "status": "Healthy", "description": null}
  ]
}
```

| Check | Healthy when | Registered |
|---|---|---|
| `database` | `SELECT 1` answers within 2 s. | Only when `DATABASE_URL` is set |
| `schema` | The migration ledger has nothing pending, read within 5 s. | Only when `DATABASE_URL` is set |
| `recovery` | The startup recovery pass has run. | Always |
| `shutdown` | This host has not been told to stop. | Always |

A legacy deployment with no `DATABASE_URL` therefore lists `recovery` and `shutdown` only, and is ready as
soon as it is up.

### Degraded is not unready

`recovery` reports **Degraded** when the startup pass ran and refused one or more matches, and the response
is still 200. A journal this build will not replay is a maintenance signal about one game; every other match
on the host is fine, and taking a working deployment out of rotation over it would turn one broken match
into an outage. The description names the match ids and the refusal, up to twenty of them:

```json
{"name": "recovery", "status": "Degraded",
 "description": "1 match(es) this build will not host: 4b1e... (UnsupportedEngineContract)"}
```

Act on it by hand: the match needs an operator, not a restart.

## 3. Graceful shutdown

On `SIGTERM` (or any other stop), in two phases.

### Phase 1: quiesce, immediately and without waiting

1. **Readiness goes false.** `/health/ready` answers 503 with the `shutdown` check `Unhealthy`, and the
   platform stops routing new players here while the players already here are still being served.
2. **The coordinator stops committing.** The flag is read under each match gate, so a command that reaches
   the gate from here on is answered `REJECT TemporaryFailure` and nothing is appended or broadcast for it.
   That is the answer the protocol already defines for a commit that did not happen: the client retries
   after it reconnects, to whichever host is serving by then.
3. **The registry stops admitting.** A socket that arrives now, or one already mid-upgrade, is refused and
   closed with 1012 rather than seated into a process that is leaving.

All three happen before anything is drained or closed, which is what makes the next phase a barrier rather
than a photograph. `POST /api/v1/steam/matches` and `.../join` answer **503** `service_unavailable` from
step 1, before the body is read and before anything is asked of Valve.

### Phase 2: drain, tell, close - inside one 20 second budget

| Step | Bound |
|---|---|
| Drain in-flight commits | 10 s, and never more than what is left of the budget |
| Broadcast `SERVER RESTART` | 5 s, and never more than what is left |
| Close every socket with 1012 | whatever is left, with a floor of 3 s |

Every step is bounded against the **remaining budget**, not against its own window. A step that overruns is
logged and abandoned, and the next one still runs: **the 1012 closes are always attempted**, whatever
happened before them. A match whose gate cannot be taken in time is skipped and counted rather than waited
on, and each socket close is itself bounded at 3 s before the socket is aborted.

That is the whole point of the budget. A wedged store or a client that stopped reading used to carry
shutdown past the platform kill deadline, and a shutdown that is killed sends 1012 to nobody: every client
it was serving sees a torn connection instead of an instruction to come back.

One summary line closes it out, and it is what an operator reads after the process is gone:

```
Shutdown: 3 live match(es), 5 socket(s) closed, 1 not drained, took 11840 ms
```

`not drained` is the number that matters. Those are matches that may have had a command in flight this host
never confirmed; their players will be re-dealt the whole log on reconnect and lose nothing, but a number
that is routinely non-zero means commits are slow enough to be worth looking at.

The process budget is 20 s and the host shutdown timeout is **25 s**, so a shutdown that spends its whole
budget still has room for the closes. `maxShutdownDelaySeconds` in `render.yaml` is 60, comfortably more
than both. Give any other platform at least 30 s before `SIGKILL`.

The heartbeat stops re-checking credentials once shutdown begins, because a re-check is a database round
trip for a socket that is about to be closed regardless of the answer. Pings carry on.

A legacy deployment has no coordinator and no socket registry. There, phase 1 flips readiness and phase 2
has nothing to do.

## 4. Metrics

The server publishes a `System.Diagnostics.Metrics` meter named **`HexWars.MatchServer`**. Anything that can
attach to a .NET meter gets everything below; a deployment with nothing but a log gets the same numbers two
other ways.

| Instrument | Kind | Tag | Counts |
|---|---|---|---|
| `hexwars.matches.created` | counter | | Matches allocated from a lobby |
| `hexwars.commands.committed` | counter | | Commands durably journalled and broadcast |
| `hexwars.commands.rejected` | counter | `reason` | CMD frames refused, by the reason sent to the issuer. CMD only |
| `hexwars.catalog.rejected` | counter | `reason` | CATALOG frames refused, and starts that could not be recorded |
| `hexwars.db.failures` | counter | `op` | Store calls that threw, by call: `append`, `catalog`, `start`, `complete`, `status`, `reload`, `journal`, `touch`, `load`, `create`, `join`, and the startup pass ones, `recovery_list`, `recovery_load`, `recovery_heal` |
| `hexwars.steam.failures` | counter | `failure` | Every Steam refusal: the exceptions Valve throws, plus `OwnershipMissing`, `Blocked` and `lobby_changed`, which this server decides |
| `hexwars.auth.failures` | counter | `stage` | Handshakes that got no seat: `frame` (never reached a credential), `credential` (a lookup that said no), `timeout` (never sent AUTH), `ticket` (Valve refused the ticket at the HTTP endpoint), `capacity` (this host was already validating as many as it will), `internal` (a failure this server did not anticipate: ours, not the caller) |
| `hexwars.reconnects` | counter | | Seats that took a socket having held one in the last 10 minutes |
| `hexwars.recovery.failures` | counter | | Matches the startup recovery pass refused |

The tags are the point of three of these. An untagged database counter says the database is unhappy and
nothing an operator can act on: a wedged append and a journal read that timed out are the same number and
different incidents. The same goes for a handshake refused before it cost a database read and one refused
by the read itself, and for a command that was not applied against a catalog that was not accepted.

The three `recovery_` operations belong to the startup pass alone. The loader they read through is the
same one every live handshake and every stale reload uses, so tagging inside it would report an outage
during an ordinary reconnect as a startup problem, and would count it twice.

A handshake refused while this host is shutting down is deliberately counted nowhere. Nothing about the
caller was wrong; it is a shutdown, and the shutdown summary already says so.
| `hexwars.matches.live` | gauge | | Matches held in memory |
| `hexwars.sockets.open` | gauge | | Live v2 sockets |
| `hexwars.outbound.queue.max` | gauge | | Deepest any outbound queue has been |
| `hexwars.command.commit.ms` | histogram | | Duration of the durable append |
| `hexwars.command.broadcast.ms` | histogram | | Accepted to the last seat having the frame queued |

### The log line

Every **60 s**, at Information, one structured line:

```
Metrics {"matchesCreated":3,"commandsCommitted":142,...,"openSockets":4,...}
```

It is on by default and is meant to be left on. The numbers that matter after an incident are the ones from
before anybody went looking.

### The endpoint

`GET /api/v1/metrics` returns the same snapshot as JSON.

| Condition | Answer |
|---|---|
| `MATCH_METRICS_TOKEN` unset | **404** - a deployment that never turned this on has no such route |
| Header `X-Metrics-Token` missing or wrong | **401** |
| Header matches | **200**, the snapshot |

The comparison is constant time over SHA-256 digests of both sides, so a wrong token leaks neither its
content nor its length. `MATCH_METRICS_TOKEN` is a **secret**: the snapshot says how many people are playing
and how the database is behaving. It is mapped on every deployment, whatever `LOBBY_PROVIDER` is set to.

Counters are totals since the process started, not rates. A match host restarts often enough that a rate
computed inside it would be computed over the wrong window; subtract two readings instead.

## 5. Verifying journals without touching them

`dotnet HexWars.NetServer.dll verify-journals` answers, read-only, the question readiness answers about the
database this host is attached to: can this build replay the matches in there. It runs no migrations and
asks Postgres for read-only sessions, so a write anywhere below it is refused by the server rather than
trusted not to happen, and it replays through the same verifier the startup recovery pass uses.

It reads `HEXWARS_VERIFY_DATABASE_URL`, falling back to `DATABASE_URL`. `--open-only` restricts it to
matches still being played. It prints one line per match and a summary; exit 0 all replay, 1 some do not,
2 it could not look. The procedure that uses it is
[Procedure G of the match recovery runbook](match-recovery-runbook.md).

It is the only tool that may be pointed at real data. `selftest-durable` builds a match to prove one
survives a restart, and drops the schema of its target to do it.

## 5. Related documents

- [Steam / Render environment authority](steam-render-environments.md) - every variable named above.
- [Protocol v2](protocol-v2.md) - the frames and close codes a shutdown produces.
- [Match data retention](match-data-retention.md) - what happens to a match nobody comes back to.
- [Render Steam Playtest runbook](render-steam-playtest-runbook.md) - the probes to configure and
  the alerts to build on the metrics above.

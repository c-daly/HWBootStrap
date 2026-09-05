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

On `SIGTERM` (or any other stop), in this order:

1. **Readiness goes false first.** `/health/ready` answers 503 with the `shutdown` check `Unhealthy`, and
   the platform stops routing new players here while the players already here are still being served.
2. `POST /api/v1/steam/matches` and `POST /api/v1/steam/matches/{id}/join` answer **503**
   `service_unavailable` immediately, before reading the body and before any call to Valve.
3. `GET /ws/v2` refuses new upgrades with **503**.
4. In-flight commits drain, up to **10 s**. This is what makes the shutdown honest: a command that was
   journalled but never broadcast would leave a player who never heard about a move that is in the record.
5. Every seated socket is sent `SERVER RESTART`.
6. Every socket is closed with **1012 Service Restart**, reason `service restart`. A client that reads 1012
   reconnects with backoff; a client that reads a torn connection shows a player a failure that never
   happened.
7. One summary line is logged: matches live, sockets closed, how long it took.

The host shutdown timeout is **25 s**, which is comfortably more than the drain plus the goodbye and less
than the patience of the platforms this deploys to. Give the container at least 30 s before `SIGKILL`.

A legacy deployment has no coordinator and no socket registry. There, step 1 happens and the rest has
nothing to do.

## 4. Metrics

The server publishes a `System.Diagnostics.Metrics` meter named **`HexWars.MatchServer`**. Anything that can
attach to a .NET meter gets everything below; a deployment with nothing but a log gets the same numbers two
other ways.

| Instrument | Kind | Tag | Counts |
|---|---|---|---|
| `hexwars.matches.created` | counter | | Matches allocated from a lobby |
| `hexwars.commands.committed` | counter | | Commands durably journalled and broadcast |
| `hexwars.commands.rejected` | counter | `reason` | Commands refused, by the reason sent to the issuer |
| `hexwars.db.failures` | counter | | Durable writes or reads that threw |
| `hexwars.steam.failures` | counter | `failure` | Refusals from the Steam Web API |
| `hexwars.auth.failures` | counter | | Tickets and socket handshakes that got no seat |
| `hexwars.reconnects` | counter | | Seats that took a socket having held one in the last 10 minutes |
| `hexwars.recovery.failures` | counter | | Matches the startup recovery pass refused |
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

## 5. Related documents

- [Steam / Render environment authority](steam-render-environments.md) - every variable named above.
- [Protocol v2](protocol-v2.md) - the frames and close codes a shutdown produces.
- [Match data retention](match-data-retention.md) - what happens to a match nobody comes back to.

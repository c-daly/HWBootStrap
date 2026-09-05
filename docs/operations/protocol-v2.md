# Protocol v2 — the match websocket (`/ws/v2`)

The wire protocol a Steam client speaks to a hosted match. It replaces the legacy `/ws` lobby, which is
frozen and still served, unchanged, wherever `LOBBY_PROVIDER` contains `Legacy`.

`/ws/v2` is mapped only when `LOBBY_PROVIDER` contains `Steam`. There is no query string on it: a match
websocket URL carries nothing but the path, and everything about who is connecting arrives in the first
frame. That is deliberate — a credential in a query string is written to every access log and proxy trace
between the client and this process.

Every frame is a UTF-8 text message. The type is the text up to the first space; the payload is the rest.
Inbound frames are capped at 64 KB.

## 1. Frames

### Client to server

| Frame | Payload | Meaning |
|---|---|---|
| `AUTH` | `<matchId> <credential>` | Must be the first frame. Exactly two tokens: the match GUID and the join credential from the create or join response. |
| `CATALOG` | barracks wire form | This seat's army list, sent once while the match is still waiting. |
| `CMD` | command wire form | A move. Answered with `APPLY` to both seats, or `REJECT` to the issuer alone. |
| `PONG` | none | Answer to a `PING`. Counts as liveness and nothing else. |

Any inbound frame counts as liveness, `PONG` included and `PONG` especially. The heartbeat asks whether the
client is still there, not what it has to say.

### Server to client

| Frame | Payload | Meaning |
|---|---|---|
| `SEAT` | `0` or `1` | The seat this credential holds. Always the first frame after a successful `AUTH`. |
| `CATALOG?` | none | This seat has not sent a catalog yet and the match is still waiting for it. |
| `START` | replay text | The dealt start state. Sent when the match becomes active, and again to any seat that reconnects into an active match. |
| `APPLY` | command wire form | A command that is durably recorded. Broadcast to both seats. |
| `REJECT` | `Malformed`, `NoSeat`, `WrongSeat`, `CatalogClosed`, `CatalogV1Required`, `TemporaryFailure`, or an engine rejection reason | The command was not applied. Sent to the issuer only. |
| `AUTH FAIL` | `invalid`, `expired`, or `unavailable` | The handshake failed. Always followed by a close with status 1008. |
| `PING` | none | Liveness probe. Answer with `PONG`. |
| `SERVER RESTART` | none | Graceful shutdown, followed by a close with status 1012. Reconnect with backoff and re-authenticate. |

## 2. The handshake

1. The client opens the socket. A browser client's `Origin` must be the request host or must appear in
   `ALLOWED_WEB_ORIGINS`, or the upgrade is refused with **403 before the socket is accepted**. Native
   clients send no `Origin` and are unaffected.
2. The first frame must be `AUTH <matchId> <credential>`. Anything else — a `CMD`, an `AUTH` with the wrong
   number of tokens, a credential this match never issued — is answered `AUTH FAIL invalid` and closed 1008.
3. A socket that sends nothing within `MATCH_AUTH_TIMEOUT_SECONDS` is closed 1008 with no frame. It has
   proved nothing and is holding a connection slot.
4. On success the server sends `SEAT <n>`, then either `CATALOG?` (the match is still waiting and this seat
   has not sent one) or `START <replay>` (the match is active).

The `AUTH` frame is never logged, at any level. What reaches a log is that a socket authenticated, the seat
it took, and the first eight characters of the match id.

## 3. Close codes

| Code | Name on the wire | When |
|---|---|---|
| 1000 | Normal closure | The socket ended normally, from either side. |
| 1001 | Endpoint unavailable | Stale: nothing inbound for `MATCH_STALE_CONNECTION_SECONDS`. |
| 1008 | Policy violation | The handshake failed, the auth deadline passed, or the client is not reading (see §5). |
| 1009 | Message too big | An inbound frame exceeded 64 KB. |
| 1012 | Service restart | Graceful shutdown, after `SERVER RESTART`. |

## 4. Timing

| Setting | Default | Effect |
|---|---|---|
| `MATCH_AUTH_TIMEOUT_SECONDS` | 10 s | How long a new socket has to send `AUTH`. |
| `MATCH_HEARTBEAT_SECONDS` | 20 s | How often the server sends `PING` on every seated socket. |
| `MATCH_STALE_CONNECTION_SECONDS` | 60 s | Silence after which a seated socket is closed 1001. Must be greater than the heartbeat, and by enough to survive a lost ping; startup refuses a configuration where it is not. |

The heartbeat is the only traffic on an idle match. Without it an idle socket is indistinguishable from a
dead one and an intermediary will eventually drop it silently; without the silence check a client that lost
power holds its seat until the process restarts, because TCP has nothing to say about a peer that has
stopped listening.

## 5. Slow clients

Each connection has a bounded outbound queue, `MATCH_OUTBOUND_QUEUE_CAPACITY` frames (default 256). When it
fills, the connection is closed with **1008 `slow client`**.

The bound is the point. The coordinator hands frames over synchronously while holding a per-match gate, so
enqueueing must never wait; with an unbounded queue that promise is kept by letting a client which has
stopped reading accumulate frames until the process runs out of memory, and every other match on the host
pays for it. A full queue is therefore treated as a decision rather than as a pause: this connection is not
slow, it is gone.

Dropping the frame instead would be worse than closing. A client that missed one `APPLY` and kept the socket
is replaying a game with a hole in it, and neither side would know. Closing sends it back through the
reconnect path, where `START` gives it the whole accepted log again.

A client that is closed this way should reconnect with backoff and re-authenticate. It will receive `START`
with the full accepted log and can fast-forward from there.

## 6. Retry and idempotency

Two different situations, two different rules. Getting them the wrong way round is how a command gets
applied twice.

**After `REJECT TemporaryFailure`, resend the same command.** It means the durable write did not happen.
Nothing was committed, nothing advanced, and nobody was told anything: the command is still usable exactly
as it was.

**After a disconnect, do not resend anything.** Reconnect, re-authenticate, and wait for `START`. It carries
the full accepted log; fast-forward through it and only then continue playing. A command sent before the
disconnect may well have been committed, and the client cannot tell from its side whether the `APPLY` was
lost on the way back or never generated.

This is safe because the server never accepts a client-supplied sequence number. Sequences are assigned by
the store under a `(match_id, sequence)` uniqueness constraint, and every command in one match is evaluated
and appended under that match's gate, one at a time. A double commit is not a race that is unlikely to
happen; it is a row the database will not accept.

## 7. Recovery guarantees

Killing this process at any command boundary and starting a fresh one over the same database cannot lose,
duplicate, reorder or alter an acknowledged action. That is the whole point of committing before
broadcasting, and it is held to by `engine/HexWars.NetServer.Tests/MatchRestartRecoveryTests.cs`, which
runs against real Postgres through the real host and disposes host A entirely - its sockets, its
coordinator and its in-memory projection - before host B is built from nothing but the rows A left behind.

What those tests establish:

| Guarantee | How it is shown |
|---|---|
| A restart keeps the journal and the game continues from it | Three commands on host A, then host B verifies one open match at startup, both seats rejoin with fresh credentials, `START` carries exactly three commands and fast-forwards to the position an independent engine replay reaches, and a fourth command lands at sequence 4. |
| No boundary is special | The same restart is taken after k = 0, 1, 2, 3, 4 and 5 commands. At every k the re-deal holds exactly k commands, the recovered position equals a direct replay of those k, the rest of the game plays out, and the journal ends as 1..5 with each sequence still holding the wire of its own command. |
| A disconnect after a commit is not a lost command | The client sees `APPLY`, drops the socket, reconnects, and `START` already carries the command. Resending it is refused by the engine (that unit has already moved this turn), so the duplicate never reaches the journal. This is why the rule in §6 is reconnect-and-wait rather than resend. |
| A failed durable write leaves nothing behind | One append is made to throw inside the real Postgres store. The issuer gets `REJECT TemporaryFailure`, the other seat is told nothing at all, the identical command sent again is accepted at sequence 1, and the journal holds exactly one row. |
| A build that cannot honour a journal refuses the whole match | With `engine_version` rewritten to an unsupported value, the startup pass reports the match under `UnsupportedEngineContract`, `AUTH` is answered `AUTH FAIL unavailable` and closed 1008, and the journal is untouched. |
| A restart while waiting does not lose a barracks | One seat sends `CATALOG` and the host goes away. After the restart that seat is not asked again, the other seat completes the start, and both are dealt the same game. |

The state comparisons are against `ReplayFile.Write` of a position built by an independent engine replay,
not against the other half of the restart: two halves compared with each other would agree happily on the
same wrong game.

### Running them

```
DOTNET_ROLL_FORWARD=Major dotnet test engine/HexWars.NetServer.Tests --filter MatchRestartRecoveryTests
```

A disposable Postgres is required. With no `HEXWARS_TEST_DATABASE_URL` set the suite starts
`postgres:16-alpine` through Testcontainers; set that variable to point at an existing throwaway database
instead. The fixture drops and recreates the public schema of whatever it is given, so it refuses any
database whose name does not contain `test` unless `HEXWARS_TEST_DATABASE_DISPOSABLE` names it exactly.

The same proof runs outside the test runner, as one command against a running database:

```
HEXWARS_TEST_DATABASE_URL=postgres://user:password@host:5432/hexwars_test \
  dotnet run --project engine/HexWars.NetServer -- selftest-durable
```

It resets the schema, composes the real server on `http://127.0.0.1:5235` with only the Steam partner API
scripted, plays the opening, stops the process, starts a second one on the same port, reconnects both
seats with new credentials and plays on. It prints `SELFTEST-DURABLE PASS` and exits 0 on success, a
diagnostic and 1 on failure, and **3** when it could not run at all: no `HEXWARS_TEST_DATABASE_URL`
(`SELFTEST-DURABLE SKIPPED`), or a database that is not marked disposable by the same rule the test
fixture applies (`SELFTEST-DURABLE REFUSED`). Exit 3 is never a pass - a self-test that came back green
for want of anything to test is worse than one that did not run.

## 8. Related documents

- `docs/operations/steam-render-environments.md` — the environment variables named above.
- `docs/operations/steam-client-configuration.md` — what the Unity client needs to reach this endpoint.

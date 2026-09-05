# Match recovery runbook

What to do when a match cannot be recovered, when the schema has to move, and when the database has to be
restored. Every procedure here is a deliberate, manual act against production data. The health surface it
starts from is [Health, metrics and graceful shutdown](health-and-shutdown.md), the deploy procedures are
the [Render Steam Playtest runbook](render-steam-playtest-runbook.md), and the schema is under
`engine/HexWars.NetServer/Persistence/Migrations/`. **Date: 2026-09-05.**

**The command log is never edited.** `match_commands` is the whole recovery story: replayed against
`start_replay` it reproduces the game exactly. A row changed or deleted there produces a game that never
happened, silently, and nothing anywhere will notice. Every procedure below writes to `matches`,
`match_join_credentials`, or nothing.

## 1. How a problem reaches you

The startup recovery pass replays every open match once, before the host serves anything, and reports what
it found in three places:

| Where | What it says |
|---|---|
| `GET /health/ready` | `recovery` check `Degraded`, description naming up to 20 match ids and the refusal for each |
| Logs | `Match <id> cannot be recovered: <Failure> <detail> - maintenance required`, at Error |
| Metrics | `recoveryFailures` in the `Metrics` line, one per refused match per pass |

**Degraded is 200, on purpose.** One refused journal is a maintenance job about one game; every other match
on the host is fine. The service keeps taking traffic, and the refused match answers every handshake with
`AUTH FAIL unavailable` until somebody acts. The failures, and what each one means:

| Failure | Meaning |
|---|---|
| `UnsupportedEngineContract` | The journal was written under engine rules this build no longer reproduces |
| `UnsupportedProtocol` | The journal speaks a protocol version this build does not |
| `CorruptStartState` | The match is active and has no start state, or the start state will not parse |
| `CorruptCommand` | A command in the log will not parse |
| `SequenceGap` | The command sequence has a hole in it |
| `CommandReplayFailed` | A command the log says was accepted is rejected on replay |
| `IssuerSeatMismatch` | A command is attributed to a seat that does not hold it |
| `NotFound` | The row is gone but something still refers to it |

## 2. Procedure A: an unrecoverable match

For every failure in the table above.

**1. Identify it.** Take the match id from the readiness description or the log line. Then read the row:

```sql
SELECT match_id, status, engine_version, protocol_version, build_id,
       created_at, started_at, last_activity_at
FROM matches WHERE match_id = '<match-id>';

SELECT count(*), min(sequence), max(sequence)
FROM match_commands WHERE match_id = '<match-id>';
```

A gap shows as `count(*)` below `max(sequence)`. An engine or protocol refusal shows in the version
columns: compare them with what this build supports.

**2. Decide: abandon, or roll back the build.**

- If the refusal is `UnsupportedEngineContract` or `UnsupportedProtocol` and the previous build **could**
  host the match, roll the service back (Render runbook, section 5) and let the players finish. Then plan
  the contract bump properly, using procedure B.
- If the journal itself is damaged (`SequenceGap`, `CorruptCommand`, `CorruptStartState`,
  `CommandReplayFailed`, `IssuerSeatMismatch`), no build can host it. Abandon it.
- If only one or two matches are affected and a rollback would interrupt everybody else, abandon them.

**3. Abandon it.** This is the only write, and it is one row:

```sql
UPDATE matches
   SET status = 'abandoned', completed_at = now()
 WHERE match_id = '<match-id>'
   AND status = 'active';
```

Use `status = 'waiting'` in the `WHERE` clause instead for a match that never started. The status
column is guarded by a trigger that permits only `waiting -> active|expired|abandoned` and
`active -> completed|expired|abandoned`, so a typo is refused rather than accepted, and `completed_at` is
required by a check constraint on every terminal status.

**4. Tell the players.** An abandoned match answers a join with 409 and the message that it has ended. Two
players lose a game in progress and will want to know why; post in the Steam Community hub.

**5. Confirm.** Restart the service: readiness returns to Healthy once no open match is refused. The
abandoned row and its command log stay for the retention window, and nothing is deleted.

## 3. Procedure B: bumping the engine contract

An engine change that alters what a command does makes every existing journal unreplayable. The version in
`EngineContract.Version` is what declares that, and the startup pass refuses anything written under a
version not in `EngineContract.SupportedVersions`.

**Keeping the old version in `SupportedVersions` is allowed only when replay results are provably
identical.** Not similar, not usually: identical. If a change alters any outcome for any reachable state,
the old version comes out of the set and existing journals must be drained first. A refactor with unchanged
behaviour may keep it.

There is no way to pause match creation yet. `MATCH_CREATION_PAUSED` is the follow-up that would give one;
until it exists, the drain is a window rather than a switch:

1. **Announce it.** Post the window in the Steam Community hub, a day ahead.
2. **Pick a low-usage window.** `matchesLive` in the `Metrics` line, and the query below, say when.

   ```sql
   SELECT status, count(*) FROM matches
    WHERE status IN ('waiting', 'active') GROUP BY status;
   ```

3. **Wait for the active matches to finish**, or abandon the stragglers with procedure A. Waiting matches
   expire on their own after 30 minutes.
4. **Confirm the count is zero** with the query above.
5. **Bump `EngineContract.Version`** and remove the old version from `SupportedVersions` unless the identity
   rule above is satisfied.
6. **Deploy** through the normal staging-first procedure.
7. **Check readiness.** A `Degraded` recovery check after the deploy means a match slipped through the
   window; abandon it with procedure A.

## 4. Procedure C: restoring the database from a backup

A restore is the most destructive tool here; read all of it before starting.

**What is lost.** Every match created after the backup point disappears: the row, its seats, its command
log. Every join credential issued after that point becomes invalid, because the credential rows go with the
backup. Players holding one get `AUTH FAIL invalid` and must rejoin from the lobby.

**What survives.** Anything the backup contains, exactly as it was. The schema goes back with it, so a
restore across a migration boundary rolls the schema back too; if `GET /health/ready` then reports pending
migrations, let the service apply them at startup.

1. Take a fresh backup of the current state first, however broken it is. It is the only copy of whatever is
   about to be discarded.
2. Restore into a **new** database instance rather than over the live one, so step 4 can be done before
   anything is switched.
3. Verify the copy replays. See procedure G.
4. Point `DATABASE_URL` at the restored instance and redeploy.
5. Communicate. Post in the Steam Community hub: matches started after the backup point are gone, and every
   player must rejoin from the lobby. Say the time of the backup point in plain terms.

## 5. Procedure D: matches stuck in waiting

A waiting match is one where seats are allocated and at least one catalog has not arrived. The retention
sweeper expires them 30 minutes after creation, so almost always the answer is to wait.

To see them:

```sql
SELECT match_id, created_at, steam_lobby_id FROM matches
 WHERE status = 'waiting' AND created_at < now() - interval '30 minutes'
 ORDER BY created_at;
```

If the sweeper is not running, expire them by hand:

```sql
UPDATE matches
   SET status = 'expired', completed_at = now()
 WHERE status = 'waiting' AND created_at < now() - interval '30 minutes';
```

A lobby whose match expired can allocate a new one: the unique index keeping one open match per lobby covers
only `waiting` and `active`.

## 6. Procedure E: revoking the credentials of a player

A credential is a bearer token. Revoking it closes the sockets holding it within the recheck interval
(`MATCH_CREDENTIAL_RECHECK_SECONDS`, 60 s by default) and refuses the next handshake.

```sql
UPDATE match_join_credentials
   SET revoked_at = now()
 WHERE steam_id = '<steam-id>' AND revoked_at IS NULL;
```

Add `AND match_id = '<match-id>'` to revoke in one match only. This ends the current sessions; the
player can join again for a new credential, so to stop them coming back add the Steam id to
`MATCH_BLOCKED_STEAM_IDS` and redeploy.

## 7. Procedure F: exporting a replay

The stored start state and the command log together are exactly the file the replay viewer reads.

```sql
SELECT start_replay FROM matches WHERE match_id = '<match-id>';

SELECT command_wire FROM match_commands
 WHERE match_id = '<match-id>' ORDER BY sequence;
```

`start_replay` is already a complete `HEXWARS-REPLAY 1` document whose last line is `CMDS 0`. To attach the
commands: replace that final `CMDS 0` line with `CMDS <n>`, where `n` is the number of rows the second query
returned, then append the `command_wire` values one per line **in sequence order**. Save it as a `.txt` and
open it with the replay viewer.

Order is the whole content of the file: a replay assembled without `ORDER BY sequence` still loads, and
shows a different game.

## 8. Procedure G: verifying that a copy of production data replays

The check to run before trusting a restore, and the honest way to rehearse an engine bump.

**Never point `selftest-durable` at real data.** It proves a match can survive a restart by BUILDING one,
and it drops the public schema of its target before it starts. It refuses any database whose name is not
marked disposable for exactly that reason, and making a name disposable to get past the refusal would
destroy the journals the restore exists to preserve.

Use `verify-journals` instead. It runs no migrations, and every statement it issues of its own runs inside
an explicit `READ ONLY` transaction, which cannot be turned off from inside itself. It replays every journal
through the same verifier the running host applies at startup, so a pass here means what a healthy
`recovery` check means.

**What that does and does not promise.** It protects against this verb writing by accident: no migration, no
repair, no statement of its own that could ever be a write. It is not a sandbox. Anything else holding the
same credentials can still write, and the connection-level default it also sets is only a default - one
`SET` turns that off. When you need the guarantee rather than the discipline, connect as a role that cannot
write:

```sql
-- OWNER-INPUT: create once per database an operator will verify.
CREATE ROLE hexwars_readonly LOGIN PASSWORD '...';
GRANT CONNECT ON DATABASE <database> TO hexwars_readonly;
GRANT USAGE ON SCHEMA public TO hexwars_readonly;
GRANT SELECT ON matches, match_players, match_commands, match_join_credentials TO hexwars_readonly;
GRANT SELECT ON schema_migrations TO hexwars_readonly;
```

Point `HEXWARS_VERIFY_DATABASE_URL` at that role and the promise stops depending on this code at all.

1. Restore the backup into a database of its own, never over production.
2. Point the verb at it and run it from the deployed image:

   ```
   HEXWARS_VERIFY_DATABASE_URL=postgres://user:pass@host:5432/restored dotnet HexWars.NetServer.dll verify-journals
   ```

   Add `--open-only` to check just the matches still being played. From a checkout the same verb is
   `dotnet run --project engine/HexWars.NetServer -- verify-journals`.

3. Read the output. One line per match, then a summary:

   ```
   4b1e0f42-... active 37 OK
   9c02a118-... active 12 CorruptCommand the command at sequence 5 does not parse
   VERIFY-JOURNALS FAIL 1/2
   ```

| Exit | Meaning | What to do |
|---|---|---|
| 0 | `VERIFY-JOURNALS PASS n/n`, every journal replays | Trust the restore |
| 1 | `VERIFY-JOURNALS FAIL k/n`, k journals will not | Abandon those matches with procedure A after the restore |
| 2 | It could not look: no address, no connection, or a schema behind this build | Fix that first; nothing was verified |

Exit 2 on a schema mismatch is deliberate. A journal read against a schema this build does not recognise
would report failures that are about the schema, and an operator acting on them would go looking at
matches that are perfectly intact.

4. Optionally, deploy a staging service against the copy and read `GET /health/ready`. The `recovery` check
   answers the same question through the host that would serve it.

## 9. Related documents

- [Health, metrics and graceful shutdown](health-and-shutdown.md)
- [Render Steam Playtest runbook](render-steam-playtest-runbook.md)
- [Match data retention](match-data-retention.md)
- [Protocol v2](protocol-v2.md)

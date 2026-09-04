# Match data retention

**Status: Decided for the first Playtest; revisit at the promotion gate.**

This document is the retention decision for the Postgres match journal that backs Steam-hosted
matches. It states what is kept, for how long, what deletes it, and what is deliberately never
stored. It is a decision record, not a proposal. Items marked `OWNER-INPUT` are the only parts
still open, and they are listed again at the end.

The schema these rules apply to is
[`001_match_journal.sql`](../../engine/HexWars.NetServer/Persistence/Migrations/001_match_journal.sql).
The build effort that introduces it is described in
[the Steam lobby hosting plan](../superpowers/plans/2026-09-03-steam-lobby-render-postgres-production.md).

## 1. Scope

Four tables hold all durable match data. Nothing else in the service writes to Postgres.

| Table | Contents |
| --- | --- |
| `matches` | One row per match: id, Steam lobby id, status, setup wire, start replay, engine and protocol version, build id, the four timestamps, winner seat. |
| `match_players` | One row per seat: match id, Steam id, seat number, normalized catalog wire, joined and last-seen timestamps. |
| `match_commands` | The accepted command journal: match id, sequence, command wire, accepted timestamp, issuer Steam id. |
| `match_join_credentials` | Hashed join credentials: SHA-256 hash, match id, Steam id, expiry, revoked timestamp. |

Child rows are deleted with their match by foreign key cascade, so `matches` is the single
retention anchor for the first three tables. Credentials expire on their own clock.

## 2. Lifecycle statuses

A match moves through `waiting -> active -> completed | expired | abandoned`, and a match that
never starts moves `waiting -> expired | abandoned` directly. The three terminal statuses are
final, and only they start the 90-day deletion clock.

- `waiting` — the match row exists and seats are allocated, no start replay yet.
- `active` — the start replay is written and the command journal is open.
- `completed` — the game reached a real end state, with a winner seat recorded.
- `expired` — the match aged out before it ever started. No winner.
- `abandoned` — the match started but went quiet. No winner.

## 3. Retention decisions

| Class | Rule | Trigger | Terminal action |
| --- | --- | --- | --- |
| Waiting match that never started | Expire after 30 minutes | Status is `waiting` and `created_at` is older than 30 minutes | Set status to `expired`, stamp `completed_at`, no winner |
| Active match gone quiet | Abandon after 7 days | Status is `active` and `last_activity_at` is older than 7 days | Set status to `abandoned`, stamp `completed_at`, no winner |
| Terminal matches and their journals | Keep 90 days | Status is `completed`, `expired` or `abandoned` and `completed_at` is older than 90 days | Hard-delete the `matches` row; cascade removes its players and commands |
| Join credentials | Delete 24 hours after expiry | `expires_at` is older than 24 hours, whether or not the credential was revoked | Row deleted |
| Player identifiers after a deletion request | Pseudonymise the Steam id | A player asks for deletion and none of their matches is still `waiting` or `active` | Rewrite the Steam id columns; the match records themselves stay for the rest of the 90-day window |

**Replay export before deletion is not enabled for the Playtest.** Exporting the start replay and
command journal to object storage before the 90-day delete is a real option and the schema supports
it, but nothing exports today. Deleted means gone. Revisit at the promotion gate.

**Pseudonymisation form.** On a deletion request the Steam id is replaced with
`deleted-<first 16 hex characters of SHA-256(steam_id)>` in `match_players.steam_id`,
`match_commands.issuer_steam_id`, and `match_join_credentials.steam_id`. The prefix keeps the value
distinguishable from a real SteamID64, and the hash keeps rows belonging to the same deleted player
joinable for debugging without carrying the identifier.

Active matches are not interrupted by a deletion request. They finish normally, or they fall into
the 7-day abandonment rule first, and the pseudonymisation runs afterwards.

**Required follow-up before this can run.** `match_players.steam_id` carries a CHECK constraint
requiring digits only, which a `deleted-` value violates. A follow-up migration
`002_pseudonymised_players.sql` must relax that constraint to accept either a canonical decimal
SteamID64 or the pseudonymised form. Until that migration ships, deletion requests cannot be
serviced by the automated path.

## 4. Backups

Backups are the automatic backups that Render provides for managed Postgres. The service runs no
dump schedule of its own, and no backup copy leaves Render.

- Plan and backup retention in days: `OWNER-INPUT`. Confirm which Render Postgres plan the service
  runs on and how many days of backups that plan keeps.
- Point-in-time recovery availability: `OWNER-INPUT`. Confirm whether the chosen plan offers it and
  what recovery window it covers.
- A restore rehearsal against a scratch database is a promotion-gate item. It has not been done.

Backups keep data past the retention windows above by design. A match hard-deleted at 90 days can
still exist inside a backup until that backup ages out, and a deletion request is therefore
satisfied in the live database first and in backups only as they roll over. State this plainly to
any player who asks.

The restore procedure itself is not in this document. It belongs to the match recovery runbook at
[`match-recovery-runbook.md`](./match-recovery-runbook.md), which is still to be written.

## 5. Sweeper contract

The rules above are enforced by a hosted service `MatchRetentionService`, implemented in the
security and operations task. It runs every 60 minutes and issues exactly these statements:

```sql
UPDATE matches SET status='expired', completed_at=now(), last_activity_at=now()
  WHERE status='waiting' AND created_at < now() - interval '30 minutes';
UPDATE matches SET status='abandoned', completed_at=now(), last_activity_at=now()
  WHERE status='active' AND last_activity_at < now() - interval '7 days';
DELETE FROM match_join_credentials WHERE expires_at < now() - interval '24 hours';
DELETE FROM matches WHERE status IN ('completed','expired','abandoned') AND completed_at < now() - interval '90 days';
```

Each statement has an index behind it in `001_match_journal.sql`, so an hourly sweep over a large
table is a range scan rather than four sequential scans:

| Sweeper statement | Index |
| --- | --- |
| Expire stale waiting matches | `ix_matches_waiting_created` on `matches (created_at) WHERE status = 'waiting'` |
| Abandon silent active matches | `ix_matches_status_activity` on `matches (status, last_activity_at)` |
| Purge expired join credentials | `ix_join_credentials_expires` on `match_join_credentials (expires_at)` |
| Purge matches past the 90-day window | `ix_matches_terminal_completed` on `matches (completed_at) WHERE status IN ('completed','expired','abandoned')` |

The purge deletes only from `matches`. Players, commands and join credentials go with it: every one
of those tables cascades from the match row.

Deleting a seat is a different thing, and the schema treats it differently. `match_join_credentials`
cascades from the seat, so removing a player takes their join tokens with them. `match_commands`
does **not**: its foreign key to the seat is `ON DELETE RESTRICT`, so deleting a player who issued
any command is refused with SQLSTATE `23503`. A journal with a hole in it replays into a different
game than the one that was played, so commands are removed only when the whole match is, never by
reaching for one seat. An operator who needs a player's rows gone deletes the match.

Two rules constrain the sweeper:

- A sweep never touches an `active` match with activity in the last 7 days. A game in progress is
  never disturbed by retention, however long it has been running, as long as players keep playing
  it.
- A sweep logs counts only. Row counts per statement, never match ids, Steam ids, command wires, or
  credential material.

Pseudonymisation is not part of the sweep. It is a deliberate, requested operation, run separately
once the follow-up migration in section 3 has shipped.

## 6. Size estimate

These are estimates, not measurements. Nothing has been measured against a real Playtest yet.

| Item | Estimate |
| --- | --- |
| Setup row | 50 B |
| Start replay | 10-30 KB |
| Commands, roughly 200 at about 40 B each | 8 KB |
| Total per match | about 40 KB |
| At 100 matches per day | about 4 MB per day |
| Full 90-day window plus indexes | about 360 MB |

That fits inside the smallest paid Render Postgres plan with a wide margin. Re-measure against real
Playtest data at the capacity gate rather than trusting these numbers.

## 7. Data not stored

The following never reaches Postgres, by design. If any of it appears in the database, that is a
bug and not a retention question.

- Raw Steam authentication tickets. A ticket is verified against Steam and discarded.
- Plaintext join credentials. Only the SHA-256 hash of the credential is stored.
- Lobby chat, and any Steam Community content.
- Player display names. The display name of the lobby owner exists only as `hw_name` in Steam lobby
  metadata, which lives in Steam, never in Postgres.
- IP addresses. Rate-limit counters are in-memory only and are lost on restart.

## 8. Open items for the promotion gate

1. Confirm the Render Postgres plan, its backup retention in days, and whether point-in-time
   recovery is available. This closes both `OWNER-INPUT` items in section 4.
2. Decide whether to enable replay export to object storage before the 90-day hard delete.
3. Confirm the deletion-request service level: how quickly a request is acknowledged, and how
   quickly it is executed once the matches of that player are no longer active.
4. Rehearse a restore against a scratch database and record the result in the match recovery
   runbook.

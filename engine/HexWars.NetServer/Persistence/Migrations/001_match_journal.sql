-- 001_match_journal: the durable match journal.
--
-- A match is a setup plus an ordered command log; replaying the log against the setup reproduces the
-- game exactly, so this schema is the whole recovery story. The constraints here are the last line of
-- defence: the partial unique index below is what makes match allocation idempotent per Steam lobby
-- even when two clients race, and the sequence primary key is what makes command append idempotent.

CREATE TABLE IF NOT EXISTS matches (
  match_id UUID PRIMARY KEY,
  steam_lobby_id TEXT NULL,
  status TEXT NOT NULL CHECK (status IN ('waiting','active','completed','expired','abandoned')),
  setup_wire TEXT NOT NULL,
  start_replay TEXT NULL,
  engine_version TEXT NOT NULL,
  protocol_version INTEGER NOT NULL,
  build_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  started_at TIMESTAMPTZ NULL,
  completed_at TIMESTAMPTZ NULL,
  last_activity_at TIMESTAMPTZ NOT NULL,
  winner_seat INTEGER NULL CHECK (winner_seat IN (0,1))
);

-- At most one open match per lobby. Completed and expired rows drop out of the index, so the same
-- lobby can host another match later.
CREATE UNIQUE INDEX IF NOT EXISTS ux_matches_open_lobby ON matches (steam_lobby_id)
  WHERE status IN ('waiting','active') AND steam_lobby_id IS NOT NULL;

-- Drives the retention reaper and the open-match recovery scan.
CREATE INDEX IF NOT EXISTS ix_matches_status_activity ON matches (status, last_activity_at);

CREATE TABLE IF NOT EXISTS match_players (
  match_id UUID NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  steam_id TEXT NOT NULL CHECK (steam_id ~ '^[0-9]{1,20}$'),
  seat INTEGER NOT NULL CHECK (seat IN (0,1)),
  catalog_wire TEXT NULL,
  joined_at TIMESTAMPTZ NOT NULL,
  last_seen_at TIMESTAMPTZ NULL,
  PRIMARY KEY (match_id, steam_id),
  UNIQUE (match_id, seat)
);

CREATE TABLE IF NOT EXISTS match_commands (
  match_id UUID NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK (sequence >= 1),
  command_wire TEXT NOT NULL,
  accepted_at TIMESTAMPTZ NOT NULL,
  issuer_steam_id TEXT NOT NULL,
  PRIMARY KEY (match_id, sequence)
);

CREATE TABLE IF NOT EXISTS match_join_credentials (
  credential_hash BYTEA PRIMARY KEY CHECK (octet_length(credential_hash) = 32),
  match_id UUID NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  steam_id TEXT NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS ix_join_credentials_match_player ON match_join_credentials (match_id, steam_id);

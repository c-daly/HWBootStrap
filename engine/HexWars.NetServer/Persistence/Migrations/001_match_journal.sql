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
  winner_seat INTEGER NULL CHECK (winner_seat IN (0,1)),

  -- The status column and the columns that describe it must agree, because recovery trusts the status
  -- word before it looks at anything else. An active or completed match without a start replay cannot
  -- be replayed at all; a terminal match without completed_at never ages out of the retention window;
  -- and a winner on anything but a completed match is a score for a game that was never finished.
  CONSTRAINT ck_matches_active_has_replay
    CHECK (status <> 'active' OR start_replay IS NOT NULL),
  CONSTRAINT ck_matches_completed_has_replay
    CHECK (status <> 'completed' OR start_replay IS NOT NULL),
  CONSTRAINT ck_matches_terminal_has_completed_at
    CHECK (status IN ('waiting','active') OR completed_at IS NOT NULL),
  CONSTRAINT ck_matches_winner_only_when_completed
    CHECK (winner_seat IS NULL OR status = 'completed')
);

-- At most one open match per lobby. Completed and expired rows drop out of the index, so the same
-- lobby can host another match later.
CREATE UNIQUE INDEX IF NOT EXISTS ux_matches_open_lobby ON matches (steam_lobby_id)
  WHERE status IN ('waiting','active') AND steam_lobby_id IS NOT NULL;

-- Drives the retention reaper and the open-match recovery scan.
CREATE INDEX IF NOT EXISTS ix_matches_status_activity ON matches (status, last_activity_at);

-- One index per statement the retention sweeper issues (docs/operations/match-data-retention.md), so an
-- hourly sweep over a large table is a range scan rather than a sequential one.
CREATE INDEX IF NOT EXISTS ix_matches_waiting_created ON matches (created_at)
  WHERE status = 'waiting';
CREATE INDEX IF NOT EXISTS ix_matches_terminal_completed ON matches (completed_at)
  WHERE status IN ('completed','expired','abandoned');

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

-- The issuer is a seat, not a free-text label: a command whose issuer holds no seat in this match could
-- never have been accepted, and on replay there would be nobody to attribute it to.
CREATE TABLE IF NOT EXISTS match_commands (
  match_id UUID NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK (sequence >= 1),
  command_wire TEXT NOT NULL,
  accepted_at TIMESTAMPTZ NOT NULL,
  issuer_steam_id TEXT NOT NULL,
  PRIMARY KEY (match_id, sequence),
  CONSTRAINT fk_match_commands_issuer_seat
    FOREIGN KEY (match_id, issuer_steam_id) REFERENCES match_players(match_id, steam_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

-- Same rule for credentials: a join token is issued to a seat, so it dies with the seat.
CREATE TABLE IF NOT EXISTS match_join_credentials (
  credential_hash BYTEA PRIMARY KEY CHECK (octet_length(credential_hash) = 32),
  match_id UUID NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  steam_id TEXT NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ NULL,
  CONSTRAINT fk_match_join_credentials_seat
    FOREIGN KEY (match_id, steam_id) REFERENCES match_players(match_id, steam_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_join_credentials_match_player ON match_join_credentials (match_id, steam_id);

-- The expired-credential purge sweeps on expires_at alone.
CREATE INDEX IF NOT EXISTS ix_join_credentials_expires ON match_join_credentials (expires_at);

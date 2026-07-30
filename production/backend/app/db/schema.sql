PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id     TEXT PRIMARY KEY,
  pipeline   TEXT NOT NULL,
  status     TEXT NOT NULL DEFAULT 'queued',
  config     TEXT NOT NULL DEFAULT '{}',
  error      TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stages (
  run_id      TEXT NOT NULL,
  name        TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'queued',
  started_at  TEXT,
  finished_at TEXT,
  duration_s  REAL,
  error       TEXT,
  traceback   TEXT,
  PRIMARY KEY (run_id, name)
);

CREATE TABLE IF NOT EXISTS events (
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id  TEXT NOT NULL,
  stage   TEXT,
  type    TEXT NOT NULL,
  ts      TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);

CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL,
  stage  TEXT NOT NULL,
  name   TEXT NOT NULL,
  value  REAL NOT NULL,
  ts     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  run_id     TEXT NOT NULL,
  stage      TEXT NOT NULL,
  name       TEXT NOT NULL,
  path       TEXT NOT NULL,
  kind       TEXT NOT NULL DEFAULT 'file',
  size_bytes INTEGER NOT NULL DEFAULT 0,
  ts         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spans (
  span_id     TEXT PRIMARY KEY,
  run_id      TEXT NOT NULL,
  stage       TEXT NOT NULL,
  parent_id   TEXT,
  type        TEXT NOT NULL,
  name        TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'ok',
  started_at  TEXT NOT NULL,
  duration_ms REAL,
  input       TEXT,
  output      TEXT,
  tokens_in   INTEGER,
  tokens_out  INTEGER,
  cost_usd    REAL,
  extra       TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id, stage);

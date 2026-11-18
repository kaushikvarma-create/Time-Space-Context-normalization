-- 001 core schema: spaces, registry, frames, observations, scene state, events, derived tables, views
-- Design rules:
--   * New object types are new ROWS in object_registry, never new columns.
--   * Missing is not zero: camera_down frames have NULL metrics and no observations.
--   * Every row carries source/provenance and (via frames) model + prompt versions.
--   * space_id is on every fact row; map_* columns on spaces are the hook for linking rooms later.


-- ---------------------------------------------------------------- dimensions

CREATE TABLE IF NOT EXISTS spaces (
  space_id      TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  space_type    TEXT NOT NULL,              -- room | parking | ... (drives vocabulary/grid config)
  rtsp_env      TEXT,                       -- name of env var holding the RTSP URL (no secrets in DB)
  grid_cols     INTEGER NOT NULL,
  grid_rows     INTEGER NOT NULL,
  tz            TEXT NOT NULL,              -- IANA name, or 'local'
  config_path   TEXT,
  -- provisions for a future multi-room map (all nullable, unused for now)
  site_id       TEXT,
  map_x         REAL,
  map_y         REAL,
  map_rotation  REAL,
  map_transform_json TEXT,
  created_utc   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS object_registry (
  object_type     TEXT PRIMARY KEY,         -- canonical label, e.g. 'person', 'car', 'chair'
  category        TEXT,                     -- person | vehicle | furniture | device | item | other
  mobility_prior  TEXT NOT NULL DEFAULT 'unknown',  -- fixed | semi_static | dynamic | unknown (seeded guess)
  review_status   TEXT NOT NULL DEFAULT 'seed',     -- seed | approved | auto_new | needs_review | merged
  first_seen_utc  TEXT,
  first_seen_frame_id INTEGER,
  first_seen_space_id TEXT,
  notes           TEXT
);

CREATE TABLE IF NOT EXISTS object_alias (
  alias        TEXT PRIMARY KEY,            -- lowercase synonym, e.g. 'couch'
  object_type  TEXT NOT NULL REFERENCES object_registry(object_type)
);

-- ---------------------------------------------------------------- facts

CREATE TABLE IF NOT EXISTS frames (
  frame_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  space_id       TEXT NOT NULL REFERENCES spaces(space_id),
  ts_utc         TEXT NOT NULL,             -- ISO8601, UTC
  ts_local       TEXT NOT NULL,
  tz             TEXT NOT NULL,
  date           TEXT NOT NULL,             -- local date YYYY-MM-DD
  dow            INTEGER NOT NULL,          -- 0=Mon .. 6=Sun
  dow_name       TEXT NOT NULL,
  hour           INTEGER NOT NULL,
  minute         INTEGER NOT NULL,
  slot_15        INTEGER NOT NULL,          -- 0..95 quarter-hour slot of the day
  week_of_year   INTEGER NOT NULL,
  is_weekend     INTEGER NOT NULL,
  part_of_day    TEXT NOT NULL,             -- night | morning | afternoon | evening
  image_path     TEXT,                      -- NULL after retention or when camera_down
  capture_status TEXT NOT NULL,             -- ok | camera_down | bad_frame
  width          INTEGER,
  height         INTEGER,
  brightness     REAL,                      -- mean luminance 0..255 (measured, not guessed)
  contrast       REAL,                      -- luminance std dev
  sharpness      REAL,                      -- variance of Laplacian (blur / lens-obstruction signal)
  detector_version TEXT,
  vlm_model      TEXT,
  prompt_version TEXT,
  detector_raw   TEXT,                      -- raw detector output (JSON), always kept
  vlm_raw        TEXT,                      -- raw VLM output (JSON), always kept
  processed_utc  TEXT,                      -- NULL until Engine 1 has run on it
  UNIQUE (space_id, ts_utc)
);
CREATE INDEX IF NOT EXISTS ix_frames_space_time ON frames(space_id, ts_utc);
CREATE INDEX IF NOT EXISTS ix_frames_dow_hour   ON frames(space_id, dow, hour);

-- One row per detected/read object per frame. Position comes from the detector box
-- (geometric grid cell); VLM-only items may have NULL cell and bbox.
CREATE TABLE IF NOT EXISTS observations (
  obs_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  frame_id     INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
  space_id     TEXT NOT NULL REFERENCES spaces(space_id),
  ts_utc       TEXT NOT NULL,
  object_type  TEXT NOT NULL REFERENCES object_registry(object_type),
  cell_col     INTEGER,                     -- 0-based grid column (NULL if unlocated)
  cell_row     INTEGER,
  cell_id      TEXT,                        -- e.g. 'R2C5'
  bbox_norm    TEXT,                        -- JSON [x1,y1,x2,y2] normalised 0..1, nullable
  status       TEXT,                        -- present | seated | standing | parked | open | closed ...
  activity     TEXT,                        -- free-vocabulary activity, nullable
  count        INTEGER NOT NULL DEFAULT 1,
  confidence   REAL,
  source       TEXT NOT NULL,               -- yolo | vlm
  attrs_json   TEXT                         -- any extra factors the VLM reports (colour, size, state...)
);
CREATE INDEX IF NOT EXISTS ix_obs_frame      ON observations(frame_id);
CREATE INDEX IF NOT EXISTS ix_obs_space_type ON observations(space_id, object_type, ts_utc);
CREATE INDEX IF NOT EXISTS ix_obs_cell       ON observations(space_id, cell_id);

-- Scene-level factors, one row per processed frame.
CREATE TABLE IF NOT EXISTS scene_state (
  frame_id        INTEGER PRIMARY KEY REFERENCES frames(frame_id) ON DELETE CASCADE,
  space_id        TEXT NOT NULL,
  ts_utc          TEXT NOT NULL,
  people_yolo     INTEGER,
  people_vlm      INTEGER,
  count_disagree  INTEGER,                  -- 1 if |yolo - vlm| > 1 (flag for review)
  vehicles_yolo   INTEGER,
  lighting        TEXT,                     -- VLM label: daylight | artificial | dim | dark
  door_state      TEXT,
  tidiness        TEXT,
  weather_visible TEXT,                     -- for outdoor/parking views
  summary         TEXT,                     -- one-line VLM description
  attrs_json      TEXT
);

-- Detected changes between consecutive processed frames of a space.
CREATE TABLE IF NOT EXISTS events (
  event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  space_id    TEXT NOT NULL,
  ts_utc      TEXT NOT NULL,
  frame_id    INTEGER REFERENCES frames(frame_id),
  event_type  TEXT NOT NULL,                -- new_object_type | became_empty | became_occupied (see 009)
  object_type TEXT,
  cell_id     TEXT,
  resolution  TEXT NOT NULL DEFAULT 'interval',  -- stills: inferred between frames, not observed
  details_json TEXT
);

-- ---------------------------------------------------------------- derived

-- Measured mobility: fraction of appearances in the same cell as the previous appearance.
CREATE TABLE IF NOT EXISTS object_mobility (
  space_id        TEXT NOT NULL,
  object_type     TEXT NOT NULL,
  n_frames        INTEGER NOT NULL,
  stable_fraction REAL,
  measured_class  TEXT,                     -- fixed | semi_static | dynamic
  updated_utc     TEXT NOT NULL,
  PRIMARY KEY (space_id, object_type)
);

-- Per (space, object type, cell, weekday, hour) averages. n is always stored.
-- Frames with capture_status != 'ok' are excluded, never counted as zero.
CREATE TABLE IF NOT EXISTS baseline_cell (
  space_id    TEXT NOT NULL,
  object_type TEXT NOT NULL,
  cell_id     TEXT NOT NULL,
  dow         INTEGER NOT NULL,
  hour        INTEGER NOT NULL,
  n_frames    INTEGER NOT NULL,             -- valid frames in this bucket
  mean_count  REAL,
  std_count   REAL,
  p50_count   REAL,
  p90_count   REAL,
  presence_rate REAL,                       -- share of valid frames with count > 0
  ewm_count   REAL,                         -- exponentially weighted mean (recent weeks count more)
  updated_utc TEXT NOT NULL,
  PRIMARY KEY (space_id, object_type, cell_id, dow, hour)
);

-- Whole-space metrics per (weekday, hour): people, vehicles, brightness, ...
CREATE TABLE IF NOT EXISTS baseline_space (
  space_id    TEXT NOT NULL,
  metric      TEXT NOT NULL,                -- people_yolo | people_vlm | vehicles_yolo | brightness
  dow         INTEGER NOT NULL,
  hour        INTEGER NOT NULL,
  n_frames    INTEGER NOT NULL,
  mean_value  REAL,
  std_value   REAL,
  p50_value   REAL,
  p90_value   REAL,
  ewm_value   REAL,
  updated_utc TEXT NOT NULL,
  PRIMARY KEY (space_id, metric, dow, hour)
);

-- ---------------------------------------------------------------- views

-- The "wide" multi-dimensional table: time dims + space + object + position + status.
CREATE VIEW IF NOT EXISTS v_observations_wide AS
SELECT f.space_id, s.name AS space_name, s.space_type,
       f.ts_utc, f.ts_local, f.date, f.dow, f.dow_name, f.hour, f.minute, f.slot_15,
       f.week_of_year, f.is_weekend, f.part_of_day,
       f.brightness, f.capture_status,
       o.object_type, r.category, r.mobility_prior, m.measured_class AS mobility_measured,
       o.cell_id, o.cell_col, o.cell_row, o.bbox_norm,
       o.status, o.activity, o.count, o.confidence, o.source, o.attrs_json
FROM observations o
JOIN frames f          ON f.frame_id = o.frame_id
JOIN spaces s          ON s.space_id = f.space_id
JOIN object_registry r ON r.object_type = o.object_type
LEFT JOIN object_mobility m ON m.space_id = f.space_id AND m.object_type = o.object_type;

-- Per-frame, per-cell counts by object type (input for heatmaps and baselines).
CREATE VIEW IF NOT EXISTS v_cell_counts AS
SELECT f.space_id, f.frame_id, f.ts_utc, f.dow, f.hour, f.part_of_day,
       o.object_type, o.cell_id, SUM(o.count) AS n
FROM observations o JOIN frames f ON f.frame_id = o.frame_id
WHERE f.capture_status = 'ok' AND o.cell_id IS NOT NULL
GROUP BY f.frame_id, o.object_type, o.cell_id;

-- Detector and VLM counts side by side per frame and object type (never silently merged).
CREATE VIEW IF NOT EXISTS v_frame_counts AS
SELECT f.space_id, f.frame_id, f.ts_local, f.dow_name, f.hour, o.object_type,
       SUM(CASE WHEN o.source='yolo' THEN o.count END) AS count_yolo,
       SUM(CASE WHEN o.source='vlm'  THEN o.count END) AS count_vlm
FROM observations o JOIN frames f ON f.frame_id=o.frame_id
WHERE f.capture_status='ok'
GROUP BY f.frame_id, o.object_type;

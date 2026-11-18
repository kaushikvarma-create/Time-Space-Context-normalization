-- events.frame_id gets ON DELETE CASCADE like every other table that hangs off frames, so deleting a frame also
-- deletes its events instead of failing. SQLite cannot change a foreign key in place: rebuild the table.
-- event_type values written by the code: new_object_type | became_empty | became_occupied
CREATE TABLE events_new (
  event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  space_id     TEXT NOT NULL,
  ts_utc       TEXT NOT NULL,
  frame_id     INTEGER REFERENCES frames(frame_id) ON DELETE CASCADE,
  event_type   TEXT NOT NULL,
  object_type  TEXT,
  cell_id      TEXT,                          -- reserved: not written yet
  resolution   TEXT NOT NULL DEFAULT 'interval',
  details_json TEXT
);
INSERT INTO events_new(event_id, space_id, ts_utc, frame_id, event_type, object_type, cell_id, resolution, details_json)
  SELECT event_id, space_id, ts_utc, frame_id, event_type, object_type, cell_id, resolution, details_json FROM events;
DROP TABLE events;
ALTER TABLE events_new RENAME TO events;
CREATE INDEX IF NOT EXISTS ix_events_space_ts ON events(space_id, ts_utc);

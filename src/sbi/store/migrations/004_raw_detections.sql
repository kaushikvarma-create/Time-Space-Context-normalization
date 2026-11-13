-- 004 raw detections: every detector box down to the storage floor, with its confidence.
-- `observations` keeps only the boxes that passed the counting threshold (so every existing consumer is
-- unchanged); this table is the raw layer underneath, so thresholds can be chosen per object type later
-- (from human labels) and counts at any threshold can be derived without re-running the model.

CREATE TABLE IF NOT EXISTS detections (
  det_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  frame_id    INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
  space_id    TEXT NOT NULL REFERENCES spaces(space_id),
  ts_utc      TEXT NOT NULL,
  raw_label   TEXT NOT NULL,              -- label after the space's detector_map; taxonomy is applied on read
  confidence  REAL NOT NULL,
  bbox_norm   TEXT NOT NULL,              -- JSON [x1,y1,x2,y2] normalised 0..1
  cell_id     TEXT,
  counted     INTEGER NOT NULL            -- 1 = passed the counting threshold (also stored as an observation)
);
CREATE INDEX IF NOT EXISTS ix_det_frame ON detections(frame_id);
CREATE INDEX IF NOT EXISTS ix_det_space_label ON detections(space_id, raw_label, ts_utc);

-- Lowest confidence the detector stored for this frame. NULL = processed before this migration (only
-- counted boxes exist), so counts at lower thresholds are unknown for that frame, not zero.
ALTER TABLE frames ADD COLUMN detector_min_conf REAL;

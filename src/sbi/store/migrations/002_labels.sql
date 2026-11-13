-- 002 ground-truth labels (human, blind)
-- Labelled blind (model output hidden), counts only for the space's label_types. Used by src/eval/vlm_eval.py.
CREATE TABLE IF NOT EXISTS label_frames (
  frame_id    INTEGER PRIMARY KEY REFERENCES frames(frame_id) ON DELETE CASCADE,
  usable      INTEGER NOT NULL DEFAULT 1,     -- 0 = frame too unclear to label
  labeler     TEXT,
  labeled_utc TEXT NOT NULL,
  note        TEXT
);
CREATE TABLE IF NOT EXISTS labels (
  frame_id    INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
  object_type TEXT NOT NULL,
  true_count  INTEGER NOT NULL,
  PRIMARY KEY (frame_id, object_type)
);

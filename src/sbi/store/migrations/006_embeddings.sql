-- 006 whole-scene image vectors from a pluggable embedder (e.g. DINOv2).
-- The raw vector is kept, not just numbers derived from it: principal components, change scores and
-- distance-to-typical are computed when the table is built, so they can be refit or redefined without
-- re-running any model. One row per (frame, model); a different model or input size is a different
-- vector space, so it gets its own rows instead of replacing the old ones.
-- No row for a frame = it was never embedded: derived columns are NaN for that slot, not zero.
CREATE TABLE IF NOT EXISTS frame_embeddings (
  frame_id INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
  model    TEXT NOT NULL,       -- embedder version, e.g. dinov2-base@448
  dim      INTEGER NOT NULL,
  vec      BLOB NOT NULL,       -- float16, little-endian (see sbi.store.embeddings)
  PRIMARY KEY (frame_id, model)
);

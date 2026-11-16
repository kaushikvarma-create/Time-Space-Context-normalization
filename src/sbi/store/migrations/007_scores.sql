-- 007 text-prompt scores: how well each photo matches a plain-English description ("an empty parking area"),
-- from a pluggable scorer (e.g. SigLIP 2).
-- One raw score per (frame, scorer, prompt). The prompt text is the key, so what was asked is on record, a new
-- prompt can be asked of every stored photo later (`sbi score`), and concepts that average several paraphrases
-- are built when the table is built, so they can be regrouped without re-running a model.
-- Scores from different models are different scales, so the scorer version is part of the key.
-- No row = never scored for that prompt: derived columns are NaN for that slot, not zero.
CREATE TABLE IF NOT EXISTS frame_scores (
  frame_id INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
  scorer   TEXT NOT NULL,       -- scorer version, e.g. siglip2-base-patch16-224
  prompt   TEXT NOT NULL,
  value    REAL NOT NULL,       -- raw score (SigLIP: a logit, so 0 is an even-odds match)
  PRIMARY KEY (frame_id, scorer, prompt)
);

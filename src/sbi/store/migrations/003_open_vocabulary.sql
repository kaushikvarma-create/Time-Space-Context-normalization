-- 003 open vocabulary: raw labels, taxonomy links, relation triples, generic per-frame features.
-- Raw model labels are kept forever; canonical names are derived from them through the taxonomy, so merging
-- or un-merging names re-maps history without re-running any model.

ALTER TABLE observations ADD COLUMN raw_label TEXT;
UPDATE observations SET raw_label = object_type WHERE raw_label IS NULL;   -- best available for pre-003 rows

ALTER TABLE object_registry ADD COLUMN merged_into TEXT;                   -- set when this name was merged ('same')

-- Links between object names. 'same' = synonyms (child is merged into parent); 'is_a' = child is a kind of parent
-- (both stay as dimensions; parent gets a roll-up). Every decision is kept for audit and can be reverted.
CREATE TABLE IF NOT EXISTS object_links (
  link_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  child         TEXT NOT NULL,
  parent        TEXT NOT NULL,
  relation      TEXT NOT NULL CHECK (relation IN ('same', 'is_a')),
  status        TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'approved', 'rejected')),
  source        TEXT NOT NULL,              -- curator | human | config
  similarity    REAL,                       -- embedding cosine that shortlisted the pair
  evidence_json TEXT,                       -- e.g. the two yes/no answers
  model         TEXT,
  created_utc   TEXT NOT NULL,
  decided_utc   TEXT,
  UNIQUE (child, parent, relation)
);

-- Cache of "is every A a B?" answers so the curator never re-asks the same question of the same model.
CREATE TABLE IF NOT EXISTS label_judgments (
  a           TEXT NOT NULL,
  b           TEXT NOT NULL,
  model       TEXT NOT NULL,
  answer      INTEGER NOT NULL,             -- 1 = yes, 0 = no
  created_utc TEXT NOT NULL,
  PRIMARY KEY (a, b, model)
);

-- Relationships seen in a frame: (person, sitting_on, chair) x3. source = vlm (read from the photo) or
-- geometry (computed from detector boxes, e.g. person box overlapping a chair box).
CREATE TABLE IF NOT EXISTS relations (
  rel_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  frame_id    INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
  space_id    TEXT NOT NULL,
  ts_utc      TEXT NOT NULL,
  subject     TEXT NOT NULL,
  predicate   TEXT NOT NULL,
  object      TEXT NOT NULL,
  raw_subject TEXT,
  raw_object  TEXT,
  count       INTEGER NOT NULL DEFAULT 1,
  source      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_rel_frame ON relations(frame_id);
CREATE INDEX IF NOT EXISTS ix_obs_raw ON observations(raw_label);

-- Generic numeric features per frame from pluggable extractors (embedding PCA, change scores, ...).
CREATE TABLE IF NOT EXISTS frame_features (
  frame_id  INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
  name      TEXT NOT NULL,
  value     REAL,
  extractor TEXT NOT NULL,
  version   TEXT NOT NULL,
  PRIMARY KEY (frame_id, name)
);

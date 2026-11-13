-- 005 segmentation masks: an outline for every stored detector box (optional segmenter, e.g. SAM 2).
-- The mask itself is kept (low resolution, compressed), not just numbers derived from it, so areas, per-cell
-- coverage and duplicate removal can be re-derived with different rules without re-running any model.

ALTER TABLE detections ADD COLUMN mask BLOB;            -- encoded by sbi.perception.masks; NULL = not segmented
ALTER TABLE detections ADD COLUMN mask_area REAL;       -- share of the image covered, measured at full resolution

-- Which segmenter outlined this frame's boxes. NULL = none ran or it failed: mask-derived numbers are
-- unknown for that frame, not zero.
ALTER TABLE frames ADD COLUMN segmenter_version TEXT;

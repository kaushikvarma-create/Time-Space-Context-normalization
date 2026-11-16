# Space Behaviour Intelligence (SBI): baseline pipeline

Camera stills every 15 min -> YOLO (counts, boxes, grid cell) + a vision LLM (semantics, scene state) ->
SQLite -> time averages by (weekday, hour). Forecasting (Chronos-2 etc.) comes later, on top of this data.
See `PLAN.md` for scope, decisions and known issues.

## Layout

```
config/
  pipeline.yaml          global settings (timezone, detector, scene reader, paths). No secrets.
  spaces/*.yaml          one file per space: camera channel, grid, vocabulary, aliases, scene fields
scripts/
  tick.sh                what cron runs (flock, timeout, IST log, rotation)
  label.sh               blind labelling UI
src/sbi/
  settings.py spaces.py  typed, validated config (unknown keys / bad references fail loudly)
  errors.py              SbiError hierarchy: ConfigError CameraError DetectorError SegmenterError EmbedderError
                         VLMError StorageError
  timeutil.py            IST time dimensions; ts_utc is canonical, everything else is local time
  grid.py                box -> grid cell (the VLM never sees the grid)
  ingest/                camera sources (NVR RTSP) and capture (image + frame row)
  perception/            Detector / Segmenter / Embedder / SceneReader / FrameSource interfaces, YOLO, SAM 2,
                         DINOv2, LM Studio reader, normaliser, mask codec
  store/                 connection, transactions, numbered migrations, data-access modules
  pipeline.py            Engine 1: frame -> observations, scene state, events (one atomic transaction)
  baseline.py            per (space, weekday, hour) averages, always with n
  embedding.py           catch-up path: image vectors for frames that have none
  tick.py cli.py         orchestration with per-space failure isolation; `python -m sbi <command>`
  labeling/ eval/        ground-truth labelling, accuracy report, label-free stability report
tests/                   pytest, fakes only (no camera / GPU / LM Studio needed)
```

## Run

```bash
cp .env.example .env            # fill in SBI_NVR_* (gitignored, chmod 600)
PYTHONPATH=src python3 -m sbi tick        # one tick (cron does this every 15 min via scripts/tick.sh)
PYTHONPATH=src python3 -m sbi eval        # accuracy vs human labels
scripts/label.sh                          # label frames (blind), then look at the eval
.venv/bin/python -m pytest                # tests
```

Other commands: `migrate`, `baseline [--space ID]`, `reprocess --space ID [--frame-id N]` (re-extract stored
frames after a prompt/model change; results are replaced, not duplicated), `curate` (link synonyms and hierarchy
among discovered object names; `scripts/curate.sh` for scheduled runs), `links` / `link-set ID approved|rejected`
(review or undo links), `table --space ID` (Chronos-ready table + column catalog in `data/exports/`),
`reprocess --space ID --detector-only` (re-run only the detector; each frame's stored VLM answer is replayed
unchanged, no VLM call), `stability --space ID --start '2026-11-14 00:00' --end '2026-11-14 05:00'` (how much
each count wobbles inside a quiet window, in IST; a noise check that needs no labels), `embed --space ID`
(store image vectors for frames that have none: history, or frames where the embedder failed),
`table --space ID --fit-end '2026-11-14 12:00'` (learn the image-vector components from earlier slots only).
Exit codes: 0 ok, 1 some space failed, 2 fatal config error.

## Conventions

- **Times:** IST everywhere a human sees them (logs, filenames, UI). `ts_utc` is stored as the canonical instant.
- **Errors:** library code raises typed `SbiError`s and never calls `sys.exit`. Catch the specific subclass you can
  handle; unexpected exceptions are bugs and are logged with a traceback at the tick boundary only.
- **Missing is not zero:** camera down / VLM failure are stored as NULL / `camera_down`, never as a count of 0.
- **Writes:** every multi-row write goes through `store.db.transaction()` (rollback on any exception). Slow work
  (detector, VLM) happens before the transaction opens.
- **Schema:** change it only via a new `store/migrations/NNN_name.sql` (idempotent, applied atomically).
- **Secrets:** only in `.env` / environment. Credentials are `SecretStr` and masked in logs and errors.
- **Paths in the DB** are relative to the project root.
- **Never give the VLM a list of object names** (it reports listed things that are not there; see
  `docs/research/2026-11-13-photo-to-numbers.md`). Prompts are open-vocabulary; naming is cleaned up afterwards by
  the taxonomy (`same` / `is_a` links, raw labels kept forever, history re-mapped on any change).
- **Detector thresholds:** the detector stores every box down to `detector.store_conf` (0.05) in `detections`,
  with its confidence. Only boxes at or above the counting threshold (`detector.conf`, or a per-label
  `detector.class_conf`) become observations, links and totals. The table also counts at each
  `table.conf_levels` level (`count.yolo_c05.<type>`); those variants are never default model inputs. Pick
  per-label thresholds from human labels (`sbi eval`), not by eye; `sbi stability` shows which are steadier.
- **Outlines:** an optional segmenter (`segmenter.kind: sam2`; `none` switches it off) outlines every stored
  box. SAM 2 only traces what is inside a box YOLO found; it does not find or name anything. The outline itself
  is stored (`detections.mask`, about 130 bytes each), so areas, per-cell coverage and duplicate removal are
  derived in the table (`area.sam2*`, `cellarea.sam2`, `count.sam2*`) and can be re-derived with other rules
  without re-running a model. A segmenter failure never loses a frame: its outline columns are NaN.
- **Image vectors:** an optional embedder (`embedder.kind: dinov2`; `none` stops new vectors, old ones stay) stores
  one whole-scene vector per photo (`frame_embeddings`, float16, 3 KB). It needs no detector and no VLM, so it
  describes changes the object counts miss. The table derives `emb.<model>.pc1..pcN` (main directions of scene
  variation), `drift_prev` (change since the last slot) and `drift_typical` (distance from the typical scene) from
  the stored vectors, so they can be redefined without re-running a model. The components are learned from the
  data in the table; for a back-test pass `--fit-end` so the future is kept out of the fit. A model change is a new
  vector space (`model` is part of the key), so old and new vectors never mix. An embedder failure never loses a
  frame: its image columns are NaN.
- **Chronos table:** regular IST grid, NaN for missing slots (never 0, never a dropped row), every dimension kept;
  the catalog says which columns are eligible and why, and a small default set is selected for models.

## Extending

- **New space:** add `config/spaces/<id>.yaml` (copy an existing one). No code changes; the next tick picks it up.
- **New detector / segmenter / embedder / scene reader / camera source:** implement the Protocol in `perception/types.py`, register its
  `kind` in `perception/factory.py` and in the settings model, select it in `config/pipeline.yaml`.
- **New object type:** nothing to do. Unknown labels become `object_registry` rows flagged `auto_new`, and
  `sbi curate` links them to existing names (synonym merge or is_a roll-up).
- **Multi-room map:** `spaces.site_id / map_x / map_y / map_transform_json` are reserved and unused so far.

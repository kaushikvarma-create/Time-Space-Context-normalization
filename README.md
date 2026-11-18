# Space Behaviour Intelligence (SBI)

Camera stills every 15 min -> five models read each photo into numbers (YOLO boxes, SAM 2 outlines, a vision LLM,
a DINOv2 image vector, SigLIP 2 description scores) -> SQLite -> one table per space -> Chronos-2 forecasts the next
4 hours, every hour. A read-only dashboard shows the numbers, the forecasts and how accurate they were.

- `PLAN.md`: scope, decisions, benchmarks and known issues, in date order.
- `docs/chronos-data-flow.md`: exactly what is given to Chronos-2 and what comes back.
- `docs/dashboard-guide.md`: what every part of the dashboard means.
- `docs/repo-map.md`: map of the code and the sweep of 18 Nov 2026 (what was checked, what is open).

## Layout

```
config/
  pipeline.yaml          global settings (timezone, detector, scene reader, paths). No secrets.
  spaces/*.yaml          one file per space: camera channel, grid, vocabulary, aliases, scene fields
scripts/
  tick.sh                what cron runs (flock, timeout, IST log, rotation)
  dashboard.sh           the dashboard (http://localhost:8502)
  label.sh               blind labelling UI
  curate.sh              scheduled vocabulary clean-up
  detector_bench/        run any detector over the benchmark photos and score it against labels
  windows/               keep WSL running after a Windows restart
src/sbi/
  settings.py spaces.py  typed, validated config (unknown keys / bad references fail loudly)
  errors.py              SbiError hierarchy: ConfigError CameraError DetectorError SegmenterError EmbedderError
                         ScorerError ForecastError VLMError StorageError
  timeutil.py            IST time dimensions; ts_utc is canonical, everything else is local time
  grid.py                box -> grid cell (the VLM never sees the grid)
  ingest/                camera sources (NVR RTSP) and capture (image + frame row)
  perception/            Detector / Segmenter / Embedder / Scorer / SceneReader / FrameSource interfaces, YOLO,
                         SAM 2, DINOv2, SigLIP 2, LM Studio reader, normaliser, mask codec
  store/                 connection, transactions, numbered migrations, data-access modules
  pipeline.py            Engine 1: frame -> observations, scene state, events (one atomic transaction)
  baseline.py            per (space, weekday, hour) averages, always with n
  sitecal.py             known-in-advance day types: site holiday file (+ optional `holidays` package)
  dashboard/             read-only Streamlit dashboard (scripts/dashboard.sh)
  forecast/              table builder; Forecaster interface (Chronos-2, baselines); target/covariate choice;
                         day-type profiles; leak-free backtest; metrics; live runner; live accuracy
  embedding.py scoring.py  catch-up paths: image vectors / probe scores for frames that lack them
  imageio.py             read a stored frame's image
  tick.py cli.py         orchestration with per-space failure isolation; `python -m sbi <command>`
  labeling/ eval/        ground-truth labelling, accuracy report, label-free stability report
tests/                   pytest, fakes only (no camera / GPU / LM Studio needed)
```

## Running v1.0

What runs on its own (cron, every 15 minutes, `scripts/tick.sh` with the project `.venv` interpreter):
capture both cameras -> YOLO boxes + SAM 2 outlines + Qwen scene reading + DINOv2 image vector + SigLIP 2 probes ->
SQLite -> baselines; and every `forecast.every_minutes` (60) a Chronos-2 forecast per space, stored with its inputs.
Every stage is isolated: a failing model is logged and skipped, capture carries on.

What you look at:
- `scripts/dashboard.sh` -> http://localhost:8502 (this machine only, read-only): now and the next 4 hours,
  forecast vs actual (also by day of week), the pipeline (what is extracted and what reaches the model), every
  dimension, and a guide. See `docs/dashboard-guide.md`.
- `PYTHONPATH=src python3 -m sbi accuracy --space parking`: stored forecasts scored against what then happened.
- `PYTHONPATH=src python3 -m sbi backtest --space parking`: simulated forecasts from past moments (rerun weekly).
- `data/logs/tick.log`: everything the tick did, in IST.

Labelling (`scripts/label.sh`, http://localhost:8501): photos come evenly from weekday/weekend x night, morning,
afternoon, evening; dense areas can be counted in four full-resolution quarter tabs. Then `sbi eval` compares your
counts with YOLO at every confidence threshold (with and without outline de-duplication) and with the VLM, and
suggests `detector.class_conf` per object type (the strictest threshold within a small margin of the best).

What you maintain: `config/site_calendar.yaml` (paste the office holiday list), `forecast_targets` and `probes` in
the space files, and ~50 labelled photos per camera (`scripts/label.sh`) to measure accuracy against truth.

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
`accuracy --space ID [--since DATE]` (score stored forecasts against what happened, vs persistence),
`forecast --space ID [--variant V] [--model chronos2|seasonal_mean|...]` (forecast with Chronos-2 and store the run and
its quantiles in `forecast_runs` / `forecasts`), `backtest --space ID` (forecast from many past moments, score against
what happened, compare models and covariate sets), `score --space ID` (score stored photos against the space's probe prompts: history, or a prompt you just added),
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
- **Detector:** YOLO26x at input size 1280, chosen by benchmark (`scripts/detector_bench/`, results in `PLAN.md`).
  Weights go in `models/` (not in git): `yolo26x.pt`, `sam2.1_l.pt`. After changing the detector, run
  `sbi reprocess --space ID --detector-only` so history is read by one detector; forecasts made before that are no
  longer scored.
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
- **Probes:** a space file can list `probes`: plain-English descriptions grouped by concept, several paraphrases
  each ("an empty parking area", "a parking lot with no vehicles", ...). An optional scorer (`scorer.kind: siglip2`)
  rates every photo against every prompt and the raw score is stored per (frame, scorer, prompt). The table has
  `probe.<scorer>.<concept>` = the mean of the concept's paraphrases (NaN if any prompt is missing for a frame).
  Add a prompt, run `sbi score`, and only the new question is asked of every stored photo. Checked against a
  detector, "people" (cabin) and "bikes" (parking) hold up, also within daytime; tidy/messy, parking "car" and
  "person" do not, so treat a probe as a measurement only after `sbi stability` and labels agree.
- **Forecasting:** `forecast_targets` in the space file say what to forecast (names or wildcards, matched against the
  table's model-ready columns). A *variant* is a named input set (`forecast.variants`: univariate, calendar,
  detector, full, all, profile, all_profile). The calendar and the day-type profiles (`prof.<target>`) are known in
  advance; everything else is past-only. `all` gives every model-ready column; `detector` and `full` give a capped
  selection. See `docs/chronos-data-flow.md`. Every forecast is stored with the model, variant, targets and covariates it used. Count-like targets are
  clipped at 0. The backtest rebuilds the column catalog from the history before each origin and gives the model
  only rows up to it, so it cannot see the future; baselines (persistence, seasonal naive, seasonal mean) are the
  bar. The live variant is `all_profile` (set in `config/pipeline.yaml`); rerun `sbi backtest` weekly to check it.
- **Chronos table:** regular IST grid, NaN for missing slots (never 0, never a dropped row), every dimension kept;
  the catalog says which columns are eligible and why, and a small default set is selected for models.

## Extending

- **New space:** add `config/spaces/<id>.yaml` (copy an existing one). No code changes; the next tick picks it up.
- **New forecaster (Timestral, TimesFM, a two-stage model):** implement `Forecaster` in `forecast/models.py`, register it in
  `build_forecaster`; live runs, the backtest and storage need no change.
- **New detector / segmenter / embedder / scorer / scene reader / camera source:** implement the Protocol in `perception/types.py`, register its
  `kind` in `perception/factory.py` and in the settings model, select it in `config/pipeline.yaml`.
- **New object type:** nothing to do. Unknown labels become `object_registry` rows flagged `auto_new`, and
  `sbi curate` links them to existing names (synonym merge or is_a roll-up).
- **Multi-room map:** `spaces.site_id / map_x / map_y / map_transform_json` are reserved and unused so far.

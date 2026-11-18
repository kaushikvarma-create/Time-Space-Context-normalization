# SBI technical deep dive

This document covers every module, setting, database table, rule and test in the repository, and how they fit
together. The figures in it were measured on the live system on 18 Nov 2026 unless stated otherwise. Code
references are written as `path::function`, relative to `src/sbi/` unless they start with another folder.

Contents

0. Overview: purpose, runtime, architecture, the life of one photo and one forecast, glossary
1. Configuration, capture and operations
2. From photo to stored numbers
3. The table and forecasting
4. Showing, checking and testing
5. Known limits and open work

## 0. Overview

### 0.1 What the system does

SBI measures what happens in a space from still photos and forecasts it.

- **Input:** one photo from each of two cameras on one Dahua-style NVR (RTSP), every 15 minutes. The admin cabin
  is channel 5 and the parking area is channel 3, both on the main stream at 2880 x 1620.
- **Measurement:** five models read each photo:
  1. a detector (YOLO26x) that boxes objects;
  2. an outliner (SAM 2) that traces each box;
  3. a vision-language model (Qwen 3.5 9B on LM Studio) that describes the scene in structured JSON;
  4. an image embedder (DINOv2 base) that summarises the whole picture as a vector;
  5. a text-image scorer (SigLIP 2 base) that rates the photo against plain-English descriptions.
- **Storage:** raw readings go into SQLite (`data/sbi.db`, WAL mode). Images are stored as files under
  `data/images/`.
- **Table:** for each space, one row per 15-minute slot and one column per measured thing. On 18 Nov 2026 that was
  723 columns for the cabin and 741 for parking.
- **Forecast:** every hour, Chronos-2 (`amazon/chronos-2`, zero-shot) reads the table and forecasts selected series
  4 hours ahead. It gives a middle estimate and a 10% to 90% range.
- **Display:** a read-only Streamlit dashboard, plus command-line reports.

### 0.2 Runtime environment

| Item | Value |
|---|---|
| Host | Windows 11 PC running WSL2 (Ubuntu 24.04) with systemd |
| GPU | NVIDIA RTX 5090, 32 GB |
| Python | 3.12 in `.venv` (created with system site packages); package code in `src/sbi`, run with `PYTHONPATH=src` |
| Key libraries | ultralytics 8.4.71, torch 2.11.0+cu128, transformers 5.15.1, chronos-forecasting 2.3.2, streamlit 1.67.0, altair 6.3.0, pandas 2.3.3, numpy 2.5.2, pydantic 2.11.10, opencv-python 4.11.0.86 |
| Scheduler | user crontab `*/15 * * * * .../scripts/tick.sh` |
| Services | systemd user service `sbi-dashboard` (lingering enabled); Windows scheduled task `SBI-WSL-Keepalive` |
| VLM host | LM Studio on the LAN, OpenAI-compatible API |
| Model weights | `models/yolo26x.pt`, `models/sam2.1_l.pt`; Hugging Face cache for DINOv2, SigLIP 2 and Chronos-2 |
| Disk use | photos about 190 MB a day, database about 9 MB a day, forecasts about 87,000 rows a day |

### 0.3 Architecture

```
                     cron, every 15 min
                            |
                    scripts/tick.sh  (flock, timeout 600 s, IST log)
                            |
                     sbi tick  [tick.py]
          +-----------------+-------------------+
          | for each space (config/spaces/*.yaml), isolated
          |
   capture [ingest/]  ->  data/images/<space>/<date>/<file>.jpg + frames row
          |
   process_frame [pipeline.py]
     detector (YOLO26x) -> outliner (SAM 2) -> image vector (DINOv2) -> description scores (SigLIP 2) -> VLM
          |                      (each optional stage fails on its own)
     one transaction: detections, observations, relations, scene_state, events, frame_embeddings, frame_scores
          |
   baselines [baseline.py]
          |
   on the hour: run_scheduled [forecast/schedule.py]
     table_for -> build_table (+ per-photo cache) -> prepare (targets, inputs, calendar, profiles)
     -> Chronos-2 predict_df -> clip counts at 0 -> forecast_runs + forecasts

   dashboard [dashboard/]  reads the same table and stored forecasts (read-only connection)
   sbi backtest / accuracy / eval / stability  -> reports in data/exports/
```

### 0.4 The life of one photo

The example is the cabin photo of 18 Nov 2026 at 11:00 IST.

1. **11:00:00** cron starts `scripts/tick.sh`. It takes a file lock so that runs never overlap, then starts
   `python -m sbi tick` under a 600-second timeout.
2. **11:00:03 to 11:00:07** the tick loads the models once for all spaces: detector, outliner, embedder, scorer,
   and a client for the VLM endpoint. A model that fails to load is logged, and its stage is skipped for this run.
3. The capture step opens the RTSP stream and discards a few buffered frames so that it keeps the freshest one. It
   then saves the JPEG and inserts a `frames` row: UTC time, local IST time fields, image metrics (brightness,
   contrast, sharpness) and the capture status.
4. `process_frame` runs the detector. Every box down to confidence 0.05 comes back with its label, already mapped
   through the space's `detector_map`. A box is marked `counted` when its confidence reaches the counting threshold
   for its type: 0.25 by default, 0.10 for chairs and motorcycles. Two box rules then remove duplicates.
5. The outliner traces every stored box, counted or not. When two counted boxes of one type have outlines that
   overlap by IoU 0.5 or more, the weaker box stops being counted. It stays stored.
6. The embedder stores one 1536-number vector of the whole picture. The scorer stores one score per description
   prompt: 63 prompts for the cabin and 60 for parking.
7. The VLM receives the photo with the camera's burned-in timestamp blacked out. It returns people (with posture),
   open-vocabulary objects with counts, relations between objects, and scene fields such as lighting and door
   state.
8. All results for the photo are written in one transaction. Existing results for the photo are cleared first, so
   reprocessing replaces rather than duplicates. The tables written are `detections`, `observations`, `relations`
   (from the VLM and from box geometry), `scene_state`, `events`, `frame_embeddings` and `frame_scores`. Finally the
   frame is marked processed.
9. The baseline for this weekday and hour is rebuilt (`baseline.rebuild`).
10. At 11:00 a forecast is due, because the slot minute is a multiple of 60. The cabin table is built. Photos that
    have not changed come from the per-photo cache, so this takes about 0.6 seconds. Chronos-2 forecasts 45 series
    for 16 slots in about 0.3 seconds, and the run is stored.
11. **11:00:30** the tick logs `tick done: 2 ok, 0 failed`.

### 0.5 The life of one forecast

1. **Table** (section 3.1): 489 rows (13 Nov 09:00 to now) by 723 columns, with a catalog that marks 412 columns
   usable.
2. **Targets** (section 3.4): `forecast_targets` patterns are matched against usable columns, giving 45 for the
   cabin.
3. **Inputs** (section 3.4): input set `all_profile`, which is:
   - every other usable column as a past-only input (360 for the cabin);
   - 7 calendar columns;
   - one day-type profile per target. These are known for the future too.
4. **Frames** (section 3.5): a past frame (rows x columns) and a future frame (16 rows of calendar and profiles).
5. **Model** (section 3.6): `Chronos2Pipeline.predict_df` with quantiles 0.1, 0.5 and 0.9.
6. **Post-processing:** counts, areas and image statistics are clipped at 0.
7. **Storage** (section 3.8): one `forecast_runs` row recording exactly what was given, and one `forecasts` row per
   series, slot and quantile.
8. **Scoring** (section 3.9): once the predicted time has passed, `sbi accuracy` and the dashboard compare each
   value with the measurement and with the "nothing changes" guess. Runs made before their photos were re-read are
   skipped.

### 0.6 Glossary

| Term | Meaning |
|---|---|
| space | one camera view, configured by one file in `config/spaces/` |
| slot | a 15-minute step, named by its start time in IST |
| frame | one captured photo, one row in `frames` |
| tick | one scheduled run: capture, read, store, forecast when due |
| detection | one detector box, stored with confidence and outline, counted or not |
| observation | a counted thing from one source (`yolo` or `vlm`), with count and grid cell |
| relation | subject, predicate, object and count, for example person sitting_on chair; source `vlm` or `geometry` |
| taxonomy | canonical object names, synonyms (`same` links) and groups (`is_a` links) |
| probe | a plain-English description scored on every photo; a concept is the mean of its paraphrases |
| dimension, series, column | one column of the table |
| catalog | per-column statistics and the reason a column is or is not usable by the model |
| target | a series Chronos-2 forecasts |
| past-only input | a series the model reads for the past only |
| known-ahead input | a series given for the future too: calendar and day-type profiles |
| variant (input set) | a named rule for which inputs the model gets: `all_profile`, `all`, `calendar`, ... |
| profile | for a target, its average at the same clock time on days of the same kind |
| run | one stored forecast: a `forecast_runs` row and its values |
| lead | how far ahead a forecast value is: step x 15 minutes |
| skill | 1 - (average miss / average miss of a simpler guess) |
| coverage, inside band | share of measured values inside the 10% to 90% range |
| backtest | forecasts replayed from past moments, using only the history before each moment |

## 1. Configuration, capture and operations

This section covers how the system is configured, how one photo per space is captured, how the 15-minute tick runs, the command line, and the scripts that keep everything running. It is based on these files:

| File | Role |
|---|---|
| `src/sbi/settings.py` | Global settings models and loaders (`config/pipeline.yaml` + environment) |
| `src/sbi/spaces.py` | Per-space configuration model and loaders (`config/spaces/*.yaml`) |
| `src/sbi/errors.py` | Typed error hierarchy |
| `src/sbi/timeutil.py` | Timezone lookup, UTC clock, derived time fields |
| `src/sbi/logging_setup.py` | Log formatter with local-time timestamps |
| `src/sbi/imageio.py` | Reading a stored frame's image from disk |
| `src/sbi/ingest/__init__.py`, `sources.py`, `capture.py` | Camera sources and the capture step |
| `src/sbi/tick.py` | One scheduled tick over all spaces |
| `src/sbi/cli.py`, `src/sbi/__main__.py`, `src/sbi/__init__.py` | Command line entry point |
| `config/pipeline.yaml`, `config/spaces/*.yaml`, `config/site_calendar.yaml`, `.env.example` | Configuration files |
| `pyproject.toml`, `.gitignore`, `.streamlit/config.toml` | Packaging, version control, Streamlit theme |
| `scripts/*.sh`, `scripts/systemd/sbi-dashboard.service`, `scripts/windows/install-autostart.ps1` | Operations |

### 1.1 Settings (`src/sbi/settings.py`)

The module docstring states the rule: everything tunable is in the YAML, and unknown keys are rejected so typos fail loudly.

#### 1.1.1 Strictness

Every settings model derives from `_Strict`, a pydantic `BaseModel` with `ConfigDict(extra="forbid", frozen=True)`.

- `extra="forbid"`: any key in `config/pipeline.yaml` that is not a declared field raises a validation error. `load_settings` turns that into a `ConfigError`.
- `frozen=True`: a loaded `Settings` object (and every nested model) cannot be modified after construction.
- Fields declared with `Literal[...]` accept only the listed strings.
- `Field(..., ge=, gt=, le=, lt=)` bounds are checked at load time.

#### 1.1.2 `Settings` (top level)

| Field | Type | Default | Notes |
|---|---|---|---|
| `root` | `Path` | required | Set by `load_settings`, never read from the YAML. Project root against which relative paths resolve. |
| `timezone` | `str` | `"Asia/Kolkata"` | Validated by `get_zone` after construction. Read by `cli.main` for `configure_logging`. Not used as the default for spaces (see 1.2). |
| `paths` | `Paths` | `Paths()` | See 1.1.3. |
| `frame_source` | `FrameSourceSettings` | `FrameSourceSettings()` | See 1.1.4. |
| `detector` | `DetectorSettings` | `DetectorSettings()` | See 1.1.5. |
| `segmenter` | `SegmenterSettings` | `SegmenterSettings()` | See 1.1.6. |
| `embedder` | `EmbedderSettings` | `EmbedderSettings()` | See 1.1.7. |
| `scorer` | `ScorerSettings` | `ScorerSettings()` | See 1.1.8. |
| `forecast` | `ForecastSettings` | `ForecastSettings()` | See 1.1.9. |
| `calendar` | `CalendarSettings` | `CalendarSettings()` | See 1.1.10. |
| `scene_reader` | `SceneReaderSettings` | required (no default) | See 1.1.11. Its `base_url` and `model` are also required. |
| `baseline` | `BaselineSettings` | `BaselineSettings()` | See 1.1.13. |
| `processing` | `ProcessingSettings` | `ProcessingSettings()` | See 1.1.13. |
| `curation` | `CurationSettings` | `CurationSettings()` | See 1.1.12. |
| `table` | `TableSettings` | `TableSettings()` | See 1.1.14. |

Methods and properties of `Settings`:

| Member | Purpose |
|---|---|
| `Settings.resolve(relative)` | Returns the path unchanged if absolute, else `root / relative`. Used for every configured path and for CLI `--out` directories. |
| `Settings.db_path` | `resolve(paths.db)`. Opened by the tick, the CLI, the dashboard (read-only) and the labelling app. |
| `Settings.image_dir` | `resolve(paths.images)`. Passed to `capture_frame` by the tick. |
| `Settings.log_dir` | `resolve(paths.logs)`. No code under `src/sbi` reads this property. |
| `Settings.cache_dir` | `resolve(paths.cache)`. `forecast/build.py` opens `cache_dir / "table_features.db"` as the per-photo feature cache. |

#### 1.1.3 `Paths`

All are strings relative to the project root unless absolute.

| Field | Type | Default | `pipeline.yaml` | Read by |
|---|---|---|---|---|
| `db` | `str` | `data/sbi.db` | `data/sbi.db` | `Settings.db_path` |
| `images` | `str` | `data/images` | `data/images` | `Settings.image_dir` (capture) |
| `logs` | `str` | `data/logs` | `data/logs` | `dashboard/app.py` reads `resolve(paths.logs) / "tick.log"` for tick timing and problems. The tick log itself is written by `scripts/tick.sh` to the hard-coded `data/logs/tick.log`. |
| `cache` | `str` | `data/cache` | `data/cache` | `Settings.cache_dir`. Code comment: derived, safe to delete, rebuilt on the next table build. |

#### 1.1.4 `FrameSourceSettings`

| Field | Type | Default | `pipeline.yaml` | Validation | Read by |
|---|---|---|---|---|---|
| `kind` | `Literal["nvr_rtsp"]` | `nvr_rtsp` | `nvr_rtsp` | only value allowed | `perception/factory.py::build_frame_source` |
| `rtsp_transport` | `Literal["tcp","udp"]` | `tcp` | `tcp` | | `ingest/sources.py::NvrRtspSource.grab` |
| `warmup_frames` | `int` | `5` | `5` | `>= 1` | `NvrRtspSource.grab` |
| `timeout_s` | `float` | `10` | `10` | `> 0` | `NvrRtspSource.grab` (converted to ms) |

#### 1.1.5 `DetectorSettings`

| Field | Type | Default | `pipeline.yaml` | Validation | Read by / meaning |
|---|---|---|---|---|---|
| `kind` | `Literal["yolo"]` | `yolo` | `yolo` | | `build_detector` |
| `weights` | `str` | `models/yolo26x.pt` | `models/yolo26x.pt` | | `YoloDetector`; resolved against root; `DetectorError` if the file is missing |
| `imgsz` | `int` | `1280` | `1280` | `>= 320` | YOLO `predict(imgsz=)`; part of the detector version string `<weights name>@<imgsz>` |
| `conf` | `float` | `0.25` | `0.25` | `0 < x < 1`, and `>= store_conf` | Default counting threshold: boxes at or above it are counted |
| `store_conf` | `float` | `0.05` | `0.05` | `0 < x < 1` | Storage floor: passed as YOLO `predict(conf=)`; every box down to it is stored, uncounted below the counting threshold. Also stored as `detector.min_conf`. |
| `class_conf` | `dict[str, float]` | `{}` | `{chair: 0.10, motorcycle: 0.10}` | each value `>= store_conf` and `< 1` | Per-label counting thresholds, via `count_threshold` |
| `iou` | `float` | `0.5` | `0.5` | `0 < x < 1` | YOLO `predict(iou=)` |
| `dedupe_labels` | `list[str]` | `["person"]` | `[person]` | | Labels where a box mostly inside a stronger box is dropped (`dedupe_contained`) |
| `dedupe_containment` | `float` | `0.8` | `0.8` | `0 < x <= 1` | Containment share for `dedupe_contained` |
| `dedupe_same_label_iou` | `float` | `0.6` | `0.6` | `0 < x <= 1` | IoU for `dedupe_same_label_iou` |
| `dedupe_outline_iou` | `float` | `0.0` | `0.5` | `0 <= x <= 1` | Passed by the tick and `reprocess` to `process_frame`. Counted boxes of one type whose outlines overlap this much count once. Needs the segmenter; `0` = off. |

Validator `_thresholds_above_floor` (model validator, after): for `conf` and every `class_conf[label]`, requires `store_conf <= value < 1`. The error names the field, e.g. `detector.class_conf[chair]=...`.

Method `count_threshold(label)` returns `class_conf.get(label, conf)`. It is called in `perception/detector_yolo.py::build_detections` to set each detection's `counted` flag.

The YAML comments record why the values were chosen: `yolo26x.pt` at 1280 was picked by a benchmark on 18 Nov 2026 (chairs exact in 80% of cabin photos vs 31% for yolo12x at 1920; 1920 gave 59%); `class_conf` 0.10 for chair and motorcycle was set from labels and the known chair count, "with duplicate outlines removed, 0.10 beats 0.25 for both".

#### 1.1.6 `SegmenterSettings`

| Field | Type | Default | `pipeline.yaml` | Validation | Read by / meaning |
|---|---|---|---|---|---|
| `kind` | `Literal["none","sam2"]` | `none` | `sam2` | | `build_segmenter`; `none` returns no segmenter, so every mask-derived column stays empty |
| `weights` | `str` | `models/sam2.1_l.pt` | `models/sam2.1_l.pt` | | `Sam2Segmenter`; resolved against root; `SegmenterError` if missing |
| `mask_width` | `int` | `384` | `384` | `32 <= x <= 2048` | Width of the stored low-resolution outline; part of version `<weights name>@<mask_width>` |
| `batch` | `int` | `64` | `64` | `>= 1` | Boxes per model call |

#### 1.1.7 `EmbedderSettings`

The docstring states that `model` and `width` are kept even when `kind` is `none`, so the table can still read vectors stored earlier.

| Field | Type | Default | `pipeline.yaml` | Validation | Read by / meaning |
|---|---|---|---|---|---|
| `kind` | `Literal["none","dinov2"]` | `none` | `dinov2` | | `build_embedder`; `none` = no new vectors |
| `model` | `str` | `facebook/dinov2-base` | `facebook/dinov2-base` | | Hugging Face id or local path |
| `width` | `int` | `448` | `448` | `56 <= x <= 1792`, and multiple of 14 (`_patch_multiple`) | Input width in pixels; height follows the photo's aspect |

Property `version` returns `f"{Path(model).name}@{width}"`, e.g. `dinov2-base@448`. It is stored with every vector and used to select vectors back out (`forecast/build.py` passes it as `embed_model`).

#### 1.1.8 `ScorerSettings`

| Field | Type | Default | `pipeline.yaml` | Read by / meaning |
|---|---|---|---|---|
| `kind` | `Literal["none","siglip2"]` | `none` | `siglip2` | `build_scorer`; `none` = no new scores |
| `model` | `str` | `google/siglip2-base-patch16-224` | same | Hugging Face id or local path |

Property `version` returns `Path(model).name`. `forecast/build.py` passes it as `probe_scorer`; `scoring.py` uses it to find frames missing scores.

#### 1.1.9 `ForecastSettings`

Module constants used here:

| Constant | Value | Meaning (from comments) |
|---|---|---|
| `ALL_KINDS` | `"*"` | In a variant: every model-ready column the pipeline extracted, no cap |
| `PROFILE_KIND` | `"prof"` | In a variant: each target's day-type profile, known in advance |
| `DEFAULT_VARIANTS` | see below | Default value of `ForecastSettings.variants` |

`DEFAULT_VARIANTS`:

| Variant | Column kinds |
|---|---|
| `all` | `cal`, `*` |
| `all_profile` | `cal`, `*`, `prof` |
| `profile` | `cal`, `prof` |
| `univariate` | (none) |
| `calendar` | `cal` |
| `detector` | `cal`, `scene`, `count`, `rollup`, `area`, `cell`, `cellarea`, `rel`, `img`, `cat` |
| `full` | the `detector` list plus `emb`, `probe` |

`pipeline.yaml` does not set `variants`, so these defaults apply.

| Field | Type | Default | `pipeline.yaml` | Validation | Read by / meaning |
|---|---|---|---|---|---|
| `model` | `str` | `amazon/chronos-2` | `amazon/chronos-2` | | `build_forecaster(..., model=)` in CLI and scheduler |
| `quantiles` | `list[float]` | `[0.1, 0.5, 0.9]` | `[0.1, 0.5, 0.9]` | see `_quantiles` | `forecast/runner.py` (predict, stored with run), backtest plan |
| `horizon` | `int` | `16` | `16` | `1..1024` | Slots ahead (16 x 15 min = 4 h); runner default, backtest default |
| `min_context` | `int` | `96` | `96` | `>= 8` | Slots required before forecasting; runner raises `ForecastError` below it; backtest default |
| `max_context` | `int` | `8192` | `8192` | `>= 8` | Newest slots used; passed as `max_slots` by `sbi forecast`, the scheduler and the dashboard. Comment: Chronos-2 reads at most 8192 (about 85 days). |
| `max_targets` | `int` | `40` | `100` | `>= 1` | Cap on targets per space (`forecast/inputs.py::resolve_targets`) |
| `cross_learning` | `bool` | `False` | `false` | | Passed to `build_forecaster` |
| `variant` | `str` | `all` | `all_profile` | must be a key of `variants` | Default covariate set for live forecasts (`forecast/runner.py`) |
| `every_minutes` | `int` | `0` | `60` | `>= 0`; non-zero must be a multiple of 15 | Scheduled forecast interval used by `forecast/schedule.py::due`; `0` = never |
| `scheduled_model` | `str` | `chronos2` | `chronos2` | | Forecaster name the tick builds |
| `variants` | `dict[str, list[str]]` | `DEFAULT_VARIANTS` | not set | | Named covariate sets |

Validators:

- `_quantiles` (field): non-empty; every value strictly between 0 and 1; list equals `sorted(set(v))` (strictly increasing, no duplicates); every value a whole percent (`abs(q*100 - round(q*100)) <= 1e-9`) because they name the columns `q10`, `q25`, ...; must contain `0.5` (the point forecast).
- `_variant_known` (model, after): `every_minutes` non-zero and not divisible by 15 is rejected; `variant` must be in `variants`. The 15 is a literal in the code; it is not read from `table.step_minutes`.

#### 1.1.10 `CalendarSettings`

Docstring: known-in-advance day types (see `sbi/sitecal.py`). All fields are read by `sitecal.py::load_calendar`, called from `forecast/build.py`.

| Field | Type | Default | `pipeline.yaml` | Validation | Meaning |
|---|---|---|---|---|---|
| `country` | `str \| None` | `IN` | `IN` | | Public holidays via the optional `holidays` package |
| `subdivision` | `str \| None` | `None` | `null` | | State code for regional holidays |
| `site_file` | `str` | `config/site_calendar.yaml` | same | | The office's own holiday list and closures; resolved as `root / site_file`. `sitecal.py` merges it over the country days ("the site file wins"). |
| `working_days` | `list[int]` | `[0,1,2,3,4]` | `[0,1,2,3,4]` | `_days`: non-empty, each in 0..6, no duplicates | Monday = 0 |

`config/site_calendar.yaml` structure: one top-level key `days`, a list of `{date: YYYY-MM-DD, name: "..."}` entries. It currently lists India's fixed-date national holidays (Republic Day, Independence Day, Gandhi Jayanti) for 2026 and 2027. Its header comment says moving festivals are deliberately not guessed and should be added from the company's list.

#### 1.1.11 `SceneReaderSettings`

All fields are read by `perception/scene_lmstudio.py::LmStudioSceneReader`, except `base_url`, which `cli.py` also uses for curation.

| Field | Type | Default | `pipeline.yaml` | Validation | Meaning |
|---|---|---|---|---|---|
| `kind` | `Literal["lmstudio_openai"]` | `lmstudio_openai` | `lmstudio_openai` | | `build_scene_reader` |
| `base_url` | `str` | required | `http://10.20.0.1:1234` | | `.../v1/chat/completions` is called. Overridden by `SBI_LMSTUDIO_URL`. |
| `model` | `str` | required | `qwen/qwen3.5-9b` | | Sent as `model`. Overridden by `SBI_VLM_MODEL`. |
| `timeout_s` | `float` | `120` | `120` | `> 0` | HTTP timeout per attempt |
| `retries` | `int` | `3` | `3` | `>= 1` | Number of attempts; `VLMError` after the last |
| `max_tokens` | `int` | `1200` | `3000` | `>= 64` | Comment: v3 answers reach about 3k characters; 1200 truncated the JSON |
| `temperature` | `float` | `0.1` | `0.1` | `0..2` | Sent as `temperature` |
| `reasoning_effort` | `str \| None` | `"none"` | `none` | | Sent only when not `None`. YAML comment: required for Qwen3.5 on LM Studio, otherwise structured output comes back empty. YAML `none` is the string `"none"`, not null. |
| `image_width` | `int` | `1344` | `1344` | `>= 256` | Width the image is downscaled to before sending (`prepare_image`) |

#### 1.1.12 `CurationSettings`

Read by `curation/services.py` (`LmStudioEmbedder`, `LmStudioJudge`) and `cli.py` (`curate`). The curation services use `scene_reader.base_url` as their server.

| Field | Type | Default | `pipeline.yaml` | Validation | Meaning |
|---|---|---|---|---|---|
| `embed_model` | `str` | `text-embedding-nomic-embed-text-v1.5` | same | | Embedding model used to shortlist candidate pairs (`/v1/embeddings`) |
| `judge_model` | `str` | `qwen/qwen3.5-9b` | same | | Chat model asked "is every A a B?" (`/v1/chat/completions`) |
| `min_similarity` | `float` | `0.80` | `0.80` | `0 < x < 1` | Embedding cosine needed to consider a pair |
| `auto_approve` | `bool` | `True` | `true` | | Links are applied without review unless `--no-auto-approve` |
| `context` | `str` | "Object names produced by a vision model describing photos of an office cabin and a parking area. In an office, 'notebook' means a paper notebook." | not set | | Prefixed to the judge prompt |
| `timeout_s` | `float` | `60` | not set | `> 0` | HTTP timeout |
| `retries` | `int` | `3` | not set | `>= 1` | Attempts |

#### 1.1.13 `BaselineSettings`, `ProcessingSettings`

| Model | Field | Type | Default | `pipeline.yaml` | Validation | Read by |
|---|---|---|---|---|---|---|
| `BaselineSettings` | `ewm_alpha` | `float` | `0.3` | `0.3` | `0 < x <= 1` | `baseline.rebuild(alpha=)` from the tick and `sbi baseline`. Comment: weight of the newest sample. |
| `ProcessingSettings` | `backlog_limit` | `int` | `5` | `5` | `>= 0` | Tick: `pending_frame_ids(..., limit)`. Unprocessed frames retried per space per tick. `0` disables backlog retries (SQL `LIMIT 0`). |

#### 1.1.14 `TableSettings`

Read by `forecast/build.py::table_for` (table building), `forecast/inputs.py` and `cli.py` (`select_covariates`), `forecast/runner.py` and `forecast/schedule.py` (`step_minutes`), and `cli.py eval` (`mask_dup_iou`).

| Field | Type | Default | `pipeline.yaml` | Validation | Meaning (from comments) |
|---|---|---|---|---|---|
| `step_minutes` | `int` | `15` | `15` | `>= 1` | Table slot length |
| `min_coverage` | `float` | `0.5` | `0.5` | `0..1` | Share of slots with a value needed to be model-ready |
| `min_presence` | `float` | `0.05` | `0.05` | `0..1` | Share of valid slots where a count is > 0 |
| `min_valid_slots` | `int` | `8` | `8` | `>= 1` | |
| `min_present_slots` | `int` | `3` | `3` | `>= 1` | A count seen in fewer slots is a flicker |
| `exclude_labels` | `list[str]` | `[wall, floor, ceiling, pillar, roof]` | same | | Building parts; their counts are VLM noise |
| `max_covariates` | `int` | `24` | `24` | `>= 0` | Default number of past covariates given to a model |
| `always_include` | `list[str]` | `[]` | `[scene.people_vlm, scene.people_yolo, scene.vehicles_yolo, count.vlm.motorcycle, count.yolo.car]` | | Always selected as covariates |
| `min_autocorr` | `float` | `0.3` | `0.3` | `-1..1` | Smoothness filter for default inputs |
| `autocorr_min_slots` | `int` | `96` | `96` | `>= 3` | Smoothness filter applies only with about one day of slots |
| `embed_components` | `int` | `4` | `4` | `0..64` | Principal components of image vectors, as `emb.<model>.pcN` |
| `conf_levels` | `list[float]` | `[0.05, 0.10]` | `[0.05, 0.10]` | `_levels`: each in (0,1); distinct after rounding to whole percent; returned sorted | Extra detector thresholds counted as `count.yolo_cNN.*` |
| `mask_dup_iou` | `float` | `0.5` | `0.5` | `0 < x <= 1` | Same-type outlines overlapping this much are one object |
| `cell_area_min` | `float` | `0.02` | `0.02` | `0 <= x < 1` | Smaller grid-cell coverage is edge spill, recorded as 0 |

#### 1.1.15 `NvrCredentials`

| Field | Type | Default | Notes |
|---|---|---|---|
| `host` | `str` | required | From `SBI_NVR_HOST` |
| `port` | `int` | `554` | From `SBI_NVR_PORT`, default `"554"` when unset |
| `user` | `str` | required | From `SBI_NVR_USER` |
| `password` | `SecretStr` | required | From `SBI_NVR_PASS`. `repr`/`str` show `**********`; the plain value is only read via `get_secret_value()` in `rtsp_url`. |

It is also `_Strict` (frozen, no extra keys).

#### 1.1.16 Loading functions

`settings.py::load_dotenv(path)`
- Minimal `KEY=VALUE` loader. Returns silently when the file does not exist.
- Skips blank lines and lines starting with `#`.
- A non-blank, non-comment line without `=` raises `ConfigError("<path>:<line>: expected KEY=VALUE")`.
- Splits on the first `=`; strips whitespace from key and value, then strips any leading/trailing `'` and `"` characters from the value.
- Uses `os.environ.setdefault`: a variable already present in the real environment (even if empty) is not overwritten. Real environment variables win.
- It does not handle `export` prefixes or trailing inline comments; these would become part of the key or value.

`settings.py::load_settings(root=None, config_file="config/pipeline.yaml") -> Settings`
1. `root = Path(root or Path.cwd()).resolve()`. The CLI passes `--root` (default `"."`), so the root is the current directory unless given.
2. `load_dotenv(root / ".env")`.
3. Reads `root / config_file` with `yaml.safe_load`; an empty file becomes `{}`. `FileNotFoundError` becomes `ConfigError("settings file not found: ...")`; `yaml.YAMLError` becomes `ConfigError("invalid YAML in ...")`.
4. Copies the `scene_reader` mapping and applies environment overrides: if `SBI_LMSTUDIO_URL` is set and non-empty it replaces `base_url`; if `SBI_VLM_MODEL` is set and non-empty it replaces `model`.
5. Constructs `Settings(root=root, **raw)`. A pydantic `ValidationError` becomes `ConfigError("invalid settings in <path>:\n<details>")`. Only `ValidationError` is caught: a top-level YAML value that is not a mapping, or a `root:` key in the YAML, would raise an uncaught `AttributeError`/`TypeError` instead.
6. Calls `get_zone(settings.timezone)` to fail fast on a bad timezone name (raises `ConfigError`).

`settings.py::load_nvr_credentials() -> NvrCredentials`
- Checks `SBI_NVR_HOST`, `SBI_NVR_USER`, `SBI_NVR_PASS`. If any is unset or empty, raises `ConfigError("missing environment variables: <names> (see .env.example)")`. Only the names are reported, never values.
- Port: `int(os.environ.get("SBI_NVR_PORT", "554"))`. A non-integer port (`ValueError`) or a pydantic `ValidationError` becomes `ConfigError("invalid NVR credentials/port in environment: ...")`.
- Called only by `tick.py::run_tick` when no frame source is injected. `load_settings` does not read NVR credentials, so every non-tick command works without them.

#### 1.1.17 Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `SBI_LMSTUDIO_URL` | `load_settings` | Overrides `scene_reader.base_url` when non-empty |
| `SBI_VLM_MODEL` | `load_settings` | Overrides `scene_reader.model` when non-empty |
| `SBI_NVR_HOST` | `load_nvr_credentials` | NVR host (required for the tick) |
| `SBI_NVR_PORT` | `load_nvr_credentials` | RTSP port, default 554 |
| `SBI_NVR_USER` | `load_nvr_credentials` | NVR user (required for the tick) |
| `SBI_NVR_PASS` | `load_nvr_credentials` | NVR password (required for the tick), held as `SecretStr` |
| `OPENCV_FFMPEG_CAPTURE_OPTIONS` | written by `NvrRtspSource.grab` | Set to `rtsp_transport;<tcp|udp>` before each capture is opened |
| `TZ`, `PYTHONPATH` | set by the shell scripts | `TZ=Asia/Kolkata`, `PYTHONPATH=src` |
| `CUDA_VISIBLE_DEVICES` | set (empty) by the systemd unit | Hides GPUs from the dashboard |

`.env.example` structure: a header comment ("Copy to .env (gitignored) and fill in. Never commit real credentials."), then the keys `SBI_NVR_HOST`, `SBI_NVR_PORT` (example `554`), `SBI_NVR_USER`, `SBI_NVR_PASS` (the three NVR identity keys are empty in the example), a comment `# LM Studio server (OpenAI-compatible)`, then `SBI_LMSTUDIO_URL` and `SBI_VLM_MODEL` with the same values as `pipeline.yaml`. The real `.env` holds the NVR credentials and is listed in `.gitignore`.

`config/pipeline.yaml` repeats the policy in its header: secrets are never in the YAML; they come from `.env` or the environment.

### 1.2 Space configuration (`src/sbi/spaces.py`)

The module docstring: adding a new space means adding one YAML file under `config/spaces/`; nothing in the code is room-specific. Every model derives from a local `_Strict` with the same `extra="forbid", frozen=True` config as in `settings.py`.

#### 1.2.1 Models

`Grid`

| Field | Type | Default | Validation | Read by |
|---|---|---|---|---|
| `cols` | `int` | required | `0 < x <= 64` | `pipeline.py` (`bbox_to_cell`), `store/registry.py` (`spaces.grid_cols`), `forecast/build.py` |
| `rows` | `int` | required | `0 < x <= 64` | same |
| `anchor` | `Literal["bottom_center","center"]` | `bottom_center` | | `pipeline.py::bbox_to_cell`: which point of a box decides its cell |

`CameraRef`

| Field | Type | Default | Validation | Meaning |
|---|---|---|---|---|
| `channel` | `int` | required | `>= 1` | NVR channel in the RTSP URL |
| `subtype` | `int` | `0` | `>= 0` | `0` = main stream, `1` = sub stream |

`VocabEntry`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `category` | `str` | required | Stored as `object_registry.category` |
| `mobility` | `Literal["fixed","semi_static","dynamic","unknown"]` | `unknown` | Stored as `object_registry.mobility_prior` |

`SpaceConfig`

| Field | Type | Default | Validation | Read by / meaning |
|---|---|---|---|---|
| `space_id` | `str` | required | regex `^[a-z][a-z0-9_]*$` | Primary key everywhere; image subdirectory name |
| `name` | `str` | required | | Registry, VLM prompt, dashboard |
| `space_type` | `str` | required | | Registry, VLM prompt; the dashboard shows vehicles only when it equals `parking` |
| `camera` | `CameraRef` | required | | `rtsp_url` |
| `tz` | `str` | `Asia/Kolkata` | `get_zone` in `_cross_checks` | Local time for all derived time fields. The default is a literal; it does not come from `Settings.timezone`. |
| `cadence_minutes` | `int` | `15` | `>= 1` | Read only by `dashboard/data.py` (expected photos per 24 h = `24*60 // cadence_minutes`). The capture cadence itself is set by cron. |
| `label_types` | `list[str]` | `[]` | each must be in `vocabulary` | Labelling app: object types a human counts |
| `mask_regions` | `list[tuple[float,float,float,float]]` | `[]` | each `0 <= x1 < x2 <= 1` and `0 <= y1 < y2 <= 1` | `scene_lmstudio.py::prepare_image` blacks these out before the VLM sees the image |
| `grid` | `Grid` | required | | Geometric grid for detector boxes |
| `vocabulary` | `dict[str, VocabEntry]` | required | keys lowercase snake_case (`^[a-z][a-z0-9_]*$`) | Seeded into `object_registry` by `register_space` |
| `aliases` | `dict[str, str]` | `{}` | targets must be in `vocabulary` | Loaded into `object_alias` (alias lower-cased) |
| `detector_map` | `dict[str, str]` | `{}` | targets must be in `vocabulary` | `build_detections`: model class name to canonical label. Classes not in the map are dropped. |
| `scene_fields` | `dict[str, list[str]]` | `{}` | each list non-empty | Extra enum fields in the VLM JSON schema |
| `probes` | `dict[str, list[str]]` | `{}` | see checks below | Prompts scored by the scorer; the table averages each concept |
| `forecast_targets` | `list[str]` | `[]` | distinct, non-blank | Column names or shell-style patterns matched against model-ready columns when a forecast runs. Spaces with an empty list are skipped by scheduled forecasts. |

#### 1.2.2 `_cross_checks`

`SpaceConfig._cross_checks` is a model validator (after). It first calls `get_zone(self.tz)`, which raises `ConfigError` directly for an unknown zone. It then collects all problems into a list and raises one `ValueError` joining them with `"; "`:

| Check | Message |
|---|---|
| Every vocabulary key matches `^[a-z][a-z0-9_]*$` | `vocabulary label ... must be lowercase snake_case` |
| Every alias target is a vocabulary key | `alias ... -> ...: target not in vocabulary` |
| Every `detector_map` target is a vocabulary key | `detector_map ... -> ...: target not in vocabulary` |
| Every `label_types` entry is a vocabulary key | `label_types entry ... not in vocabulary` |
| Every mask region is normalised with `x1<x2`, `y1<y2` within `[0,1]` | `mask region ... must be normalised [x1,y1,x2,y2] ...` |
| `forecast_targets` entries non-blank and distinct | `forecast_targets must be distinct, non-empty column names or patterns` |
| Every probe concept matches the snake_case regex | `probe concept ... must be lowercase snake_case` |
| Every probe concept has at least one prompt | `probe concept ... has no prompts` |
| Every prompt is non-blank and at most 200 characters | `probe prompt ... must be 1-200 characters` |
| No prompt is shared by two concepts | `probe prompt ... is used by both ... and ...` |
| No concept repeats a prompt | `probe concept ... repeats a prompt` |
| Every scene field has at least one allowed value | `scene field ... has no allowed values` |

Alias keys and `detector_map` keys are not pattern-checked (they may contain spaces, e.g. `cell phone`).

#### 1.2.3 Property and functions

- `SpaceConfig.probe_prompts` (property): every distinct prompt across all concepts, in first-seen order (`dict.fromkeys`). Used by `pipeline.py` (scoring during processing) and `scoring.py` (finding frames missing scores).
- `spaces.py::load_space(path) -> SpaceConfig`: `yaml.safe_load` the file (empty becomes `{}`) and construct `SpaceConfig`. `FileNotFoundError` becomes `ConfigError("space file not found")`; `yaml.YAMLError` becomes `ConfigError("invalid YAML in ...")`; `ValidationError` or `TypeError` (for example a top-level list) becomes `ConfigError("invalid space config ...")`. A bad `tz` is different: `get_zone` raises `ConfigError` inside the validator, pydantic does not wrap it (it is not a `ValueError`), so it propagates unchanged as `ConfigError("unknown timezone ...")` without the file name (checked with the project interpreter).
- `spaces.py::discover_space_files(root) -> list[Path]`: `sorted((root / "config" / "spaces").glob("*.yaml"))`. Only `.yaml` files directly in that directory are found; `.yml` files and subdirectories are ignored.

#### 1.2.4 `config/spaces/admin_cabin.yaml`

| Key | Value |
|---|---|
| `space_id` / `name` / `space_type` | `admin_cabin` / `Admin cabin` / `room` |
| `camera` | channel `5`, subtype `0` (main stream). Comment: host, user and password come from the environment and must never be put here. |
| `tz` / `cadence_minutes` | `Asia/Kolkata` / `15` |
| `label_types` | `person`, `chair`, `monitor`, `laptop`, `bag` |
| `mask_regions` | one region `[0.64, 0.03, 0.97, 0.09]`: the burned-in camera timestamp |
| `grid` | 8 columns x 6 rows, anchor `bottom_center` (comment: floor objects use the box's bottom centre) |

Vocabulary (20 types):

| Type | Category | Mobility |
|---|---|---|
| person | person | dynamic |
| chair | furniture | semi_static |
| sofa | furniture | fixed |
| table | furniture | fixed |
| desk | furniture | fixed |
| laptop | device | semi_static |
| monitor | device | fixed |
| phone | device | dynamic |
| bag | item | dynamic |
| bottle | item | dynamic |
| cup | item | dynamic |
| book | item | semi_static |
| plant | furniture | fixed |
| door | fixture | fixed |
| whiteboard | fixture | fixed |
| screen | device | fixed |
| keyboard | device | semi_static |
| mouse | device | semi_static |
| helmet | item | dynamic |
| desktop_pc | device | fixed |

Aliases (26): `couch`, `settee` -> sofa; `tv`, `television` -> screen; `display`, `computer`, `computer monitor` -> monitor; `notebook` -> laptop; `cell phone`, `mobile`, `mobile phone` -> phone; `backpack`, `handbag` -> bag; `potted plant` -> plant; `mug` -> cup; `worker`, `man`, `woman`, `people` -> person; `water bottle` -> bottle; `computer tower`, `pc`, `cpu`, `tower` -> desktop_pc; `office chair` -> chair; `motorcycle helmet` -> helmet.

Detector map (16 COCO classes): `person` -> person, `chair` -> chair, `couch` -> sofa, `potted plant` -> plant, `dining table` -> table, `laptop` -> laptop, `tv` -> monitor, `keyboard` -> keyboard, `mouse` -> mouse, `cell phone` -> phone, `backpack` -> bag, `handbag` -> bag, `suitcase` -> bag (comment: the detector calls the large black bag on the desk a suitcase), `bottle` -> bottle, `cup` -> cup, `book` -> book. Note that the detector class `tv` maps to `monitor`, while the VLM alias `tv` maps to `screen`.

Scene fields: `lighting` [daylight, artificial, dim, dark]; `door_state` [open, closed, unknown]; `tidiness` [tidy, ok, messy, unknown].

Probes: 21 concepts, 3 prompts each, 63 prompts in total: `people`, `empty`, `lit`, `dark`, `messy`, `tidy`, `meeting`, `computer_work`, `sitting`, `standing`, `crowded`, `one_person`, `screens_on`, `screens_off`, `sunlight`, `door_open`, `eating`, `phone_call`, `bags_on_desk`, `cleaning`, `papers`. The file comment says to run `sbi score --space admin_cabin` after adding a concept or prompt, and to check `sbi stability` before trusting concepts that mostly track occupancy or brightness.

Forecast targets: `scene.people_yolo`, `scene.people_vlm`, `count.yolo.*`, `area.sam2.*`, `probe.*`. The comment notes that threshold variants (`count.yolo_c05.*`) match only by exact name.

#### 1.2.5 `config/spaces/parking.yaml`

Header comment: do not extract licence plates; counts and positions only.

| Key | Value |
|---|---|
| `space_id` / `name` / `space_type` | `parking` / `Parking area` / `parking` |
| `camera` | channel `3`, subtype `0` |
| `tz` / `cadence_minutes` | `Asia/Kolkata` / `15` |
| `label_types` | `person`, `car`, `motorcycle` |
| `mask_regions` | `[0.64, 0.03, 0.97, 0.09]` |
| `grid` | 12 columns x 8 rows, anchor `bottom_center` |

Vocabulary (13 types):

| Type | Category | Mobility | Comment in file |
|---|---|---|---|
| car | vehicle | semi_static | |
| bicycle | vehicle | semi_static | not `bike`: in Indian English "bike" usually means motorbike |
| motorcycle | vehicle | semi_static | |
| truck | vehicle | semi_static | |
| bus | vehicle | dynamic | |
| person | person | dynamic | |
| helmet | item | dynamic | |
| barrier | fixture | fixed | |
| pole | fixture | fixed | |
| chair | furniture | semi_static | the guard's desk: found in every labelled photo |
| bottle | item | dynamic | |
| bag | item | dynamic | |
| gate | fixture | fixed | |

Aliases (9): `cycle`, `pushbike` -> bicycle; `scooter`, `motorbike` -> motorcycle; `automobile`, `vehicle`, `suv` -> car; `van` -> truck; `pedestrian` -> person.

Detector map (11): `car`, `bicycle`, `motorcycle`, `truck`, `bus`, `person`, `chair`, `bottle` map to themselves; `backpack`, `handbag`, `suitcase` -> bag.

Scene fields: `lighting` [daylight, artificial, dim, dark]; `weather_visible` [clear, cloudy, rain, fog, unknown]; `occupancy_level` [empty, sparse, moderate, full].

Probes: 20 concepts, 3 prompts each, 60 prompts: `bikes`, `empty`, `car`, `night`, `day`, `person`, `full`, `half_full`, `arriving`, `parking_bike`, `helmets`, `bicycles`, `truck`, `wet`, `shadows`, `lights_on`, `group`, `cleaning`, `disorder`, `orderly`.

Forecast targets: `scene.vehicles_yolo`, `scene.people_yolo`, `count.yolo.*`, `count.vlm.motorcycle`, `area.sam2.*`, `probe.*`.

Both files load cleanly through `load_space` (checked with the project interpreter).

### 1.3 Error hierarchy (`src/sbi/errors.py`)

The module docstring states the boundary rule: library code raises these; only the CLI decides exit codes. Code catches the specific subclass it can handle; anything else is a bug and propagates to the top-level handler, which logs the traceback.

| Class | Parent | Meaning (docstring) | Raised by (examples) | Handled where |
|---|---|---|---|---|
| `SbiError` | `Exception` | Base for all expected, handled failures | `cli.py` raises it directly for usage errors (unknown space, no space files, conflicting flags) | `cli.main`: exit 2. `tick._run_space`: space fails, others continue. |
| `ConfigError` | `SbiError` | Missing or invalid configuration (settings, space file, environment) | `load_settings`, `load_dotenv`, `load_nvr_credentials`, `load_space`, `get_zone`, factory functions for unknown kinds, `sitecal.py` | Space file errors in the tick: that space is skipped. Missing NVR credentials in the tick: fatal, exit 2. |
| `CameraError` | `SbiError` | A frame could not be obtained | `NvrRtspSource.grab`, `FileSource.grab` | `capture_frame` catches it and records `capture_status='camera_down'` |
| `DetectorError` | `SbiError` | Detector failed to load or run | `YoloDetector` load and `detect` | Tick: at load, logged and the detector is `None` (frames queue); per frame, the backlog loop stops for that space. `reprocess`: load error is fatal (exit 2); per-frame error counted as failed. |
| `SegmenterError` | `SbiError` | Mask model could not load or run; frames still processed without masks | `Sam2Segmenter` | Tick: at load, logged and continued without outlines. `process_frame`: per frame, boxes stored without outlines. `reprocess`: load error is fatal (comment: do not silently wipe stored outlines). |
| `EmbedderError` | `SbiError` | Embedding model could not load or run | `DinoV2Embedder` | Tick: logged at load; per frame in `process_frame` |
| `ScorerError` | `SbiError` | Text-prompt scoring model could not load or run | `Siglip2Scorer` | Tick: logged at load; per frame in `process_frame` |
| `ForecastError` | `SbiError` | Forecaster could not load or run, or not enough data | `forecast/runner.py`, forecast models | Scheduled forecasts: logged per space. CLI: exit 2. |
| `VLMError` | `SbiError` | Scene reader failed after retries or answered unusably | `LmStudioSceneReader` | `process_frame` stores the frame without VLM fields |
| `StorageError` | `SbiError` | Database or filesystem failure | `store/db.py::connect`, `transaction`, `capture_frame` image write, `process_frame` for a missing frame | Inside `_run_space`: that space fails. From `connect`/`migrate` at tick start: exit 2. |
| `CurationError` | `SbiError` | Curator's embedding or judge model failed | `curation/services.py` | CLI `curate`: exit 2 |

Exceptions that are not `SbiError`:
- Inside `_run_space`: caught by `except Exception`, logged with `log.exception("%s: unexpected error (this is a bug)")`, and the space outcome becomes `ok=False, error="unexpected: <repr>"`. Other spaces continue.
- Inside scheduled forecasting: caught per space in `run_scheduled` and again around the whole call in `tick._forecasts`; they never change the tick result.
- Anywhere else (for example in `run_tick` setup outside the guarded blocks, or in a non-tick CLI command): not caught by `cli.main`. Python prints the traceback and the process exits with status 1.

### 1.4 Time handling (`src/sbi/timeutil.py`, `src/sbi/logging_setup.py`)

The module docstring: `ts_utc` is the canonical stored instant; every derived field (date, weekday, hour, slot) is computed in the space's local timezone (IST by default) so averages group by wall-clock time.

| Function / class | Purpose and behaviour |
|---|---|
| `timeutil.py::get_zone(tz_name)` | Returns `ZoneInfo(tz_name)`. `ZoneInfoNotFoundError`, `ValueError` or `OSError` become `ConfigError("unknown timezone ... (use an IANA name such as 'Asia/Kolkata')")`. |
| `timeutil.py::now_utc()` | `datetime.now(timezone.utc)`, timezone-aware. |
| `timeutil.py::TimeDims` | Frozen dataclass of derived time fields (below). `as_dict()` returns `dataclasses.asdict(self)`. |
| `timeutil.py::part_of_day(hour)` | `night` for hour < 6, `morning` < 12, `afternoon` < 18, else `evening`. |
| `timeutil.py::time_dims(ts, tz_name)` | Raises `ValueError` for a naive datetime. Converts to UTC, then to the zone, and fills `TimeDims`. |

`TimeDims` fields, with an example for `2026-11-13T03:41:43Z` in `Asia/Kolkata` (computed with the project interpreter):

| Field | Type | Rule | Example |
|---|---|---|---|
| `ts_utc` | `str` | UTC ISO, seconds | `2026-11-13T03:41:43+00:00` |
| `ts_local` | `str` | local ISO, seconds, with offset | `2026-11-13T09:11:43+05:30` |
| `tz` | `str` | `local.tzname()` or the zone name if empty | `IST` |
| `date` | `str` | local date ISO | `2026-11-13` |
| `dow` | `int` | `weekday()`, Monday = 0 | `4` |
| `dow_name` | `str` | `%a` | `Fri` |
| `hour`, `minute` | `int` | local | `9`, `11` |
| `slot_15` | `int` | `hour*4 + minute//15`, 0..95 | `36` |
| `week_of_year` | `int` | ISO week | `46` |
| `is_weekend` | `int` | `1` if `dow >= 5` | `0` |
| `part_of_day` | `str` | `part_of_day(hour)` | `morning` |

These fields are written into every `frames` row by `store/frames.py::insert_frame`.

Logging:

- `logging_setup.py::LocalTimeFormatter(fmt, tz_name)`: a `logging.Formatter` whose `formatTime` ignores `datefmt` and returns `datetime.fromtimestamp(record.created, zone).isoformat(timespec="seconds")`, e.g. `2026-11-18T09:15:03+05:30`. An unknown zone raises `ConfigError` at construction.
- `logging_setup.py::configure_logging(tz_name="Asia/Kolkata", level=logging.INFO, log_file=None)`: format `"%(asctime)s %(levelname)-7s %(name)s: %(message)s"`. Always adds a `StreamHandler(sys.stderr)`; adds a `FileHandler(log_file)` (creating its parent directory) when `log_file` is given. Clears all existing root handlers first, then attaches the handlers and sets the root level.
- `cli.main` calls `configure_logging(settings.timezone)` with no `log_file`. Persistent tick logs come from `scripts/tick.sh` redirecting stderr into `data/logs/tick.log`. The script's own lines use `date -Is` under `TZ=Asia/Kolkata`, which produces the same ISO-with-offset style.
- Logger names: modules use `logging.getLogger(__name__)` (e.g. `sbi.tick`, `sbi.ingest.capture`, `sbi.ingest.sources`); `cli.py` uses `logging.getLogger("sbi")`.

### 1.5 Capture (`src/sbi/ingest/`)

`ingest/__init__.py` holds only the docstring "Frame acquisition: camera sources and capture (store image + frame row)."

#### 1.5.1 Frame sources (`ingest/sources.py`)

The module docstring: credentials never appear in logs or exception messages. Any source satisfies the `perception/types.py::FrameSource` protocol: `grab(space) -> np.ndarray` returning a BGR image and raising `CameraError` on failure.

`sources.py::rtsp_url(creds, camera, *, redact=False) -> str`
- Builds a Dahua-style NVR URL: `rtsp://<user>:<password>@<host>:<port>/cam/realmonitor?channel=<channel>&subtype=<subtype>`.
- User and password are URL-encoded with `urllib.parse.quote(..., safe='')`; the password is read with `SecretStr.get_secret_value()`.
- With `redact=True` the auth part is the literal `***:***`. The redacted form is the only one used in messages.

`sources.py::NvrRtspSource(settings: FrameSourceSettings, creds: NvrCredentials)`
- Built by `perception/factory.py::build_frame_source` when `frame_source.kind == "nvr_rtsp"`.
- `grab(space)`:
  1. Sets the process environment variable `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;<tcp|udp>`. The comment says it must be set before the capture is opened because OpenCV's FFmpeg backend reads it.
  2. Computes the redacted URL (`where`) for messages.
  3. Opens `cv2.VideoCapture(<real URL>, cv2.CAP_FFMPEG)`.
  4. Calls `cap.set(CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)` and `cap.set(CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)` with `timeout_ms = int(timeout_s * 1000)` (10 000 ms by default). These calls happen after the constructor has already tried to open the stream. OpenCV documents both properties as open-time parameters, so they are unlikely to bound this open or the reads. The overall bound is the `timeout 600` in `scripts/tick.sh`.
  5. If `not cap.isOpened()`: raises `CameraError("cannot open stream <redacted URL>")`.
  6. Reads `warmup_frames` frames (5 by default), keeping the last one that decoded (`ok` true). The comment: discard buffered frames, keep the freshest.
  7. If no frame decoded: raises `CameraError("stream opened but no frame decoded: <redacted URL>")`.
  8. `cap.release()` always runs (in `finally`).

`sources.py::FileSource(path)`
- Docstring: reads a still from disk, for tests and offline re-runs.
- `grab(space)` ignores `space`, returns `cv2.imread(path)`, and raises `CameraError("cannot read image file ...")` when `imread` returns `None`.
- It cannot be selected from `pipeline.yaml` (`frame_source.kind` only allows `nvr_rtsp`); it is passed in through `run_tick(source=...)`.

#### 1.5.2 Image metrics and the bad-frame rule (`ingest/capture.py`)

`capture.py::compute_metrics(img) -> ImageMetrics` converts BGR to grayscale and returns:

| Metric | Formula |
|---|---|
| `width` | `img.shape[1]` |
| `height` | `img.shape[0]` |
| `brightness` | mean of the grayscale pixel values (0..255) |
| `contrast` | standard deviation of the grayscale pixel values |
| `sharpness` | variance of `cv2.Laplacian(gray, cv2.CV_64F)` |

`ImageMetrics` is the frozen dataclass in `store/frames.py` with those five fields.

`capture.py::is_bad_frame(m) -> bool`: `contrast < 1.0 and (brightness < 5 or brightness > 250)`. The docstring: flat near-black or near-white frames usually mean a bad decode, not an empty room.

#### 1.5.3 `CaptureResult` and file naming

`capture.py::CaptureResult` (frozen dataclass):

| Field | Type | Meaning |
|---|---|---|
| `frame_id` | `int` | Row id in `frames` |
| `status` | `str` | `ok`, `camera_down` or `bad_frame` |
| `created` | `bool` | `False` if a row for this (space, timestamp) already existed |
| `error` | `str \| None` | Camera error text or the bad-frame reason |

`capture.py::_image_name(ts_local)`: removes `-` and `:` from the local ISO timestamp and appends `.jpg`, e.g. `2026-11-13T09:11:43+05:30` becomes `20261113T091143+0530.jpg`.

Storage path: `<image_dir>/<space_id>/<local date>/<_image_name>`, e.g. `data/images/admin_cabin/2026-11-13/20261113T091143+0530.jpg`. The path stored in the database is relative to the project root when the file is under it, else absolute.

#### 1.5.4 `capture.py::capture_frame`

Signature: `capture_frame(conn, space, source, *, root, image_dir, now=None) -> CaptureResult`. The module docstring: a camera failure is data, not an exception; it is recorded as `capture_status='camera_down'` with NULL metrics, so averages exclude it and nothing is read as an empty room. A disk failure is an error (`StorageError`).

Steps:
1. `dims = time_dims(now or now_utc(), space.tz)`. The tick passes its own `now`, so the frame's timestamp is the tick's start time, not the moment the image was grabbed.
2. `source.grab(space)`. A `CameraError` is caught; its text is kept and logged at WARNING: `"<space>: camera unavailable: <error>"`. Other exceptions propagate.
3. Default outcome: `status="camera_down"`, no path, no metrics.
4. If an image was grabbed: compute metrics.
   - Bad frame: `status="bad_frame"`, `error="flat near-black/near-white frame"`, WARNING `"<space>: bad frame (brightness %.1f, contrast %.2f)"`. The image is not written; the metrics are kept.
   - Otherwise `status="ok"`: create the directory, write the JPEG with `cv2.imwrite(..., [IMWRITE_JPEG_QUALITY, 92])`. `imwrite` returning `False` or any `OSError` becomes `StorageError("cannot write image ...")`. No frame row is written in that case.
5. Inside `store/db.py::transaction` (`BEGIN IMMEDIATE ... COMMIT`), call `store/frames.py::insert_frame`.
6. INFO log `"<space>: <status> frame_id=<id>"`, with `" (already existed)"` appended when the row was not new.

`insert_frame` writes one row with `INSERT OR IGNORE`:

| Column | Value |
|---|---|
| `space_id` | `space.space_id` |
| `ts_utc`, `ts_local`, `tz`, `date`, `dow`, `dow_name`, `hour`, `minute`, `slot_15`, `week_of_year`, `is_weekend`, `part_of_day` | from `TimeDims` |
| `image_path` | relative path for `ok`, `NULL` for `camera_down` and `bad_frame` |
| `capture_status` | `ok` / `camera_down` / `bad_frame` |
| `width`, `height`, `brightness`, `contrast`, `sharpness` | metrics for `ok` and `bad_frame`; `NULL` for `camera_down` |

If the insert is ignored (same `space_id` and `ts_utc` already present), the existing `frame_id` is looked up and returned with `created=False`. The image file written in step 4 overwrites the file at the same path in that case.

#### 1.5.5 Reading a stored image (`src/sbi/imageio.py`)

`imageio.py::read_frame_image(frame, root) -> np.ndarray | None`: returns `None` when `frame["image_path"]` is empty; otherwise `cv2.imread(str(root / image_path))`, which returns `None` for a missing or unreadable file. An absolute stored path wins over the root because of `pathlib` join rules. `pipeline.py::process_frame` treats `None` as "image unreadable" and skips the frame.

### 1.6 The tick (`src/sbi/tick.py`)

The module docstring: for every space, capture a frame, run Engine 1, refresh that hour's baseline, and every `forecast.every_minutes` store a forecast per space. A problem in one space never stops the others, and a perception outage never stops capture.

#### 1.6.1 `SpaceOutcome`

Frozen dataclass returned per space: `space_id: str`, `ok: bool`, `frame_id: int | None = None`, `capture_status: str | None = None`, `processed: int = 0`, `error: str | None = None`. A camera outage produces `ok=True` with `capture_status="camera_down"`, because the outage is recorded as data.

#### 1.6.2 `tick.py::run_tick`

Signature: `run_tick(settings, space_files, *, source=None, detector=None, reader=None, use_vlm=True, now=None, segmenter=None, embedder=None, scorer=None, forecaster=None) -> list[SpaceOutcome]`. Every component can be injected (tests, alternative implementations); otherwise it is built from settings.

Order of operations:

1. `now = now or now_utc()`. One timestamp is used for every space in this tick.
2. `conn = connect(settings.db_path)` (WAL mode, `busy_timeout` 30 s, foreign keys on). A `StorageError` here propagates to the CLI.
3. `migrate(conn)`: applies pending schema migrations.
4. For each space file: `load_space`, then `store/registry.py::register_space` (upserts the `spaces` row with the config path relative to the root when possible; seeds `object_registry` with `INSERT OR IGNORE` so reviewed rows are never overwritten; loads aliases with `INSERT OR REPLACE`). An `SbiError` logs `"skipping space file <name>: <error>"` and appends `SpaceOutcome(<file stem>, False, error=...)`.
5. Frame source: if not injected, `build_frame_source(settings.frame_source, load_nvr_credentials())`. A `ConfigError` (missing credentials) is not caught: the tick stops before any capture and no frame rows are written. The comment marks this as fatal.
6. Detector: if not injected, `build_detector`. A `DetectorError` logs `"detector unavailable, frames will be queued unprocessed: ..."` and leaves `detector=None`.
7. Scene reader: if not injected and `use_vlm`, `build_scene_reader(settings.scene_reader)`. This only creates an HTTP session; it does not contact the server.
8. Segmenter: only if not injected and a detector exists. `SegmenterError` logs `"segmenter unavailable, frames will be stored without outlines: ..."`.
9. Embedder: same rule. `EmbedderError` logs `"embedder unavailable, frames will be stored without image vectors: ..."`.
10. Scorer: same rule. `ScorerError` logs `"scorer unavailable, frames will be stored without probe scores: ..."`.
11. For each loaded space, in file order: `_run_space(...)`.
12. `_forecasts(conn, settings, spaces, now, forecaster)`.
13. Return the outcomes. The connection is closed in `finally`.

`build_segmenter`, `build_embedder` and `build_scorer` return `None` when their `kind` is `none`. All models are loaded once per tick, before the first capture; each tick is a new process, so models are loaded on every tick.

Models built per stage, with the current `pipeline.yaml`:

| Stage | Builder | Model | Loaded when | Failure at load |
|---|---|---|---|---|
| Frame source | `build_frame_source` | NVR RTSP | always | fatal (`ConfigError`) |
| Detector | `build_detector` | YOLO `models/yolo26x.pt` at 1280 | always | logged; capture continues, frames queue |
| Scene reader | `build_scene_reader` | LM Studio `qwen/qwen3.5-9b` | unless `--no-vlm` | not caught (construction does no I/O) |
| Segmenter | `build_segmenter` | SAM 2 `models/sam2.1_l.pt` | only with a detector | logged; no outlines |
| Embedder | `build_embedder` | DINOv2 `facebook/dinov2-base` @448 | only with a detector | logged; no vectors |
| Scorer | `build_scorer` | SigLIP 2 `google/siglip2-base-patch16-224` | only with a detector | logged; no probe scores |
| Forecaster | `build_forecaster` (in `forecast/schedule.py`) | `chronos2` with `amazon/chronos-2` | only when a forecast is due | logged; no forecasts this tick |

#### 1.6.3 `tick.py::_run_space`

1. `capture_frame(conn, space, source, root=settings.root, image_dir=settings.image_dir, now=now)`.
2. If a detector exists:
   - `todo` starts with the new frame id if `cap.status == "ok"`.
   - Appends `pending_frame_ids(conn, space_id, settings.processing.backlog_limit)`, excluding the new frame id. That query selects frames with `capture_status='ok'` and `processed_utc IS NULL`, oldest first, `LIMIT backlog_limit`.
   - For each id, `pipeline.py::process_frame(conn, space, fid, detector, reader, root=..., segmenter=..., embedder=..., scorer=..., dedupe_outline_iou=settings.detector.dedupe_outline_iou)`.
     - `DetectorError`: ERROR `"<space> frame <id>: detector failed, leaving it queued: ..."` and `break` (comment: do not hammer a broken detector with the rest of the backlog).
     - Other `SbiError`: ERROR `"<space> frame <id>: processing failed, leaving it queued: ..."` and `continue`.
     - If the result has `processed=True`, the counter increases and an INFO line is logged:
       `"<space> frame <id>: people yolo=<n> vlm=<n> boxes=<counted> stored=<all>"` followed by optional suffixes: `" (no VLM)"` when the VLM answer was not usable; `" (no outlines)"` when a segmenter exists but produced no masks; `" (no image vector)"` when an embedder exists but no vector was stored; `" (no probe scores)"` when a scorer exists, the space has probes, and no scores were stored.
3. If at least one frame was processed: `baseline.rebuild(conn, space_id, alpha=settings.baseline.ewm_alpha, dow=d.dow, hour=d.hour)` with `d = time_dims(now, space.tz)`. Only the baseline bucket for the tick's local weekday and hour is recomputed.
4. Returns `SpaceOutcome(space_id, True, cap.frame_id, cap.status, processed)`. Processing errors that were logged and skipped do not make the outcome fail.
5. Any `SbiError` escaping steps 1-3 (for example `StorageError` from the image write or a transaction): ERROR `"<space>: tick failed: ..."`, outcome `ok=False`.
6. Any other exception: `log.exception("<space>: unexpected error (this is a bug)")`, outcome `ok=False, error="unexpected: <repr>"`.

When the detector is missing, captured `ok` frames keep `processed_utc` NULL. Later ticks process the new frame plus up to `backlog_limit` older ones per space.

#### 1.6.4 Scheduled forecasts

`tick.py::_forecasts` lazily imports `forecast/schedule.py::run_scheduled` (comment: most ticks are not forecast ticks), logs `"<space>: forecast run <id> stored"` for each stored run, and wraps the whole call in `except Exception` with `log.exception("scheduled forecasting crashed (this is a bug); capture is unaffected")`. It never changes the tick's outcomes.

`forecast/schedule.py::due(now_utc, tz, every_minutes, step_minutes=15)`: `False` when `every_minutes <= 0`; otherwise converts `now` to the space's local time, takes minutes since midnight, floors to `step_minutes`, and returns whether the result is divisible by `every_minutes`. With `every_minutes: 60`, forecasts are made on the tick whose slot starts on the hour.

`run_scheduled` selects spaces that have `forecast_targets` and are due. If none, it returns immediately without loading a model. Otherwise it builds the forecaster once (`fs.scheduled_model`, `fs.model`, `fs.cross_learning`); an `SbiError` there is logged as `"forecaster unavailable, no forecasts this time: ..."`. Per space it builds the table with `max_slots=forecast.max_context` and calls `forecast_space`; `SbiError` and other exceptions are logged and isolated per space.

#### 1.6.5 `scripts/tick.sh`

Header comments: one scheduled tick for all spaces; overlapping runs are skipped (flock); a hung run is killed after 10 minutes; all times are IST; output is appended to `data/logs/tick.log`, rotated at 10 MB.

Line by line:

| Line | Effect |
|---|---|
| `set -uo pipefail` | Unset variables are errors; pipelines fail on any failing stage. `-e` is not set, so the script continues after a failed command and can inspect `rc`. |
| `cd "$(dirname "$0")/.."` | Runs from the project root wherever cron starts it. |
| `export TZ=Asia/Kolkata PYTHONPATH=src` | IST for `date` and the process; makes `sbi` importable without installation. |
| `LOG=data/logs/tick.log`; `mkdir -p data/logs` | Log path (hard-coded, not read from `paths.logs`). |
| `[ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 10485760 ] && mv "$LOG" "$LOG.1"` | If the log exceeds 10 485 760 bytes (10 MiB), it is renamed to `tick.log.1`, replacing any earlier `.1`. One old generation is kept. This runs before the lock is taken. |
| `exec 9>data/.tick.lock` | Opens file descriptor 9 on the lock file for the life of the script. |
| `if ! flock -n 9; then ... exit 0; fi` | Non-blocking exclusive lock. If another tick holds it, appends `"<date -Is> WARNING tick skipped: previous tick still running"` and exits 0. |
| `PY=.venv/bin/python`; `[ -x "$PY" ] \|\| PY=python3` | Uses the project venv (comment: system packages plus Chronos-2 for scheduled forecasts); falls back to the system `python3` when the venv is missing. |
| `timeout 600 "$PY" -m sbi tick >> "$LOG" 2>&1` | Runs the tick with a 600 s limit; stdout and stderr (all Python logging) are appended to the log. |
| `rc=$?` and `case $rc` | `0`: nothing extra. `1`: appends `WARNING tick finished with failed space(s), rc=1`. `124` (the `timeout` exit status): appends `ERROR tick killed after 600s timeout`. Anything else: appends `ERROR tick failed, rc=<rc>`. |
| `exit $rc` | Returns the tick's status to cron. |

Cron entry (from `README.md` and the installed crontab):

```
*/15 * * * * <project root>/scripts/tick.sh
```

It runs at minutes 0, 15, 30 and 45 of every hour. The README installs it with `(crontab -l; echo "*/15 * * * * $PWD/scripts/tick.sh") | crontab -`. `scripts/curate.sh` notes that cron runs in UTC on this machine; `*/15` gives the same minutes in both zones because IST is UTC+05:30.

### 1.7 Command line (`src/sbi/cli.py`, `__main__.py`, `__init__.py`)

- `src/sbi/__init__.py`: docstring "Space Behaviour Intelligence: camera stills -> structured scene data -> baselines." and `__version__ = "0.1.0"`.
- `src/sbi/__main__.py`: `sys.exit(main())`, so `python -m sbi <command>` runs `cli.main`.
- `pyproject.toml` declares no console-script entry point; the CLI is run as `python -m sbi` with `PYTHONPATH=src` from the project root (the scripts do this).

#### 1.7.1 `cli.py::main(argv=None) -> int`

1. Parses arguments with `_parser()`. argparse itself exits with status 2 on invalid arguments.
2. `load_settings(args.root)`, then `configure_logging(settings.timezone)`.
3. `tick` is handled first and does not open a separate connection (see 1.7.3).
4. Every other command opens `connect(settings.db_path)`, runs `migrate(conn)`, executes the command, and closes the connection in `finally`. Every command therefore applies pending migrations.
5. Returns `0` unless the command returned `1` earlier.
6. `except SbiError as e`: logs `log.error("%s", e)` if the root logger has handlers, otherwise prints `error: <e>` to stderr (this happens when `load_settings` fails before logging is configured). Returns `2`.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | `tick`: at least one space outcome has `ok=False`. `reprocess`, `embed`, `score`: at least one frame failed. Also Python's status for an uncaught non-`SbiError` exception. |
| `2` | Any `SbiError` reaching `main` (configuration, missing credentials, storage at startup, unknown space, model load failure in a non-tick command, etc.). Also argparse usage errors. |

The module docstring lists the codes as "0 ok, 1 some spaces failed, 2 fatal setup/config error".

Helpers:
- `cli.py::_parser()`: builds the parser; `prog="sbi"`; the module docstring is the description; subcommands are required.
- `cli.py::_spaces_by_id(root)`: loads every discovered space file and returns `{space_id: SpaceConfig}`. Any invalid space file raises `ConfigError` and stops the command, even when another space was requested.
- `cli.py::_build_table(conn, settings, space, start=None, end=None, fit_end=None, max_slots=None)`: lazy import and call of `forecast/build.py::table_for`.

#### 1.7.2 Global option

| Option | Default | Effect |
|---|---|---|
| `--root` | `.` | Project root passed to `load_settings`. Must come before the subcommand. |

#### 1.7.3 Subcommands

| Command | Arguments (default) | What it does |
|---|---|---|
| `tick` | `spaces` (zero or more YAML paths; default `config/spaces/*.yaml`); `--no-vlm` (flag) | Paths given are resolved against the current directory, not `--root`. With none, `discover_space_files(settings.root)`. No files: `SbiError("no space files found under config/spaces/")`, exit 2. Calls `run_tick(settings, files, use_vlm=not no_vlm)`, logs `"tick done: <n> ok, <m> failed"`, returns 1 if any outcome failed. |
| `migrate` | none | Applies migrations (as every command does) and prints `schema version <PRAGMA user_version>`. |
| `baseline` | `--space` (optional) | Full `baseline.rebuild` (all weekday/hour buckets) with `ewm_alpha`, for the given space or every `space_id` in the `spaces` table. Logs `"baseline rebuilt for <id>"`. |
| `reprocess` | `--space` (required); `--frame-id` (int, repeatable; default all frames); `--no-vlm`; `--detector-only` | Unknown space: exit 2. Builds the detector and segmenter; their load errors are fatal (the segmenter comment: stop, do not silently wipe stored outlines). Then rejects `--detector-only` with `--no-vlm` (`SbiError`). The reader is built unless `--no-vlm`. Default ids: every frame of the space with `capture_status='ok'`, ordered by `ts_utc` (the help text says "all processed frames"; the query does not filter on processing). Calls `process_frame(..., replay_vlm=detector_only, segmenter=..., dedupe_outline_iou=...)`; no embedder or scorer is passed. A per-frame `SbiError` (including `DetectorError`) is logged as `"frame <id> not reprocessed: ..."` and counted. Logs `"reprocessed <done>/<n> frames[, <failed> failed]"`; returns 1 if any failed. |
| `eval` | none | Prints `eval/vlm_eval.py::report(conn, mask_dup_iou=table.mask_dup_iou)`. |
| `curate` | `--no-auto-approve` (flag) | Runs `curation/curator.py::curate` with `LmStudioEmbedder` and `LmStudioJudge` on `scene_reader.base_url`, `min_similarity`, and `auto_approve = curation.auto_approve and not --no-auto-approve`. Prints counts (names, candidate pairs, questions asked/cached, new links, re-mapped rows) and one line per new link. |
| `links` | `--status` (`proposed`, `approved`, `rejected`; optional) | Lists rows of `object_links` (optionally filtered by status) ordered by `link_id`, then the count. |
| `link-set` | `link_id` (int); `status` (`approved` or `rejected`) | Calls `set_link_status` and prints its result. A `ValueError` becomes `SbiError`, exit 2. |
| `table` | `--space` (required); `--start`; `--end`; `--out` (`data/exports`); `--fit-end` | Builds the table for the window. Writes `<space>_table.csv` (index label `timestamp`) and `<space>_catalog.csv` under `resolve(--out)`. Prints slot count, range and `slot_status` counts, then dimensions by kind. Runs `select_covariates` (no targets, table settings) and rewrites the catalog with a `default_input` column. `--fit-end`: learn the image-vector components only from slots before this local time. |
| `forecast` | `--space` (required); `--variant` (default `forecast.variant`); `--horizon` (int, default `forecast.horizon`); `--model` (`chronos2`; help lists `chronos2 \| seasonal_mean \| seasonal_naive \| persistence`) | Builds the forecaster and a table limited to `max_context` slots, calls `forecast_space`, which stores the run. Prints the run id, model, origin (IST), counts, the first 4 steps for up to 6 targets with all quantile columns, and where it was stored. |
| `backtest` | `--space` (required); `--horizon` (int); `--min-context` (int); `--step` (int, `8` = 2 hours); `--variants` (comma list; default all of `forecast.variants`); `--models` (`chronos2,seasonal_mean,seasonal_naive,persistence`); `--out` (`data/exports`) | Validates variant names (`SbiError` if unknown). Builds the whole table, computes origins; none: `SbiError` naming the slots needed (`min_context + horizon`). Rebuilds the table with `fit_end` at the first origin (comment: image-vector components learned from history before the first origin only). Runs every model, writes `<space>_backtest_raw.csv` and `<space>_backtest.csv`, prints the summary and skipped origins. |
| `accuracy` | `--space` (required); `--since` (local time); `--out` (`data/exports`) | Scores stored forecasts whose time has passed against the table. No points: prints a message, exit 0. Otherwise writes `<space>_accuracy.csv` (per target) and prints the per-lead table. |
| `score` | `--space` (required); `--limit` (int) | Requires the space to have `probes` and `scorer.kind` not `none` (each else `SbiError`). Runs `scoring.py::score_missing`; logs done/failed; returns 1 if any failed. |
| `embed` | `--space` (required); `--limit` (int) | Requires a known space and `embedder.kind` not `none`. Runs `embedding.py::embed_missing`; returns 1 if any failed. |
| `stability` | `--space` (required); `--start`, `--end` (local time); `--min-mean` (float, `0.0`); `--out` (`data/exports`) | Runs `eval/stability.py::stability` on the whole table, writes `<space>_stability.csv`, prints rows whose absolute window mean is at least `--min-mean`. |

### 1.8 Operations

#### 1.8.1 `scripts/dashboard.sh`

- `set -euo pipefail`; `cd` to the project root; `export TZ=Asia/Kolkata PYTHONPATH=src`.
- `exec .venv/bin/streamlit run src/sbi/dashboard/app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false "$@"`.
- The dashboard is reachable only from this machine (bound to 127.0.0.1). Extra arguments pass through. There is no fallback interpreter: the venv must exist.

#### 1.8.2 `scripts/label.sh`

Same structure as `dashboard.sh`, running `src/sbi/labeling/app.py` on port 8501 (the blind labelling UI). The header notes it needs the `.venv` with Streamlit.

#### 1.8.3 `scripts/curate.sh`

- Header: links synonyms (`same`) and hierarchy (`is_a`) among VLM object names, applies them and re-maps history. It uses a lock separate from `tick.sh` so it never delays a capture. Review with `python3 -m sbi links`; undo with `python3 -m sbi link-set ID rejected`. Suggested crontab for 03:00 IST (cron runs in UTC on this machine): `30 21 * * * <root>/scripts/curate.sh`.
- `set -uo pipefail`; `cd` to the root; `TZ=Asia/Kolkata`, `PYTHONPATH=src`.
- Log `data/logs/curate.log` (no rotation). Lock `data/.curate.lock` via `exec 9>` and `flock -n 9`; if held, logs `WARNING curation skipped: previous run still active` and exits 0.
- Logs `curation start`, runs `timeout 3600 python3 -m sbi curate >> "$LOG" 2>&1`, logs `curation end rc=<rc>`, exits with that code.
- It runs the system `python3`, not `.venv/bin/python` as `tick.sh` does.

#### 1.8.4 systemd user service (`scripts/systemd/sbi-dashboard.service`)

Header: the read-only dashboard as a user service; starts with the machine and restarts if it stops. `@ROOT@` is filled in by the installer.

| Directive | Value | Effect |
|---|---|---|
| `Description` | `SBI dashboard (http://localhost:8502)` | |
| `After` | `network.target` | Starts after networking |
| `Type` | `simple` | |
| `WorkingDirectory` | `@ROOT@` | Project root |
| `ExecStart` | `@ROOT@/scripts/dashboard.sh` | |
| `Restart` | `always` | Restarted whenever it stops |
| `RestartSec` | `5` | 5 s between restarts |
| `Environment` | `CUDA_VISIBLE_DEVICES=` | Empty value hides all GPUs. Comment: stay small next to the 15-minute capture run, no GPU use, modest memory. |
| `MemoryMax` | `4G` | Memory cap for the service |
| `WantedBy` | `default.target` | Started with the user's default target |

#### 1.8.5 `scripts/install-dashboard-service.sh`

- `set -euo pipefail`. Header lists removal (`systemctl --user disable --now sbi-dashboard` and deleting the unit file), status and journal commands. Re-running replaces the service.
- Computes `ROOT` as the absolute project root, creates `~/.config/systemd/user`, writes the unit with `sed "s|@ROOT@|$ROOT|g"`.
- `systemctl --user daemon-reload`, then `systemctl --user enable --now sbi-dashboard.service`.
- `loginctl enable-linger "$USER"` so the user manager starts at boot without a Linux login (comment: WSL is kept running by the Windows task). On failure it prints that the dashboard will start at the first login instead and continues.
- Prints the first 5 lines of `systemctl --user --no-pager status`.

#### 1.8.6 Windows scheduled task (`scripts/windows/install-autostart.ps1`)

Purpose (header): keep the WSL distro, and so cron and the 15-minute tick, running after a Windows restart. A hidden `sleep infinity` inside the distro holds WSL open, so systemd keeps cron running.

- Parameters: `-Distro` (mandatory), `-TaskName` (default `SBI-WSL-Keepalive`). `$ErrorActionPreference = "Stop"`.
- Validates the distro against `wsl.exe -l -q` (NUL characters removed, lines trimmed); throws if not found, listing the known distros.
- Action: `%WINDIR%\System32\conhost.exe --headless %WINDIR%\System32\wsl.exe -d <Distro> --exec /bin/sleep infinity`. Comment: `conhost --headless` runs `wsl.exe` without a console window (Windows 11).
- Trigger: at logon of the current user (`$env:USERDOMAIN\$env:USERNAME`). The task starts at Windows logon, not at boot.
- Settings: no execution time limit; restart up to 999 times at 1-minute intervals; allowed on and not stopped on batteries; `StartWhenAvailable`; `MultipleInstances IgnoreNew`.
- Principal: current user, `Interactive` logon, `Limited` run level.
- `Register-ScheduledTask ... -Force` (replaces an existing task), then `Start-ScheduledTask`, waits 3 s and prints the task state.
- Usage from WSL: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w scripts/windows/install-autostart.ps1)" -Distro "$WSL_DISTRO_NAME"`. Removal: `Unregister-ScheduledTask -TaskName SBI-WSL-Keepalive`.

#### 1.8.7 `.streamlit/config.toml`

Theme for the local Streamlit apps (the comment says nothing here affects the data): `base = "light"`, `primaryColor = "#2a78d6"`, `backgroundColor = "#fcfcfb"`, `secondaryBackgroundColor = "#f1f0ec"`, `textColor = "#0b0b0b"`, `font = "sans serif"`; `[client] toolbarMode = "minimal"`. Streamlit reads it from the working directory; both launch scripts `cd` to the project root first.

#### 1.8.8 `.gitignore`

| Entry | What it excludes |
|---|---|
| `data/` | Runtime data: database, images, logs, cache, exports, lock files |
| `__pycache__/`, `*.pyc` | Python bytecode |
| `.env` | Credentials. `.env.example` says: copy to `.env` (gitignored), never commit real credentials. |
| `models/` | Model weights (`models/yolo26x.pt`, `models/sam2.1_l.pt`) |
| `.venv/` | The project virtual environment |
| `graphify-out/` | Output of the graphify tool |

#### 1.8.9 `pyproject.toml`

| Section | Content |
|---|---|
| `[project]` | `name = "sbi"`, `version = "0.1.0"`, description "Space Behaviour Intelligence: camera stills -> structured scene data -> baselines", `requires-python = ">=3.10"` |
| `dependencies` | `opencv-python`, `numpy`, `pandas`, `pydantic>=2`, `pyyaml`, `requests` |
| `[project.optional-dependencies]` | see below |
| `[tool.pytest.ini_options]` | `pythonpath = ["src"]`, `testpaths = ["tests"]`, `addopts = "-q"` |
| `[tool.setuptools.packages.find]` | `where = ["src"]` |
| `[tool.setuptools.package-data]` | `sbi = ["store/migrations/*.sql"]` (migration SQL ships with the package) |

Optional dependency groups:

| Group | Packages | Used for (from comments) |
|---|---|---|
| `detector` | `ultralytics>=8.4` | YOLO detector and SAM 2 outlines. YOLO26 weights need ultralytics 8.4 or newer. |
| `vision` | `torch`, `transformers` | DINOv2 image vectors and SigLIP 2 scores |
| `ui` | `streamlit>=1.50`, `altair` | Dashboard and labelling tool |
| `forecast` | `chronos-forecasting>=2.0` | Chronos-2 |
| `holidays` | `holidays` | Optional automatic public holidays; the site calendar file works without it |
| `dev` | `pytest` | Tests |

A comment records the versions known to work in the 18 Nov 2026 install: ultralytics 8.4.71, transformers 5.15.1, torch 2.11.0, chronos-forecasting 2.3.2. The file has no `[build-system]` table and no console-script entry.

#### 1.8.10 Runtime files under `data/`

| Path | Written by |
|---|---|
| `data/sbi.db` (+ WAL files) | Tick and CLI (`paths.db`) |
| `data/images/<space>/<date>/<ts>.jpg` | `capture_frame` |
| `data/logs/tick.log`, `tick.log.1` | `scripts/tick.sh` |
| `data/logs/curate.log` | `scripts/curate.sh` |
| `data/.tick.lock`, `data/.curate.lock` | `flock` in the two scripts |
| `data/cache/table_features.db` | `forecast/build.py` feature cache |
| `data/exports/*.csv` | CLI `table`, `backtest`, `accuracy`, `stability` (default `--out`) |

## 2. From photo to stored numbers

This section covers what happens to one stored photo after capture: which models look at it, what each one returns, how the results are cleaned, and exactly which database rows are written. It also covers the database itself (every migration, table, column, key and index), the catch-up jobs that fill in missing image vectors and prompt scores, the vocabulary taxonomy and curator, and the per-weekday/hour baselines.

Code lives in:

| Area | Files |
|---|---|
| Interfaces and component construction | `src/sbi/perception/types.py`, `src/sbi/perception/factory.py` |
| Models | `src/sbi/perception/detector_yolo.py`, `segmenter_sam2.py`, `embedder_dinov2.py`, `scorer_siglip2.py`, `scene_lmstudio.py` |
| Pure helpers (no model, no IO) | `src/sbi/perception/dedupe.py`, `masks.py`, `geometry.py`, `src/sbi/grid.py` |
| Label handling | `src/sbi/perception/normalise.py`, `src/sbi/taxonomy.py`, `src/sbi/curation/` |
| Orchestration for one frame | `src/sbi/pipeline.py` |
| Storage | `src/sbi/store/` and `src/sbi/store/migrations/001..008` |
| Catch-up and derived tables | `src/sbi/embedding.py`, `src/sbi/scoring.py`, `src/sbi/baseline.py` |

`src/sbi/perception/__init__.py` contains only the docstring "Perception components (Engine 1): detector, scene reader, label normalisation." The code calls the per-frame stage "Engine 1".

### 2.1 The per-frame flow at a glance

`src/sbi/pipeline.py::process_frame` handles one row of `frames`. All model work runs first, outside any database transaction. Then every row for the frame is written in one transaction.

```
frames row (capture_status='ok', image_path set)
  -> read image (BGR)
  -> detector.detect            YOLO26x @1280, every box >= store_conf, `counted` flag, two box dedupe rules
  -> segmenter.segment          SAM 2, one outline per box (optional; failure -> no outlines)
  -> uncount_outline_duplicates outlines of one type that coincide -> weaker box uncounted (optional)
  -> embedder.embed             DINOv2 whole-scene vector (optional; failure -> no vector)
  -> scorer.score               SigLIP 2 logit per probe prompt (optional; failure -> no scores)
  -> reader.read / replay       VLM scene reading as JSON (optional; failure -> stored error)
  -> ONE transaction:
       clear old results -> detections -> observations (yolo) -> geometry relations
       -> observations (vlm) -> vlm relations -> scene_state -> events -> frames columns
       -> frame_embeddings -> frame_scores
```

The module docstring of `pipeline.py` states the rules: slow work runs before the write transaction so the database lock is held briefly; all rows of a frame are written in one transaction so a failure leaves the frame as it was; re-running a frame replaces its earlier results; only boxes marked `counted` become observations, links and totals; missing results are stored as NULL or no row ("unknown, not zero").

### 2.2 Perception interfaces (`perception/types.py`)

The module docstring says: to add a new detector, segmenter, embedder, scorer, scene reader or camera source, implement the Protocol, register it in `factory.py`, and add its `kind` to the settings model. No pipeline code changes.

#### 2.2.1 Dataclasses

`Detection` (frozen) — one detector box.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `label` | `str` | — | Label already mapped through the space's `detector_map` |
| `conf` | `float` | — | Detector confidence |
| `bbox` | `BBox` = `tuple[float, float, float, float]` | — | Normalised `[x1, y1, x2, y2]` |
| `counted` | `bool` | `True` | `False` means below the counting threshold: stored as a raw detection, never counted |

`SegMask` (frozen, `eq=False` because it holds a numpy array) — one outline.

| Field | Type | Meaning |
|---|---|---|
| `area` | `float` | Share of the image covered by the outline (comment: measured at full resolution) |
| `grid` | `np.ndarray` | Low-resolution boolean copy of the outline; this is what gets stored |

`SceneObject` (frozen) — one object kind read by the VLM.

| Field | Type | Meaning |
|---|---|---|
| `label` | `str` | Raw label as read by the model (normalised later against the registry) |
| `count` | `int` | Count of that kind |
| `status` | `str` | One value of `STATUS_VALUES` |

`Relation` (frozen) — one relation triple read by the VLM.

| Field | Type | Meaning |
|---|---|---|
| `subject` | `str` | Raw label |
| `predicate` | `str` | One of the fixed `PREDICATES` |
| `object` | `str` | Raw label |
| `count` | `int` | How many times the relation is seen |

`SceneReading` (frozen) — the whole VLM result for one photo.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `people_count` | `int` | — | People the VLM counted |
| `people` | `tuple[dict, ...]` | — | One dict per person (`posture`, `activity` per the JSON schema) |
| `objects` | `tuple[SceneObject, ...]` | — | Object kinds with counts and status |
| `scene` | `dict[str, str \| None]` | — | Space-specific scene fields (lighting, tidiness, ...) |
| `summary` | `str` | — | One-line description |
| `raw` | `str` | — | Raw model text, always stored |
| `model` | `str` | — | Model name |
| `prompt_version` | `str` | — | Prompt version (`PROMPT_VERSION`) |
| `relations` | `tuple[Relation, ...]` | `()` | Relation triples |
| `latency_s` | `float` | `0.0`, `compare=False` | Seconds for the successful request |

#### 2.2.2 Protocols

| Protocol | Attributes | Method | Contract (from docstrings) |
|---|---|---|---|
| `Detector` | `version: str`, `min_conf: float` (lowest confidence it returns, the storage floor, recorded per frame) | `detect(image, space) -> list[Detection]` | Every box down to `min_conf`; `counted` marks boxes above the counting threshold. Raise `DetectorError` on failure. |
| `Segmenter` | `version: str` | `segment(image, boxes: list[BBox]) -> list[SegMask]` | One outline per box, same order. Raise `SegmenterError`. |
| `Embedder` | `version: str` (model + preprocessing; a different version is a different vector space), `dim: int` | `embed(image) -> np.ndarray` | One float32 vector of length `dim` for the whole scene, BGR image in. Raise `EmbedderError`. |
| `Scorer` | `version: str` (scores from different models are different scales, never mixed) | `score(image, prompts: list[str]) -> list[float]` | One raw score per prompt, same order, BGR image in. Raise `ScorerError`. |
| `SceneReader` | — | `read(image, space) -> SceneReading`; `parse(raw, space) -> SceneReading` | `read` raises `VLMError` after retries. `parse` rebuilds a reading from a stored raw answer without a model call and raises `ValueError` if unusable. |
| `FrameSource` | — | `grab(space) -> np.ndarray` | Return a BGR image; raise `CameraError`. |

All error classes are in `src/sbi/errors.py` and derive from `SbiError`.

### 2.3 Building components (`perception/factory.py`)

Each builder maps the settings `kind` to a class. Model libraries are imported inside the branch ("lazy: keep torch out of imports that do not need it").

| Function | `kind` | Returns | Other `kind` |
|---|---|---|---|
| `build_detector(settings, root)` | `"yolo"` | `YoloDetector` | `ConfigError` |
| `build_segmenter(settings, root)` | `"none"` / `"sam2"` | `None` / `Sam2Segmenter` | `ConfigError` |
| `build_embedder(settings, root)` | `"none"` / `"dinov2"` | `None` / `DinoV2Embedder` | `ConfigError` |
| `build_scorer(settings, root)` | `"none"` / `"siglip2"` | `None` / `Siglip2Scorer` | `ConfigError` |
| `build_scene_reader(settings)` | `"lmstudio_openai"` | `LmStudioSceneReader` | `ConfigError` |
| `build_frame_source(settings, creds)` | `"nvr_rtsp"` | `ingest.sources.NvrRtspSource` | `ConfigError` |

The settings models in `src/sbi/settings.py` declare each `kind` as a `Literal`, so an unknown kind is already rejected when settings load; the `ConfigError` branch is a second guard.

Failure behaviour of the constructors:

| Class | Missing weights | Load failure |
|---|---|---|
| `YoloDetector` | `DetectorError("detector weights not found: ...")` | Any exception wrapped as `DetectorError` |
| `Sam2Segmenter` | `SegmenterError("segmenter weights not found: ...")` | Wrapped as `SegmenterError` |
| `DinoV2Embedder` | (no file check; Hugging Face id or local path) | Wrapped as `EmbedderError` |
| `Siglip2Scorer` | (no file check) | Wrapped as `ScorerError` |
| `LmStudioSceneReader` | — | Constructor does not contact the server; it only stores settings, a `requests.Session` and a sleep function |

Relative weight paths are resolved against the project `root`. How callers react (from `src/sbi/tick.py::run_tick`): a `DetectorError` at build time is logged and the tick keeps capturing, so frames wait unprocessed in the backlog; segmenter, embedder and scorer build failures are logged and the tick continues without that component; those three are built only when a detector exists. `sbi reprocess` (`src/sbi/cli.py`) lets a `SegmenterError` stop the command, with the comment "do not silently wipe stored outlines".

Version strings, stored with results:

| Component | `version` | Example with current config |
|---|---|---|
| Detector | `f"{weights.name}@{imgsz}"` | `yolo26x.pt@1280` |
| Segmenter | `f"{weights.name}@{mask_width}"` | `sam2.1_l.pt@384` |
| Embedder | `EmbedderSettings.version` = `f"{Path(model).name}@{width}"` | `dinov2-base@448` |
| Scorer | `ScorerSettings.version` = `Path(model).name` | `siglip2-base-patch16-224` |
| Scene reader | (no version attribute) `SceneReading.model` = settings `model`, plus `PROMPT_VERSION` | `qwen/qwen3.5-9b`, `v3` |

### 2.4 Detector (`perception/detector_yolo.py`, `perception/dedupe.py`)

#### 2.4.1 Settings

`config/pipeline.yaml` block `detector`, validated by `DetectorSettings`:

| Key | Config value | Code default | Meaning |
|---|---|---|---|
| `kind` | `yolo` | `yolo` | Only implementation |
| `weights` | `models/yolo26x.pt` | same | Chosen by benchmark on 18 Nov 2026 (config comment: chairs exact in 80% of cabin photos vs 31% for yolo12x at 1920) |
| `imgsz` | `1280` | `1280` (≥ 320) | Inference size. Config comment: larger sizes were worse here (1920: 59%) |
| `conf` | `0.25` | `0.25` (0 < v < 1) | Counting threshold: boxes at or above it are counted |
| `store_conf` | `0.05` | `0.05` (0 < v < 1) | Storage floor: every box down to here is kept in `detections` |
| `class_conf` | `{chair: 0.10, motorcycle: 0.10}` | `{}` | Per-label counting thresholds. Config comment: set from labels and the known chair count; with duplicate outlines removed, 0.10 beats 0.25 for both |
| `iou` | `0.5` | `0.5` | IoU passed to the model's own NMS |
| `dedupe_labels` | `[person]` | `["person"]` | Labels where a box mostly inside a stronger box is dropped |
| `dedupe_containment` | `0.8` | `0.8` | Containment threshold for that rule |
| `dedupe_same_label_iou` | `0.6` | `0.6` | Same-label box IoU threshold |
| `dedupe_outline_iou` | `0.5` | `0.0` (0 = off) | Outline IoU at which counted boxes of one type are one object; needs the segmenter |

`DetectorSettings._thresholds_above_floor` rejects the settings unless `store_conf <= v < 1` for `conf` and for every `class_conf` value. `DetectorSettings.count_threshold(label)` returns `class_conf.get(label, conf)`. The key is the label after `detector_map` (for example `chair`, `motorcycle`), not the model's class name.

#### 2.4.2 `YoloDetector`

- `__init__(settings, root)`: resolves the weights path, raises `DetectorError` if the file does not exist, silences the warning "Unable to import Axes3D" (comment: environment quirk with two matplotlib installs), loads `ultralytics.YOLO`, picks device `0` when CUDA is available else `"cpu"`. Sets `version = f"{weights.name}@{imgsz}"` and `min_conf = store_conf`.
- `detect(image, space)`: calls `model.predict(image, imgsz=imgsz, conf=store_conf, iou=iou, verbose=False, device=...)`, takes the first result, and builds a list of `RawBox = (normalised xyxy list, confidence, model class name)` from `r.boxes.xyxyn`, `r.boxes.conf`, `r.boxes.cls` and `model.names`. Any exception in this part becomes `DetectorError("detector inference failed: ...")`. Then returns `build_detections(raw, space, cfg)`.

The model runs once, at the storage floor. The counting decision is made afterwards in Python.

#### 2.4.3 `build_detections(raw, space, cfg)`

Pure post-processing (no model):

1. Class whitelist. `label = space.detector_map.get(cls_name)`. A class that is not a key of the space's `detector_map` is dropped (comment: "noise (e.g. refrigerator in a car park)"). The space file validator requires every `detector_map` target to be in the space's vocabulary.
2. Builds `Detection(label, round(conf, 3), bbox with each value rounded to 4 decimals, counted = float(conf) >= cfg.count_threshold(label))`. The `counted` test uses the unrounded confidence.
3. `dedupe_same_label_iou(dets, cfg.dedupe_same_label_iou)`.
4. `dedupe_contained(dets, set(cfg.dedupe_labels), cfg.dedupe_containment)`.

The returned list is ordered by descending confidence (both dedupe functions sort that way).

Current `detector_map` entries:

| Space | COCO class -> label |
|---|---|
| `admin_cabin` | person->person, chair->chair, couch->sofa, potted plant->plant, dining table->table, laptop->laptop, tv->monitor, keyboard->keyboard, mouse->mouse, cell phone->phone, backpack->bag, handbag->bag, suitcase->bag (file comment: the detector calls the large black bag on the desk a suitcase), bottle->bottle, cup->cup, book->book |
| `parking` | car, bicycle, motorcycle, truck, bus, person, chair, bottle map to themselves; backpack, handbag, suitcase -> bag |

#### 2.4.4 Box dedupe rules (`perception/dedupe.py`)

Helpers: `_area(b)` = `max(0, x2-x1) * max(0, y2-y1)`; `_inter(a, b)` = area of the intersection box; `iou(a, b)` = `inter / max(area_a + area_b - inter, 1e-9)`.

| Function | Applies to | Rule | Keeps |
|---|---|---|---|
| `dedupe_same_label_iou(dets, thr)` | All labels | Walk boxes by descending confidence. Drop a box if a kept box with the same label has box IoU **strictly greater** than `thr` (0.6). | The stronger box |
| `dedupe_contained(dets, labels, thr)` | Only labels in `dedupe_labels` (`person`) | Walk by descending confidence. Drop a box if a kept box of the same label satisfies `inter / max(min(area_this, area_kept), 1e-9) > thr` (0.8), that is, the intersection covers more than 80% of the smaller of the two boxes. Boxes of other labels are always kept. | The stronger box |

Both functions sort with Python's stable sort on `-conf`, so for equal confidence the box that came earlier in the input wins. The docstring of `dedupe_same_label_iou` gives the reason: different source classes can collapse to one label after mapping (handbag + backpack -> bag) and box the same object twice. The docstring of `dedupe_contained` gives the case (a second box on a seated person's torso) and the trade-off (a person almost fully hidden behind another may be dropped too).

Both rules run on all boxes down to `store_conf`, counted or not. Because boxes are visited strongest first and labels share one threshold, a counted box can only be removed by a stronger box of the same label, which is also counted. The module docstring states the intent: "the counted set is exactly what running the model at the counting threshold would give."

#### 2.4.5 Outline dedupe rule: `uncount_outline_duplicates(dets, outlines, min_iou)`

Runs in `pipeline.process_frame` only when a segmenter produced outlines for the frame and `dedupe_outline_iou > 0` (currently 0.5).

- Raises `ValueError` if `len(outlines) != len(dets)`. `outlines[i]` belongs to `dets[i]`.
- Groups the indexes of **counted** detections by `label`. Uncounted detections are not considered.
- For each label group, calls `masks.drop_duplicate_masks([(conf, outline), ...], min_iou)` and gets the positions to keep.
- Every counted detection not kept is replaced by a copy with `counted=False`. It is not removed: it is still stored as a raw detection with its outline.
- Order of the list is kept.

Tie-breaking comes from `drop_duplicate_masks` (section 2.6): strongest confidence first; for equal confidence the earlier item in the group wins; outlines with zero pixels are never dropped. The comparison uses `>= min_iou` on outline IoU at the stored low resolution.

Docstring reason: boxes overlap a lot between real neighbours (chairs in a row, parked motorcycles) but their outlines do not, so this removes doubles that the box rules cannot, and allows a low counting threshold without counting objects twice.

#### 2.4.6 What `counted` means

After all three rules, `counted=True` means: the box's label is whitelisted for the space, its confidence is at or above `count_threshold(label)`, it survived both box dedupe rules, and (when outlines exist and the rule is on) no stronger counted box of the same label has an outline with IoU ≥ 0.5. Only counted boxes become `observations` rows with `source='yolo'`, geometry relations, `people_yolo` and `vehicles_yolo`. Every box that survived the two box rules is stored in `detections`, with `counted` 1 or 0.

### 2.5 Segmenter (`perception/segmenter_sam2.py`)

Module docstring: SAM 2 does not name or find objects; it only traces the outline of what is inside each box it is given.

Settings (`segmenter` block, `SegmenterSettings`):

| Key | Config | Code default | Meaning |
|---|---|---|---|
| `kind` | `sam2` | `none` | `none` = no outlines; mask columns stay NULL |
| `weights` | `models/sam2.1_l.pt` | same | SAM 2.1 large |
| `mask_width` | `384` | `384` (32..2048) | Width of the stored outline (config comment: 384 x 216 for a 16:9 camera) |
| `batch` | `64` | `64` (≥ 1) | Boxes per model call |

`Sam2Segmenter`:

- `__init__(settings, root)`: checks the weights file (else `SegmenterError`), loads `ultralytics.SAM`, keeps a reference to `torch`, selects device `0` or `"cpu"`. Load errors become `SegmenterError`. `version = f"{weights.name}@{mask_width}"`.
- `_outlines(image, pixel)`: one model call with `bboxes=pixel` (pixel coordinates). Returns `r.masks.data.float()` as an `(n, H, W)` tensor in box order, or `None` when `r.masks` is `None` or the number of masks differs from the number of boxes (docstring: the model drops a box it finds nothing in).
- `segment(image, boxes)`:
  1. Empty `boxes` -> `[]`.
  2. Grid size: `gw = mask_width`, `gh = max(1, round(gw * h / w))` from the photo's height and width.
  3. Boxes are processed in chunks of `batch`. Normalised boxes are converted to pixels (`x * w`, `y * h`).
  4. If `_outlines` returns `None` for a chunk ("short batch"), every box in that chunk is sent again on its own. If no single box returns a mask, it raises `SegmenterError("segmenter returned no mask for any of N boxes")`. Otherwise boxes without a mask get a zero tensor shaped like the first mask found, and the chunk is concatenated in box order. It logs how many boxes had nothing to outline ("stored as empty outlines").
  5. `area` per box = mean of the mask tensor over height and width, rounded to 6 decimals.
  6. `grid` per box = `torch.nn.functional.interpolate(mask, size=(gh, gw), mode="area") >= 0.5`, moved to CPU as a numpy boolean array.
  7. Returns one `SegMask(area, grid)` per box, in input order.
- Errors: a `SegmenterError` raised inside is re-raised; any other exception becomes `SegmenterError("segmenter inference failed: ...")`. One failing chunk fails the whole call; there are no partial results.

The pipeline passes every stored box (counted and uncounted) to the segmenter, so every row in `detections` gets an outline when segmentation succeeds.

### 2.6 Mask codec and mask arithmetic (`perception/masks.py`)

Pure numpy. Module docstring: mostly-empty grids compress to a few hundred bytes.

| Function | Purpose and behaviour |
|---|---|
| `encode_mask(grid) -> bytes` | Requires a 2-D `bool` array, else `ValueError`. Output = 4-byte header `struct "<HH"` (height, width, little-endian unsigned 16-bit) + `zlib.compress(np.packbits(grid).tobytes(), 6)`. |
| `decode_mask(blob) -> np.ndarray` | Reads the header, decompresses, `np.unpackbits(..., count=h*w)`, reshapes to `(h, w)` bool. `struct.error`, `zlib.error` or `ValueError` become `ValueError("corrupt mask blob: ...")`. |
| `mask_iou(a, b) -> float` | `|a AND b| / |a OR b|`; returns `0.0` when the union is empty. |
| `mask_extent(mask)` | `(pixels set, first row, row end, first column, column end)`; ends are exclusive. All zeros for an empty mask. |
| `_iou_at_least(a, ea, b, eb, min_iou) -> bool` | Same answer as `mask_iou(a, b) >= min_iou`, with pruning: if either mask is empty returns `0.0 >= min_iou`; if `min(na, nb) / max(na, nb) < min_iou` returns `False` (IoU can never exceed smaller area / larger area); if the two bounding extents do not intersect returns `0.0 >= min_iou`; otherwise counts the intersection only inside the overlap of the two extents and returns `inter / (na + nb - inter) >= min_iou`. Docstring: about 20x less work than `mask_iou`. |
| `drop_duplicate_masks(items, min_iou, extents=None) -> list[int]` | `items` are `(confidence, mask)` for boxes of one object type. Visits indexes by descending confidence (stable, so equal confidence keeps input order). An item with a non-empty mask is dropped if `_iou_at_least` holds against any already kept item. Empty masks are always kept ("cannot be compared"). Returns the kept indexes sorted ascending. `extents` can be passed so the same outline is not measured again when compared under several sources. |
| `cell_shares(mask, cols, rows) -> np.ndarray` | Share of each grid cell covered by the mask, shape `(rows, cols)`. Pixel row `r` belongs to cell row `r * rows // h`, pixel column `c` to cell column `c * cols // w` (docstring: cell edges match `grid.point_to_cell`). Uses `np.bincount`; a cell with zero pixels divides by 1. |

Users: the pipeline uses `encode_mask` and (through `dedupe.uncount_outline_duplicates`) `drop_duplicate_masks`. `forecast/mask_features.py` uses `cell_shares`, `drop_duplicate_masks` and `mask_extent`; `forecast/table.py` and `eval/thresholds.py` use `decode_mask`.

### 2.7 Geometry and grid cells

#### 2.7.1 `perception/geometry.py`

Module docstring: deterministic links computed from detector boxes (no model judgement, fully reproducible).

- `overlap_links(dets, min_ios=0.3) -> list[tuple[str, str, str, int]]`: for every unordered pair of detections with **different** labels, computes `_inter(a, b) / smaller box area` (intersection over smaller). If the smaller area is positive and the ratio is `>= min_ios`, the pair `tuple(sorted((label_a, label_b)))` is counted once. Returns `(subject, "overlaps", object, n)` for each pair, sorted by `(subject, object)`. The subject is the alphabetically first label, so `(chair, person)` and `(person, chair)` are the same link. The pipeline calls it with the default `0.3` on counted detections only.
- `union_area(boxes) -> float`: exact area covered by a set of normalised boxes, overlaps counted once. Drops boxes with zero width or height. Builds the sorted distinct x and y edges, and for each elementary rectangle adds its area if its centre lies inside any box (inclusive edges). Returns `0.0` for no boxes. Docstring: a "how full" measure that does not depend on separating packed or doubly-boxed objects. Used by `forecast/table.py`.

#### 2.7.2 `grid.py`

Module docstring: the VLM never sees the grid; cells are computed from detector boxes.

| Name | Behaviour |
|---|---|
| `BBox` | `tuple[float, float, float, float]` |
| `_EDGE = 0.999999` | Keeps a coordinate of exactly 1.0 inside the last cell |
| `cell_id(col, row)` | `f"R{row}C{col}"`, 0-based, for example `R2C5` |
| `point_to_cell(x, y, cols, rows)` | `(min(max(int(x*cols), 0), cols-1), min(max(int(y*rows), 0), rows-1))`; out-of-range values are clamped |
| `bbox_anchor(bbox, anchor)` | `"bottom_center"` -> `((x1+x2)/2, y2)`; `"center"` -> box centre; other value -> `ValueError` |
| `bbox_to_cell(bbox, cols, rows, anchor)` | Anchor point, each coordinate capped at `_EDGE`, then `point_to_cell`; returns `(col, row, cell_id)` |
| `bbox_to_cells(bbox, cols, rows)` | Every cell between the top-left and bottom-right corner cells, inclusive. Not used by the pipeline. |

The space file sets `grid.cols`, `grid.rows` (each 1..64) and `grid.anchor` (default `bottom_center`). Current spaces: `admin_cabin` 8 x 6, `parking` 12 x 8, both `bottom_center` (comment: "use the box's bottom-centre (feet / wheels) for the cell").

### 2.8 Whole-scene embedder (`perception/embedder_dinov2.py`)

Settings (`embedder` block, `EmbedderSettings`):

| Key | Config | Code default | Meaning |
|---|---|---|---|
| `kind` | `dinov2` | `none` | `none` = no new vectors; old ones stay |
| `model` | `facebook/dinov2-base` | same | Hugging Face id or local path |
| `width` | `448` | `448` (56..1792, must be a multiple of 14) | Input width; height follows the photo's aspect (config comment: 448 x 252) |

Module docstring: the photo is resized whole (no centre crop, which would drop the edges of a wide camera view). The camera's burned-in clock was tested and does not move the vector measurably, so nothing is masked.

| Function / method | Behaviour |
|---|---|
| `PATCH = 14`, `_MEAN`, `_STD` | Patch size; ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` |
| `prepare(image_bgr, width)` | `ValueError` on an empty image. `out_h = max(14, round(width * h / w / 14) * 14)`. Resize to `(width, out_h)` with `INTER_AREA`, BGR -> RGB, scale to 0..1, subtract mean, divide by std, transpose to `(3, H, W)` float32. |
| `combine(cls_token, patch_tokens)` | Returns `[unit(CLS) | unit(mean of patch tokens)]` as float32. `unit` divides by the L2 norm unless the norm is 0. Docstring: cosine similarity on the whole vector is then the average of the two halves' cosines. |
| `DinoV2Embedder.__init__(settings, root)` | If `model` is relative and `root / model` exists, uses the local path. Loads `transformers.AutoModel` with `local_files_only=True` first; on `OSError` loads normally (network). Device `"cuda"` or `"cpu"`, model in eval mode. `dim = 2 * model.config.hidden_size`. Errors become `EmbedderError`. `version = settings.version`. |
| `DinoV2Embedder.embed(image)` | Runs `prepare`, a batch of one, under `torch.inference_mode()`. Takes `last_hidden_state[0]`; token 0 is CLS, tokens 1.. are patches. Any exception becomes `EmbedderError("embedding failed: ...")`. Raises `EmbedderError` if the vector size is not `dim` or any value is not finite. |

For `dinov2-base` (hidden size 768) the vector has 1536 numbers. Storage converts it to little-endian float16 (`store/embeddings.py`, see `frame_embeddings` in section 2.13.5): 3072 bytes per frame.

### 2.9 Text-prompt scorer (`perception/scorer_siglip2.py`)

Settings (`scorer` block, `ScorerSettings`):

| Key | Config | Code default |
|---|---|---|
| `kind` | `siglip2` | `none` (no new scores) |
| `model` | `google/siglip2-base-patch16-224` | same |

Module docstring: SigLIP scores every prompt independently (a sigmoid per prompt, so several can be true at once, or none), unlike CLIP's pick-one-of-N. The raw logit is stored: 0 is an even-odds match, and logits keep their resolution where probabilities would saturate. The photo is squashed whole to the model's square input size.

| Item | Behaviour |
|---|---|
| `MAX_TOKENS = 64` | Text padding length |
| `Siglip2Scorer.__init__(settings, root)` | Local path rule as for the embedder. Device `"cuda"` or `"cpu"`; dtype `float16` on CUDA, `float32` on CPU. Loads `AutoProcessor` and `AutoModel` with `local_files_only=True`, falling back to a normal load on `OSError`. Errors become `ScorerError`. `version = settings.version`. |
| `Siglip2Scorer.score(image, prompts)` | Empty `prompts` -> `[]`. BGR -> RGB. Processor called with prompts lowercased, `padding="max_length"`, `max_length=64`, `return_tensors="pt"`. Floating tensors are cast to the model dtype. Returns `logits_per_image[0]` as floats. Exceptions become `ScorerError("scoring failed: ...")`; a result whose shape is not `(len(prompts),)` or that has non-finite values raises `ScorerError`. |

Prompts come from the space file's `probes` (concept -> list of paraphrases). `SpaceConfig.probe_prompts` returns every distinct prompt in a stable order (`dict.fromkeys` over the concepts' lists). The space validator requires concept names in snake_case, at least one prompt per concept, prompts of 1-200 characters, no prompt repeated in a concept or shared by two concepts. What is stored is one row per `(frame, scorer version, prompt text)` with the raw logit. Averaging paraphrases into a concept happens later in the table builder.

### 2.10 VLM scene reader (`perception/scene_lmstudio.py`)

#### 2.10.1 Settings

`scene_reader` block, `SceneReaderSettings`:

| Key | Config | Code default | Meaning |
|---|---|---|---|
| `kind` | `lmstudio_openai` | same | Any OpenAI-compatible vision endpoint |
| `base_url` | `http://10.20.0.1:1234` | required | Env override `SBI_LMSTUDIO_URL` (config comment) |
| `model` | `qwen/qwen3.5-9b` | required | Env override `SBI_VLM_MODEL` |
| `timeout_s` | `120` | `120` | Per-request timeout |
| `retries` | `3` | `3` (≥ 1) | Total attempts |
| `max_tokens` | `3000` | `1200` | Config comment: v3 answers reach about 3k characters; 1200 truncated the JSON |
| `temperature` | `0.1` | `0.1` | |
| `reasoning_effort` | `none` | `"none"` | Config comment: required for Qwen3.5 on LM Studio, otherwise structured output comes back empty. `null` omits the field. |
| `image_width` | `1344` | `1344` (≥ 256) | Width of the image sent |

#### 2.10.2 Constants and gotchas

Module docstring notes from testing:

- Qwen3.5 thinks by default. With `json_schema` output the JSON then lands in `reasoning_content` and `content` is empty; `reasoning_effort: "none"` fixes it. The reader still falls back to `reasoning_content`.
- Counts move with prompt wording, so `PROMPT_VERSION` is stored on every frame and bumped on any edit.
- Never put a list of object names in the prompt. A test on 2026-11-13 found that with absent decoy names in a "reuse these names" list, the VLM reported all six decoys in half of the cabin photos; with the open prompt it reported none. Naming consistency is handled afterwards by the taxonomy and curator.

| Constant | Value |
|---|---|
| `PROMPT_VERSION` | `"v3"` (v3: open vocabulary, relation triples; v2: fixed status list, rubrics) |
| `STATUS_VALUES` | `present, in_use, idle, moving, parked, open, closed, on, off, unknown` |
| `POSTURES` | `seated, standing, walking, other` |
| `PREDICATES` | `sitting_on, standing_at, using, holding, carrying, wearing, riding, on, in, near, next_to, walking_past, parked_at` |

#### 2.10.3 Request

`prepare_image(image, mask_regions, width) -> str`:

1. Copies the image and fills each normalised `mask_regions` rectangle with black (`cv2.rectangle(..., thickness=-1)`). Both current spaces mask `[0.64, 0.03, 0.97, 0.09]`, the burned-in overlay region at the top right. This masking is done only here; the embedder and scorer receive the unmasked image.
2. Resizes to `width x int(width * h / w)` with `INTER_AREA` (1344 wide).
3. Encodes JPEG quality 90; a failed encode raises `VLMError`. Returns base64 text.

`build_prompt(space) -> str` — fixed English text with the space's `name` and `space_type` inserted. It instructs: describe only what is visible; do not identify anyone and do not read licence plates; list every distinct kind of physical object with short, generic, singular, lowercase nouns, one entry per kind with count and one status (each status explained); estimate counts when packed or hidden; list clearly visible relations as subject/predicate/object/count with examples; use `unknown` for scene fields that cannot be seen; `door_state` only if a door is clearly visible; tidiness definitions (tidy / messy / ok). It contains no list of object names.

`build_schema(space) -> dict` — JSON schema; every property is required:

| Property | Schema |
|---|---|
| `people_count` | integer |
| `people` | array of objects `{posture: enum POSTURES, activity: string}`, both required |
| `objects` | array of objects `{label: string, count: integer, status: enum STATUS_VALUES}`, all required |
| `relations` | array of objects `{subject: string, predicate: enum PREDICATES, object: string, count: integer}`, all required |
| one property per `space.scene_fields` entry | string enum of that field's allowed values |
| `summary` | string |

Scene fields per space: `admin_cabin` has `lighting [daylight, artificial, dim, dark]`, `door_state [open, closed, unknown]`, `tidiness [tidy, ok, messy, unknown]`; `parking` has `lighting`, `weather_visible [clear, cloudy, rain, fog, unknown]`, `occupancy_level [empty, sparse, moderate, full]`.

`LmStudioSceneReader._payload(image, space)` builds:

```
{"model": model, "temperature": temperature, "max_tokens": max_tokens,
 "messages": [{"role": "user", "content": [
     {"type": "text", "text": build_prompt(space)},
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}],
 "response_format": {"type": "json_schema",
                     "json_schema": {"name": "scene", "strict": true, "schema": build_schema(space)}},
 "reasoning_effort": reasoning_effort        # only when not None
}
```

#### 2.10.4 `read(image, space)`: retries and timeouts

- URL: `f"{base_url.rstrip('/')}/v1/chat/completions"`. The payload is built once, before the first attempt.
- Up to `retries` attempts (3). Each attempt is one `session.post(url, json=payload, timeout=timeout_s)`.

| Outcome of an attempt | Action |
|---|---|
| `requests.RequestException` (network, timeout) | Transient: retry |
| HTTP 400-499 | Raise `VLMError("scene reader rejected the request (HTTP n): ...")` at once (comment: retrying cannot help) |
| HTTP ≥ 500 | Retry |
| Other status | Take `choices[0].message`, text = `content` or `reasoning_content` or `""`, call `parse_reading`; return on success. `ValueError`, `KeyError`, `IndexError`, `TypeError` (bad, empty or misshapen answer) -> retry |

- Each failure is logged as a warning. Between attempts it sleeps `2 * attempt` seconds (2 s, then 4 s). After the last attempt: `VLMError("scene reader failed after N attempts: <last error>")`.
- `latency_s` is the time of the successful attempt, rounded to 0.01 s.

#### 2.10.5 Parsing and validation: `parse_reading(text, space, *, model, latency_s)`

| Part | Rule |
|---|---|
| Top level | `json.loads`; must be a dict, else `ValueError` |
| `people_count` | Required key; must be a non-negative `int` and not a `bool`, else `ValueError` |
| `objects` | Required key; each item needs `label` (non-empty string after strip) and `count` (non-negative int, not bool), else `ValueError`. `status` not in `STATUS_VALUES` becomes `"unknown"`. Label is stored stripped. |
| `relations` | Optional; each item kept only if subject and object are non-empty strings, predicate is in `PREDICATES`, and count is an int ≥ 1 (not bool). Malformed items are dropped with a debug log, not fatal. |
| Scene fields | For each `space.scene_fields` name: value kept if in the allowed list, else `"unknown"` if `"unknown"` is allowed, else `None` |
| `people` | `tuple(data.get("people") or ())`; items are not validated |
| `summary` | `str(data.get("summary", ""))` |

`people_count` is not checked against the length of `people`. The returned `SceneReading` has `raw=text`, `prompt_version=PROMPT_VERSION`.

`LmStudioSceneReader.parse(raw, space)` calls `parse_reading(raw, space, model=settings.model, latency_s=0.0)`. It is used to replay a stored answer (section 2.12.2).

#### 2.10.6 How failures are stored

The pipeline catches `VLMError`. The frame is still written with detector results; `scene_state.people_vlm`, scene fields and `summary` are NULL; no `source='vlm'` observations or relations are written; `frames.vlm_model` and `frames.prompt_version` are NULL; `frames.vlm_raw` holds `json.dumps({"error": "<message>"})`. When no reader is configured (`--no-vlm`), `vlm_raw` is NULL.

### 2.11 Label normalisation, taxonomy and curation

#### 2.11.1 Normal forms (`taxonomy.py`)

| Function | Behaviour |
|---|---|
| `spaced(label)` | Strip, lowercase, underscores to spaces, collapse whitespace. `"Office_Chair "` -> `"office chair"` |
| `snake(label)` | `spaced(label)` with spaces as underscores -> `"office_chair"` |
| `singular_candidates(k)` | `[k]`, then `k[:-3] + "y"` if it ends in `ies`, then `k[:-2]` if it ends in `es`, then `k[:-1]` if it ends in `s` but not `ss`. The original form is always tried first. Examples: `"boxes"` -> `["boxes", "box", "boxe"]`; `"batteries"` -> `["batteries", "battery", "batteri", "batterie"]`; `"glass"` -> `["glass"]`. |

#### 2.11.2 Write-time mapping: `perception/normalise.py::canonical`

`canonical(conn, raw_label, *, space_id, frame_id, ts_utc, category="other") -> (object_type, is_new)`. Must be called inside a transaction (it may write).

1. `k = spaced(raw_label)`; empty -> `ValueError("empty object label")`.
2. For each `cand` in `singular_candidates(k)`: `store/registry.py::find_object_type(conn, cand)`. The first hit is returned with `is_new=False`.
3. No hit: `new = snake(raw_label)` (the label as given, not singularised) and `add_auto_new_object_type(...)` creates a registry row with `review_status='auto_new'` and an `events` row of type `new_object_type`. Returns `(new, True)`.

`find_object_type(conn, candidate)`:
- Exact match of `candidate.replace(" ", "_")` against `object_registry.object_type`; else
- match of `candidate.replace("_", " ")` against `object_alias.alias`;
- either hit is passed through `_follow_merges`, which follows `merged_into` until it reaches a row with no `merged_into` (cycle-safe via a `seen` set).
- Returns `None` if neither matches.

`add_auto_new_object_type(conn, object_type, *, raw_label, space_id, frame_id, ts_utc, category="other")` inserts `object_registry(object_type, category, mobility_prior='unknown', review_status='auto_new', first_seen_utc, first_seen_frame_id, first_seen_space_id)` and `events(space_id, ts_utc, frame_id, event_type='new_object_type', object_type, resolution='interval', details_json={"raw_label": ...})`. The pipeline never passes `category`, so new rows get `category='other'`. Because this runs inside the frame's transaction, a rollback also removes the new registry row.

#### 2.11.3 Read-time mapping: the `Taxonomy` class

Module docstring: two kinds of link. `same` = synonyms, materialised as `object_alias` rows and `object_registry.merged_into`. `is_a` = "backpack is_a bag": both names stay as their own dimension, and the parent also gets a roll-up dimension that sums its descendants. Resolution is a pure function of the stored links, so changing a link and re-resolving re-maps all history.

`Taxonomy` (dataclass) fields:

| Field | Type | Content |
|---|---|---|
| `known` | `set[str]` | Every `object_registry.object_type` (canonical or merged) |
| `alias` | `dict[str, str]` | `object_alias.alias` -> `object_type` |
| `merged_into` | `dict[str, str]` | Rows with `merged_into` set |
| `parents` | `dict[str, set[str]]` | Approved `is_a` links, both ends followed through merges |

| Method / function | Behaviour |
|---|---|
| `_follow(name)` | Follow `merged_into`, cycle-safe |
| `lookup(raw)` | For each singular candidate of `spaced(raw)`: if its snake form is in `known` return `_follow` of it; else if the spaced candidate is an alias return `_follow(alias target)`. `None` if unknown. Same rule as `find_object_type`, but in memory. |
| `resolve(raw)` | `lookup(raw)` or `snake(raw)` |
| `ancestors(name)` | All `is_a` ancestors, transitive, each followed through merges, excluding `name`; cycle-safe |
| `load_taxonomy(conn)` | Builds the object from `object_registry`, `object_alias`, and `object_links WHERE relation='is_a' AND status='approved'`; a link whose two ends resolve to the same name is ignored |
| `reresolve_observations(conn)` | Caller holds the transaction. For every `observations` row with a `raw_label`, sets `object_type = lookup(raw_label)` when lookup finds a name and it differs. For every `relations` row with a `raw_subject`, sets `subject = lookup(raw_subject) or subject` and `object = lookup(raw_object) or object` when either changes. Returns the number of changed rows. Unknown raw labels keep their current value. `detections` is not touched (its `raw_label` is mapped on read). |

`forecast/table.py` uses `load_taxonomy` and `ancestors` to build roll-ups.

#### 2.11.4 Seeding the registry: `store/registry.py::register_space`

Called at the start of every tick for each space file (`tick.py::run_tick`). In one transaction:

- Upserts `spaces` (`space_id, name, space_type, grid_cols, grid_rows, tz, config_path, created_utc`); on conflict it updates everything except `created_utc`.
- For every vocabulary entry: `INSERT OR IGNORE INTO object_registry(object_type, category, mobility_prior, review_status='seed')`. Existing rows are never overwritten.
- For every alias: `INSERT OR REPLACE INTO object_alias(alias.lower(), target)`. Config aliases therefore overwrite any alias row with the same key on every tick.

`category_of(conn, object_type)` returns the registry category or `None`.

#### 2.11.5 Curation (`curation/curator.py`, `curation/services.py`)

Module docstring method, validated 2026-11-13 on 79 candidate pairs from real VLM output (11 links, all correct; the one-shot "group these names" and single-question variants produced results such as "desk is_a keyboard"):

1. Shortlist name pairs by text-embedding cosine ≥ `min_similarity`.
2. Ask two independent yes/no questions, "is every A a B?" and "is every B an A?".

   | A->B | B->A | Result |
   |---|---|---|
   | yes | yes | `same` |
   | yes | no | A `is_a` B |
   | no | yes | B `is_a` A |
   | no | no | no link |

3. Store the link with evidence; if auto-approved, apply it and re-map history from raw labels.

Settings (`curation` block, `CurationSettings`):

| Key | Config | Code default |
|---|---|---|
| `embed_model` | `text-embedding-nomic-embed-text-v1.5` | same |
| `judge_model` | `qwen/qwen3.5-9b` | same |
| `min_similarity` | `0.80` | `0.80` |
| `auto_approve` | `true` | `True` (CLI `--no-auto-approve` turns it off for a run) |
| `context` | (not set) | "Object names produced by a vision model describing photos of an office cabin and a parking area. In an office, 'notebook' means a paper notebook." |
| `timeout_s` | (not set) | `60` |
| `retries` | (not set) | `3` |

Both services use the scene reader's `base_url` (`cli.py`).

Services:

| Item | Behaviour |
|---|---|
| `TextEmbedder` Protocol | `name`; `embed(texts) -> np.ndarray` of L2-normalised rows; raise `CurationError` |
| `PairJudge` Protocol | `name`; `is_every(a, b) -> bool`; raise `CurationError` |
| `_post(session, url, payload, *, timeout, retries, sleep)` | Up to `retries` attempts. HTTP 4xx -> `CurationError` at once. Other HTTP errors (`raise_for_status`), `requests.RequestException` and `ValueError` (bad JSON) -> retry after `2 * attempt` seconds. After the last attempt -> `CurationError`. |
| `LmStudioEmbedder.embed(texts)` | POST `{base_url}/v1/embeddings` with `model = embed_model` and inputs prefixed `"clustering: "`. Sorts `data` by `index`, checks the row count, returns rows divided by `max(norm, 1e-12)`. |
| `LmStudioJudge.is_every(a, b)` | POST `{base_url}/v1/chat/completions`, `temperature 0`, `max_tokens 20`, `reasoning_effort "none"`, strict JSON schema `{"answer": enum ["yes", "no"]}`. Prompt: `"{context} Is every '{a}' a '{b}'? Answer yes only if '{a}' is the same thing as, or a specific kind of, '{b}'. Being found together or being related does not count."` Reads `content` or `reasoning_content`; any other answer -> `CurationError`. |

Curator functions:

| Function | Behaviour |
|---|---|
| `CurationReport` | `names`, `candidates`, `questions_asked`, `questions_cached`, `links` (list of dicts), `reresolved_rows` |
| `_judge(conn, judge, a, b, report)` | Looks up `label_judgments(a, b, model)`; a hit counts as cached. Otherwise asks the judge with `spaced(a)`, `spaced(b)` outside any transaction, then stores the answer in its own transaction (`INSERT OR REPLACE`). |
| `_support(conn, name)` | `(trusted, n)`: trusted = 1 if `review_status` is `seed` or `approved`; n = number of `observations` with that `object_type` |
| `pick_canonical(conn, a, b)` | Compares `(trusted, n_observations, -len(name), name)` for both names; the larger tuple becomes the canonical (parent). Order: configured/reviewed names, then the more observed, then the shorter, then the alphabetically later name. |
| `apply_same(conn, child, parent)` | Caller holds the transaction. Sets child `merged_into=parent, review_status='merged'`; re-points rows whose `merged_into` was the child; re-points aliases that targeted the child; adds alias `spaced(child) -> parent`. |
| `undo_same(conn, child, parent)` | Clears `merged_into` and sets `review_status='approved'` on the child (only if it was merged into that parent); deletes the alias `spaced(child) -> parent`. |
| `_record(conn, child, parent, relation, *, approved, similarity, evidence, model)` | `INSERT OR IGNORE INTO object_links` with `source='curator'`, status `approved` or `proposed`, `similarity` rounded to 4 decimals, `evidence_json`, `decided_utc` only when approved. Returns the new `link_id`, or 0 if the triple already existed. |
| `curate(conn, embedder, judge, *, min_similarity, auto_approve)` | See below. |
| `set_link_status(conn, link_id, status)` | Human decision, `status` must be `approved` or `rejected` (else `ValueError`). In one transaction: for a `same` link, approving a not-yet-approved link runs `apply_same`; rejecting an approved link runs `undo_same`. Then sets `status`, `source='human'`, `decided_utc`, and runs `reresolve_observations`. Returns a dict with the link and `reresolved_rows`. |

`curate` in order:

1. `names` = registry rows with `merged_into IS NULL`, sorted. Fewer than 2 -> return.
2. Embeds `spaced(name)` for all names; similarity = dot product of the normalised vectors.
3. Candidate pairs: every pair `i < j` with similarity ≥ `min_similarity` that does not already appear in `object_links` in either direction (any relation, any status). Sorted by similarity, highest first.
4. For each pair: skip it if either name has been merged (checked against the database each time, so earlier merges in the same run count). Ask both questions through `_judge`. If both are no, skip. Otherwise, in one transaction: yes/yes -> `pick_canonical`, record a `same` link, and if a new link was recorded and auto-approve is on, `apply_same`; one yes -> record an `is_a` link with child = the name that "is every" the other. Recorded links are appended to the report.
5. If anything was applied, one more transaction runs `reresolve_observations` and stores the count in the report.

History re-mapping works because `observations.raw_label`, `relations.raw_subject` and `relations.raw_object` keep the model's original text; `reresolve_observations` recomputes canonical names from them. `is_a` links do not change any stored row; they only change roll-ups computed at read time.

### 2.12 `pipeline.process_frame` end to end

Signature:

```
process_frame(conn, space, frame_id, detector, reader, *, root, replay_vlm=False,
              segmenter=None, embedder=None, scorer=None, dedupe_outline_iou=0.0) -> FrameResult
```

Callers: `tick.py::_run_space` (the new frame plus backlog; passes segmenter, embedder, scorer and `settings.detector.dedupe_outline_iou`) and `cli.py` `reprocess` (passes segmenter, `replay_vlm=args.detector_only` and `dedupe_outline_iou`; it does not pass an embedder or scorer).

#### 2.12.1 Before the transaction (exact order)

| Step | Code | Failure behaviour |
|---|---|---|
| 1 | `get_frame(conn, frame_id)` | Missing row -> `StorageError("frame N does not exist")` |
| 2 | Check `capture_status == 'ok'` and `image_path` set | Otherwise returns `FrameResult(frame_id, False, reason=f"capture_status={status}")` and writes nothing |
| 3 | `imageio.read_frame_image(frame, root)` (`cv2.imread` of `root / image_path`) | `None` -> returns `FrameResult(frame_id, False, reason="image unreadable")` |
| 4 | `detections = detector.detect(img, space)` | `DetectorError` propagates; nothing written |
| 5 | If a segmenter is given: `masks = segmenter.segment(img, [d.bbox for d in detections])`; a length mismatch raises `SegmenterError` | `SegmenterError` -> `masks = None`, warning logged, frame continues without outlines |
| 6 | `seg_version = segmenter.version` if a segmenter ran and `masks` is not `None`, else `None` | — |
| 7 | If `masks` is not `None` and `dedupe_outline_iou > 0`: `detections = uncount_outline_duplicates(detections, [m.grid for m in masks], dedupe_outline_iou)` | — |
| 8 | If an embedder is given: `vector = embedder.embed(img)`; a shape other than `(dim,)` raises `EmbedderError` | `EmbedderError` -> `vector = None`, warning logged |
| 9 | If a scorer is given and `space.probe_prompts` is non-empty: `values = scorer.score(img, prompts)`; a length mismatch raises `ScorerError`; `scores = dict(zip(prompts, values))` | `ScorerError` -> `scores` stays `None`, warning logged |
| 10 | VLM: if `replay_vlm`, `_stored_reading(frame, space, reader)`; elif a reader is given, `reader.read(img, space)` | Replay: `StorageError` propagates, nothing written. Live: `VLMError` -> `vlm_error = str(e)`, warning logged |

Only the listed error types are caught. Any other exception (including `DetectorError` and `StorageError`) propagates and the frame stays unprocessed. In the tick, a `DetectorError` stops processing the rest of that space's list for this tick ("do not hammer a broken detector"); another `SbiError` skips that frame and continues.

Segmentation runs even when there are no detections; `segment([])` returns `[]`, so `seg_version` is set and `masks_ok` is true for a frame with zero boxes.

#### 2.12.2 `_stored_reading(frame, space, reader)`: `replay_vlm`

`replay_vlm=True` re-runs the detector (and segmenter) but rebuilds the VLM part from the stored answer without calling the model (`sbi reprocess --detector-only`).

| `frames.vlm_raw` | Result |
|---|---|
| `NULL` | `(None, None)`: no reading, no error; `vlm_raw` stays NULL |
| Starts with `{"error"` | `(None, stored error text)`; the error is stored again |
| Anything else, no reader | `StorageError("replaying a stored VLM answer needs a scene reader to parse it")` |
| Anything else | `reader.parse(raw, space)`; `ValueError`, `KeyError` or `TypeError` -> `StorageError("frame N: stored VLM answer cannot be replayed: ...")`. On success, the reading's `model` and `prompt_version` are replaced with the frame's stored `vlm_model` and `prompt_version`. |

`cli.py` refuses `--detector-only` together with `--no-vlm`.

#### 2.12.3 Inside the transaction

`with transaction(conn):` runs `_write(...)`, then `put_embedding(conn, frame_id, embedder.version, vector)` if a vector exists, then `put_scores(conn, frame_id, scorer.version, scores)` if scores exist. A failure anywhere rolls back every row of the frame.

`_write(conn, space, frame, all_detections, reading, vlm_error, detector, masks, seg_version)` in exact order:

1. **Clear.** `results.clear_frame_results(conn, frame_id)` deletes the frame's `observations`, `detections`, `relations`, `scene_state`, and `events` except `new_object_type` events. `frame_embeddings` and `frame_scores` are not cleared, so a failed embedder or scorer on a re-run never wipes earlier vectors or scores.
2. **Detections (raw layer).** For every detection (counted or not), in list order: cell from `bbox_to_cell(bbox, grid.cols, grid.rows, grid.anchor)`; `insert_detection` with `raw_label=d.label`, `confidence`, `bbox`, `cell_id`, `counted`, `mask=encode_mask(masks[i].grid)` and `mask_area=masks[i].area` when masks exist, else NULL.
3. **Detector observations.** For each counted detection: `canonical(d.label)`, cell from `bbox_to_cell`, `insert_observation(object_type=canonical, source='yolo', raw_label=d.label, cell=(col, row, id), bbox, confidence=d.conf)`, `count=1`.
4. **Geometry relations.** For each `(subject, "overlaps", object, n)` from `overlap_links(counted detections)`: canonicalise both labels; `insert_relation(subject, predicate='overlaps', obj, count=n, source='geometry', raw_subject, raw_object)`.
5. **Detector totals.** `people_yolo` = number of counted detections whose `label == "person"`. `vehicles_yolo` = number of counted detections whose `category_of(conn, d.label) == "vehicle"`. Both use the detector-mapped label, not the canonical one.
6. **VLM rows** (only when `reading` is not `None`):
   - `people_vlm = reading.people_count`.
   - One observation `object_type=canonical("person")`, `source='vlm'`, `raw_label='person'`, `count=people_vlm` (written even when the count is 0), `status` = most common non-empty `posture` in `reading.people` (or NULL), `attrs_json={"people": [...]}`.
   - One observation per `reading.objects` entry, skipping labels whose stripped lowercase form is `person` or `people` (comment: already stored from the structured people list): `object_type=canonical(o.label)`, `source='vlm'`, `raw_label=o.label`, `count=o.count`, `status=o.status`, no cell and no bbox.
   - One relation per `reading.relations` entry: canonical subject and object, the VLM predicate, `count`, `source='vlm'`, raw subject and object kept.
   - Scene fields split by `SCENE_COLUMNS = {"lighting", "door_state", "tidiness", "weather_visible"}`: those go to `scene_state` columns; all other fields (for example `occupancy_level` in `parking`) go to `scene_state.attrs_json`.
7. **Scene state.** `insert_scene_state(people_yolo, people_vlm, vehicles_yolo, columns, summary=reading.summary or NULL, extra or NULL)`. `count_disagree` is NULL when `people_vlm` is NULL, else `1` if `|people_yolo - people_vlm| > 1`, else `0`.
8. **Occupancy events.** `prev = previous_people(conn, space_id, ts_utc)` = the people count of the latest `scene_state` row of the space with `ts_utc` strictly earlier (VLM count if not NULL, else detector count), or `None`. `now_n = people_vlm` if not NULL else `people_yolo`. If `prev` is not `None`: `prev > 0 and now_n == 0` -> event `became_empty`; `prev == 0 and now_n > 0` -> event `became_occupied`. `details_json = {"prev": prev, "now": now_n}`; `object_type`, `cell_id` NULL; `resolution` defaults to `interval`.
9. **Frame bookkeeping.** `mark_processed` updates the `frames` row:

| Column | Value |
|---|---|
| `detector_version` | `detector.version` |
| `detector_min_conf` | `detector.min_conf` (the storage floor) |
| `segmenter_version` | `seg_version` (NULL when no segmenter or it failed) |
| `vlm_model` | `reading.model` or NULL |
| `prompt_version` | `reading.prompt_version` or NULL |
| `detector_raw` | JSON list of `{"label", "conf", "bbox", "counted"}` for every stored detection (after the outline rule) |
| `vlm_raw` | `reading.raw`; or `{"error": ...}` JSON; or NULL |
| `processed_utc` | Current UTC time, ISO seconds |

`_write` returns a `FrameResult`; `process_frame` then sets `embedded` and `scored`.

#### 2.12.4 `FrameResult`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `frame_id` | `int` | — | |
| `processed` | `bool` | — | `False` when skipped before any model ran |
| `reason` | `str \| None` | `None` | Why it was skipped |
| `people_yolo` | `int \| None` | `None` | |
| `people_vlm` | `int \| None` | `None` | |
| `boxes` | `int` | `0` | Counted boxes |
| `boxes_stored` | `int` | `0` | All raw detections, counted or not |
| `vlm_ok` | `bool` | `False` | A reading (live or replayed) was written |
| `masks_ok` | `bool` | `False` | A segmenter ran for this frame |
| `embedded` | `bool` | `False` | A vector was stored |
| `scored` | `bool` | `False` | Probe scores were stored |

#### 2.12.5 Backlog

`store/frames.py::pending_frame_ids(conn, space_id, limit)` returns frames with `capture_status='ok'` and `processed_utc IS NULL`, oldest first, up to `limit`. `config/pipeline.yaml` `processing.backlog_limit: 5` (code default 5, ≥ 0) sets that limit. Each tick processes the newly captured frame (when its status is `ok`) and then the pending frames other than it.

### 2.13 The database

#### 2.13.1 Connection: `store/db.py::connect(path)`

- Creates the parent directory, opens `sqlite3.connect(path, timeout=30, isolation_level=None)` (autocommit mode). `OSError` or `sqlite3.Error` -> `StorageError`.
- `row_factory = sqlite3.Row`.
- `PRAGMA foreign_keys=ON`, `PRAGMA journal_mode=WAL` (comment: readers such as the dashboard and labeller do not block the writer), `PRAGMA busy_timeout=30000`.

The database file is `data/sbi.db` (`paths.db`).

#### 2.13.2 `transaction(conn)`

Context manager. Module docstring: because the connection is in autocommit mode, all multi-statement writes must go through `transaction()`, so a half-written frame can never be committed by a later, unrelated write.

1. If `conn.in_transaction` -> `StorageError("nested transaction: the caller already holds one")`. It is not re-entrant.
2. `BEGIN IMMEDIATE` (takes the write lock at the start); a failure -> `StorageError`.
3. Any exception in the block (`BaseException`) -> `ROLLBACK`, then re-raise.
4. On success `COMMIT`; a failed commit -> `ROLLBACK` and `StorageError("commit failed: ...")`.

The writer functions in `store/results.py`, `store/embeddings.py`, `store/scores.py`, `store/forecasts.py`, `store/frames.py::insert_frame` and `registry.add_auto_new_object_type` assume the caller holds the transaction. `register_space` opens its own.

#### 2.13.3 `migrate(conn, directory=MIGRATIONS_DIR) -> int`

- Reads the current version from `PRAGMA user_version`.
- `_migration_files(directory)` globs `[0-9][0-9][0-9]_*.sql`, sorted, and requires the numbers to be exactly `1..n` with no gaps or duplicates, else `StorageError`.
- For each file with number greater than the current version, runs `conn.executescript("BEGIN IMMEDIATE;\n<file>\nPRAGMA user_version = n;\nCOMMIT;")`. Each migration and its version bump are atomic. On `sqlite3.Error` it rolls back if a transaction is open and raises `StorageError("migration <file> failed: ...")`.
- Returns the highest file number, or the current version if there are no files.
- Docstring: migrations must be safe for databases created before versioning existed (use `IF NOT EXISTS`).

`migrate` runs at the start of every tick, every CLI database command, and in the labelling app.

#### 2.13.4 Migrations in order

| File | Adds |
|---|---|
| `001_core.sql` | Tables `spaces`, `object_registry`, `object_alias`, `frames`, `observations`, `scene_state`, `events`, `object_mobility`, `baseline_cell`, `baseline_space`; views `v_observations_wide`, `v_cell_counts`, `v_frame_counts`; indexes on frames and observations |
| `002_labels.sql` | Tables `label_frames`, `labels` (blind human labels) |
| `003_open_vocabulary.sql` | `observations.raw_label` (back-filled from `object_type`), `object_registry.merged_into`, tables `object_links`, `label_judgments`, `relations`, `frame_features`; indexes `ix_rel_frame`, `ix_obs_raw` |
| `004_raw_detections.sql` | Table `detections`; `frames.detector_min_conf` |
| `005_masks.sql` | `detections.mask`, `detections.mask_area`, `frames.segmenter_version` |
| `006_embeddings.sql` | Table `frame_embeddings` |
| `007_scores.sql` | Table `frame_scores` |
| `008_forecasts.sql` | Tables `forecast_runs`, `forecasts` |

Design rules written at the top of `001_core.sql`: new object types are new rows in `object_registry`, never new columns; missing is not zero (camera_down frames have NULL metrics and no observations); every row carries provenance and, through `frames`, model and prompt versions; `space_id` is on every fact row; `map_*` columns on `spaces` are a hook for linking rooms later.

#### 2.13.5 Tables

Column comments below are taken from the SQL files. "Writers" and "readers" name the Python modules that run SQL against the table.

##### `spaces` (001)

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `space_id` | TEXT | PRIMARY KEY | |
| `name` | TEXT | NOT NULL | |
| `space_type` | TEXT | NOT NULL | room, parking, ... |
| `rtsp_env` | TEXT | | Name of an env var holding the RTSP URL (no secrets in the DB). Not written by current code |
| `grid_cols` | INTEGER | NOT NULL | |
| `grid_rows` | INTEGER | NOT NULL | |
| `tz` | TEXT | NOT NULL | IANA name or `local` |
| `config_path` | TEXT | | |
| `site_id`, `map_x`, `map_y`, `map_rotation`, `map_transform_json` | TEXT / REAL | nullable | Provisions for a future multi-room map; unused |
| `created_utc` | TEXT | NOT NULL | |

Writer: `registry.register_space`. Readers: `cli.py`, views.

##### `object_registry` (001, 003)

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `object_type` | TEXT | PRIMARY KEY | Canonical label |
| `category` | TEXT | | person, vehicle, furniture, device, item, other |
| `mobility_prior` | TEXT | NOT NULL DEFAULT `'unknown'` | fixed, semi_static, dynamic, unknown (seeded guess) |
| `review_status` | TEXT | NOT NULL DEFAULT `'seed'` | SQL comment lists seed, approved, auto_new, needs_review. Code writes `seed` (register_space), `auto_new` (normalise), `merged` (apply_same), `approved` (undo_same) |
| `first_seen_utc` | TEXT | | Set for auto_new rows |
| `first_seen_frame_id` | INTEGER | | |
| `first_seen_space_id` | TEXT | | |
| `notes` | TEXT | | Not written by current code |
| `merged_into` | TEXT | (003) | Set when this name was merged as a synonym |

Writers: `register_space`, `add_auto_new_object_type`, `curator.apply_same` / `undo_same`. Readers: `find_object_type`, `category_of`, `load_taxonomy`, curator.

##### `object_alias` (001)

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `alias` | TEXT | PRIMARY KEY | Lowercase synonym, for example `couch` |
| `object_type` | TEXT | NOT NULL, REFERENCES `object_registry(object_type)` | |

Writers: `register_space`, `apply_same`, `undo_same`. Readers: `find_object_type`, `load_taxonomy`.

##### `frames` (001, 004, 005)

| Column | Type | Constraint | Meaning / writer |
|---|---|---|---|
| `frame_id` | INTEGER | PRIMARY KEY AUTOINCREMENT | |
| `space_id` | TEXT | NOT NULL, REFERENCES `spaces` | |
| `ts_utc` | TEXT | NOT NULL | ISO 8601 UTC |
| `ts_local` | TEXT | NOT NULL | |
| `tz` | TEXT | NOT NULL | |
| `date` | TEXT | NOT NULL | Local date `YYYY-MM-DD` |
| `dow` | INTEGER | NOT NULL | 0 = Monday .. 6 = Sunday |
| `dow_name` | TEXT | NOT NULL | |
| `hour` | INTEGER | NOT NULL | |
| `minute` | INTEGER | NOT NULL | |
| `slot_15` | INTEGER | NOT NULL | 0..95 quarter-hour of the day |
| `week_of_year` | INTEGER | NOT NULL | |
| `is_weekend` | INTEGER | NOT NULL | |
| `part_of_day` | TEXT | NOT NULL | night, morning, afternoon, evening |
| `image_path` | TEXT | | NULL after retention or when camera_down |
| `capture_status` | TEXT | NOT NULL | ok, camera_down, bad_frame |
| `width`, `height` | INTEGER | | |
| `brightness` | REAL | | Mean luminance 0..255 |
| `contrast` | REAL | | Luminance std dev |
| `sharpness` | REAL | | Variance of Laplacian |
| `detector_version` | TEXT | | `mark_processed` |
| `vlm_model` | TEXT | | `mark_processed` |
| `prompt_version` | TEXT | | `mark_processed` |
| `detector_raw` | TEXT | | JSON of every stored box, always kept |
| `vlm_raw` | TEXT | | Raw VLM answer, or `{"error": ...}` |
| `processed_utc` | TEXT | | NULL until Engine 1 has run |
| `detector_min_conf` | REAL | (004) | Lowest confidence stored for the frame. NULL = processed before 004, so counts below the counting threshold are unknown for that frame |
| `segmenter_version` | TEXT | (005) | NULL = no segmenter ran or it failed; mask-derived numbers unknown |
| | | UNIQUE `(space_id, ts_utc)` | |

Indexes: `ix_frames_space_time (space_id, ts_utc)`, `ix_frames_dow_hour (space_id, dow, hour)`.

Writers: `store/frames.py::insert_frame` (called by `ingest/capture.py`; columns up to `sharpness`) and `results.mark_processed`. Readers: most modules (pipeline, backfills, baseline, table builder, dashboard, labelling, eval).

`store/frames.py`:
- `ImageMetrics` (frozen): `width`, `height`, `brightness`, `contrast`, `sharpness`.
- `insert_frame(conn, space_id, dims, *, status, image_path, metrics) -> (frame_id, created)`: `INSERT OR IGNORE` with all `TimeDims` fields, image path, status and metrics (NULL when `metrics` is `None`). A repeat for the same `(space_id, ts_utc)` returns the existing id with `created=False`.
- `get_frame(conn, frame_id)`: the row or `None`.
- `pending_frame_ids(conn, space_id, limit)`: see 2.12.5.

##### `observations` (001, 003)

One row per detected or read object per frame.

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `obs_id` | INTEGER | PRIMARY KEY AUTOINCREMENT | |
| `frame_id` | INTEGER | NOT NULL, REFERENCES `frames` ON DELETE CASCADE | |
| `space_id` | TEXT | NOT NULL, REFERENCES `spaces` | |
| `ts_utc` | TEXT | NOT NULL | |
| `object_type` | TEXT | NOT NULL, REFERENCES `object_registry` | Canonical name |
| `cell_col` | INTEGER | | 0-based grid column, NULL if unlocated |
| `cell_row` | INTEGER | | |
| `cell_id` | TEXT | | For example `R2C5` |
| `bbox_norm` | TEXT | | JSON `[x1, y1, x2, y2]`, nullable |
| `status` | TEXT | | VLM status; for the VLM person row, the most common posture |
| `activity` | TEXT | | Free-vocabulary activity. Not written by current code (activities are in `attrs_json`) |
| `count` | INTEGER | NOT NULL DEFAULT 1 | |
| `confidence` | REAL | | Detector confidence for `yolo` rows |
| `source` | TEXT | NOT NULL | `yolo` or `vlm` |
| `attrs_json` | TEXT | | VLM person row: `{"people": [...]}` |
| `raw_label` | TEXT | (003) | Label before taxonomy; pre-003 rows were back-filled with `object_type` |

Indexes: `ix_obs_frame (frame_id)`, `ix_obs_space_type (space_id, object_type, ts_utc)`, `ix_obs_cell (space_id, cell_id)`, `ix_obs_raw (raw_label)` (003).

`yolo` rows: one per counted box, `count=1`, with cell, bbox and confidence. `vlm` rows: one person row plus one per object kind, with `count` and `status`, no cell or bbox.

Writer: `results.insert_observation`; updates by `taxonomy.reresolve_observations`. Readers: `baseline.py`, `forecast/table.py`, `curator._support`, `eval/vlm_eval.py`, `labeling/core.py`, the views.

`insert_observation(conn, *, frame_id, space_id, ts_utc, object_type, source, raw_label=None, count=1, cell=None, bbox=None, confidence=None, status=None, attrs=None)`: `raw_label` defaults to `object_type`; `bbox` and `attrs` are stored as JSON, or NULL when empty.

##### `scene_state` (001)

One row per processed frame.

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `frame_id` | INTEGER | PRIMARY KEY, REFERENCES `frames` ON DELETE CASCADE | |
| `space_id` | TEXT | NOT NULL | |
| `ts_utc` | TEXT | NOT NULL | |
| `people_yolo` | INTEGER | | Counted `person` boxes |
| `people_vlm` | INTEGER | | VLM `people_count`; NULL when no reading |
| `count_disagree` | INTEGER | | 1 if `|yolo - vlm| > 1`, 0 otherwise, NULL without VLM |
| `vehicles_yolo` | INTEGER | | Counted boxes whose label has category `vehicle` |
| `lighting` | TEXT | | VLM: daylight, artificial, dim, dark |
| `door_state` | TEXT | | |
| `tidiness` | TEXT | | |
| `weather_visible` | TEXT | | For outdoor or parking views |
| `summary` | TEXT | | One-line VLM description |
| `attrs_json` | TEXT | | Scene fields not in `SCENE_COLUMNS` |

Writer: `results.insert_scene_state`. Readers: `results.previous_people`, `baseline.py`, `forecast/table.py`, `dashboard/data.py`.

##### `events` (001)

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `event_id` | INTEGER | PRIMARY KEY AUTOINCREMENT | |
| `space_id` | TEXT | NOT NULL | |
| `ts_utc` | TEXT | NOT NULL | |
| `frame_id` | INTEGER | REFERENCES `frames(frame_id)` (no cascade) | |
| `event_type` | TEXT | NOT NULL | SQL comment lists new_object_type, object_appeared, object_gone, became_empty, became_occupied. Code writes `new_object_type`, `became_empty`, `became_occupied` |
| `object_type` | TEXT | | Set for `new_object_type` |
| `cell_id` | TEXT | | Not written by current code |
| `resolution` | TEXT | NOT NULL DEFAULT `'interval'` | Stills: inferred between frames, not observed |
| `details_json` | TEXT | | `{"raw_label": ...}` or `{"prev": n, "now": n}` |

Writers: `registry.add_auto_new_object_type`, `results.insert_event`. Deleted by `clear_frame_results` (except `new_object_type`).

`insert_event(conn, *, space_id, ts_utc, frame_id, event_type, details=None)`.

##### `object_mobility` (001)

Measured mobility: fraction of appearances in the same cell as the previous appearance.

| Column | Type | Constraint |
|---|---|---|
| `space_id` | TEXT | NOT NULL |
| `object_type` | TEXT | NOT NULL |
| `n_frames` | INTEGER | NOT NULL |
| `stable_fraction` | REAL | |
| `measured_class` | TEXT | fixed, semi_static, dynamic |
| `updated_utc` | TEXT | NOT NULL |
| | | PRIMARY KEY `(space_id, object_type)` |

No Python code writes this table; it is only joined in `v_observations_wide`.

##### `baseline_cell` (001)

Per (space, object type, cell, weekday, hour). Frames with `capture_status != 'ok'` are excluded, never counted as zero.

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `space_id` | TEXT | NOT NULL | |
| `object_type` | TEXT | NOT NULL | |
| `cell_id` | TEXT | NOT NULL | |
| `dow` | INTEGER | NOT NULL | |
| `hour` | INTEGER | NOT NULL | |
| `n_frames` | INTEGER | NOT NULL | Valid frames in the bucket |
| `mean_count` | REAL | | |
| `std_count` | REAL | | |
| `p50_count` | REAL | | |
| `p90_count` | REAL | | |
| `presence_rate` | REAL | | Share of valid frames with count > 0 |
| `ewm_count` | REAL | | Exponentially weighted mean |
| `updated_utc` | TEXT | NOT NULL | |
| | | PRIMARY KEY `(space_id, object_type, cell_id, dow, hour)` | |

Writer and reader: `baseline.py` (section 2.15).

##### `baseline_space` (001)

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `space_id` | TEXT | NOT NULL | |
| `metric` | TEXT | NOT NULL | SQL comment lists people_total, vehicles_total, brightness, occupied_cells; code writes `people_yolo`, `people_vlm`, `vehicles_yolo`, `brightness` |
| `dow` | INTEGER | NOT NULL | |
| `hour` | INTEGER | NOT NULL | |
| `n_frames` | INTEGER | NOT NULL | Non-NULL values used |
| `mean_value`, `std_value`, `p50_value`, `p90_value`, `ewm_value` | REAL | | |
| `updated_utc` | TEXT | NOT NULL | |
| | | PRIMARY KEY `(space_id, metric, dow, hour)` | |

##### `label_frames` (002)

Human labels, made blind (model output hidden), counts only for the space's `label_types`.

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `frame_id` | INTEGER | PRIMARY KEY, REFERENCES `frames` ON DELETE CASCADE | |
| `usable` | INTEGER | NOT NULL DEFAULT 1 | 0 = frame too unclear to label |
| `labeler` | TEXT | | |
| `labeled_utc` | TEXT | NOT NULL | |
| `note` | TEXT | | |

##### `labels` (002)

| Column | Type | Constraint |
|---|---|---|
| `frame_id` | INTEGER | NOT NULL, REFERENCES `frames` ON DELETE CASCADE |
| `object_type` | TEXT | NOT NULL |
| `true_count` | INTEGER | NOT NULL |
| | | PRIMARY KEY `(frame_id, object_type)` |

Both label tables are written by `labeling/core.py` and read by `eval/vlm_eval.py` and `eval/thresholds.py`.

##### `object_links` (003)

Links between object names. Every decision is kept for audit and can be reverted.

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `link_id` | INTEGER | PRIMARY KEY AUTOINCREMENT | |
| `child` | TEXT | NOT NULL | |
| `parent` | TEXT | NOT NULL | |
| `relation` | TEXT | NOT NULL, CHECK in (`same`, `is_a`) | `same` = child merged into parent; `is_a` = child is a kind of parent |
| `status` | TEXT | NOT NULL DEFAULT `'proposed'`, CHECK in (`proposed`, `approved`, `rejected`) | |
| `source` | TEXT | NOT NULL | curator, human, config |
| `similarity` | REAL | | Embedding cosine that shortlisted the pair |
| `evidence_json` | TEXT | | The two yes/no answers and the names |
| `model` | TEXT | | Judge model |
| `created_utc` | TEXT | NOT NULL | |
| `decided_utc` | TEXT | | |
| | | UNIQUE `(child, parent, relation)` | |

Writers: `curator._record`, `set_link_status`. Readers: `curate`, `load_taxonomy`, `cli.py links`.

##### `label_judgments` (003)

Cache of "is every A a B?" answers so the curator never asks the same model the same question twice.

| Column | Type | Constraint |
|---|---|---|
| `a` | TEXT | NOT NULL |
| `b` | TEXT | NOT NULL |
| `model` | TEXT | NOT NULL |
| `answer` | INTEGER | NOT NULL (1 = yes, 0 = no) |
| `created_utc` | TEXT | NOT NULL |
| | | PRIMARY KEY `(a, b, model)` |

##### `relations` (003)

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `rel_id` | INTEGER | PRIMARY KEY AUTOINCREMENT | |
| `frame_id` | INTEGER | NOT NULL, REFERENCES `frames` ON DELETE CASCADE | |
| `space_id` | TEXT | NOT NULL | |
| `ts_utc` | TEXT | NOT NULL | |
| `subject` | TEXT | NOT NULL | Canonical |
| `predicate` | TEXT | NOT NULL | VLM predicate or `overlaps` |
| `object` | TEXT | NOT NULL | Canonical |
| `raw_subject` | TEXT | | |
| `raw_object` | TEXT | | |
| `count` | INTEGER | NOT NULL DEFAULT 1 | |
| `source` | TEXT | NOT NULL | `vlm` (read from the photo) or `geometry` (from detector boxes) |

Index: `ix_rel_frame (frame_id)`. Writer: `results.insert_relation` (raw subject and object default to the canonical ones); updated by `reresolve_observations`. Reader: `forecast/table.py`.

##### `frame_features` (003)

Generic numeric features per frame from pluggable extractors.

| Column | Type | Constraint |
|---|---|---|
| `frame_id` | INTEGER | NOT NULL, REFERENCES `frames` ON DELETE CASCADE |
| `name` | TEXT | NOT NULL |
| `value` | REAL | |
| `extractor` | TEXT | NOT NULL |
| `version` | TEXT | NOT NULL |
| | | PRIMARY KEY `(frame_id, name)` |

`forecast/table.py` reads it into `feat.<name>` columns. No code in `src/sbi` writes it.

##### `detections` (004, 005)

Every detector box down to the storage floor. `observations` keeps only counted boxes; this is the raw layer, so thresholds can be chosen per object type later and counts at any threshold can be derived without re-running the model.

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `det_id` | INTEGER | PRIMARY KEY AUTOINCREMENT | |
| `frame_id` | INTEGER | NOT NULL, REFERENCES `frames` ON DELETE CASCADE | |
| `space_id` | TEXT | NOT NULL, REFERENCES `spaces` | |
| `ts_utc` | TEXT | NOT NULL | |
| `raw_label` | TEXT | NOT NULL | Label after `detector_map`; taxonomy applied on read |
| `confidence` | REAL | NOT NULL | Rounded to 3 decimals |
| `bbox_norm` | TEXT | NOT NULL | JSON `[x1, y1, x2, y2]` |
| `cell_id` | TEXT | | From the anchor rule |
| `counted` | INTEGER | NOT NULL | 1 = counted (also an observation) |
| `mask` | BLOB | (005) | `encode_mask` output; NULL = not segmented |
| `mask_area` | REAL | (005) | Share of the image covered |

Indexes: `ix_det_frame (frame_id)`, `ix_det_space_label (space_id, raw_label, ts_utc)`. Writer: `results.insert_detection`. Readers: `forecast/table.py`, `eval/thresholds.py`.

##### `frame_embeddings` (006)

One row per `(frame, model)`. A different model or input size is a different vector space, so it gets its own rows. No row = never embedded (derived columns are NaN, not zero).

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `frame_id` | INTEGER | NOT NULL, REFERENCES `frames` ON DELETE CASCADE | |
| `model` | TEXT | NOT NULL | Embedder version, for example `dinov2-base@448` |
| `dim` | INTEGER | NOT NULL | |
| `vec` | BLOB | NOT NULL | float16 little-endian |
| | | PRIMARY KEY `(frame_id, model)` | |

`store/embeddings.py`:

| Function | Behaviour |
|---|---|
| `encode_vector(vec)` | Requires a non-empty 1-D vector of finite numbers (else `ValueError`); returns `astype("<f2").tobytes()`. Module docstring: the float16 rounding error (about 1e-3 relative) is far below the scene-to-scene differences these vectors are used for. |
| `decode_vector(blob, dim)` | `ValueError` unless `len(blob) == 2 * dim`; returns float32 |
| `put_embedding(conn, frame_id, model, vec)` | `INSERT OR REPLACE`; other models' rows are left alone |
| `load_embeddings(conn, space_id, model, frame_ids=None)` | Returns `({frame_id: vector}, [corrupt frame ids])`. Corrupt rows are left out. Mixed dimensions within one model -> `StorageError`. |
| `frames_missing_embedding(conn, space_id, model)` | Frames with `capture_status='ok'`, `image_path` not NULL and no row for this model, ordered by `ts_utc` |

Readers: `forecast/table.py` (through `load_embeddings`), `dashboard/data.py`.

##### `frame_scores` (007)

One raw score per `(frame, scorer, prompt)`. The prompt text is the key, so what was asked is on record and a new prompt can be asked of every stored photo later. Scorer version is part of the key because different models are different scales. No row = never scored.

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `frame_id` | INTEGER | NOT NULL, REFERENCES `frames` ON DELETE CASCADE | |
| `scorer` | TEXT | NOT NULL | Scorer version |
| `prompt` | TEXT | NOT NULL | |
| `value` | REAL | NOT NULL | Raw score (SigLIP logit; 0 = even odds) |
| | | PRIMARY KEY `(frame_id, scorer, prompt)` | |

`store/scores.py`:

| Function | Behaviour |
|---|---|
| `put_scores(conn, frame_id, scorer, scores)` | `ValueError` if any value is not finite; otherwise `INSERT OR REPLACE` per prompt. Other prompts and scorers are untouched. |
| `load_scores(conn, space_id, scorer, frame_ids=None)` | `{frame_id: {prompt: value}}` |
| `frames_missing_scores(conn, space_id, scorer, prompts)` | For each frame with `capture_status='ok'` and `image_path` not NULL, oldest first, the list of `prompts` it has no score for; frames missing none are left out |

Reader also: `dashboard/data.py`.

##### `forecast_runs` (008)

One run = one forecaster call for one space at one origin (the last observed slot).

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `run_id` | INTEGER | PRIMARY KEY AUTOINCREMENT | |
| `space_id` | TEXT | NOT NULL, REFERENCES `spaces` | |
| `created_utc` | TEXT | NOT NULL | |
| `origin_local` | TEXT | NOT NULL | Last observed slot, naive local time (IST) |
| `model` | TEXT | NOT NULL | For example `chronos2:amazon/chronos-2` |
| `variant` | TEXT | NOT NULL | Covariate-set name |
| `horizon` | INTEGER | NOT NULL | Slots ahead |
| `step_minutes` | INTEGER | NOT NULL | |
| `context_slots` | INTEGER | NOT NULL | Rows of history given to the model |
| `targets_json` | TEXT | NOT NULL | |
| `covariates_json` | TEXT | NOT NULL | Past-only covariates; calendar columns listed too when used |
| `quantiles_json` | TEXT | NOT NULL | |

Index: `ix_fruns_space (space_id, origin_local)`.

##### `forecasts` (008)

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `run_id` | INTEGER | NOT NULL, REFERENCES `forecast_runs` ON DELETE CASCADE | |
| `target` | TEXT | NOT NULL | Table column name |
| `ts_local` | TEXT | NOT NULL | Slot forecast, naive local time |
| `step` | INTEGER | NOT NULL | 1 = slot right after the origin |
| `quantile` | REAL | NOT NULL | Quantiles are rows, so the set can change |
| `value` | REAL | NOT NULL | |
| | | PRIMARY KEY `(run_id, target, ts_local, quantile)` | |

`store/forecasts.py` (writers assume a transaction):

| Function | Behaviour |
|---|---|
| `insert_run(conn, *, space_id, origin_local, model, variant, horizon, step_minutes, context_slots, targets, covariates, quantiles, created=None)` | Inserts a run; lists stored as JSON; `created` defaults to now (UTC). Returns `run_id`. |
| `insert_forecasts(conn, run_id, forecast)` | Input: long frame with `timestamp`, `target`, `step` and quantile columns named `q<NN>`. Quantile columns are detected as names starting with `q` followed by digits (`q10` -> 0.10). None found -> `StorageError`. NaN values are skipped. Returns rows stored. |
| `load_forecast(conn, run_id)` | Wide frame `target, timestamp, step, q10, q50, ...`, sorted by target and step; empty run -> empty frame with `target, timestamp, step` |
| `latest_run(conn, space_id)` | Row with the highest `run_id` for the space, or `None` |

Readers: `forecast/accuracy.py`, `dashboard/data.py`.

#### 2.13.6 Views (001)

| View | Content |
|---|---|
| `v_observations_wide` | Observations joined to frame time dimensions, space name and type, registry category and mobility prior, and `object_mobility.measured_class` (LEFT JOIN), with cell, bbox, status, activity, count, confidence, source, attrs |
| `v_cell_counts` | Per frame, object type and cell: `SUM(count)` as `n`, for frames with `capture_status='ok'` and non-NULL `cell_id` |
| `v_frame_counts` | Per frame and object type: `count_yolo` and `count_vlm` side by side (never merged), for `capture_status='ok'` frames |

No Python module in `src/sbi` queries these views.

#### 2.13.7 Who writes what for one processed frame

| Table | Rows per processed frame |
|---|---|
| `detections` | One per stored box (counted or not) |
| `observations` | One per counted box (`yolo`); with a reading, one person row plus one per object kind (`vlm`) |
| `relations` | One per overlapping label pair (`geometry`); with a reading, one per valid VLM relation (`vlm`) |
| `scene_state` | Exactly one |
| `events` | 0 or 1 occupancy event; one `new_object_type` per new registry name |
| `object_registry` | One `auto_new` row per new name |
| `frames` | Processing columns updated |
| `frame_embeddings` | 0 or 1 (replaced) |
| `frame_scores` | 0 or one per probe prompt (replaced) |

### 2.14 Catch-up backfills (`embedding.py`, `scoring.py`)

Both share the storage function of the live pipeline, so both paths write identical rows.

#### 2.14.1 `embedding.py::embed_missing(conn, space_id, embedder, *, root, limit=None) -> EmbedReport`

- "Missing" = `frames_missing_embedding(conn, space_id, embedder.version)`: frames of the space with `capture_status='ok'`, a non-NULL `image_path`, and no `frame_embeddings` row for this embedder version, oldest first. A frame embedded by a different model or width counts as missing.
- Loop: stop when `limit` is set and `done >= limit`. Read the image; unreadable -> warning, `failed += 1`, continue. Embed; a shape other than `(dim,)` or an `EmbedderError` -> warning, `failed += 1`, continue. Otherwise `put_embedding` in its own transaction, `done += 1`.
- `limit` counts frames actually embedded, not frames attempted (module docstring: so a frame that keeps failing cannot use up every limited run). Nothing is written for a failed frame.
- `EmbedReport(done, failed)`. CLI `sbi embed --space ... [--limit]` returns exit code 1 if any frame failed, and refuses to run when `embedder.kind` is `none`.

#### 2.14.2 `scoring.py::score_missing(conn, space, scorer, *, root, limit=None) -> ScoreReport`

- "Missing" = `frames_missing_scores(conn, space_id, scorer.version, space.probe_prompts)`: for each usable frame, the current probe prompts that have no `(frame, scorer version, prompt)` row. Two uses: scoring history after the scorer is switched on, and asking a new prompt of all past photos.
- Only the missing prompts are sent to the scorer for each frame.
- Same loop rules as `embed_missing`: `limit` counts frames actually scored; unreadable image or `ScorerError` (including a wrong number of scores) -> `failed += 1`, skip; success -> `put_scores` in its own transaction.
- `ScoreReport(done, failed)`.

### 2.15 Baselines (`baseline.py`)

Module docstring rules: only processed frames with `capture_status='ok'` count; `n_frames` is always stored; a cell with no detection in a valid frame counts as 0; camera_down, bad_frame and unprocessed frames are excluded entirely.

Settings: `baseline.ewm_alpha: 0.3` (code default 0.3, 0 < v ≤ 1).

| Function | Behaviour |
|---|---|
| `_quantile(sorted_vals, p)` | Linear interpolation between the two nearest ranks: index `(n-1) * p` |
| `summarise(vals, alpha)` | `mean` (`statistics.fmean`), `std` (population std `pstdev`, 0.0 for one value), `p50`, `p90`, and `ewm`: starts at the first value, then `ewm = alpha * v + (1 - alpha) * ewm` for each later value. Values must be in time order. |
| `rebuild(conn, space_id, *, alpha=0.3, dow=None, hour=None)` | Recomputes baselines for a space, or for one `(dow, hour)` bucket when both are given |

`rebuild` in order:

1. Selects frames joined to `scene_state` (so only processed frames), `capture_status='ok'`, the space, and the bucket filter if given, ordered by `ts_utc`. Columns: `frame_id, dow, hour, brightness, people_yolo, people_vlm, vehicles_yolo`.
2. Groups them into `(dow, hour)` buckets.
3. In one transaction: deletes the existing `baseline_cell` and `baseline_space` rows for the space (or only for the bucket).
4. For each bucket:
   - Reads `observations` with `source='yolo'` and non-NULL `cell_id` for the bucket's frames, summed per `(frame_id, object_type, cell_id)`.
   - For each `(object_type, cell_id)` seen at least once in the bucket: the value list has one entry per frame of the bucket, 0 for frames without that type in that cell. Inserts a `baseline_cell` row with `n_frames = number of frames`, mean, std, p50, p90, `presence_rate = share of values > 0`, ewm. Pairs never seen in the bucket get no row.
   - For each metric `people_yolo`, `people_vlm`, `vehicles_yolo`, `brightness`: drops NULL values (for example `people_vlm` when the VLM failed), and if any remain inserts a `baseline_space` row with `n_frames = number of non-NULL values`.

Callers: the tick calls `rebuild` for the bucket of the tick time (`time_dims(now, space.tz)`) after it processed at least one frame of the space; `sbi baseline` rebuilds all buckets for one space or all spaces.

## 3. The table and forecasting

This section covers everything between the stored readings in SQLite and a stored, scored forecast: the per-space
table, the per-photo column cache, the column catalog, the site calendar, the choice of targets and inputs, the
forecasters, the metrics, the backtest, live (stored) forecasts, the scheduled forecast and live accuracy.

### 3.1 Module map

| File | Purpose |
|---|---|
| `src/sbi/forecast/__init__.py` | Package marker. Docstring: "the regular, model-agnostic table that time-series models read". |
| `src/sbi/forecast/build.py` | `table_for`: the single entry point that builds a space's table with every setting applied (CLI, scheduled tick, dashboard). |
| `src/sbi/forecast/table.py` | `build_table` (grid, frame picking, every column family), `frame_columns` (per-photo columns), the catalog, `select_covariates`, `all_covariates`, `calendar`, `to_chronos`. |
| `src/sbi/forecast/feature_cache.py` | `FeatureCache`: a separate SQLite file that keeps each photo's derived columns between builds. |
| `src/sbi/forecast/mask_features.py` | `mask_features`: counts, areas and per-cell areas from stored outlines (masks) of one photo. |
| `src/sbi/forecast/embed_features.py` | `embedding_features`: principal components and drift measures from whole-scene image vectors. |
| `src/sbi/forecast/probe_features.py` | `concept_values`: one number per probe concept from the stored prompt scores of one photo. |
| `src/sbi/forecast/profile_features.py` | `day_profile`, `add_profiles`: the day-type profile (`prof.<target>`), a known-in-advance input. |
| `src/sbi/forecast/inputs.py` | `resolve_targets`, `prepare`, `predict`: choosing targets and inputs, running a forecaster, clipping at 0. Shared by live runs and the backtest. |
| `src/sbi/forecast/models.py` | `Forecaster` protocol, three baselines, `Chronos2Forecaster`, `build_forecaster`, `qname`. |
| `src/sbi/forecast/metrics.py` | `evaluate`, `pinball`, `summarise`, `overall`: scoring forecasts against measured values. |
| `src/sbi/forecast/backtest.py` | Rolling-origin backtest: `BacktestPlan`, `origins_for`, `run_backtest`. |
| `src/sbi/forecast/runner.py` | `forecast_space`: one live forecast, stored in `forecast_runs` / `forecasts`. |
| `src/sbi/forecast/schedule.py` | `due`, `run_scheduled`: forecasts made by the 15-minute tick. |
| `src/sbi/forecast/accuracy.py` | Live accuracy of stored forecasts: `live_points`, `outdated_runs`, `by_target`, `pooled_skill`, `by_lead`, `by_weekday`. |
| `src/sbi/sitecal.py` | `CalendarSpec`, `load_calendar`, `read_site_file`, `country_days`: holidays and working days. |
| `src/sbi/store/forecasts.py` | `insert_run`, `insert_forecasts`, `load_forecast`, `latest_run`: storage of runs and predictions. |

Conventions used throughout:

- Timestamps in the table are naive local time (IST for both spaces), one row per `table.step_minutes` (15) slot.
- A column name is `<kind>.<source>.<name...>` (for example `count.yolo.chair`, `cell.yolo.R3C4.chair`,
  `rel.vlm.person.sitting_on.chair`). Some families have only two parts (`cal.hour`, `scene.people_yolo`,
  `img.brightness`, `cat.lighting`, `feat.<name>`).
- `kind` is the first part. `source` is the second part when the name has at least three parts, else `""`.

### 3.2 Where a table comes from: `src/sbi/forecast/build.py::table_for`

`table_for(conn, settings, space, *, start=None, end=None, fit_end=None, max_slots=None, use_cache=True) -> TableResult`

The one place the CLI, the scheduled tick and the dashboard get a table from, "so they always agree". It:

1. Creates `FeatureCache(settings.cache_dir / "table_features.db")` when `use_cache` is True, else passes no cache.
   `settings.cache_dir` resolves `paths.cache` (`data/cache` in `config/pipeline.yaml`) against the project root.
2. Loads the site calendar with `load_calendar(settings.calendar, settings.root, range(this_year - 3, this_year + 2))`,
   where `this_year = date.today().year`. The year range is therefore `this_year - 3` through `this_year + 1`.
3. Calls `build_table` with:

| `build_table` argument | Value passed by `table_for` |
|---|---|
| `space_id`, `tz` | `space.space_id`, `space.tz` |
| `step_minutes` | `settings.table.step_minutes` |
| `start`, `end` | passed through |
| `min_coverage`, `min_presence`, `min_valid_slots`, `min_present_slots` | `settings.table.*` |
| `exclude_labels`, `conf_levels` | `tuple(settings.table.exclude_labels)`, `tuple(settings.table.conf_levels)` |
| `mask_dup_iou`, `cell_area_min` | `settings.table.*` |
| `grid` | `(space.grid.cols, space.grid.rows)` (admin_cabin 8 x 6, parking 12 x 8) |
| `embed_model` | `settings.embedder.version` = `f"{Path(embedder.model).name}@{embedder.width}"` (for example `dinov2-base@448`) |
| `embed_components` | `settings.table.embed_components` |
| `embed_fit_end` | the `fit_end` argument |
| `probe_scorer` | `settings.scorer.version` = `Path(scorer.model).name` (for example `siglip2-base-patch16-224`) |
| `probe_groups` | `space.probes` (concept -> list of prompts) |
| `calendar_spec` | the loaded `CalendarSpec` |
| `cache`, `max_slots` | as above / passed through |

Callers and the `max_slots` they pass:

| Caller | `max_slots` | `fit_end` |
|---|---|---|
| `sbi forecast` (`src/sbi/cli.py`) | `forecast.max_context` | none |
| `run_scheduled` (scheduled tick) | `forecast.max_context` | none |
| Dashboard (`src/sbi/dashboard/app.py`) | `forecast.max_context` | none |
| `sbi backtest` | none (whole history) | first call none; second call the first origin's timestamp |
| `sbi accuracy` | none (whole history) | none |
| `sbi table` | none | `--fit-end` option |

The table is identical with and without the cache; the cache only changes build time.

### 3.3 `src/sbi/forecast/table.py::build_table`, step by step

Signature and defaults:

```
build_table(conn, space_id, *, tz, step_minutes=15, start=None, end=None, min_coverage=0.5,
            min_presence=0.05, min_valid_slots=8, min_present_slots=3, exclude_labels=(),
            conf_levels=(), mask_dup_iou=0.5, grid=None, cell_area_min=0.02, embed_model=None,
            embed_components=4, embed_fit_end=None, probe_scorer=None, probe_groups=None,
            calendar_spec=CalendarSpec(), cache=None, max_slots=None) -> TableResult
```

The module docstring states the rules the builder follows:

- Regular grid with no gaps. A slot without a usable frame is a row of NaN, never a missing row and never 0, because
  Chronos-2 masks NaN but needs regular timestamps.
- One frame per slot: the usable one closest to the slot start. Manual test runs and retries are dropped.
- Names come from raw labels through the current taxonomy, so merging synonyms re-maps all history.
- In a usable frame, an object type the source did not report is 0 (seen and absent). If the VLM failed for a frame,
  every VLM-derived column is NaN for that slot, while detector columns are still filled.
- Every dimension is kept. The catalog marks which ones are model-ready and says why the others are not.

#### 3.3.1 Step 1: load the taxonomy

`tax = load_taxonomy(conn)` (`src/sbi/taxonomy.py`). It carries `known` (registry object types), `alias` (spaced alias ->
object type), `merged_into`, and `parents` (approved `is_a` links, both ends resolved through merges). The table uses:

- `src/sbi/forecast/table.py::_name(tax, raw, fallback)`: `tax.lookup(raw)` when `raw` is non-empty, else `None`;
  falls back to `fallback` when the lookup gives nothing. `lookup` returns the canonical name of a raw label or `None`
  for an unknown label.
- `tax.ancestors(name)`: all transitive `is_a` ancestors, excluding the name itself, cycle-safe.

#### 3.3.2 Step 2: pick one frame per slot: `src/sbi/forecast/table.py::_pick_frames`

`_pick_frames(conn, space_id, tz, step) -> DataFrame` indexed by slot.

1. Reads `frame_id, ts_utc, capture_status, processed_utc, vlm_raw, brightness, contrast, sharpness,
   detector_min_conf, segmenter_version` from `frames` for the space. No rows: raises
   `SbiError("no frames stored for space ...")`.
2. `local` = `ts_utc` parsed as UTC, converted to `tz`, made naive.
3. `slot = floor(local, step minutes)`; `offset = (local - slot)` in seconds (always >= 0).
4. `usable = capture_status == "ok" AND processed_utc IS NOT NULL`.
5. `status`:

| Condition | `status` |
|---|---|
| `usable` | `ok` |
| `capture_status == "ok"` but not processed | `unprocessed` |
| otherwise | the stored `capture_status` value itself |

6. `vlm_ok = usable AND vlm_raw IS NOT NULL AND NOT vlm_raw.startswith('{"error"')`.
7. `rank = 0` if usable else `1`.
8. Sort by `(slot, rank, offset)` and keep the first row per slot. So a slot gets its usable frame with the smallest
   offset from the slot start; if it has no usable frame, it gets the unusable frame with the smallest offset (its
   status then explains why the slot is empty).

#### 3.3.3 Step 3: the slot grid and `max_slots`

```
index = pd.date_range(Timestamp(start) if start else frames.index.min(),
                      Timestamp(end)   if end   else frames.index.max(),
                      freq=f"{step_minutes}min", name="timestamp")
if max_slots is not None and len(index) > max_slots:
    index = index[-max_slots:]
```

- Without `start`/`end` the grid runs from the first frame's slot to the newest frame's slot (whatever that frame's
  status), with no gaps.
- `start` and `end` are used as given; they are not floored to the step. A value off the slot boundary gives a grid
  that does not line up with the slots.
- `max_slots` keeps only the newest `max_slots` rows, after `start`/`end`. Its docstring: "for callers that cannot use
  more history than that". Live forecasts and the dashboard pass `forecast.max_context` (8192), which is the most
  Chronos-2 reads (`ForecastSettings.max_context` comment: about 85 days of 15-minute slots).
- Frames outside the grid are ignored (they are simply not re-indexed in).

#### 3.3.4 Step 4: META columns

`META = ["slot_status", "frame_id", "vlm_ok"]`. They are the first three columns and are never in the catalog.

| Column | Value |
|---|---|
| `slot_status` | the picked frame's `status` (`ok`, `unprocessed`, or the raw `capture_status`); `no_frame` when the slot has no frame at all |
| `frame_id` | the picked frame's id (also set for unusable frames); NaN for `no_frame` |
| `vlm_ok` | the picked frame's `vlm_ok` as bool; `False` for `no_frame` |

Two boolean masks over the grid drive every NaN rule below:

- `valid = slot_status == "ok"`
- `vlm_ok = valid & t["vlm_ok"]`

#### 3.3.5 Step 5: calendar columns: `src/sbi/forecast/table.py::calendar`

`calendar(index, spec=CalendarSpec()) -> DataFrame`. All seven columns are known in advance. They are computed for
every row whatever its status, and the same function builds the future frame's calendar in `to_chronos`.

| Column | Formula |
|---|---|
| `cal.hour` | `hour` (0..23) |
| `cal.minute_of_day` | `hour * 60 + minute` |
| `cal.dow` | `dayofweek` (Monday = 0 .. Sunday = 6) |
| `cal.is_weekend` | `1 if dayofweek >= 5 else 0` (fixed Saturday/Sunday; does not use `working_days`) |
| `cal.slot_of_day` | `(hour * 60 + minute) // 15` (the divisor is the constant 15, not `step_minutes`) |
| `cal.is_holiday` | `1 if date in spec.holidays else 0` |
| `cal.is_working_day` | `1 if dayofweek in spec.working_days and not is_holiday else 0` |

See 3.7 for where `spec.holidays` and `spec.working_days` come from.

#### 3.3.6 Step 6: "did the source look?" masks per source token

These masks decide where a derived column is 0 (looked, saw nothing) and where it is NaN (did not look).

```
floor     = detector_min_conf of the picked frame (numeric, NaN when missing)
level_ok  = { "yolo_cNN": valid & (floor <= level + 1e-9)   for each level in conf_levels }
outlined  = valid & (segmenter_version IS NOT NULL)
mask_sources = {"sam2": None, "sam2_cNN": level for each level}     # passed to mask_features
source_ok = { **level_ok,
              "sam2": outlined,
              "sam2_cNN": outlined & level_ok["yolo_cNN"]  for each level }
```

- `detector_min_conf` records how low the detector's stored boxes go for that frame. A frame processed before
  low-confidence boxes were stored has NaN there, so `floor <= level` is False and all `yolo_cNN` and `sam2_cNN`
  columns are NaN for it (unknown, not zero), as the module docstring requires.
- `src/sbi/forecast/table.py::level_source(level, base="yolo")` builds the token: `f"{base}_c{round(level*100):02d}"`,
  so 0.05 -> `yolo_c05`, 0.10 -> `yolo_c10`, and with `base="sam2"` -> `sam2_c05`.

#### 3.3.7 Step 7: per-photo columns: `src/sbi/forecast/table.py::frame_columns`

```
frame_columns(conn, tax, frame_ids, *, conf_levels, mask_sources, mask_dup_iou, grid, cell_area_min)
    -> {frame_id: {column: value}}
```

Only the picked frames of `valid` slots are passed. `fid_slot` maps each such frame id to its slot, and `stamps`
maps it to `str(processed_utc)` (used by the cache, 3.4). The call goes through
`_frame_columns_cached` (3.4); without a cache it calls `frame_columns` directly.

`frame_columns` is a pure function of the stored rows, the vocabulary links and the arguments. Each frame starts as a
`defaultdict(float)`, so a column appears only when something contributed to it; a photo with nothing stored gets an
empty dict. Frame ids are processed in batches of `_SQL_BATCH = 500` per SQL statement (SQLite limits bound
variables). Within a batch, four reads happen in this order.

**(a) `observations` rows** (`frame_id, source, object_type, raw_label, cell_id, count`). For each row, with
`name = _name(tax, raw_label, object_type)`:

| Column | Update |
|---|---|
| `count.<source>.<name>` | `+= count` |
| `rollup.<source>.<anc>` for every `anc` in `tax.ancestors(name) ∪ {name}` | `+= count` |
| `cell.yolo.<cell_id>.<name>` | `+= count`, only when `source == "yolo"` and `cell_id` is set |

So `rollup.<src>.<P>` = count of `P` itself plus the counts of every name that has `P` as an `is_a` ancestor (the
code comment: "roll-up = the name itself + all is_a descendants"). `<source>` is the observation's source token
(`yolo` for counted detector boxes, `vlm` for the vision-language model). Cell ids are `R<row>C<col>`
(`src/sbi/grid.py::cell_id`).

**(b) counted detector boxes for `area.yolo`**: `observations` rows with `source='yolo' AND bbox_norm IS NOT NULL`.
Each row's normalised box `[x1, y1, x2, y2]` (JSON) is appended to the list keyed `("yolo", name, frame_id)`.

**(c) `detections` rows** (every stored detector box, counted or not: `frame_id, raw_label, confidence, counted,
bbox_norm, mask`). With `name = _name(tax, raw_label, raw_label)`:

- For each level `lv` in `conf_levels` with `confidence >= lv`:
  `count.yolo_cNN.<name> += 1` and the box is appended to the list keyed `("yolo_cNN", name, frame_id)`.
- If `mask` is not NULL, `(name, confidence, bool(counted), decode_mask(mask))` is added to the frame's outline list.
  `src/sbi/perception/masks.py::decode_mask` raises `ValueError("corrupt mask blob: ...")` on a corrupt blob; this is
  not caught in `frame_columns`, so it propagates out of `build_table`.

**(d) outline columns**: for each frame with outlines, `mask_features(outlines, sources=mask_sources,
min_iou=mask_dup_iou, grid=grid, cell_min=cell_area_min)` (3.3.8); every returned value is added to the frame's dict.

**(e) box-area columns**: for each `(src, name, frame)` box list:
`area.<src>.<name> += round(union_area(boxes), 5)`, where `src` is `yolo` or `yolo_cNN`.
`src/sbi/perception/geometry.py::union_area(boxes)` is the exact area covered by a set of normalised boxes, overlaps
counted once (the share of the image covered). Degenerate boxes (`x2 <= x1` or `y2 <= y1`) are dropped; an empty list
gives 0. It splits the plane on all distinct box edges and adds each elementary rectangle whose centre lies in any box.

**(f) `relations` rows** (`frame_id, source, subject, predicate, object, raw_subject, raw_object, count`):
`rel.<source>.<_name(tax, raw_subject, subject)>.<predicate>.<_name(tax, raw_object, object)> += count`.
The table docstring names the sources `vlm` and `geometry`.

Summary of the column families produced per photo:

| Family | Source tokens | Value |
|---|---|---|
| `count.<src>.<type>` | `yolo`, `vlm` (observations) | sum of `count` |
| `count.yolo_cNN.<type>` | one per conf level | number of stored boxes with `confidence >= 0.NN` |
| `count.sam2.<type>`, `count.sam2_cNN.<type>` | from outlines | boxes left after outline de-duplication (3.3.8) |
| `rollup.<src>.<parent>` | `yolo`, `vlm` | the parent's count plus all `is_a` descendants' counts |
| `cell.yolo.<cell>.<type>` | `yolo` only | sum of `count` in that grid cell |
| `area.yolo.<type>` | counted boxes | `round(union_area(boxes), 5)` |
| `area.yolo_cNN.<type>` | all boxes `>= 0.NN` | `round(union_area(boxes), 5)` |
| `area.sam2.<type>`, `area.sam2_cNN.<type>` | from outlines | share of the image covered by the union of outlines |
| `cellarea.sam2.<cell>.<type>` | counted boxes' outlines only | share of the cell covered |
| `rel.<src>.<subject>.<predicate>.<object>` | `vlm`, `geometry` | sum of `count` |

#### 3.3.8 Outline columns: `src/sbi/forecast/mask_features.py::mask_features`

```
mask_features(boxes, *, sources, min_iou, grid=None, cell_min=0.0, cell_sources=("sam2",)) -> {column: value}
```

`boxes` is a list of `(object type, confidence, counted, outline)` for one frame. `sources` maps a source token to a
confidence level; `None` means "the counted boxes". The table passes
`{"sam2": None, "sam2_c05": 0.05, "sam2_c10": 0.10}` (with the configured levels).

1. `mask_extent` of every outline is measured once and reused by every source:
   `(pixels set, first row, row end, first column, column end)`, all zero for an empty outline.
2. For each source, a box is selected when `counted` (level `None`) or when `confidence >= level`. Selected boxes are
   grouped by type.
3. Per type:
   - `count.<src>.<type> = len(drop_duplicate_masks(items, min_iou, extents))`.
     `src/sbi/perception/masks.py::drop_duplicate_masks` walks the boxes from highest to lowest confidence and drops a
     box whose outline has IoU `>= min_iou` with an already kept one ("the same object boxed twice: the stronger one is
     kept"). Empty outlines cannot be compared and are kept. IoU = `|A ∩ B| / |A ∪ B|` in pixels; the helper
     `_iou_at_least` gives the same answer as the full IoU with shortcuts (disjoint bounding boxes share nothing; IoU
     cannot exceed smaller area / larger area).
   - `area.<src>.<type> = round(mean(union of all selected outlines of that type), 5)`: the share of outline pixels
     set. The union uses all selected outlines, before de-duplication (de-duplication only changes the count).
   - Only when `grid` is given and `src` is in `cell_sources` (`("sam2",)`, the counted boxes): `shares =
     cell_shares(union, cols, rows)`, the share of each grid cell covered (cell edges match `grid.point_to_cell`:
     pixel row `r` belongs to cell row `r * rows // h`, likewise columns). For every cell with
     `share >= max(cell_min, 1e-9)`: `cellarea.<src>.R<row>C<col>.<type> = round(share, 4)`. Cells below
     `cell_min` (`table.cell_area_min`, 0.02) are left out, so they become 0 in the table (the settings comment:
     "smaller cell coverage is edge spill, recorded as 0").

Per-cell columns exist for `sam2` only "because per-cell columns multiply fast"; the module docstring notes that
outlines are stored, so other sources can be derived later without any model.

#### 3.3.9 Step 8: from per-photo dicts to columns, and roll-up pruning

1. `blocks[column][slot] = value` for every photo's columns (via `fid_slot`).
2. Roll-ups without children are removed. `has_children` = every name that appears as a parent in `tax.parents`.
   Every `rollup.*` column whose third part (`col.split(".", 2)[2]`) is not in `has_children` is deleted. The code
   comment: otherwise they duplicate `count.*`.
3. `counts = DataFrame(blocks, index=grid, dtype=float).fillna(0.0)`: every slot first gets 0 where the column has
   no value.

#### 3.3.10 Step 9: NaN versus 0

For each column in `counts`, `src = col.split(".")[1]`, and rows where the source did not look are set to NaN:

| Source token in the column | Kept (0 where absent) only where | NaN otherwise |
|---|---|---|
| `vlm` (`count.vlm.*`, `rollup.vlm.*`, `rel.vlm.*`) | slot `ok` and `vlm_ok` | VLM failed, no frame, unusable frame |
| `yolo` (`count`, `rollup`, `cell`, `area`) | slot `ok` | no usable frame |
| `yolo_cNN` | slot `ok` and `detector_min_conf <= level + 1e-9` | frame read before boxes that low were stored |
| `sam2` | slot `ok` and `segmenter_version` set | frame never outlined |
| `sam2_cNN` | the `sam2` rule and the `yolo_cNN` rule | either fails |
| any other token (`geometry`, ...) | slot `ok` | no usable frame |

Consequences:

- A column exists only if at least one valid slot in the grid produced it. With `max_slots` or `start`/`end`, the
  set of columns depends on the window.
- Within a valid slot, an object type that the source did not report is 0.
- Detector columns stay filled when only the VLM failed.

#### 3.3.11 Step 10: scene, categorical, image and frame-feature columns

**`scene.*` (numeric).** Read from `scene_state` for the space: `people_yolo, people_vlm, vehicles_yolo,
count_disagree`, restricted to picked valid frames, re-indexed to slots, prefixed `scene.`. Slots with no
`scene_state` row are NaN. The builder does not apply the `vlm_ok` mask here; `scene.people_vlm` is NaN after a VLM
failure because `people_vlm` is stored as NULL in that case (`src/sbi/pipeline.py` stores `None`).

**`cat.*` (strings).** The fixed fields `CAT_COLUMNS = ["lighting", "door_state", "tidiness", "weather_visible"]`
from `scene_state`, plus every key of `attrs_json` (each key becomes a column, for example `cat.occupancy_level` for
parking's scene field). Prefixed `cat.`, re-indexed, then every row where `vlm_ok` is False is set to NaN.
The four fixed fields exist as columns for every space, even one whose VLM never fills them (they are then all NaN
and the catalog marks them not ready for too few valid slots).

**`img.*`.** `brightness, contrast, sharpness` of the picked frame, prefixed `img.`, NaN where the slot is not `ok`.

**`feat.*`.** `frame_features` rows (`frame_id, name, value`) joined to `frames` for the space, restricted to picked
valid frames, pivoted to one column per `name` (`pivot_table`, mean when a name repeats), prefixed `feat.`.
Slots without a row are NaN. No rows at all: no `feat.*` columns.

#### 3.3.12 Step 11: image-vector columns: `_embedding_columns` and `embed_features.py`

`src/sbi/forecast/table.py::_embedding_columns(conn, space_id, index, fid_slot, model, components, fit_end)`

- Returns an empty frame when `model` is not set, there are no valid frames, or no vectors are stored.
- `load_embeddings(conn, space_id, model, set(fid_slot))` (`src/sbi/store/embeddings.py`) returns vectors of the
  picked frames and a list of corrupt rows. Corrupt rows are left out (their slot reads as unknown) with a warning
  naming up to 5 frames. Vectors of mixed dimensions for one model raise `StorageError` (not caught).
- Builds a `(slots, dim)` float32 matrix of NaN, filling the rows of slots that have a vector.
- `fit = index < Timestamp(fit_end)` when `fit_end` is given (strictly earlier slots), else `None`.
- Column source name: the part of `model` before `@`, with every character outside `[A-Za-z0-9_-]` replaced by `_`
  ("a column source has no dots"). `dinov2-base@448` -> `emb.dinov2-base.<feature>`.

`src/sbi/forecast/embed_features.py::embedding_features(vectors, *, components, fit=None) -> {name: (slots,) array}`

| Output | Definition |
|---|---|
| `pc1 .. pcN` (`N = components`, `table.embed_components` = 4) | `(v_t - mean) · axis_i` for every slot with a vector |
| `drift_prev` | `cosine_distance(v_t, v_{t-1})`: how much the scene changed in one slot |
| `drift_typical` | `cosine_distance(v_t, mean)`: how unusual the scene looks |

Rules:

- `have` = rows with no NaN. `use = have` (or `have & fit`). Rows without a vector are NaN in every output.
- `cosine_distance(a, b)` = row-wise `1 - (a·b) / (|a| |b|)`, NaN where either row is NaN or has zero length,
  clipped to `[0, 2]`.
- `drift_prev`: the first row is NaN; any row whose previous slot has no vector is NaN. With one slot, all NaN.
- `src/sbi/forecast/embed_features.py::principal_axes(x, count)`: `mean = x.mean(0)`; covariance
  `C = (x - mean)^T (x - mean) / max(n - 1, 1)`; eigen-decomposition with `numpy.linalg.eigh`; the top
  `min(count, max(n - 1, 0))` eigenvectors by eigenvalue, as unit-length rows. Sign rule: each axis is flipped so that
  its largest-magnitude loading is positive ("so the same data always gives the same columns"). Fewer than `count`
  axes come back when the data has fewer directions of variation (3 rows give at most 2).
- The mean and axes are fit only when `use.sum() >= 2`, in float64. Otherwise every `pc` and `drift_typical` is NaN.
  Missing axes leave their `pc` column all NaN.
- `drift_prev` does not depend on the fit.
- `fit_end` and leakage: the module docstring says a backtest must pass the training window, "otherwise a back-test
  would see the future through the components". `sbi backtest` builds its table with `fit_end` = the first origin's
  timestamp (3.11).

Image-vector columns need no detector and no VLM, so they exist for every embedded frame (of a valid slot).

#### 3.3.13 Step 12: probe columns: `_probe_columns` and `probe_features.py`

`src/sbi/forecast/table.py::_probe_columns(conn, space_id, index, fid_slot, scorer, groups)`

- Returns an empty frame when `scorer` is not set, `groups` (the space's `probes`) is empty, there are no valid
  frames, or no scores are stored.
- `load_scores(conn, space_id, scorer, set(fid_slot))` (`src/sbi/store/scores.py`) gives
  `{frame_id: {prompt: score}}`.
- One column per concept in `groups`, all NaN at first. For each frame with scores,
  `concept_values(by_prompt, groups)` fills its slot.
- Column name: `probe.<scorer with non [A-Za-z0-9_-] replaced by _>.<concept>`, for example
  `probe.siglip2-base-patch16-224.people`.

`src/sbi/forecast/probe_features.py::concept_values(scores, groups) -> {concept: value}`:
`value = mean(score of each prompt of the concept)` when every prompt has a score; NaN when any prompt is missing
("a mean over fewer paraphrases would be a different measurement, not a missing one") or the prompt list is empty.
For SigLIP the score is a raw logit (0 = even odds).

Concepts per space (from `config/spaces/*.yaml`, three paraphrases each):

| Space | Concepts |
|---|---|
| `admin_cabin` (21) | people, empty, lit, dark, messy, tidy, meeting, computer_work, sitting, standing, crowded, one_person, screens_on, screens_off, sunlight, door_open, eating, phone_call, bags_on_desk, cleaning, papers |
| `parking` (20) | bikes, empty, car, night, day, person, full, half_full, arriving, parking_bike, helmets, bicycles, truck, wet, shadows, lights_on, group, cleaning, disorder, orderly |

#### 3.3.14 Step 13: column order and the result

```
table = [META: slot_status, frame_id, vlm_ok]
      + [cal.*: hour, minute_of_day, dow, is_weekend, slot_of_day, is_holiday, is_working_day]
      + scene.* (people_yolo, people_vlm, vehicles_yolo, count_disagree)
      + img.* (brightness, contrast, sharpness)
      + count.*   (sorted by name)
      + area.*    (sorted)
      + rollup.*  (sorted)
      + cell.*    (sorted)
      + cellarea.* (sorted)
      + rel.*     (sorted)
      + cat.*
      + feat.*
      + emb.*  (pc1..pcN, drift_prev, drift_typical)
      + probe.* (in the order of the space file's probes)
```

Sorting is plain string order across sources inside a family (for example `count.sam2.*`, then `count.sam2_c05.*`,
..., `count.vlm.*`, `count.yolo.*`, `count.yolo_c05.*`).

`src/sbi/forecast/table.py::TableResult` (frozen dataclass):

| Field | Content |
|---|---|
| `table` | the DataFrame; index = naive local timestamps, regular step, named `timestamp` |
| `catalog` | one row per non-META column (3.5) |
| `cal` | the `CalendarSpec` used, so future calendar rows match the table's |

The catalog is built by `_catalog(table, min_coverage, min_presence, min_valid_slots, min_present_slots,
frozenset(exclude_labels))`.

### 3.4 The per-photo column cache: `src/sbi/forecast/feature_cache.py`

Purpose (module docstring): deriving the per-photo columns "means decoding and comparing every outline of every
photo, which was most of the table's build time and grew with every photo taken". A photo's columns only change when
it is read again, the vocabulary links change, or the derivation settings change, so they are computed once and kept.

**File.** `<paths.cache>/table_features.db` (default `data/cache/table_features.db`), a separate SQLite file, "not the
main database, which the dashboard opens read-only". `paths.cache` is described as "derived, safe to delete: rebuilt
on the next table build".

**Connection.** `FeatureCache._connect()` creates the parent directory, opens SQLite with `timeout=15`, sets
`PRAGMA journal_mode=WAL` and creates the table if needed.

**Schema.**

```sql
CREATE TABLE IF NOT EXISTS frame_columns(
  frame_id INTEGER NOT NULL,
  key      TEXT NOT NULL,     -- fingerprint of everything the columns depend on besides the photo
  stamp    TEXT NOT NULL,     -- when the photo was last read (str(processed_utc))
  cols     TEXT NOT NULL,     -- JSON {column: value}
  PRIMARY KEY(frame_id, key))
```

**Key.** `src/sbi/forecast/feature_cache.py::fingerprint(*parts)` =
first 16 hex characters of `sha1(json.dumps(parts, sort_keys=True, default=sorted))`. `default=sorted` turns sets
into sorted lists. `src/sbi/forecast/table.py::_frame_columns_cached` passes, in order:

| Part | Meaning |
|---|---|
| `FEATURES_VERSION` | code version of `frame_columns` (currently `1`; "bump when frame_columns() changes what it computes: old cache entries then miss") |
| `space_id` | the space |
| `conf_levels` | detector threshold variants |
| `mask_sources` | `{"sam2": None, "sam2_cNN": level, ...}` |
| `mask_dup_iou` | outline de-duplication IoU |
| `grid` | `(cols, rows)` |
| `cell_area_min` | per-cell edge-spill cut-off |
| `sorted(tax.known)` | registry object types |
| `tax.alias` | alias -> object type |
| `tax.merged_into` | merges |
| `{k: sorted(v) for k, v in tax.parents.items()}` | approved `is_a` links |

**Stamp.** `str(frames["processed_utc"])` of the picked frame. Re-processing a photo (for example `sbi reprocess`)
writes a new `processed_utc` (`src/sbi/store/results.py` updates it together with `detector_min_conf` and
`segmenter_version`), so its entry misses.

**Lookup flow (`_frame_columns_cached`).**

1. No cache: `frame_columns` for all ids.
2. `have = cache.load(key, stamps)`.
3. `missing` = ids not in `have`. If any, `fresh = frame_columns(missing)`, `cache.store(key, stamps, fresh)`,
   `have.update(fresh)`.

**`FeatureCache.load(key, stamps)`.** Empty `stamps`: `{}`. Otherwise selects rows with this `key` for the ids in
batches of `_BATCH = 500`, and keeps a row only if its `stamp` equals the current stamp. JSON is decoded per row.
Any `sqlite3.Error`, `OSError` or `ValueError`: a warning "table cache unreadable (...): computing without it" and
`{}` (everything is recomputed).

**`FeatureCache.store(key, stamps, columns)`.** Empty `columns`: nothing. Otherwise, in one transaction: delete rows of
the same frame ids under any other key (`key <> ?`), then `INSERT OR REPLACE` `(frame_id, key, stamp, json.dumps(cols))`.
Photos with no derived columns are stored too (as `{}`), because `frame_columns` returns an entry for every id.
Any `sqlite3.Error` or `OSError`: warning "table cache not written (...)" and the build continues.

**Invalidation summary.**

| Change | Effect |
|---|---|
| Photo re-read (new `processed_utc`) | stamp differs: that photo misses and is recomputed, then replaced |
| Any table setting in the key, vocabulary link, alias, merge, registry type, or `FEATURES_VERSION` | key differs: every photo misses; on store, old-key rows of the stored photos are deleted |
| Cache file missing, locked or corrupt | logged; columns computed directly |

Entries of photos that are never stored again under a new key stay in the file until it is deleted.

### 3.5 The catalog: `_catalog` and `build_catalog`

`src/sbi/forecast/table.py::build_catalog(table, *, min_coverage, min_presence, min_valid_slots, min_present_slots,
exclude_labels=())` is the public wrapper around `_catalog`. Its docstring: "The column catalog for any slice of a
table (a backtest builds it from the history before each forecast, so column selection never sees the future)."

`_catalog` makes one row per column that is not in `META`. Per column `s`:

```
parts    = col.split(".");  kind = parts[0];  source = parts[1] if len(parts) >= 3 else ""
nn       = s.dropna()
numeric  = kind != "cat" and s has a numeric dtype
coverage = mean(s is not NaN)                                  over all slots of the table
present  = count(nn > 0) if numeric else len(nn)
presence = mean(nn > 0) if numeric and nn non-empty else NaN   over valid (non-NaN) slots
std      = population std of nn (ddof=0) if numeric and nn non-empty else NaN
autocorr1 = s.autocorr(1) if numeric and len(nn) > 2 and std > 0 else NaN   (warnings suppressed)
distinct = number of distinct non-NaN values
obj      = parts[-1] if kind in OBJECT_KINDS else None
```

`COUNT_KINDS = {count, rollup, cell, rel, area, cellarea}`; `OBJECT_KINDS = {count, rollup, cell, area, cellarea}`
(families whose last name part is an object type). `s.autocorr(1)` is the pandas lag-1 Pearson correlation of the
series with itself shifted by one slot; pairs with a NaN are skipped.

**Readiness rules, checked in this order (first match wins):**

| # | Condition | `model_ready` | `reason` |
|---|---|---|---|
| 1 | `kind == "cal"` | True | `known in advance` |
| 2 | `obj in exclude_labels` (`wall, floor, ceiling, pillar, roof`) | False | `structural object (building part)` |
| 3 | `len(nn) < min_valid_slots` (8) | False | `only N valid slots (< 8)` |
| 4 | `coverage < min_coverage` (0.5) | False | `coverage X% < 50%` |
| 5 | `distinct < 2` | False | `constant` |
| 6 | `kind in COUNT_KINDS` and (`present < min_present_slots` (3) or `presence < min_presence` (0.05)) | False | `present in only N slots (flicker)` |
| 7 | otherwise | True | `ok` |

Notes:

- Calendar columns are always ready, even when constant (for example `cal.is_holiday` over a span with no holiday).
- Rule 2 applies to object families only; `rel.*` names are not checked against `exclude_labels`.
- Rule 6 applies to areas, cells, cell areas and relations as well as counts. `cat`, `scene`, `img`, `feat`, `emb`
  and `probe` columns are not subject to it.

**Catalog columns:**

| Column | Content |
|---|---|
| `column` | table column name |
| `kind` | first name part |
| `source` | second name part for names of three or more parts, else `""` |
| `known_future` | `kind == "cal"` |
| `variant` | `is_variant(kind, source)` |
| `coverage` | rounded to 3 decimals |
| `valid_slots` | `len(nn)` |
| `present_slots` | `present` |
| `presence` | rounded to 3, or None |
| `mean` | mean of `nn`, rounded to 3, or None (non-numeric / empty) |
| `std` | rounded to 3, or None |
| `autocorr1` | rounded to 3, or None |
| `distinct` | distinct non-NaN values |
| `model_ready` | bool |
| `reason` | text, as in the table above |

**`src/sbi/forecast/table.py::is_variant(kind, source)`**: True when the source ends in `_cNN` (regex `_c\d\d$`:
`yolo_c05`, `sam2_c10`, ...) or when `source == "sam2" and kind == "count"`. Docstring: "A near-copy of a primary
series (another threshold, or the same count after outline de-duplication)." So `area.sam2.*` and `cellarea.sam2.*`
are primary series; `count.sam2.*`, every `*_cNN` column are variants.

The `sbi table` command also adds a `default_input` column (whether `select_covariates` with `targets=[]` picks the
column) to the exported catalog CSV; that column is not part of `_catalog`.

### 3.6 Choosing past-only covariates

#### 3.6.1 `src/sbi/forecast/table.py::select_covariates`

```
select_covariates(catalog, *, targets, max_covariates, always, min_autocorr, autocorr_min_slots, kinds=None) -> list[str]
```

The capped default set ("with 19 rows of history, 180 inputs would be noise", module docstring). Steps:

1. `ready` = catalog rows with `model_ready`, `kind != "cal"`, and `column not in targets`.
2. If `kinds` is given, keep only rows whose `kind` is in it.
3. `chosen` = the `always` columns that are in `ready`, in the order of `always`. Variants are allowed here.
4. `rest` = `ready` minus `chosen`, minus every `variant` row.
5. `ac = autocorr1` with missing values as 0.
6. Smoothness filter, only when `catalog["valid_slots"].max() >= autocorr_min_slots` (96): keep rows with
   `kind == "cat"` or `ac >= min_autocorr` (0.3). Because calendar columns have no NaN, the maximum equals the table
   length, so the filter switches on once the table has about a day of slots. The docstring: about ±0.23 noise in
   autocorrelation at 19 slots; "real occupancy changes smoothly, model noise flickers".
7. Sort `rest` by `ac`, then `coverage`, then `std`, all descending (missing `std` last).
8. Return `chosen + first max(0, max_covariates - len(chosen)) of rest`.

`table.always_include` in `config/pipeline.yaml`: `scene.people_vlm, scene.people_yolo, scene.vehicles_yolo,
count.vlm.motorcycle, count.yolo.car`. `table.max_covariates` = 24. These settings matter only for the `detector`
and `full` variants (3.8.2) and the `sbi table` report.

#### 3.6.2 `src/sbi/forecast/table.py::all_covariates`

`all_covariates(catalog, *, targets)` = every `model_ready` column with `kind != "cal"` that is not a target, in table
order. No cap, no smoothness filter, variants and `cat.*` included. Docstring: "Chronos-2 attends across all of them
and decides what is useful."

### 3.7 Calendar and site calendar: `src/sbi/sitecal.py`

Docstring: holidays and closures become known-future model inputs (`cal.is_holiday`, `cal.is_working_day`), "the
kind of covariate published Chronos-2 uses lean on most".

**`CalendarSpec`** (frozen dataclass):

| Field | Default | Meaning |
|---|---|---|
| `holidays` | `frozenset()` | dates that are not normal days |
| `working_days` | `frozenset({0,1,2,3,4})` | weekday numbers, Monday = 0 |
| `names` | `{}` | date -> name, display only (excluded from equality and hash) |

**`read_site_file(path) -> {date: name}`.** A missing file is an empty calendar. Invalid YAML raises
`ConfigError("<path>: not valid YAML: ...")`. The document's `days` list is read; each entry must be a mapping whose
`date` YAML parses as a date (written `YYYY-MM-DD` unquoted), else `ConfigError("<path>: days[i] needs a date written
as YYYY-MM-DD ...")`. The name is `str(entry["name"])`, or `"closed"` when it is missing or empty.

Site file format (`config/site_calendar.yaml`):

```yaml
days:
  - {date: 2026-01-26, name: Republic Day}
  - {date: 2026-08-15, name: Independence Day}
```

The shipped file lists only India's fixed-date national holidays for 2026 and 2027 (Republic Day, Independence Day,
Gandhi Jayanti); its comment says moving festivals are deliberately not guessed and should be added from the
company's list.

**`country_days(country, subdivision, years) -> {date: name}`.** No `country`: `{}`. If the optional `holidays`
package cannot be imported: `{}`, with one info log per process ("`holidays` package not installed: using the site
calendar file only"). Otherwise `holidays.country_holidays(country, subdiv=subdivision, years=list(years))`; a
`KeyError` or `NotImplementedError` (unsupported country/subdivision) becomes `ConfigError`. In the project's
`.venv` (checked 18 Nov 2026) `holidays` is not installed, so only the site file is in effect.

**`load_calendar(settings, root, years) -> CalendarSpec`.** `site = read_site_file(root / settings.site_file)`;
`names = {**country_days(...), **site}` (the site file wins on a shared date);
returns `CalendarSpec(frozenset(names), frozenset(settings.working_days), names)`.

**`CalendarSettings`** (`src/sbi/settings.py`, `calendar:` block):

| Key | Code default | `config/pipeline.yaml` | Meaning |
|---|---|---|---|
| `country` | `"IN"` | `IN` | country for the `holidays` package |
| `subdivision` | `None` | `null` | state code for regional holidays |
| `site_file` | `config/site_calendar.yaml` | same | site holiday list, relative to the root |
| `working_days` | `[0,1,2,3,4]` | same | validator: non-empty, distinct, each in 0..6 |

### 3.8 Inputs: `src/sbi/forecast/inputs.py`

Shared by live forecasts and the backtest "so both make exactly the same choices".

Constants:

- `NONNEGATIVE_KINDS = {scene, count, rollup, area, cell, cellarea, rel, img}`: kinds clipped at 0 after forecasting.
- `NOT_TARGETS = {cal, cat}`: calendar (known in advance) and categorical strings are never targets.

#### 3.8.1 `resolve_targets(patterns, catalog, max_targets) -> list[str]`

1. `ready` = catalog rows with `model_ready` and `kind not in NOT_TARGETS`. `columns` = their names in catalog (table)
   order. `primary` = the ready names that are not variants.
2. For each pattern, in the order written:
   - Contains any of `*`, `?`, `[`: wildcard. `hits` = ready primary columns matching with `fnmatchcase`
     (case-sensitive shell-style). Variants are skipped, "so a broad pattern does not fill the cap with near-copies".
   - Otherwise: exact name. A variant matches when named exactly.
   - No hits: warning "forecast target %r matches no model-ready column".
   - New hits are appended in table order, without duplicates.
3. Nothing chosen: `ForecastError("none of the forecast targets matches a model-ready column")`.
4. More than `max_targets`: info log, and only the first `max_targets` are kept.

`forecast.max_targets`: code default 40, `config/pipeline.yaml` 100.

Space targets (`forecast_targets`):

| Space | Patterns |
|---|---|
| `admin_cabin` | `scene.people_yolo`, `scene.people_vlm`, `count.yolo.*`, `area.sam2.*`, `probe.*` |
| `parking` | `scene.vehicles_yolo`, `scene.people_yolo`, `count.yolo.*`, `count.vlm.motorcycle`, `area.sam2.*`, `probe.*` |

`count.yolo.*` does not match `count.yolo_c05.*` (the pattern requires a dot right after `yolo`), and variants are
excluded from wildcards anyway. `SpaceConfig` rejects empty or duplicate patterns.

#### 3.8.2 Variants: `DEFAULT_VARIANTS` (`src/sbi/settings.py`)

A variant is a named list of column kinds the model may see. `"cal"` means the calendar; `ALL_KINDS = "*"` means
every model-ready extracted column, uncapped; `PROFILE_KIND = "prof"` means each target's day-type profile.

| Variant | Kinds | Past-only covariates | Calendar | Profiles | Future frame columns |
|---|---|---|---|---|---|
| `all` (code default `forecast.variant`) | `cal, *` | `all_covariates` | yes | no | `id, timestamp`, 7 `cal.*` |
| `all_profile` (`config/pipeline.yaml` live variant) | `cal, *, prof` | `all_covariates` | yes | yes | `id, timestamp`, 7 `cal.*`, `prof.<target>` per target |
| `profile` | `cal, prof` | none | yes | yes | `id, timestamp`, 7 `cal.*`, `prof.*` |
| `univariate` | (empty) | none | no | no | `id, timestamp` |
| `calendar` | `cal` | none | yes | no | `id, timestamp`, 7 `cal.*` |
| `detector` | `cal, scene, count, rollup, area, cell, cellarea, rel, img, cat` | `select_covariates(kinds=...)`, capped | yes | no | `id, timestamp`, 7 `cal.*` |
| `full` | `detector` kinds + `emb, probe` | `select_covariates(kinds=...)`, capped | yes | no | `id, timestamp`, 7 `cal.*` |

`feat.*` columns are reachable only through `*`. `ForecastSettings.variants` defaults to `DEFAULT_VARIANTS`; the
YAML does not override it. The validator requires `forecast.variant` to be a key of `variants`.

#### 3.8.3 `prepare(result, space_id, patterns, kinds, *, horizon, settings, table_settings) -> Prepared`

Inputs for one forecast from `result`, "whose table ends at the forecast origin".

1. `targets = resolve_targets(patterns, result.catalog, settings.max_targets)`.
2. `past_kinds = kinds` without `"cal"` and `"prof"`.
3. Covariates:
   - `"*"` in `past_kinds`: `all_covariates(catalog, targets=targets)` (any other kinds listed with it are ignored).
   - else, non-empty `past_kinds`: `select_covariates(catalog, targets=targets,
     max_covariates=table.max_covariates, always=table.always_include, min_autocorr=table.min_autocorr,
     autocorr_min_slots=table.autocorr_min_slots, kinds=past_kinds)`.
   - else: `[]`.
4. `calendar = "cal" in kinds`.
5. `context, future = to_chronos(result, space_id, targets, horizon, covariates=covariates, calendar_inputs=calendar)`.
   An explicit list is always passed, so the selection defaults inside `to_chronos` are not used here.
6. If `"prof" in kinds`: `add_profiles(context, future, targets, working=result.table["cal.is_working_day"],
   future_working=calendar(future timestamps, result.cal)["cal.is_working_day"])` with the default
   `min_same_weekday=2` (not configurable through settings).
7. Returns `Prepared(targets, covariates, calendar, context, future, profiles)`.

`Prepared` (frozen dataclass): `targets`, `covariates` (past-only), `calendar` (bool), `context`, `future`,
`profiles` (bool, default False).

#### 3.8.4 `src/sbi/forecast/table.py::to_chronos`

```
to_chronos(result, space_id, targets, horizon, covariates=None, calendar_inputs=True, **select_kw)
    -> (context_df, future_df)
```

The long format of `Chronos2Pipeline.predict_df(id_column="id", timestamp_column="timestamp", target=targets)`.

1. `horizon < 1`: `ValueError("horizon must be >= 1")`. Any target not in the table:
   `ValueError("unknown target columns: [...]")`.
2. `cal_cols` = every `cal.*` column in table order when `calendar_inputs`, else none.
3. `covariates is None`: `select_covariates(catalog, targets, **({"max_covariates": 24, "always": [],
   "min_autocorr": 0.3, "autocorr_min_slots": 96} | select_kw))`.
4. **Context frame**: `table[targets + covariates + cal_cols]` (in that order). Every `cat.*` column is cast to
   `object` with NaN replaced by `None` (strings; Chronos-2 treats them as categories). The index becomes the
   `timestamp` column (a column named `index` is renamed to `timestamp`), and `id = space_id` is inserted as the first
   column. Columns: `id, timestamp, <targets>, <covariates>, <cal.*>`. Rows: every slot of the table, NaN kept.
5. Step: `table.index.freq`, or the inferred frequency when the index has none.
6. **Future frame**: `future_index = date_range(last timestamp + step, periods=horizon, freq=step)`;
   `calendar(future_index, result.cal)` with the index as `timestamp`, and `id` first. Columns: `id, timestamp`,
   7 `cal.*`. When `calendar_inputs` is False, only `id, timestamp` ("nothing known in advance: the future frame is
   just the clock").

The future frame never contains a target or a past-only covariate. Profiles, when used, are added afterwards by
`add_profiles`.

#### 3.8.5 Day-type profiles: `src/sbi/forecast/profile_features.py`

Docstring: a zero-shot model is told the weekday and working-day flag, but with a few days of history it cannot
connect them to behaviour ("after an empty weekend it expects an empty Monday"); the profile spells it out.

`day_profile(history, working, future_index, future_working, *, min_same_weekday=2) -> (past_profile, future_profile)`

For one series. Per row: `v` (value, NaN allowed), `tod = hour*60 + minute`, `dow`, `work` (1/0), `has = v is not
NaN`. `overall = nanmean(v)` (NaN when the series has no value).

Levels, tried from most to least specific; a row keeps the first level that is allowed:

| Level | Group keys | Needed number of other days with a value |
|---|---|---|
| 1 | `dow, tod` (same weekday, same clock time) | `min_same_weekday` (2) |
| 2 | `work, tod` (same day type, same clock time) | 1 |
| 3 | `tod` (any day, same clock time) | 1 |
| 4 | none | the series' overall average |

Past rows (leave own day out). For each level, with `total` = sum of `v` in the row's group and `count` = number of
rows with a value in the group:

```
others = count - has_row
loo    = (total - (v_row if has_row else 0)) / others
```

A row takes `loo` at the first level where `others >= need`. Each group has at most one row per day (one slot per
clock time per day), so `others` is the number of other days with a value. The profile is "never a copy of the value
it sits beside".

Future rows: the group mean over all history rows of the matching group, `total / count`, at the first level where
`count >= need`. Rows still empty after level 3 get `overall`. The result is NaN only when the history has no value at
all.

`add_profiles(context, future, targets, working, future_working, *, min_same_weekday=2) -> (context, future, names)`
adds one `prof.<target>` column per target to both frames (so the model treats it as known in advance), computed
from the context frame's own values and timestamps. In the backtest the context is the history before the origin,
so the profile cannot contain the future.

#### 3.8.6 `predict(forecaster, prepared, quantiles, catalog) -> DataFrame`

1. `out = forecaster.forecast(prepared.context, prepared.future, targets=prepared.targets, quantiles=quantiles)`.
2. `nonnegative` = catalog columns whose kind is in `NONNEGATIVE_KINDS`.
3. For rows whose `target` is in `nonnegative`, every quantile column (`qname(q)`) is clipped at a lower bound of 0.

Clipped: `scene, count, rollup, area, cell, cellarea, rel, img`. Not clipped: `probe` (SigLIP logits can be
negative), `emb` (components can be negative), and `feat`. Docstring: "the one domain rule models cannot know".
Live forecasts and the backtest both go through this function.

### 3.9 Forecasters: `src/sbi/forecast/models.py`

`qname(q) = f"q{round(q * 100):02d}"`: 0.1 -> `q10`, 0.5 -> `q50`, 0.9 -> `q90`, 0.05 -> `q05`.

**`Forecaster` protocol.**

| Member | Meaning |
|---|---|
| `name: str` | stored with every run, for example `chronos2:amazon/chronos-2` |
| `probabilistic: bool` | False: every quantile equals the point forecast, intervals mean nothing |
| `uses_covariates: bool` | False: results do not depend on the covariate set |
| `forecast(context, future, *, targets, quantiles) -> DataFrame` | long frame `timestamp, target, step, q..`; raises `ForecastError` if the model fails or a target has no usable history |

`step` is 1 for the first slot after the history.

**Baselines** (`_Baseline`: `probabilistic = False`, `uses_covariates = False`). `_Baseline.forecast` loops over
targets: `_series(context, target)` builds a float series indexed by timestamp and raises
`ForecastError("target ... has no observed values in the history")` when it is all NaN; `_predict(history, when)`
gives one value per future timestamp; `_long` writes the same value into every quantile column.

| Class | `name` | Rule |
|---|---|---|
| `Persistence` | `persistence` | the last non-NaN value, held flat |
| `SeasonalNaive` | `seasonal_naive` | for each future time `t`: the value at `t - d days` for the smallest `d` in 1..7 where that timestamp is in the history and not NaN; persistence when none |
| `SeasonalMean` | `seasonal_mean` | for each `t`: mean of history values at the same weekday and clock time when there are at least 2 such points; else the mean of the same clock time on any day; else persistence |

`SeasonalMean` is "the weekday-by-hour average, at slot resolution". All baselines ignore covariates, calendar and
profiles.

**`Chronos2Forecaster(model="amazon/chronos-2", *, cross_learning=False)`.** `probabilistic = True`,
`uses_covariates = True`, `name = f"chronos2:{model}"`.

- Loading: imports `torch` and `chronos.Chronos2Pipeline`, then
  `Chronos2Pipeline.from_pretrained(model, device_map="cuda" if torch.cuda.is_available() else "cpu")`. Any exception
  becomes `ForecastError("failed to load forecaster <model>: ...")`.
- `forecast(context, future, *, targets, quantiles)` calls:

```python
self._pipe.predict_df(
    context,
    future_df=future,
    prediction_length=len(future),          # = horizon
    quantile_levels=list(quantiles),
    id_column="id",
    timestamp_column="timestamp",
    target=list(targets),
    cross_learning=self.cross_learning,
)
```

  Any exception becomes `ForecastError("forecasting failed: ...")`.
- Output shaping: rename `target_name` -> `target`; for each quantile the model's column named `str(q)` (for example
  `"0.1"`) must exist, else `ForecastError("model output has no column for quantile ...")`, and is copied as float to
  `qname(q)`; sort by `target, timestamp`; `step = cumcount within target + 1`; return only
  `timestamp, target, step, q..` with a fresh index.

**`build_forecaster(kind, *, model="amazon/chronos-2", cross_learning=False)`.** `"chronos2"` ->
`Chronos2Forecaster(model, cross_learning=...)`; a key of `BASELINES` (`persistence`, `seasonal_naive`,
`seasonal_mean`) -> a new baseline instance; anything else ->
`ForecastError("unknown forecaster ...; choose from chronos2, persistence, seasonal_mean, seasonal_naive")`.

### 3.10 Metrics: `src/sbi/forecast/metrics.py`

`POINT_COLUMNS = [timestamp, target, step, y, pred, abs_err, pinball, covered]`.

**`pinball(y, pred, tau)`** = `max(tau * d, (tau - 1) * d)` with `d = y - pred`.

**`evaluate(forecast, truth, quantiles, *, probabilistic) -> DataFrame`.** `truth` has a timestamp index and one column
per target.

1. `y = truth.at[timestamp, target]` when both exist, else NaN. Rows with NaN `y` are dropped ("an unmeasured slot
   has no right answer"). None left: an empty frame with `POINT_COLUMNS`.
2. `pred = q50`.
3. `abs_err = |y - pred|`.
4. `pinball = mean over the quantiles tau of pinball(y, q_tau, tau)`.
5. `covered = 1.0 if q_min <= y <= q_max else 0.0`, using the lowest and highest quantile (q10..q90: an 80%
   interval); NaN for non-probabilistic forecasters.

**`summarise(raw, *, reference="seasonal_mean") -> DataFrame`.** `raw` = `evaluate` rows plus `origin, forecaster,
variant`. Empty input: an empty frame with columns `forecaster, variant, target, n, mae, wql, coverage, skill`.
The reference's `abs_err` per `(origin, target, timestamp)` (duplicates dropped) is the reference error. Per
`(forecaster, variant, target)`:

| Output | Formula |
|---|---|
| `n` | number of points |
| `mae` | mean `abs_err` over all the group's points |
| `wql` | `2 * sum(pinball) / sum(|y|)`; NaN when `sum(|y|) = 0` |
| `coverage` | mean `covered`; NaN when all NaN |
| `skill` | `1 - MAE_same / MAE_ref`, both over the points the group shares with the reference (inner join on `origin, target, timestamp`); NaN when there are none or `MAE_ref` is 0 |

Positive skill is better than the reference; the reference itself scores 0.

**`overall(summary)`.** Per `(forecaster, variant)`: `targets` (distinct), `points` (sum of `n`), `skill` (mean over
targets), `wql` (mean), `coverage` (mean), sorted by `skill` descending, NaN last. "Each target counts once, so busy
series do not drown quiet ones."

### 3.11 Backtest: `src/sbi/forecast/backtest.py`

Rolling-origin backtest: forecast from many past moments and score against what then happened. Its protocol rules
"each exist to keep the future out of the forecast".

**`BacktestPlan`** (frozen): `horizon`, `step` (slots between origins), `min_context`, `quantiles`, `variants`
(name -> kinds), `patterns` (target patterns), `catalog_args` (thresholds for `build_catalog`, the same as the
table's), `cal` (`CalendarSpec`, default empty).

**`BacktestRun`**: `raw` (scored points), `origins` (number of origins), `failures` (`"forecaster/variant"` ->
origins skipped).

**`origins_for(n_rows, plan)`** = `range(min_context - 1, n_rows - horizon, step)`: row positions with
`min_context` rows up to and including the origin and a full horizon of truth after it. The last origin position
is at most `n_rows - horizon - 1`.

**`run_backtest(table, space_id, forecasters, plan, *, settings, table_settings) -> BacktestRun`.**

1. No origins: `ForecastError("not enough history to backtest: N slots, need at least min_context + horizon")`.
2. For each origin position `pos`:
   - `history = table.iloc[:pos + 1]`; `truth = table.iloc[pos + 1 : pos + 1 + horizon]`.
   - `result = TableResult(history, build_catalog(history, **plan.catalog_args), plan.cal)`: the catalog, and so the
     targets, covariate selection, autocorrelation and readiness, are rebuilt from the history only.
   - `origin = history.index[-1]`.
   - For each forecaster: if `uses_covariates` is False (baselines), one pass with variant `"-"` and kinds `[]`
     (once per origin, since covariates cannot change their output); otherwise one pass per plan variant.
   - Each pass: `prepare(...)` then `predict(...)`. A `ForecastError` increments `failures["<name>/<variant>"]`, logs
     a warning, and the pass is skipped. Other exceptions propagate.
   - `evaluate(out, truth, quantiles, probabilistic=fc.probabilistic)`, then `origin`, `forecaster` (name) and
     `variant` columns are added.
3. `raw` = concatenation of all scored parts (an empty frame when none).

Leak-free rules:

| Rule | How |
|---|---|
| The model sees only rows up to the origin | `history = table.iloc[:pos+1]`; `to_chronos` builds the context from it |
| Column readiness and selection never see the future | catalog rebuilt per origin |
| Only calendar values are given for the horizon | `to_chronos` future frame; profiles computed from `history` |
| Image-vector components fit on the first history only | the CLI builds the table with `fit_end` = first origin's timestamp; `fit = index < fit_end` |

Notes: per-photo columns (counts, areas, outlines, probes) are per-photo values and do not depend on other slots.
A column produced only by later photos exists in `history` as all-0 or all-NaN, and is then not ready
(constant or too few values).

`sbi backtest` (`src/sbi/cli.py`) wiring: builds the whole table once to find the origins, then rebuilds it with
`fit_end = str(whole.index[first_origin])`. Options: `--horizon` (default `forecast.horizon`), `--min-context`
(default `forecast.min_context`), `--step` (default 8 = 2 hours), `--variants` (default all of `forecast.variants`),
`--models` (default `chronos2,seasonal_mean,seasonal_naive,persistence`). It writes
`<space>_backtest_raw.csv` and `<space>_backtest.csv` (`summarise` with reference `seasonal_mean`) and prints
`overall`.

### 3.12 Live runs: `src/sbi/forecast/runner.py::forecast_space`

```
forecast_space(conn, space, result, forecaster, *, settings, table_settings, variant=None, horizon=None,
               step_minutes=15) -> ForecastRun
```

Checks, in order:

| Check | Error (`ForecastError`) |
|---|---|
| `variant` (default `settings.variant`) must be a key of `settings.variants` | `unknown variant ...; choose from [...]` |
| `horizon` defaults to `settings.horizon` | |
| `len(result.table) >= settings.min_context` (96) | `<space>: N slots of history, need at least 96 (forecast.min_context)` |
| `space.forecast_targets` non-empty | `<space> has no forecast_targets in its space file` |

Then `prepare(result, space_id, space.forecast_targets, settings.variants[variant], ...)`, `predict(forecaster,
prepared, settings.quantiles, result.catalog)`, `origin = table.index[-1]` (the newest slot of the table, whatever its
status). Stored in one transaction (`src/sbi/store/db.py::transaction`):

`insert_run(...)` -> one `forecast_runs` row:

| Column | Value |
|---|---|
| `run_id` | autoincrement |
| `space_id` | the space |
| `created_utc` | `now_utc().isoformat()` |
| `origin_local` | `origin.isoformat()` (naive IST, "last observed slot") |
| `model` | `forecaster.name` |
| `variant` | variant name |
| `horizon` | slots ahead |
| `step_minutes` | `settings.table.step_minutes` as passed by the caller |
| `context_slots` | `len(table)` (all rows, NaN rows included) |
| `targets_json` | JSON list of target names |
| `covariates_json` | JSON list: past covariate names, then the literal `"cal.*"` if the calendar was given, then `"prof.*"` if profiles were given |
| `quantiles_json` | JSON list, for example `[0.1, 0.5, 0.9]` |

`insert_forecasts(conn, run_id, forecast)` -> one `forecasts` row per non-NaN value:

| Column | Value |
|---|---|
| `run_id` | the run |
| `target` | table column name |
| `ts_local` | `Timestamp(timestamp).isoformat()` |
| `step` | 1 = slot right after the origin |
| `quantile` | from the column name: `int(name[1:]) / 100` for every column named `q` + digits |
| `value` | the (clipped) forecast |

Primary key `(run_id, target, ts_local, quantile)`; `ON DELETE CASCADE` from `forecast_runs`. A frame without any
`q<digits>` column raises `StorageError`. Returns the number of rows stored.

`load_forecast(conn, run_id)` returns a wide frame `target, timestamp, step, q..` sorted by `target, step`
(column name `"q" + str(round(quantile * 100))`); empty: columns `target, timestamp, step` only.
`latest_run(conn, space_id)` returns the newest `forecast_runs` row by `run_id`.

`ForecastRun` (frozen): `run_id, origin, targets, covariates` (past-only names), `forecast` (the frame).

### 3.13 Scheduled forecasts: `src/sbi/forecast/schedule.py`

**`due(now_utc, tz, every_minutes, step_minutes=15) -> bool`.** False when `every_minutes <= 0`. Otherwise
`minute = local hour * 60 + local minute` in the space's time zone, floored to the step:
`(minute - minute % step_minutes) % every_minutes == 0`. With `every_minutes = 60` the tick at hh:00 (and any tick
inside hh:00..hh:14) is due.

`ForecastSettings` validator: `every_minutes` must be 0 or a multiple of 15. Code default 0 (off);
`config/pipeline.yaml` sets 60.

**`run_scheduled(conn, settings, spaces, now_utc, *, forecaster=None) -> list[ScheduledForecast]`.**

1. `todo` = spaces with `forecast_targets` for which `due(...)` is True. None: `[]`, and no model is loaded.
2. If no forecaster was handed in: `build_forecaster(forecast.scheduled_model, model=forecast.model,
   cross_learning=forecast.cross_learning)`. An `SbiError` is logged and every `todo` space gets
   `ScheduledForecast(space_id, None, error)`.
3. Per space: `forecast_space(conn, space, table_for(conn, settings, space, max_slots=forecast.max_context),
   forecaster, settings=forecast, table_settings=settings.table, step_minutes=settings.table.step_minutes)`.
   - Success: `ScheduledForecast(space_id, run_id)`.
   - `SbiError`: error log, `ScheduledForecast(space_id, None, str(e))`.
   - Any other exception: `log.exception` ("this is a bug"), `ScheduledForecast(space_id, None, "unexpected: ...")`.
     One space's failure does not stop the others.

`ScheduledForecast` (frozen): `space_id`, `run_id` (None on failure), `error` (None on success).

The tick (`src/sbi/tick.py::_forecasts`) calls `run_scheduled` after every space has been captured and processed, and
wraps it in a last-resort `except Exception` that logs and never changes the tick's outcome. The module docstring:
"nothing here can stop or fail a capture".

### 3.14 Live accuracy: `src/sbi/forecast/accuracy.py`

Scores the real, stored forecasts (not simulations) against what was later measured. Reference: persistence, "the
last value measured before the forecast was made". `skill = 1 - MAE / MAE_persistence` on the same points.

**`outdated_runs(conn, space_id, *, tolerance=0.05) -> set[int]`.** Runs made from a history that has since been read
again (for example `sbi reprocess` after a detector change); scoring them "would measure the change of detector, not
the forecast".

1. Frames of the space with `processed_utc` set, ordered by `ts_utc`; runs of the space (`run_id, created_utc`).
   Either empty: `set()`.
2. Times are compared as naive UTC.
3. For each run: `made = created_utc`; `before` = number of photos with `ts_utc <= made`;
   outdated when `before > 0` and `(number of those photos with processed_utc > made) / before > tolerance`.
   A few late reads (for example a photo retried after a VLM outage) stay under the 5% tolerance.

**`live_points(conn, space_id, table, *, since=None) -> DataFrame`.** One row per stored forecast point with a
measured truth.

1. `since_iso = Timestamp(since).isoformat()` or `""`; runs of the space with `origin_local >= since_iso` (text
   comparison of ISO strings), ordered by `run_id`.
2. Skip outdated runs; skip runs whose `load_forecast` is empty.
3. `points = evaluate(forecast, table, quantiles_json, probabilistic = model not in BASELINES)`; skip if empty.
4. `history = table.loc[:origin]`; `last[target]` = last non-NaN value of `history[target]`, NaN if none (or the
   target is not in the table).
5. `persist_err = |y - last[target]|`.
6. `scale[target] = mean(|table[target].diff()|)` over the whole table passed in (the typical 15-minute change, as in
   MASE).
7. `lead_min = step * run.step_minutes`; adds `run_id, model, variant, origin`.

The table passed in decides the truth and the persistence value: `sbi accuracy` passes the whole history, the
dashboard passes the table cut to `forecast.max_context`.

**`_skill(g)`** (per-target, unscaled): over rows with `persist_err` set,
`1 - mean(abs_err) / mean(persist_err)`; NaN when the persistence mean is 0 or NaN.

**`by_target(points)`.** Per `(model, variant, target)`: `n`, `mae`, `persist_mae`, `skill` (`_skill`), `coverage`
(mean `covered`, NaN when all NaN). Empty input: empty frame with these columns.

**`pooled_skill(g)`.** Over rows with `persist_err` set and `scale > 0`:

```
pooled_skill = 1 - Σ (abs_err / scale) / Σ (persist_err / scale)        (NaN when the denominator is 0)
```

The module docstring: averaging per-series ratios instead "lets a series that happened not to move (persistence
error ~0) produce a huge negative skill that swamps everything else".

**`lead_bucket(minutes)`** with `LEADS`:

| Lead time `step * step_minutes` | Bucket |
|---|---|
| `<= 60` | `0-1 h` |
| `<= 120` | `1-2 h` |
| `<= 240` | `2-4 h` |
| larger | `4 h +` |

With `horizon = 16` and 15-minute slots: steps 1-4 -> `0-1 h`, 5-8 -> `1-2 h`, 9-16 -> `2-4 h`.

**`by_lead(points)`.** Per `(model, variant, lead)`: `runs` (distinct run ids), `targets`, `points`, `skill`
(`pooled_skill`), `coverage`; sorted by model, variant, then bucket order. Empty input: empty frame with these
columns.

**`by_weekday(points, target=None)`.** Groups by the weekday of the predicted timestamp, Monday first; days with no
scored point are left out.

| Mode | Columns |
|---|---|
| one `target` | `day, forecasts checked, average measured (mean y), average forecast (mean pred), average miss (mean abs_err), miss if nothing changed (mean persist_err), inside band (mean covered)` |
| all series (`target=None`) | `day, forecasts checked, skill (pooled_skill), inside band` |

For all series, averages "would mix units", so only the pooled skill and the band share are given.

### 3.15 Settings reference

**`table:` block** (`TableSettings`):

| Key | Code default | YAML | Used by |
|---|---|---|---|
| `step_minutes` | 15 | 15 | grid step; `due`; `step_minutes` stored with runs |
| `min_coverage` | 0.5 | 0.5 | catalog rule 4 |
| `min_presence` | 0.05 | 0.05 | catalog rule 6 |
| `min_valid_slots` | 8 | 8 | catalog rule 3 |
| `min_present_slots` | 3 | 3 | catalog rule 6 |
| `exclude_labels` | wall, floor, ceiling, pillar, roof | same | catalog rule 2 |
| `max_covariates` | 24 | 24 | `select_covariates` cap |
| `always_include` | `[]` | 5 columns (3.6.1) | `select_covariates` |
| `min_autocorr` | 0.3 | 0.3 | smoothness filter |
| `autocorr_min_slots` | 96 | 96 | smoothness filter switch |
| `embed_components` | 4 | 4 | `emb.*.pc1..pcN` |
| `conf_levels` | [0.05, 0.10] | same | `yolo_cNN`, `sam2_cNN` sources; validator: each in (0, 1), distinct after rounding to whole percents; returned sorted |
| `mask_dup_iou` | 0.5 | 0.5 | outline de-duplication |
| `cell_area_min` | 0.02 | 0.02 | `cellarea` cut-off |

**`forecast:` block** (`ForecastSettings`):

| Key | Code default | YAML | Notes |
|---|---|---|---|
| `model` | `amazon/chronos-2` | same | Hugging Face id |
| `quantiles` | [0.1, 0.5, 0.9] | same | validator: strictly increasing, in (0, 1), whole percents, must include 0.5 |
| `horizon` | 16 | 16 | 1..1024 slots |
| `min_context` | 96 | 96 | >= 8 |
| `max_context` | 8192 | 8192 | >= 8; `max_slots` for live runs and the dashboard |
| `max_targets` | 40 | 100 | >= 1 |
| `cross_learning` | False | false | passed to `predict_df` |
| `variant` | `all` | `all_profile` | must be a key of `variants` |
| `every_minutes` | 0 | 60 | 0 or a multiple of 15 |
| `scheduled_model` | `chronos2` | `chronos2` | `build_forecaster` kind for the tick |
| `variants` | `DEFAULT_VARIANTS` | not set | 3.8.2 |

### 3.16 Worked example (as of 18 Nov 2026)

Figures from `docs/chronos-data-flow.md`: the live system on 18 Nov 2026 at 10:30 IST, about 5 days of history.
Arithmetic shown in brackets follows from the code.

**Table.** 487 slots for each space. `487 = 5 days x 96 + 7`, so the grid starts at 18 Nov 10:30 minus
`486 x 15 min` = Fri 13 Nov 09:00. 487 is above `min_context` (96), so forecasts are allowed, and below `max_context`
(8192), so `max_slots` trims nothing. 487 is at least `autocorr_min_slots` (96), so the smoothness filter would apply
to the `detector`/`full` variants (it does not apply to `all_profile`). Admin cabin: 718 columns, 416 model-ready in
the catalog. Parking: 598 columns, 232 model-ready. Build time 5.2 s (cabin) and 1.8 s (parking).

**Targets.** Live variant `all_profile` = `["cal", "*", "prof"]`.

| Targets by family | Admin cabin | Parking |
|---|---|---|
| Scene totals | 2 | 2 |
| Counts per object type (`count.yolo.*`, plus `count.vlm.motorcycle` for parking) | 11 | 5 |
| Area per type (`area.sam2.*`) | 11 | 4 |
| Probe concepts (`probe.*`) | 21 | 20 |
| Total (under `max_targets` = 100) | 45 | 31 |

All 21 cabin and all 20 parking concepts were model-ready targets.

**Past-only inputs** (`all_covariates`: every ready non-cal, non-target column):

| Family | Admin cabin | Parking |
|---|---|---|
| `cellarea.*` | 102 | 50 |
| `count.*` (VLM counts, `yolo_cNN`, `sam2`) | 78 | 35 |
| `cell.*` | 62 | 19 |
| `area.*` (box-based, other thresholds) | 55 | 20 |
| `rel.*` | 45 | 18 |
| `emb.*` (4 components + 2 drifts) | 6 | 6 |
| `rollup.*` | 6 | 5 |
| `img.*` | 3 | 3 |
| `cat.*` | 1 | 2 |
| `scene.*` not forecast | 0 | 1 |
| Total | 358 | 159 |

**Frames handed to Chronos-2.**

| | Admin cabin | Parking |
|---|---|---|
| Context rows | 487 | 487 |
| Context columns | 457 [= `id` 1 + `timestamp` 1 + 45 targets + 358 covariates + 7 `cal.*` + 45 `prof.*`] | 230 [= 2 + 31 + 159 + 7 + 31] |
| Future rows | 16 [= `horizon`] | 16 |
| Future columns | 54 [= `id` + `timestamp` + 7 `cal.*` + 45 `prof.*`] | 40 [= 2 + 7 + 31] |
| Missing values in the context | 0.36% | 0.34% |
| Chronos-2 time | 0.29 s | 0.09 s |

In the cabin context, 357 of the 358 covariates are numeric and one is text (`cat.*`, cast to object with `None`
for missing).

**A profile value.** On Wed 18 Nov at 10:00 the measured `scene.people_yolo` was 4 and `prof.scene.people_yolo` was
3.33. History holds no other Wednesday, so level 1 (needs 2 other same-weekday days) fails; level 2 (working days at
10:00, own day left out) uses Fri, Mon and Tue.

**The call.** `predict_df(context, future_df=future, prediction_length=16, quantile_levels=[0.1, 0.5, 0.9],
id_column="id", timestamp_column="timestamp", target=[45 names], cross_learning=False)`.

**Output.** One row per target per future slot: cabin 720 rows [= 45 x 16], parking 496 [= 31 x 16], with columns
`timestamp, target, step, q10, q50, q90`. First cabin rows for `scene.people_yolo`: 10:45 step 1 (1.26, 2.39, 3.66),
11:00 step 2 (0.65, 1.85, 3.26), 11:15 step 3 (1.62, 2.97, 4.31). `scene.*`, `count.*` and `area.*` targets are then
clipped at 0; the 21/20 `probe.*` targets are not.

**Storage.** One `forecast_runs` row per space with `context_slots = 487`, `horizon = 16`, `variant = all_profile`,
`covariates_json` = 358 (or 159) names + `"cal.*"` + `"prof.*"`. `forecasts` rows: up to 2160 for the cabin
[= 720 x 3 quantiles] and 1488 for parking [= 496 x 3], fewer if any value is NaN.

**Lead buckets for these runs.** Steps 1-4 (10:45-11:30) -> `0-1 h`, 5-8 -> `1-2 h`, 9-16 -> `2-4 h`.

**Backtest on that history (from the doc).** 46 origins over 5 days; skill against `seasonal_mean`: cabin 0.417,
parking 0.350; persistence: 0.299 and 0.169.


## 4. Showing, checking and testing

This section covers the code that shows the data (the dashboard), the code that collects human ground truth (the labelling tool), the code that checks model output against that truth or against itself (`sbi.eval`), the one-off detector benchmark scripts, and the test suite.

| Part | Files | Writes to the database? |
|---|---|---|
| Dashboard data layer | `src/sbi/dashboard/data.py` | No. Opens the database with `mode=ro`. |
| Dashboard app | `src/sbi/dashboard/app.py`, `.streamlit/config.toml`, `docs/dashboard-guide.md` | No. |
| Labelling tool | `src/sbi/labeling/core.py`, `src/sbi/labeling/app.py` | Yes: `label_frames`, `labels`. Also runs migrations once at start. |
| Evaluation | `src/sbi/eval/vlm_eval.py`, `thresholds.py`, `stability.py` | No. |
| Detector benchmark | `scripts/detector_bench/bench.py`, `score.py` | No. Opens `data/sbi.db` with `mode=ro`; writes JSON files under `data/exports/detector_bench/`. |
| Tests | `tests/conftest.py`, `tests/test_*.py` | Only temporary databases under pytest's `tmp_path`. |

### 4.1 Dashboard data layer (`src/sbi/dashboard/data.py`)

The module docstring states the two rules of the layer: every function returns plain DataFrames, lists or dicts so it can be tested without Streamlit, and nothing in it writes. All times shown are local (IST). Table timestamps are already local and naive. UTC columns from the database (`frames.ts_utc`) are converted here with `sbi.timeutil.get_zone`.

#### 4.1.1 Connection and time helper

| Function | Purpose and behaviour |
|---|---|
| `data.py::connect_readonly(path)` | Opens `file:{path}?mode=ro` as a URI with `timeout=30` and sets `row_factory = sqlite3.Row`. Any write raises `sqlite3.OperationalError` ("attempt to write a readonly database"); `tests/test_dashboard.py::test_latest_forecast_and_the_read_only_connection` checks this. |
| `data.py::_local(ts_utc, tz)` | Converts an ISO UTC string to `"%Y-%m-%d %H:%M"` in the space's zone. Returns `"-"` for `None` or an empty string. |

#### 4.1.2 Functions kept from the v1.0 dashboard

| Function | Inputs | Output |
|---|---|---|
| `data.py::health(conn, spaces, now_utc)` | connection, list of `SpaceConfig`, aware UTC `datetime` | One row per space about the last 24 hours. Not called by `app.py`; covered by tests only. |
| `data.py::log_problems(path, *, limit=15)` | path of the tick log | The last `limit` lines that contain `" WARNING "` or `" ERROR "`, right-stripped, newest last. `[]` if the file does not exist. The file is read with `errors="replace"`. |
| `data.py::latest_forecast(conn, space_id)` | connection, space id | `(run, forecast)`. `run` is the newest `forecast_runs` row (highest `run_id`) as a dict plus two parsed keys: `targets` (from `targets_json`) and `covariates` (from `covariates_json`). `forecast` is `sbi.store.forecasts.load_forecast(conn, run_id)`: a wide frame with `target, timestamp, step, q10, q50, q90, ...` (one `qNN` column per stored quantile, `NN = round(quantile*100)`). With no run: `(None, empty DataFrame)`. |
| `data.py::chart_data(table, forecast, target, *, days)` | the space table, the wide forecast, a column name, a window in days | Long frame with columns `timestamp, kind, value, low, high`. See below. |
| `data.py::backtest_overall(exports, space_id)` | exports folder, space id | Reads `{exports}/{space_id}_backtest.csv` (written by `sbi backtest`) and returns `sbi.forecast.metrics.overall()` of it: one row per (forecaster, variant) with `targets, points, skill, wql, coverage`, sorted by skill. Empty frame if the file is missing. The import of `overall` is lazy. |

SQL run by `health()` for each space (`since` = `now_utc - 24 h` as ISO text):

```sql
SELECT COUNT(*) n, SUM(capture_status='ok') ok, SUM(processed_utc IS NOT NULL) processed,
       SUM(vlm_raw IS NOT NULL AND vlm_raw NOT LIKE '{"error"%') vlm, SUM(segmenter_version IS NOT NULL) outlined,
       SUM(EXISTS(SELECT 1 FROM frame_embeddings e WHERE e.frame_id=f.frame_id)) embedded,
       SUM(EXISTS(SELECT 1 FROM frame_scores c WHERE c.frame_id=f.frame_id)) scored, MAX(ts_utc) last
FROM frames f WHERE space_id=? AND ts_utc >= ?;
SELECT origin_local, model FROM forecast_runs WHERE space_id=? ORDER BY run_id DESC LIMIT 1;
```

`health()` output columns:

| Column | Value |
|---|---|
| `space` | space id |
| `last photo (IST)` | `_local(MAX(ts_utc))`. This includes failed captures (the test shows a failed 11:45 capture as the last photo). |
| `photos 24 h` | `"{ok} / {24*60 // cadence_minutes}"`, for example `"3 / 96"` |
| `failed captures` | `n - ok` |
| `processed`, `VLM`, `outlines`, `image vector`, `probes` | share of `ok` frames that reached the stage, formatted `"{:.0%}"`; `"-"` when `ok` is 0 |
| `latest forecast (IST)` | `origin_local` with `T` replaced by a space, first 16 characters; `"-"` with no run |
| `forecaster` | `forecast_runs.model`; `"-"` with no run |

`chart_data()` details:

- Measured part: present only when `target` is a table column. It takes `table[target]` from `table.index.max() - days` to the end (both ends included, so one day of 15-minute slots gives 97 rows). `value` is the column as float; `low` and `high` are NaN.
- Forecast part: present only when the forecast frame is not empty, has rows for `target`, and has a `q50` column. Quantile columns are the columns named `q` + digits, sorted as strings. `value = q50`, `low` = the first of those names, `high` = the last.
- With neither part, it returns an empty frame with the five columns.

#### 4.1.3 Readable names: `SOURCE_NAMES`, `KIND_NAMES`, `_source()`, `label()`

`SOURCE_NAMES` maps source tokens to words: `yolo` -> `detector`, `vlm` -> `VLM`, `sam2` -> `outline`, `geometry` -> `geometry`.

`KIND_NAMES` maps a column family (the first part of a column name) to a heading:

| Family | Heading |
|---|---|
| `scene` | Scene totals |
| `count` | Counts per object type |
| `rollup` | Counts per object group |
| `area` | Area covered per type |
| `cell` | Counts per grid cell |
| `cellarea` | Area per grid cell |
| `rel` | Relations between objects |
| `img` | Image statistics |
| `cat` | Scene fields (text) |
| `emb` | Image vector |
| `probe` | Scene description scores |
| `cal` | Calendar |
| `feat` | Other features |

`data.py::_source(token)` splits the token at the first `_c`. If the part after `_c` is all digits, it returns `"{name}, confidence 0.{digits}+"` (`yolo_c05` -> `detector, confidence 0.05+`; `sam2_c10` -> `outline, confidence 0.10+`). Otherwise it returns `SOURCE_NAMES.get(token, token)`.

`data.py::label(column)` splits the column at `.` and applies the first matching rule:

| Rule | Example input | Example output |
|---|---|---|
| `scene.<name>_<source>` with a known source and a non-empty name | `scene.people_yolo` | `people (detector)` |
| other `scene.<x>` | `scene.count_disagree` | `count disagree` (underscores to spaces) |
| `count` or `rollup` with 3 parts | `count.vlm.motorcycle` | `motorcycle count (VLM)` |
| `area` with 3 parts | `area.sam2_c10.person` | `person area share (outline, confidence 0.10+)` |
| `cell` or `cellarea` with 4 parts | `cellarea.sam2.c3r2.chair` | `chair area in cell c3r2 (outline)`; `cell` gives `... count in cell ...` |
| `rel` with 5 parts | `rel.vlm.person.sitting_on.chair` | `person sitting on chair (VLM)` |
| `probe` | `probe.siglip2.screens_on` | `'screens on' score` |
| `emb` | `emb.dinov2-base.pc1` | `image vector pc1` |
| `img`, `cat`, `cal` | `img.brightness` | `brightness` (last part, underscores to spaces) |
| anything else | `odd` | `odd` (unchanged) |

#### 4.1.4 Cache key: `table_stamp()`

`data.py::table_stamp(conn, space_id)` returns a 4-tuple that changes exactly when the space's table or its stored forecasts could change:

```sql
SELECT COALESCE(MAX(frame_id), 0), COUNT(processed_utc), COALESCE(MAX(processed_utc), '') FROM frames WHERE space_id=?;
SELECT COALESCE(MAX(run_id), 0) FROM forecast_runs WHERE space_id=?;
```

The tuple is `(max frame_id, number of read frames, max run_id, max processed_utc)`.

| Event | Element that changes |
|---|---|
| a new photo row (any capture status) | 1st |
| a photo read for the first time | 2nd and 4th |
| a photo read again (`sbi reprocess`) | 4th (`processed_utc` is rewritten) |
| a new forecast run | 3rd |

For a space with no frames the tuple is `(0, 0, 0, "")`. The app uses `stamp[0] == 0` to detect "no photos stored yet". The app passes the tuple as an argument to its cached functions, so it is part of the cache key (section 4.2.3).

#### 4.1.5 Values shown on the "Now" and "Forecast vs actual" tabs

| Function | Purpose and behaviour |
|---|---|
| `data.py::usual(series, when, *, before=None)` | Mean of `series` at the same clock time on earlier days. NaNs are dropped. With `before`, only rows whose calendar day is earlier than `before`'s day are used. The rest is grouped by minute of day (`hour*60 + minute`) and averaged, then looked up at each timestamp of `when`. Returns a float Series indexed by `when`; NaN where no row has that clock time. |
| `data.py::summary(table, forecast, targets, *, leads=(4, 8, 16), step_minutes=15)` | One row per target (see below). |
| `data.py::series_chart(table, forecast, target, *, hours)` | `chart_data(..., days=hours/24)` plus a third kind, `usual`. Returns `chart_data`'s result unchanged when it is empty or when `target` is not a table column. Otherwise `when` = the sorted unique timestamps of the measured and forecast rows, and `usual(table[target], when, before=table.index.max())` with NaNs dropped is appended as `kind="usual"` rows (low/high NaN). So the usual line uses every day before the table's last day and spans both the shown history and the forecast window. |
| `data.py::lead_history(conn, space_id, table, target, step, skip_runs=())` | Every stored forecast of `target` made `step` slots ahead, next to what was measured then (see below). |

`summary()` row construction:

1. Targets that are not table columns, or whose numeric values are all NaN, are skipped.
2. `at` = the last timestamp with a value; `now` = that value; `usual` = `usual(s, [at], before=at)` (earlier days only).
3. Forecast rows of the target are taken. Quantile columns are sorted by their number.
4. For each lead `L` in `leads`, the column name is `"+{L*step_minutes//60} h"` when `L*step_minutes` is a whole number of hours, else `"+{L*step_minutes} min"`. With the defaults the names are `+1 h`, `+2 h`, `+4 h`.
5. The value is `q50` of the single row with `step == L`. `"{name} low"` is the lowest quantile and `"{name} high"` the highest. All three are NaN when there is not exactly one such row or there is no `q50`.

Output columns: `column, series (label()), at, now, usual, +1 h, +1 h low, +1 h high, +2 h, ..., +4 h high`.

`lead_history()` runs:

```sql
SELECT f.run_id, f.ts_local, f.quantile, f.value FROM forecasts f JOIN forecast_runs r USING(run_id)
WHERE r.space_id=? AND f.target=? AND f.step=?
```

Then:

1. Rows of runs in `skip_runs` are removed. The app passes `sbi.forecast.accuracy.outdated_runs()`: runs for which more than 5% of the photos taken before the run were read after the run was made.
2. When several runs predicted the same `ts_local` at this lead, only the newest run (highest `run_id`) is kept.
3. The rows are pivoted to one column per quantile. Without a 0.5 column the result is empty.
4. `forecast` = the 0.5 quantile, `low` = the lowest quantile column, `high` = the highest.
5. `measured` = `table[target]` reindexed at each timestamp. It is NaN until that time is in the table.
6. Sorted by timestamp. Columns: `timestamp, forecast, low, high, measured`. An empty result still has these columns.

#### 4.1.6 Values shown on the "Pipeline" and "All dimensions" tabs

`STAGES` is a list of (stage name, SQL condition on `frames f`):

| Stage | Condition |
|---|---|
| `captured` | `capture_status='ok'` |
| `read by detector` | `processed_utc IS NOT NULL` |
| `read by VLM` | `vlm_raw IS NOT NULL AND vlm_raw NOT LIKE '{"error"%'` |
| `outlined` | `segmenter_version IS NOT NULL` |
| `image vector` | `EXISTS(SELECT 1 FROM frame_embeddings e WHERE e.frame_id=f.frame_id)` |
| `description scores` | `EXISTS(SELECT 1 FROM frame_scores c WHERE c.frame_id=f.frame_id)` |

| Function | Purpose and behaviour |
|---|---|
| `data.py::funnel(conn, space, table)` | Counts one photo per slot: the frame ids in `table["frame_id"]` (the photo the table uses for each slot). First row: `slots since first photo` = `len(table)`, share 1.0. Then one row per stage: `SELECT COUNT(*) FROM frames f WHERE frame_id IN (ids) AND <condition>`, with `share = photos / slots` (NaN when there are no slots). With no frame ids every stage is 0. Columns: `stage, photos, share`. |
| `data.py::capture_grid(table)` | One row per slot. `status` starts as `ok`. It becomes `VLM failed` where `vlm_ok` is false, then `no photo` where `slot_status != "ok"`, then `not read yet` where `slot_status == "unprocessed"` (later rules win). Columns: `day` (`"%a %d %b"`), `date` (midnight), `time` (`"%H:%M"`), `minute` (minute of day), `status`. |
| `data.py::gaps(table)` | Runs of consecutive slots whose `slot_status` is neither `ok` nor `unprocessed`. A captured-but-unread photo is not a gap. The step is `index[1] - index[0]` (15 minutes if the table has one row). Consecutive bad slots are grouped where the difference equals the step. Columns: `from, to, slots`. Empty frame with those columns when there are none. |
| `data.py::tick_timing(path, *, last=96)` | Reads the tick log line by line. The first token of each line is its timestamp. A line containing `detector` and ` loaded ` sets the start; the next line containing `tick done` closes it and records the duration in seconds. A `tick done` with no open start is ignored; an unparseable timestamp (`ValueError`) is skipped. Uses the last `last` durations. Returns `{"ticks": 0}` if the file is missing or there are no durations, else `{"ticks", "median_s", "p95_s", "max_s"}`. The detector is the first model `sbi.tick.run_tick` builds, so this measures from the first model load to the end of the tick. |
| `data.py::dimension_flow(catalog, run)` | Per column family: `extracted` (catalog rows), `usable` (sum of `model_ready`), `given_as_input` (columns listed in the run's covariates; every `cal` column counts when the run lists `cal.*`), `forecast` (columns in the run's targets). Adds `what` from `KIND_NAMES`, sorts by `extracted` descending, renames `kind` to `family`, and appends a `total` row with empty `what`. With `run=None`, `given_as_input` and `forecast` are 0. |
| `data.py::dimensions(table, catalog, run, forecast)` | One row per catalog column except `cal` (see below). |
| `data.py::header_status(conn, spaces, now_utc)` | One dict per space for the header band (see below). |

`dimensions()` row fields:

| Field | Value |
|---|---|
| `dimension` | `label(column)` |
| `column`, `family` | catalog `column` and `kind` |
| `now` | the value in the last table row as float, for numeric families (all except `cat`); `None` if missing |
| `now (text)` | for `cat` columns, the last row's value as text; `None` otherwise or if missing |
| `usual now` | `usual(s, [last index], before=last index)`; `None` if the column has no values |
| `min`, `max` | over the whole table; `None` if no values |
| `+1 h forecast` | `q50` of the latest run at `step == 4` for that column (from a `target -> q50` dict); `None` if not forecast |
| `coverage` | catalog `coverage` |
| `use` | `forecast` if the column is a run target; `input to model` if in the run's covariates; otherwise `table only: {catalog reason}` |

`header_status()` SQL per space:

```sql
SELECT f.ts_utc, c.people_yolo, c.vehicles_yolo FROM frames f LEFT JOIN scene_state c USING(frame_id)
WHERE f.space_id=? AND f.capture_status='ok' AND f.processed_utc IS NOT NULL ORDER BY f.ts_utc DESC LIMIT 1;
SELECT origin_local FROM forecast_runs WHERE space_id=? ORDER BY run_id DESC LIMIT 1;
```

Each entry has `space_id, name, photo, age_min, people, vehicles, forecast, state`:

- `photo`: local `HH:MM` of the latest captured and read photo.
- `age_min`: minutes since that photo, as an int, never below 0.
- `people`: `scene_state.people_yolo`.
- `vehicles`: `scene_state.vehicles_yolo`, only when `space.space_type == "parking"`; otherwise `None`.
- `forecast`: characters 11 to 16 of the latest `origin_local` (`HH:MM`, no date), or `None`.
- `state`: `live` when the age is at most 20 minutes, `late` at most 60, `stopped` above 60. With no read photo the state is `no photos` and every photo field is `None`.

#### 4.1.7 SQL summary

| Function | Tables read |
|---|---|
| `health` | `frames`, `frame_embeddings`, `frame_scores`, `forecast_runs` |
| `latest_forecast` | `forecast_runs`, `forecasts` (via `load_forecast`) |
| `table_stamp` | `frames`, `forecast_runs` |
| `lead_history` | `forecasts`, `forecast_runs` |
| `funnel` | `frames`, `frame_embeddings`, `frame_scores` |
| `header_status` | `frames`, `scene_state`, `forecast_runs` |
| `log_problems`, `tick_timing` | `data/logs/tick.log` (no SQL) |
| `backtest_overall` | `data/exports/{space}_backtest.csv` (no SQL) |

### 4.2 Dashboard app (`src/sbi/dashboard/app.py`)

#### 4.2.1 Launch

`scripts/dashboard.sh` changes to the repository root and sets `TZ=Asia/Kolkata` and `PYTHONPATH=src`. It then runs `.venv/bin/streamlit run src/sbi/dashboard/app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false`. The page is at http://localhost:8502 and is reachable from this machine only. `scripts/install-dashboard-service.sh` installs it as a user service (described in another section). The project root is the current directory, or `SBI_ROOT` if set (`load_settings(os.environ.get("SBI_ROOT"))`).

`st.set_page_config(layout="wide", page_title="SBI", initial_sidebar_state="collapsed")`. A small CSS block sets the top and bottom padding of the main container and tabular numerals for metric values.

#### 4.2.2 Colours and constants

| Name | Value | Use |
|---|---|---|
| `INK` | `#2f2f2c` | measured line on the accuracy chart; "Average measured value" tile bar |
| `MUTED` | `#a3a29a` | the usual line; the vertical "latest photo" rule; muted text in the legend |
| `BLUE` | `#2a78d6` | default series colour; forecast line and band on the accuracy chart; blue tiles |
| `GRID` | `#e6e5e0` | grid, axis domain and tick colour |
| `SLATE` | `#52514e` | default tile bar; colour for any non-headline series |
| `SERIES` | `#2a78d6`, `#eb6834`, `#1baf7a`, `#6250d6` | one colour per headline series (blue, orange, green, violet); the tile and its chart share it |
| `GOOD`, `WARN`, `BAD` | `#0ca30c`, `#eda100`, `#d03b3b` | reserved for state: captured / degraded / missing |
| `STATUS` | `ok` `#b5dfc0`, `not read yet` `#8c8b84`, `VLM failed` = `WARN`, `no photo` = `BAD` | capture grid cell colours |
| `USE_TINT` | `forecast` `#dbe9fb`, `input to model` `#d9f0e6`, `table only` `#f1f0ec` | cell backgrounds for where a dimension goes (blue, green, grey) |
| `STATE_DOT` | `live` `#35c46a`, `late` `#f0b429`, `stopped` `#ef6461`, `no photos` `#8d99a6` | header dots |
| `HOURS` | `24 h`: 24, `48 h`: 48, `7 days`: 168 | history choices on tab 1 |
| `LEADS` | `1 h ahead`: 4, `2 h ahead`: 8, `4 h ahead`: 16 | lead choices on tab 2, in slots |
| `WHOLE_THINGS` | `scene.`, `count.`, `rollup.`, `cell.` | prefixes shown with one decimal |
| `TABS` | `Now and next 4 hours`, `Forecast vs actual`, `Pipeline`, `All dimensions`, `Guide` | tab labels |
| `GUIDE` | `<repo>/docs/dashboard-guide.md`, found from the code location (`Path(__file__).parents[3]`), not from `SBI_ROOT` | guide tab |

The header band uses its own colours: background `#1d2733`, separator `#34414f`, secondary text `#aab4c0` and `#d5dbe2`.

#### 4.2.3 Setup, caching and cache keys

| Function | Decorator | Behaviour |
|---|---|---|
| `app.py::_setup()` | `@st.cache_resource` | Loads settings and a dict `space_id -> SpaceConfig` from `discover_space_files(settings.root)`. Built once per server process. |
| `app.py::_table(space_id, stamp)` | `@st.cache_data(max_entries=4, show_spinner="Reading the database ...")` | Opens its own read-only connection and returns `(table, catalog)` from `sbi.forecast.build.table_for(conn, settings, space, max_slots=settings.forecast.max_context)`. `max_context` defaults to 8192 slots. The connection is closed in `finally`. |
| `app.py::_points(space_id, stamp)` | `@st.cache_data(max_entries=4, show_spinner=False)` | Opens its own read-only connection and returns `sbi.forecast.accuracy.live_points(conn, space_id, table)` for the cached table. `live_points` leaves out outdated runs. |

The cache key of both data caches is the argument tuple `(space_id, stamp)`, with `stamp = data.table_stamp(conn, space_id)`. A reload or the timed refresh with no new photo, read or forecast hits the cache. Each cache keeps at most four entries. `settings` and `spaces` are module globals and are not part of the key. `table_for` also uses a per-photo feature cache file (`table_features.db` in the settings cache folder), described in another section.

At start-up the module calls `_setup()` and opens and closes one read-only connection to check the database. `SbiError` or `OSError` shows `st.error("Setup problem: ...")` and stops the script.

#### 4.2.4 Formatting helpers

| Function | Behaviour |
|---|---|
| `app.py::num(v, column="")` | `"-"` for `None` or NaN. For columns starting with a `WHOLE_THINGS` prefix, or any value with absolute size 100 or more: one decimal with a trailing `.0` removed. Otherwise three significant figures (`.3g`). |
| `app.py::band(v, lo, hi, column="")` | `"-"` when `v` is NaN, else `"{num(v)}  ({num(lo)} to {num(hi)})"`. |
| `app.py::tile(where, label_, value, note=None, color=SLATE)` | HTML block: a 4 px left border in `color`, the label (0.85 rem), the value (2 rem, tabular numerals), and a note line (0.8 rem; kept at a minimum height when empty). |
| `KEY` | HTML legend placed above series charts: `── measured`, `┄┄ forecast, shaded = 10% to 90%`, `···· usual at this time of day` (muted), `| latest photo` (muted). |
| `app.py::header(statuses, clock)` | Builds the header band HTML (section 4.2.7). |
| `app.py::state_colour(share)` | `GOOD` at 0.98 or more, `WARN` at 0.90 or more, else `BAD`. |
| `app.py::headline(space, run, table)` | The series shown first (below). |
| `app.py::table_tz(space)` | `get_zone(space.tz)`; used for the page clock. |

`headline()`:

1. `targets` = the latest run's targets. With no run: every table column starting with `scene.`.
2. `named` = entries of `space.forecast_targets` that appear literally in `targets`. Patterns such as `count.yolo.*` never match literally.
3. `counts` = targets starting with `count.` that are not in `named`, sorted by descending mean over the whole table.
4. Returns the first four of `named + counts`, or the first four targets if that list is empty.

#### 4.2.5 Chart construction (Altair)

`app.py::_plain(chart, height)` gives every chart the same look: the given height, padding `{left 4, right 28, top 4, bottom 4}` (room on the right so the last time label is not cut), grid/domain/tick colour `GRID`, label colour `#52514e`, label size 11, and no view frame.

`app.py::series_figure(cd, title, now, height=240, color=BLUE)` draws one `series_chart()` frame in four layers, in this order:

| Layer | Mark | Encoding |
|---|---|---|
| area | `mark_area(opacity=0.16, color=color)` on rows with `kind == "forecast"` | `y=low`, `y2=high`: the forecast band |
| rule | `mark_rule(color=MUTED, strokeWidth=1)` on a one-row frame | `x=now` (the table's last slot) |
| lines | `mark_line(strokeWidth=1.8)` | `y=value` with `zero=True`. Colour by `kind`: measured = `color`, usual = `MUTED`, forecast = `color`. Dash by `kind`: measured `[1,0]` (solid), usual `[1,3]` (dotted), forecast `[5,3]` (dashed). No legends. |
| hover | `mark_circle(size=70, opacity=0)` | Invisible points with tooltip: time (`%a %d %b %H:%M`), kind, value, low, high (`.3g`) |

The x axis is `timestamp:T`, format `%a %H:%M`, labels horizontal, about 6 ticks. The title is anchored at the start, font size 13.

`app.py::lead_figure(h, title)` draws a `lead_history()` frame, height 340:

| Layer | Mark | Encoding |
|---|---|---|
| area | `mark_area(opacity=0.18, color=BLUE)` on all rows | `low` to `high` |
| lines | `mark_line(strokeWidth=1.8)` on a long frame (`measured` and `forecast` stacked, NaN rows dropped) | colour `measured` = `INK`, `forecast` = `BLUE`; legend at the top, no title |
| hover | `mark_circle(size=70, opacity=0)` at `y=forecast` | tooltip: time, measured, forecast, low, high (`.3g`) |

The x axis is formatted `%a %H:%M` with about 8 ticks. Forecasts whose time has not come yet have no measured point, so the blue line continues past the end of the dark line.

`app.py::grid_figure(g)` draws `capture_grid()` as a heat map:

- `mark_rect` with a 1 px `#fcfcfb` stroke.
- x = `minute` (ordinal), axis title "time of day", labels every 180 minutes, written `H:00`.
- y = `day`, sorted by date.
- colour = `status` with the `STATUS` domain and colours; legend at the top.
- tooltip: day, time, status.
- height `120 + 34 * days`; no view frame.

#### 4.2.6 Page refresh and the per-refresh connection

```python
@st.fragment(run_every=60)
def page() -> None:
    conn = data.connect_readonly(settings.db_path)
    try:
        _page(conn)
    finally:
        conn.close()
```

The whole page body is one fragment that Streamlit runs again every 60 seconds. The connection is opened inside each run and closed at its end. The docstring gives the reason: Streamlit runs the timed refresh in a different thread from the first page load, and an SQLite connection may only be used in the thread that created it. A connection kept for the session would fail with "SQLite objects created in a thread can only be used in that same thread". `tests/test_dashboard.py::test_the_app_opens_its_database_connection_inside_each_refresh` checks this rule with `ast`: no module-level assignment (or assignment in a module-level `try`) calls `connect_readonly`, and `page()` assigns one and calls `conn.close()`.

Each refresh does cheap queries every time: header status, `table_stamp`, the latest forecast, the latest photo path, outdated runs, lead history, funnel, and the log file reads. The table and the live accuracy points are rebuilt only when the stamp changes.

#### 4.2.7 Page structure

Query parameters: `?space=<space_id>&tab=<now|accuracy|pipeline|all|guide>`. An unknown or missing `space` selects the first space. An unknown `tab` selects `now`. These only set the initial selection (`first_space`, `first_tab`).

Order of elements in `app.py::_page(conn)`:

1. **Clock.** `clock` = now in UTC. `local` = the same time in the first space's zone.
2. **Header band** (`header(data.header_status(...), local)`), dark background:
   - Top line: title "Space Behaviour Intelligence", subtitle "a photo every 15 minutes, read into numbers, forecast 4 hours ahead", and on the right the clock as `%a %d %b %Y, %H:%M IST`.
   - One cell per space: a coloured dot (`STATE_DOT`), the space name in bold, and the state word (`live`, `late`, `stopped`, `no photos`).
   - Detail line under each space: `no photos yet` when there is no read photo. Otherwise `photo HH:MM (N min ago)`, then `· N vehicles, N people` (vehicles for parking only; singular forms "1 vehicle", "1 person"), then `· forecast HH:MM` when a run exists.
3. **Space switch.** A horizontal `st.radio` in the left of two columns (`[3, 2]`). Its label is hidden. It shows `spaces[s].name` and starts at `first_space`.
4. **Empty-space and error checks.** If `stamp[0]` is 0: `st.info("No photos stored yet for {name}.")` and the run ends. If `_table()` raises `SbiError`: `st.error("Could not read {name}: ...")` and the run ends.
5. **Status line** (right column, right-aligned): `showing <name> up to <last slot %a %d %b %H:%M> | refreshes every minute, last at HH:MM:SS`.
6. **Tabs.** `st.tabs(TABS, default=TABS[first_tab])`. The Guide tab is filled first.

Widget keys: `pick_{space}`, `acc_{space}`, `fam_{space}` and `plot_{space}` are per space. `hours`, `lead`, `use` and `search` are shared by all spaces.

#### 4.2.8 Tab 1: Now and next 4 hours

| Order | Element | Fed by |
|---|---|---|
| 1 | `heads = headline(space, run, table)` | `latest_forecast`, cached table |
| 2 | Left column (2 of 5): the newest photo with `capture_status='ok'` (`SELECT image_path FROM frames WHERE space_id=? AND capture_status='ok' ORDER BY ts_utc DESC LIMIT 1`), shown when the file exists under `settings.root`. The photo may not be read yet. | SQL |
| 3 | Right column (3 of 5): one tile per headline series. Label = `label()`. Value = `num(now)`. Note = `"+d vs usual u"` (`+` shown when now >= usual), or no note when usual is NaN. Bar colour = the series' `SERIES` colour, fixed per space (`colour = {heads[i]: SERIES[i]}`). | `data.summary(table, forecast, heads)` |
| 4 | Right column: forecast table, height 250. Columns: `series`, `now`, `usual at this time`, and for each of `+1 h`, `+2 h`, `+4 h` a column `"<lead>  (10% to 90%)"` with `band(median, low, high)`. Rows = the run's targets, or `heads` with no run. | `data.summary(table, forecast, targets or heads)` |
| 5 | `st.info("No forecast stored yet for this space.")` when there is no run | `latest_forecast` |
| 6 | Controls: multiselect **Series** (options: targets, or heads with no run; default: heads; labels from `label()`) and horizontal radio **History** (`24 h`, `48 h`, `7 days`) | `HOURS` |
| 7 | `KEY` legend | - |
| 8 | One chart per picked series, alternating between two columns. Colour = the series' `SERIES` colour, or `SLATE` for any series that is not a headline. Empty frames are skipped. | `data.series_chart(table, forecast, target, hours=...)` -> `series_figure` |

#### 4.2.9 Tab 2: Forecast vs actual

With no run targets: `st.info("No forecast stored yet for this space.")`. Otherwise:

| Order | Element | Fed by |
|---|---|---|
| 1 | Selectbox **Series** (run targets, `label()`) and horizontal radio **Forecast made** (`1 h ahead`, `2 h ahead`, `4 h ahead`) | run targets, `LEADS` |
| 2 | Caption, when some runs are outdated: "N earlier forecast runs are not scored here: the photos behind them have since been read again (a detector change) ..." | `sbi.forecast.accuracy.outdated_runs(conn, space_id)` |
| 3 | `h = lead_history(..., LEADS[lead], skip_runs=old_runs)`; `done` = rows with a measured value. With none: `st.info("No forecast at this lead has a measured outcome yet.")` | `data.lead_history` |
| 4 | Four tiles: **Forecasts checked** (`len(done)`); **Average miss** (mean of absolute forecast minus measured, `.2g`, blue bar); **Average measured value** (`.2g`, `INK` bar); **Measured inside the shaded range** (share with low <= measured <= high, `.0%`, note "expected about 80%", blue bar) | `done` |
| 5 | Chart "<label>: forecast made <lead> vs what was measured" | `lead_figure(h)` |
| 6 | When live points exist: heading "**<label> by day of week**" with an explanation, then the per-weekday table for this series (rounded to 2; `inside band` shown as a percent) | `_points` -> `by_weekday(points, target)` |
| 7 | "**All series together.**" with an explanation of skill and coverage, then the per-lead table (rounded to 3). `model` is dropped. Renamed: `variant` -> `inputs given`, `lead` -> `how far ahead`, `runs` -> `forecast runs`, `targets` -> `series`, `points` -> `forecasts checked`, `coverage` -> `inside band`. | `by_lead(points)` |
| 8 | "**All series together, by day of week.**" table (rounded to 3) | `by_weekday(points)` |
| 9 | Per-series table, height 320. Adds `series` = `label(target)`; drops `model` and `target`. Renamed: `variant` -> `inputs given`, `n` -> `forecasts checked`, `mae` -> `average miss`, `persist_mae` -> `miss if nothing changed`, `coverage` -> `inside band`. | `by_target(points)` |
| 10 | When `data/exports/{space}_backtest.csv` exists: "**Backtest** (`sbi backtest`): forecasts replayed from past moments, skill vs the weekday-by-time average." and the table (rounded to 3) | `data.backtest_overall(settings.resolve("data/exports"), space_id)` |

The scoring functions (`live_points`, `outdated_runs`, `by_lead`, `by_target`, `by_weekday`) belong to `sbi.forecast.accuracy` and are described in the forecasting section.

#### 4.2.10 Tab 3: Pipeline

| Order | Element | Fed by |
|---|---|---|
| 1 | Tile **Photos captured**: `"{captured} / {slots}"`, note `"{share:.1%} of slots"`, bar colour `state_colour(share)` | `data.funnel` rows 1 and 0 |
| 2 | Tile **Numbers per photo**: total `extracted` | `data.dimension_flow` total row |
| 3 | Tile **Usable series**: total `usable`, note "seen often enough, not constant" | same |
| 4 | Tile **Given to Chronos-2**: total `given_as_input + forecast`, note "N forecast, M inputs", blue bar | same |
| 5 | Tile **Time per 15-min run**: median seconds, note "slowest N s of last T". `-` and no note when there are no ticks. | `data.tick_timing(<logs>/tick.log)` |
| 6 | Left (2 of 5): "**Photos through each stage**" table; `share` shown as a progress bar titled "share of slots" | `data.funnel` |
| 7 | Left: "**Slots without a photo: N**" (sum of `slots`), then the gap table with `from`/`to` formatted `%a %d %b %H:%M` when there are gaps | `data.gaps` |
| 8 | Right (3 of 5): "**What is extracted and where it goes** (latest forecast run, <origin>, inputs '<variant>')". Table height 495. `given_as_input` is renamed `input to model`. Non-zero cells in `input to model` get the green tint; non-zero cells in `forecast` get the blue tint. | `data.dimension_flow(catalog, run)` |
| 9 | "**Every 15-minute slot**" grid chart | `data.capture_grid` -> `grid_figure` |
| 10 | "**Warnings and errors in the run log: N**" and the lines in a code block | `data.log_problems(<logs>/tick.log)` (last 15) |

`<logs>` is `settings.resolve(settings.paths.logs)`, by default `data/logs`.

#### 4.2.11 Tab 4: All dimensions

1. `dims = data.dimensions(table, catalog, run, forecast)`. Columns `now`, `usual now`, `min`, `max` and `+1 h forecast` are rounded to 4 significant figures (`float(f"{v:.4g}")`). Missing `now (text)` is shown as an empty string.
2. Filters in three columns:
   - **Family**: multiselect of families, shown with `KIND_NAMES`.
   - **Use**: multiselect of `forecast`, `input to model`, `table only`. It matches the part of `use` before `:`.
   - **Search**: text input, placeholder "chair, motorcycle, meeting ...". Case-insensitive plain substring match on `dimension` or `column`.
3. Caption "N of M dimensions".
4. Table, height 420. The `use` column is tinted with `USE_TINT` by its prefix. `now` is a small column. `coverage` is shown as a progress bar (percent).
5. Selectbox **Plot** over the non-`cat` columns of the filtered view, shown as `"<label>   [<column>]"`. The chosen column is drawn over 7 days (`series_chart(..., hours=168)`), with `KEY`, in `series_figure(..., height=300)` with the default colour `BLUE`.

#### 4.2.12 Tab 5: Guide

The tab renders `docs/dashboard-guide.md` with `st.markdown`. If the file is missing it shows "The guide file `docs/dashboard-guide.md` is missing.". Section 4.7 lists where the guide and the code differ.

#### 4.2.13 Theme file `.streamlit/config.toml`

Streamlit reads this file from the working directory. Both launch scripts change to the repository root, so it applies to the dashboard and the labelling tool. It changes only the look.

| Key | Value |
|---|---|
| `[theme] base` | `light` |
| `primaryColor` | `#2a78d6` (same as `BLUE`) |
| `backgroundColor` | `#fcfcfb` (also the grid cell stroke) |
| `secondaryBackgroundColor` | `#f1f0ec` (same as the `table only` tint) |
| `textColor` | `#0b0b0b` |
| `font` | `sans serif` |
| `[client] toolbarMode` | `minimal` |

### 4.3 Labelling tool (`src/sbi/labeling/`)

A person counts objects in a photo without seeing the model counts, to avoid anchoring. After saving, the model counts are shown next to theirs. `scripts/label.sh` runs `src/sbi/labeling/app.py` with Streamlit on `127.0.0.1:8501`, with `TZ=Asia/Kolkata` and `PYTHONPATH=src`.

#### 4.3.1 `core.py`

| Name | Purpose and behaviour |
|---|---|
| `core.py::progress(conn, space_id)` | `(labelled, labellable)`. Labelled = `label_frames` rows of the space (usable or not). Labellable = frames with `capture_status='ok' AND processed_utc IS NOT NULL`. |
| `STRATA` | The 8 strata `(weekend, part)`: weekend in `(0, 1)`, part in `morning, afternoon, evening, night`. Weekday strata come first. |
| `core.py::stratum(dow, part_of_day)` | `(int(dow >= 5), part_of_day)`. Saturday and Sunday (`dow` 5, 6) are weekend. |
| `core.py::next_unlabeled(conn, space_id)` | Picks the next frame (below). |
| `core.py::quadrants(img)` | Cuts the photo at `h//2`, `w//2` into `Top left`, `Top right`, `Bottom left`, `Bottom right` at full resolution, without overlap, so each pixel is in exactly one quarter. |
| `core.py::save_label(conn, frame_id, counts, *, usable, labeler, note="")` | Raises `ValueError` if any count is negative. In one transaction: `INSERT OR REPLACE INTO label_frames(frame_id, usable, labeler, labeled_utc, note)` (`labeled_utc` = now UTC, seconds); `DELETE FROM labels WHERE frame_id=?`; and, only when usable, one `labels(frame_id, object_type, true_count)` row per type (zeros included). |
| `core.py::model_counts(conn, frame_id, types)` | For each type: `{"yolo": SUM(count) of observations with source 'yolo', "vlm": the same for 'vlm'}`. `vlm` is `None` when the frame's `vlm_raw` is missing or starts with `{"error"`. |
| `core.py::load_frame_image(conn, frame_id, root)` | Reads `frames.image_path` (absolute, or relative to `root`) with OpenCV. Raises `StorageError` when there is no path or the file cannot be read. |
| `core.py::draw_boxes(conn, frame_id, img)` | Copies the image and draws every `observations` row with `source='yolo'` and a `bbox_norm`, as a yellow (BGR `0,255,255`) 3 px rectangle with `"<label> <conf:.2f>"` above it. |
| `core.py::to_rgb_small(img, width=1400)` | Resizes to `width` keeping the aspect ratio and converts BGR to RGB. |

`next_unlabeled()` choice of the next photo:

1. Counts existing labels per stratum (`label_frames` joined to `frames.dow`, `frames.part_of_day`).
2. Lists unlabelled candidates (captured ok, read, no `label_frames` row) in a fixed pseudo-random order: `ORDER BY (frame_id * 7919) % 10007`. The first candidate in that order is kept for each stratum.
3. Returns the candidate of the stratum with the fewest labels. Ties go to the earlier stratum in `STRATA`; strata not in `STRATA` sort last. Returns `None` when no candidate is left.

The result is the same between sessions. The first labels cover weekday and weekend, day and night, instead of mostly empty nights or only the newest photos.

Stored data: `label_frames` (one row per handled frame; `usable` 0 means skipped as too unclear or unreadable; `labeler`, `labeled_utc`, `note`) and `labels` (primary key `(frame_id, object_type)`, `true_count`; rows only for usable frames). Both are defined in migration `002_labels.sql`.

#### 4.3.2 `app.py`

| Step | Behaviour |
|---|---|
| Setup | `_init()` (`@st.cache_resource`) loads settings, opens a connection once to run `migrate()` and closes it, and loads the spaces. Then `conn = connect(settings.db_path)` is opened at module level on every script run, so each run (each thread) has its own connection. The docstring gives the reason: an SQLite connection may only be used by the thread that opened it. `SbiError` shows "Setup problem" and stops. The per-run connection is not closed explicitly. |
| Sidebar | Selectbox **Space**; text input **Your name / initials** (default `me`); a progress bar `"{done} labelled of {total} available"`; caption "Target: about 50 per space, spread over different hours and days." |
| Types | `space.label_types`. If empty: a warning and stop. |
| Session state | `space`, `fid` (current frame), `revealed` (frame id whose model counts are shown), `last_counts`. Changing space picks a new `fid` with `next_unlabeled()` and clears `revealed`. No frame left: `st.success("No unlabelled frames left ...")` and stop. |
| Title | `"<space> \| frame <id> \| <dow_name> <ts_local YYYY-MM-DD HH:MM> IST (<part_of_day>)"` |
| Unreadable image | `st.error`, and a **Skip this frame** button that saves the frame as unusable with note `"unreadable: <error>"` and moves on. |
| Counting view | Left (3 of 4): tabs `Whole photo` (resized to 1400 px), then the four quadrant tabs, each shown at its own width (full resolution), with a caption to count dense areas per quarter and add them up. Right: form **Count what you can see** with one number input per type (0 to 200, step 1, key `"{fid}_{type}"`), a caption to count partly hidden objects, a checkbox **Too unclear to label (skip)**, a **Note** input, and a **Save** button. |
| Save | `save_label(..., usable=not unusable)`. When usable, the model counts are revealed for this frame. When unusable, it moves straight to the next frame. |
| Revealed view | Left: the photo with yellow detector boxes (caption "Yellow boxes = YOLO detections"). Right: table **You vs models** with columns `type, you, yolo, vlm` (`n/a` when the VLM failed). A **Next frame** button picks the next frame. |

`tests/test_labelling_app.py` runs the app twice in two different threads with Streamlit's `AppTest` to check that the per-run connection works.

### 4.4 Evaluation (`src/sbi/eval/`)

`sbi eval` (in `sbi/cli.py`) prints `vlm_eval.report(conn, mask_dup_iou=settings.table.mask_dup_iou)`. That report includes the threshold report. `sbi stability` runs the stability report.

#### 4.4.1 `vlm_eval.py`: detector and VLM counts against human labels

`_SQL` returns one row per usable label: `space_id, part_of_day, frame_id, object_type, true_count`, plus:

- `yolo`: `COALESCE(SUM(count), 0)` of `observations` with `source='yolo'` and the same `object_type`.
- `vlm`: the same with `source='vlm'`, but `NULL` when the frame's `vlm_raw` is `NULL` or starts with `{"error"`.

Only frames with `label_frames.usable = 1` are included.

| Function | Behaviour |
|---|---|
| `vlm_eval.py::load_labelled(conn)` | Runs `_SQL` into a DataFrame. |
| `vlm_eval.py::summarise(df, by)` | Melts `yolo` and `vlm` into `source`/`pred` and drops rows with no prediction (a failed VLM is left out, not counted as 0). `err = pred - true_count`. Groups by `by + ["source"]`: `n`, `bias` (mean signed error), `mae`, `within1` (% with abs error at most 1), `exact` (% exact), `mean_true`. Rounded to 2. |
| `vlm_eval.py::report(conn, *, mask_dup_iou=0.5)` | With no labels: `"no labels yet: run scripts/label.sh"`. Otherwise the text below. |

Report sections, in order:

1. `"<N> labelled frames"`.
2. `"== by space and object type"`: `summarise(df, ["space_id", "object_type"])`.
3. `"== people by part of day (look for night/low-light bias)"`: `summarise` of `person` rows by `space_id, part_of_day`.
4. The output of `thresholds.report(conn, min_iou=mask_dup_iou)`, appended after a blank line when it is not empty.

The module docstring says bias matters more than MAE: a detector that always undercounts at night corrupts baselines even if its error looks small.

#### 4.4.2 `thresholds.py`: a confidence threshold per object type from the labels

Every detector box down to the storage floor is stored in `detections` with its confidence and, when the segmenter ran, its outline. A count at any threshold can therefore be recomputed for every labelled photo without running a model again.

| Constant | Value |
|---|---|
| `LEVELS` | `0.05, 0.10, ..., 0.50` (10 levels) |
| `CURRENT` | `0.25`: the "current rule" `yolo@0.25` |
| `MIN_LABELS` | `10`: fewer labels for a (space, type) give no recommendation |

| Function | Behaviour |
|---|---|
| `thresholds.py::sweep(conn, *, levels=LEVELS, min_iou=0.5)` | One row per (labelled frame, object type, rule, level) with `pred` and `true` (below). |
| `thresholds.py::score(points)` | Groups by `space_id, object_type, rule, level`: `n`, `bias`, `mae`, `exact` (share 0 to 1), `mean_true`. Empty input gives an empty frame with those columns. |
| `thresholds.py::recommend(scores, *, rel_margin=0.05, abs_margin=0.05)` | One row per (space, type) (below). |
| `thresholds.py::class_conf_suggestion(rec)` | The `detector.class_conf` entries implied by the recommendations (below). |
| `thresholds.py::report(conn, *, min_iou=0.5)` | Text report; `""` when there are no usable labels. |

`sweep()`:

1. Reads the labels of usable frames: `space_id, frame_id, object_type, true_count`, the frame's `detector_min_conf` (as `floor`) and `segmenter_version` (as `seg`).
2. Reads `frame_id, raw_label, confidence, mask` from `detections` for those frames. `raw_label` in `detections` holds the detector label after the space's `detector_map`.
3. For each label and level:
   - The level is skipped when the frame's floor is `NULL` or above the level. The frame was not read that low, so the count is unknown, not zero.
   - `yolo` rule: the number of boxes of that type with confidence >= level. The `counted` flag is not used.
   - `outline` rule: only when the frame has a `segmenter_version` and every kept box has a mask. The masks are decoded, and the count is `len(drop_duplicate_masks(items, min_iou))`: same-type outlines that overlap at least `min_iou` count once.

`recommend()` for each (space, type) group:

| Field | Value |
|---|---|
| `n` | the largest `n` over the group's rows |
| `mean_true` | from the first row, rounded to 2 |
| `current_mae`, `current_bias` | of `yolo@0.25`, NaN if that row is missing |
| `best`, `best_mae` | the row with the lowest MAE (first such row on ties) |
| `recommended`, `rec_mae`, `rec_bias`, `why` | chosen by the rules below |

Rules, applied in order:

1. **Too few labels.** `n < MIN_LABELS`: recommended `-`, why `"only n labels (< 10)"`.
2. **Margin.** The acceptable rows are those with `mae <= best_mae * (1 + rel_margin) + abs_margin`, by default `best_mae * 1.05 + 0.05`.
3. **Keep current.** If `yolo@0.25` exists and is acceptable, the recommendation is `yolo@0.25` with why `"current threshold is as good as the best: keep"`. A tie is no evidence for a change.
4. **Otherwise** the acceptable rows are sorted by (`rule == "yolo"`, level), both descending. The first row is chosen: any plain `yolo` option comes before any `outline` option, and within a rule the highest (strictest) level wins. Why: `"strictest option within a small margin of the best MAE"`.

`class_conf_suggestion()` keeps only recommendations starting with `yolo@`. It groups them by object type over all spaces. It returns `{type: level}` only when all spaces agree on one level and that level differs from 0.25, because `detector.class_conf` is a global setting. Outline picks, `-`, the current threshold, and disagreeing spaces give no entry.

`report()` contains:

1. A header line explaining `yolo@t`, `outline@t` and the current rule.
2. The recommendation table rounded to 2.
3. When there are suggestions: `suggested config/pipeline.yaml -> detector.class_conf: {type: 0.10, ...}`.
4. For each `outline@` recommendation: `"<space>/<type>: outline de-duplication beats plain counting; use the table column count.sam2_cNN.<type> as that series"`, where `NN` is the level times 100, two digits.

#### 4.4.3 `stability.py`: steadiness without labels

In a window where the real scene does not change (for example a car park at night), every slot-to-slot change in a count is measurement noise. A count that is consistently biased is usable for forecasting; one that jitters is not. The report measures steadiness, not correctness.

`KINDS = ("scene", "count", "area", "rollup", "emb", "probe")`. Other families (`cat`, `cell`, `cellarea`, `rel`, `img`, `cal`) are not reported.

`stability.py::stability(table, *, start=None, end=None, kinds=KINDS)`:

1. **Window.** `table.loc[start:end]` with `pd.Timestamp` bounds in local time. A missing bound leaves that side open.
2. **Columns.** Every table column whose first part is in `kinds`.
3. **Steps.** `step = |diff|` of the column inside the window. A step is NaN when either neighbour is missing, so only pairs of consecutive slots that both have a value are compared. `pairs` = the number of such steps. Columns with no pairs are skipped.
4. **Values per column:**

| Field | Formula |
|---|---|
| `noise` | mean of `step` (mean absolute change between consecutive valid slots in the window), 5 decimals |
| `spread` | population standard deviation (`ddof=0`) over the whole table, 4 decimals |
| `snr` | `spread / noise`, 2 decimals. NaN when `spread == 0` (the column never moves: blind, not stable). `inf` when `noise == 0` and `spread > 0`. |
| `window_mean` | mean of the column in the window, 3 decimals |
| `changed_share` | share of steps that are greater than 0, 3 decimals |
| `kind`, `source`, `object` | first part; second part when the name has at least 3 parts (else `""`); last part |

5. **Order.** Sorted by `object` ascending, then `snr` descending with NaN last. For each object, `inf` comes first and blind columns last.

`sbi stability --space S [--start "2026-11-14 00:00"] [--end ...] [--min-mean X] [--out data/exports]` builds the space table, writes `<out>/<space>_stability.csv`, and prints the rows whose `|window_mean| >= min_mean`, without the `kind` and `source` columns. The quiet window is chosen by the user. The CLI help suggests a night window.

### 4.5 Detector benchmark scripts (`scripts/detector_bench/`)

These are one-off scripts, not part of the package. They read `data/sbi.db` read-only and compare candidate detectors on stored photos.

#### 4.5.1 `bench.py`: run one candidate

Usage: `.venv/bin/python scripts/detector_bench/bench.py NAME`. Output: `data/exports/detector_bench/NAME.json`. The folder is created on import.

**Photo selection** (`bench.py::frames()`): every second `admin_cabin` photo with `capture_status='ok'` in `ts_utc` order, plus every photo that has a `labels` row (any space). The result is a dict `frame_id -> (image_path, brightness, ts_local, space_id)`. Photos that OpenCV cannot read are skipped.

**Candidate names** (`bench.py::make(name)`):

| Name form | Model | Behaviour |
|---|---|---|
| `yolo12x@SIZE`, `yolo26x@SIZE`, `yolo11x@SIZE` | `ultralytics.YOLO("models/<weights>.pt")` | `ultra()` at input size `SIZE` |
| `...+tta` | same | `augment=True` (test-time augmentation) |
| `...+clahe` | same | CLAHE on the L channel of LAB (clip 3.0, 8x8 tiles), only for photos with stored brightness below 100 |
| `...+tile` | same | the whole photo plus a left tile (0 to 60% of the width) and a right tile (40% to 100%), each at the full input size. A tile box within 8 px of an inner tile edge is dropped ("the other view has it whole"). Tile x coordinates are mapped back to the whole photo; tiles span the full height. `+tta` and `+clahe` are ignored in this mode. Overlapping boxes are not merged here; `score.py` de-duplicates. |
| `rtdetr*@SIZE` | `ultralytics.RTDETR("models/<weights>.pt")` | `ultra()` at `SIZE` |
| `yoloe*@SIZE[+p2]` | `ultralytics.YOLOE`, one model per space with `set_classes(prompts, get_text_pe(prompts))` | Text prompts. Cabin: `PROMPTS` (12 COCO-like names), or `PROMPTS2` with `+p2` (13 descriptive names such as "office chair", "computer monitor", "desktop computer tower"). Parking: `PARK_PROMPTS` (person, car, motorcycle, bicycle, truck, bus), or `PARK_PROMPTS2` with `+p2` (adds scooter). Changes the working directory to `models/`. |
| `gdino*` | Hugging Face `IDEA-Research/grounding-dino-base` on CUDA | Text `"a . b . c ."` from `PROMPTS` or `PARK_PROMPTS`; box and text thresholds 0.15; boxes normalised by width and height |
| `owlv2*` | Hugging Face `google/owlv2-base-patch16-ensemble` on CUDA | The photo is padded to a square with grey (127) at the bottom/right. Queries `"a photo of a <prompt>"`; threshold 0.05; boxes normalised by the original size, with x2/y2 clipped to the photo |
| anything else | - | `SystemExit("unknown candidate ...")` |

`bench.py::ultra(model, img, imgsz, augment=False, conf=0.05)` runs an Ultralytics predict with `conf=0.05`, `iou=0.5`, `device=0`. It returns `(class name, conf rounded to 3, [x1, y1, x2, y2] normalised, rounded to 4)` per box.

**Output file**: `{frame_id: {"ts", "space", "brightness", "boxes": [[name, conf, [x1, y1, x2, y2]], ...]}}`. The script prints `"NAME: N photos in Ts (s/photo)"`.

#### 4.5.2 `score.py`: compare all candidates

`score.py` scores every `*.json` in `data/exports/detector_bench/`. It prints one row per candidate, sorted by `avg miss`.

| Item | Rule |
|---|---|
| Class mapping `MAP` | Detector names are lower-cased and stripped, then mapped: `chair`/`office chair` -> chair; `person`; `tv`/`monitor`/`computer monitor` -> monitor; `laptop`/`laptop computer` -> laptop; `backpack`/`handbag` -> bag; `car`; `motorcycle`/`scooter` -> motorcycle; `couch` -> sofa. Names not in `MAP` are ignored. |
| `score.py::iou(a, b)` | Box IoU with a `1e-9` floor on the union. |
| `score.py::counts(boxes, label, thr, dedupe=0.6)` | Greedy per-class de-duplication: boxes sorted by confidence descending; a box is kept when it maps to `label`, has `conf >= thr`, and has IoU of at most 0.6 with every box already kept. Returns the number kept. |
| Thresholds `THR` | `0.10, 0.15, ..., 0.80` (15 values) |
| Chair truth | 7 chairs in every cabin photo (`admin_cabin`). |
| Chair split | The cabin photos in file order. Even positions are the tuning half; odd positions are the test half. The chair threshold is the `THR` value with the lowest mean `|count - 7|` on the tuning half. Results are reported on the test half only. |
| Night | `brightness < 95`; a missing brightness counts as day (200). |
| Other types | `person, monitor, laptop, bag, car, motorcycle` are scored against the hand labels (`labels` table) on every photo of the candidate that has a label for that type. The threshold is 0.25, except `monitor`, whose threshold is chosen (lowest mean miss over `THR`) on those same labelled photos. |

Output columns per candidate:

| Column | Meaning |
|---|---|
| `detector` | the JSON file name |
| `chair thr` | the threshold chosen on the tuning half |
| `chairs =7`, `within 1`, `avg miss` | on the test half at that threshold |
| `=7 day`, `=7 night` | share exactly 7 by day and by night |
| `min`, `max` | lowest and highest chair count on the test half |
| `=7 @0.25` | share exactly 7 at a fixed 0.25 threshold |
| `<type> miss` | mean absolute error against the labels (NaN with no labels) |

A last line prints the number of test photos, the night count and the number of labelled photos. These come from the last candidate processed.

### 4.6 Tests

#### 4.6.1 Strategy and how to run

- No test touches the camera, the GPU, or LM Studio (`tests/conftest.py` docstring). Every model and every external source is replaced by a fake: frame source, detector, scene reader, segmenter, image embedder, probe scorer, forecasters, an HTTP session, the SAM 2 callable, the Chronos-2 pipeline, the text embedder and judge used for curation, and the `holidays` package.
- Storage is real. Each test gets a fresh SQLite file under pytest's `tmp_path`, with every migration applied, so SQL, migrations and transactions are tested for real.
- Space configuration is real: `config/spaces/admin_cabin.yaml`, `config/spaces/parking.yaml` and `config/pipeline.yaml` from the repository are loaded or copied into the temporary root.
- Both Streamlit apps are run end to end with `streamlit.testing.v1.AppTest` against a temporary root (`SBI_ROOT`).
- `tests/test_segmenter.py` is skipped if `torch` is not installed (`pytest.importorskip("torch")`).
- Test modules import helpers from each other (for example `DETS` and `cap` from `test_flow`, `make_table` from `test_forecast`). `tests/__init__.py` makes `tests` a package so these relative imports work.

Run from the repository root:

```
.venv/bin/python -m pytest -q -p no:warnings
```

`pyproject.toml` sets `pythonpath = ["src"]`, `testpaths = ["tests"]` and `addopts = "-q"`. With the extra `-q` the run prints only progress dots and no summary line; the exit code is 0 when everything passes. The suite collects and passes **254 tests** (counted 2026-11-18). It took about 17 s on this machine.

| File | Test functions | Collected tests |
|---|---|---|
| `test_dashboard.py` | 16 | 16 |
| `test_detections.py` | 12 | 12 |
| `test_embeddings.py` | 26 | 27 |
| `test_feature_cache.py` | 8 | 8 |
| `test_flow.py` | 23 | 23 |
| `test_forecast.py` | 28 | 28 |
| `test_foundation.py` | 11 | 17 |
| `test_labelling.py` | 10 | 10 |
| `test_labelling_app.py` | 1 | 1 |
| `test_masks.py` | 12 | 12 |
| `test_perception.py` | 14 | 19 |
| `test_probes.py` | 18 | 24 |
| `test_profiles.py` | 8 | 8 |
| `test_segmenter.py` | 3 | 3 |
| `test_store.py` | 10 | 16 |
| `test_v1_ops.py` | 15 | 15 |
| `test_vocabulary.py` | 15 | 15 |
| **Total** | **230** | **254** |

#### 4.6.2 `conftest.py`: fakes, helpers and fixtures

Helpers:

| Name | Behaviour |
|---|---|
| `REPO` | the repository root |
| `synthetic_image(seed=0, w=320, h=180)` | random `uint8` BGR image with values 40 to 219, reproducible per seed |
| `make_reading(people=2, objects=(("chair", 3, "idle"),), scene=None, raw="{}")` | a `SceneReading` with `people` seated/typing entries, the given objects, a default scene `{lighting: artificial, door_state: unknown, tidiness: ok}`, summary "test scene", model `fake-vlm`, prompt version `vtest` |
| `det(label, bbox, conf=0.9, counted=True)` | builds a `Detection` |
| `thursday_ist(hour=10, minute=0, week=0)` | Thursday 2026-11-12 at the given IST time plus `week * 7` days, as an aware UTC `datetime` |
| `__all__` | re-exports `CameraError`, `DetectorError`, `VLMError` |

Fakes:

| Fake | Version / attributes | Behaviour |
|---|---|---|
| `FakeSource(items)` | `calls` | `grab(space)` returns the items in order and repeats the last one when the list is exhausted. An `Exception` instance in the list is raised instead. |
| `FakeDetector(dets=(), error=None)` | `version="fake-det@1"`, `min_conf=0.05`, `calls` | `detect(image, space)` returns a copy of `dets`, or raises `error`. |
| `FakeReader(reading=None, error=None)` | `calls` | `read(image, space)` returns `reading` (default `make_reading()`) or raises `error`. `parse(raw, space)` returns the same reading with `raw` replaced, so stored answers can be replayed. |
| `FakeSegmenter(masks=None, error=None)` | `version="fake-seg@64"`, `calls` | `segment(image, boxes)` raises `error`; returns the given `masks`; or, by default, one `SegMask` per box: a filled rectangle on a 64x36 grid with `area = box width * box height`. |
| `FakeEmbedder(vectors=None, error=None)` | `version="fake-emb@8"`, `dim=8`, `calls` | `embed(image)` raises `error`; pops the next given vector; or returns `1 + mean(image)/255 * arange(8)`. |
| `FakeScorer(values=None, error=None)` | `version="fake-scorer"`, `asked` | `score(image, prompts)` records the prompts, raises `error`, or returns `values[p]` per prompt, else `len(p) % 7 - 3`. |

Fixtures:

| Fixture | Behaviour |
|---|---|
| `root` | `tmp_path` |
| `settings` | `Settings(root=root, scene_reader=SceneReaderSettings(base_url="http://fake", model="fake-vlm"))` |
| `space` | `admin_cabin.yaml` loaded from the repository |
| `parking` | `parking.yaml` loaded from the repository |
| `conn` | a connection to `root/t.db`, migrated, with both spaces registered; closed after the test |
| `space_dir` | `root/config/spaces/` with both real space files copied in (for tick tests) |

#### 4.6.3 Helpers shared between test modules

| Module | Helpers |
|---|---|
| `test_flow.py` | `DETS` (two persons, a bag, a chair); `cap(conn, space, root, source, now)` (capture through `capture_frame` into `root/data/images`); `count(conn, table, **where)`; `processed(...)` (capture and process); `make_root(...)` (sorted space files) |
| `test_forecast.py` | `make_table(n=200)`: a synthetic table starting 2026-11-12 00:00 where `scene.people_yolo` equals the row position (so any leak shows as a larger value), plus count, variant, late-starting, flat, area, probe, emb, img, cat and calendar columns; `result_for(df)`; `context_of`, `future_of`; `SpyForecaster` (persistence with ±1 intervals that records what it was shown, can be told to fail); `Negative` (predicts below zero); `forecast_frame`, `plan`, `prep`, `live`; `StubPipeline`, `adapter`, `raw_output` (Chronos-2 stub); constants `Q`, `FS`, `TS`, `CATALOG_ARGS` |
| `test_detections.py` | `LOW` (two uncounted low-confidence boxes); `build` |
| `test_dashboard.py` | `_forecast(table, target, values, step0=1)`: a wide forecast frame after the table's end |
| `test_labelling.py` | `frame_at`, `labelled`; `BIKES` (5 motorcycles, confidence 0.9 to 0.07); `TWICE` (one motorcycle boxed twice) |
| `test_masks.py` | `rect(...)` boolean rectangle on a grid; `run` |
| `test_embeddings.py`, `test_probes.py` | `frame`, `run`, `build`; `E0`, `E1`, `E2` unit vectors; `space_with`; `ShortScorer` |
| `test_feature_cache.py` | `photos`, `build`; fixture `counted` (spies on `frame_columns` and records how many photos each call derived) |
| `test_profiles.py` | `week_series`, `working`, `points`, `DAY` |
| `test_v1_ops.py` | `write`, `scheduled`, `tick`, `runs`, `store_run`, `IST` |
| `test_vocabulary.py` | `run_frame`, `reading_with`, a text `FakeEmbedder` (same-group names get identical vectors), `FakeJudge`, `seed_names`, `build` |
| `test_perception.py` | `GOOD` (valid VLM answer), `Resp`, `ok_body`, `Session` (fake HTTP session), `reader` |
| `test_segmenter.py` | `FakeSam` (fake SAM 2 callable that can drop boxes or fail), `segmenter` (builds a `Sam2Segmenter` without loading weights), `IMG`, `BOXES` |
| `test_store.py` | `_canon`, `_frame` |
| `test_foundation.py` | `_write` (mutated copy of `parking.yaml`), `_settings_root` |

#### 4.6.4 Tests per file

##### `test_dashboard.py`: dashboard data and a smoke run of the real app

| Test | What it checks |
|---|---|
| `test_health_counts_the_last_24_hours_per_stage` | `health()` counts only the last 24 hours: photos vs expected, failed captures, stage shares, last photo, latest forecast and model; `-` for a space with no photos |
| `test_log_problems_keeps_only_warnings_and_errors_newest_last` | only WARNING/ERROR lines, newest last, the `limit`, a missing file |
| `test_chart_data_joins_measured_history_and_the_forecast_band` | one day of measured rows (97) followed by forecast median and band; unknown target gives empty; no forecast gives measured only |
| `test_latest_forecast_and_the_read_only_connection` | the latest run and its quantiles; `(None, empty)` for a space with no run; a write on the read-only connection fails |
| `test_the_dashboard_app_runs_end_to_end` | `AppTest` run of `app.py`: no exception, the five tab labels, the header, "no photos yet" for parking, the guide text, tiles in order, at least 4 tables and 3 charts; switching to parking shows "No photos stored yet" |
| `test_the_app_opens_its_database_connection_inside_each_refresh` | AST check: no module-level connection; `page()` opens and closes one |
| `test_header_status_says_whether_each_space_is_live` | photo time, age, people, no vehicles for the cabin; `live`/`late`/`stopped` limits at 20 and 60 minutes; `no photos` for parking |
| `test_labels_read_like_words` | `label()` rules for thresholds, outlines, scene, counts, cell areas, relations, probes and unknown names |
| `test_usual_is_the_mean_at_the_same_clock_time_on_earlier_days_only` | `before` leaves today out; without it every day counts; NaN with no earlier data |
| `test_summary_gives_now_usual_and_the_forecast_at_each_lead` | now, usual, the +1/+2/+4 h median and band, unknown columns skipped, NaN with no forecast |
| `test_series_chart_adds_the_usual_line_over_history_and_forecast` | 97 measured + 2 forecast + 99 usual rows on the same timestamps |
| `test_lead_history_pairs_each_forecast_with_what_was_measured` | newest run wins for the same moment; measured filled or NaN; empty for another space or a missing step |
| `test_capture_grid_gaps_and_stamp` | grid statuses; gap runs; `table_stamp` before capture, after capture, after reading, and after a re-read |
| `test_funnel_counts_one_photo_per_slot_through_each_stage` | slot and stage counts with a missing slot and an unread photo |
| `test_tick_timing_reads_run_durations_from_the_log` | median, p95 and max from load/done pairs; `last`; a missing file |
| `test_dimension_flow_and_the_dimension_list_say_where_each_column_goes` | per-family numbers, `cal.*` counted as every calendar column, totals, no run; `dimensions()` values, `use` text and `cat` text values |

##### `test_detections.py`: boxes below the counting threshold, detector-only reprocess, stability

| Test | What it checks |
|---|---|
| `test_counting_thresholds_must_not_be_below_the_storage_floor` | `DetectorSettings`/`TableSettings` validation; per-class thresholds; `level_source(0.05) == "yolo_c05"` |
| `test_build_detections_whitelists_marks_counted_and_honours_per_label_thresholds` | labels outside the vocabulary are dropped; `counted` set by threshold; `class_conf` |
| `test_low_confidence_boxes_never_change_the_counted_set` | adding weak boxes leaves the counted set equal to a run at the counting threshold |
| `test_uncounted_boxes_are_stored_raw_but_never_counted` | detections stored with `counted=0`, their cells, `detector_min_conf`, no geometry links from weak boxes, reprocessing replaces |
| `test_detector_only_reprocess_keeps_the_stored_vlm_answer` | `replay_vlm=True` makes no VLM call and keeps VLM rows and provenance |
| `test_detector_only_reprocess_keeps_a_stored_vlm_failure_as_unknown` | a stored VLM error stays unknown after replay |
| `test_unreplayable_stored_answer_fails_loudly_and_leaves_the_frame_untouched` | `StorageError` when a stored answer cannot be parsed or no reader is given; data unchanged |
| `test_lmstudio_reader_parses_a_stored_answer_without_a_network_call` | `LmStudioSceneReader.parse` makes no HTTP call |
| `test_union_area_counts_overlaps_once` | box union area for degenerate, nested, overlapping and separate boxes |
| `test_table_counts_at_each_threshold_and_old_frames_are_unknown_not_zero` | `count.yolo_cNN.*` and `area.yolo_cNN.*` columns; frames read with a higher floor are NaN; levels are opt-in |
| `test_threshold_variants_are_not_default_model_inputs_unless_asked_for` | variant columns are model-ready but not selected unless listed in `always` |
| `test_stability_ranks_steady_and_responsive_above_jittery_and_flags_blind_columns` | order, `noise`, `snr` inf/NaN, `changed_share`, `pairs` with gaps, `cat` excluded, empty window |

##### `test_embeddings.py`: whole-scene image vectors

| Test | What it checks |
|---|---|
| `test_vector_codec_round_trips_within_half_precision_and_rejects_bad_input` | float16 codec round trip, size, rejection of NaN/2-D/empty and corrupt bytes |
| `test_models_coexist_replace_and_stay_inside_their_space` | one vector per (frame, model), replacement, per-space loading |
| `test_corrupt_and_mixed_dimension_rows` | corrupt rows returned as unknown; mixed dimensions raise `StorageError` |
| `test_frames_missing_embedding_skips_done_failed_captures_and_other_models` | backlog query |
| `test_embedder_settings_version_width_rule_and_factory` | version string, width multiple of 14, `kind="none"` builds nothing |
| `test_prepare_keeps_the_whole_frame_at_a_patch_aligned_size` | DINOv2 resize to patch-aligned sizes, normalisation, empty input |
| `test_prepare_converts_bgr_to_rgb` | channel order |
| `test_combine_is_cls_then_patch_mean_each_unit_length` | CLS + patch-mean vector, unit length, zero vectors safe |
| `test_cosine_distance_rows_and_missing_values` | row-wise cosine distance with NaN and zero rows |
| `test_drift_prev_is_nan_next_to_a_missing_slot_and_exact_elsewhere` | `drift_prev` around gaps |
| `test_pc1_follows_the_main_direction_with_a_deterministic_sign` | PCA direction, sign, `drift_typical`, orthonormal axes |
| `test_every_axis_has_its_largest_loading_positive_whatever_the_data` | sign rule over 12 random seeds |
| `test_fit_window_keeps_later_data_out_of_the_components` | components fit only on the fit window; later rows still projected |
| `test_too_little_data_gives_unknown_not_zero_and_extra_components_are_nan` | 1 row, 3 rows, constant rows, 0 rows |
| `test_pipeline_stores_one_vector_per_frame_and_model` | storage and replacement on re-run |
| `test_embedder_failure_keeps_the_frame_without_a_vector` (2 cases: raises, wrong size) | frame still processed, no vector |
| `test_a_failed_rerun_never_wipes_a_stored_vector` | stored vector kept |
| `test_without_an_embedder_nothing_is_written` | no rows |
| `test_tick_embeds_every_space_and_survives_an_embedder_that_cannot_load` | tick continues when the embedder cannot be built |
| `test_backfill_fills_only_what_is_missing_and_reports_failures` | `embed_missing` done/failed counts, `limit` |
| `test_backfill_survives_an_embedder_error_on_one_frame` | one error does not stop the backfill |
| `test_table_has_image_vector_columns_and_nan_where_no_vector` | `emb.*` columns, NaN where missing, `drift_prev` values, slot not voided |
| `test_table_without_a_model_or_vectors_has_no_image_columns` | no `emb.*` columns without a model name or stored vectors |
| `test_table_fit_end_changes_what_the_components_are_fit_on` | `embed_fit_end` |
| `test_corrupt_vector_is_a_gap_with_a_warning` | corrupt row gives NaN and a warning |
| `test_image_vector_columns_are_eligible_inputs_and_in_the_stability_report` | model-ready, selected as covariate, present in `stability()` |

##### `test_feature_cache.py`: the per-photo column cache

| Test | What it checks |
|---|---|
| `test_the_table_is_identical_with_a_cold_cache_a_warm_cache_and_no_cache` | identical tables and catalogs; the warm build derives nothing |
| `test_only_new_and_re_read_photos_are_derived_again` | a new photo and a re-read photo are the only misses |
| `test_settings_and_vocabulary_changes_miss_the_cache` | another setting or a new alias derives everything again; old keys are dropped |
| `test_a_broken_cache_never_breaks_the_table` | unreadable or unwritable cache file gives the same table and a warning |
| `test_cache_entries_follow_key_and_stamp_and_batch_over_many_photos` | key/stamp matching, more than two SQL batches, `fingerprint` order-independence |
| `test_frame_columns_covers_every_photo_asked_for_also_beyond_one_sql_batch` | unknown ids give `{}`; forced small batches give the same result |
| `test_max_slots_keeps_the_newest_slots_only` | `max_slots` |
| `test_table_for_uses_the_cache_folder_from_settings` | `table_for` writes `table_features.db` in the cache folder; `use_cache=False`; `max_slots` |

##### `test_flow.py`: capture, pipeline, baseline, tick, labelling and eval

| Test | What it checks |
|---|---|
| `test_capture_ok_stores_ist_named_relative_image` | IST fields and a relative IST-named image path |
| `test_camera_failure_is_data_not_zero` | `camera_down` row, no image |
| `test_flat_black_frame_is_flagged_not_stored` | `bad_frame`, no image |
| `test_duplicate_timestamp_is_idempotent_and_returns_the_right_id` | one row per (space, time) |
| `test_duplicate_returns_its_own_id_even_after_other_rows_were_inserted` | regression for `INSERT OR IGNORE` + `lastrowid` |
| `test_disk_failure_raises_storage_error_and_records_nothing` | unwritable image folder |
| `test_pipeline_writes_detector_vlm_scene_and_registry` | observations, cells, scene state, new object registry entry, provenance |
| `test_vlm_failure_keeps_detector_results_with_null_vlm_fields` | VLM fields NULL, never 0; error JSON stored |
| `test_detector_failure_propagates_and_writes_nothing` | frame stays in the backlog |
| `test_midway_failure_rolls_back_everything_and_cannot_leak_into_later_commits` | regression for a half-written frame |
| `test_reprocessing_replaces_results_instead_of_duplicating` | replacement |
| `test_non_ok_and_missing_frames` | not processed; unknown frame id raises |
| `test_occupancy_events` | `became_empty`, `became_occupied` |
| `test_baseline_counts_empty_cells_as_zero_but_excludes_missing_frames` | baseline means and presence |
| `test_baseline_vlm_failure_is_excluded_not_zero` | VLM failures left out |
| `test_bucket_scoped_rebuild_leaves_other_buckets_alone` | partial rebuild |
| `test_tick_isolates_a_failing_space_and_keeps_the_other` | an unexpected error in one space |
| `test_camera_down_in_one_space_does_not_fail_the_tick` | camera failure is data |
| `test_invalid_space_file_is_skipped_others_run` | bad space file |
| `test_detector_outage_still_captures_and_backlog_recovers` | backlog drained on the next tick |
| `test_missing_camera_credentials_is_a_fatal_config_error` | `ConfigError` |
| `test_labelling_flow_and_eval` | `progress`, `next_unlabeled` serves every frame once, `save_label`, `model_counts`, `vlm_eval.summarise` bias/within1/exact, `report` |
| `test_labelling_validation_and_unusable_frames` | negative counts rejected; unusable frames store no `labels`; "no labels yet" report; missing frame image raises |

##### `test_forecast.py`: forecasting

| Test | What it checks |
|---|---|
| `test_forecast_settings_rules` | default variant, variant sets, quantile validation |
| `test_forecast_targets_in_space_files_and_their_validation` | `forecast_targets` in the space files; duplicates and blanks rejected |
| `test_persistence_holds_the_last_value_and_every_quantile_equals_it` | `Persistence` |
| `test_seasonal_naive_takes_the_same_clock_time_from_the_latest_earlier_day_that_has_one` | `SeasonalNaive` with gaps and short history |
| `test_seasonal_mean_uses_weekday_buckets_only_with_two_points_else_time_of_day_else_last_value` | `SeasonalMean` fallbacks |
| `test_a_target_with_no_history_is_an_error_and_unknown_models_are_named` | `ForecastError`s; `build_forecaster` |
| `test_pinball_loss_by_hand` | pinball loss values |
| `test_evaluate_scores_each_point_and_skips_unmeasured_slots` | `evaluate` errors, coverage, pinball |
| `test_summary_skill_compares_with_the_reference_on_the_same_points_only` | `summarise` skill and wql; `overall` |
| `test_targets_come_from_model_ready_columns_in_pattern_order_without_variants` | `resolve_targets` |
| `test_unmatched_pattern_is_reported_but_does_not_stop_the_others` | warning for an unmatched pattern |
| `test_variants_decide_which_inputs_exist` | context/future columns per input set |
| `test_all_variant_gives_every_model_ready_column_uncapped` | the `all` input set ignores caps |
| `test_counts_are_never_forecast_below_zero_but_other_kinds_may_be` | non-negative clip for counts only |
| `test_origins_leave_enough_history_behind_and_a_full_horizon_ahead` | `origins_for` |
| `test_the_model_never_sees_rows_after_the_origin_and_the_future_frame_is_only_the_clock` | no leak in the backtest |
| `test_scores_use_the_right_truth_slot_for_every_step` | truth alignment per step |
| `test_columns_are_judged_from_the_history_before_each_origin_not_the_whole_table` | per-origin catalog |
| `test_baselines_run_once_per_origin_and_models_once_per_variant` | call counts |
| `test_a_failing_forecaster_is_skipped_at_those_origins_and_counted` | failure counts |
| `test_too_little_history_is_an_error_with_the_numbers` | minimum context message |
| `test_chronos_adapter_maps_the_real_output_format_and_passes_the_request_through` | Chronos-2 output mapping and request arguments |
| `test_chronos_adapter_turns_model_failures_and_missing_quantiles_into_forecast_errors` | error mapping |
| `test_the_backtest_and_live_runs_both_apply_the_non_negative_rule` | clip in both paths |
| `test_forecast_rows_round_trip_with_their_run` | `insert_run`, `insert_forecasts`, `load_forecast`, `latest_run`, key violation |
| `test_live_forecast_stores_the_run_with_the_inputs_it_used` | stored run metadata per input set |
| `test_live_forecast_refuses_bad_requests_and_a_failed_model_stores_nothing` | refusals; nothing stored on failure |
| `test_cli_forecast_and_backtest_end_to_end_with_baselines` | `sbi forecast` and `sbi backtest` exit codes and output; backtest CSV written |

##### `test_foundation.py`: time, grid, configuration

| Test | What it checks |
|---|---|
| `test_time_dims_are_ist` | UTC to IST fields |
| `test_local_date_rolls_over_at_ist_midnight_not_utc` | local date at IST midnight |
| `test_time_dims_rejects_naive_and_bad_zone` | naive datetime and unknown zone |
| `test_grid_cells_and_edges` | cell lookup, clamping, bottom-centre anchor, multi-cell boxes |
| `test_real_space_configs_validate` | shipped space files load |
| `test_invalid_space_configs_are_rejected_with_a_clear_message` (7 cases) | bad alias, detector map, label types, mask region, timezone, unknown key, label name |
| `test_missing_space_file` | "not found" |
| `test_settings_load_and_env_override` | `SBI_LMSTUDIO_URL` override, `db_path` |
| `test_settings_reject_unknown_keys_and_missing_file` | "invalid settings", "not found" |
| `test_credentials_missing_and_secret_never_in_repr` | NVR credentials checks; password never in `repr`/`str`; bad port |
| `test_rtsp_url_redaction` | percent-encoding and redaction |

##### `test_labelling.py`: photo order, quarters, thresholds from labels

| Test | What it checks |
|---|---|
| `test_photos_come_from_the_least_labelled_day_type_first` | one photo from each day type before any repeat; exhaustion; `STRATA` and `stratum()` |
| `test_the_order_is_stable_between_sessions` | the same next frame twice |
| `test_quadrants_cover_the_photo_once_at_full_resolution` | names, full coverage, no overlap |
| `test_counts_are_recomputed_at_every_threshold_from_stored_boxes` | `sweep()` counts per level; no outline rule without outlines; `score()` MAE and bias |
| `test_outline_rule_removes_a_bike_boxed_twice` | outline de-duplication in `sweep()` |
| `test_frames_read_only_down_to_a_higher_floor_are_not_scored_below_it` | levels under the frame's floor are left out |
| `test_unusable_photos_are_ignored` | unusable frames, including a stray `labels` row, give no points and an empty report |
| `test_recommendation_is_the_strictest_option_close_to_the_best_and_needs_enough_labels` | best and recommended `yolo@0.10` for bikes; keep `yolo@0.25` on a tie; suggestion `{motorcycle: 0.10}`; fewer than `MIN_LABELS` gives `-` |
| `test_suggestions_skip_outline_picks_the_current_threshold_and_disagreeing_spaces` | `class_conf_suggestion` returns `{}` |
| `test_eval_report_includes_the_threshold_section` | `vlm_eval.report` contains the threshold section and the `class_conf` line |

##### `test_labelling_app.py`: the labelling app under Streamlit

| Test | What it checks |
|---|---|
| `test_the_labelling_app_survives_runs_on_different_threads` | two `AppTest` runs on two threads with the cached setup raise no exception |

##### `test_masks.py`: outlines

| Test | What it checks |
|---|---|
| `test_mask_codec_round_trips_and_rejects_bad_input` | mask codec, size, errors |
| `test_duplicate_outlines_keep_the_stronger_box_only` | `mask_iou`, `drop_duplicate_masks`, nested outlines are not copies |
| `test_cell_shares_match_the_grid` | per-cell coverage |
| `test_mask_features_per_source_dedupe_union_and_cells` | `count.sam2*`, `area.sam2*` (union), `cellarea.*` and `cell_min` |
| `test_every_stored_box_gets_an_outline_and_counts_are_unaffected` | masks stored for every box; `segmenter_version` |
| `test_segmenter_failure_keeps_the_frame_with_unknown_outlines` | error or wrong mask count gives no masks and a NULL version |
| `test_no_boxes_with_a_working_segmenter_is_known_empty_not_unknown` | version set with zero boxes; NULL without a segmenter |
| `test_segmenter_factory_and_missing_weights` | off by default; missing weights raise |
| `test_tick_runs_without_outlines_when_the_segmenter_cannot_load` | tick continues |
| `test_table_outline_columns_and_frames_without_outlines_are_unknown` | outline columns NaN without outlines; values; variants; not default inputs |
| `test_uncount_outline_duplicates_keeps_the_stronger_box_per_object_and_type` | `uncount_outline_duplicates` rules and errors |
| `test_pipeline_counts_an_object_boxed_twice_once_when_asked_and_keeps_both_boxes` | `dedupe_outline_iou` in the pipeline; off by default; no effect without a working segmenter |

##### `test_perception.py`: box de-duplication and the scene reader

| Test | What it checks |
|---|---|
| `test_same_label_overlap_is_collapsed_but_distinct_boxes_survive` | `dedupe_same_label_iou` |
| `test_different_labels_never_merge` | different labels kept |
| `test_contained_person_box_dropped_adjacent_people_kept` | `dedupe_contained` |
| `test_contained_dedupe_only_applies_to_chosen_labels` | only chosen labels; `iou` |
| `test_parse_valid_and_coerces_harmless_deviations` | unknown status and enum values become `unknown` |
| `test_parse_rejects_unusable_shapes` (6 cases) | bad people count, empty label, negative count, missing objects |
| `test_parse_rejects_non_json_and_non_object` | empty, non-JSON, array |
| `test_schema_and_prompt_follow_the_space` | schema fields and prompt per space |
| `test_reader_success_sends_schema_and_reasoning_effort` | request body |
| `test_reader_falls_back_to_reasoning_content_when_content_is_empty` | fallback field |
| `test_reader_retries_transient_failures_then_succeeds` | retries on connection error, 503, garbage |
| `test_reader_does_not_retry_client_errors` | HTTP 400 not retried |
| `test_reader_gives_up_with_vlmerror_after_retries` | `VLMError` after 3 attempts |
| `test_reader_omits_reasoning_effort_when_unset` | no `reasoning_effort` key |

##### `test_probes.py`: text-prompt probes

| Test | What it checks |
|---|---|
| `test_shipped_space_files_define_probes_and_a_stable_prompt_order` | probe groups and prompt order |
| `test_bad_probe_definitions_are_rejected_with_the_reason` (6 cases) | name, empty, length, shared and repeated prompts |
| `test_scorer_settings_and_factory` | off by default; version from model path |
| `test_scores_replace_per_prompt_and_scorers_stay_apart` | storage; non-finite rejected |
| `test_missing_scores_name_exactly_the_prompts_each_frame_lacks` | backlog per prompt |
| `test_read_frame_image_handles_relative_absolute_and_missing` | `read_frame_image` |
| `test_concept_is_the_mean_of_its_paraphrases_and_unknown_if_any_is_missing` | `concept_values` |
| `test_pipeline_asks_every_prompt_once_and_stores_each_score` | storage and replacement |
| `test_scorer_failure_keeps_the_frame_without_scores` (2 cases: raises, wrong count) | frame processed, no scores |
| `test_a_failed_rerun_never_wipes_stored_scores` | stored scores kept |
| `test_no_scorer_or_no_probes_means_no_model_call_and_no_rows` | no call, no rows |
| `test_tick_scores_every_space_and_survives_a_scorer_that_cannot_load` | tick continues |
| `test_backfill_scores_history_and_a_new_prompt_asks_only_the_new_question` | `score_missing` |
| `test_backfill_counts_unreadable_images_and_scorer_errors_and_keeps_going` | failures counted; `limit` |
| `test_table_has_one_column_per_concept_averaging_paraphrases_and_nan_where_unscored` | `probe.*` columns |
| `test_a_concept_with_a_prompt_missing_for_a_frame_is_unknown_for_that_frame` | partial concepts are NaN |
| `test_table_without_a_scorer_name_or_scores_has_no_probe_columns` | no probe columns |
| `test_probe_columns_appear_in_the_stability_report` | `probe` kind in `stability()` |

##### `test_profiles.py`: day-type profiles and accuracy by weekday

| Test | What it checks |
|---|---|
| `test_profile_uses_the_same_day_type_and_leaves_the_rows_own_day_out` | working/non-working profile, own day excluded |
| `test_profile_switches_to_the_same_weekday_once_it_has_been_seen_often_enough` | same-weekday switch and `min_same_weekday` |
| `test_profile_follows_the_clock_time_and_falls_back_when_a_kind_of_day_is_new` | fallbacks, gaps, empty series |
| `test_profiles_reach_the_model_as_known_in_advance_columns_only_when_asked` | `prof.*` columns in context and future |
| `test_a_profile_never_sees_beyond_the_history_it_is_given` | no leak |
| `test_add_profiles_returns_the_new_column_names` | column names and order |
| `test_by_weekday_shows_which_days_are_forecast_badly` | `by_weekday` per target and pooled |
| `test_runs_made_before_their_history_was_read_again_are_not_scored` | `outdated_runs` with tolerance; `live_points`; `data.lead_history` with `skip_runs` |

##### `test_segmenter.py`: SAM 2 wrapper with a fake model

| Test | What it checks |
|---|---|
| `test_one_outline_per_box_in_order_at_the_stored_size` | output size, order, areas, batching |
| `test_a_box_with_nothing_in_it_gets_an_empty_outline_and_the_others_keep_theirs` | per-box retry and empty outline; log message |
| `test_nothing_at_all_or_a_model_failure_is_a_segmenter_error` | `SegmenterError` messages |

##### `test_store.py`: migrations, transactions, normaliser

| Test | What it checks |
|---|---|
| `test_migrate_fresh_is_idempotent` | 8 migrations, tables and columns present |
| `test_migrate_upgrades_a_pre_versioning_database` | `user_version` 0 database upgraded |
| `test_migration_gap_is_rejected` | numbering gap |
| `test_failed_migration_rolls_back_atomically` | rollback |
| `test_transaction_commits_and_rolls_back` | `transaction()` |
| `test_nested_transaction_is_refused` | nested refused |
| `test_foreign_keys_are_enforced` | foreign keys on |
| `test_known_labels_map_to_registry` (7 cases) | aliases, plurals, case, spaces |
| `test_new_label_becomes_a_reviewable_row_and_event` | `auto_new` row and event with valid JSON |
| `test_empty_label_is_rejected` | `ValueError` |

##### `test_v1_ops.py`: site calendar, scheduled forecasts, live accuracy

| Test | What it checks |
|---|---|
| `test_site_file_entries_are_read_and_bad_entries_are_named` | `read_site_file` |
| `test_the_shipped_site_file_holds_only_fixed_date_national_days` | 26 Jan, 15 Aug, 2 Oct |
| `test_country_holidays_are_optional_and_the_site_file_wins` | fake `holidays` module; site names win |
| `test_working_days_are_validated` | `CalendarSettings.working_days` |
| `test_calendar_columns_mark_holidays_and_working_days_and_reach_the_future_frame` | `cal.is_holiday`, `cal.is_working_day` in future |
| `test_backtest_horizons_use_the_table_calendar` | holiday visible in backtest futures |
| `test_due_on_the_slot_grid_in_local_time` | `due()`; `every_minutes` validation and default 0 |
| `test_the_tick_stores_a_forecast_per_space_only_when_due` | no model loaded when not due; one model when due |
| `test_scheduled_forecasts_really_store_runs_end_to_end` | runs stored per space |
| `test_forecasting_problems_never_touch_capture` | model load failure, model error, a bug: capture unaffected |
| `test_run_scheduled_reports_each_space` | per-space result and error |
| `test_an_unexpected_error_in_one_space_does_not_stop_the_next` | isolation |
| `test_live_accuracy_scores_stored_forecasts_against_what_happened_and_against_persistence` | `live_points`, `by_target`, `since`, scale |
| `test_accuracy_by_lead_time_and_point_baselines_have_no_interval` | `by_lead`, `lead_bucket` |
| `test_a_series_that_did_not_move_cannot_swamp_the_pooled_skill` | pooled scaled skill |

##### `test_vocabulary.py`: open vocabulary and the table

| Test | What it checks |
|---|---|
| `test_singular_candidates_do_not_mangle_double_s` | plural handling |
| `test_taxonomy_resolution_follows_aliases_merges_and_is_a_closure` | alias, merge chain, ancestors with a cycle |
| `test_curate_merges_synonyms_links_hierarchy_and_remaps_history` | `curate` with fake embedder and judge; history re-mapped; raw labels kept; no re-asking |
| `test_unrelated_pairs_never_link_and_proposals_need_approval` | proposals and approval |
| `test_rejecting_a_merge_restores_both_dimensions` | rejection and re-resolution |
| `test_canonical_prefers_configured_names` | `pick_canonical` |
| `test_overlap_links_are_symmetric_counted_and_ignore_same_label` | geometry links |
| `test_prompt_v3_has_no_name_list_and_schema_has_relations` | prompt and schema |
| `test_parse_keeps_valid_relations_and_drops_malformed` | relation parsing |
| `test_pipeline_stores_raw_labels_vlm_relations_and_geometry_links` | stored raw labels and relations; replacement |
| `test_table_grid_gaps_missing_semantics_and_vlm_failure` | slot grid, statuses, earliest frame per slot, absent = 0, missing = NaN |
| `test_table_rollups_cells_relations_and_catalog` | roll-ups, cells, relations, catalog reasons |
| `test_structural_objects_are_kept_but_not_eligible_and_selection_is_capped` | structural objects; `select_covariates` cap and `always` |
| `test_to_chronos_shapes_match_predict_df_contract` | context/future shapes |
| `test_table_requires_data` | "no frames" error |


## 5. Known limits and open work

### 5.1 Measurement and forecasting limits

| Limit | Effect | What would change it |
|---|---|---|
| Forecasts do not know weekdays yet | After an empty weekend, Monday morning was forecast at 0.7 people against 2.45 measured. With 5 days of history each weekday has been seen once. | More weeks of history. Fine-tuning Chronos-2 on this site after about a month (`Chronos2Pipeline.fit` exists; not attempted to completion). |
| One chair in about six cabin photos is miscounted | Usually one chair hidden behind a person or bag. | A detector trained on these two camera views; needs boxes drawn on a few hundred photos. |
| Parking motorcycles: average miss 2.0 of about 8 on labels | The camera angle makes parked bikes overlap. Left loose by decision. | Same as above. |
| Monitors: 0.71 average miss on labels | Some screens are seen edge-on. | More labels, then a per-type threshold. |
| Cabin has 7 labelled photos | Threshold advice needs at least 10 per type (`eval/thresholds.py::MIN_LABELS`). | More labels. |
| Holiday list has 3 dates | The working-day flag is wrong on other office holidays. | Fill in `config/site_calendar.yaml`. |
| Each space is forecast alone | Parking is not shown the cabin's series. | A joint forecast with `cross_learning` or one item per site. |
| Some targets duplicate others | `count.yolo.person` equals `scene.people_yolo`. | Trim `forecast_targets`. |
| Live accuracy reads every stored forecast | About 87,000 rows a day; the accuracy tab slows over months. | Score a recent window by default. |
| Photos and database exist only on this PC | A disk failure loses everything. | A copy to another machine or drive. |
| Ultralytics (YOLO26, SAM 2 wrapper) is AGPL-3.0 and the repo is public | Matters only if distributed or sold. | A licence decision. |

### 5.2 Code inconsistencies found while writing this document

None of these affects today's live results. Each was confirmed by reading the code. They are listed so they can be
fixed deliberately.

**Capture and operations**

- `ingest/sources.py::NvrRtspSource.grab` sets the open and read timeouts after `cv2.VideoCapture(url)` has already
  opened the stream. `frame_source.timeout_s` therefore probably has no effect. The real limit is `timeout 600` in
  `tick.sh`.
- `tick.py`: one `now` is used for every space, so a frame's time is the tick start, not the moment of grabbing.
- `tick.py::_run_space` rebuilds only the baseline bucket of the tick time. Backlog frames from other hours wait for
  `sbi baseline`. It also reports a space as ok when one of its frames failed to process (the error is logged).
- `tick.sh` rotates the log before taking the lock, so a skipped run can rename the log while a running tick still
  writes to it. `curate.sh` uses the system `python3`, not `.venv`, and its log is never rotated.
- `settings.py::load_settings` catches only `ValidationError`. A YAML file whose top level is not a mapping raises
  `AttributeError` or `TypeError`, not `ConfigError`. `paths.logs` is ignored by `tick.sh`, which hard-codes
  `data/logs/tick.log`. The comment says `timezone` is the default for spaces, but `SpaceConfig.tz` defaults to a
  fixed `Asia/Kolkata`.
- `cli.py reprocess`: the `--frame-id` help says "all processed frames" but every `ok` frame is selected. Embedder
  and scorer are not passed, so reprocessing never fills missing vectors or scores (`sbi embed` / `sbi score` do).
- The space-file comments say the vocabulary is "shown to the VLM". The v3 prompt is open-vocabulary and does not
  include it. In `admin_cabin.yaml`, detector class `tv` maps to `monitor` but the alias `tv` maps to `screen`.

**Perception, pipeline and store**

- `perception/detector_yolo.py::build_detections` decides `counted` from the unrounded confidence but stores it
  rounded to 3 decimals. A box at 0.2496 is stored as 0.250 with counted = 0.
- `pipeline._write` skips VLM object labels only when they are exactly `person` or `people`. A label such as
  `pedestrian` becomes a second `vlm` person observation.
- `perception/scene_lmstudio.py::parse_reading` does not type-check relation and people items. A malformed item
  raises `AttributeError`, not `VLMError`. The strict JSON schema normally prevents this.
- `curation/curator.py::undo_same` does not restore aliases and merges that `apply_same` re-pointed, and it sets
  `review_status='approved'` rather than the earlier status. `apply_same` writes `review_status='merged'`, a value
  the schema comment does not list.
- `store/registry.py::register_space` lowercases config aliases but keeps underscores, while lookups use spaced
  form. It also overwrites curator-made aliases with the same key on every tick.
- Schema comments and the code disagree:
  - `baseline_space.metric` is written as people_yolo / people_vlm / vehicles_yolo / brightness.
  - `events` types `object_appeared` / `object_gone` are never written.
- Parts of the schema have no writer:
  - tables and columns: `object_mobility`, `frame_features`, `observations.activity`, `events.cell_id`, the
    `spaces` map columns;
  - views: the three `v_*` views are never queried.
- `events.frame_id` lacks `ON DELETE CASCADE`, unlike the other child tables.
- `store/db.py::migrate` returns the highest file number even when the database's version is higher.

**Table and forecasting**

- `store/forecasts.py::load_forecast` names a 0.05 quantile `q5`, but `forecast/models.py::qname` gives `q05`.
  Quantiles below 0.10 would break live accuracy. The defaults (0.1, 0.5, 0.9) are not affected.
- `forecast/table.py::calendar` computes `cal.slot_of_day` with a fixed 15, not `step_minutes`. The docstring says
  image-vector directions are fit "on embed_fit_end and earlier"; the code uses strictly earlier slots.
- A corrupt stored outline raises inside `frame_columns` and fails the whole table build. A corrupt image vector is
  skipped with a warning.
- `build_table` does not round `start` / `end` to the slot step. The `conf_levels` validator message says "two
  decimals at most" but checks distinctness after rounding to whole percents.

**Dashboard, evaluation and benchmark**

- `dashboard/app.py` catches only `SbiError` and `OSError` at start-up. A missing database raises
  `sqlite3.OperationalError` and shows a traceback.
- `dashboard/data.py::chart_data` sorts quantile columns as text. `summary` and `dimensions` assume 15-minute steps.
  The page clock uses the first space's time zone and fails with no space files. `health()` is tested but no longer
  shown.
- `eval/thresholds.py`:
  - `CURRENT = 0.25` ignores `class_conf`.
  - `sweep` ignores the outline de-duplication now used when counting.
  - `recommend` lets any plain-box option beat any outline option, not only on ties.
  - `report` can name a `count.sam2_cNN` column that exists only for levels in `table.conf_levels`.
- `labeling/app.py` does not close its per-run connection.
- `scripts/detector_bench/bench.py`:
  - the usage text omits `+tile`;
  - `+tile` ignores `+tta` and `+clahe`.
- `scripts/detector_bench/score.py`:
  - tunes and reports the monitor threshold on the same photos;
  - scores hand-labelled cabin photos against 7 chairs whatever their label says.

### 5.3 Where the documentation and the code differ

These were corrected in `docs/chronos-data-flow.md` and `docs/dashboard-guide.md` when this document was written.

- Stored input names: the code stores markers `cal.*` and `prof.*`, not the calendar and profile columns.
- The flicker rule applies to all count-like families.
- Building parts are excluded.
- The table cache and history cap exist.
- The clipping kinds.
- The dashboard's header states, the headline rule, the meaning of green, and the last-15 log lines.

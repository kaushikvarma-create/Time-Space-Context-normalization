# Space Behaviour Intelligence: Implementation Plan

Merges two things:
- **Observation layer** (our design): fixed camera, zones, strict-JSON VLM extraction, SQLite storage.
- **Roadmap deck** (Space Behaviour Intelligence, Model Pipeline Roadmap): Chronos-2, LightGBM, behaviour models, conformal anomalies, feedback loop.

Unverified items are marked **[verify]**. Nothing here has been run yet.

---

## Current scope (decided)

**Baseline first:** capture, structure, store, and time-averaged views. Forecasting (Chronos-2, LightGBM), behaviour models, and anomaly scoring are deferred until weeks of data exist.

| Item | Decision |
|---|---|
| Inputs | Photos (stills) via RTSP on demand, every 15 min, 24/7 |
| Spaces | `admin_cabin` (a few people) and `parking` (bikes, cars). Config per space type |
| Position | Grid only (room 8x6, parking 12x8). Cells computed from detector boxes; the VLM never sees the grid. No named zones yet |
| Models | YOLO (benchmark yolo12x vs yolo11x) for counts and boxes; Qwen via LM Studio server for semantics and scene state |
| Objects | New object types are new **rows** in `object_registry` (flagged `auto_new`), never new columns. Names normalised via vocabulary + alias map |
| Mobility | Seeded prior (fixed / semi_static / dynamic), then measured from how often an object stays in the same cell |
| Storage | SQLite. `space_id` on every row; nullable `map_*` columns on `spaces` for linking rooms later |
| Averages | `baseline_cell` and `baseline_space` per (weekday, hour) with sample count n; missing frames excluded, never zero |
| Dashboard | Streamlit, only if small: weekday x hour heatmap, grid heatmap on reference photo, new-objects table |
| Privacy | Keep simple; no identification of people; no licence plates |
| Ground truth | Human corrects proposed labels on about 50 frames (VLM must not grade itself) |

Cameras (from the deepstream work): one Dahua-style NVR, `rtsp://<user>:<pass>@<host>:554/cam/realmonitor?channel=N&subtype=S`. `admin_cabin` = channel 5, `parking` = channel 3, main stream (2880x1620). Credentials live only in `.env` / environment (see `.env.example`), never in the repo.
LM Studio: `http://10.20.0.1:1234`, model `qwen/qwen3.5-9b`. Use `reasoning_effort: "none"` or structured output returns empty `content` (thinking mode swallows the JSON).

First real-frame findings (1 frame per camera, eyeballed, NOT a benchmark):
- Cabin: Qwen counted 4 people (matches); YOLO gave 5 (duplicate box on one seated person). Detector-is-better-at-counting is not supported yet; test with labelled frames.
- Parking: YOLO 12-16 two-wheelers, Qwen 15-20 depending on prompt wording (25+ visible). Dense rows are undercounted by both; treat bike counts as low-confidence and track per-cell occupancy too.
- Qwen's count moved ~25% with a prompt change, so freeze `prompt_version` and re-measure on any edit. Qwen listed a `door` in the cabin that I could not see: likely hallucination, so object presence needs the labelled check too.
- yolo11x vs yolo12x: indistinguishable on these frames (both fast, ~40-90 ms on the 5090). Per-space class whitelist (`detector_map`) is needed to drop noise like refrigerator/tv.
- The camera burns in a timestamp (top-right); masked before the VLM sees the image (`mask_regions`).

Code lives in `src/sbi/` (see README.md for the layout, conventions and how to extend). Refactored 2026-11-13 into a
package with typed config, swappable perception components, numbered migrations, atomic writes and 75 tests.
Live since 2026-11-13: `scripts/tick.sh` runs `python -m sbi tick` from cron every 15 min (log: `data/logs/tick.log`, IST; DB: `data/sbi.db`).
Prompt v2 (fixed `status` list, scene rubrics, `unknown` allowed): tidiness stopped flipping; door_state mostly `unknown` (consider dropping for the cabin).
Labelling: `scripts/label.sh` (blind labelling, then reveals model counts); `PYTHONPATH=src python3 -m sbi eval` reports bias/MAE/within-1 per type and part of day.
Early parking signal (unlabelled, so a hypothesis only): morning bike count rose 6 -> 13 in YOLO; VLM bike count was much jumpier (25,25,20,15) and ~2x YOLO's level. YOLO may be better for trend, VLM for level.

Low-confidence detections (2026-11-13, migration 004): YOLO runs at a 0.05 storage floor and every box is kept in
`detections` with its confidence. The counted set (>= 0.25) is unchanged, verified identical on all stored frames.
The table adds `count.yolo_c05|c10.<type>` and `area.yolo*.<type>` (share of the image covered by boxes).
Why: on one parking frame YOLO counted 9 two-wheelers at 0.25 and 22 at 0.05 (about 20+ by eye). Unvalidated.
Decide per-type thresholds (`detector.class_conf`) from human labels, plus `sbi stability` on night windows.
Open-vocabulary detector tests (docs/research, Part 4): YOLOE undercounts worse than YOLO12x, SAM 2 alone cannot
count packed bikes, SAM 3 is untested (gated weights). SAM 2 masks on YOLO boxes are a later option for
duplicate removal and true mask area.

SAM 2 outlines (2026-11-13, migration 005): SAM 2.1-L outlines every stored YOLO box (about 0.2 s per photo).
Outlines are stored; the table derives `area.sam2*.<type>`, `cellarea.sam2.<cell>.<type>` and `count.sam2*.<type>`
(same-type boxes whose outlines overlap with IoU >= 0.5 count once). On parking frame 24: 22 boxes at >= 0.05
become 19 after outline de-duplication (about 20+ by eye). Unvalidated until labels exist. Known limit: SAM 2
cannot add objects YOLO missed, and one box around a cluster of bikes still counts as one.

Image vectors (2026-11-16, migration 006): DINOv2-base turns every photo into one 1536-number vector (448 x 252
input, [CLS | mean patch], about 40 ms and 0.4 GB). Stored raw per (frame, model); the table derives
`emb.dinov2-base.pc1..pc4`, `drift_prev` and `drift_typical`. Probe before building: the camera's burned-in clock
does not move the vector (night-to-night change 0.0055 with it, 0.0053 with it blanked), and night-to-night noise
is 14x (cabin) to 100x (parking) smaller than the day-to-night change. On the 606 stored frames the columns track
occupancy without any detector (parking pc1 vs motorcycle count r = 0.89; cabin drift_typical vs people r = 0.91;
both confounded by the daily cycle, so not independent evidence yet). Unvalidated until labels exist. Per-cell
vectors (a grid of local embeddings) are the natural next extension and are not stored yet.

Probes (2026-11-16, migration 007): SigLIP 2 base (375M) scores every photo against each space's plain-English
prompts (18 per space, 6 concepts of 3 paraphrases); raw logits stored per (frame, scorer, prompt); the table
averages paraphrases into `probe.<scorer>.<concept>`. Tested first on 78 frames: paraphrase agreement mostly 0.8-0.98;
base and so400m (1.1B) were a wash, so base. On 308 slots, cabin "people" r = 0.92 and parking "bikes" r = 0.78 with the
detector counts (0.92 / 0.72 inside 09-19h, so not just the daily cycle); cabin tidy/messy (~0), parking "car"
(0.01) and "person" (tracks brightness) do not hold up. Embedding the VLM's own summary text was tested and rejected:
wording varies at night (night noise as large as the 3-day spread in the cabin). Next extractors on the list: metric
depth, floor/"stuff" segmentation, pose keypoints (no identification), per-cell patch features, pixel change and
camera-health flags.

Known issues / not yet done:
- Subjective VLM fields are unstable (cabin `tidiness` flipped tidy -> messy 30 s apart; `door_state` reported with no door visible). Do not use for alerts until measured.
- VLM `status` is free text and often useless ("occupied or empty"). Constrain to an enum per object type.
- Nested chair boxes in the cabin are not deduped (chairs legitimately overlap); a car was once read as a truck and a `bus` appeared in YOLO.
- New-object path verified only on synthetic labels; the real model followed the vocabulary so far.
- Not built: mobility measurement, appeared/gone events, alert logic, dashboard, 50-frame labelled check.
- Demo DB at `data/demo.db` has n=3 in a single (Thu, 16h) bucket, so its averages show the shape only.

---

## 0. Decisions to lock before writing code

| # | Decision | Options | Recommendation | Why it matters |
|---|---|---|---|---|
| D1 | Capture cadence | hourly stills / 15-min stills / video + event triggers | **DECIDED: photos (stills) first.** Cadence still open: 15-min preferred, hourly acceptable as v0. See section 1a | The deck's 15-min grid and 1h-ahead forecasts assume it. At hourly, 4 weeks is about 670 rows per series, too thin for weekly seasonality |
| D2 | Privacy posture | store images / blur faces / counts only | **Blur faces at ingest, keep raw images N days, keep structured data indefinitely** | Admin cabin. Check company policy and local law first |
| D3 | Identify individuals? | yes / no | **No.** Counts and anonymous activity only | Avoids most compliance risk |
| D4 | Pilot scope | one space | **One common area, one fixed camera** | Prove the loop before generalising |
| D5 | VLM serving | local Qwen via vLLM / API | **Local Qwen (vLLM)** if GPU allows | Keeps images on-prem; cost is flat |
| D6 | Primary goal | analytics vs alerting | Decide with stakeholder | Changes how much Phase 6 matters |
| D7 | Timezone and clock | | **Store UTC plus local TZ name** | One clock is a deck rule |

Output of Phase 0: a one-page `DECISIONS.md` with these answered.

---

## 1. Target architecture

```
Camera (fixed)
  -> Ingest (every 15 min): grab frame, blur faces, save image, log capture status
  -> Engine 1: [cheap change gate] -> VLM pass 1 (list) -> VLM pass 2 (zone + attrs)
  -> Validate + normalise (schema check, label vocabulary, zone check)
  -> Memory (SQLite/Postgres): frames, detections, space_state, events, forecasts, alerts
  -> Engine 2: baselines -> Chronos-2 (zero-shot) -> LightGBM (per space) -> behaviour models
  -> Engine 3: numeric / behaviour / layout / rule signals -> conformal score -> VLM explanation -> alert
  -> Review UI -> feedback labels -> retraining -> model registry
```

**Key principle:** raw output is always kept, so any layer can be recomputed after prompt or model improvements.

### 1a. Photos-first adjustments and Engine 1 spec

The deck assumes video. With stills the following change:

| Deck assumption | With stills |
|---|---|
| Detect + **track** | No tracking (gaps too large for identity). Per-frame detection only |
| Events: entered / left | Interval diffs between frames, marked `resolution=interval`, not observed events |
| Survival / FIM-PP behaviour models | Durations quantised to cadence; treat as optional experiments (Phase 6) |
| Clips as evidence | Key frames only |
| Numeric forecasting (Chronos-2, LightGBM) | Unaffected. This is the primary path |

**Engine 1 is a hybrid, not VLM-only** (changed from earlier draft):

| Output | Source | Reason |
|---|---|---|
| Person count, boxes, zone from box centre | Person detector (YOLO-class **[verify** model choice, benchmark on our camera**]**) | VLMs are weak at counting; detectors are cheap |
| Object presence (chairs, laptops, etc.) | Detector for common classes, VLM for the rest | Cover vocabulary gaps |
| Activity, posture, scene state (door, lighting, tidiness) | VLM (Qwen) | Boxes cannot express these |
| Cross-check | Compare detector count vs VLM count | Disagreement beyond 1 flags the frame for review |
| Layout change | Reference-frame comparison (feature/embedding diff); VLM only to describe | Does not need the VLM to detect |

Bias to test for explicitly (more dangerous than random noise): undercounting at night, in specific zones, or when people are seated or occluded. Report accuracy per hour and per zone in Phase 2.

Engine 1 needs its own spec page before Phase 1 ends: chosen detector, VLM, prompt version, run frequency, compute per frame, and cost per day.

---

## 2. Repository layout (proposed)

```
Time-Space-Context-normalization/
  PLAN.md
  DECISIONS.md
  config/
    space_office_common.yaml      # zones, vocabulary, cadence, timezone
    reference_frame.jpg
  src/
    ingest/        capture.py, blur.py
    vlm/           prompts.py, extract.py, validate.py, normalise.py
    store/         schema.sql, db.py
    features/      grid.py, calendar.py, lags.py
    models/        baselines.py, chronos_zs.py, lgbm.py, behaviour.py
    anomaly/       signals.py, conformal.py, explain.py
    eval/          vlm_eval.py, forecast_eval.py, anomaly_eval.py
  data/            (gitignored) images, db, labels
  notebooks/
  tests/
```

---

## 3. Phased plan

Timeline assumes one pilot space and one person at roughly 50 to 70% time. The deck's 12 weeks is kept as the reference. Data collection time cannot be compressed.

### Phase 1: Perception layer (Week 1-2) [our design; feeds deck Stage 2]

**Goal:** a frame becomes a reliable structured record.

**Tasks**
1. Mount or confirm the fixed camera. Save a reference frame.
2. Define zones. Draw polygons on the reference frame and write them to `space_office_common.yaml`. Aim for 5 to 8 zones.
3. Define the controlled vocabulary (about 20 to 40 object labels, plus activity and posture labels).
4. Write the JSON schema (see section 4) and use constrained decoding (vLLM guided JSON) **[verify** the exact flag for your vLLM version**]**.
5. Write the two-pass prompts. Pass 1 lists people and objects. Pass 2 assigns zone and attributes per item.
6. Validation layer: reject or retry on schema failure, unknown zone, unknown label, or count mismatch between passes.
7. Normaliser maps synonyms to the vocabulary ("couch" to "sofa").
8. Add a change gate (cheap): pixel or embedding difference versus the previous frame. If unchanged, copy the previous state with `provenance=inferred`. This is a cost control for later.
9. Ingest scheduler (cron or systemd timer) with retry and a capture-status log (`ok`, `camera_down`, `bad_frame`).

**Position strategy (decision tree)**
- Default: VLM chooses a zone from the given list.
- If a bbox is needed, add a grounding detector (Grounding DINO or YOLO-World) and assign zone by bbox centre inside polygon. Compare against VLM zone and log disagreements.
- Camera drift check: compare each day's frame to the reference via feature matching. Alert if displacement exceeds a threshold.

**Exit gate G1:** pipeline runs unattended for 3 days with at least 98% of frames producing a schema-valid record or an explicit `camera_down`/`failed` status.

### Phase 2: Ground truth and VLM evaluation (Week 2-3, overlaps data collection)

**Goal:** know how wrong the VLM is, per output type.

**Tasks**
1. Sample 100 frames stratified by hour of day, weekday/weekend, and lighting. Include night and hard cases.
2. Hand-label: person count, zone per person, object presence per vocabulary item, posture/activity.
3. Build `vlm_eval.py` and report per field:
   - person count: MAE, exact-match rate
   - zone assignment accuracy
   - object precision/recall per label
   - activity accuracy
   - accuracy by hour, to expose night degradation
4. Iterate prompt, vocabulary, and model (compare Qwen sizes; test with and without two-pass).
5. Freeze `prompt_version` and `model_version`. Record both on every record.

**Exit gate G2 (fills the deck's blank `[__]%`):** agree thresholds with the stakeholder. Suggested starting targets, to be adjusted after seeing real numbers:
- person count within ±1 in at least 90% of frames
- zone accuracy at least 90%
- no label with precision under 80% used for alerting

If counts miss the gate, forecasting on counts is not trustworthy. Fix perception before Phase 4.

### Phase 3: Data collection and memory (Week 2-6)

**Goal:** 4+ weeks of clean, time-gridded data.

**Tasks**
1. Create the schema (section 4). Keep `frames`, `detections`, `space_state`, `events`, `evidence`.
2. Derive `space_state` per interval: counts per zone, activity shares, object presence, lighting, door state.
3. Derive `events` by diffing consecutive states (entered, left, became_empty, object_moved, object_new, object_gone). At hourly cadence these are coarse; mark `resolution=interval`.
4. Enforce deck rules in code:
   - **Missing is not zero:** `camera_down` rows have NULL values, never 0.
   - Provenance column: `detected | inferred | sensor`.
   - Confidence and model/prompt version on every value.
   - One clock.
5. Evidence policy: keep key frames for N days (suggest 30), keep alert evidence longer; clips only if video exists.
6. Daily data-quality report: % missing, % inferred, schema failure rate, mean confidence.
7. Weekly: label 20 more frames to track VLM drift.

**Exit gate G3:** at least 4 weeks of data, under X% missing (agree with stakeholder; suggested 10%), VLM accuracy still within G2 on the latest labelled sample.

### Phase 4: Baselines and zero-shot forecasting (Week 3-7, start as soon as ~2 weeks exist)

**Goal:** the first forecasts and a benchmark to beat. Deck Stage 1.

**Tasks**
1. Baselines first (cheap, required for gates):
   - same time last week
   - mean by (weekday, hour)
   - seasonal naive with median
2. Aggregate to the chosen grid (D1). Series: total people, per-zone people, per-zone occupied flag.
3. Chronos-2 zero-shot **[verify** the package and API, e.g. `chronos-forecasting`, model id on Hugging Face, and covariate input format**]**:
   - inputs: history, calendar covariates, holidays; bookings if available
   - mask missing values (do not fill with 0)
   - horizons: 1h, 4h, 1 day ahead
   - store quantiles (10/50/90) with `model_version`; never overwrite
4. Backtest by rolling origin on time-ordered data. Never random.
5. Metrics: MAE or WQL against baselines, and **80% interval coverage**.
6. Optional challengers behind one interface: TimesFM, Toto, others named in the deck **[verify** current availability and licences**]**.

**Exit gate G4 (from deck):** beats "same time last week" on backtest, and the 80% interval covers about 80% of actuals (suggest 75 to 85%). If it fails, the baseline serves and we document why.

### Phase 5: Training data and per-space numeric model (Week 5-8)

Deck Stages 3 and 4A.

**Tasks**
1. Feature builder: calendar, lags (1 step, 1h, 1 day, 1 week), rolling means, neighbouring-zone values, Chronos-2 forecast as a feature.
2. Labels: future value at +1h, +4h, +1 day.
3. Split by time: earliest ~60% train, next ~20% validation, latest ~20% test. Check no leakage (assert max train time < min test time).
4. LightGBM quantile models (10/50/90). Tune lightly. Record feature importance.
5. Champion/challenger per space: lower backtest error serves; Chronos-2 stays fallback for gaps and new spaces.
6. Register the model with version, data window, and backtest score in a registry table.

**Caveat:** with 4 to 6 weeks of data, LightGBM may not beat Chronos-2 or the baseline. That is a valid result; the gate decides.

**Exit gate G5:** the chosen champion beats baseline on the held-out test period; coverage within tolerance. Repeat the check separately for weekends, night, and post-downtime periods.

### Phase 6: Behaviour models (Week 6-9)

Deck Stage 4B. Do this after numeric forecasts work.

**Tasks**
1. Define states for the space (suggest: `empty`, `sparse`, `active`, `crowded`, plus optional `meeting_like` or `break_like` from activity labels). Keep few; each needs enough examples.
2. Day-one methods (no training): transition table from observed data, typical durations per state.
3. Trained: LightGBM classifier for next state with probabilities; survival model (e.g. Cox or gradient-boosted survival) for time-in-state **[verify** the library choice**]**.
4. Event timing (FIM-PP) is an experiment only. Run it only if events are dense enough; at coarse cadence it likely is not.
5. Evaluate: macro-F1 for next state against the "most frequent transition" baseline; duration MAE or C-index against category default.

**Exit gate G6:** beats the category/transition-table default on the held-out period.

### Phase 7: Anomaly detection (Week 9-11)

Deck Stage 5. Start simple, add signals one at a time.

**Signals**
1. **Numeric:** actual outside the forecast interval for k consecutive steps. Calibrate the threshold with conformal prediction on a rolling window to target a chosen false-alarm rate **[verify** exact method, e.g. split or adaptive conformal**]**.
2. **Behaviour:** rare transition (probability under p) or duration above the survival model's high quantile.
3. **Layout:** per-object persistent presence versus the reference layout. Flag if a fixture is missing or moved for at least m consecutive frames. Guard against false alarms from camera drift.
4. **Rules:** human-editable YAML rules, e.g. "door open and zero people 22:00-06:00". Start with 5 to 10 hand-written rules.

**Scoring and explanation**
- Combine signals into one score and severity tier (low / medium / high) with transparent weights.
- A VLM explains the flagged frame in plain language. The explanation is descriptive only; **unusual is not dangerous**, so a human confirms.
- Alert payload: score, signals fired, forecast vs actual, key frames, timestamp.

**Exit gate G7:** run in shadow mode (no notifications) for 2 weeks. Reviewers label alerts. Agree precision/recall targets and max false alarms per day with the stakeholder (deck's `[__]`).

### Phase 8: Production loop (Week 11-14)

Deck Stage 6.

**Tasks**
1. Review UI (simple): alert list, frames, confirm/reject, optional reason.
2. Feedback store: reviewer labels feed the anomaly thresholds, rules, and training labels.
3. Scheduled retraining (weekly) with quality checks and a promote-only-if-better rule.
4. Drift watch (daily): VLM schema failure rate, mean confidence, forecast error, interval coverage, camera displacement.
5. Alerting on the system itself: camera down, VLM down, DB full.
6. Retention job: delete images past the retention window; keep derived data.
7. Runbook: how to roll back a model, re-extract with a new prompt, add a new space.

**Exit gate G8:** 2 weeks of stable operation with drift watch green and review turnaround acceptable.

---

## 4. Data model (starter DDL)

```sql
CREATE TABLE frames (
  frame_id      TEXT PRIMARY KEY,           -- ISO ts of capture
  ts_utc        TEXT NOT NULL,
  ts_local      TEXT NOT NULL,
  tz            TEXT NOT NULL,
  date          TEXT, dow INTEGER, hour INTEGER,
  image_path    TEXT,                       -- NULL after retention
  capture_status TEXT NOT NULL,             -- ok | camera_down | bad_frame | failed
  model_version TEXT, prompt_version TEXT,
  raw_output    TEXT                        -- full VLM JSON, always kept
);

CREATE TABLE detections (
  frame_id TEXT REFERENCES frames(frame_id),
  idx INTEGER,
  label TEXT NOT NULL,                      -- controlled vocabulary
  zone TEXT,
  bbox TEXT,                                -- normalised [x1,y1,x2,y2], nullable
  attrs_json TEXT,
  confidence REAL,
  provenance TEXT NOT NULL                  -- detected | inferred | sensor
);

CREATE TABLE space_state (
  ts_utc TEXT, zone TEXT,
  people_count REAL,                        -- NULL if missing, never 0 for downtime
  occupied INTEGER,
  activity_json TEXT,
  objects_json TEXT,
  provenance TEXT, confidence REAL,
  PRIMARY KEY (ts_utc, zone)
);

CREATE TABLE events (
  event_id INTEGER PRIMARY KEY, ts_utc TEXT, zone TEXT,
  type TEXT, resolution TEXT, details_json TEXT
);

CREATE TABLE forecasts (
  made_at TEXT, target_ts TEXT, zone TEXT, series TEXT,
  q10 REAL, q50 REAL, q90 REAL,
  model_name TEXT, model_version TEXT,
  PRIMARY KEY (made_at, target_ts, zone, series, model_name, model_version)
);

CREATE TABLE alerts (
  alert_id INTEGER PRIMARY KEY, ts_utc TEXT, score REAL, severity TEXT,
  signals_json TEXT, explanation TEXT, frame_ids TEXT,
  review_status TEXT, reviewer_note TEXT
);

CREATE TABLE model_registry (
  model_id TEXT PRIMARY KEY, kind TEXT, trained_on_from TEXT, trained_on_to TEXT,
  backtest_metric_json TEXT, status TEXT   -- candidate | champion | retired
);
```

## 5. Extraction schema (per frame)

```json
{
  "people_count": 3,
  "people": [{"zone": "sofa_area", "posture": "seated", "activity": "laptop", "confidence": 0.8}],
  "objects": [{"label": "chair", "zone": "table_A", "state": "empty", "confidence": 0.7}],
  "scene": {"lighting": "daylight", "door": "closed", "tidiness": "ok"},
  "notes": ""
}
```

Prompt skeleton (final wording to be tuned in Phase 2):
- System: role, "describe only what is visible, do not guess identity".
- Context: zone list with short descriptions plus the reference image or layout text; allowed labels; output schema.
- Rules: "if unsure, lower confidence; never invent items; count only fully or mostly visible people".

---

## 6. Gates summary

| Gate | Phase | Measure | Target (initial, adjustable) |
|---|---|---|---|
| G1 | 1 | Unattended valid-record rate | at least 98% over 3 days |
| G2 | 2 | VLM count/zone/label accuracy on labelled sample | count within ±1 in 90%+, zone 90%+ |
| G3 | 3 | Data completeness and quality | 4+ weeks, under 10% missing |
| G4 | 4 | Beats last-week baseline, 80% interval coverage | coverage 75 to 85% |
| G5 | 5 | Champion beats baseline on test period | also holds on weekends, night |
| G6 | 6 | Next-state F1, duration error | beats default |
| G7 | 7 | Alert precision/recall, false alarms per day | agreed with stakeholder |
| G8 | 8 | 2 weeks stable operation | drift green |

## 7. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| VLM count noise | Corrupts all forecasts | G2 gate, per-hour accuracy tracking, change gate and smoothing, prefer occupied/not-occupied if counts are unreliable |
| Hourly cadence too sparse | No 15-min grid, weak seasonality | Decide D1 early; sample faster during pilot |
| Camera shifts | Zones and layout signals break | Daily drift check, alert on displacement |
| Night/low-light errors | False anomalies | Per-hour eval, separate thresholds, IR camera if needed |
| Little training data | LightGBM underperforms | Keep Chronos-2 as fallback; gates decide |
| VLM or prompt change breaks history | Inconsistent series | Version everything; re-extract from kept images; never mix versions in a training window silently |
| Privacy or compliance | Project blocked | D2/D3 up front; blur at ingest; retention job |
| Alert fatigue | Reviewers ignore alerts | Shadow mode, false-alarm budget, severity tiers |
| GPU or cost limits | Pipeline falls behind | Change gate, smaller Qwen variant, batch off-peak |
| Over-engineering | Time lost | Phases 6 to 8 only after G5 passes |

## 8. Timeline (reference)

| Weeks | Phases |
|---|---|
| 0 | Decisions (section 0) |
| 1-2 | Phase 1 perception |
| 2-3 | Phase 2 labelling and VLM eval; collection running |
| 2-6 | Phase 3 data collection |
| 3-7 | Phase 4 baselines + Chronos-2 |
| 5-8 | Phase 5 features + LightGBM |
| 6-9 | Phase 6 behaviour |
| 9-11 | Phase 7 anomalies (shadow mode) |
| 11-14 | Phase 8 production loop |

## 9. First five actions

1. Answer D1 to D7 and write `DECISIONS.md`.
2. Confirm camera position and save a reference frame.
3. Define zones and vocabulary in `config/`.
4. Stand up Qwen via vLLM and run Pass 1/2 on 20 frames.
5. Create `schema.sql` and a minimal ingest script.

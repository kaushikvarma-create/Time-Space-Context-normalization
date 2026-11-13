# Turning a photo into numbers without losing information

Date: 2026-11-13 (IST). Data: 34 frames (17 per camera), 09:11-12:30 IST, one weekday morning.
Status: research + experiments. Nothing here is in production yet.

## The question

Chronos-2 (and any time-series model) only takes numbers. If we reduce each photo to a few hand-picked counts
(people, bikes, cars) we throw away resolution (where things are) and context (everything we did not think to
count). How do we keep as much of the photo as possible, and how do we handle YOLO and the VLM disagreeing?

## Part 1. "When YOLO and the VLM disagree, rerun and average"

### What the experiments show

**Reruns of the same input are identical.** YOLO is deterministic; the VLM at temperature 0.1 returned 24, 24, 24.
Averaging identical numbers changes nothing. Reruns only help if each one looks at the photo *differently*.

**YOLO confidence threshold (E1, all 34 frames).** Mean count per frame at different thresholds:

| | 0.05 | 0.10 | 0.15 | 0.25 (current) | 0.40 | 0.60 | VLM |
|---|---|---|---|---|---|---|---|
| parking two-wheelers | 17.4 | 12.4 | 10.6 | 9.0 | 7.4 | 3.2 | 15.5 |
| parking cars | 1.6 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | |
| cabin people | 4.9 | 4.4 | 4.3 | 4.1 | 3.6 | 3.0 | 4.0 |

- 48% of all two-wheeler detections score below 0.25, and 96% of car detections score above 0.5.
- **The right threshold differs by class.** One global 0.25 is wrong for packed bikes and irrelevant for cars.
- I visually inspected the low-confidence crops (0.05-0.40).
  - Most "motorcycle" boxes are **real, partly hidden two-wheelers**.
  - Person boxes are mixed: real seated people half-hidden by chairs, plus junk (one on the burned-in "Parking" text overlay, one on a thin sliver).

**Different views of one frame (E2, parking 12:30 IST).** My reference count from zoomed crops is **about 25
two-wheelers (±4)**, 4 cars and 1 person. This is one frame counted by me, not a labelled ground truth.

| Method | Two-wheelers |
|---|---|
| YOLO current (1920px, conf 0.25) | 8 |
| YOLO native 2880px | 5 |
| YOLO conf 0.10 | 13 |
| YOLO test-time augmentation | 12 |
| YOLO SAHI-style tiles 2x2 / 3x2 (conf .15) | 12 / 12 |
| VLM temp 0.1 (x3) | 24, 24, 24 |
| VLM 2x2 tiles, summed | 30 |
| VLM temp 0.9 (x7) | 13-43, median 32 |

- **Cars and people:** YOLO gets them right.
- **Dense rows of two-wheelers:** YOLO undercounts by half or more whatever the setting. COCO YOLO is the wrong tool here.
- **Averaging is the wrong fix.** A plain average of YOLO (≈10) and the VLM (≈24) gives ≈17, which is worse than the VLM alone.
- **High-temperature sampling** gives a wide spread. That spread is useful as an *uncertainty* signal, but the median overshoots.

### What the literature says (details in the research notes below)
- **VLM counting** degrades as the count grows and when object types are mixed (VLMCountBench 2025; LVLM-Count, TMLR 2026).
- **Whole-image vs tiled bias:** whole-image VLM counts tend to under-count, and tiled counts tend to over-count because of edge effects. Combining the two cuts the error a lot (WS-COC 2026). Our frame showed the same pattern: 24 whole vs 30 tiled, reference ≈25.
- **Point-then-count** prompting beats asking for a number directly (Molmo/PixMo 2024).
- **Self-consistency:** no study of it specifically for VLM counting was found. Expect a median of samples to reduce variance but not remove bias.
- **Purpose-built counters:** open-world counters such as **CountGD** reach MAE ≈3.7 zero-shot on a car-park benchmark (CARPK).

### Recommendation
1. **Store every detection down to conf 0.05 with its confidence**, and apply per-class thresholds when deriving counts.
   - The threshold then becomes a setting we can change and re-derive, with no need to re-run images.
2. **Set per-class thresholds from labelled frames**, choosing the threshold that minimises count error rather than the usual detection score (mAP).
   - Until labels exist, keep 0.25 for people and cars. For two-wheelers, no threshold makes YOLO reliable anyway.
3. **Two-wheelers need a different estimator:** VLM whole-image plus VLM tiles combined, and/or CountGD, and/or area occupancy from segmentation masks (see Part 2).
4. **Combine estimators by calibrating, not by averaging raw numbers.**
   - On ~50 labelled frames, fit a simple correction for each estimator (true ≈ a + b·estimate).
   - Weight each estimator by how accurate it proved, and store the combined value plus an uncertainty band.
   - Use disagreement and sample spread as that uncertainty.
5. **Extra views only when they are informative:** for example, VLM tiles and samples when estimators disagree by more than X, so cost stays bounded.

## Part 2. Keeping as much of the photo as possible

### Principle: store a rich, versioned feature vector per photo; let each model choose what it reads

The photo is always kept, so nothing is ever lost permanently. The question is which numbers we compute and keep
per frame. Proposal: a long-format `frame_features(frame_id, name, value, extractor, version)` table, filled by
pluggable extractors. The forecaster's series builder selects from it. Adding a new extractor never changes the
schema.

### Layers, with what our data says about each

| Layer | Example numbers | Dims | Evidence so far |
|---|---|---|---|
| L0 image statistics | brightness, contrast, sharpness | 3 | already stored; free |
| L1 calibrated counts **per class per grid cell** | people in R2C3, cars per cell | cells × few classes, sparse | YOLO is fine for people and cars; bikes need L1b |
| L1b dense-object estimator | two-wheelers via VLM whole+tiles / CountGD | 1-2 per region | the VLM was closest to the reference on one frame; needs labels |
| L2 area occupancy per region | fraction of each cell covered by "motorcycle" / "person" masks (SAM 3) | cells × few | literature only; robust under occlusion; not tested yet |
| L3 global embedding → PCA | DINOv2/v3 or SigLIP 2, top 8-16 components | 8-16 | parking: PC1 explains 60%, corr +0.78 (VLM) / +0.88 (YOLO) with bike counts; cabin: corr +0.84/+0.86 with people |
| L3b per-cell embedding change | 1 - cos(cell now, cell at same weekday-hour baseline) | = number of cells | per-cell change map correctly localised where bikes and cars arrived and the person walked, with no class names |
| L3c novelty score | distance of the whole frame to its weekday-hour baseline | 1 | global drift tracked bike arrivals (corr +0.82) |
| L4 VLM scene states | lighting, door, tidiness, occupancy level | few categorical | stable after prompt v2; categorical covariates |
| L5 text concept probes (CLIP) | similarity to "an empty office", "people in a meeting"... | per probe | **weak and inconsistent**: corr ±0.4, and "an office with many people working" was *negatively* correlated with people (-0.14). Not reliable yet |

**Caveat on the correlations.** They come from one morning (17 frames per camera) in which bikes, people and time of day
all rose together. Part of every correlation is just "it got later". They show the signals are plausible, not that
they work. A few weeks of data, including evenings and weekends, is needed.

### What goes into Chronos-2
- **Targets (what we forecast):** a small number of calibrated, meaningful series, e.g. people per space, two-wheelers
  in parking, occupied-cell fraction. They must be interpretable, because the alerts and baselines are built on them.
- **Covariates (context that helps):** calendar (known future), PCA components, novelty, per-region change or occupancy
  aggregated to a few regions, VLM states, brightness.
- **Raw embeddings stay in storage, not in the model.** 384-768 dimensions per frame (per-cell features would be
  18-37k) is far too many covariates. PCA is fitted on history and refit rarely, with its version stored.
### Chronos-2 inputs (checked against the paper, the HF card and the `chronos-forecasting` source; V = verified)
- **Multivariate targets:** yes. `target` can be a list of columns. [V]
- **Past-only covariates:** yes, any extra column in `df`. [V]
- **Known-future covariates:** yes, columns that are also in `future_df`. [V]
- **Categorical covariates:** yes, passed as strings. [V]
- **Missing values:** NaN is supported in both the context and the covariates. It is masked and left out of the scaling. [V]
  - **But the timestamps must have no gaps.** A camera-down step must be a row with NaN; leaving the row out breaks frequency inference. [V] This matches our "missing is not zero" rule exactly.
- **Context and horizon:** context 8192 steps (about 85 days at 15 minutes); horizon 1024 steps natively. Our 4- and 96-step horizons and 1-4 weeks of context fit. [V]
- **API:**

  ```python
  Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda").predict_df(
      context_df, future_df=..., prediction_length=96, quantile_levels=[0.1, 0.5, 0.9],
      id_column="id", timestamp_column="timestamp", target=...)
  ```

  - The output has one row per future step, with a column for each quantile. [V]
  - Model sizes: 120M, or 28M (`autogluon/chronos-2-small`). [V]
  - Licence: Apache-2.0. [V]
- **Many covariates:**
  - The report sets no cap and shows no study of how many covariates help or hurt. [V]
  - Its own docs say cross-series learning "must be tested". [V]
  - Community reports say unrelated series can slightly degrade forecasts, and that simple covariate effects are captured less well than by TabPFN-TS. [S]
  - So we add covariates in steps and measure each step: counts plus calendar, then plus per-region counts, then plus PCA components.
- **Challengers:**
  - **Timestral** (Mistral, Oct 2026) matches Chronos-2 on input types. [V] The claim that it is more accurate is the vendor's own. [S]
  - **TimesFM 3.0** has covariates but fills gaps by linear interpolation instead of masking them. Its weights are non-commercial. [V]
  - **Toto 2.0** has no covariates yet. [V]
  - **Moirai 2.0** dropped covariates and is CC-BY-NC. [V]
- **Prior use of camera-derived inputs in any time-series foundation model:** none found. The closest is PDFM (arXiv 2411.07207), which pairs geospatial embeddings with TimesFM. The ablations are therefore part of the work, not a formality.

## Prior work (survey summary)

[V] = checked on the source page, [S] = from abstract or summary only.
- Jacobs, Roman, Pless, *Consistent Temporal Variations in Many Outdoor Scenes*, CVPR 2007. PCA of long webcam sequences gives time series that track weather and season. This is the closest precedent for embedding → PCA → time series. [S]
- Rolf et al., *MOSAIKS*, Nature Comms 2021. Store one generic image featurisation, then predict many targets later. [S]
- Kim et al., Sensors 2023. Camera-based occupancy counts had to be corrected before use in energy forecasting. [S]
- CountGD (arXiv 2407.04619): open-world counting with text and/or exemplars, CARPK MAE 3.83 / 3.68 zero-shot. [V]
- VLMCountBench (arXiv 2510.04401) [V]; LVLM-Count (arXiv 2412.00686) [V]; WS-COC (arXiv 2602.12774) [V via summary];
  Molmo/PixMo point-then-count (arXiv 2409.17146) [V].
- SAHI sliced inference (arXiv 2202.06934). [S]
- SAM 3 (Meta, Nov 2025): instance masks for every match of a noun phrase. [V]
- DINOv3 (arXiv 2508.10104), SigLIP 2 (arXiv 2502.14786): image embeddings, the latter with matching text embeddings. [S]
- PatchCore / VadCLIP: embedding distance-to-normal for anomalies. [S]
- **Gap:** no published pipeline was found that feeds frozen CCTV frame embeddings into a time-series foundation model as covariates. We would be doing something new, so validate carefully.
- No public benchmark of dense parked two-wheelers exists. We need our own labelled set.

## Decisions needed / next steps
1. **Labelling** (blocks calibration of everything): about 50 frames per camera, spread over hours, with per-class counts.
2. **Store all detections ≥0.05** and move thresholds into derivation (small change).
3. Add a `frame_features` table and an extractor interface. First extractors: L0, L1 per-cell counts, L3 DINOv2 global plus per-cell change, L3c novelty.
4. **Two-wheeler estimator bake-off** on the labelled set: VLM whole + tiles combined, CountGD, SAM 3 area occupancy, YOLO + SAHI.
5. Revisit L5 text probes only after labels exist.

## Part 3. Open vocabulary, links, and the Chronos table (experiments 13:00-14:30 IST)

### How to make the VLM name things
| Prompt | Distinct names (cabin / parking) | Same names photo-to-photo | Problem |
|---|---|---|---|
| Open ("list every kind of object") | 29 / 18 | 76% / 67% | synonyms (bag/backpack, water bottle/bottle), building parts, small items flicker |
| Registry-aware ("reuse these names") | 15 / 10 | 100% / 90% | **priming**: see below |

- **Decoy test.** Six absent objects (refrigerator, dog, sofa, bicycle, whiteboard, umbrella) were added to the "reuse these names" list. The VLM reported all six in half of the cabin photos. The open prompt reported none.
- **The old production prompt was also primed.** It had a "preferred labels" hint, and over 20 frames that hint produced:
  - `door`, `whiteboard`, `plant` and `screen`, which the open prompt never reported;
  - renamings: `desktop_pc` for computer tower, `book` for notebook.
- **Rule: the VLM never sees a list of names.** Prompt v3 is open-vocabulary. Naming is fixed afterwards, in text.

### Merging and linking names (the curator)
- **One-shot "group these 40 names" (Qwen 9B): unusable.** Everything came out as category "structure", and it merged almost nothing.
- **One question per pair: still bad.** It read "related" as "a kind of": "desk is_a keyboard", "monitor is_a mouse".
- **Two independent yes/no questions** ("is every A a B?" and "is every B an A?"), on pairs shortlisted by embedding cosine ≥0.80:
  - On the experiment set: 79 pairs gave 11 links, all correct.
  - On live data: 97 pairs gave 10 links. 9 were sensible: wall_mounted_fan = wall_fan, which is a kind of fan; green_bottle, pc_tower, poster, gate and fence each linked to their parent.
  - 1 was wrong: "motorcycle is_a bike". The cause was an ambiguous name. Our `bike` meant bicycle, but in Indian English "bike" usually means a motorbike. The link was rejected and the config type renamed to `bicycle`.
- **Links:**
  - `same` merges names into one dimension.
  - `is_a` keeps both names and adds a roll-up dimension for the parent.
  - Raw labels are stored forever, so any link change re-maps history.
  - Every decision is audited and reversible.

### Relationship triples
- VLM triples such as (person, sitting_on, chair) overlapped 40-70% between photos 15 minutes apart. Part of that change is real (people move), part is naming drift.
- Prompt v3 uses a fixed list of predicates, and the subject and object names go through the same taxonomy.
- Geometric links (detector boxes overlapping) are deterministic and free, e.g. car-overlaps-motorcycle.

### The table handed to Chronos
- **Size today:** 199 dimensions for the cabin and 105 for parking from 19 slots. That covers counts from both models, roll-ups, per-cell counts, relations, scene fields, image statistics and calendar.
- **Eligibility (hard rules, with the reason recorded):**
  - excluded: building parts (`wall`, `floor`, ...), counts seen in fewer than 3 slots, constant columns, low coverage.
  - result: cabin 135 of 199 eligible, parking 59 of 105.
- **Default inputs:**
  - a capped set of 24, with key series always included;
  - ranked by lag-1 autocorrelation. Real occupancy changes smoothly; model noise flickers. Parking cars scored 0.94, motorcycles 0.6-0.8, walls -0.04.
  - Smoothness only filters once there is about a day of slots. With 19 points it is ±0.23 noise.
  - People are real but fast-changing (autocorrelation ≈0), so they are always included rather than filtered.
- **Verified with the real model:** `amazon/chronos-2` (chronos-forecasting 2.3.2) accepted the parking table as built: 2 targets, 24 past covariates, 5 known-future calendar columns, and NaN rows for a simulated outage. It returned 10/50/90% quantiles. The forecasts on 19 points are not meaningful yet; this was a format check.

## Part 4. SAM / open-vocabulary detectors (2026-11-13, ~15:30 IST)

**Question:** can a text-prompted detector measure every name the VLM discovers, and fix the dense two-wheeler undercount?

- **SAM 1/2:** class-agnostic (masks without names), so they cannot produce per-type counts on their own.
- **SAM 3 (text prompt → every instance + presence score):** the real candidate. Its weights are gated on Hugging Face; access had not been granted when this ran. The script is ready (`sam3_eval.py`, same design as below).
- **YOLOE-26x-seg (ungated, Ultralytics, text-prompted detect+segment), run on all 58 stored frames:**
  - **Prompts:** each frame was prompted with the core types, every raw VLM label for that frame, and 6 decoys.
  - **Decoys:** 0 hits at confidence ≥0.25 in 58 frames. Unlike the VLM, a name list does not make it see absent things.
  - **Cheap:** about 1.2 s/frame at imgsz 1920; 2.0 GB VRAM.
  - **Worse than YOLO12x on COCO types:**
    - Parking motorcycles: 3–6 vs YOLO 6–13 vs VLM 8–20. On frame 24 (by eye, roughly 20+), it boxes clusters of bikes as one.
    - Cabin monitors: 0–2 vs YOLO 4–7 (by eye, about 5 on frame 23).
  - **Class competition:** when "bag" and "backpack" are both prompted, bags land on "backpack". Prompt sets would need the taxonomy applied first.
  - **Finds things COCO cannot:** helmets in parking (0–6 per frame; YOLO has no helmet class). People and chairs match YOLO closely.
- **Verdict:** YOLOE does not replace YOLO12x for counting and does not fix the two-wheeler undercount. It is a possible extra source for non-COCO types only. SAM 3 remains to be tested on the dense two-wheeler case, which is the one that matters.
- **SAM 2.1-L with YOLO12x (COCO), run on parking frame 24 and cabin frame 23:**
  - **Everything mode (no prompts):**
    - It outlines big things: cars, the floor, the kiosk.
    - It leaves the packed two-wheeler rows almost entirely unsegmented (1 scooter out of roughly 20+).
    - It cannot count bikes. 1.7 s per frame.
  - **YOLO boxes → SAM 2 masks:**
    - Counts stay YOLO's.
    - It adds mask area per type, e.g. motorcycles cover 7.0% of frame 24. That is a "how full" number that does not depend on separating each bike.
    - 0.1–0.4 s per frame; 5.3 GB VRAM peak.
  - **The real lead came from YOLO itself:**
    - On frame 24, YOLO12x motorcycles at conf ≥0.25 = 9 and at ≥0.05 = 22 (by eye, roughly 20+).
    - Drawn on the image, most 0.05–0.25 boxes sit on real, distinct bikes in the packed rows; a few are nested or overlapping.
    - Cabin frame 23 at ≥0.05 adds plausible extras (chairs 7→11, monitors 5→9), some of which are probably duplicates.
    - The threshold must be chosen per class from labelled frames, not by eye from one image. This supports storing all boxes ≥0.05 with their confidence and counting at several thresholds; SAM 2 mask overlap can remove nested duplicates.

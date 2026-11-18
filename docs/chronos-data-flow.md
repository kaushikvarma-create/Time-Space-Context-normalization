# How data is fed to Chronos-2

What exactly is handed to the forecasting model, in what shape, and what comes back. Numbers are from the live
system on 18 Nov 2026 at 10:30 IST (5 days of history). Code references are in brackets.

## The short version

```
photo every 15 min
   -> 5 models read it into numbers                      (stored raw in SQLite)
   -> table: one row per 15-minute slot, one column per measured thing        [forecast/table.py]
   -> pick what to forecast (targets) and what to show the model (inputs)     [forecast/inputs.py]
   -> two frames: the past (context) and the next 4 hours (future)            [forecast/table.py: to_chronos]
   -> Chronos-2 predict_df(context, future)                                   [forecast/models.py]
   -> 3 numbers per series per future slot (low, middle, high)
   -> counts clipped at 0, stored in forecast_runs / forecasts                [forecast/runner.py]
```

It runs every hour on the hour, inside the 15-minute scheduled run, right after the newest photo has been read
[forecast/schedule.py]. Nothing is trained on your data: Chronos-2 is used as it comes ("zero-shot").

| | Admin cabin | Parking |
|---|---|---|
| History rows given (15-minute slots) | 487 | 487 |
| Series forecast (targets) | 45 | 31 |
| Past-only inputs | 358 | 159 |
| Known-ahead inputs | 52 (7 calendar + 45 profiles) | 38 (7 calendar + 31 profiles) |
| Columns in the past frame | 457 | 230 |
| Rows in the future frame | 16 (4 hours) | 16 |
| Missing values in the past frame | 0.36% | 0.34% |
| Time to build the table | 5.2 s | 1.8 s |
| Time for Chronos-2 to forecast | 0.29 s | 0.09 s |

## Step 1: the table

One table per space [forecast/build.py: table_for -> forecast/table.py: build_table].

- **Rows:** every 15-minute slot from the first photo to the newest, with no gaps. A slot without a usable photo is
  a row of empty values. It is never dropped and never filled with 0, because Chronos needs evenly spaced rows and
  treats an empty value as "unknown".
- **One photo per slot:** the usable photo closest to the slot start.
- **Columns:** everything extracted, built from the raw stored readings each time. Admin cabin 718 columns, parking
  598. The families (counts, areas, grid cells, relations, image statistics, image vector, description scores,
  scene fields, calendar) are listed in `docs/dashboard-guide.md`.
- **Catalog:** for every column the table records whether it is usable ("model-ready"): values in at least half the
  slots, at least 8 values, not constant, and for counts seen in at least 3 slots and 5% of slots. Admin cabin 416
  usable, parking 232.

## Step 2: what to forecast (targets)

Each space file lists `forecast_targets`: column names or wildcards [spaces.py, inputs.py: resolve_targets].

- Admin cabin: `scene.people_yolo`, `scene.people_vlm`, `count.yolo.*`, `area.sam2.*`, `probe.*`
- Parking: `scene.vehicles_yolo`, `scene.people_yolo`, `count.yolo.*`, `count.vlm.motorcycle`, `area.sam2.*`, `probe.*`

The patterns are matched against usable columns only, in the order written, up to `forecast.max_targets` (100).
Wildcards skip near-copies of a series (the same count at another confidence threshold). Text columns and the
calendar are never targets.

| Targets by family | Admin cabin | Parking |
|---|---|---|
| Scene totals (people, vehicles) | 2 | 2 |
| Counts per object type | 11 | 5 |
| Area covered per type | 11 | 4 |
| Description scores | 21 | 20 |
| **Total** | **45** | **31** |

## Step 3: what else the model sees from the past (past-only inputs)

The live input set is `all_profile` = `["cal", "*", "prof"]` (`forecast.variant` in `config/pipeline.yaml`).

`*` means every usable column that is not a target and not the calendar, with no cap and no filtering
[table.py: all_covariates].

| Past-only inputs by family | Admin cabin | Parking |
|---|---|---|
| Area per grid cell | 102 | 50 |
| Counts (VLM counts, other thresholds, outline-deduplicated) | 78 | 35 |
| Counts per grid cell | 62 | 19 |
| Area per type (box-based, other thresholds) | 55 | 20 |
| Relations between objects | 45 | 18 |
| Image vector (4 main directions + 2 change measures) | 6 | 6 |
| Counts per object group | 6 | 5 |
| Image statistics (brightness, contrast, sharpness) | 3 | 3 |
| Scene fields in words (given as text categories) | 1 | 2 |
| Scene totals not forecast | 0 | 1 |
| **Total** | **358** | **159** |

The model is given their past values only. It is not told their future, and it does not forecast them.

## Step 4: what the model is told about the future (known-ahead inputs)

Two kinds of column appear in both the past frame and the future frame, so the model knows their values for the
next 4 hours.

**Calendar, 7 columns** [table.py: calendar]: hour, minute of day, day of week, weekend flag, slot of day, holiday
flag, working-day flag. Holidays come from `config/site_calendar.yaml`.

**Day-type profile, one column per target** (`prof.<target>`) [forecast/profile_features.py]. For each slot it is the
average of that same series at the same clock time on other days of the same kind, using the most specific level
that has enough days:

1. the same weekday, once at least 2 other such days exist;
2. otherwise the same day type (working day or not);
3. otherwise any day;
4. otherwise the series' overall average.

Past rows leave their own day out, so the profile is never a copy of the value next to it. It is computed only from
the history being handed over, so it cannot contain the future.

Example from the live cabin frame: on Wed 18 Nov at 10:00 the measured `scene.people_yolo` was 4 and its profile was
3.33, the average at 10:00 on the other working days (Fri, Mon, Tue).

## Step 5: the two frames handed over

[table.py: to_chronos, then profile_features.py: add_profiles]

**Past frame ("context"), admin cabin: 487 rows x 457 columns**

| Column group | Count | Contents |
|---|---|---|
| `id` | 1 | the space name, the same on every row |
| `timestamp` | 1 | slot start in IST, exactly 15 minutes apart |
| targets | 45 | measured values |
| past-only inputs | 358 | measured values (357 numeric, 1 text) |
| calendar | 7 | whole numbers |
| profiles | 45 | numbers |

The last rows of the real frame (a few columns shown):

| timestamp | scene.people_yolo | scene.people_vlm | img.brightness | cal.hour | cal.dow | cal.is_working_day | prof.scene.people_yolo |
|---|---|---|---|---|---|---|---|
| 18 Nov 10:00 | 4 | 4 | 122.2 | 10 | 2 | 1 | 3.33 |
| 18 Nov 10:15 | 2 | 2 | 121.8 | 10 | 2 | 1 | 4.00 |
| 18 Nov 10:30 | 3 | 3 | 121.9 | 10 | 2 | 1 | 2.33 |

**Future frame, admin cabin: 16 rows x 54 columns** (`id`, `timestamp`, 7 calendar, 45 profiles)

| timestamp | cal.hour | cal.dow | cal.is_working_day | prof.scene.people_yolo |
|---|---|---|---|---|
| 18 Nov 10:45 | 10 | 2 | 1 | 2.00 |
| 18 Nov 11:00 | 11 | 2 | 1 | 1.67 |
| 18 Nov 11:15 | 11 | 2 | 1 | 2.67 |

The future frame never contains a target or a past-only input.

Checks made on the live frames: timestamps are perfectly regular; 0.36% of values are missing; only the 2 slots
that had no photo are empty across all measured columns; the newest row is complete.

## Step 6: the call

[forecast/models.py: Chronos2Forecaster.forecast]

```python
pipeline.predict_df(
    context,                       # the past frame
    future_df=future,              # the future frame
    prediction_length=16,          # 4 hours of 15-minute slots  (forecast.horizon)
    quantile_levels=[0.1, 0.5, 0.9],                            # (forecast.quantiles)
    id_column="id", timestamp_column="timestamp",
    target=[...45 names...],       # the columns to forecast
    cross_learning=False,          # (forecast.cross_learning)
)
```

How Chronos-2 reads this:

- Columns named in `target` are forecast.
- Columns that are also in the future frame are inputs known in advance.
- Every other column is an input known only for the past.
- Text columns are treated as categories.
- Empty values are masked out, not read as 0.

About the model: `amazon/chronos-2`, 119.5 million parameters, running on the GPU. It can read up to 8,192 slots
of history (about 85 days at this cadence) and reads them in pieces of 16 slots (4 hours). Inside one call all
series of the space attend to each other, which is how the inputs can influence the forecasts. It has no knowledge
of this site beyond the rows it is given in that call.

## Step 7: what comes back and what is done with it

Chronos returns one row per target per future slot with the three requested levels: 45 x 16 = 720 rows for the
cabin, 31 x 16 = 496 for parking.

| timestamp | target | step | q10 | q50 | q90 |
|---|---|---|---|---|---|
| 18 Nov 10:45 | scene.people_yolo | 1 | 1.26 | 2.39 | 3.66 |
| 18 Nov 11:00 | scene.people_yolo | 2 | 0.65 | 1.85 | 3.26 |
| 18 Nov 11:15 | scene.people_yolo | 3 | 1.62 | 2.97 | 4.31 |

- **q50** is the middle estimate. **q10 to q90** is the range given an 80% chance of containing the true value.
- **One rule is applied afterwards** [inputs.py: predict]: counts, areas and image statistics cannot be negative,
  so their three values are clipped at 0. Description scores and image-vector values can be negative and are left
  alone. Nothing else is changed.
- **Stored** [runner.py, store/forecasts.py]: one row in `forecast_runs` (space, time made, start time, model,
  input set, horizon, number of history rows, target names, input names, levels) and one row per value in
  `forecasts`. So every forecast can be traced to exactly what the model was given.
- **Used by:** the dashboard (charts, tables), `sbi accuracy` (scores each stored forecast once its time has
  passed), and the "Forecast vs actual" tab.

## Rules that keep it honest

| Rule | Where |
|---|---|
| Missing is not zero: no photo, or a failed model, gives an empty value | table.py |
| The future frame carries only calendar and profiles, never measurements | to_chronos |
| The backtest rebuilds the catalog, targets, inputs and profiles from the history before each forecast moment | backtest.py |
| Image-vector directions are learned only from history before the first backtest moment | `embed_fit_end` |
| Live forecasts and the backtest go through the same `prepare` and `predict` functions | inputs.py |
| A forecast made before its history was re-read (detector change) is not scored | accuracy.py: outdated_runs |

## Evidence that the inputs are used

- Scrambling the past-only inputs changes the forecasts (average change 0.14 across all cabin targets; measured on the
  morning of 18 Nov, before the detector change). Scrambling
  the future calendar barely does (0.004). Chronos raises no warnings on the frames.
- Backtest on 5 days, 46 forecast moments, skill against the weekday-by-time average: cabin 0.417, parking 0.350
  (holding the last value: 0.299 and 0.169).

## Known limits

1. **It does not know weekdays yet.** With 5 days of history each weekday has been seen once. After the empty
   weekend it forecast Monday morning at 0.7 people against 2.45 measured. Profiles helped slightly; more weeks
   are what fix this. First real tests: Sat 21 Nov and Mon 23 Nov.
2. **Some inputs repeat each other.** For example `count.yolo.person` is the same series as `scene.people_yolo`,
   and about 30 input columns have a near-identical twin. Harmless, but they add nothing.
3. **Table build time grows with history** (about 1 second per 100 slots for the cabin). At 85 days that would be
   over a minute per forecast. See "Open improvements" in `docs/repo-map.md`.
4. **Past 85 days** the history exceeds what Chronos-2 can read in one call; older rows are then ignored by it.
5. **Each space is forecast on its own.** Parking is not shown the cabin's series, or the other way round.
6. **Horizon is 4 hours.** `forecast.horizon` can be raised; accuracy further out has not been measured.

## Settings that change what is fed

| Setting | File | Effect |
|---|---|---|
| `forecast_targets` | `config/spaces/<space>.yaml` | which series are forecast |
| `probes` | `config/spaces/<space>.yaml` | which plain-English descriptions are scored on each photo |
| `forecast.variant` | `config/pipeline.yaml` | the input set: `all_profile`, `all`, `calendar`, `profile`, `full`, `detector`, `univariate` |
| `forecast.horizon`, `quantiles`, `max_targets` | `config/pipeline.yaml` | how far ahead, which levels, how many targets |
| `forecast.every_minutes` | `config/pipeline.yaml` | how often a forecast is stored (60) |
| `table.min_coverage`, `min_presence`, ... | `config/pipeline.yaml` | what counts as a usable column |
| `config/site_calendar.yaml` | | holidays and closures behind the working-day flag |

To see the frames yourself: `PYTHONPATH=src .venv/bin/python -m sbi table --space admin_cabin` writes the table and
its catalog to `data/exports/`.

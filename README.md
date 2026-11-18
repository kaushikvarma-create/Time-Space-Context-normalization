# Space Behaviour Intelligence (SBI)

SBI turns two security cameras into time series and forecasts them.

Every 15 minutes it takes one still from each camera (an admin cabin and a parking area). Five models read each
photo into numbers: YOLO26x boxes, SAM 2 outlines, a vision-language model, a DINOv2 image vector and SigLIP 2
description scores. Everything is stored in SQLite. Each space gets one table, with a row every 15 minutes and about
600 to 750 columns. Once an hour Chronos-2 reads that table and forecasts the next 4 hours. A read-only dashboard
shows the numbers, the forecasts and how accurate past forecasts were.

| Document | What it covers |
|---|---|
| `docs/technical-deep-dive.md` | every module, setting, table, rule and test, in detail |
| `docs/chronos-data-flow.md` | exactly what is given to Chronos-2 and what comes back |
| `docs/dashboard-guide.md` | what every part of the dashboard means (also the dashboard's Guide tab) |
| `docs/repo-map.md` | mind map of the code, the 18 Nov 2026 sweep, open improvements |
| `docs/research/2026-11-13-photo-to-numbers.md` | research behind the photo-to-numbers design |
| `PLAN.md` | decisions, benchmarks and results in date order |

## What runs on its own

| What | How | When |
|---|---|---|
| Capture and read both cameras, update baselines | cron runs `scripts/tick.sh` | every 15 minutes |
| Chronos-2 forecast per space, stored with its inputs | inside the same run | on the hour (`forecast.every_minutes: 60`) |
| Dashboard at http://localhost:8502 | systemd user service `sbi-dashboard` | always; restarts itself, starts at boot |
| Keep WSL (and so cron) running after a Windows restart | Windows scheduled task `SBI-WSL-Keepalive` | at Windows logon |

Every stage is isolated. If one model fails, that failure is logged and its columns are left empty for that photo.
The rest of the photo is still stored, and the other space is unaffected. All times shown to people are IST.

## Quick start

Python 3.10 or newer, an NVIDIA GPU (the models also run on CPU, slowly), and an OpenAI-compatible vision endpoint
(LM Studio) for the scene reader. Model weights go in `models/`, which is not in git: `yolo26x.pt`, `sam2.1_l.pt`.
The Hugging Face models (DINOv2, SigLIP 2, Chronos-2) download on first use.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[detector,vision,forecast,ui,dev]'
cp .env.example .env            # fill in SBI_NVR_HOST/PORT/USER/PASS and the LM Studio URL; keep it chmod 600
PYTHONPATH=src .venv/bin/python -m sbi migrate
PYTHONPATH=src .venv/bin/python -m sbi tick                # one run by hand
.venv/bin/python -m pytest -q -p no:warnings                # tests: fakes only, no camera/GPU/LM Studio needed
```

To make it run on its own, follow these three steps:

```bash
(crontab -l; echo "*/15 * * * * $PWD/scripts/tick.sh") | crontab -
scripts/install-dashboard-service.sh
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w scripts/windows/install-autostart.ps1)" -Distro "$WSL_DISTRO_NAME"
```

## Commands

All commands run as `PYTHONPATH=src .venv/bin/python -m sbi <command>` from the project root.

| Command | What it does |
|---|---|
| `tick` | capture and read every space, update baselines, forecast when due (what cron runs) |
| `migrate` | apply pending database migrations |
| `reprocess --space ID [--frame-id N] [--detector-only] [--no-vlm]` | read stored photos again after a model or setting change; results are replaced, not duplicated. `--detector-only` keeps each photo's stored VLM answer |
| `embed --space ID` / `score --space ID` | add image vectors / description scores to photos that lack them (history, or a newly added prompt) |
| `table --space ID [--fit-end T]` | write the table and its column catalog to `data/exports/` |
| `forecast --space ID [--variant V] [--model M]` | forecast now and store the run |
| `backtest --space ID [--variants ...] [--models ...]` | replay forecasts from past moments and score them |
| `accuracy --space ID [--since DATE]` | score stored forecasts against what was later measured |
| `eval` | compare model counts with human labels, and suggest per-type thresholds |
| `stability --space ID --start T --end T` | how much each count wobbles in a quiet window (no labels needed) |
| `baseline [--space ID]` | rebuild weekday-by-hour averages |
| `curate`, `links`, `link-set ID approved\|rejected` | link synonyms and object groups, review or undo links |

Exit codes: 0 ok, 1 a space failed, 2 fatal configuration error.

| Script | Use |
|---|---|
| `scripts/label.sh` | labelling tool at http://localhost:8501: count objects in photos to measure accuracy |
| `scripts/dashboard.sh` | run the dashboard in the foreground (the service uses this) |
| `scripts/detector_bench/bench.py`, `score.py` | benchmark any detector setting on the stored photos and score it against labels |

## Layout

```
config/
  pipeline.yaml            global settings: paths, every model, table, forecast, calendar. No secrets.
  spaces/*.yaml            one file per camera: channel, grid, vocabulary, aliases, detector map, probes, targets
  site_calendar.yaml       office holidays and closures (fill this in)
scripts/                   tick.sh, dashboard.sh, label.sh, curate.sh, detector_bench/, systemd/, windows/
src/sbi/
  settings.py spaces.py    typed, validated configuration (unknown keys fail loudly)
  errors.py                SbiError hierarchy; every expected failure is one of these
  ingest/                  camera sources (NVR RTSP) and capture (image file + frame row)
  perception/              model interfaces and implementations: YOLO, SAM 2, DINOv2, SigLIP 2, LM Studio VLM;
                           box and outline de-duplication; mask codec; label normaliser
  pipeline.py              one photo -> every stored result, in one transaction
  store/                   SQLite connection, transactions, numbered migrations, data access
  taxonomy.py curation/    object names: synonyms and is-a groups; raw labels kept forever
  baseline.py              weekday-by-hour averages
  forecast/                table builder and its per-photo cache; target and input choice; day-type profiles;
                           Chronos-2 and baselines; backtest; live runs; live accuracy
  sitecal.py               holidays and working days (site file, optional `holidays` package)
  dashboard/               read-only Streamlit dashboard (data layer + page)
  labeling/ eval/          human labelling tool; accuracy, threshold and stability reports
  tick.py cli.py           the scheduled run and the command line
tests/                     pytest; fakes only
data/                      not in git: sbi.db, images/, logs/, cache/, exports/, backups/
models/                    not in git: detector and outline weights
```

## Rules the code follows

- **Missing is not zero.** A slot with no photo, or a model that failed, gives an empty value, never a count of 0.
- **Raw readings are kept.** Every detector box down to confidence 0.05 is stored with its outline, as are the VLM's
  full answer, every description score and every image vector. Table columns are derived from them, so a rule can
  change and history is re-derived without running a model again.
- **Measure, do not smooth.** If a count is wrong, the fix is a better detector or counting rule, chosen by
  benchmark against labels (`PLAN.md`, 18 Nov 2026). Values are not post-corrected.
- **No future leaks.** The backtest rebuilds every choice from the history before each forecast moment. Only the
  calendar and day-type profiles are given for the future.
- **One transaction per photo.** Slow model work happens first, then every write for a photo commits together.
- **Schema changes only by migration** (`src/sbi/store/migrations/NNN_*.sql`).
- **Secrets only in `.env`.** Credentials are masked in logs and errors. The dashboard and labelling tool listen on
  this machine only.
- **The VLM is never shown a list of object names.** Named lists make it report things that are not there.
  Prompts are open-vocabulary, and names are tidied afterwards by the taxonomy.

## Extending

| To add | Do this |
|---|---|
| A space (camera) | add `config/spaces/<id>.yaml` (copy one). The next run picks it up. |
| A forecast series | add its column name or a wildcard to `forecast_targets` in the space file. |
| A description to score | add a concept with a few paraphrases under `probes`, then run `sbi score --space ID`. |
| A model (detector, outliner, embedder, scorer, scene reader, camera source) | implement the Protocol in `perception/types.py`, register its `kind` in `perception/factory.py` and the settings, select it in `config/pipeline.yaml`. After a detector change, run `sbi reprocess --detector-only`. |
| A forecaster | implement `Forecaster` in `forecast/models.py` and register it in `build_forecaster`. Live runs, backtest and storage need no change. |
| An object type | nothing. Unknown labels become registry rows flagged `auto_new`, and `sbi curate` links them. |

See `docs/technical-deep-dive.md` for the details behind every line above.

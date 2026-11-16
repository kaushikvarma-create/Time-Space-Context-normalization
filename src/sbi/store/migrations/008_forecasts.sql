-- 008 forecasts: what a forecaster predicted, from which table state and inputs.
-- One run = one forecaster call for one space at one origin (the last observed slot). The inputs are recorded
-- with the run (targets, covariates, covariate-set name, model), so a stored forecast can always be traced back and
-- later compared with what was actually measured. Quantiles are rows, not columns, so the quantile set can change.
CREATE TABLE IF NOT EXISTS forecast_runs (
  run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  space_id      TEXT NOT NULL REFERENCES spaces(space_id),
  created_utc   TEXT NOT NULL,
  origin_local  TEXT NOT NULL,        -- last observed slot, naive local time (IST), as in the table
  model         TEXT NOT NULL,        -- forecaster name, e.g. chronos2:amazon/chronos-2
  variant       TEXT NOT NULL,        -- covariate-set name (univariate, calendar, detector, full, ...)
  horizon       INTEGER NOT NULL,     -- slots ahead
  step_minutes  INTEGER NOT NULL,
  context_slots INTEGER NOT NULL,     -- rows of history given to the model
  targets_json  TEXT NOT NULL,
  covariates_json TEXT NOT NULL,      -- past-only covariates; calendar columns are listed here too when used
  quantiles_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fruns_space ON forecast_runs(space_id, origin_local);

CREATE TABLE IF NOT EXISTS forecasts (
  run_id   INTEGER NOT NULL REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
  target   TEXT NOT NULL,             -- table column name
  ts_local TEXT NOT NULL,             -- the slot being forecast, naive local time (IST)
  step     INTEGER NOT NULL,          -- 1 = the slot right after the origin
  quantile REAL NOT NULL,
  value    REAL NOT NULL,
  PRIMARY KEY (run_id, target, ts_local, quantile)
);

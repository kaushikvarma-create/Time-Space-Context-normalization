"""Forecast runs and their predictions (tables forecast_runs, forecasts). Writers assume a transaction."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pandas as pd

from ..errors import StorageError
from ..timeutil import now_utc


def insert_run(conn: sqlite3.Connection, *, space_id: str, origin_local: str, model: str, variant: str, horizon: int,
               step_minutes: int, context_slots: int, targets: list[str], covariates: list[str],
               quantiles: list[float], created: datetime | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO forecast_runs(space_id, created_utc, origin_local, model, variant, horizon, step_minutes, "
        "context_slots, targets_json, covariates_json, quantiles_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (space_id, (created or now_utc()).isoformat(), origin_local, model, variant, horizon, step_minutes,
         context_slots, json.dumps(targets), json.dumps(covariates), json.dumps(quantiles)))
    return int(cur.lastrowid)


def insert_forecasts(conn: sqlite3.Connection, run_id: int, forecast: pd.DataFrame) -> int:
    """forecast: long frame with columns timestamp, target, step and one column per quantile named q<NN>
    (q10, q50, ...). Returns the number of rows stored."""
    qcols = {c: int(c[1:]) / 100 for c in forecast.columns if c.startswith("q") and c[1:].isdigit()}
    if not qcols:
        raise StorageError("forecast frame has no quantile columns (q10, q50, ...)")
    rows = [(run_id, r["target"], pd.Timestamp(r["timestamp"]).isoformat(), int(r["step"]), tau, float(r[c]))
            for _, r in forecast.iterrows() for c, tau in qcols.items() if pd.notna(r[c])]
    conn.executemany("INSERT INTO forecasts(run_id, target, ts_local, step, quantile, value) VALUES(?,?,?,?,?,?)", rows)
    return len(rows)


def load_forecast(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """Wide frame: target, timestamp, step, q10, q50, ... for one run."""
    df = pd.read_sql_query("SELECT target, ts_local, step, quantile, value FROM forecasts WHERE run_id=?", conn,
                           params=(run_id,))
    if df.empty:
        return pd.DataFrame(columns=["target", "timestamp", "step"])
    df["q"] = df["quantile"].map(lambda q: f"q{round(q * 100):02d}")   # as forecast.models.qname: q05, q10, q50, ...
    wide = df.pivot_table(index=["target", "ts_local", "step"], columns="q", values="value").reset_index()
    wide = wide.rename(columns={"ts_local": "timestamp"})
    wide["timestamp"] = pd.to_datetime(wide["timestamp"])
    return wide.sort_values(["target", "step"]).reset_index(drop=True).rename_axis(columns=None)


def latest_run(conn: sqlite3.Connection, space_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM forecast_runs WHERE space_id=? ORDER BY run_id DESC LIMIT 1",
                        (space_id,)).fetchone()

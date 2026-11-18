"""How good were the forecasts actually made? Every stored run is scored against what was later measured.

Unlike the backtest (simulated forecasts from the past), this scores the real, stored forecasts, so it answers
"is the running system any good" and grows on its own as time passes. Each point is also compared with persistence
(the last value measured before the forecast was made), the simplest honest bar: skill = 1 - MAE / MAE_persistence
on the same points (positive = better than holding the last value). Points whose truth is not measured yet, or
missing, are left out.

Summaries across series pool the errors after dividing each series by its typical 15-minute change (mean |change|
between consecutive slots over the table, as in MASE). Averaging per-series ratios instead lets a series that happened
not to move (persistence error ~0) produce a huge negative skill that swamps everything else.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from ..store.forecasts import load_forecast
from .metrics import evaluate
from .models import BASELINES

LEADS = [(60, "0-1 h"), (120, "1-2 h"), (240, "2-4 h"), (10**9, "4 h +")]


def lead_bucket(minutes: int) -> str:
    return next(label for limit, label in LEADS if minutes <= limit)


def outdated_runs(conn: sqlite3.Connection, space_id: str, *, tolerance: float = 0.05) -> set[int]:
    """Forecast runs made from a history that has since been read again (`sbi reprocess`, for example after a
    detector change). Their forecasts predicted the old readings, so scoring them against the new ones would
    measure the change of detector, not the forecast. A run is outdated when more than `tolerance` of the photos
    taken before it was made were read after it was made. (A few late reads, such as a photo retried after a VLM
    outage, do not count.)"""
    frames = pd.read_sql_query("SELECT ts_utc, processed_utc FROM frames WHERE space_id=? AND processed_utc IS NOT NULL "
                               "ORDER BY ts_utc", conn, params=(space_id,))
    runs = conn.execute("SELECT run_id, created_utc FROM forecast_runs WHERE space_id=?", (space_id,)).fetchall()
    if frames.empty or not runs:
        return set()
    def utc(values) -> np.ndarray:                                       # plain UTC datetime64, comparable everywhere
        return pd.to_datetime(values, utc=True).tz_localize(None).to_numpy()
    taken, read = utc(frames["ts_utc"].to_numpy()), utc(frames["processed_utc"].to_numpy())
    out = set()
    for run_id, created in runs:
        made = utc([created])[0]
        before = int(np.searchsorted(taken, made, side="right"))           # photos taken up to the run
        if before and float((read[:before] > made).mean()) > tolerance:
            out.add(int(run_id))
    return out


def live_points(conn: sqlite3.Connection, space_id: str, table: pd.DataFrame, *, since: str | None = None) -> pd.DataFrame:
    """One row per stored forecast point that now has a measured truth. Runs whose history has since been read
    again (see outdated_runs) are left out."""
    since_iso = pd.Timestamp(since).isoformat() if since else ""     # stored origins are ISO text: compare like with like
    runs = conn.execute("SELECT run_id, model, variant, origin_local, step_minutes, quantiles_json FROM forecast_runs "
                        "WHERE space_id=? AND origin_local >= ? ORDER BY run_id", (space_id, since_iso)).fetchall()
    skip = outdated_runs(conn, space_id)
    parts = []
    for run in runs:
        if run["run_id"] in skip:
            continue
        f = load_forecast(conn, run["run_id"])
        if f.empty:
            continue
        quantiles = json.loads(run["quantiles_json"])
        points = evaluate(f, table, quantiles, probabilistic=run["model"] not in BASELINES)
        if points.empty:
            continue
        origin = pd.Timestamp(run["origin_local"])
        history = table.loc[:origin]
        last = {t: (history[t].dropna().iloc[-1] if t in history and history[t].notna().any() else np.nan)
                for t in points["target"].unique()}
        points["persist_err"] = (points["y"] - points["target"].map(last)).abs()
        points["scale"] = points["target"].map({t: float(table[t].diff().abs().mean()) if t in table else np.nan
                                                for t in points["target"].unique()})
        points["lead_min"] = points["step"] * run["step_minutes"]
        points["run_id"], points["model"], points["variant"], points["origin"] = (run["run_id"], run["model"],
                                                                                   run["variant"], origin)
        parts.append(points)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _skill(g: pd.DataFrame) -> float:
    ok = g[g["persist_err"].notna()]
    base = float(ok["persist_err"].mean()) if len(ok) else np.nan
    return 1 - float(ok["abs_err"].mean()) / base if base and not np.isnan(base) and base > 0 else np.nan


def by_target(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=["model", "variant", "target", "n", "mae", "persist_mae", "skill", "coverage"])
    rows = []
    for (m, v, t), g in points.groupby(["model", "variant", "target"], sort=False):
        rows.append({"model": m, "variant": v, "target": t, "n": len(g), "mae": float(g["abs_err"].mean()),
                     "persist_mae": float(g["persist_err"].mean()), "skill": _skill(g),
                     "coverage": float(g["covered"].mean()) if g["covered"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def pooled_skill(g: pd.DataFrame) -> float:
    """1 - sum(error / scale) / sum(persistence error / scale) over series with a usable scale."""
    ok = g[g["persist_err"].notna() & (g["scale"] > 0)]
    base = float((ok["persist_err"] / ok["scale"]).sum())
    return 1 - float((ok["abs_err"] / ok["scale"]).sum()) / base if base > 0 else np.nan


def by_lead(points: pd.DataFrame) -> pd.DataFrame:
    """Per (model, variant, lead time): pooled scaled skill across series (see the module note), and coverage."""
    if points.empty:
        return pd.DataFrame(columns=["model", "variant", "lead", "runs", "targets", "points", "skill", "coverage"])
    p = points.assign(lead=points["lead_min"].map(lead_bucket))
    rows = []
    for (m, v, lead), g in p.groupby(["model", "variant", "lead"], sort=False):
        rows.append({"model": m, "variant": v, "lead": lead, "runs": g["run_id"].nunique(),
                     "targets": g["target"].nunique(), "points": len(g), "skill": pooled_skill(g),
                     "coverage": float(g["covered"].mean()) if g["covered"].notna().any() else np.nan})
    order = {label: i for i, (_, label) in enumerate(LEADS)}
    return pd.DataFrame(rows).sort_values(["model", "variant", "lead"], key=lambda s: s.map(order) if s.name == "lead" else s).reset_index(drop=True)


WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def by_weekday(points: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Scores per weekday of the moment that was predicted, Monday first; days with no scored forecast are left out.
    For one `target`: what was measured and forecast on average, the average miss and the miss of holding the last
    value. For all series together (target=None) those averages would mix units, so only the pooled skill and the
    share inside the band are given. Shows whether weekends and Mondays are forecast as well as midweek."""
    cols = (["day", "forecasts checked", "average measured", "average forecast", "average miss",
             "miss if nothing changed", "inside band"] if target else ["day", "forecasts checked", "skill", "inside band"])
    if points.empty:
        return pd.DataFrame(columns=cols)
    p = points[points["target"] == target] if target else points
    rows = []
    for dow, g in p.groupby(pd.to_datetime(p["timestamp"]).dt.dayofweek):
        inside = float(g["covered"].mean()) if g["covered"].notna().any() else np.nan
        if target:
            rows.append({"day": WEEKDAYS[dow], "forecasts checked": len(g), "average measured": float(g["y"].mean()),
                         "average forecast": float(g["pred"].mean()), "average miss": float(g["abs_err"].mean()),
                         "miss if nothing changed": float(g["persist_err"].mean()), "inside band": inside})
        else:
            rows.append({"day": WEEKDAYS[dow], "forecasts checked": len(g), "skill": pooled_skill(g), "inside band": inside})
    return pd.DataFrame(rows, columns=cols)

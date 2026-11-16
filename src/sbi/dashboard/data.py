"""Everything the dashboard shows, as plain DataFrames (testable without Streamlit). Read-only: nothing here writes.

All times are local (IST): table timestamps already are; UTC columns from the database are converted here.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..spaces import SpaceConfig
from ..store.forecasts import load_forecast
from ..timeutil import get_zone


def connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _local(ts_utc: str | None, tz: str) -> str:
    if not ts_utc:
        return "-"
    return pd.Timestamp(ts_utc).tz_convert(get_zone(tz)).strftime("%Y-%m-%d %H:%M")


def health(conn: sqlite3.Connection, spaces: list[SpaceConfig], now_utc: datetime) -> pd.DataFrame:
    """One row per space: is it capturing, and is every stage keeping up over the last 24 hours."""
    since = (now_utc - timedelta(hours=24)).isoformat()
    rows = []
    for s in spaces:
        r = conn.execute(
            "SELECT COUNT(*) n, SUM(capture_status='ok') ok, SUM(processed_utc IS NOT NULL) processed, "
            "SUM(vlm_raw IS NOT NULL AND vlm_raw NOT LIKE '{\"error\"%') vlm, SUM(segmenter_version IS NOT NULL) outlined, "
            "SUM(EXISTS(SELECT 1 FROM frame_embeddings e WHERE e.frame_id=f.frame_id)) embedded, "
            "SUM(EXISTS(SELECT 1 FROM frame_scores c WHERE c.frame_id=f.frame_id)) scored, MAX(ts_utc) last "
            "FROM frames f WHERE space_id=? AND ts_utc >= ?", (s.space_id, since)).fetchone()
        fc = conn.execute("SELECT origin_local, model FROM forecast_runs WHERE space_id=? ORDER BY run_id DESC LIMIT 1",
                          (s.space_id,)).fetchone()
        ok = r["ok"] or 0
        share = lambda k: f"{(r[k] or 0) / ok:.0%}" if ok else "-"
        rows.append({"space": s.space_id, "last photo (IST)": _local(r["last"], s.tz),
                     "photos 24 h": f"{ok} / {24 * 60 // s.cadence_minutes}", "failed captures": (r["n"] or 0) - ok,
                     "processed": share("processed"), "VLM": share("vlm"), "outlines": share("outlined"),
                     "image vector": share("embedded"), "probes": share("scored"),
                     "latest forecast (IST)": fc["origin_local"].replace("T", " ")[:16] if fc else "-",
                     "forecaster": fc["model"] if fc else "-"})
    return pd.DataFrame(rows)


def log_problems(path: Path, *, limit: int = 15) -> list[str]:
    """The most recent WARNING / ERROR lines of the tick log (newest last)."""
    if not path.exists():
        return []
    with path.open(errors="replace") as fh:
        lines = [ln.rstrip() for ln in fh if " WARNING " in ln or " ERROR " in ln]
    return lines[-limit:]


def latest_forecast(conn: sqlite3.Connection, space_id: str) -> tuple[dict | None, pd.DataFrame]:
    row = conn.execute("SELECT * FROM forecast_runs WHERE space_id=? ORDER BY run_id DESC LIMIT 1", (space_id,)).fetchone()
    if row is None:
        return None, pd.DataFrame()
    run = dict(row) | {"targets": json.loads(row["targets_json"]), "covariates": json.loads(row["covariates_json"])}
    return run, load_forecast(conn, row["run_id"])


def chart_data(table: pd.DataFrame, forecast: pd.DataFrame, target: str, *, days: float) -> pd.DataFrame:
    """Long frame for one chart: measured values over the last `days`, then the forecast median and band.
    Columns: timestamp, kind ('measured' | 'forecast'), value, low, high."""
    out = []
    if target in table.columns:
        recent = table[target].loc[table.index.max() - timedelta(days=days):]
        out.append(pd.DataFrame({"timestamp": recent.index, "kind": "measured", "value": recent.to_numpy(float),
                                 "low": float("nan"), "high": float("nan")}))
    if not forecast.empty:
        f = forecast[forecast["target"] == target]
        qs = sorted(c for c in f.columns if c.startswith("q") and c[1:].isdigit())
        if len(f) and "q50" in qs:
            out.append(pd.DataFrame({"timestamp": pd.to_datetime(f["timestamp"]), "kind": "forecast",
                                     "value": f["q50"].to_numpy(float), "low": f[qs[0]].to_numpy(float),
                                     "high": f[qs[-1]].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["timestamp", "kind", "value", "low", "high"])


def backtest_overall(exports: Path, space_id: str) -> pd.DataFrame:
    """The last `sbi backtest` summary for the space, if one was written."""
    path = exports / f"{space_id}_backtest.csv"
    if not path.exists():
        return pd.DataFrame()
    from ..forecast.metrics import overall
    return overall(pd.read_csv(path))

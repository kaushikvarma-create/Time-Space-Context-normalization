"""Scoring forecasts against what was measured.

Per point: absolute error of the median, pinball loss averaged over the quantiles, and whether the outer quantiles
(e.g. q10..q90, an 80% interval) contained the truth. Summaries add weighted quantile loss (WQL: 2 x total pinball
loss / total |truth|, the usual scale-free form) and a skill score against a reference forecaster on exactly the
same points: 1 - MAE / MAE_reference (positive = better than the reference, 0 = equal).

Slots where the truth is missing (NaN) are skipped: an unmeasured slot has no right answer.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .models import qname


POINT_COLUMNS = ["timestamp", "target", "step", "y", "pred", "abs_err", "pinball", "covered"]


def pinball(y: np.ndarray, pred: np.ndarray, tau: float) -> np.ndarray:
    d = y - pred
    return np.maximum(tau * d, (tau - 1) * d)


def evaluate(forecast: pd.DataFrame, truth: pd.DataFrame, quantiles: Sequence[float], *, probabilistic: bool) -> pd.DataFrame:
    """One row per forecast point with a measured truth. `truth`: index = timestamp, one column per target."""
    f = forecast.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    f["y"] = [truth.at[t, g] if (t in truth.index and g in truth.columns) else np.nan
              for t, g in zip(f["timestamp"], f["target"])]
    f = f[f["y"].notna()].copy()
    if f.empty:
        return pd.DataFrame(columns=POINT_COLUMNS)
    y = f["y"].to_numpy(float)
    f["pred"] = f[qname(0.5)].to_numpy(float)
    f["abs_err"] = np.abs(y - f["pred"].to_numpy())
    f["pinball"] = np.mean([pinball(y, f[qname(q)].to_numpy(float), q) for q in quantiles], axis=0)
    lo, hi = f[qname(min(quantiles))].to_numpy(float), f[qname(max(quantiles))].to_numpy(float)
    f["covered"] = ((y >= lo) & (y <= hi)).astype(float) if probabilistic else np.nan
    return f[POINT_COLUMNS]


def summarise(raw: pd.DataFrame, *, reference: str = "seasonal_mean") -> pd.DataFrame:
    """raw: rows from evaluate() plus columns origin, forecaster, variant. One row per (forecaster, variant, target)
    with n, mae, wql, coverage and skill against `reference` on the same (origin, target, timestamp) points."""
    if raw.empty:
        return pd.DataFrame(columns=["forecaster", "variant", "target", "n", "mae", "wql", "coverage", "skill"])
    key = ["origin", "target", "timestamp"]
    ref = raw[raw["forecaster"] == reference][key + ["abs_err"]].rename(columns={"abs_err": "ref_err"})
    ref = ref.drop_duplicates(key)
    rows = []
    for (fc, var, tgt), g in raw.groupby(["forecaster", "variant", "target"], sort=False):
        denom = float(np.abs(g["y"]).sum())
        both = g.merge(ref, on=key, how="inner")
        ref_mae = float(both["ref_err"].mean()) if len(both) else np.nan
        mae_same = float(both["abs_err"].mean()) if len(both) else np.nan
        rows.append({"forecaster": fc, "variant": var, "target": tgt, "n": len(g), "mae": float(g["abs_err"].mean()),
                     "wql": 2 * float(g["pinball"].sum()) / denom if denom > 0 else np.nan,
                     "coverage": float(g["covered"].mean()) if g["covered"].notna().any() else np.nan,
                     "skill": 1 - mae_same / ref_mae if ref_mae and not np.isnan(ref_mae) and ref_mae > 0 else np.nan})
    return pd.DataFrame(rows)


def overall(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (forecaster, variant): the mean over targets of skill and wql (each target counts once, so busy
    series do not drown quiet ones), the mean coverage, and how many targets and points went into it."""
    if summary.empty:
        return summary
    g = summary.groupby(["forecaster", "variant"], sort=False)
    out = g.agg(targets=("target", "nunique"), points=("n", "sum"), skill=("skill", "mean"), wql=("wql", "mean"),
                coverage=("coverage", "mean")).reset_index()
    return out.sort_values("skill", ascending=False, na_position="last").reset_index(drop=True)

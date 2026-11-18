"""What this kind of day has looked like at each clock time so far: a known-in-advance input for the forecaster.

A zero-shot model is told the weekday and whether it is a working day, but with a few days of history it cannot yet
connect those flags to behaviour: after an empty weekend it expects an empty Monday. The profile spells the
connection out. For every slot it gives the average of the SAME series at the same clock time on other days of the
same kind, where "same kind" is the most specific level that has enough days:

    1. the same weekday            (needs at least `min_same_weekday` other such days)
    2. the same day type           (working day or not; needs one other such day)
    3. any day                     (needs one other day)
    4. the series' overall average

Past rows leave their own day out, so the profile is never a copy of the value it sits beside; the model sees the
same kind of number in the past as it is given for the future. Everything is computed from the history handed in,
so a backtest that passes the history before each forecast keeps the future out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def day_profile(history: pd.Series, working: np.ndarray, future_index: pd.DatetimeIndex, future_working: np.ndarray,
                *, min_same_weekday: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(profile for each history row, profile for each future row) of one series. `working` / `future_working` are
    1 for a working day, 0 otherwise, aligned with `history.index` / `future_index`. NaN only when the history has
    no value at all."""
    v = pd.to_numeric(history, errors="coerce").to_numpy(float)
    idx = history.index
    past = pd.DataFrame({"v": v, "tod": idx.hour * 60 + idx.minute, "dow": idx.dayofweek,
                         "work": np.asarray(working, int), "has": (~np.isnan(v)).astype(int)})
    fut = pd.DataFrame({"tod": future_index.hour * 60 + future_index.minute, "dow": future_index.dayofweek,
                        "work": np.asarray(future_working, int)})
    overall = float(np.nanmean(v)) if past["has"].any() else float("nan")
    out_past = np.full(len(past), np.nan)
    out_fut = np.full(len(fut), np.nan)
    for keys, need in ((["dow", "tod"], min_same_weekday), (["work", "tod"], 1), (["tod"], 1)):
        g = past.groupby(keys)
        total, count = g["v"].transform("sum").to_numpy(), g["has"].transform("sum").to_numpy()
        others = count - past["has"].to_numpy()                       # days of this kind other than the row's own
        with np.errstate(invalid="ignore", divide="ignore"):
            loo = (total - np.nan_to_num(v)) / others
        fill = np.isnan(out_past) & (others >= need)
        out_past[fill] = loo[fill]
        table = g.agg(total=("v", "sum"), count=("has", "sum")).reset_index()
        m = fut.merge(table, on=keys, how="left")
        ok = np.isnan(out_fut) & (m["count"].fillna(0).to_numpy() >= need)
        out_fut[ok] = (m["total"].to_numpy() / m["count"].to_numpy())[ok]
    out_past[np.isnan(out_past)] = overall
    out_fut[np.isnan(out_fut)] = overall
    return out_past, out_fut


def add_profiles(context: pd.DataFrame, future: pd.DataFrame, targets: list[str], working: np.ndarray,
                 future_working: np.ndarray, *, min_same_weekday: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add one `prof.<target>` column per target to both frames (so the model treats it as known in advance).
    Returns (context, future, names of the added columns)."""
    past_cols, fut_cols = {}, {}
    past_index = pd.DatetimeIndex(pd.to_datetime(context["timestamp"]))
    future_index = pd.DatetimeIndex(pd.to_datetime(future["timestamp"]))
    for t in targets:
        p, f = day_profile(pd.Series(context[t].to_numpy(), index=past_index), working, future_index, future_working,
                           min_same_weekday=min_same_weekday)
        past_cols[f"prof.{t}"], fut_cols[f"prof.{t}"] = p, f
    return (pd.concat([context, pd.DataFrame(past_cols, index=context.index)], axis=1),
            pd.concat([future, pd.DataFrame(fut_cols, index=future.index)], axis=1), list(past_cols))

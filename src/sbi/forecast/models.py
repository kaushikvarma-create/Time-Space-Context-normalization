"""Forecasters: anything that turns a table's history into quantile forecasts, behind one interface.

To add a model (Timestral, TimesFM, a LightGBM stage, ...): implement `Forecaster`, register it in `build_forecaster`.
Nothing else changes: live forecasts, the backtest and storage all talk to the interface.

Input: `context` (long frame: id, timestamp, targets, past-only covariates, calendar columns) and `future` (id,
timestamp and the calendar columns, i.e. what is known in advance), in the layout Chronos-2's predict_df uses.
Output: a long frame with columns timestamp, target, step (1 = first slot after the history) and one column per
quantile named q<NN> (q10, q50, q90).

The baselines are deliberately simple and use no covariates: they are the bar a model must clear.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from ..errors import ForecastError

log = logging.getLogger(__name__)


def qname(q: float) -> str:
    return f"q{round(q * 100):02d}"


class Forecaster(Protocol):
    name: str                  # stored with every run, e.g. chronos2:amazon/chronos-2
    probabilistic: bool        # False: every quantile equals the point forecast, so intervals mean nothing
    uses_covariates: bool      # False: results do not depend on the covariate set

    def forecast(self, context: pd.DataFrame, future: pd.DataFrame, *, targets: list[str],
                 quantiles: Sequence[float]) -> pd.DataFrame:
        """Raise ForecastError if the model fails or a target has no usable history."""


def _long(future: pd.DataFrame, target: str, values: np.ndarray, quantiles: Sequence[float]) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": future["timestamp"].to_numpy(), "target": target,
                        "step": np.arange(1, len(future) + 1)})
    for q in quantiles:
        out[qname(q)] = values.astype(float)
    return out


def _series(context: pd.DataFrame, target: str) -> pd.Series:
    s = pd.Series(context[target].to_numpy(dtype=float), index=pd.to_datetime(context["timestamp"]))
    if s.notna().sum() == 0:
        raise ForecastError(f"target {target!r} has no observed values in the history")
    return s


class _Baseline:
    probabilistic = False
    uses_covariates = False

    def _predict(self, history: pd.Series, when: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, context, future, *, targets, quantiles):
        parts = []
        when = pd.DatetimeIndex(pd.to_datetime(future["timestamp"]))
        for t in targets:
            parts.append(_long(future, t, self._predict(_series(context, t), when), quantiles))
        return pd.concat(parts, ignore_index=True)


class Persistence(_Baseline):
    """The last observed value, held flat."""
    name = "persistence"

    def _predict(self, history, when):
        return np.full(len(when), float(history.dropna().iloc[-1]))


class SeasonalNaive(_Baseline):
    """The value at the same clock time on the most recent earlier day that has one (looking back up to 7 days);
    persistence where none exists."""
    name = "seasonal_naive"

    def _predict(self, history, when):
        last = float(history.dropna().iloc[-1])
        out = []
        for t in when:
            value = np.nan
            for days in range(1, 8):
                probe = t - timedelta(days=days)
                if probe in history.index and not np.isnan(history[probe]):     # the history ends at the origin
                    value = float(history[probe])
                    break
            out.append(last if np.isnan(value) else value)
        return np.array(out)


class SeasonalMean(_Baseline):
    """Mean of the same weekday and clock time in the history when it has at least two such points, else the mean
    of the same clock time on any day, else persistence. This is the weekday-by-hour average, at slot resolution."""
    name = "seasonal_mean"

    def _predict(self, history, when):
        h = history.dropna()
        frame = pd.DataFrame({"v": h.to_numpy(), "dow": h.index.dayofweek, "tod": h.index.hour * 60 + h.index.minute})
        by_dow = frame.groupby(["dow", "tod"])["v"].agg(["mean", "count"])
        by_tod = frame.groupby("tod")["v"].mean()
        last = float(h.iloc[-1])
        out = []
        for t in when:
            tod = t.hour * 60 + t.minute
            key = (t.dayofweek, tod)
            if key in by_dow.index and by_dow.loc[key, "count"] >= 2:
                out.append(float(by_dow.loc[key, "mean"]))
            elif tod in by_tod.index:
                out.append(float(by_tod.loc[tod]))
            else:
                out.append(last)
        return np.array(out)


class Chronos2Forecaster:
    probabilistic = True
    uses_covariates = True

    def __init__(self, model: str = "amazon/chronos-2", *, cross_learning: bool = False):
        self.cross_learning = cross_learning
        self.name = f"chronos2:{model}"
        try:
            import torch
            from chronos import Chronos2Pipeline
            self._pipe = Chronos2Pipeline.from_pretrained(model, device_map="cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:  # third-party loaders raise many types; wrap into our boundary error
            raise ForecastError(f"failed to load forecaster {model}: {e}") from e
        log.info("forecaster %s loaded", self.name)

    def forecast(self, context, future, *, targets, quantiles):
        try:
            raw = self._pipe.predict_df(context, future_df=future, prediction_length=len(future),
                                        quantile_levels=list(quantiles), id_column="id", timestamp_column="timestamp",
                                        target=list(targets), cross_learning=self.cross_learning)
        except Exception as e:
            raise ForecastError(f"forecasting failed: {e}") from e
        out = raw.rename(columns={"target_name": "target"})
        for q in quantiles:
            if str(q) not in out.columns:
                raise ForecastError(f"model output has no column for quantile {q}")
            out[qname(q)] = out[str(q)].astype(float)
        out = out.sort_values(["target", "timestamp"])
        out["step"] = out.groupby("target").cumcount() + 1
        return out[["timestamp", "target", "step"] + [qname(q) for q in quantiles]].reset_index(drop=True)


BASELINES = {"persistence": Persistence, "seasonal_naive": SeasonalNaive, "seasonal_mean": SeasonalMean}


def build_forecaster(kind: str, *, model: str = "amazon/chronos-2", cross_learning: bool = False) -> Forecaster:
    if kind == "chronos2":
        return Chronos2Forecaster(model, cross_learning=cross_learning)
    if kind in BASELINES:
        return BASELINES[kind]()
    raise ForecastError(f"unknown forecaster {kind!r}; choose from chronos2, {', '.join(sorted(BASELINES))}")

"""Choosing what to forecast and what to give the model: targets from patterns, a covariate set by variant.

Shared by live forecasts and the backtest so both make exactly the same choices.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from fnmatch import fnmatchcase

import pandas as pd

from ..errors import ForecastError
from ..settings import ALL_KINDS, ForecastSettings, TableSettings
from .models import Forecaster, qname
from .table import TableResult, all_covariates, select_covariates, to_chronos

log = logging.getLogger(__name__)
NONNEGATIVE_KINDS = {"scene", "steady", "count", "rollup", "area", "cell", "cellarea", "rel", "img"}   # counts, shares, image stats
NOT_TARGETS = {"cal", "cat"}      # calendar is known in advance; categorical fields are strings, not series


def resolve_targets(patterns: list[str], catalog: pd.DataFrame, max_targets: int) -> list[str]:
    """Model-ready numeric columns named by the patterns (exact names or shell-style wildcards), in the order the
    patterns are written. Threshold variants (count.yolo_c05.*) match only when named exactly, so a broad pattern
    does not fill the cap with near-copies. Raises ForecastError if nothing matches."""
    ready = catalog[catalog["model_ready"] & ~catalog["kind"].isin(NOT_TARGETS)]
    columns = list(ready["column"])
    primary = set(ready.loc[~ready["variant"], "column"])
    chosen: list[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            hits = [c for c in columns if c in primary and fnmatchcase(c, pattern)]
        else:
            hits = [c for c in columns if c == pattern]
        if not hits:
            log.warning("forecast target %r matches no model-ready column", pattern)
        chosen += [c for c in hits if c not in chosen]
    if not chosen:
        raise ForecastError("none of the forecast targets matches a model-ready column")
    if len(chosen) > max_targets:
        log.info("forecasting the first %d of %d matching targets (forecast.max_targets)", max_targets, len(chosen))
    return chosen[:max_targets]


@dataclass(frozen=True)
class Prepared:
    targets: list[str]
    covariates: list[str]          # past-only, chosen by the catalog's selection rule
    calendar: bool                 # calendar columns are given as known-future inputs
    context: pd.DataFrame
    future: pd.DataFrame


def prepare(result: TableResult, space_id: str, patterns: list[str], kinds: list[str], *, horizon: int,
            settings: ForecastSettings, table_settings: TableSettings) -> Prepared:
    """Inputs for one forecast from `result` (whose table ends at the forecast origin). `kinds` is the variant:
    column kinds the model may see, with "cal" meaning the calendar."""
    targets = resolve_targets(patterns, result.catalog, settings.max_targets)
    past_kinds = [k for k in kinds if k != "cal"]
    covariates: list[str] = []
    if ALL_KINDS in past_kinds:                       # everything the pipeline extracted, uncapped
        covariates = all_covariates(result.catalog, targets=targets)
    elif past_kinds:
        covariates = select_covariates(result.catalog, targets=targets, max_covariates=table_settings.max_covariates,
                                       always=table_settings.always_include, min_autocorr=table_settings.min_autocorr,
                                       autocorr_min_slots=table_settings.autocorr_min_slots, kinds=past_kinds)
    calendar = "cal" in kinds
    context, future = to_chronos(result, space_id, targets, horizon, covariates=covariates, calendar_inputs=calendar)
    return Prepared(targets, covariates, calendar, context, future)


def predict(forecaster: Forecaster, prepared: Prepared, quantiles: list[float], catalog: pd.DataFrame) -> pd.DataFrame:
    """Run the forecaster on prepared inputs and apply the one domain rule models cannot know: a count, an area
    share or an image statistic is never negative, so those targets' quantiles are clipped at 0. (Probe scores and
    image-vector components can be negative and are left alone.) Live forecasts and the backtest both go through
    here, so they always behave alike."""
    out = forecaster.forecast(prepared.context, prepared.future, targets=prepared.targets, quantiles=quantiles)
    nonnegative = set(catalog.loc[catalog["kind"].isin(NONNEGATIVE_KINDS), "column"])
    cols = [qname(q) for q in quantiles]
    mask = out["target"].isin(nonnegative)
    out.loc[mask, cols] = out.loc[mask, cols].clip(lower=0.0)
    return out

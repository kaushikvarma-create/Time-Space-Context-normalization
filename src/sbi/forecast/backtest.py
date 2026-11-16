"""Rolling-origin backtest: forecast from many past moments and score against what then happened.

Protocol (each rule exists to keep the future out of the forecast):
  * An origin is a slot with enough history behind it. The model sees only rows up to the origin; the truth is the
    next `horizon` rows.
  * The catalog (which columns are usable, how smooth they are) and therefore targets and covariates are rebuilt from
    the history before every origin, never from the whole table.
  * Calendar values for the horizon are known in advance, so they are legitimately given; nothing else is.
  * Image-vector components (emb.*) are the one thing fit on the whole table by the table builder. The caller builds
    the table with embed_fit_end = the first origin, so they are learned from the first history only.

A forecaster that fails at one origin is skipped there (counted) and the run continues. Baselines do not use
covariates, so they run once per origin under the variant name "-".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..errors import ForecastError
from ..settings import ForecastSettings, TableSettings
from .inputs import predict, prepare
from .metrics import evaluate
from .models import Forecaster
from .table import TableResult, build_catalog

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestPlan:
    horizon: int
    step: int                                   # slots between origins
    min_context: int
    quantiles: list[float]
    variants: dict[str, list[str]]
    patterns: list[str]
    catalog_args: dict                          # thresholds for build_catalog (same as the table's)


@dataclass
class BacktestRun:
    raw: pd.DataFrame
    origins: int
    failures: dict[str, int] = field(default_factory=dict)      # "forecaster/variant" -> origins skipped


def origins_for(n_rows: int, plan: BacktestPlan) -> list[int]:
    """Row positions usable as origins: min_context rows behind, a full horizon of truth ahead."""
    return list(range(plan.min_context - 1, n_rows - plan.horizon, plan.step))


def run_backtest(table: pd.DataFrame, space_id: str, forecasters: list[Forecaster], plan: BacktestPlan, *,
                 settings: ForecastSettings, table_settings: TableSettings) -> BacktestRun:
    origins = origins_for(len(table), plan)
    if not origins:
        raise ForecastError(f"not enough history to backtest: {len(table)} slots, need at least "
                            f"{plan.min_context + plan.horizon}")
    parts: list[pd.DataFrame] = []
    failures: dict[str, int] = {}
    for pos in origins:
        history = table.iloc[:pos + 1]
        truth = table.iloc[pos + 1:pos + 1 + plan.horizon]
        result = TableResult(history, build_catalog(history, **plan.catalog_args))
        origin = history.index[-1]
        for fc in forecasters:
            for variant, kinds in ([("-", [])] if not fc.uses_covariates else list(plan.variants.items())):
                label = f"{fc.name}/{variant}"
                try:
                    p = prepare(result, space_id, plan.patterns, kinds, horizon=plan.horizon, settings=settings,
                                table_settings=table_settings)
                    out = predict(fc, p, plan.quantiles, result.catalog)
                except ForecastError as e:
                    failures[label] = failures.get(label, 0) + 1
                    log.warning("backtest origin %s, %s skipped: %s", origin, label, e)
                    continue
                scored = evaluate(out, truth, plan.quantiles, probabilistic=fc.probabilistic)
                scored["origin"], scored["forecaster"], scored["variant"] = origin, fc.name, variant
                parts.append(scored)
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return BacktestRun(raw, len(origins), failures)

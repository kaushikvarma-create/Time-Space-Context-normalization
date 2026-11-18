"""A live forecast: build inputs from the whole table, run a forecaster, store the run and its predictions."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import pandas as pd

from ..errors import ForecastError
from ..settings import ForecastSettings, TableSettings
from ..spaces import SpaceConfig
from ..store.db import transaction
from ..store.forecasts import insert_forecasts, insert_run
from .inputs import predict, prepare
from .models import Forecaster
from .table import TableResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastRun:
    run_id: int
    origin: pd.Timestamp
    targets: list[str]
    covariates: list[str]
    forecast: pd.DataFrame


def forecast_space(conn: sqlite3.Connection, space: SpaceConfig, result: TableResult, forecaster: Forecaster, *,
                   settings: ForecastSettings, table_settings: TableSettings, variant: str | None = None,
                   horizon: int | None = None, step_minutes: int = 15) -> ForecastRun:
    variant = variant or settings.variant
    if variant not in settings.variants:
        raise ForecastError(f"unknown variant {variant!r}; choose from {sorted(settings.variants)}")
    horizon = horizon or settings.horizon
    table = result.table
    if len(table) < settings.min_context:
        raise ForecastError(f"{space.space_id}: {len(table)} slots of history, need at least {settings.min_context} "
                            f"(forecast.min_context)")
    if not space.forecast_targets:
        raise ForecastError(f"{space.space_id} has no forecast_targets in its space file")
    prepared = prepare(result, space.space_id, space.forecast_targets, settings.variants[variant], horizon=horizon,
                       settings=settings, table_settings=table_settings)
    forecast = predict(forecaster, prepared, settings.quantiles, result.catalog)
    origin = table.index[-1]
    inputs = prepared.covariates + (["cal.*"] if prepared.calendar else []) + (["prof.*"] if prepared.profiles else [])
    with transaction(conn):
        run_id = insert_run(conn, space_id=space.space_id, origin_local=origin.isoformat(), model=forecaster.name,
                            variant=variant, horizon=horizon, step_minutes=step_minutes, context_slots=len(table),
                            targets=prepared.targets, covariates=inputs, quantiles=settings.quantiles)
        insert_forecasts(conn, run_id, forecast)
    log.info("%s: forecast run %d (%s, %s): %d targets, %d covariates, %d slots ahead", space.space_id, run_id,
             forecaster.name, variant, len(prepared.targets), len(prepared.covariates), horizon)
    return ForecastRun(run_id, origin, prepared.targets, prepared.covariates, forecast)

"""Forecasts made by the scheduled tick: every `forecast.every_minutes`, for every space, stored like any other run.

Isolation is the rule, as for the other optional stages: the forecaster loads only when a forecast is due, a model
that cannot load or a space that cannot be forecast is logged and skipped, and nothing here can stop or fail a
capture. Unexpected exceptions are logged with a traceback (they are bugs) and also contained.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ..errors import SbiError
from ..settings import Settings
from ..spaces import SpaceConfig
from ..timeutil import get_zone
from .build import table_for
from .models import Forecaster, build_forecaster
from .runner import forecast_space

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledForecast:
    space_id: str
    run_id: int | None
    error: str | None = None


def due(now_utc: datetime, tz: str, every_minutes: int, step_minutes: int = 15) -> bool:
    """True when the tick's slot (local time, floored to the step) is a multiple of every_minutes."""
    if every_minutes <= 0:
        return False
    local = now_utc.astimezone(get_zone(tz))
    minute = local.hour * 60 + local.minute
    return (minute - minute % step_minutes) % every_minutes == 0


def run_scheduled(conn: sqlite3.Connection, settings: Settings, spaces: list[SpaceConfig], now_utc: datetime, *,
                  forecaster: Forecaster | None = None) -> list[ScheduledForecast]:
    fs = settings.forecast
    todo = [s for s in spaces if s.forecast_targets and due(now_utc, s.tz, fs.every_minutes, settings.table.step_minutes)]
    if not todo:
        return []
    if forecaster is None:
        try:
            forecaster = build_forecaster(fs.scheduled_model, model=fs.model, cross_learning=fs.cross_learning)
        except SbiError as e:
            log.error("forecaster unavailable, no forecasts this time: %s", e)
            return [ScheduledForecast(s.space_id, None, str(e)) for s in todo]
    out: list[ScheduledForecast] = []
    for space in todo:
        try:
            run = forecast_space(conn, space, table_for(conn, settings, space), forecaster, settings=fs,
                                 table_settings=settings.table, step_minutes=settings.table.step_minutes)
            out.append(ScheduledForecast(space.space_id, run.run_id))
        except SbiError as e:
            log.error("%s: scheduled forecast failed: %s", space.space_id, e)
            out.append(ScheduledForecast(space.space_id, None, str(e)))
        except Exception as e:  # noqa: BLE001  isolation boundary: a forecasting bug must not touch capture
            log.exception("%s: unexpected error in scheduled forecast (this is a bug)", space.space_id)
            out.append(ScheduledForecast(space.space_id, None, f"unexpected: {e!r}"))
    return out

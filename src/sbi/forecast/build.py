"""Build a space's table with every setting applied. The one place the CLI, the scheduled tick and the dashboard
get a table from, so they always agree."""
from __future__ import annotations

import sqlite3
from datetime import date

from ..settings import Settings
from ..sitecal import load_calendar
from ..spaces import SpaceConfig
from .table import TableResult, build_table


def table_for(conn: sqlite3.Connection, settings: Settings, space: SpaceConfig, *, start: str | None = None,
              end: str | None = None, fit_end: str | None = None) -> TableResult:
    tc = settings.table
    this_year = date.today().year
    cal = load_calendar(settings.calendar, settings.root, range(this_year - 3, this_year + 2))
    return build_table(conn, space.space_id, tz=space.tz, step_minutes=tc.step_minutes, start=start, end=end,
                       min_coverage=tc.min_coverage, min_presence=tc.min_presence,
                       min_valid_slots=tc.min_valid_slots, min_present_slots=tc.min_present_slots,
                       exclude_labels=tuple(tc.exclude_labels), conf_levels=tuple(tc.conf_levels),
                       mask_dup_iou=tc.mask_dup_iou, grid=(space.grid.cols, space.grid.rows),
                       cell_area_min=tc.cell_area_min, embed_model=settings.embedder.version,
                       embed_components=tc.embed_components, embed_fit_end=fit_end,
                       probe_scorer=settings.scorer.version, probe_groups=space.probes, calendar_spec=cal,
                       steady_labels=tuple(space.steady_labels), steady_window=tc.steady_window,
                       steady_min_values=tc.steady_min_values)

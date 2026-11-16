"""Days known in advance to be different: public holidays and site closures. They become known-future model
inputs (cal.is_holiday, cal.is_working_day), the kind of covariate published Chronos-2 uses lean on most.

Sources, merged:
  * the site calendar file (config/site_calendar.yaml): the office's own holiday list and closures. This is the
    authoritative source, because offices observe their own list, and it needs no network or extra package;
  * optionally the `holidays` package for the configured country (and state), if it is installed.
Working days (default Monday-Friday) are configurable, because a site may work Saturdays.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from .errors import ConfigError
from .settings import CalendarSettings

log = logging.getLogger(__name__)
_noted_missing_package = False


@dataclass(frozen=True)
class CalendarSpec:
    holidays: frozenset[date] = frozenset()
    working_days: frozenset[int] = frozenset({0, 1, 2, 3, 4})      # Monday = 0
    names: dict = field(default_factory=dict, compare=False, hash=False)  # date -> name, for display only


def read_site_file(path: Path) -> dict[date, str]:
    """{date: name} from the site calendar file; a missing file is an empty calendar."""
    if not path.exists():
        return {}
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e
    out: dict[date, str] = {}
    for i, entry in enumerate(doc.get("days") or []):
        if not isinstance(entry, dict) or not isinstance(entry.get("date"), date):
            raise ConfigError(f"{path}: days[{i}] needs a date written as YYYY-MM-DD (got {entry!r})")
        out[entry["date"]] = str(entry.get("name") or "closed")
    return out


def country_days(country: str | None, subdivision: str | None, years: range) -> dict[date, str]:
    """Public holidays from the optional `holidays` package; empty (with a note in the log) if it is absent."""
    if not country:
        return {}
    try:
        import holidays
    except ImportError:
        global _noted_missing_package
        if not _noted_missing_package:                 # once per process, not on every table build
            log.info("`holidays` package not installed: using the site calendar file only")
            _noted_missing_package = True
        return {}
    try:
        return dict(holidays.country_holidays(country, subdiv=subdivision, years=list(years)).items())
    except (KeyError, NotImplementedError) as e:
        raise ConfigError(f"calendar.country {country!r} / subdivision {subdivision!r} not supported: {e}") from e


def load_calendar(settings: CalendarSettings, root: Path, years: range) -> CalendarSpec:
    site = read_site_file(root / settings.site_file)
    names = {**country_days(settings.country, settings.subdivision, years), **site}      # the site file wins
    return CalendarSpec(frozenset(names), frozenset(settings.working_days), names)

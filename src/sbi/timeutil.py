"""Time handling. ts_utc is the canonical stored instant; every derived field (date, weekday, hour, slot)
is computed in the space's local timezone (IST by default) so averages group by wall-clock time."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import ConfigError


def get_zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as e:
        raise ConfigError(f"unknown timezone {tz_name!r} (use an IANA name such as 'Asia/Kolkata')") from e


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TimeDims:
    ts_utc: str
    ts_local: str
    tz: str
    date: str
    dow: int            # 0 = Monday
    dow_name: str
    hour: int
    minute: int
    slot_15: int        # 0..95
    week_of_year: int
    is_weekend: int
    part_of_day: str    # night | morning | afternoon | evening

    def as_dict(self) -> dict:
        return asdict(self)


def part_of_day(hour: int) -> str:
    return "night" if hour < 6 else "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"


def time_dims(ts: datetime, tz_name: str) -> TimeDims:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    utc = ts.astimezone(timezone.utc)
    local = utc.astimezone(get_zone(tz_name))
    return TimeDims(
        ts_utc=utc.isoformat(timespec="seconds"),
        ts_local=local.isoformat(timespec="seconds"),
        tz=local.tzname() or tz_name,
        date=local.date().isoformat(),
        dow=local.weekday(),
        dow_name=local.strftime("%a"),
        hour=local.hour,
        minute=local.minute,
        slot_15=local.hour * 4 + local.minute // 15,
        week_of_year=local.isocalendar().week,
        is_weekend=int(local.weekday() >= 5),
        part_of_day=part_of_day(local.hour),
    )

"""Frame rows: insert (idempotent), fetch, backlog, and processing bookkeeping."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..timeutil import TimeDims


@dataclass(frozen=True)
class ImageMetrics:
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float


def insert_frame(conn: sqlite3.Connection, space_id: str, dims: TimeDims, *, status: str,
                 image_path: str | None, metrics: ImageMetrics | None) -> tuple[int, bool]:
    """Insert a frame; returns (frame_id, created). A repeat for the same (space, ts) returns the existing id.
    Caller owns the transaction."""
    m = metrics
    cur = conn.execute(
        """INSERT OR IGNORE INTO frames(space_id,ts_utc,ts_local,tz,date,dow,dow_name,hour,minute,slot_15,
             week_of_year,is_weekend,part_of_day,image_path,capture_status,width,height,brightness,contrast,sharpness)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (space_id, dims.ts_utc, dims.ts_local, dims.tz, dims.date, dims.dow, dims.dow_name, dims.hour, dims.minute,
         dims.slot_15, dims.week_of_year, dims.is_weekend, dims.part_of_day, image_path, status,
         m.width if m else None, m.height if m else None, m.brightness if m else None,
         m.contrast if m else None, m.sharpness if m else None))
    if cur.rowcount == 1:
        return int(cur.lastrowid), True
    row = conn.execute("SELECT frame_id FROM frames WHERE space_id=? AND ts_utc=?", (space_id, dims.ts_utc)).fetchone()
    return int(row[0]), False


def get_frame(conn: sqlite3.Connection, frame_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM frames WHERE frame_id=?", (frame_id,)).fetchone()


def pending_frame_ids(conn: sqlite3.Connection, space_id: str, limit: int) -> list[int]:
    """Usable frames that Engine 1 has not processed yet (oldest first), e.g. after a detector outage."""
    rows = conn.execute(
        """SELECT frame_id FROM frames WHERE space_id=? AND capture_status='ok' AND processed_utc IS NULL
           ORDER BY ts_utc LIMIT ?""", (space_id, limit)).fetchall()
    return [r[0] for r in rows]

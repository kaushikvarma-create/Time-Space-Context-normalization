"""Spaces and the object registry (canonical object types + aliases)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..spaces import SpaceConfig
from ..timeutil import now_utc
from .db import transaction


def register_space(conn: sqlite3.Connection, space: SpaceConfig, config_path: Path | str) -> None:
    """Upsert the space, seed its vocabulary (never overwriting reviewed rows), and load aliases."""
    with transaction(conn):
        conn.execute(
            """INSERT INTO spaces(space_id,name,space_type,grid_cols,grid_rows,tz,config_path,created_utc)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(space_id) DO UPDATE SET name=excluded.name, space_type=excluded.space_type,
                 grid_cols=excluded.grid_cols, grid_rows=excluded.grid_rows, tz=excluded.tz,
                 config_path=excluded.config_path""",
            (space.space_id, space.name, space.space_type, space.grid.cols, space.grid.rows, space.tz,
             str(config_path), now_utc().isoformat(timespec="seconds")))
        for label, v in space.vocabulary.items():
            conn.execute("""INSERT OR IGNORE INTO object_registry(object_type,category,mobility_prior,review_status)
                            VALUES(?,?,?,'seed')""", (label, v.category, v.mobility))
        for alias, target in space.aliases.items():
            conn.execute("INSERT OR REPLACE INTO object_alias(alias,object_type) VALUES(?,?)", (alias.lower(), target))


def _follow_merges(conn: sqlite3.Connection, name: str) -> str:
    seen: set[str] = set()
    while name not in seen:
        seen.add(name)
        row = conn.execute("SELECT merged_into FROM object_registry WHERE object_type=?", (name,)).fetchone()
        if not row or not row[0]:
            break
        name = row[0]
    return name


def find_object_type(conn: sqlite3.Connection, candidate: str) -> str | None:
    """Exact registry match on snake_case, else alias match on the spaced form; follows synonym merges."""
    snake = candidate.replace(" ", "_")
    row = conn.execute("SELECT object_type FROM object_registry WHERE object_type=?", (snake,)).fetchone()
    if row:
        return _follow_merges(conn, row[0])
    row = conn.execute("SELECT object_type FROM object_alias WHERE alias=?", (candidate.replace("_", " "),)).fetchone()
    return _follow_merges(conn, row[0]) if row else None


def add_auto_new_object_type(conn: sqlite3.Connection, object_type: str, *, raw_label: str, space_id: str,
                             frame_id: int, ts_utc: str, category: str = "other") -> None:
    """Create a registry row flagged for review and log the discovery. Caller owns the transaction."""
    conn.execute(
        """INSERT INTO object_registry(object_type,category,mobility_prior,review_status,
             first_seen_utc,first_seen_frame_id,first_seen_space_id) VALUES(?,?,'unknown','auto_new',?,?,?)""",
        (object_type, category, ts_utc, frame_id, space_id))
    conn.execute(
        """INSERT INTO events(space_id,ts_utc,frame_id,event_type,object_type,resolution,details_json)
           VALUES(?,?,?,'new_object_type',?,'interval',?)""",
        (space_id, ts_utc, frame_id, object_type, json.dumps({"raw_label": raw_label})))


def category_of(conn: sqlite3.Connection, object_type: str) -> str | None:
    row = conn.execute("SELECT category FROM object_registry WHERE object_type=?", (object_type,)).fetchone()
    return row[0] if row else None

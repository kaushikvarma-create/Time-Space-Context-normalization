"""Map any label (model free text or detector class) to a canonical object_registry entry at write time.

Order: registry / alias / merged synonym (via the taxonomy, including naive singular forms) -> otherwise create
a new registry row flagged `auto_new` (plus a new_object_type event). New object types are new ROWS, never new
columns. Must be called inside a transaction (it may write). The raw label is stored separately by the caller,
so later taxonomy changes can re-map history.
"""
from __future__ import annotations

import sqlite3

from ..store.registry import add_auto_new_object_type, find_object_type
from ..taxonomy import singular_candidates, snake, spaced


def canonical(conn: sqlite3.Connection, raw_label: str, *, space_id: str, frame_id: int, ts_utc: str,
              category: str = "other") -> tuple[str, bool]:
    """Return (canonical_object_type, is_new)."""
    k = spaced(raw_label)
    if not k:
        raise ValueError("empty object label")
    for cand in singular_candidates(k):
        found = find_object_type(conn, cand)
        if found:
            return found, False
    new = snake(raw_label)
    add_auto_new_object_type(conn, new, raw_label=raw_label, space_id=space_id, frame_id=frame_id,
                             ts_utc=ts_utc, category=category)
    return new, True

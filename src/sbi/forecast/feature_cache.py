"""A cache of the per-photo columns the table derives from stored readings (counts, areas, grid cells, relations).

Deriving them means decoding and comparing every outline of every photo, which was most of the table's build
time and grew with every photo taken. A photo's columns only change when the photo is read again, the vocabulary
links change, or the derivation settings change, so they are computed once and kept here.

The cache is a separate SQLite file (not the main database, which the dashboard opens read-only). Each entry is
one photo under one `key`; the key is a fingerprint of everything the columns depend on besides the photo
(settings, vocabulary, code version) and `stamp` is when the photo was last read. A different key or stamp is a
miss. The cache is only an accelerator: any problem with it is logged and the columns are computed directly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)
FrameColumns = dict[int, dict[str, float]]       # frame_id -> {column name: value}
_BATCH = 500                                     # ids per SQL statement (SQLite limits bound variables)


def fingerprint(*parts) -> str:
    """A short, stable key for any JSON-serialisable description of what the cached columns depend on."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=sorted).encode()).hexdigest()[:16]


class FeatureCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS frame_columns(frame_id INTEGER NOT NULL, key TEXT NOT NULL, "
                     "stamp TEXT NOT NULL, cols TEXT NOT NULL, PRIMARY KEY(frame_id, key))")
        return conn

    def load(self, key: str, stamps: dict[int, str]) -> FrameColumns:
        """Cached columns for the photos in `stamps` ({frame_id: when it was last read}) that are still current."""
        if not stamps:
            return {}
        out: FrameColumns = {}
        try:
            conn = self._connect()
            try:
                ids = list(stamps)
                for i in range(0, len(ids), _BATCH):
                    part = ids[i:i + _BATCH]
                    for fid, stamp, cols in conn.execute(
                            f"SELECT frame_id, stamp, cols FROM frame_columns WHERE key=? AND frame_id IN "
                            f"({','.join('?' * len(part))})", [key, *part]):
                        if stamp == stamps[fid]:
                            out[int(fid)] = json.loads(cols)
            finally:
                conn.close()
        except (sqlite3.Error, OSError, ValueError) as e:
            log.warning("table cache unreadable (%s): computing without it", e)
            return {}
        return out

    def store(self, key: str, stamps: dict[int, str], columns: FrameColumns) -> None:
        """Keep these photos' columns under `key`; entries of the same photos under older keys are dropped."""
        if not columns:
            return
        try:
            conn = self._connect()
            try:
                with conn:
                    ids = list(columns)
                    for i in range(0, len(ids), _BATCH):
                        part = ids[i:i + _BATCH]
                        conn.execute(f"DELETE FROM frame_columns WHERE key<>? AND frame_id IN "
                                     f"({','.join('?' * len(part))})", [key, *part])
                    conn.executemany("INSERT OR REPLACE INTO frame_columns(frame_id, key, stamp, cols) VALUES(?,?,?,?)",
                                     [(fid, key, stamps[fid], json.dumps(cols)) for fid, cols in columns.items()])
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as e:
            log.warning("table cache not written (%s)", e)

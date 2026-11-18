"""Connection handling, explicit transactions, and numbered SQL migrations.

The connection is in autocommit mode (isolation_level=None); all multi-statement writes must go through
`transaction()`, which commits on success and ROLLS BACK on any exception, so a half-written frame can
never be committed by a later, unrelated write.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..errors import StorageError

log = logging.getLogger(__name__)
MIGRATIONS_DIR = Path(__file__).with_name("migrations")


def connect(path: Path | str) -> sqlite3.Connection:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), timeout=30, isolation_level=None)
    except (OSError, sqlite3.Error) as e:
        raise StorageError(f"cannot open database {p}: {e}") from e
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")   # readers (dashboard, labeller) don't block the writer
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, or ROLLBACK if the block raises. Not re-entrant."""
    if conn.in_transaction:
        raise StorageError("nested transaction: the caller already holds one")
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.Error as e:
        raise StorageError(f"cannot start transaction: {e}") from e
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        try:
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            conn.execute("ROLLBACK")
            raise StorageError(f"commit failed: {e}") from e


def _migration_files(directory: Path) -> list[tuple[int, Path]]:
    files = sorted(directory.glob("[0-9][0-9][0-9]_*.sql"))
    numbered = [(int(f.name[:3]), f) for f in files]
    versions = [n for n, _ in numbered]
    if versions != list(range(1, len(versions) + 1)):
        raise StorageError(f"migrations must be numbered 001.. without gaps or duplicates, found {versions}")
    return numbered


def migrate(conn: sqlite3.Connection, directory: Path = MIGRATIONS_DIR) -> int:
    """Apply pending migrations in order, each atomically; returns the resulting schema version.

    Migrations must be idempotent-safe for databases created before versioning existed (use IF NOT EXISTS).
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    files = _migration_files(directory)
    for n, path in files:
        if n <= current:
            continue
        log.info("applying migration %s", path.name)
        script = f"BEGIN IMMEDIATE;\n{path.read_text()}\nPRAGMA user_version = {n};\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error as e:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise StorageError(f"migration {path.name} failed: {e}") from e
    return max(current, files[-1][0]) if files else current

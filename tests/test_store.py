"""Migrations, transactions, registry, normaliser."""
import json
import sqlite3

import pytest

from sbi.errors import StorageError
from sbi.perception.normalise import canonical
from sbi.store.db import MIGRATIONS_DIR, connect, migrate, transaction


def test_migrate_fresh_is_idempotent(tmp_path):
    c = connect(tmp_path / "x.db")
    assert migrate(c) == 3 and migrate(c) == 3
    assert c.execute("PRAGMA user_version").fetchone()[0] == 3
    names = {r[0] for r in c.execute("SELECT name FROM sqlite_master")}
    assert {"frames", "observations", "labels", "v_frame_counts", "v_observations_wide", "object_links",
            "label_judgments", "relations", "frame_features"} <= names
    assert "raw_label" in {r[1] for r in c.execute("PRAGMA table_info(observations)")}


def test_migrate_upgrades_a_pre_versioning_database(tmp_path):
    """The live DB was created before versioning existed (user_version 0, tables present)."""
    c = connect(tmp_path / "old.db")
    c.executescript((MIGRATIONS_DIR / "001_core.sql").read_text())
    assert c.execute("PRAGMA user_version").fetchone()[0] == 0
    assert migrate(c) == 3


def test_migration_gap_is_rejected(tmp_path):
    d = tmp_path / "m"; d.mkdir()
    (d / "001_a.sql").write_text("CREATE TABLE a(x);"); (d / "003_c.sql").write_text("CREATE TABLE c(x);")
    with pytest.raises(StorageError, match="without gaps"):
        migrate(connect(tmp_path / "g.db"), d)


def test_failed_migration_rolls_back_atomically(tmp_path):
    d = tmp_path / "m"; d.mkdir()
    (d / "001_a.sql").write_text("CREATE TABLE a(x); INSERT INTO nope VALUES(1);")
    c = connect(tmp_path / "f.db")
    with pytest.raises(StorageError, match="001_a.sql"):
        migrate(c, d)
    assert c.execute("PRAGMA user_version").fetchone()[0] == 0
    assert c.execute("SELECT name FROM sqlite_master WHERE name='a'").fetchone() is None
    assert not c.in_transaction


def test_transaction_commits_and_rolls_back(conn):
    with transaction(conn):
        conn.execute("INSERT INTO object_registry(object_type,category) VALUES('t1','x')")
    with pytest.raises(RuntimeError):
        with transaction(conn):
            conn.execute("INSERT INTO object_registry(object_type,category) VALUES('t2','x')")
            raise RuntimeError("boom")
    types = {r[0] for r in conn.execute("SELECT object_type FROM object_registry")}
    assert "t1" in types and "t2" not in types and not conn.in_transaction


def test_nested_transaction_is_refused(conn):
    with transaction(conn):
        with pytest.raises(StorageError, match="nested"):
            with transaction(conn):
                pass


def test_foreign_keys_are_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO observations(frame_id,space_id,ts_utc,object_type,source) "
                     "VALUES(999,'parking','t','person','yolo')")


# ---- normaliser
def _canon(conn, label):
    with transaction(conn):
        return canonical(conn, label, space_id="admin_cabin", frame_id=_frame(conn), ts_utc="2026-11-13T00:00:00+00:00")


def _frame(conn):
    row = conn.execute("SELECT frame_id FROM frames LIMIT 1").fetchone()
    if row:
        return row[0]
    conn.execute("INSERT INTO frames(space_id,ts_utc,ts_local,tz,date,dow,dow_name,hour,minute,slot_15,week_of_year,"
                 "is_weekend,part_of_day,capture_status) VALUES('admin_cabin','t','t','IST','d',0,'Mon',0,0,0,1,0,"
                 "'night','ok')")
    return conn.execute("SELECT frame_id FROM frames").fetchone()[0]


@pytest.mark.parametrize("raw,expected", [("chair", "chair"), ("Couch", "sofa"), ("Couches", "sofa"),
                                          ("water bottle", "bottle"), ("OFFICE  CHAIR", "chair"),
                                          ("Chairs", "chair"), ("desktop_pc", "desktop_pc")])
def test_known_labels_map_to_registry(conn, raw, expected):
    assert _canon(conn, raw) == (expected, False)


def test_new_label_becomes_a_reviewable_row_and_event(conn):
    assert _canon(conn, 'Fire "Extinguisher"') == ('fire_"extinguisher"', True)   # odd chars must not break anything
    assert _canon(conn, 'Fire "Extinguisher"')[1] is False                        # second time it is known
    row = conn.execute("SELECT review_status FROM object_registry WHERE object_type=?",
                       ('fire_"extinguisher"',)).fetchone()
    assert row[0] == "auto_new"
    ev = conn.execute("SELECT details_json FROM events WHERE event_type='new_object_type'").fetchone()
    assert json.loads(ev[0]) == {"raw_label": 'Fire "Extinguisher"'}              # valid JSON despite the quotes


def test_empty_label_is_rejected(conn):
    with pytest.raises(ValueError):
        _canon(conn, "   ")

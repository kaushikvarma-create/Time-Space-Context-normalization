"""The per-photo column cache: the table is identical with it, photos are derived once, and anything that could
change a photo's columns is a miss."""
from datetime import timedelta

import pandas as pd
import pytest

import sbi.forecast.table as table_mod
from sbi.forecast.build import table_for
from sbi.forecast.feature_cache import FeatureCache, fingerprint
from sbi.forecast.table import build_table, frame_columns
from sbi.pipeline import process_frame
from sbi.store.db import connect, migrate
from sbi.store.registry import register_space
from sbi.taxonomy import load_taxonomy

from .conftest import FakeDetector, FakeReader, FakeSource, synthetic_image, thursday_ist
from .test_flow import DETS, cap


def photos(conn, space, root, n=6):
    ids = []
    for i in range(n):
        fid = cap(conn, space, root, FakeSource([synthetic_image(i)]), thursday_ist(9, 0) + timedelta(minutes=15 * i)).frame_id
        process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root)
        ids.append(fid)
    return ids


@pytest.fixture
def counted(monkeypatch):
    """How many photos frame_columns() was asked to derive, per call."""
    calls = []
    real = table_mod.frame_columns
    def spy(conn, tax, frame_ids, **kw):
        calls.append(len(frame_ids))
        return real(conn, tax, frame_ids, **kw)
    monkeypatch.setattr(table_mod, "frame_columns", spy)
    return calls


def build(conn, space, cache=None, **kw):
    return build_table(conn, "admin_cabin", tz=space.tz, conf_levels=(0.05,), grid=(8, 6), cache=cache, **kw)


def test_the_table_is_identical_with_a_cold_cache_a_warm_cache_and_no_cache(conn, space, root, tmp_path, counted):
    photos(conn, space, root)
    plain = build(conn, space)
    cache = FeatureCache(tmp_path / "c" / "features.db")                  # the folder is created on first use
    cold, warm = build(conn, space, cache), build(conn, space, cache)
    pd.testing.assert_frame_equal(plain.table, cold.table)
    pd.testing.assert_frame_equal(plain.table, warm.table)
    pd.testing.assert_frame_equal(plain.catalog, warm.catalog)
    assert counted == [6, 6]                                              # no cache, cold cache; the warm build derived nothing
    assert any(c.startswith("count.yolo.") for c in plain.table.columns)


def test_only_new_and_re_read_photos_are_derived_again(conn, space, root, tmp_path, counted):
    ids = photos(conn, space, root, 4)
    cache = FeatureCache(tmp_path / "features.db")
    build(conn, space, cache)
    fid = cap(conn, space, root, FakeSource([synthetic_image(9)]), thursday_ist(10, 0)).frame_id
    process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root)
    build(conn, space, cache)
    assert counted == [4, 1]                                              # just the new photo
    conn.execute("UPDATE frames SET processed_utc='2030-01-01T00:00:00+00:00' WHERE frame_id=?", (ids[0],))
    conn.execute("UPDATE observations SET count=count+5 WHERE frame_id=? AND source='yolo'", (ids[0],))   # as a re-read would
    again = build(conn, space, cache)
    assert counted == [4, 1, 1]                                           # just the re-read photo
    pd.testing.assert_frame_equal(again.table, build(conn, space).table)  # and its new values are in the table


def test_settings_and_vocabulary_changes_miss_the_cache(conn, space, root, tmp_path, counted):
    photos(conn, space, root, 3)
    cache = FeatureCache(tmp_path / "features.db")
    build(conn, space, cache); build(conn, space, cache, mask_dup_iou=0.9); build(conn, space, cache, mask_dup_iou=0.9)
    assert counted == [3, 3]                                              # another setting: derived again, then cached
    conn.execute("INSERT INTO object_alias(alias, object_type) VALUES('stool', 'chair')")
    plain = build(conn, space, mask_dup_iou=0.9)
    pd.testing.assert_frame_equal(build(conn, space, cache, mask_dup_iou=0.9).table, plain.table)
    assert counted == [3, 3, 3, 3]                                        # the vocabulary changed: all derived again
    one = __import__("sqlite3").connect(tmp_path / "features.db")
    assert one.execute("SELECT COUNT(*), COUNT(DISTINCT key) FROM frame_columns").fetchone() == (3, 1)   # old keys dropped


def test_a_broken_cache_never_breaks_the_table(conn, space, root, tmp_path, caplog):
    photos(conn, space, root, 2)
    bad = tmp_path / "features.db"
    bad.write_text("this is not a database")
    with caplog.at_level("WARNING"):
        t = build(conn, space, FeatureCache(bad))
    pd.testing.assert_frame_equal(t.table, build(conn, space).table)
    assert "table cache unreadable" in caplog.text
    blocked = tmp_path / "file"; blocked.write_text("x")
    pd.testing.assert_frame_equal(build(conn, space, FeatureCache(blocked / "sub" / "features.db")).table, t.table)


def test_cache_entries_follow_key_and_stamp_and_batch_over_many_photos(tmp_path):
    cache = FeatureCache(tmp_path / "f.db")
    stamps = {i: f"t{i}" for i in range(1, 1202)}                         # more than two SQL batches
    cache.store("k1", stamps, {i: {"count.yolo.chair": float(i)} for i in stamps})
    got = cache.load("k1", stamps)
    assert len(got) == 1201 and got[1201] == {"count.yolo.chair": 1201.0}
    assert cache.load("k2", stamps) == {} and cache.load("k1", {}) == {}
    assert set(cache.load("k1", {5: "t5", 6: "re-read"})) == {5}          # a different stamp is a miss
    cache.store("k1", stamps, {})                                         # nothing to store: no error
    assert fingerprint(1, {"b": 2, "a": {3, 1}}) == fingerprint(1, {"a": {1, 3}, "b": 2}) != fingerprint(2, {"a": {1, 3}, "b": 2})


def test_frame_columns_covers_every_photo_asked_for_also_beyond_one_sql_batch(conn, space, root, monkeypatch):
    ids = photos(conn, space, root, 5)
    tax = load_taxonomy(conn)
    kw = dict(conf_levels=(0.05,), mask_sources={"sam2": None}, mask_dup_iou=0.5, grid=(8, 6), cell_area_min=0.02)
    whole = frame_columns(conn, tax, ids + [999], **kw)
    assert set(whole) == set(ids) | {999} and whole[999] == {} and whole[ids[0]]
    monkeypatch.setattr(table_mod, "_SQL_BATCH", 2)                       # force three batches
    assert frame_columns(conn, tax, ids + [999], **kw) == whole


def test_max_slots_keeps_the_newest_slots_only(conn, space, root):
    photos(conn, space, root, 6)
    full, last = build(conn, space), build(conn, space, max_slots=4)
    assert len(full.table) == 6 and len(last.table) == 4 and last.table.index[-1] == full.table.index[-1]
    pd.testing.assert_frame_equal(last.table[[c for c in last.table if c.startswith("count.")]],
                                  full.table.iloc[-4:][[c for c in last.table if c.startswith("count.")]], check_freq=False)
    assert len(build(conn, space, max_slots=100).table) == 6


def test_table_for_uses_the_cache_folder_from_settings(settings, space, root, counted):
    c = connect(settings.db_path); migrate(c); register_space(c, space, "config/spaces/admin_cabin.yaml")
    photos(c, space, root, 3)
    a = table_for(c, settings, space); b = table_for(c, settings, space); plain = table_for(c, settings, space, use_cache=False)
    assert (settings.cache_dir / "table_features.db").exists() and counted == [3, 3]      # cached build, then uncached
    pd.testing.assert_frame_equal(a.table, b.table); pd.testing.assert_frame_equal(a.table, plain.table)
    assert len(table_for(c, settings, space, max_slots=2).table) == 2

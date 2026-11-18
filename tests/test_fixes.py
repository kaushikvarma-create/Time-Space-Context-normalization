"""Regression tests for the inconsistencies fixed on 18 Nov 2026 (see docs/technical-deep-dive.md section 5.2)."""
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from sbi.errors import ConfigError
from sbi.forecast.table import CORRUPT_OUTLINE, build_table, calendar
from sbi.perception.detector_yolo import build_detections
from sbi.perception.scene_lmstudio import parse_reading
from sbi.pipeline import process_frame
from sbi.settings import DetectorSettings, TableSettings, load_dotenv, load_settings
from sbi.store.db import connect, migrate
from sbi.store.forecasts import insert_forecasts, insert_run, load_forecast
from sbi.tick import run_tick

from .conftest import FakeDetector, FakeReader, FakeSegmenter, FakeSource, det, make_reading, synthetic_image, thursday_ist
from .test_flow import DETS, cap, make_root


def test_counted_always_agrees_with_the_stored_confidence(space):
    cfg = DetectorSettings(conf=0.25)
    d = build_detections([([0.1, 0.1, 0.2, 0.2], 0.2496, "chair"), ([0.3, 0.1, 0.4, 0.2], 0.2494, "chair")], space, cfg)
    assert [(x.conf, x.counted) for x in d] == [(0.25, True), (0.249, False)]


def test_dotenv_handles_export_quotes_and_trailing_comments(tmp_path, monkeypatch):
    for k in ("SBI_T_A", "SBI_T_B", "SBI_T_C", "SBI_T_D"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text("export SBI_T_A=one\nSBI_T_B='two # not a comment'\nSBI_T_C=three  # note\nSBI_T_D=\"x'\"\n")
    load_dotenv(tmp_path / ".env")
    assert [os.environ[k] for k in ("SBI_T_A", "SBI_T_B", "SBI_T_C", "SBI_T_D")] == ["one", "two # not a comment", "three", "x'"]


def test_a_settings_file_that_is_not_a_mapping_is_a_config_error(tmp_path):
    (tmp_path / "config").mkdir()
    for text, match in (("- a\n- b\n", "must be a mapping"), ("root: /x\n", "'root' is not a setting")):
        (tmp_path / "config" / "pipeline.yaml").write_text(text)
        with pytest.raises(ConfigError, match=match):
            load_settings(tmp_path)


def test_conf_levels_must_be_whole_percents():
    with pytest.raises(ValueError, match="two decimals"):
        TableSettings(conf_levels=[0.055])
    assert TableSettings(conf_levels=[0.10, 0.05]).conf_levels == [0.05, 0.10]


def test_any_name_for_a_person_from_the_vlm_is_not_counted_twice(conn, space, root):
    reading = make_reading(people=2, objects=(("persons", 2, "idle"), ("chair", 3, "idle")))
    fid = cap(conn, space, root, FakeSource([synthetic_image()])).frame_id
    process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(reading), root=root)
    rows = conn.execute("SELECT object_type, count FROM observations WHERE frame_id=? AND source='vlm' ORDER BY object_type",
                        (fid,)).fetchall()
    assert [tuple(r) for r in rows] == [("chair", 3), ("person", 2)]


def test_malformed_vlm_items_raise_value_error_or_are_dropped(space):
    base = {"people_count": 1, "objects": [{"label": "chair", "count": 2}], "people": [{"posture": "seated"}, "x"],
            "relations": ["bad", {"subject": "person", "predicate": "sitting_on", "object": "chair", "count": 1}]}
    r = parse_reading(json.dumps(base), space, model="m", latency_s=0)
    assert len(r.people) == 1 and len(r.relations) == 1
    for bad in ({**base, "objects": ["chair"]}, {**base, "objects": {"chair": 2}}):
        with pytest.raises(ValueError):
            parse_reading(json.dumps(bad), space, model="m", latency_s=0)


def test_undoing_a_merge_restores_everything_it_changed(conn):
    from sbi.curation.curator import apply_same, undo_same
    from sbi.store.db import transaction
    with transaction(conn):
        for name, status in (("sofa", "seed"), ("couchette", "auto_new"), ("lounger", "auto_new")):
            conn.execute("INSERT OR IGNORE INTO object_registry(object_type, review_status) VALUES(?, ?)", (name, status))
        conn.execute("UPDATE object_registry SET review_status='auto_new' WHERE object_type='couchette'")
        conn.execute("UPDATE object_registry SET merged_into='couchette' WHERE object_type='lounger'")
        conn.execute("INSERT OR REPLACE INTO object_alias(alias, object_type) VALUES('day bed', 'couchette')")
        conn.execute("INSERT INTO object_links(child, parent, relation, status, source, created_utc) "
                     "VALUES('couchette', 'sofa', 'same', 'approved', 'test', 'now')")
        before = (conn.execute("SELECT object_type, merged_into, review_status FROM object_registry ORDER BY 1").fetchall(),
                  conn.execute("SELECT alias, object_type FROM object_alias ORDER BY 1").fetchall())
        apply_same(conn, "couchette", "sofa")
    assert conn.execute("SELECT merged_into FROM object_registry WHERE object_type='lounger'").fetchone()[0] == "sofa"
    with transaction(conn):
        undo_same(conn, "couchette", "sofa")
    after = (conn.execute("SELECT object_type, merged_into, review_status FROM object_registry ORDER BY 1").fetchall(),
             conn.execute("SELECT alias, object_type FROM object_alias ORDER BY 1").fetchall())
    assert after == before
    assert "undo" not in json.loads(conn.execute("SELECT evidence_json FROM object_links").fetchone()[0])


def test_deleting_a_frame_deletes_its_events(conn, space, root):
    fid = cap(conn, space, root, FakeSource([synthetic_image()])).frame_id
    conn.execute("INSERT INTO events(space_id, ts_utc, frame_id, event_type) VALUES('admin_cabin', 't', ?, 'became_empty')", (fid,))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DELETE FROM frames WHERE frame_id=?", (fid,))
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_migrate_never_reports_a_lower_version_than_the_database_has(tmp_path):
    c = connect(tmp_path / "v.db")
    c.execute("PRAGMA user_version = 99")
    assert migrate(c) == 99


def test_quantiles_below_ten_percent_keep_their_names(conn):
    rid = insert_run(conn, space_id="admin_cabin", origin_local="2026-11-12T10:00:00", model="m", variant="all", horizon=1,
                     step_minutes=15, context_slots=1, targets=["a"], covariates=[], quantiles=[0.05, 0.5, 0.95])
    insert_forecasts(conn, rid, pd.DataFrame({"timestamp": [pd.Timestamp("2026-11-12 10:15")], "target": "a", "step": 1,
                                              "q05": 1.0, "q50": 2.0, "q95": 3.0}))
    assert {"q05", "q50", "q95"} <= set(load_forecast(conn, rid).columns)


def test_slot_of_day_follows_the_step():
    idx = pd.date_range("2026-11-12 01:00", periods=2, freq="30min")
    assert calendar(idx)["cal.slot_of_day"].tolist() == [2, 3]
    assert calendar(pd.date_range("2026-11-12 01:00", periods=1, freq="15min"))["cal.slot_of_day"].tolist() == [4]


def test_an_unreadable_outline_leaves_that_photos_outline_columns_empty(conn, space, root, caplog):
    ids = []
    for m in (0, 15):
        fid = cap(conn, space, root, FakeSource([synthetic_image(m)]), thursday_ist(9, m)).frame_id
        process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root, segmenter=FakeSegmenter())
        ids.append(fid)
    conn.execute("UPDATE detections SET mask=x'00' WHERE det_id=(SELECT MIN(det_id) FROM detections WHERE frame_id=?)", (ids[0],))
    with caplog.at_level("WARNING"):
        t = build_table(conn, "admin_cabin", tz=space.tz, grid=(8, 6)).table
    area = [c for c in t.columns if c.startswith("area.sam2.")]
    assert area and t[area].iloc[0].isna().all() and t[area].iloc[1].notna().all()
    assert t["count.yolo.person"].notna().all() and CORRUPT_OUTLINE not in t.columns and "unreadable stored outline" in caplog.text


def test_start_and_end_are_floored_to_the_slot_grid(conn, space, root):
    for m in (0, 15, 30):
        fid = cap(conn, space, root, FakeSource([synthetic_image(m)]), thursday_ist(9, m)).frame_id
        process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root)
    t = build_table(conn, "admin_cabin", tz=space.tz, start="2026-11-12 09:07", end="2026-11-12 09:40").table
    assert [str(x) for x in t.index] == ["2026-11-12 09:00:00", "2026-11-12 09:15:00", "2026-11-12 09:30:00"]
    assert (t["slot_status"] == "ok").all()


def test_tick_times_each_photo_rebuilds_every_bucket_and_fails_a_space_whose_photo_failed(settings, space_dir, monkeypatch):
    files = make_root(settings.root, space_dir)
    src = FakeSource([synthetic_image()])
    out = run_tick(settings, files, source=src, reader=FakeReader())          # no `now`: each photo timed when taken
    c = connect(settings.db_path)
    stamps = [r[0] for r in c.execute("SELECT ts_utc FROM frames ORDER BY frame_id")]
    assert len(stamps) == 2 and all(o.ok for o in out)
    import sbi.tick as tick_mod
    rebuilt = []
    monkeypatch.setattr(tick_mod.baseline, "rebuild", lambda conn, sid, alpha, dow, hour: rebuilt.append((sid, dow, hour)))
    # an old unprocessed photo (another hour) is in the backlog: its bucket is rebuilt too
    from sbi.ingest.capture import capture_frame
    from sbi.spaces import load_space
    cab = load_space(space_dir / "admin_cabin.yaml")
    capture_frame(c, cab, FakeSource([synthetic_image(3)]), root=settings.root, image_dir=settings.image_dir,
                  now=thursday_ist(6, 0))
    run_tick(settings, files, source=src, detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist(10, 0))
    cab_buckets = {(d, h) for s, d, h in rebuilt if s == "admin_cabin"}
    assert (3, 6) in cab_buckets and (3, 10) in cab_buckets                   # Thursday 06:00 and 10:00
    out = {o.space_id: o for o in run_tick(settings, files, source=src, detector=FakeDetector(error=None, dets=[]),
                                           reader=FakeReader(), now=thursday_ist(10, 15))}
    assert out["admin_cabin"].ok and out["admin_cabin"].failed == 0
    from sbi.errors import StorageError
    def broken(*a, **k):
        raise StorageError("disk full")
    monkeypatch.setattr(tick_mod, "process_frame", broken)
    out = {o.space_id: o for o in run_tick(settings, files, source=src, detector=FakeDetector(DETS), reader=FakeReader(),
                                           now=thursday_ist(10, 30))}
    assert not out["admin_cabin"].ok and out["admin_cabin"].failed >= 1 and "could not be processed" in out["admin_cabin"].error


def test_dashboard_leads_follow_the_runs_step():
    from sbi.dashboard import data
    from .test_forecast import make_table
    table = make_table(200)
    ts = pd.date_range(table.index[-1] + timedelta(minutes=30), periods=8, freq="30min")
    fc = pd.DataFrame({"target": "scene.people_yolo", "timestamp": ts, "step": range(1, 9),
                       "q05": 0.0, "q50": [float(v) for v in range(1, 9)], "q95": 9.0})
    row = data.summary(table, fc, ["scene.people_yolo"], step_minutes=30).iloc[0]
    assert (row["+1 h"], row["+2 h"], row["+4 h"]) == (2.0, 4.0, 8.0) and (row["+1 h low"], row["+1 h high"]) == (0.0, 9.0)
    cd = data.chart_data(table, fc, "scene.people_yolo", days=1)
    assert cd.loc[cd["kind"] == "forecast", ["low", "high"]].iloc[0].tolist() == [0.0, 9.0]   # q05..q95, not text order

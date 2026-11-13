"""capture -> pipeline -> baseline -> tick, with fakes."""
import json
from pathlib import Path

import numpy as np
import pytest

from sbi import baseline
from sbi.errors import CameraError, DetectorError, StorageError, VLMError
from sbi.ingest.capture import capture_frame
from sbi.labeling import core as labeling
from sbi.eval import vlm_eval
from sbi.pipeline import process_frame
from sbi.tick import run_tick

from .conftest import FakeDetector, FakeReader, FakeSource, det, make_reading, synthetic_image, thursday_ist

DETS = [det("person", (0.10, 0.20, 0.20, 0.60)), det("person", (0.60, 0.20, 0.70, 0.60)),
        det("bag", (0.30, 0.50, 0.35, 0.60)), det("chair", (0.40, 0.30, 0.55, 0.90))]


def cap(conn, space, root, source, now=None):
    return capture_frame(conn, space, source, root=root, image_dir=root / "data" / "images",
                         now=now or thursday_ist())


def count(conn, table, **where):
    q = f"SELECT COUNT(*) FROM {table}" + (" WHERE " + " AND ".join(f"{k}=?" for k in where) if where else "")
    return conn.execute(q, tuple(where.values())).fetchone()[0]


# ---------------------------------------------------------------- capture
def test_capture_ok_stores_ist_named_relative_image(conn, space, root):
    r = cap(conn, space, root, FakeSource([synthetic_image()]))
    row = conn.execute("SELECT * FROM frames WHERE frame_id=?", (r.frame_id,)).fetchone()
    assert (r.status, r.created) == ("ok", True)
    assert row["ts_local"] == "2026-11-12T10:00:00+05:30" and row["hour"] == 10 and row["dow_name"] == "Thu"
    assert row["image_path"] == "data/images/admin_cabin/2026-11-12/20261112T100000+0530.jpg"
    assert (root / row["image_path"]).exists() and row["brightness"] > 0 and row["width"] == 320


def test_camera_failure_is_data_not_zero(conn, space, root):
    r = cap(conn, space, root, FakeSource([CameraError("cannot open stream rtsp://***:***@h/x")]))
    row = conn.execute("SELECT * FROM frames WHERE frame_id=?", (r.frame_id,)).fetchone()
    assert r.status == "camera_down" and row["image_path"] is None and row["brightness"] is None
    assert not (root / "data" / "images").exists()


def test_flat_black_frame_is_flagged_not_stored(conn, space, root):
    r = cap(conn, space, root, FakeSource([np.zeros((90, 160, 3), np.uint8)]))
    row = conn.execute("SELECT capture_status, image_path FROM frames WHERE frame_id=?", (r.frame_id,)).fetchone()
    assert r.status == "bad_frame" and tuple(row) == ("bad_frame", None)
    assert not (root / "data" / "images").exists()


def test_duplicate_timestamp_is_idempotent_and_returns_the_right_id(conn, space, root):
    a = cap(conn, space, root, FakeSource([synthetic_image()]))
    b = cap(conn, space, root, FakeSource([synthetic_image(1)]))
    assert (a.frame_id, a.created, b.frame_id, b.created) == (a.frame_id, True, a.frame_id, False)
    assert count(conn, "frames") == 1


def test_duplicate_returns_its_own_id_even_after_other_rows_were_inserted(conn, space, parking, root):
    """Regression: INSERT OR IGNORE + lastrowid returned the id of whichever row was inserted last."""
    a = cap(conn, space, root, FakeSource([synthetic_image()]))
    other = cap(conn, parking, root, FakeSource([synthetic_image(3)]))          # a different frame, inserted later
    again = cap(conn, space, root, FakeSource([synthetic_image(1)]))            # duplicate of `a`
    assert other.frame_id != a.frame_id and (again.frame_id, again.created) == (a.frame_id, False)


def test_disk_failure_raises_storage_error_and_records_nothing(conn, space, root):
    blocker = root / "blocked"; blocker.write_text("i am a file")
    with pytest.raises(StorageError, match="cannot write image"):
        capture_frame(conn, space, FakeSource([synthetic_image()]), root=root, image_dir=blocker / "images",
                      now=thursday_ist())
    assert count(conn, "frames") == 0


# ---------------------------------------------------------------- pipeline
def processed(conn, space, root, detector=None, reader=None, now=None, source=None):
    c = cap(conn, space, root, source or FakeSource([synthetic_image()]), now)
    res = process_frame(conn, space, c.frame_id, detector or FakeDetector(DETS), reader or FakeReader(), root=root)
    return c.frame_id, res


def test_pipeline_writes_detector_vlm_scene_and_registry(conn, space, root):
    reading = make_reading(people=2, objects=(("chair", 6, "idle"), ("Fire Extinguisher", 1, "present")))
    fid, res = processed(conn, space, root, reader=FakeReader(reading))
    assert res.processed and (res.people_yolo, res.people_vlm, res.boxes, res.vlm_ok) == (2, 2, 4, True)
    cells = {r["object_type"]: r["cell_id"] for r in conn.execute(
        "SELECT object_type,cell_id FROM observations WHERE frame_id=? AND source='yolo' AND object_type!='person'", (fid,))}
    assert cells["bag"] == "R3C2" and cells["chair"] == "R5C3"        # geometric, bottom-centre anchor on an 8x6 grid
    assert count(conn, "observations", frame_id=fid, source="vlm") == 3       # person + chair + fire_extinguisher
    s = conn.execute("SELECT * FROM scene_state WHERE frame_id=?", (fid,)).fetchone()
    assert (s["people_yolo"], s["people_vlm"], s["count_disagree"], s["lighting"]) == (2, 2, 0, "artificial")
    assert conn.execute("SELECT review_status FROM object_registry WHERE object_type='fire_extinguisher'").fetchone()[0] == "auto_new"
    f = conn.execute("SELECT * FROM frames WHERE frame_id=?", (fid,)).fetchone()
    assert f["processed_utc"] and f["detector_version"] == "fake-det@1" and f["prompt_version"] == "vtest"


def test_vlm_failure_keeps_detector_results_with_null_vlm_fields(conn, space, root):
    fid, res = processed(conn, space, root, reader=FakeReader(error=VLMError("LM Studio down")))
    assert res.processed and not res.vlm_ok and res.people_vlm is None
    s = conn.execute("SELECT * FROM scene_state WHERE frame_id=?", (fid,)).fetchone()
    assert s["people_vlm"] is None and s["count_disagree"] is None and s["people_yolo"] == 2   # NULL, never 0
    assert count(conn, "observations", frame_id=fid, source="vlm") == 0 and count(conn, "observations", frame_id=fid) == 4
    assert json.loads(conn.execute("SELECT vlm_raw FROM frames WHERE frame_id=?", (fid,)).fetchone()[0])["error"]


def test_detector_failure_propagates_and_writes_nothing(conn, space, root):
    c = cap(conn, space, root, FakeSource([synthetic_image()]))
    with pytest.raises(DetectorError):
        process_frame(conn, space, c.frame_id, FakeDetector(error=DetectorError("cuda oom")), FakeReader(), root=root)
    assert count(conn, "observations") == 0 and count(conn, "scene_state") == 0
    assert conn.execute("SELECT processed_utc FROM frames").fetchone()[0] is None      # stays in the backlog


def test_midway_failure_rolls_back_everything_and_cannot_leak_into_later_commits(conn, space, root):
    """Regression: a half-written frame used to be committed by the next, unrelated write."""
    bad = make_reading(objects=(("chair", 2, "idle"), ("", 1, "idle")))      # empty label blows up mid-write
    c = cap(conn, space, root, FakeSource([synthetic_image()]))
    with pytest.raises(ValueError):
        process_frame(conn, space, c.frame_id, FakeDetector(DETS), FakeReader(bad), root=root)
    assert not conn.in_transaction
    cap(conn, space, root, FakeSource([synthetic_image(2)]), now=thursday_ist(10, 15))   # a later, unrelated commit
    assert count(conn, "observations") == 0 and count(conn, "scene_state") == 0
    assert conn.execute("SELECT processed_utc FROM frames WHERE frame_id=?", (c.frame_id,)).fetchone()[0] is None


def test_reprocessing_replaces_results_instead_of_duplicating(conn, space, root):
    fid, _ = processed(conn, space, root)
    before = count(conn, "observations", frame_id=fid)
    process_frame(conn, space, fid, FakeDetector(DETS[:1]), FakeReader(make_reading(people=1, objects=())), root=root)
    assert count(conn, "observations", frame_id=fid) == 2          # 1 yolo + 1 vlm person
    assert before == 6 and count(conn, "scene_state", frame_id=fid) == 1


def test_non_ok_and_missing_frames(conn, space, root):
    c = cap(conn, space, root, FakeSource([CameraError("x")]))
    res = process_frame(conn, space, c.frame_id, FakeDetector(DETS), FakeReader(), root=root)
    assert not res.processed and "camera_down" in res.reason
    with pytest.raises(StorageError, match="does not exist"):
        process_frame(conn, space, 9999, FakeDetector(), FakeReader(), root=root)


def test_occupancy_events(conn, space, root):
    processed(conn, space, root, now=thursday_ist(10, 0))                                       # 2 people
    processed(conn, space, root, FakeDetector([]), FakeReader(make_reading(0, ())), thursday_ist(10, 15))
    processed(conn, space, root, FakeDetector(DETS), FakeReader(), thursday_ist(10, 30))
    kinds = [r[0] for r in conn.execute("SELECT event_type FROM events ORDER BY ts_utc")]
    assert kinds == ["became_empty", "became_occupied"]


# ---------------------------------------------------------------- baseline
def test_baseline_counts_empty_cells_as_zero_but_excludes_missing_frames(conn, space, root):
    only_one = [det("person", (0.10, 0.20, 0.20, 0.60))]
    processed(conn, space, root, FakeDetector(only_one), now=thursday_ist(10, 0, 0))
    processed(conn, space, root, FakeDetector(only_one), now=thursday_ist(10, 0, 1))
    processed(conn, space, root, FakeDetector([]), FakeReader(make_reading(0, ())), now=thursday_ist(10, 0, 2))
    cap(conn, space, root, FakeSource([CameraError("down")]), now=thursday_ist(10, 0, 3))      # excluded entirely
    baseline.rebuild(conn, "admin_cabin", alpha=0.5)
    b = conn.execute("SELECT * FROM baseline_cell WHERE object_type='person'").fetchone()
    assert b["n_frames"] == 3 and b["mean_count"] == pytest.approx(2 / 3) and b["presence_rate"] == pytest.approx(2 / 3)
    ppl = conn.execute("SELECT * FROM baseline_space WHERE metric='people_yolo'").fetchone()
    assert ppl["n_frames"] == 3 and ppl["mean_value"] == pytest.approx(2 / 3)


def test_baseline_vlm_failure_is_excluded_not_zero(conn, space, root):
    processed(conn, space, root, reader=FakeReader(make_reading(4, ())), now=thursday_ist(10, 0, 0))
    processed(conn, space, root, reader=FakeReader(error=VLMError("x")), now=thursday_ist(10, 0, 1))
    baseline.rebuild(conn, "admin_cabin")
    v = conn.execute("SELECT * FROM baseline_space WHERE metric='people_vlm'").fetchone()
    assert v["n_frames"] == 1 and v["mean_value"] == 4.0


def test_bucket_scoped_rebuild_leaves_other_buckets_alone(conn, space, root):
    processed(conn, space, root, now=thursday_ist(10))
    processed(conn, space, root, now=thursday_ist(14))
    baseline.rebuild(conn, "admin_cabin")
    conn.execute("UPDATE baseline_space SET mean_value=-1 WHERE hour=14")
    baseline.rebuild(conn, "admin_cabin", dow=3, hour=10)
    assert conn.execute("SELECT mean_value FROM baseline_space WHERE hour=14 LIMIT 1").fetchone()[0] == -1


# ---------------------------------------------------------------- tick
def make_root(root: Path, space_dir: Path):
    return sorted(space_dir.glob("*.yaml"))


def test_tick_isolates_a_failing_space_and_keeps_the_other(settings, space_dir):
    class Mixed:
        def grab(self, space):
            if space.space_id == "parking":
                raise RuntimeError("driver crash (a bug)")
            return synthetic_image()
    out = run_tick(settings, make_root(settings.root, space_dir), source=Mixed(), detector=FakeDetector(DETS),
                   reader=FakeReader(), now=thursday_ist())
    by = {o.space_id: o for o in out}
    assert by["admin_cabin"].ok and by["admin_cabin"].processed == 1
    assert not by["parking"].ok and "unexpected" in by["parking"].error


def test_camera_down_in_one_space_does_not_fail_the_tick(settings, space_dir):
    class OneDown:
        def grab(self, space):
            if space.space_id == "parking":
                raise CameraError("offline")
            return synthetic_image()
    out = {o.space_id: o for o in run_tick(settings, make_root(settings.root, space_dir), source=OneDown(),
                                           detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist())}
    assert out["parking"].ok and out["parking"].capture_status == "camera_down" and out["parking"].processed == 0
    assert out["admin_cabin"].processed == 1


def test_invalid_space_file_is_skipped_others_run(settings, space_dir):
    (space_dir / "broken.yaml").write_text("space_id: broken\nunknown_key: 1\n")
    out = run_tick(settings, make_root(settings.root, space_dir), source=FakeSource([synthetic_image()]),
                   detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist())
    ok = {o.space_id for o in out if o.ok}
    assert ok == {"admin_cabin", "parking"} and any(o.space_id == "broken" and not o.ok for o in out)


def test_detector_outage_still_captures_and_backlog_recovers(settings, space_dir):
    """No weights in the tmp root -> real YoloDetector fails to load -> frames queue; next tick drains them."""
    files = make_root(settings.root, space_dir)
    src = FakeSource([synthetic_image()])
    first = run_tick(settings, files, source=src, reader=FakeReader(), now=thursday_ist(10, 0))
    assert all(o.ok and o.capture_status == "ok" and o.processed == 0 for o in first)
    from sbi.store.db import connect
    c = connect(settings.db_path)
    assert c.execute("SELECT COUNT(*) FROM frames WHERE processed_utc IS NULL").fetchone()[0] == 2
    second = run_tick(settings, files, source=src, detector=FakeDetector(DETS), reader=FakeReader(),
                      now=thursday_ist(10, 15))
    assert all(o.processed == 2 for o in second)                       # new frame + 1 backlog frame per space
    assert c.execute("SELECT COUNT(*) FROM frames WHERE processed_utc IS NULL").fetchone()[0] == 0


def test_missing_camera_credentials_is_a_fatal_config_error(settings, space_dir, monkeypatch):
    from sbi.errors import ConfigError
    for k in ("SBI_NVR_HOST", "SBI_NVR_USER", "SBI_NVR_PASS"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ConfigError, match="SBI_NVR"):
        run_tick(settings, make_root(settings.root, space_dir), detector=FakeDetector(), reader=FakeReader())


# ---------------------------------------------------------------- labelling + eval
def test_labelling_flow_and_eval(conn, parking, space, root):
    ids = [processed(conn, space, root, now=thursday_ist(10, 15 * i))[0] for i in range(3)]
    assert labeling.progress(conn, "admin_cabin") == (0, 3)
    served = []
    for _ in range(3):
        fid = labeling.next_unlabeled(conn, "admin_cabin"); served.append(fid)
        labeling.save_label(conn, fid, {"person": 3, "chair": 3}, usable=True, labeler="t")
    assert sorted(served) == sorted(ids) and labeling.next_unlabeled(conn, "admin_cabin") is None
    assert labeling.progress(conn, "admin_cabin") == (3, 3)
    mc = labeling.model_counts(conn, ids[0], ["person", "chair"])
    assert mc["person"] == {"yolo": 2, "vlm": 2} and mc["chair"]["vlm"] == 3
    df = vlm_eval.load_labelled(conn)
    out = vlm_eval.summarise(df, ["space_id", "object_type"]).set_index(["object_type", "source"])
    assert out.loc[("person", "yolo"), "bias"] == -1.0 and out.loc[("person", "yolo"), "within1"] == 100.0
    assert out.loc[("chair", "vlm"), "exact"] == 100.0
    assert "labelled frames" in vlm_eval.report(conn)


def test_labelling_validation_and_unusable_frames(conn, space, root):
    fid, _ = processed(conn, space, root)
    with pytest.raises(ValueError):
        labeling.save_label(conn, fid, {"person": -1}, usable=True, labeler="t")
    labeling.save_label(conn, fid, {}, usable=False, labeler="t", note="blurry")
    assert count(conn, "labels") == 0 and count(conn, "label_frames", usable=0) == 1
    assert vlm_eval.report(conn) == "no labels yet: run scripts/label.sh"
    with pytest.raises(StorageError):
        labeling.load_frame_image(conn, 9999, root)

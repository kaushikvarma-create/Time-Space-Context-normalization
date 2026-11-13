"""Raw detections below the counting threshold: storage, thresholds, table columns, detector-only reprocess,
and the label-free stability report."""
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sbi.errors import StorageError, VLMError
from sbi.eval.stability import stability
from sbi.forecast.table import build_table, level_source, select_covariates
from sbi.perception.detector_yolo import build_detections
from sbi.perception.geometry import union_area
from sbi.perception.scene_lmstudio import LmStudioSceneReader
from sbi.pipeline import process_frame
from sbi.settings import DetectorSettings, SceneReaderSettings, TableSettings

from .conftest import FakeDetector, FakeReader, FakeSource, det, make_reading, synthetic_image, thursday_ist
from .test_flow import DETS, cap, count

LOW = [det("chair", (0.70, 0.30, 0.80, 0.60), conf=0.12, counted=False),
       det("person", (0.85, 0.10, 0.95, 0.50), conf=0.07, counted=False)]


# ---------------------------------------------------------------- settings and thresholds
def test_counting_thresholds_must_not_be_below_the_storage_floor():
    with pytest.raises(ValidationError, match="store_conf"):
        DetectorSettings(conf=0.03, store_conf=0.05)
    with pytest.raises(ValidationError, match=r"class_conf\[motorcycle\]"):
        DetectorSettings(class_conf={"motorcycle": 0.01})
    cfg = DetectorSettings(class_conf={"motorcycle": 0.10})
    assert cfg.count_threshold("motorcycle") == 0.10 and cfg.count_threshold("car") == 0.25
    with pytest.raises(ValidationError, match="conf_levels"):
        TableSettings(conf_levels=[0.05, 0.051])
    assert TableSettings(conf_levels=[0.10, 0.05]).conf_levels == [0.05, 0.10] and level_source(0.05) == "yolo_c05"


def test_build_detections_whitelists_marks_counted_and_honours_per_label_thresholds(space):
    raw = [([0.1, 0.1, 0.2, 0.5], 0.90, "person"), ([0.5, 0.1, 0.6, 0.5], 0.24, "person"),
           ([0.3, 0.3, 0.4, 0.6], 0.12, "chair"), ([0.7, 0.7, 0.9, 0.9], 0.80, "zebra")]
    out = {(d.label, d.conf): d.counted for d in build_detections(raw, space, DetectorSettings())}
    assert out == {("person", 0.9): True, ("person", 0.24): False, ("chair", 0.12): False}     # zebra not whitelisted
    relaxed = build_detections(raw, space, DetectorSettings(class_conf={"chair": 0.10}))
    assert {d.label for d in relaxed if d.counted} == {"person", "chair"}


def test_low_confidence_boxes_never_change_the_counted_set(space):
    """The counted boxes must be exactly what running the model at the counting threshold gives, so history
    stays comparable. Low boxes sit on top of, inside and next to strong ones here."""
    cfg = DetectorSettings()
    strong = [([0.10, 0.10, 0.30, 0.60], 0.90, "person"), ([0.50, 0.20, 0.70, 0.70], 0.60, "chair"),
              ([0.52, 0.22, 0.71, 0.71], 0.40, "chair"), ([0.80, 0.50, 0.90, 0.60], 0.30, "handbag")]
    weak = [([0.11, 0.11, 0.29, 0.59], 0.20, "person"), ([0.15, 0.20, 0.25, 0.40], 0.08, "person"),
            ([0.50, 0.20, 0.70, 0.70], 0.24, "chair"), ([0.80, 0.50, 0.90, 0.60], 0.10, "backpack"),
            ([0.01, 0.01, 0.05, 0.05], 0.06, "chair")]
    at_threshold = build_detections(strong, space, cfg)
    with_low = build_detections(strong + weak, space, cfg)
    assert [d for d in with_low if d.counted] == at_threshold and all(d.counted for d in at_threshold)
    assert [d.conf for d in with_low if not d.counted] == [0.06]       # the rest were duplicates of stronger boxes


# ---------------------------------------------------------------- pipeline storage
def test_uncounted_boxes_are_stored_raw_but_never_counted(conn, space, root):
    c = cap(conn, space, root, FakeSource([synthetic_image()]))
    res = process_frame(conn, space, c.frame_id, FakeDetector(DETS + LOW), FakeReader(), root=root)
    assert (res.boxes, res.boxes_stored, res.people_yolo) == (4, 6, 2)
    assert count(conn, "observations", frame_id=c.frame_id, source="yolo") == 4
    assert count(conn, "detections", frame_id=c.frame_id) == 6 and count(conn, "detections", counted=1) == 4
    low = conn.execute("SELECT raw_label, confidence, cell_id FROM detections WHERE counted=0 ORDER BY confidence").fetchall()
    assert [tuple(r) for r in low] == [("person", 0.07, "R3C7"), ("chair", 0.12, "R3C6")]   # bottom edge y=0.5 is row 3 of 6
    f = conn.execute("SELECT detector_min_conf, detector_raw FROM frames WHERE frame_id=?", (c.frame_id,)).fetchone()
    assert f["detector_min_conf"] == 0.05
    assert [d["counted"] for d in json.loads(f["detector_raw"])] == [True] * 4 + [False] * 2
    assert conn.execute("SELECT people_yolo FROM scene_state").fetchone()[0] == 2
    # a low box overlapping a counted one must not create a geometric link
    process_frame(conn, space, c.frame_id, FakeDetector(DETS + [det("laptop", (0.10, 0.20, 0.20, 0.60), 0.1, False)]),
                  FakeReader(), root=root)
    assert count(conn, "relations", source="geometry", object="laptop") == 0
    assert count(conn, "detections", frame_id=c.frame_id) == 5         # reprocessing replaces, never duplicates


def test_detector_only_reprocess_keeps_the_stored_vlm_answer(conn, space, root):
    reading = make_reading(people=3, objects=(("chair", 5, "idle"),), raw='{"stored": true}')
    c = cap(conn, space, root, FakeSource([synthetic_image()]))
    process_frame(conn, space, c.frame_id, FakeDetector(DETS), FakeReader(reading), root=root)
    conn.execute("UPDATE frames SET vlm_model='old-vlm', prompt_version='v-old' WHERE frame_id=?", (c.frame_id,))
    before = conn.execute("SELECT object_type, count FROM observations WHERE source='vlm' ORDER BY 1").fetchall()

    reader = FakeReader(reading)
    res = process_frame(conn, space, c.frame_id, FakeDetector(DETS[:1] + LOW), reader, root=root, replay_vlm=True)
    assert reader.calls == 0 and res.vlm_ok and (res.people_yolo, res.people_vlm) == (1, 3)     # no model call
    assert conn.execute("SELECT object_type, count FROM observations WHERE source='vlm' ORDER BY 1").fetchall() == before
    f = conn.execute("SELECT vlm_model, prompt_version, vlm_raw FROM frames").fetchone()
    assert tuple(f) == ("old-vlm", "v-old", '{"stored": true}')                                 # provenance kept
    assert count(conn, "observations", source="yolo") == 1 and count(conn, "detections") == 3


def test_detector_only_reprocess_keeps_a_stored_vlm_failure_as_unknown(conn, space, root):
    c = cap(conn, space, root, FakeSource([synthetic_image()]))
    process_frame(conn, space, c.frame_id, FakeDetector(DETS), FakeReader(error=VLMError("down")), root=root)
    res = process_frame(conn, space, c.frame_id, FakeDetector(DETS), FakeReader(), root=root, replay_vlm=True)
    assert not res.vlm_ok and conn.execute("SELECT people_vlm FROM scene_state").fetchone()[0] is None
    assert json.loads(conn.execute("SELECT vlm_raw FROM frames").fetchone()[0])["error"]
    assert count(conn, "observations", source="vlm") == 0


def test_unreplayable_stored_answer_fails_loudly_and_leaves_the_frame_untouched(conn, space, root):
    class Unparseable(FakeReader):
        def parse(self, raw, space):
            raise ValueError("people_count missing")
    c = cap(conn, space, root, FakeSource([synthetic_image()]))
    process_frame(conn, space, c.frame_id, FakeDetector(DETS), FakeReader(), root=root)
    with pytest.raises(StorageError, match="cannot be replayed"):
        process_frame(conn, space, c.frame_id, FakeDetector(DETS[:1]), Unparseable(), root=root, replay_vlm=True)
    assert count(conn, "observations", source="yolo") == 4 and count(conn, "observations", source="vlm") == 2
    with pytest.raises(StorageError, match="needs a scene reader"):
        process_frame(conn, space, c.frame_id, FakeDetector(DETS), None, root=root, replay_vlm=True)


def test_lmstudio_reader_parses_a_stored_answer_without_a_network_call(space):
    class NoNetwork:
        def post(self, *a, **k):
            raise AssertionError("parse() must not call the model")
    reader = LmStudioSceneReader(SceneReaderSettings(base_url="http://fake", model="m"), session=NoNetwork())
    raw = json.dumps({"people_count": 1, "people": [], "objects": [{"label": "mug", "count": 2, "status": "present"}],
                      "lighting": "artificial", "summary": "s"})
    r = reader.parse(raw, space)
    assert r.people_count == 1 and r.objects[0].label == "mug" and r.raw == raw


# ---------------------------------------------------------------- geometry and table
def test_union_area_counts_overlaps_once():
    assert union_area([]) == 0.0 and union_area([(0.2, 0.2, 0.2, 0.9)]) == 0.0            # degenerate box
    assert union_area([(0.0, 0.0, 0.5, 0.5)]) == pytest.approx(0.25)
    assert union_area([(0.0, 0.0, 0.5, 0.5), (0.1, 0.1, 0.4, 0.4)]) == pytest.approx(0.25)      # nested duplicate
    assert union_area([(0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75)]) == pytest.approx(0.4375)
    assert union_area([(0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)]) == pytest.approx(0.02)


def build(conn, **kw):
    return build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, conf_levels=(0.05, 0.10), **kw)


def test_table_counts_at_each_threshold_and_old_frames_are_unknown_not_zero(conn, space, root):
    for i, minute in enumerate((0, 15, 30)):
        c = cap(conn, space, root, FakeSource([synthetic_image(i)]), thursday_ist(10, minute))
        process_frame(conn, space, c.frame_id, FakeDetector(DETS + LOW), FakeReader(), root=root)
    old = conn.execute("SELECT frame_id FROM frames ORDER BY ts_utc LIMIT 1").fetchone()[0]
    conn.execute("DELETE FROM detections WHERE frame_id=?", (old,))          # a frame from before migration 004
    conn.execute("UPDATE frames SET detector_min_conf=NULL WHERE frame_id=?", (old,))
    res = build(conn)
    t, cat = res.table, res.catalog.set_index("column")
    assert t["count.yolo.chair"].tolist() == [1, 1, 1] and t["count.yolo.person"].tolist() == [2, 2, 2]
    assert np.isnan(t["count.yolo_c05.chair"].iloc[0]) and t["count.yolo_c05.chair"].tolist()[1:] == [2, 2]
    assert t["count.yolo_c10.chair"].tolist()[1:] == [2, 2]                   # the 0.12 chair
    assert t["count.yolo_c05.person"].tolist()[1:] == [3, 3] and t["count.yolo_c10.person"].tolist()[1:] == [2, 2]
    assert t["area.yolo.person"].tolist() == pytest.approx([0.08] * 3)          # two 0.1 x 0.4 boxes; all history
    assert np.isnan(t["area.yolo_c05.person"].iloc[0]) and t["area.yolo_c05.person"].iloc[1] == pytest.approx(0.12)
    assert cat.loc["count.yolo_c05.chair", "variant"] and not cat.loc["count.yolo.chair", "variant"]
    assert cat.loc["area.yolo.person", "kind"] == "area" and cat.loc["count.yolo_c05.chair", "coverage"] == 0.667
    a_floor_of_point_ten = conn.execute("SELECT frame_id FROM frames ORDER BY ts_utc DESC LIMIT 1").fetchone()[0]
    conn.execute("UPDATE frames SET detector_min_conf=0.10 WHERE frame_id=?", (a_floor_of_point_ten,))
    t2 = build(conn).table
    assert np.isnan(t2["count.yolo_c05.chair"].iloc[2]) and t2["count.yolo_c10.chair"].iloc[2] == 2
    assert "count.yolo_c05.chair" not in build_table(conn, "admin_cabin", tz="Asia/Kolkata").table   # levels are opt-in


def test_threshold_variants_are_not_default_model_inputs_unless_asked_for(conn, space, root):
    for i in range(12):
        low = [det("chair", (0.70, 0.30, 0.80, 0.60), 0.12, False)] * (i % 3)
        c = cap(conn, space, root, FakeSource([synthetic_image(i)]), thursday_ist(9, 0) + timedelta(minutes=15 * i))
        process_frame(conn, space, c.frame_id, FakeDetector(DETS[: 2 + i % 3] + low), FakeReader(), root=root)
    res = build(conn)
    cat = res.catalog.set_index("column")
    assert cat.loc["count.yolo_c05.chair", "model_ready"]
    opts = dict(targets=[], max_covariates=50, min_autocorr=0.3, autocorr_min_slots=1000)
    assert not [c for c in select_covariates(res.catalog, always=[], **opts) if ".yolo_c" in c]
    assert select_covariates(res.catalog, always=["count.yolo_c05.chair"], **opts)[0] == "count.yolo_c05.chair"


# ---------------------------------------------------------------- stability report
def test_stability_ranks_steady_and_responsive_above_jittery_and_flags_blind_columns():
    idx = pd.date_range("2026-11-13 18:00", periods=24, freq="15min")
    night = slice(8, 24)                                                    # 20:00 onwards: nothing really changes
    steady = np.r_[np.linspace(20, 6, 8), np.full(16, 6.0)]                 # empties in the evening, then flat
    jittery = steady + np.r_[np.zeros(8), np.tile([0, 3], 8)]               # same level, wobbles at night
    gappy = steady.copy(); gappy[[10, 11, 15]] = np.nan
    t = pd.DataFrame({"count.yolo.motorcycle": steady, "count.vlm.motorcycle": jittery,
                      "count.yolo_c05.motorcycle": np.full(24, 20.0), "area.yolo.motorcycle": gappy,
                      "cat.lighting": "dark", "slot_status": "ok"}, index=idx)
    rep = stability(t, start=str(idx[night][0])).set_index("column")
    assert list(rep.index) == ["count.yolo.motorcycle", "area.yolo.motorcycle", "count.vlm.motorcycle",
                               "count.yolo_c05.motorcycle"]                 # best first, blind last
    assert rep.loc["count.yolo.motorcycle", "noise"] == 0 and np.isinf(rep.loc["count.yolo.motorcycle", "snr"])
    assert rep.loc["count.vlm.motorcycle", "noise"] == 3.0 and rep.loc["count.vlm.motorcycle", "changed_share"] == 1.0
    assert np.isnan(rep.loc["count.yolo_c05.motorcycle", "snr"])            # never moves at all: blind, not stable
    assert rep.loc["count.yolo.motorcycle", "pairs"] == 15 and rep.loc["area.yolo.motorcycle", "pairs"] == 10
    assert "cat.lighting" not in rep.index and stability(t, start="2030-01-01").empty

"""Outlines (segmentation masks): codec, arithmetic, pipeline storage and failure handling, table columns."""
import numpy as np
import pytest

from sbi.errors import SegmenterError
from sbi.forecast.mask_features import mask_features
from sbi.forecast.table import build_table, is_variant, select_covariates
from sbi.perception.factory import build_segmenter
from sbi.perception.masks import cell_shares, decode_mask, drop_duplicate_masks, encode_mask, mask_iou
from sbi.perception.types import SegMask
from sbi.pipeline import process_frame
from sbi.settings import SegmenterSettings
from sbi.tick import run_tick

from .conftest import FakeDetector, FakeReader, FakeSegmenter, FakeSource, det, synthetic_image, thursday_ist
from .test_detections import LOW
from .test_flow import DETS, cap, count


def rect(x1, y1, x2, y2, shape=(36, 64)) -> np.ndarray:
    g = np.zeros(shape, dtype=bool)
    g[y1:y2, x1:x2] = True
    return g


# ---------------------------------------------------------------- codec and arithmetic
def test_mask_codec_round_trips_and_rejects_bad_input():
    rng = np.random.default_rng(0)
    for g in (rect(3, 4, 20, 30), np.zeros((216, 384), bool), rng.random((7, 13)) > 0.5):
        back = decode_mask(encode_mask(g))
        assert back.shape == g.shape and back.dtype == bool and (back == g).all()
    assert len(encode_mask(rect(100, 50, 180, 120, (216, 384)))) < 200          # mostly-empty grids stay tiny
    with pytest.raises(ValueError, match="2-D boolean"):
        encode_mask(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="corrupt"):
        decode_mask(b"\x04\x00\x04\x00not-zlib")


def test_duplicate_outlines_keep_the_stronger_box_only():
    a, shifted, other, empty = rect(0, 0, 20, 20), rect(2, 0, 22, 20), rect(40, 0, 60, 20), rect(0, 0, 0, 0)
    assert mask_iou(a, a) == 1.0 and mask_iou(a, other) == 0.0 and mask_iou(empty, empty) == 0.0
    assert mask_iou(a, shifted) == pytest.approx(18 / 22)
    items = [(0.3, shifted), (0.9, a), (0.2, other), (0.1, empty), (0.08, empty)]
    assert drop_duplicate_masks(items, 0.5) == [1, 2, 3, 4]       # the 0.3 copy of `a` goes; empties are kept
    assert drop_duplicate_masks(items, 0.9) == [0, 1, 2, 3, 4]    # stricter rule: not the same object
    nested = [(0.9, rect(0, 0, 40, 30)), (0.5, rect(5, 5, 15, 15))]
    assert drop_duplicate_masks(nested, 0.5) == [0, 1]            # one bike inside a cluster outline is NOT a copy


def test_cell_shares_match_the_grid():
    s = cell_shares(rect(0, 0, 32, 18), cols=8, rows=6)           # left half, top half of a 64x36 grid
    assert s.shape == (6, 8) and s[:3, :4].min() == 1.0 and s[3:, :].max() == 0.0 and s[:, 4:].max() == 0.0
    assert cell_shares(rect(0, 0, 4, 6), cols=8, rows=6)[0, 0] == pytest.approx(0.5)
    assert cell_shares(np.ones((5, 7), bool), cols=3, rows=2).min() == 1.0           # sizes that do not divide


def test_mask_features_per_source_dedupe_union_and_cells():
    bike, copy, weak = rect(0, 0, 16, 12), rect(1, 0, 17, 12), rect(32, 18, 48, 30)
    boxes = [("motorcycle", 0.9, True, bike), ("motorcycle", 0.2, False, copy), ("motorcycle", 0.07, False, weak),
             ("car", 0.8, True, rect(0, 24, 8, 36))]
    f = mask_features(boxes, sources={"sam2": None, "sam2_c10": 0.10, "sam2_c05": 0.05}, min_iou=0.5,
                      grid=(8, 6), cell_min=0.25, cell_sources=("sam2", "sam2_c10", "sam2_c05"))
    assert (f["count.sam2.motorcycle"], f["count.sam2_c10.motorcycle"], f["count.sam2_c05.motorcycle"]) == (1, 1, 2)
    assert f["area.sam2.motorcycle"] == pytest.approx(16 * 12 / (64 * 36), abs=1e-5)
    assert f["area.sam2_c10.motorcycle"] == pytest.approx(17 * 12 / (64 * 36), abs=1e-5)       # union, not a sum
    assert f["cellarea.sam2.R0C0.motorcycle"] == 1.0 and f["cellarea.sam2.R5C0.car"] == 1.0
    assert "cellarea.sam2.R0C2.motorcycle" not in f                # nothing there
    assert "cellarea.sam2_c10.R0C2.motorcycle" not in f            # 1/8 of the cell: below cell_min, left out
    assert f["cellarea.sam2_c05.R3C4.motorcycle"] == 1.0 and "count.sam2.bus" not in f
    assert mask_features([], sources={"sam2": None}, min_iou=0.5) == {}
    default = mask_features(boxes, sources={"sam2": None, "sam2_c05": 0.05}, min_iou=0.5, grid=(8, 6))
    assert not [k for k in default if k.startswith("cellarea.sam2_c05.")] and "cellarea.sam2.R0C0.motorcycle" in default


# ---------------------------------------------------------------- pipeline
def run(conn, space, root, segmenter, dets=DETS + LOW, now=None):
    c = cap(conn, space, root, FakeSource([synthetic_image()]), now or thursday_ist())
    return c.frame_id, process_frame(conn, space, c.frame_id, FakeDetector(dets), FakeReader(), root=root,
                                     segmenter=segmenter)


def test_every_stored_box_gets_an_outline_and_counts_are_unaffected(conn, space, root):
    fid, res = run(conn, space, root, FakeSegmenter())
    assert res.masks_ok and (res.boxes, res.boxes_stored, res.people_yolo) == (4, 6, 2)
    rows = conn.execute("SELECT raw_label, confidence, mask, mask_area, bbox_norm FROM detections ORDER BY det_id").fetchall()
    assert len(rows) == 6 and all(r["mask"] is not None for r in rows)            # uncounted boxes included
    assert rows[0]["mask_area"] == pytest.approx(0.04) and decode_mask(rows[0]["mask"]).shape == (36, 64)
    assert decode_mask(rows[3]["mask"]).sum() == decode_mask(encode_mask(rect(26, 11, 35, 32))).sum()   # the chair's own box
    assert conn.execute("SELECT segmenter_version FROM frames").fetchone()[0] == "fake-seg@64"
    assert count(conn, "observations", source="yolo") == 4


def test_segmenter_failure_keeps_the_frame_with_unknown_outlines(conn, space, root):
    for seg in (FakeSegmenter(error=SegmenterError("cuda oom")), FakeSegmenter(masks=[SegMask(0.1, rect(0, 0, 2, 2))])):
        fid, res = run(conn, space, root, seg, now=thursday_ist(11, 0 if seg.error else 15))
        assert res.processed and not res.masks_ok and res.boxes == 4          # wrong mask count = failure too
        f = conn.execute("SELECT segmenter_version, processed_utc FROM frames WHERE frame_id=?", (fid,)).fetchone()
        assert f["segmenter_version"] is None and f["processed_utc"]
        assert count(conn, "detections", frame_id=fid) == 6
        assert conn.execute("SELECT COUNT(mask) FROM detections WHERE frame_id=?", (fid,)).fetchone()[0] == 0


def test_no_boxes_with_a_working_segmenter_is_known_empty_not_unknown(conn, space, root):
    fid, res = run(conn, space, root, FakeSegmenter(), dets=[])
    assert res.masks_ok and conn.execute("SELECT segmenter_version FROM frames").fetchone()[0] == "fake-seg@64"
    fid2, res2 = run(conn, space, root, None, now=thursday_ist(10, 15))
    assert not res2.masks_ok and conn.execute("SELECT segmenter_version FROM frames WHERE frame_id=?", (fid2,)).fetchone()[0] is None


def test_segmenter_factory_and_missing_weights(tmp_path):
    assert build_segmenter(SegmenterSettings(), tmp_path) is None                     # default: switched off
    with pytest.raises(SegmenterError, match="weights not found"):
        build_segmenter(SegmenterSettings(kind="sam2", weights="models/nope.pt"), tmp_path)


def test_tick_runs_without_outlines_when_the_segmenter_cannot_load(settings, space_dir):
    broken = settings.model_copy(update={"segmenter": SegmenterSettings(kind="sam2", weights="models/nope.pt")})
    out = run_tick(broken, sorted(space_dir.glob("*.yaml")), source=FakeSource([synthetic_image()]),
                   detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist())
    assert all(o.ok and o.processed == 1 for o in out)
    out = run_tick(settings, sorted(space_dir.glob("*.yaml")), source=FakeSource([synthetic_image(1)]),
                   detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist(10, 15), segmenter=FakeSegmenter())
    assert all(o.ok and o.processed == 1 for o in out)
    import sqlite3
    c = sqlite3.connect(settings.db_path)
    assert c.execute("SELECT COUNT(*), COUNT(segmenter_version) FROM frames").fetchone() == (4, 2)


# ---------------------------------------------------------------- table
def test_table_outline_columns_and_frames_without_outlines_are_unknown(conn, space, root):
    twice = DETS + [det("chair", (0.41, 0.31, 0.55, 0.90), conf=0.12, counted=False)] + LOW   # chair boxed twice
    run(conn, space, root, None, dets=twice, now=thursday_ist(10, 0))                # before a segmenter existed
    run(conn, space, root, FakeSegmenter(), dets=twice, now=thursday_ist(10, 15))
    run(conn, space, root, FakeSegmenter(error=SegmenterError("down")), dets=twice, now=thursday_ist(10, 30))
    run(conn, space, root, FakeSegmenter(), dets=twice, now=thursday_ist(10, 45))
    res = build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, conf_levels=(0.05, 0.10),
                      grid=(space.grid.cols, space.grid.rows), cell_area_min=0.02)
    t, cat = res.table, res.catalog.set_index("column")
    assert t["count.yolo.chair"].tolist() == [1, 1, 1, 1] and t["count.yolo_c05.chair"].tolist() == [3, 3, 3, 3]
    for col, value in (("count.sam2.chair", 1), ("count.sam2_c05.chair", 2), ("count.sam2_c10.chair", 2),
                       ("count.sam2_c05.person", 3)):                                # 3 chair boxes, 2 real chairs
        assert np.isnan(t[col].iloc[0]) and np.isnan(t[col].iloc[2]) and t[col].iloc[1] == t[col].iloc[3] == value, col
    assert t["area.sam2.person"].iloc[1] == pytest.approx(2 * 7 * 15 / (64 * 36), abs=1e-4)   # two 7x15-cell person outlines
    assert t["area.sam2.person"].iloc[1] <= t["area.sam2_c05.person"].iloc[1]
    cells = [c for c in t.columns if c.startswith("cellarea.sam2.") and c.endswith(".chair")]
    assert cells and np.isnan(t[cells[0]].iloc[0]) and 0 < t[cells[0]].iloc[1] <= 1
    assert cat.loc["area.sam2.person", "kind"] == "area" and cat.loc[cells[0], "kind"] == "cellarea"
    assert cat.loc["count.sam2.chair", "variant"] and cat.loc["count.sam2_c05.chair", "variant"]
    assert not cat.loc["area.sam2.person", "variant"] and cat.loc["area.sam2_c05.person", "variant"]
    assert is_variant("cellarea", "sam2_c10") and not is_variant("cellarea", "sam2")
    assert not [c for c in t.columns if c.startswith("cellarea.sam2_c")]              # per-cell: counted boxes only
    sel = select_covariates(res.catalog, targets=[], max_covariates=500, always=[], min_autocorr=0.3, autocorr_min_slots=1000)
    assert not [c for c in sel if "_c05" in c or "_c10" in c or c.startswith("count.sam2.")]

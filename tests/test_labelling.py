"""Labelling v2: an even spread of photos over day types, zoom quarters, and choosing thresholds from the labels."""
from datetime import timedelta

import numpy as np
import pytest

from sbi.labeling import core as labeling
from sbi.eval import thresholds, vlm_eval
from sbi.labeling.core import STRATA, quadrants, stratum
from sbi.pipeline import process_frame

from .conftest import FakeDetector, FakeReader, FakeSegmenter, FakeSource, det, synthetic_image, thursday_ist
from .test_flow import cap


def frame_at(conn, space, root, when, dets=(), segmenter=None, detector_cls=FakeDetector):
    fid = cap(conn, space, root, FakeSource([synthetic_image()]), when).frame_id
    process_frame(conn, space, fid, detector_cls(dets), FakeReader(), root=root, segmenter=segmenter)
    return fid


# ---------------------------------------------------------------- which photo next
def test_photos_come_from_the_least_labelled_day_type_first(conn, parking, root):
    sat = thursday_ist(10) + timedelta(days=2)
    groups = {
        "weekday morning": [thursday_ist(9, 15 * i) for i in range(4)],
        "weekday night": [thursday_ist(2, 15 * i) for i in range(4)],
        "weekend afternoon": [sat + timedelta(hours=4, minutes=15 * i) for i in range(2)],
    }
    by_fid = {frame_at(conn, parking, root, t): name for name, ts in groups.items() for t in ts}
    served = []
    for _ in range(6):
        fid = labeling.next_unlabeled(conn, "parking")
        served.append(by_fid[fid])
        labeling.save_label(conn, fid, {"person": 0}, usable=True, labeler="t")
    assert set(served[:3]) == set(groups)                          # one from each day type before any repeats
    assert served[3:6].count("weekend afternoon") == 1             # then round again, until that group runs out
    for _ in range(4):
        labeling.save_label(conn, labeling.next_unlabeled(conn, "parking"), {"person": 0}, usable=True, labeler="t")
    assert labeling.next_unlabeled(conn, "parking") is None
    assert len(STRATA) == 8 and stratum(5, "night") == (1, "night") and stratum(4, "night") == (0, "night")


def test_the_order_is_stable_between_sessions(conn, parking, root):
    for i in range(6):
        frame_at(conn, parking, root, thursday_ist(10, 0) + timedelta(minutes=15 * i))
    assert labeling.next_unlabeled(conn, "parking") == labeling.next_unlabeled(conn, "parking")


def test_quadrants_cover_the_photo_once_at_full_resolution():
    img = np.arange(9 * 16 * 3, dtype=np.uint32).reshape(9, 16, 3)
    parts = quadrants(img)
    assert [n for n, _ in parts] == ["Top left", "Top right", "Bottom left", "Bottom right"]
    assert sum(p.shape[0] * p.shape[1] for _, p in parts) == 9 * 16                  # nothing missing, nothing twice
    assert (parts[0][1] == img[:4, :8]).all() and (parts[3][1] == img[4:, 8:]).all()


# ---------------------------------------------------------------- thresholds from labels
BIKES = [det("motorcycle", (0.05 * k, 0.5, 0.05 * k + 0.04, 0.7), conf=c, counted=c >= 0.25)
         for k, c in enumerate([0.9, 0.6, 0.3, 0.12, 0.07])]
TWICE = [det("motorcycle", (0.1, 0.5, 0.2, 0.7), conf=0.9), det("motorcycle", (0.1, 0.5, 0.2, 0.7), conf=0.4)]


def labelled(conn, space, root, dets, truth, n=1, segmenter=None, start=thursday_ist(10)):
    for i in range(n):
        fid = frame_at(conn, space, root, start + timedelta(minutes=15 * i), dets, segmenter=segmenter)
        labeling.save_label(conn, fid, truth, usable=True, labeler="t")


def test_counts_are_recomputed_at_every_threshold_from_stored_boxes(conn, parking, root):
    labelled(conn, parking, root, BIKES, {"motorcycle": 4, "person": 0})
    pts = thresholds.sweep(conn).set_index(["object_type", "rule", "level"])
    got = {lv: int(pts.loc[("motorcycle", "yolo", lv), "pred"]) for lv in (0.05, 0.10, 0.25, 0.30, 0.50)}
    assert got == {0.05: 5, 0.10: 4, 0.25: 3, 0.30: 3, 0.50: 2}
    assert int(pts.loc[("person", "yolo", 0.05), "pred"]) == 0 and ("motorcycle", "outline", 0.05) not in pts.index
    s = thresholds.score(thresholds.sweep(conn)).set_index(["object_type", "rule", "level"])
    assert s.loc[("motorcycle", "yolo", 0.10), "mae"] == 0 and s.loc[("motorcycle", "yolo", 0.25), "bias"] == -1


def test_outline_rule_removes_a_bike_boxed_twice(conn, parking, root):
    labelled(conn, parking, root, TWICE, {"motorcycle": 1}, segmenter=FakeSegmenter())
    pts = thresholds.sweep(conn).set_index(["object_type", "rule", "level"])
    assert pts.loc[("motorcycle", "yolo", 0.25), "pred"] == 2 and pts.loc[("motorcycle", "outline", 0.25), "pred"] == 1
    assert pts.loc[("motorcycle", "outline", 0.45), "pred"] == 1                    # the weaker copy is gone anyway


def test_frames_read_only_down_to_a_higher_floor_are_not_scored_below_it(conn, parking, root):
    class OldDetector(FakeDetector):
        min_conf = 0.25
    fid = frame_at(conn, parking, root, thursday_ist(10), BIKES[:3], detector_cls=OldDetector)
    labeling.save_label(conn, fid, {"motorcycle": 3}, usable=True, labeler="t")
    levels = sorted(thresholds.sweep(conn)["level"].unique())
    assert min(levels) == 0.25                                                      # 0.05-0.20 unknown, not zero


def test_unusable_photos_are_ignored(conn, parking, root):
    fid = frame_at(conn, parking, root, thursday_ist(10), BIKES)
    labeling.save_label(conn, fid, {}, usable=False, labeler="t")
    assert thresholds.sweep(conn).empty and thresholds.report(conn) == ""
    conn.execute("INSERT INTO labels(frame_id, object_type, true_count) VALUES(?, 'motorcycle', 3)", (fid,))  # stray row
    assert thresholds.sweep(conn).empty                                            # still ignored: frame is unusable


def test_recommendation_is_the_strictest_option_close_to_the_best_and_needs_enough_labels(conn, parking, root):
    labelled(conn, parking, root, BIKES, {"motorcycle": 4, "person": 0}, n=thresholds.MIN_LABELS)
    rec = thresholds.recommend(thresholds.score(thresholds.sweep(conn))).set_index("object_type")
    bike = rec.loc["motorcycle"]
    assert bike["best"] == "yolo@0.10" and bike["recommended"] == "yolo@0.10" and bike["current_mae"] == 1.0
    assert rec.loc["person", "recommended"] == "yolo@0.25" and "keep" in rec.loc["person", "why"]   # a tie is no evidence
    assert thresholds.class_conf_suggestion(rec.reset_index()) == {"motorcycle": 0.10}
    few = thresholds.recommend(thresholds.score(thresholds.sweep(conn).query("frame_id < 3")))
    assert (few["recommended"] == "-").all() and few["why"].str.contains("only").all()


def test_suggestions_skip_outline_picks_the_current_threshold_and_disagreeing_spaces():
    import pandas as pd
    rec = pd.DataFrame({"space_id": ["a", "b", "a", "a", "a"], "object_type": ["person", "person", "chair", "bag", "car"],
                        "recommended": ["yolo@0.10", "yolo@0.30", "yolo@0.25", "outline@0.10", "-"]})
    assert thresholds.class_conf_suggestion(rec) == {}


def test_eval_report_includes_the_threshold_section(conn, parking, root):
    labelled(conn, parking, root, BIKES, {"motorcycle": 4, "person": 0}, n=thresholds.MIN_LABELS)
    text = vlm_eval.report(conn)
    assert "detector threshold per object type" in text and "class_conf: {motorcycle: 0.10}" in text

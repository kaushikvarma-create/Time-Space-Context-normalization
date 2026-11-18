"""Steady counts of things that do not move: the pooled median, its table columns, and the space setting."""
from datetime import timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from sbi.dashboard import data
from sbi.forecast.steady_features import pooled_median
from sbi.forecast.table import build_table
from sbi.pipeline import process_frame
from sbi.perception.types import Detection
from sbi.spaces import SpaceConfig

from .conftest import FakeDetector, FakeReader, FakeSource, synthetic_image, thursday_ist
from .test_flow import cap

NAN = float("nan")


def test_pooled_median_is_the_middle_of_every_counters_readings_in_the_window():
    r = np.array([[7, 6], [9, 7], [8, 6], [7, 7]], float)                 # two counters, four slots
    assert pooled_median(r, window=2, min_values=1).tolist() == [6.0, 7.0, 7.0, 7.0]
    #   slot 0: {7,6} -> lower middle 6 | slot 1: {7,6,9,7} -> 7 | slot 2: {9,7,8,6} -> 7 | slot 3: {8,6,7,7} -> 7
    assert pooled_median(r, window=1, min_values=1).tolist() == [6.0, 7.0, 6.0, 7.0]    # one slot: the lower of two
    assert pooled_median(r, window=10, min_values=1)[-1] == 7.0                          # all eight readings


def test_pooled_median_looks_only_backwards():
    r = np.array([[7.0]] * 6 + [[20.0]] * 6)
    out = pooled_median(r, window=4, min_values=1)
    assert out[:6].tolist() == [7.0] * 6                                   # the jump ahead is not seen
    assert out[6:].tolist() == [7.0, 7.0, 20.0, 20.0, 20.0, 20.0]           # it shows once it is over half the window


def test_pooled_median_needs_enough_readings_and_ignores_missing_ones():
    r = np.array([[NAN, NAN], [7, NAN], [NAN, NAN], [9, 7], [NAN, 7]], float)
    out = pooled_median(r, window=3, min_values=2)
    assert np.isnan(out[:3]).all() and out[3] == 7.0 and out[4] == 7.0     # {7,9,7} then {9,7,7}
    assert np.isnan(pooled_median(r, window=3, min_values=4)).all()
    assert pooled_median(np.empty((0, 2)), 3, 1).tolist() == [] and np.isnan(pooled_median(np.empty((3, 0)), 3, 1)).all()
    for bad in ((0, 1), (3, 0)):
        with pytest.raises(ValueError):
            pooled_median(r, *bad)
    with pytest.raises(ValueError, match="2-D"):
        pooled_median(np.array([1.0, 2.0]), 3, 1)


def test_a_noisy_counter_and_a_biased_one_still_give_the_true_count():
    rng = np.random.default_rng(0)
    detector = 7 + rng.choice([-2, -1, 0, 1, 1, 2, 3], 400)                # jumps around, leans high
    vlm = 7 + rng.choice([-1, -1, 0, 0, 0], 400)                           # steadier, leans low
    out = pooled_median(np.column_stack([detector, vlm]).astype(float), window=96, min_values=24)
    assert (out[12:] == 7).all() and np.isnan(out[:11]).all()             # exact from the first slot it exists


def chairs(n):
    return [Detection("chair", 0.9, (0.1 + 0.05 * i, 0.1, 0.14 + 0.05 * i, 0.2)) for i in range(n)]


def test_the_table_gets_a_steady_count_for_listed_types_only(conn, space, root):
    seen = [7, 9, 7, 4, 7, 8, 7, 7]                                        # the detector's chair count, slot by slot
    for i, n in enumerate(seen):
        fid = cap(conn, space, root, FakeSource([synthetic_image(i)]), thursday_ist(9, 0) + timedelta(minutes=15 * i)).frame_id
        process_frame(conn, space, fid, FakeDetector(chairs(n)), FakeReader(), root=root)
    res = build_table(conn, "admin_cabin", tz=space.tz, steady_labels=("chair", "sofa"), steady_window=4, steady_min_values=3)
    t = res.table
    assert t["count.yolo.chair"].tolist() == [float(n) for n in seen]                     # the raw count is untouched
    vlm = t["count.vlm.chair"].tolist() if "count.vlm.chair" in t else None
    pooled = np.column_stack([seen] + ([vlm] if vlm else [])).astype(float)
    assert t["steady.all.chair"].equals(type(t["steady.all.chair"])(pooled_median(pooled, 4, 3), index=t.index))
    assert "steady.all.sofa" not in t.columns                              # never counted by anything: no column
    cat = res.catalog.set_index("column")
    assert cat.loc["steady.all.chair", "kind"] == "steady" and not cat.loc["steady.all.chair", "variant"]
    plain = build_table(conn, "admin_cabin", tz=space.tz).table
    assert not [c for c in plain.columns if c.startswith("steady.")]       # nothing listed, nothing added
    assert data.label("steady.all.chair") == "chair count (steady, 24 h)"
    latest = int(t["steady.all.chair"].dropna().iloc[-1])
    assert data.fixtures(t) == [("chair", latest)] and data.fixtures(plain) == []


def test_steady_types_default_to_fixed_vocabulary_and_are_validated(space, parking):
    assert "chair" in space.steady_labels and "person" not in space.steady_labels and "laptop" not in space.steady_labels
    assert parking.steady_labels == ["barrier", "pole", "gate"]
    default = SpaceConfig.model_validate({**space.model_dump(), "steady_types": None})
    assert "monitor" in default.steady_labels and "chair" not in default.steady_labels    # chair is semi_static
    assert SpaceConfig.model_validate({**space.model_dump(), "steady_types": []}).steady_labels == []
    with pytest.raises(ValidationError, match="not in vocabulary"):
        SpaceConfig.model_validate({**space.model_dump(), "steady_types": ["throne"]})
    with pytest.raises(ValidationError, match="repeats"):
        SpaceConfig.model_validate({**space.model_dump(), "steady_types": ["chair", "chair"]})

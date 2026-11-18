"""SAM 2 wrapper logic with a fake model (no weights, no GPU): one outline per box, also when the model drops one."""
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from sbi.errors import SegmenterError

torch = pytest.importorskip("torch")
from sbi.perception.segmenter_sam2 import Sam2Segmenter  # noqa: E402


class FakeSam:
    """Returns a full-frame mask per box, except boxes whose left edge is in `empty` (pixels): those are dropped,
    as the real model does for a box it finds nothing in."""
    def __init__(self, empty=(), fail=False):
        self.empty, self.fail, self.calls = set(empty), fail, []

    def __call__(self, image, bboxes, verbose, device):
        self.calls.append(len(bboxes))
        if self.fail:
            raise RuntimeError("cuda out of memory")
        kept = [b for b in bboxes if round(b[0]) not in self.empty]
        masks = None
        if kept:
            data = torch.zeros((len(kept), 20, 40))
            for i, b in enumerate(kept):
                data[i, :, : int(round(b[0])) + 1] = 1.0                 # wider mask for boxes further right
            masks = SimpleNamespace(data=data)
        return [SimpleNamespace(masks=masks)]


def segmenter(model, batch=64):
    s = object.__new__(Sam2Segmenter)                                    # skip weight loading
    s.cfg = SimpleNamespace(mask_width=20, batch=batch)
    s._torch, s._model, s._device, s.version = torch, model, "cpu", "fake@20"
    return s


IMG = np.zeros((20, 40, 3), np.uint8)
BOXES = [(0.0, 0.0, 0.5, 0.5), (0.25, 0.0, 0.75, 0.5), (0.5, 0.0, 1.0, 0.5)]       # left edges at pixel 0, 10, 20


def test_one_outline_per_box_in_order_at_the_stored_size():
    model = FakeSam()
    out = segmenter(model).segment(IMG, BOXES)
    assert len(out) == 3 and all(m.grid.shape == (10, 20) and m.grid.dtype == bool for m in out)
    assert [round(m.area, 3) for m in out] == [0.025, 0.275, 0.525] and model.calls == [3]
    assert segmenter(FakeSam()).segment(IMG, []) == []
    batched = FakeSam(); assert len(segmenter(batched, batch=2).segment(IMG, BOXES)) == 3 and batched.calls == [2, 1]


def test_a_box_with_nothing_in_it_gets_an_empty_outline_and_the_others_keep_theirs(caplog):
    model = FakeSam(empty={10})
    with caplog.at_level(logging.INFO):
        out = segmenter(model).segment(IMG, BOXES)
    assert [round(m.area, 3) for m in out] == [0.025, 0.0, 0.525]        # still three, in box order
    assert not out[1].grid.any() and out[0].grid.any() and out[2].grid.any()
    assert model.calls == [3, 1, 1, 1] and "1 of 3 boxes had nothing to outline" in caplog.text


def test_nothing_at_all_or_a_model_failure_is_a_segmenter_error():
    with pytest.raises(SegmenterError, match="no mask for any of 3 boxes"):
        segmenter(FakeSam(empty={0, 10, 20})).segment(IMG, BOXES)
    with pytest.raises(SegmenterError, match="inference failed: cuda out of memory"):
        segmenter(FakeSam(fail=True)).segment(IMG, BOXES)

"""Pure box-cleanup functions (no model, no IO), so they are easy to test and tune."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..grid import BBox
from .masks import drop_duplicate_masks
from .types import Detection


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: BBox, b: BBox) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: BBox, b: BBox) -> float:
    i = _inter(a, b)
    return i / max(_area(a) + _area(b) - i, 1e-9)


def dedupe_same_label_iou(dets: list[Detection], thr: float) -> list[Detection]:
    """After class mapping, different source classes can collapse to one label (handbag+backpack -> bag) and
    box the same object twice. Keep the stronger box when same-label IoU exceeds thr."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        if not any(k.label == d.label and iou(k.bbox, d.bbox) > thr for k in kept):
            kept.append(d)
    return kept


def dedupe_contained(dets: list[Detection], labels: set[str], thr: float) -> list[Detection]:
    """For the given labels, drop a weaker box that lies mostly inside a stronger one (e.g. a second box on
    a seated person's torso). Trade-off: a person almost fully hidden behind another may be dropped too."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        if d.label in labels:
            small = _area(d.bbox)
            if any(k.label == d.label and
                   _inter(k.bbox, d.bbox) / max(min(small, _area(k.bbox)), 1e-9) > thr for k in kept):
                continue
        kept.append(d)
    return kept


def uncount_outline_duplicates(dets: list[Detection], outlines: list[np.ndarray], min_iou: float) -> list[Detection]:
    """Two counted boxes of one object type whose OUTLINES nearly coincide are one object boxed twice: the weaker
    one stops being counted (it stays stored as a raw detection, with its outline). Boxes overlap a lot between
    real neighbours (chairs in a row, parked motorcycles); their outlines do not, so this removes doubles that the
    box rules cannot, and lets a low counting threshold be used without counting objects twice. `outlines[i]`
    belongs to `dets[i]`. Order is kept."""
    if len(outlines) != len(dets):
        raise ValueError(f"{len(outlines)} outlines for {len(dets)} detections")
    by_label: dict[str, list[int]] = {}
    for i, d in enumerate(dets):
        if d.counted:
            by_label.setdefault(d.label, []).append(i)
    out = list(dets)
    for idx in by_label.values():
        kept = set(drop_duplicate_masks([(dets[i].conf, outlines[i]) for i in idx], min_iou))
        for pos, i in enumerate(idx):
            if pos not in kept:
                out[i] = replace(dets[i], counted=False)
    return out

"""Deterministic links computed from detector boxes (no model judgement, fully reproducible).

(a, overlaps, b): two boxes with different labels whose intersection covers at least `min_ios` of the smaller
box (e.g. a person box overlapping a chair box: someone sitting at or standing by it). Pairs are counted and
ordered alphabetically so (chair, person) and (person, chair) are the same link.
"""
from __future__ import annotations

from collections import Counter

from .dedupe import _area, _inter
from .types import Detection


def overlap_links(dets: list[Detection], min_ios: float = 0.3) -> list[tuple[str, str, str, int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for i, a in enumerate(dets):
        for b in dets[i + 1:]:
            if a.label == b.label:
                continue
            smaller = min(_area(a.bbox), _area(b.bbox))
            if smaller > 0 and _inter(a.bbox, b.bbox) / smaller >= min_ios:
                counts[tuple(sorted((a.label, b.label)))] += 1
    return [(s, "overlaps", o, n) for (s, o), n in sorted(counts.items())]


def union_area(boxes: list) -> float:
    """Exact area covered by a set of normalised boxes (overlaps counted once): the share of the image that is
    covered. A 'how full' measure that does not depend on separating packed or doubly-boxed objects."""
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return 0.0
    xs = sorted({v for b in boxes for v in (b[0], b[2])})
    ys = sorted({v for b in boxes for v in (b[1], b[3])})
    total = 0.0
    for i in range(len(xs) - 1):
        cx = (xs[i] + xs[i + 1]) / 2
        for j in range(len(ys) - 1):
            cy = (ys[j] + ys[j + 1]) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
                total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return total

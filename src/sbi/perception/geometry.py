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

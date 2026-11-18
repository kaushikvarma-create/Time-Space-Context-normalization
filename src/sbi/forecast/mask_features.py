"""Numbers derived from stored outlines for ONE frame (pure; the table builder calls this per frame).

For each source (the counted boxes, and all boxes at or above each confidence level) and each object type:
  count.<src>.<type>            boxes left after removing same-type boxes that outline the same object
  area.<src>.<type>             share of the image covered by the outlines (overlaps counted once)
  cellarea.<src>.<cell>.<type>  share of a grid cell covered (cells below `cell_min` are left out); only for
                                the sources in `cell_sources`, because per-cell columns multiply fast. The
                                outlines are stored, so other sources can be derived later without any model.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..grid import cell_id
from ..perception.masks import cell_shares, drop_duplicate_masks, mask_extent

MaskedBox = tuple[str, float, bool, np.ndarray]      # (object type, confidence, counted, outline)


def mask_features(boxes: list[MaskedBox], *, sources: dict[str, float | None], min_iou: float,
                  grid: tuple[int, int] | None = None, cell_min: float = 0.0,
                  cell_sources: tuple[str, ...] = ("sam2",)) -> dict[str, float]:
    """`sources` maps a source token to its confidence level; None means 'the counted boxes'."""
    out: dict[str, float] = {}
    extents = [mask_extent(mask) for _, _, _, mask in boxes]         # measured once, reused by every source
    for src, level in sources.items():
        by_type: dict[str, list[int]] = defaultdict(list)
        for i, (name, conf, counted, _) in enumerate(boxes):
            if counted if level is None else conf >= level:
                by_type[name].append(i)
        for name, idx in by_type.items():
            items = [(boxes[i][1], boxes[i][3]) for i in idx]
            out[f"count.{src}.{name}"] = float(len(drop_duplicate_masks(items, min_iou, [extents[i] for i in idx])))
            union = np.logical_or.reduce([m for _, m in items])
            out[f"area.{src}.{name}"] = round(float(union.mean()), 5)
            if grid is not None and src in cell_sources:
                shares = cell_shares(union, *grid)
                for r, c in zip(*np.nonzero(shares >= max(cell_min, 1e-9))):
                    out[f"cellarea.{src}.{cell_id(int(c), int(r))}.{name}"] = round(float(shares[r, c]), 4)
    return out

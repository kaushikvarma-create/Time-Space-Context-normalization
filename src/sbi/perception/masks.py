"""Mask codec and mask arithmetic (pure numpy, no model, no IO).

A stored mask is a small boolean grid (the segmenter picks the size, e.g. 384 wide) packed to bits and
zlib-compressed, with its height and width in a 4-byte header. Mostly-empty grids compress to a few hundred bytes.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

_HEADER = struct.Struct("<HH")       # height, width


def encode_mask(grid: np.ndarray) -> bytes:
    if grid.ndim != 2 or grid.dtype != bool:
        raise ValueError("mask must be a 2-D boolean array")
    return _HEADER.pack(*grid.shape) + zlib.compress(np.packbits(grid).tobytes(), 6)


def decode_mask(blob: bytes) -> np.ndarray:
    try:
        h, w = _HEADER.unpack_from(blob)
        bits = np.unpackbits(np.frombuffer(zlib.decompress(blob[_HEADER.size:]), dtype=np.uint8), count=h * w)
    except (struct.error, zlib.error, ValueError) as e:
        raise ValueError(f"corrupt mask blob: {e}") from e
    return bits.reshape(h, w).astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / union if union else 0.0


def mask_extent(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """(pixels set, first row, row end, first column, column end) of a mask; all zero when it is empty."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return 0, 0, 0, 0, 0
    return int(mask.sum()), int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _iou_at_least(a: np.ndarray, ea: tuple, b: np.ndarray, eb: tuple, min_iou: float) -> bool:
    """mask_iou(a, b) >= min_iou, without touching pixels when the answer is already decided: outlines whose
    bounding boxes do not meet share nothing, and IoU can never exceed smaller area / larger area. Otherwise the
    overlap is counted only inside the boxes' intersection. Same answer as mask_iou, about 20x less work."""
    na, nb = ea[0], eb[0]
    if not na or not nb:
        return 0.0 >= min_iou
    if min(na, nb) / max(na, nb) < min_iou:
        return False
    r0, r1, c0, c1 = max(ea[1], eb[1]), min(ea[2], eb[2]), max(ea[3], eb[3]), min(ea[4], eb[4])
    if r0 >= r1 or c0 >= c1:
        return 0.0 >= min_iou
    inter = int(np.logical_and(a[r0:r1, c0:c1], b[r0:r1, c0:c1]).sum())
    return inter / (na + nb - inter) >= min_iou


def drop_duplicate_masks(items: list[tuple[float, np.ndarray]], min_iou: float,
                         extents: list[tuple] | None = None) -> list[int]:
    """Indexes to keep. `items` are (confidence, mask) for boxes of ONE object type. Two boxes whose outlines
    are nearly the same shape are the same object boxed twice: the stronger one is kept. Empty outlines
    cannot be compared and are kept. `extents` (mask_extent of each item) can be passed to avoid measuring the
    same outline again when it is compared under several sources."""
    if extents is None:
        extents = [mask_extent(m) for _, m in items]
    kept: list[int] = []
    for i in sorted(range(len(items)), key=lambda k: -items[k][0]):
        m, e = items[i][1], extents[i]
        if e[0] and any(_iou_at_least(m, e, items[k][1], extents[k], min_iou) for k in kept):
            continue
        kept.append(i)
    return sorted(kept)


def cell_shares(mask: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Share of each grid cell covered by the mask, shape (rows, cols). Cell edges match grid.point_to_cell."""
    h, w = mask.shape
    cell = ((np.arange(h) * rows // h)[:, None] * cols + (np.arange(w) * cols // w)[None, :]).ravel()
    size = np.bincount(cell, minlength=rows * cols)
    hit = np.bincount(cell[mask.ravel()], minlength=rows * cols)
    return (hit / np.maximum(size, 1)).reshape(rows, cols)

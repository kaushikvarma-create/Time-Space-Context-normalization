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


def drop_duplicate_masks(items: list[tuple[float, np.ndarray]], min_iou: float) -> list[int]:
    """Indexes to keep. `items` are (confidence, mask) for boxes of ONE object type. Two boxes whose outlines
    are nearly the same shape are the same object boxed twice: the stronger one is kept. Empty outlines
    cannot be compared and are kept."""
    kept: list[int] = []
    for i in sorted(range(len(items)), key=lambda k: -items[k][0]):
        m = items[i][1]
        if m.any() and any(mask_iou(m, items[k][1]) >= min_iou for k in kept):
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

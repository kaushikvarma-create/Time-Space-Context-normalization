"""Geometric grid positioning. The VLM never sees the grid; cells are computed from detector boxes."""
from __future__ import annotations

BBox = tuple[float, float, float, float]
_EDGE = 0.999999  # keeps a coordinate of exactly 1.0 inside the last cell


def cell_id(col: int, row: int) -> str:
    return f"R{row}C{col}"


def point_to_cell(x: float, y: float, cols: int, rows: int) -> tuple[int, int]:
    """Normalised point -> (col, row), 0-based. Out-of-range values are clamped to the grid."""
    return min(max(int(x * cols), 0), cols - 1), min(max(int(y * rows), 0), rows - 1)


def bbox_anchor(bbox: BBox, anchor: str = "bottom_center") -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    if anchor == "bottom_center":
        return (x1 + x2) / 2, y2
    if anchor == "center":
        return (x1 + x2) / 2, (y1 + y2) / 2
    raise ValueError(f"unknown anchor {anchor!r}")


def bbox_to_cell(bbox: BBox, cols: int, rows: int, anchor: str = "bottom_center") -> tuple[int, int, str]:
    x, y = bbox_anchor(bbox, anchor)
    c, r = point_to_cell(min(x, _EDGE), min(y, _EDGE), cols, rows)
    return c, r, cell_id(c, r)


def bbox_to_cells(bbox: BBox, cols: int, rows: int) -> list[str]:
    """All cells a box overlaps (useful for large objects such as cars)."""
    x1, y1, x2, y2 = bbox
    c1, r1 = point_to_cell(min(x1, _EDGE), min(y1, _EDGE), cols, rows)
    c2, r2 = point_to_cell(min(x2, _EDGE), min(y2, _EDGE), cols, rows)
    return [cell_id(c, r) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]

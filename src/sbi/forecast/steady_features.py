"""Steady counts for things that do not move (chairs, monitors, desks): pure numpy, no IO.

A per-photo count of furniture is a noisy reading of a number that hardly ever changes. In the admin cabin the
detector reported 4 to 10 chairs for a room that always holds 7: it mistakes dark shapes for chairs at night and
loses chairs behind people by day, while the VLM leans the other way. Their errors differ, so the middle value of
EVERY counter's readings over the last day is close to the truth, and it moves only when the room really changes.

    steady.all.<type> = median of all counters' readings of <type> over the trailing window (24 hours by default)

It looks only backwards, so a backtest can use it without seeing the future. A real change (a chair removed) shows
up once more than half the window has seen it.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pooled_median(readings: np.ndarray, window: int, min_values: int) -> np.ndarray:
    """`readings` has one row per slot and one column per counter (NaN = no reading). Returns, per slot, the median
    of every reading in that slot and the `window - 1` slots before it; NaN where fewer than `min_values` readings
    exist. The median is always one of the readings (never 6.5 chairs): of two middle values the lower is taken."""
    if window < 1 or min_values < 1:
        raise ValueError("window and min_values must be at least 1")
    a = np.asarray(readings, dtype=float)
    if a.ndim != 2:
        raise ValueError("readings must be 2-D: slots x counters")
    n, k = a.shape
    if n == 0 or k == 0:
        return np.full(n, np.nan)
    padded = np.vstack([np.full((window - 1, k), np.nan), a])
    windows = sliding_window_view(padded, window, axis=0).reshape(n, -1)
    enough = (~np.isnan(windows)).sum(axis=1) >= min_values
    out = np.full(n, np.nan)
    if enough.any():
        out[enough] = np.nanquantile(windows[enough], 0.5, axis=1, method="lower")
    return out

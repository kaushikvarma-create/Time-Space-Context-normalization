"""How steady is each count? A label-free check of measurement noise.

For forecasting, a count that is consistently biased is usable; one that jitters is not. In a window where
the real scene does not change (a car park at night), every slot-to-slot change in a count is measurement
noise. This report compares that noise with how much the same column moves over the whole history:

  noise    mean |change| between consecutive valid slots inside the quiet window
  spread   standard deviation over the whole table (how much the column moves at all)
  snr      spread / noise. High = steady when nothing happens AND responsive when something does.
           A column that never moves (spread 0) is not 'stable', it is blind: snr is NaN and it sorts last.

This does not replace human labels: it measures steadiness, not correctness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KINDS = ("scene", "count", "area", "rollup")


def stability(table: pd.DataFrame, *, start: str | None = None, end: str | None = None,
              kinds: tuple[str, ...] = KINDS) -> pd.DataFrame:
    """One row per numeric column of the given kinds. `start`/`end` (local time) bound the quiet window;
    spread always uses the whole table. Only pairs of consecutive slots that both have a value are compared."""
    window = table.loc[pd.Timestamp(start) if start else None: pd.Timestamp(end) if end else None]
    rows = []
    for col in table.columns:
        parts = col.split(".")
        if parts[0] not in kinds:
            continue
        full = pd.to_numeric(table[col], errors="coerce")
        step = pd.to_numeric(window[col], errors="coerce").diff().abs()   # NaN when either neighbour is missing
        pairs = int(step.notna().sum())
        if pairs == 0:
            continue
        noise, spread = float(step.mean()), float(full.std(ddof=0))
        level = float(pd.to_numeric(window[col], errors="coerce").mean())
        if spread == 0:
            snr = np.nan
        elif noise == 0:
            snr = np.inf
        else:
            snr = spread / noise
        rows.append({"column": col, "kind": parts[0], "source": parts[1] if len(parts) >= 3 else "",
                     "object": parts[-1], "pairs": pairs, "window_mean": round(level, 3), "noise": round(noise, 4),
                     "changed_share": round(float((step.dropna() > 0).mean()), 3), "spread": round(spread, 4),
                     "snr": snr if np.isnan(snr) or np.isinf(snr) else round(snr, 2)})
    out = pd.DataFrame(rows, columns=["column", "kind", "source", "object", "pairs", "window_mean", "noise",
                                      "changed_share", "spread", "snr"])
    return out.sort_values(["object", "snr"], ascending=[True, False], na_position="last").reset_index(drop=True)

"""Time-averaged baselines per (space, weekday, hour).

Rules: only processed frames with capture_status='ok' count; n_frames is always stored; a cell with no
detection in a valid frame counts as 0 (the frame was seen and the cell was empty), while camera_down /
bad_frame / unprocessed frames are excluded entirely (missing is not zero).
"""
from __future__ import annotations

import statistics
import sqlite3
from collections import defaultdict

from .store.db import transaction
from .timeutil import now_utc


def _quantile(sorted_vals: list[float], p: float) -> float:
    i = (len(sorted_vals) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


def summarise(vals: list[float], alpha: float) -> dict[str, float]:
    """mean/std/p50/p90 and an exponentially weighted mean (vals must be in chronological order)."""
    s = sorted(vals)
    ewm = vals[0]
    for v in vals[1:]:
        ewm = alpha * v + (1 - alpha) * ewm
    return {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "p50": _quantile(s, 0.5), "p90": _quantile(s, 0.9), "ewm": ewm}


def rebuild(conn: sqlite3.Connection, space_id: str, *, alpha: float = 0.3, dow: int | None = None,
            hour: int | None = None) -> None:
    """Recompute baselines for a space; pass dow+hour to refresh only one bucket (cheap per tick)."""
    scope, args = "", [space_id]
    if dow is not None and hour is not None:
        scope, args = " AND f.dow=? AND f.hour=?", [space_id, dow, hour]
    frames = conn.execute(
        f"""SELECT f.frame_id,f.dow,f.hour,f.brightness,s.people_yolo,s.people_vlm,s.vehicles_yolo
            FROM frames f JOIN scene_state s ON s.frame_id=f.frame_id
            WHERE f.capture_status='ok' AND f.space_id=?{scope} ORDER BY f.ts_utc""", args).fetchall()
    buckets: dict[tuple[int, int], list[sqlite3.Row]] = defaultdict(list)
    for fr in frames:
        buckets[(fr["dow"], fr["hour"])].append(fr)
    now = now_utc().isoformat(timespec="seconds")

    with transaction(conn):
        if scope:
            conn.execute("DELETE FROM baseline_cell WHERE space_id=? AND dow=? AND hour=?", (space_id, dow, hour))
            conn.execute("DELETE FROM baseline_space WHERE space_id=? AND dow=? AND hour=?", (space_id, dow, hour))
        else:
            conn.execute("DELETE FROM baseline_cell WHERE space_id=?", (space_id,))
            conn.execute("DELETE FROM baseline_space WHERE space_id=?", (space_id,))
        for (b_dow, b_hour), frs in buckets.items():
            ids = [x["frame_id"] for x in frs]
            marks = ",".join("?" * len(ids))
            per: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)   # (type, cell) -> {frame_id: n}
            for r in conn.execute(
                    f"""SELECT frame_id,object_type,cell_id,SUM(count) n FROM observations
                        WHERE source='yolo' AND cell_id IS NOT NULL AND frame_id IN ({marks})
                        GROUP BY frame_id,object_type,cell_id""", ids):
                per[(r["object_type"], r["cell_id"])][r["frame_id"]] = r["n"]
            for (otype, cid), by_frame in per.items():
                vals = [float(by_frame.get(i, 0)) for i in ids]      # zeros for valid frames with nothing there
                st = summarise(vals, alpha)
                conn.execute("INSERT INTO baseline_cell VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (space_id, otype, cid, b_dow, b_hour, len(ids), st["mean"], st["std"], st["p50"],
                              st["p90"], sum(v > 0 for v in vals) / len(vals), st["ewm"], now))
            metrics = {"people_yolo": [x["people_yolo"] for x in frs],
                       "people_vlm": [x["people_vlm"] for x in frs],        # None when the VLM failed: excluded
                       "vehicles_yolo": [x["vehicles_yolo"] for x in frs],
                       "brightness": [x["brightness"] for x in frs]}
            for name, raw in metrics.items():
                vals = [float(v) for v in raw if v is not None]
                if vals:
                    st = summarise(vals, alpha)
                    conn.execute("INSERT INTO baseline_space VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                 (space_id, name, b_dow, b_hour, len(vals), st["mean"], st["std"], st["p50"],
                                  st["p90"], st["ewm"], now))

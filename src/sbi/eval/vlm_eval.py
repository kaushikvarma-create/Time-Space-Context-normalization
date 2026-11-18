"""Compare YOLO and VLM counts with human labels.

Reports per space/type/source: n, bias (mean signed error), MAE, % within +-1, % exact. Bias matters more
than MAE: a detector that always undercounts at night corrupts baselines even if its error looks small.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

_SQL = """
SELECT f.space_id, f.part_of_day, f.frame_id, l.object_type, l.true_count,
       (SELECT COALESCE(SUM(count),0) FROM observations o
         WHERE o.frame_id=f.frame_id AND o.object_type=l.object_type AND o.source='yolo') AS yolo,
       CASE WHEN f.vlm_raw IS NULL OR f.vlm_raw LIKE '{"error"%' THEN NULL ELSE
         (SELECT COALESCE(SUM(count),0) FROM observations o
           WHERE o.frame_id=f.frame_id AND o.object_type=l.object_type AND o.source='vlm') END AS vlm
FROM labels l JOIN frames f ON f.frame_id=l.frame_id
JOIN label_frames lf ON lf.frame_id=l.frame_id AND lf.usable=1
"""


def load_labelled(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(_SQL, conn)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    long = df.melt(id_vars=["space_id", "part_of_day", "frame_id", "object_type", "true_count"],
                   value_vars=["yolo", "vlm"], var_name="source", value_name="pred").dropna(subset=["pred"])
    long["err"] = long["pred"] - long["true_count"]
    out = long.groupby(by + ["source"]).agg(
        n=("err", "size"), bias=("err", "mean"), mae=("err", lambda s: s.abs().mean()),
        within1=("err", lambda s: (s.abs() <= 1).mean() * 100), exact=("err", lambda s: (s == 0).mean() * 100),
        mean_true=("true_count", "mean"))
    return out.round(2).reset_index()


def report(conn: sqlite3.Connection, *, mask_dup_iou: float = 0.5, **threshold_kw) -> str:
    """`threshold_kw` go to eval.thresholds.report (the counting rule and thresholds in use)."""
    from .thresholds import report as threshold_report
    df = load_labelled(conn)
    if df.empty:
        return "no labels yet: run scripts/label.sh"
    pd.set_option("display.width", 200)
    text = (f"{df['frame_id'].nunique()} labelled frames\n\n== by space and object type\n"
            f"{summarise(df, ['space_id', 'object_type']).to_string(index=False)}\n\n"
            f"== people by part of day (look for night/low-light bias)\n"
            f"{summarise(df[df.object_type == 'person'], ['space_id', 'part_of_day']).to_string(index=False)}")
    extra = threshold_report(conn, min_iou=mask_dup_iou, **threshold_kw)
    return text + ("\n\n" + extra if extra else "")

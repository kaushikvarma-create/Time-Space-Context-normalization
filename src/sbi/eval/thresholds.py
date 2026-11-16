"""Choose a detector confidence threshold per object type from human labels.

Every box down to the storage floor is kept with its confidence (and its outline), so a count at ANY threshold can be
recomputed for every labelled photo without re-running a model. This sweeps thresholds for two counting rules:
  yolo@t     boxes of that type with confidence >= t (the counting rule the pipeline uses)
  outline@t  the same boxes after removing same-type boxes whose outlines overlap (count.sam2* in the table)
and scores each against the human counts (bias, MAE, exact share).

Choosing honestly from ~50 labels: the lowest-MAE option is noisy. If the current rule (yolo@0.25) is within a small
margin of the best, it stays: a tie is no evidence for a change. Otherwise the recommendation is the STRICTEST
threshold (fewest false positives) within that margin, with the plain yolo rule winning ties because it is what
`detector.class_conf` controls. Types with too few labels get no recommendation.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from ..perception.masks import decode_mask, drop_duplicate_masks

LEVELS = [round(0.05 * k, 2) for k in range(1, 11)]          # 0.05 .. 0.50
CURRENT = 0.25
MIN_LABELS = 10


def sweep(conn: sqlite3.Connection, *, levels: list[float] = LEVELS, min_iou: float = 0.5) -> pd.DataFrame:
    """One row per (labelled frame, object type, rule, threshold): pred and true counts."""
    labels = pd.read_sql_query(
        "SELECT f.space_id, l.frame_id, l.object_type, l.true_count, f.detector_min_conf AS floor, "
        "f.segmenter_version AS seg FROM labels l JOIN label_frames lf ON lf.frame_id=l.frame_id AND lf.usable=1 "
        "JOIN frames f ON f.frame_id=l.frame_id", conn)
    if labels.empty:
        return pd.DataFrame(columns=["space_id", "frame_id", "object_type", "rule", "level", "pred", "true"])
    ids = sorted(labels["frame_id"].unique().tolist())
    boxes: dict[tuple[int, str], list[tuple[float, bytes | None]]] = {}
    for fid, label, conf, mask in conn.execute(
            f"SELECT frame_id, raw_label, confidence, mask FROM detections WHERE frame_id IN ({','.join('?' * len(ids))})", ids):
        boxes.setdefault((fid, label), []).append((float(conf), mask))
    rows = []
    for r in labels.itertuples():
        found = boxes.get((r.frame_id, r.object_type), [])
        for lv in levels:
            if r.floor is None or r.floor > lv + 1e-9:             # this frame was not read that low: unknown
                continue
            kept = [(c, m) for c, m in found if c >= lv]
            rows.append((r.space_id, r.frame_id, r.object_type, "yolo", lv, len(kept), r.true_count))
            if r.seg is not None and all(m is not None for _, m in kept):
                items = [(c, decode_mask(m)) for c, m in kept]
                rows.append((r.space_id, r.frame_id, r.object_type, "outline", lv,
                             len(drop_duplicate_masks(items, min_iou)), r.true_count))
    return pd.DataFrame(rows, columns=["space_id", "frame_id", "object_type", "rule", "level", "pred", "true"])


def score(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=["space_id", "object_type", "rule", "level", "n", "bias", "mae", "exact", "mean_true"])
    p = points.assign(err=points["pred"] - points["true"])
    return p.groupby(["space_id", "object_type", "rule", "level"]).agg(
        n=("err", "size"), bias=("err", "mean"), mae=("err", lambda s: s.abs().mean()),
        exact=("err", lambda s: (s == 0).mean()), mean_true=("true", "mean")).reset_index()


def recommend(scores: pd.DataFrame, *, rel_margin: float = 0.05, abs_margin: float = 0.05) -> pd.DataFrame:
    """Per (space, type): the current rule (yolo@0.25), the best option, and the recommendation (see module note)."""
    rows = []
    for (space, otype), g in scores.groupby(["space_id", "object_type"]):
        n = int(g["n"].max())
        cur = g[(g["rule"] == "yolo") & (g["level"] == CURRENT)]
        best = g.loc[g["mae"].idxmin()]
        row = {"space_id": space, "object_type": otype, "n": n, "mean_true": round(float(g["mean_true"].iloc[0]), 2),
               "current_mae": float(cur["mae"].iloc[0]) if len(cur) else np.nan,
               "current_bias": float(cur["bias"].iloc[0]) if len(cur) else np.nan,
               "best": f"{best['rule']}@{best['level']:.2f}", "best_mae": float(best["mae"])}
        if n < MIN_LABELS:
            rows.append(row | {"recommended": "-", "rec_mae": np.nan, "rec_bias": np.nan,
                               "why": f"only {n} labels (< {MIN_LABELS})"})
            continue
        ok = g[g["mae"] <= best["mae"] * (1 + rel_margin) + abs_margin].copy()
        if len(cur) and ((ok["rule"] == "yolo") & (ok["level"] == CURRENT)).any():
            rows.append(row | {"recommended": f"yolo@{CURRENT:.2f}", "rec_mae": row["current_mae"],
                               "rec_bias": row["current_bias"], "why": "current threshold is as good as the best: keep"})
            continue
        ok["simple"] = (ok["rule"] == "yolo").astype(int)
        pick = ok.sort_values(["simple", "level"], ascending=False).iloc[0]
        rows.append(row | {"recommended": f"{pick['rule']}@{pick['level']:.2f}", "rec_mae": float(pick["mae"]),
                           "rec_bias": float(pick["bias"]),
                           "why": "strictest option within a small margin of the best MAE"})
    return pd.DataFrame(rows)


def class_conf_suggestion(rec: pd.DataFrame) -> dict[str, float]:
    """detector.class_conf entries implied by the recommendations: only plain-yolo picks that differ from the
    current threshold, and only where every space that labels the type agrees (the setting is global)."""
    out: dict[str, float] = {}
    for otype, g in rec[rec["recommended"].str.startswith("yolo@")].groupby("object_type"):
        levels = {float(r.split("@")[1]) for r in g["recommended"]}
        if len(levels) == 1 and (lv := levels.pop()) != CURRENT:
            out[otype] = lv
    return out


def report(conn: sqlite3.Connection, *, min_iou: float = 0.5) -> str:
    pts = sweep(conn, min_iou=min_iou)
    if pts.empty:
        return ""
    rec = recommend(score(pts))
    pd.set_option("display.width", 220)
    lines = ["== detector threshold per object type (yolo@t = boxes with confidence >= t; outline@t = the same after "
             "outline de-duplication; current rule = yolo@0.25)",
             rec.round(2).to_string(index=False)]
    sug = class_conf_suggestion(rec)
    if sug:
        lines.append("suggested config/pipeline.yaml -> detector.class_conf: "
                     + "{" + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(sug.items())) + "}")
    outline = rec[rec["recommended"].str.startswith("outline@")]
    for r in outline.itertuples():
        lines.append(f"{r.space_id}/{r.object_type}: outline de-duplication beats plain counting; use the table column "
                     f"count.sam2_c{round(float(r.recommended.split('@')[1]) * 100):02d}.{r.object_type} as that series")
    return "\n".join(lines)

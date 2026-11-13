"""Build the Chronos-ready table: one row per 15-minute slot (IST), one column per dimension.

Rules (each one is a decision, not an accident):
  * Regular grid with no gaps. A slot without a usable frame is a ROW OF NaN, never a missing row and never 0
    (Chronos-2 masks NaN but needs regular timestamps).
  * One frame per slot: the usable one closest to the slot start (manual test runs and retries are dropped).
  * Names come from raw labels through the CURRENT taxonomy, so merging synonyms re-maps all history.
  * In a usable frame, an object type the source did not report is 0 (seen and absent). If the VLM failed for a
    frame, every VLM-derived column is NaN for that slot (unknown), while detector columns are still filled.
  * Every dimension is kept. The catalog marks which ones are model-ready (enough coverage, not constant,
    not a one-off flicker) and says why the others are not, so nothing is silently dropped.

Model inputs: the catalog marks each column eligible or not (with the reason), and select_covariates() picks a
small default set; with 19 rows of history, 180 inputs would be noise. Everything stays in the table.

Column families (prefix.source.name):
  cal.*    calendar, known in advance (future covariates)       scene.*  per-frame scene numbers
  img.*    measured image statistics                             cat.*    categorical scene fields (strings)
  count.yolo|vlm.<type>      rollup.yolo|vlm.<parent>  (parent + all is_a descendants)
  cell.yolo.<cell>.<type>    rel.vlm|geometry.<subject>.<predicate>.<object>      feat.<name> (frame_features)
  area.yolo.<type>           share of the image covered by that type's counted boxes (overlaps counted once)

Detector threshold variants: `yolo` is the counted set (detector.conf / class_conf). For every level in
table.conf_levels there is also count.yolo_cNN.<type> and area.yolo_cNN.<type>, built from ALL stored boxes with
confidence >= 0.NN. They are NaN for frames processed before low-confidence boxes were stored (unknown, not
zero). Variants are marked in the catalog and never picked as default model inputs: which threshold is right
per object type is decided from human labels, not here.
"""
from __future__ import annotations

import json
import re
import sqlite3
import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import SbiError
from ..perception.geometry import union_area
from ..taxonomy import Taxonomy, load_taxonomy

META = ["slot_status", "frame_id", "vlm_ok"]
CAT_COLUMNS = ["lighting", "door_state", "tidiness", "weather_visible"]
COUNT_KINDS = {"count", "rollup", "cell", "rel", "area"}
OBJECT_KINDS = {"count", "rollup", "cell", "area"}       # families whose last name part is an object type
_VARIANT = re.compile(r"^yolo_c\d\d$")


def level_source(level: float) -> str:
    """Source token for a detector confidence level: 0.05 -> 'yolo_c05'."""
    return f"yolo_c{round(level * 100):02d}"


@dataclass(frozen=True)
class TableResult:
    table: pd.DataFrame       # index: naive local timestamps (IST), regular step
    catalog: pd.DataFrame     # one row per dimension


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "cal.hour": index.hour, "cal.minute_of_day": index.hour * 60 + index.minute, "cal.dow": index.dayofweek,
        "cal.is_weekend": (index.dayofweek >= 5).astype(int), "cal.slot_of_day": (index.hour * 60 + index.minute) // 15,
    }, index=index)


def _pick_frames(conn: sqlite3.Connection, space_id: str, tz: str, step: int) -> pd.DataFrame:
    f = pd.read_sql_query("SELECT frame_id, ts_utc, capture_status, processed_utc, vlm_raw, brightness, contrast, "
                          "sharpness, detector_min_conf FROM frames WHERE space_id=?", conn, params=(space_id,))
    if f.empty:
        raise SbiError(f"no frames stored for space {space_id!r}")
    local = pd.to_datetime(f["ts_utc"], utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    f["slot"] = local.dt.floor(f"{step}min")
    f["offset"] = (local - f["slot"]).dt.total_seconds()
    usable = (f["capture_status"] == "ok") & f["processed_utc"].notna()
    f["status"] = np.where(usable, "ok", np.where(f["capture_status"] == "ok", "unprocessed", f["capture_status"]))
    f["vlm_ok"] = usable & f["vlm_raw"].notna() & ~f["vlm_raw"].fillna("").str.startswith('{"error"')
    f["rank"] = (~usable).astype(int)
    return f.sort_values(["slot", "rank", "offset"]).drop_duplicates("slot").set_index("slot")


def _name(tax: Taxonomy, raw: str | None, fallback: str) -> str:
    return (tax.lookup(raw) if raw else None) or fallback


def build_table(conn: sqlite3.Connection, space_id: str, *, tz: str, step_minutes: int = 15,
                start: str | None = None, end: str | None = None, min_coverage: float = 0.5,
                min_presence: float = 0.05, min_valid_slots: int = 8, min_present_slots: int = 3,
                exclude_labels: tuple[str, ...] = (), conf_levels: tuple[float, ...] = ()) -> TableResult:
    tax = load_taxonomy(conn)
    frames = _pick_frames(conn, space_id, tz, step_minutes)
    index = pd.date_range(pd.Timestamp(start) if start else frames.index.min(),
                          pd.Timestamp(end) if end else frames.index.max(), freq=f"{step_minutes}min", name="timestamp")
    t = pd.DataFrame(index=index)
    t["slot_status"] = frames["status"].reindex(index).fillna("no_frame")
    t["frame_id"] = frames["frame_id"].reindex(index)
    t["vlm_ok"] = frames["vlm_ok"].astype(bool).reindex(index, fill_value=False)
    valid = t["slot_status"] == "ok"
    vlm_ok = valid & t["vlm_ok"]
    t = t.join(calendar(index))

    floor = pd.to_numeric(frames["detector_min_conf"], errors="coerce").reindex(index)
    level_ok = {level_source(lv): valid & (floor <= lv + 1e-9) for lv in conf_levels}   # was the frame read that low?

    fid_slot = {int(fid): slot for slot, fid in t.loc[valid, "frame_id"].items()}
    blocks: dict[str, pd.Series] = {}

    def add(col: str, slot, value: float) -> None:
        blocks.setdefault(col, pd.Series(0.0, index=index))
        blocks[col].at[slot] += value

    if fid_slot:
        marks = ",".join("?" * len(fid_slot))
        ids = list(fid_slot)
        for fid, src, otype, raw, cell, n in conn.execute(
                f"SELECT frame_id, source, object_type, raw_label, cell_id, count FROM observations "
                f"WHERE frame_id IN ({marks})", ids):
            slot, name = fid_slot[fid], _name(tax, raw, otype)
            add(f"count.{src}.{name}", slot, n)
            for anc in tax.ancestors(name) | {name}:      # roll-up = the name itself + all is_a descendants
                add(f"rollup.{src}.{anc}", slot, n)
            if src == "yolo" and cell:
                add(f"cell.yolo.{cell}.{name}", slot, n)
        boxes: dict[tuple[str, str, object], list] = defaultdict(list)       # (source, name, slot) -> boxes
        for fid, otype, raw, bbox in conn.execute(
                f"SELECT frame_id, object_type, raw_label, bbox_norm FROM observations "
                f"WHERE source='yolo' AND bbox_norm IS NOT NULL AND frame_id IN ({marks})", ids):
            boxes[("yolo", _name(tax, raw, otype), fid_slot[fid])].append(json.loads(bbox))
        if conf_levels:
            for fid, raw, conf, bbox in conn.execute(
                    f"SELECT frame_id, raw_label, confidence, bbox_norm FROM detections WHERE frame_id IN ({marks})", ids):
                slot, name = fid_slot[fid], _name(tax, raw, raw)
                for lv in conf_levels:
                    if conf >= lv:
                        add(f"count.{level_source(lv)}.{name}", slot, 1)
                        boxes[(level_source(lv), name, slot)].append(json.loads(bbox))
        for (src, name, slot), bs in boxes.items():
            add(f"area.{src}.{name}", slot, round(union_area(bs), 5))
        for fid, src, s, p, o, rs, ro, n in conn.execute(
                f"SELECT frame_id, source, subject, predicate, object, raw_subject, raw_object, count FROM relations "
                f"WHERE frame_id IN ({marks})", ids):
            add(f"rel.{src}.{_name(tax, rs, s)}.{p}.{_name(tax, ro, o)}", fid_slot[fid], n)

    # keep only roll-ups for names that actually have is_a descendants (otherwise they duplicate count.*)
    has_children = {p for parents in tax.parents.values() for p in parents}
    for col in [c for c in blocks if c.startswith("rollup.")]:
        if col.split(".", 2)[2] not in has_children:
            del blocks[col]

    counts = pd.DataFrame(blocks, index=index)
    for col in counts.columns:                       # 0 only where the source actually looked; NaN otherwise
        src = col.split(".")[1]
        counts.loc[~(vlm_ok if src == "vlm" else level_ok.get(src, valid)), col] = np.nan

    scene = pd.read_sql_query(
        "SELECT frame_id, people_yolo, people_vlm, vehicles_yolo, count_disagree, lighting, door_state, tidiness, "
        "weather_visible, attrs_json FROM scene_state WHERE space_id=?", conn, params=(space_id,))
    scene = scene[scene["frame_id"].isin(list(fid_slot))]
    scene.index = scene["frame_id"].map(fid_slot)
    num = scene[["people_yolo", "people_vlm", "vehicles_yolo", "count_disagree"]].add_prefix("scene.")
    cats = scene[CAT_COLUMNS].copy()
    for i, a in scene["attrs_json"].dropna().items():
        for k, v in json.loads(a).items():
            cats.loc[i, k] = v
    cats = cats.add_prefix("cat.").reindex(index)
    cats.loc[~vlm_ok] = np.nan
    img = frames[["brightness", "contrast", "sharpness"]].reindex(index).add_prefix("img.")
    img.loc[~valid] = np.nan

    feats = pd.read_sql_query("SELECT ff.frame_id, ff.name, ff.value FROM frame_features ff JOIN frames f USING(frame_id) "
                              "WHERE f.space_id=?", conn, params=(space_id,))
    feats = feats[feats["frame_id"].isin(list(fid_slot))]
    feat_w = (feats.assign(slot=feats["frame_id"].map(fid_slot)).pivot_table(index="slot", columns="name", values="value")
              .add_prefix("feat.").reindex(index)) if not feats.empty else pd.DataFrame(index=index)

    def family(prefix: str) -> list[str]:
        return sorted(c for c in counts.columns if c.startswith(prefix))
    ordered = pd.concat([num.reindex(index), img, counts[family("count.")], counts[family("area.")],
                         counts[family("rollup.")], counts[family("cell.")], counts[family("rel.")], cats, feat_w], axis=1)
    table = pd.concat([t, ordered], axis=1)
    return TableResult(table, _catalog(table, min_coverage, min_presence, min_valid_slots, min_present_slots,
                                       frozenset(exclude_labels)))


def _catalog(table: pd.DataFrame, min_coverage: float, min_presence: float, min_valid: int, min_present_slots: int,
             exclude_labels: frozenset[str]) -> pd.DataFrame:
    rows = []
    for col in table.columns:
        if col in META:
            continue
        s = table[col]
        parts = col.split(".")
        kind, source = parts[0], (parts[1] if len(parts) >= 3 else "")
        nn = s.dropna()
        numeric = kind != "cat" and pd.api.types.is_numeric_dtype(s)
        coverage = float(s.notna().mean())
        present = int((nn > 0).sum()) if numeric else len(nn)
        presence = float((nn > 0).mean()) if numeric and len(nn) else np.nan
        std = float(nn.std(ddof=0)) if numeric and len(nn) else np.nan
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)       # constant stretches make autocorr undefined
            ac = float(s.autocorr(1)) if numeric and len(nn) > 2 and std > 0 else np.nan
        distinct = int(nn.nunique())
        obj = parts[-1] if kind in OBJECT_KINDS else None
        if kind == "cal":
            ready, why = True, "known in advance"
        elif obj in exclude_labels:
            ready, why = False, "structural object (building part)"
        elif len(nn) < min_valid:
            ready, why = False, f"only {len(nn)} valid slots (< {min_valid})"
        elif coverage < min_coverage:
            ready, why = False, f"coverage {coverage:.0%} < {min_coverage:.0%}"
        elif distinct < 2:
            ready, why = False, "constant"
        elif kind in COUNT_KINDS and (present < min_present_slots or presence < min_presence):
            ready, why = False, f"present in only {present} slots (flicker)"
        else:
            ready, why = True, "ok"
        rows.append({"column": col, "kind": kind, "source": source, "known_future": kind == "cal",
                     "variant": bool(_VARIANT.match(source)),
                     "coverage": round(coverage, 3), "valid_slots": len(nn), "present_slots": present,
                     "presence": None if np.isnan(presence) else round(presence, 3),
                     "mean": round(float(nn.mean()), 3) if numeric and len(nn) else None,
                     "std": None if np.isnan(std) else round(std, 3),
                     "autocorr1": None if np.isnan(ac) else round(ac, 3), "distinct": distinct,
                     "model_ready": ready, "reason": why})
    return pd.DataFrame(rows)


def select_covariates(catalog: pd.DataFrame, *, targets: list[str], max_covariates: int, always: list[str],
                      min_autocorr: float, autocorr_min_slots: int) -> list[str]:
    """Default model inputs: `always` columns (if eligible), then eligible columns ranked by smoothness
    (lag-1 autocorrelation; real occupancy changes smoothly, model noise flickers) and coverage.
    Threshold variants (count.yolo_cNN.*) are near-copies of the counted series, so they are only used when
    named in `always`; otherwise they would crowd real inputs out of the cap.
    Smoothness only filters once there are enough slots to estimate it (about +-0.23 noise at 19 slots).
    Every other column stays in the table; this only picks what a model gets by default."""
    ready = catalog[catalog["model_ready"] & (catalog["kind"] != "cal") & ~catalog["column"].isin(targets)].copy()
    chosen = [c for c in always if c in set(ready["column"])]
    rest = ready[~ready["column"].isin(chosen) & ~ready["variant"]].copy()
    rest["ac"] = rest["autocorr1"].astype(float).fillna(0.0)
    if len(catalog) and catalog["valid_slots"].max() >= autocorr_min_slots:
        rest = rest[(rest["kind"] == "cat") | (rest["ac"] >= min_autocorr)]
    rest = rest.sort_values(["ac", "coverage", "std"], ascending=False)
    return chosen + list(rest["column"])[: max(0, max_covariates - len(chosen))]


def to_chronos(result: TableResult, space_id: str, targets: list[str], horizon: int,
               covariates: list[str] | None = None, **select_kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(context_df, future_df) in the long format of Chronos2Pipeline.predict_df(id_column='id',
    timestamp_column='timestamp', target=targets). Calendar columns go in both frames (known-future covariates);
    every other covariate is past-only. Default covariates = select_covariates(**select_kw)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    table, cat = result.table, result.catalog
    missing = [c for c in targets if c not in table.columns]
    if missing:
        raise ValueError(f"unknown target columns: {missing}")
    cal_cols = [c for c in table.columns if c.startswith("cal.")]
    if covariates is None:
        opts = {"max_covariates": 24, "always": [], "min_autocorr": 0.3, "autocorr_min_slots": 96} | select_kw
        covariates = select_covariates(cat, targets=targets, **opts)
    context = table[targets + covariates + cal_cols].copy()
    for c in context.columns:
        if c.startswith("cat."):
            context[c] = context[c].astype(object).where(context[c].notna(), None)
    context = context.reset_index().rename(columns={"index": "timestamp"})
    context.insert(0, "id", space_id)
    step = table.index.freq or pd.tseries.frequencies.to_offset(pd.infer_freq(table.index))
    future_index = pd.date_range(table.index[-1] + step, periods=horizon, freq=step, name="timestamp")
    future = calendar(future_index).reset_index()
    future.insert(0, "id", space_id)
    return context, future

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

Outline (mask) columns, when a segmenter ran: sources `sam2` (counted boxes) and `sam2_cNN` (all boxes >= 0.NN):
  count.sam2*.<type>  boxes after removing same-type boxes that outline the same object
  area.sam2*.<type>   share of the image covered by outlines      cellarea.sam2*.<cell>.<type>  share of a cell
cellarea is built for the counted boxes only. All are NaN for frames no segmenter outlined. Only area.sam2.*
and cellarea.sam2.* can be default inputs.

Image-vector columns, when an embedder ran (see forecast/embed_features.py):
  emb.<model>.pc1..pcN   position along the main directions in which the scene varies (N = embed_components)
  emb.<model>.drift_prev / drift_typical   change since the previous slot / distance from the typical scene
They need no detector and no VLM, so they exist for every embedded frame. NaN for frames never embedded.
The principal directions are fit on `embed_fit_end` and earlier when given, otherwise on the whole table.

Probe columns, when a scorer ran (see forecast/probe_features.py):
  probe.<scorer>.<concept>   mean raw score of the concept's paraphrased prompts (SigLIP: a logit, 0 = even odds)
Concepts come from the space file (`probes`). They are named, continuous descriptions of the scene (is it crowded,
dark, cluttered, ...) and say nothing the detector did not need to find. NaN for frames not scored, or when any
prompt of the concept is missing for that frame.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import warnings
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import SbiError
from ..perception.geometry import union_area
from ..perception.masks import decode_mask
from ..store.embeddings import load_embeddings
from ..store.scores import load_scores
from ..sitecal import CalendarSpec
from ..taxonomy import Taxonomy, load_taxonomy
from .embed_features import embedding_features
from .feature_cache import FeatureCache, fingerprint
from .probe_features import concept_values
from .mask_features import mask_features

log = logging.getLogger(__name__)
META = ["slot_status", "frame_id", "vlm_ok"]
CAT_COLUMNS = ["lighting", "door_state", "tidiness", "weather_visible"]
COUNT_KINDS = {"count", "rollup", "cell", "rel", "area", "cellarea"}
OBJECT_KINDS = {"count", "rollup", "cell", "area", "cellarea"}   # families whose last name part is an object type
_LEVEL = re.compile(r"_c\d\d$")


def level_source(level: float, base: str = "yolo") -> str:
    """Source token for a detector confidence level: 0.05 -> 'yolo_c05'."""
    return f"{base}_c{round(level * 100):02d}"


def is_variant(kind: str, source: str) -> bool:
    """A near-copy of a primary series (another threshold, or the same count after outline de-duplication)."""
    return bool(_LEVEL.search(source)) or (source == "sam2" and kind == "count")


@dataclass(frozen=True)
class TableResult:
    table: pd.DataFrame       # index: naive local timestamps (IST), regular step
    catalog: pd.DataFrame     # one row per dimension
    cal: CalendarSpec = CalendarSpec()    # holidays / working days, so future calendar rows match the table's


def calendar(index: pd.DatetimeIndex, spec: CalendarSpec = CalendarSpec()) -> pd.DataFrame:
    """Calendar columns, all known in advance. is_holiday / is_working_day come from the site calendar."""
    holiday = np.array([d in spec.holidays for d in index.date], dtype=bool)
    working = np.isin(index.dayofweek, sorted(spec.working_days)) & ~holiday
    return pd.DataFrame({
        "cal.hour": index.hour, "cal.minute_of_day": index.hour * 60 + index.minute, "cal.dow": index.dayofweek,
        "cal.is_weekend": (index.dayofweek >= 5).astype(int), "cal.slot_of_day": (index.hour * 60 + index.minute) // 15,
        "cal.is_holiday": holiday.astype(int), "cal.is_working_day": working.astype(int),
    }, index=index)


def _pick_frames(conn: sqlite3.Connection, space_id: str, tz: str, step: int) -> pd.DataFrame:
    f = pd.read_sql_query("SELECT frame_id, ts_utc, capture_status, processed_utc, vlm_raw, brightness, contrast, "
                          "sharpness, detector_min_conf, segmenter_version FROM frames WHERE space_id=?", conn, params=(space_id,))
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


FEATURES_VERSION = 1          # bump when frame_columns() changes what it computes: old cache entries then miss
_SQL_BATCH = 500              # ids per SQL statement (SQLite limits bound variables)


def frame_columns(conn: sqlite3.Connection, tax: Taxonomy, frame_ids: list[int], *, conf_levels: tuple[float, ...],
                  mask_sources: dict[str, float | None], mask_dup_iou: float, grid: tuple[int, int] | None,
                  cell_area_min: float) -> dict[int, dict[str, float]]:
    """For each photo, every column derived from its stored objects, boxes, outlines and relations:
    count.* rollup.* cell.* area.* cellarea.* rel.*  ({frame_id: {column: value}}). A photo with nothing stored gets
    an empty dict. Pure function of the stored rows, the vocabulary links and the arguments."""
    out: dict[int, dict[str, float]] = {int(f): defaultdict(float) for f in frame_ids}
    for start in range(0, len(frame_ids), _SQL_BATCH):
        ids = [int(f) for f in frame_ids[start:start + _SQL_BATCH]]
        marks = ",".join("?" * len(ids))
        for fid, src, otype, raw, cell, n in conn.execute(
                f"SELECT frame_id, source, object_type, raw_label, cell_id, count FROM observations "
                f"WHERE frame_id IN ({marks})", ids):
            cols, name = out[fid], _name(tax, raw, otype)
            cols[f"count.{src}.{name}"] += n
            for anc in tax.ancestors(name) | {name}:      # roll-up = the name itself + all is_a descendants
                cols[f"rollup.{src}.{anc}"] += n
            if src == "yolo" and cell:
                cols[f"cell.yolo.{cell}.{name}"] += n
        boxes: dict[tuple[str, str, int], list] = defaultdict(list)          # (source, name, frame) -> boxes
        for fid, otype, raw, bbox in conn.execute(
                f"SELECT frame_id, object_type, raw_label, bbox_norm FROM observations "
                f"WHERE source='yolo' AND bbox_norm IS NOT NULL AND frame_id IN ({marks})", ids):
            boxes[("yolo", _name(tax, raw, otype), fid)].append(json.loads(bbox))
        outlines: dict[int, list] = defaultdict(list)                       # frame -> (type, conf, counted, mask)
        for fid, raw, conf, counted, bbox, mask in conn.execute(
                f"SELECT frame_id, raw_label, confidence, counted, bbox_norm, mask FROM detections "
                f"WHERE frame_id IN ({marks})", ids):
            name = _name(tax, raw, raw)
            for lv in conf_levels:
                if conf >= lv:
                    out[fid][f"count.{level_source(lv)}.{name}"] += 1
                    boxes[(level_source(lv), name, fid)].append(json.loads(bbox))
            if mask is not None:
                outlines[fid].append((name, conf, bool(counted), decode_mask(mask)))
        for fid, masked in outlines.items():
            for col, value in mask_features(masked, sources=mask_sources, min_iou=mask_dup_iou, grid=grid,
                                            cell_min=cell_area_min).items():
                out[fid][col] += value
        for (src, name, fid), bs in boxes.items():
            out[fid][f"area.{src}.{name}"] += round(union_area(bs), 5)
        for fid, src, subj, p, o, rs, ro, n in conn.execute(
                f"SELECT frame_id, source, subject, predicate, object, raw_subject, raw_object, count FROM relations "
                f"WHERE frame_id IN ({marks})", ids):
            out[fid][f"rel.{src}.{_name(tax, rs, subj)}.{p}.{_name(tax, ro, o)}"] += n
    return {fid: dict(cols) for fid, cols in out.items()}


def _frame_columns_cached(conn: sqlite3.Connection, tax: Taxonomy, stamps: dict[int, str], cache: FeatureCache | None,
                          *, space_id: str, **kw) -> dict[int, dict[str, float]]:
    """frame_columns() for the photos in `stamps`, taking unchanged photos from the cache when there is one."""
    ids = list(stamps)
    if cache is None:
        return frame_columns(conn, tax, ids, **kw)
    key = fingerprint(FEATURES_VERSION, space_id, kw["conf_levels"], kw["mask_sources"], kw["mask_dup_iou"], kw["grid"],
                      kw["cell_area_min"], sorted(tax.known), tax.alias, tax.merged_into,
                      {k: sorted(v) for k, v in tax.parents.items()})
    have = cache.load(key, stamps)
    missing = [f for f in ids if f not in have]
    if missing:
        fresh = frame_columns(conn, tax, missing, **kw)
        cache.store(key, stamps, fresh)
        have.update(fresh)
    return have


def build_table(conn: sqlite3.Connection, space_id: str, *, tz: str, step_minutes: int = 15,
                start: str | None = None, end: str | None = None, min_coverage: float = 0.5,
                min_presence: float = 0.05, min_valid_slots: int = 8, min_present_slots: int = 3,
                exclude_labels: tuple[str, ...] = (), conf_levels: tuple[float, ...] = (),
                mask_dup_iou: float = 0.5, grid: tuple[int, int] | None = None,
                cell_area_min: float = 0.02, embed_model: str | None = None, embed_components: int = 4,
                embed_fit_end: str | None = None, probe_scorer: str | None = None,
                probe_groups: dict[str, list[str]] | None = None,
                calendar_spec: CalendarSpec = CalendarSpec(), cache: FeatureCache | None = None,
                max_slots: int | None = None) -> TableResult:
    """`cache` keeps each photo's derived columns between builds (same table, much faster). `max_slots` keeps only
    the newest that many slots (after `start`/`end`), for callers that cannot use more history than that."""
    tax = load_taxonomy(conn)
    frames = _pick_frames(conn, space_id, tz, step_minutes)
    index = pd.date_range(pd.Timestamp(start) if start else frames.index.min(),
                          pd.Timestamp(end) if end else frames.index.max(), freq=f"{step_minutes}min", name="timestamp")
    if max_slots is not None and len(index) > max_slots:
        index = index[-max_slots:]
    t = pd.DataFrame(index=index)
    t["slot_status"] = frames["status"].reindex(index).fillna("no_frame")
    t["frame_id"] = frames["frame_id"].reindex(index)
    t["vlm_ok"] = frames["vlm_ok"].astype(bool).reindex(index, fill_value=False)
    valid = t["slot_status"] == "ok"
    vlm_ok = valid & t["vlm_ok"]
    t = t.join(calendar(index, calendar_spec))

    floor = pd.to_numeric(frames["detector_min_conf"], errors="coerce").reindex(index)
    level_ok = {level_source(lv): valid & (floor <= lv + 1e-9) for lv in conf_levels}   # was the frame read that low?
    outlined = valid & frames["segmenter_version"].notna().reindex(index, fill_value=False)
    mask_sources: dict[str, float | None] = {"sam2": None, **{level_source(lv, "sam2"): lv for lv in conf_levels}}
    source_ok = {**level_ok, "sam2": outlined,
                 **{level_source(lv, "sam2"): outlined & level_ok[level_source(lv)] for lv in conf_levels}}

    fid_slot = {int(fid): slot for slot, fid in t.loc[valid, "frame_id"].items()}
    stamps = {int(fid): str(frames["processed_utc"].get(slot)) for slot, fid in t.loc[valid, "frame_id"].items()}
    per_frame = _frame_columns_cached(conn, tax, stamps, cache, space_id=space_id, conf_levels=conf_levels,
                                      mask_sources=mask_sources, mask_dup_iou=mask_dup_iou, grid=grid,
                                      cell_area_min=cell_area_min)
    blocks: dict[str, dict] = defaultdict(dict)                              # column -> {slot: value}
    for fid, cols in per_frame.items():
        slot = fid_slot[fid]
        for col, value in cols.items():
            blocks[col][slot] = value

    # keep only roll-ups for names that actually have is_a descendants (otherwise they duplicate count.*)
    has_children = {p for parents in tax.parents.values() for p in parents}
    for col in [c for c in blocks if c.startswith("rollup.")]:
        if col.split(".", 2)[2] not in has_children:
            del blocks[col]

    counts = pd.DataFrame({col: pd.Series(by_slot, dtype=float) for col, by_slot in blocks.items()},
                          index=index).fillna(0.0)
    for col in counts.columns:                       # 0 only where the source actually looked; NaN otherwise
        src = col.split(".")[1]
        counts.loc[~(vlm_ok if src == "vlm" else source_ok.get(src, valid)), col] = np.nan

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

    emb_w = _embedding_columns(conn, space_id, index, fid_slot, embed_model, embed_components, embed_fit_end)
    probe_w = _probe_columns(conn, space_id, index, fid_slot, probe_scorer, probe_groups or {})

    def family(prefix: str) -> list[str]:
        return sorted(c for c in counts.columns if c.startswith(prefix))
    ordered = pd.concat([num.reindex(index), img, counts[family("count.")], counts[family("area.")],
                         counts[family("rollup.")], counts[family("cell.")],
                         counts[family("cellarea.")], counts[family("rel.")], cats, feat_w, emb_w, probe_w], axis=1)
    table = pd.concat([t, ordered], axis=1)
    return TableResult(table, _catalog(table, min_coverage, min_presence, min_valid_slots, min_present_slots,
                                       frozenset(exclude_labels)), calendar_spec)


def _embedding_columns(conn: sqlite3.Connection, space_id: str, index: pd.DatetimeIndex, fid_slot: dict,
                       model: str | None, components: int, fit_end: str | None) -> pd.DataFrame:
    """emb.<model>.* columns from the stored image vectors; an empty frame when there are none."""
    empty = pd.DataFrame(index=index)
    if not model or not fid_slot:
        return empty
    vectors, corrupt = load_embeddings(conn, space_id, model, set(fid_slot))
    if corrupt:
        log.warning("%s: %d corrupt image vectors left out (frames %s)", space_id, len(corrupt), corrupt[:5])
    if not vectors:
        return empty
    mat = np.full((len(index), next(iter(vectors.values())).size), np.nan, dtype=np.float32)
    position = {slot: i for i, slot in enumerate(index)}
    for fid, vec in vectors.items():
        mat[position[fid_slot[fid]]] = vec
    fit = None if fit_end is None else np.asarray(index < pd.Timestamp(fit_end))
    name = re.sub(r"[^A-Za-z0-9_-]", "_", model.split("@")[0])        # a column source has no dots
    return pd.DataFrame({f"emb.{name}.{k}": v for k, v in embedding_features(mat, components=components,
                                                                            fit=fit).items()}, index=index)


def _probe_columns(conn: sqlite3.Connection, space_id: str, index: pd.DatetimeIndex, fid_slot: dict,
                   scorer: str | None, groups: dict[str, list[str]]) -> pd.DataFrame:
    """probe.<scorer>.<concept> columns from the stored prompt scores; an empty frame when there are none."""
    empty = pd.DataFrame(index=index)
    if not scorer or not groups or not fid_slot:
        return empty
    scores = load_scores(conn, space_id, scorer, set(fid_slot))
    if not scores:
        return empty
    name = re.sub(r"[^A-Za-z0-9_-]", "_", scorer)                    # a column source has no dots
    position = {slot: i for i, slot in enumerate(index)}
    cols = {c: np.full(len(index), np.nan) for c in groups}
    for fid, by_prompt in scores.items():
        row = position[fid_slot[fid]]
        for concept, value in concept_values(by_prompt, groups).items():
            cols[concept][row] = value
    return pd.DataFrame({f"probe.{name}.{c}": values for c, values in cols.items()}, index=index)


def build_catalog(table: pd.DataFrame, *, min_coverage: float, min_presence: float, min_valid_slots: int,
                  min_present_slots: int, exclude_labels: tuple[str, ...] = ()) -> pd.DataFrame:
    """The column catalog for any slice of a table (a backtest builds it from the history before each forecast, so
    column selection never sees the future)."""
    return _catalog(table, min_coverage, min_presence, min_valid_slots, min_present_slots, frozenset(exclude_labels))


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
                     "variant": is_variant(kind, source),
                     "coverage": round(coverage, 3), "valid_slots": len(nn), "present_slots": present,
                     "presence": None if np.isnan(presence) else round(presence, 3),
                     "mean": round(float(nn.mean()), 3) if numeric and len(nn) else None,
                     "std": None if np.isnan(std) else round(std, 3),
                     "autocorr1": None if np.isnan(ac) else round(ac, 3), "distinct": distinct,
                     "model_ready": ready, "reason": why})
    return pd.DataFrame(rows)


def select_covariates(catalog: pd.DataFrame, *, targets: list[str], max_covariates: int, always: list[str],
                      min_autocorr: float, autocorr_min_slots: int, kinds: Collection[str] | None = None) -> list[str]:
    """Default model inputs: `always` columns (if eligible), then eligible columns ranked by smoothness
    (lag-1 autocorrelation; real occupancy changes smoothly, model noise flickers) and coverage.
    Variants (other thresholds such as count.yolo_cNN.*, and de-duplicated counts) are near-copies of a primary
    series, so they are only used when named in `always`; otherwise they would crowd real inputs out of the cap.
    Smoothness only filters once there are enough slots to estimate it (about +-0.23 noise at 19 slots).
    `kinds` limits the choice to those column kinds (a covariate-set variant); None means any kind.
    Every other column stays in the table; this only picks what a model gets by default."""
    ready = catalog[catalog["model_ready"] & (catalog["kind"] != "cal") & ~catalog["column"].isin(targets)].copy()
    if kinds is not None:
        ready = ready[ready["kind"].isin(set(kinds))]
    chosen = [c for c in always if c in set(ready["column"])]
    rest = ready[~ready["column"].isin(chosen) & ~ready["variant"]].copy()
    rest["ac"] = rest["autocorr1"].astype(float).fillna(0.0)
    if len(catalog) and catalog["valid_slots"].max() >= autocorr_min_slots:
        rest = rest[(rest["kind"] == "cat") | (rest["ac"] >= min_autocorr)]
    rest = rest.sort_values(["ac", "coverage", "std"], ascending=False)
    return chosen + list(rest["column"])[: max(0, max_covariates - len(chosen))]


def all_covariates(catalog: pd.DataFrame, *, targets: list[str]) -> list[str]:
    """Every model-ready column except the calendar and the targets: counts, areas, cells, relations, scene fields,
    image statistics, image-vector components and probes, threshold variants included. No cap and no smoothness
    filter: Chronos-2 attends across all of them and decides what is useful. Ordered as the table is."""
    ready = catalog[catalog["model_ready"] & (catalog["kind"] != "cal") & ~catalog["column"].isin(targets)]
    return list(ready["column"])


def to_chronos(result: TableResult, space_id: str, targets: list[str], horizon: int,
               covariates: list[str] | None = None, calendar_inputs: bool = True, **select_kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(context_df, future_df) in the long format of Chronos2Pipeline.predict_df(id_column='id',
    timestamp_column='timestamp', target=targets). Calendar columns go in both frames (known-future covariates);
    every other covariate is past-only. Default covariates = select_covariates(**select_kw)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    table, cat = result.table, result.catalog
    missing = [c for c in targets if c not in table.columns]
    if missing:
        raise ValueError(f"unknown target columns: {missing}")
    cal_cols = [c for c in table.columns if c.startswith("cal.")] if calendar_inputs else []
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
    future = calendar(future_index, result.cal).reset_index()
    future.insert(0, "id", space_id)
    if not calendar_inputs:                           # nothing known in advance: the future frame is just the clock
        future = future[["id", "timestamp"]]
    return context, future

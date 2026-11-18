"""Everything the dashboard shows, as plain DataFrames (testable without Streamlit). Read-only: nothing here writes.

All times are local (IST): table timestamps already are; UTC columns from the database are converted here.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..spaces import SpaceConfig
from ..store.forecasts import load_forecast
from ..timeutil import get_zone


def connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _local(ts_utc: str | None, tz: str) -> str:
    if not ts_utc:
        return "-"
    return pd.Timestamp(ts_utc).tz_convert(get_zone(tz)).strftime("%Y-%m-%d %H:%M")


def health(conn: sqlite3.Connection, spaces: list[SpaceConfig], now_utc: datetime) -> pd.DataFrame:
    """One row per space: is it capturing, and is every stage keeping up over the last 24 hours."""
    since = (now_utc - timedelta(hours=24)).isoformat()
    rows = []
    for s in spaces:
        r = conn.execute(
            "SELECT COUNT(*) n, SUM(capture_status='ok') ok, SUM(processed_utc IS NOT NULL) processed, "
            "SUM(vlm_raw IS NOT NULL AND vlm_raw NOT LIKE '{\"error\"%') vlm, SUM(segmenter_version IS NOT NULL) outlined, "
            "SUM(EXISTS(SELECT 1 FROM frame_embeddings e WHERE e.frame_id=f.frame_id)) embedded, "
            "SUM(EXISTS(SELECT 1 FROM frame_scores c WHERE c.frame_id=f.frame_id)) scored, MAX(ts_utc) last "
            "FROM frames f WHERE space_id=? AND ts_utc >= ?", (s.space_id, since)).fetchone()
        fc = conn.execute("SELECT origin_local, model FROM forecast_runs WHERE space_id=? ORDER BY run_id DESC LIMIT 1",
                          (s.space_id,)).fetchone()
        ok = r["ok"] or 0
        share = lambda k: f"{(r[k] or 0) / ok:.0%}" if ok else "-"
        rows.append({"space": s.space_id, "last photo (IST)": _local(r["last"], s.tz),
                     "photos 24 h": f"{ok} / {24 * 60 // s.cadence_minutes}", "failed captures": (r["n"] or 0) - ok,
                     "processed": share("processed"), "VLM": share("vlm"), "outlines": share("outlined"),
                     "image vector": share("embedded"), "probes": share("scored"),
                     "latest forecast (IST)": fc["origin_local"].replace("T", " ")[:16] if fc else "-",
                     "forecaster": fc["model"] if fc else "-"})
    return pd.DataFrame(rows)


def log_problems(path: Path, *, limit: int = 15) -> list[str]:
    """The most recent WARNING / ERROR lines of the tick log (newest last)."""
    if not path.exists():
        return []
    with path.open(errors="replace") as fh:
        lines = [ln.rstrip() for ln in fh if " WARNING " in ln or " ERROR " in ln]
    return lines[-limit:]


def latest_forecast(conn: sqlite3.Connection, space_id: str) -> tuple[dict | None, pd.DataFrame]:
    row = conn.execute("SELECT * FROM forecast_runs WHERE space_id=? ORDER BY run_id DESC LIMIT 1", (space_id,)).fetchone()
    if row is None:
        return None, pd.DataFrame()
    run = dict(row) | {"targets": json.loads(row["targets_json"]), "covariates": json.loads(row["covariates_json"])}
    return run, load_forecast(conn, row["run_id"])


def chart_data(table: pd.DataFrame, forecast: pd.DataFrame, target: str, *, days: float) -> pd.DataFrame:
    """Long frame for one chart: measured values over the last `days`, then the forecast median and band.
    Columns: timestamp, kind ('measured' | 'forecast'), value, low, high."""
    out = []
    if target in table.columns:
        recent = table[target].loc[table.index.max() - timedelta(days=days):]
        out.append(pd.DataFrame({"timestamp": recent.index, "kind": "measured", "value": recent.to_numpy(float),
                                 "low": float("nan"), "high": float("nan")}))
    if not forecast.empty:
        f = forecast[forecast["target"] == target]
        qs = sorted(c for c in f.columns if c.startswith("q") and c[1:].isdigit())
        if len(f) and "q50" in qs:
            out.append(pd.DataFrame({"timestamp": pd.to_datetime(f["timestamp"]), "kind": "forecast",
                                     "value": f["q50"].to_numpy(float), "low": f[qs[0]].to_numpy(float),
                                     "high": f[qs[-1]].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["timestamp", "kind", "value", "low", "high"])


def backtest_overall(exports: Path, space_id: str) -> pd.DataFrame:
    """The last `sbi backtest` summary for the space, if one was written."""
    path = exports / f"{space_id}_backtest.csv"
    if not path.exists():
        return pd.DataFrame()
    from ..forecast.metrics import overall
    return overall(pd.read_csv(path))


# ------------------------------------------------------------------ v1.1 dashboard: now, forecast, pipeline, columns

SOURCE_NAMES = {"yolo": "detector", "vlm": "VLM", "sam2": "outline", "geometry": "geometry"}
KIND_NAMES = {"scene": "Scene totals", "steady": "Steady counts of things that do not move", "count": "Counts per object type", "rollup": "Counts per object group",
              "area": "Area covered per type", "cell": "Counts per grid cell", "cellarea": "Area per grid cell",
              "rel": "Relations between objects", "img": "Image statistics", "cat": "Scene fields (text)",
              "emb": "Image vector", "probe": "Scene description scores", "cal": "Calendar", "feat": "Other features"}


def _source(token: str) -> str:
    """'yolo' -> 'detector'; 'yolo_c05' -> 'detector, confidence 0.05+' (every stored box at or above that level)."""
    base, _, level = token.partition("_c")
    name = SOURCE_NAMES.get(base, base)
    return f"{name}, confidence 0.{level}+" if level.isdigit() else SOURCE_NAMES.get(token, token)


def label(column: str) -> str:
    """A readable name for a table column: 'count.yolo.chair' -> 'chair count (detector)'."""
    parts = column.split(".")
    kind, src = parts[0], _source(parts[1]) if len(parts) > 1 else ""
    if kind == "scene":
        name, _, source = parts[1].rpartition("_")
        return f"{name} ({SOURCE_NAMES[source]})" if source in SOURCE_NAMES and name else parts[1].replace("_", " ")
    if kind == "steady" and len(parts) == 3:
        return f"{parts[2]} count (steady, 24 h)"
    if kind in ("count", "rollup") and len(parts) == 3:
        return f"{parts[2]} count ({src})"
    if kind == "area" and len(parts) == 3:
        return f"{parts[2]} area share ({src})"
    if kind in ("cell", "cellarea") and len(parts) == 4:
        return f"{parts[3]} {'count' if kind == 'cell' else 'area'} in cell {parts[2]} ({src})"
    if kind == "rel" and len(parts) == 5:
        return f"{parts[2]} {parts[3].replace('_', ' ')} {parts[4]} ({src})"
    if kind == "probe":
        return f"'{parts[-1].replace('_', ' ')}' score"
    if kind == "emb":
        return f"image vector {parts[-1]}"
    if kind in ("img", "cat", "cal"):
        return parts[-1].replace("_", " ")
    return column


def table_stamp(conn: sqlite3.Connection, space_id: str) -> tuple[int, int, int]:
    """Changes exactly when the space's table or stored forecasts could change (a new or re-read photo, a new
    forecast run), so a cached table is rebuilt only then."""
    f = conn.execute("SELECT COALESCE(MAX(frame_id), 0), COUNT(processed_utc) FROM frames WHERE space_id=?",
                     (space_id,)).fetchone()
    r = conn.execute("SELECT COALESCE(MAX(run_id), 0) FROM forecast_runs WHERE space_id=?", (space_id,)).fetchone()
    return int(f[0]), int(f[1]), int(r[0])


def usual(series: pd.Series, when: pd.DatetimeIndex, *, before: pd.Timestamp | None = None) -> pd.Series:
    """The series' mean at the same clock time on earlier days (NaN where no earlier day has that time)."""
    s = series.dropna()
    if before is not None:
        s = s[s.index.normalize() < before.normalize()]
    by_time = s.groupby(s.index.hour * 60 + s.index.minute).mean()
    return pd.Series(by_time.reindex(when.hour * 60 + when.minute).to_numpy(float), index=when)


def summary(table: pd.DataFrame, forecast: pd.DataFrame, targets: list[str], *,
            leads: tuple[int, ...] = (4, 8, 16), step_minutes: int = 15) -> pd.DataFrame:
    """One row per target: the latest measured value, the usual value at this clock time, and the stored forecast
    (median, low, high) at each lead (in slots)."""
    rows = []
    for t in targets:
        if t not in table.columns:
            continue
        s = pd.to_numeric(table[t], errors="coerce").dropna()
        if s.empty:
            continue
        at = s.index[-1]
        row = {"column": t, "series": label(t), "at": at, "now": float(s.iloc[-1]),
               "usual": float(usual(s, pd.DatetimeIndex([at]), before=at).iloc[0])}
        f = forecast[forecast["target"] == t] if not forecast.empty else forecast
        qs = sorted((c for c in f.columns if c.startswith("q") and c[1:].isdigit()), key=lambda c: int(c[1:]))
        for lead in leads:
            hit = f[f["step"] == lead] if len(f) else f
            name = f"+{lead * step_minutes // 60} h" if lead * step_minutes % 60 == 0 else f"+{lead * step_minutes} min"
            ok = len(hit) == 1 and "q50" in qs
            row[name] = float(hit["q50"].iloc[0]) if ok else float("nan")
            row[f"{name} low"] = float(hit[qs[0]].iloc[0]) if ok else float("nan")
            row[f"{name} high"] = float(hit[qs[-1]].iloc[0]) if ok else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def series_chart(table: pd.DataFrame, forecast: pd.DataFrame, target: str, *, hours: float) -> pd.DataFrame:
    """Long frame for one chart. Rows: timestamp, kind ('measured' | 'usual' | 'forecast'), value, low, high.
    'usual' is the mean at that clock time on earlier days, over both the shown history and the forecast window."""
    cd = chart_data(table, forecast, target, days=hours / 24)
    if cd.empty or target not in table.columns:
        return cd
    when = pd.DatetimeIndex(sorted(set(pd.to_datetime(cd["timestamp"]))))
    u = usual(pd.to_numeric(table[target], errors="coerce"), when, before=table.index.max()).dropna()
    extra = pd.DataFrame({"timestamp": u.index, "kind": "usual", "value": u.to_numpy(float),
                          "low": float("nan"), "high": float("nan")})
    return pd.concat([cd, extra], ignore_index=True)


def lead_history(conn: sqlite3.Connection, space_id: str, table: pd.DataFrame, target: str, step: int) -> pd.DataFrame:
    """Every stored forecast of `target` made `step` slots ahead, next to what was then measured.
    Columns: timestamp, forecast, low, high, measured (NaN until the time has passed). When several runs forecast
    the same moment at that lead, the newest run is used."""
    df = pd.read_sql_query(
        "SELECT f.run_id, f.ts_local, f.quantile, f.value FROM forecasts f JOIN forecast_runs r USING(run_id) "
        "WHERE r.space_id=? AND f.target=? AND f.step=?", conn, params=(space_id, target, step))
    cols = ["timestamp", "forecast", "low", "high", "measured"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[df["run_id"] == df.groupby("ts_local")["run_id"].transform("max")]
    wide = df.pivot_table(index="ts_local", columns="quantile", values="value")
    if 0.5 not in wide.columns:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({"timestamp": pd.to_datetime(wide.index), "forecast": wide[0.5].to_numpy(float),
                        "low": wide[wide.columns.min()].to_numpy(float), "high": wide[wide.columns.max()].to_numpy(float)})
    truth = pd.to_numeric(table[target], errors="coerce") if target in table.columns else pd.Series(dtype=float)
    out["measured"] = truth.reindex(out["timestamp"]).to_numpy(float)
    return out.sort_values("timestamp").reset_index(drop=True)


STAGES = [("captured", "capture_status='ok'"), ("read by detector", "processed_utc IS NOT NULL"),
          ("read by VLM", "vlm_raw IS NOT NULL AND vlm_raw NOT LIKE '{\"error\"%'"),
          ("outlined", "segmenter_version IS NOT NULL"),
          ("image vector", "EXISTS(SELECT 1 FROM frame_embeddings e WHERE e.frame_id=f.frame_id)"),
          ("description scores", "EXISTS(SELECT 1 FROM frame_scores c WHERE c.frame_id=f.frame_id)")]


def funnel(conn: sqlite3.Connection, space: SpaceConfig, table: pd.DataFrame) -> pd.DataFrame:
    """Photo slots from the first photo to the latest, and how many got through each stage. One photo per slot
    (the one the table uses), so the numbers add up to slots, not to raw frames."""
    ids = [int(i) for i in table["frame_id"].dropna()]
    slots = len(table)
    rows = [{"stage": "slots since first photo", "photos": slots, "share": 1.0}]
    marks = ",".join("?" * len(ids))
    for name, cond in STAGES:
        n = conn.execute(f"SELECT COUNT(*) FROM frames f WHERE frame_id IN ({marks}) AND {cond}", ids).fetchone()[0] if ids else 0
        rows.append({"stage": name, "photos": int(n), "share": n / slots if slots else float("nan")})
    return pd.DataFrame(rows)


def capture_grid(table: pd.DataFrame) -> pd.DataFrame:
    """One row per slot: day, minute of day, and status: 'ok', 'VLM failed', 'not read yet' (captured, waiting for
    the next run to read it) or 'no photo'."""
    status = pd.Series("ok", index=table.index)
    status[~table["vlm_ok"].astype(bool)] = "VLM failed"
    status[table["slot_status"] != "ok"] = "no photo"
    status[table["slot_status"] == "unprocessed"] = "not read yet"
    return pd.DataFrame({"day": table.index.strftime("%a %d %b"), "date": table.index.normalize(),
                         "time": table.index.strftime("%H:%M"), "minute": table.index.hour * 60 + table.index.minute,
                         "status": status.to_numpy()})


def gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive slots without a photo: start, end, slots. A photo that is captured but not read yet is
    not a gap."""
    bad = table.index[~table["slot_status"].isin(["ok", "unprocessed"])]
    if not len(bad):
        return pd.DataFrame(columns=["from", "to", "slots"])
    step = table.index[1] - table.index[0] if len(table) > 1 else pd.Timedelta(minutes=15)
    group = (pd.Series(bad).diff() != step).cumsum().to_numpy()
    runs = pd.Series(bad).groupby(group).agg(["min", "max", "size"])
    return runs.rename(columns={"min": "from", "max": "to", "size": "slots"}).reset_index(drop=True)


def tick_timing(path: Path, *, last: int = 96) -> dict:
    """Seconds each of the last `last` ticks took (first model load to 'tick done'), from the tick log."""
    if not path.exists():
        return {"ticks": 0}
    start, durations = None, []
    with path.open(errors="replace") as fh:
        for ln in fh:
            stamp = ln.split(" ", 1)[0]
            if "detector" in ln and " loaded " in ln:
                start = stamp
            elif "tick done" in ln and start is not None:
                try:
                    durations.append((pd.Timestamp(stamp) - pd.Timestamp(start)).total_seconds())
                except ValueError:
                    pass
                start = None
    d = pd.Series(durations[-last:], dtype=float)
    if d.empty:
        return {"ticks": 0}
    return {"ticks": len(d), "median_s": float(d.median()), "p95_s": float(d.quantile(0.95)), "max_s": float(d.max())}


def dimension_flow(catalog: pd.DataFrame, run: dict | None) -> pd.DataFrame:
    """Per column family: how many dimensions are extracted, how many are usable (model-ready), how many the latest
    forecast run gave to the model as inputs, and how many it forecast."""
    sent = set(run["covariates"]) if run else set()
    targets = set(run["targets"]) if run else set()
    cal_sent = "cal.*" in sent
    c = catalog.assign(
        sent=[(col in sent) or (kind == "cal" and cal_sent) for col, kind in zip(catalog["column"], catalog["kind"])],
        forecast=catalog["column"].isin(targets))
    g = c.groupby("kind").agg(extracted=("column", "size"), usable=("model_ready", "sum"),
                              given_as_input=("sent", "sum"), forecast=("forecast", "sum")).astype(int)
    g.insert(0, "what", [KIND_NAMES.get(k, k) for k in g.index])
    g = g.sort_values("extracted", ascending=False).reset_index().rename(columns={"kind": "family"})
    total = {"family": "total", "what": "", **{k: int(g[k].sum()) for k in ("extracted", "usable", "given_as_input", "forecast")}}
    return pd.concat([g, pd.DataFrame([total])], ignore_index=True)


def dimensions(table: pd.DataFrame, catalog: pd.DataFrame, run: dict | None, forecast: pd.DataFrame) -> pd.DataFrame:
    """One row per extracted dimension: latest value, usual value at this time, range, coverage, and what happens
    to it (forecast / given to the model / kept only in the table, with the reason)."""
    sent = set(run["covariates"]) if run else set()
    targets = set(run["targets"]) if run else set()
    next_hour = {}
    if not forecast.empty and "q50" in forecast.columns:
        next_hour = forecast[forecast["step"] == 4].set_index("target")["q50"].to_dict()
    last = table.iloc[-1] if len(table) else pd.Series(dtype=object)
    rows = []
    for r in catalog.itertuples(index=False):
        if r.kind == "cal":
            continue
        numeric = r.kind != "cat"
        s = pd.to_numeric(table[r.column], errors="coerce").dropna() if numeric else pd.Series(dtype=float)
        use = "forecast" if r.column in targets else "input to model" if r.column in sent else f"table only: {r.reason}"
        rows.append({"dimension": label(r.column), "column": r.column, "family": r.kind,
                     "now": float(last[r.column]) if numeric and pd.notna(last.get(r.column)) else None,
                     "now (text)": None if numeric or pd.isna(last.get(r.column)) else str(last[r.column]),
                     "usual now": float(usual(s, pd.DatetimeIndex([table.index[-1]]), before=table.index[-1]).iloc[0]) if len(s) else None,
                     "min": float(s.min()) if len(s) else None, "max": float(s.max()) if len(s) else None,
                     "+1 h forecast": next_hour.get(r.column), "coverage": r.coverage, "use": use})
    return pd.DataFrame(rows)


def header_status(conn: sqlite3.Connection, spaces: list[SpaceConfig], now_utc: datetime) -> list[dict]:
    """One entry per space for the page header: the latest photo that has been read (local time and age in minutes),
    what the detector counted in it, when the latest forecast was made, and a state word:
    'live' (photo at most 20 minutes old), 'late' (at most an hour), 'stopped' (older) or 'no photos'."""
    out = []
    for s in spaces:
        r = conn.execute(
            "SELECT f.ts_utc, c.people_yolo, c.vehicles_yolo FROM frames f LEFT JOIN scene_state c USING(frame_id) "
            "WHERE f.space_id=? AND f.capture_status='ok' AND f.processed_utc IS NOT NULL ORDER BY f.ts_utc DESC LIMIT 1",
            (s.space_id,)).fetchone()
        fc = conn.execute("SELECT origin_local FROM forecast_runs WHERE space_id=? ORDER BY run_id DESC LIMIT 1",
                          (s.space_id,)).fetchone()
        entry = {"space_id": s.space_id, "name": s.name, "photo": None, "age_min": None, "people": None, "vehicles": None,
                 "forecast": fc["origin_local"][11:16] if fc else None, "state": "no photos"}
        if r is not None:
            taken = pd.Timestamp(r["ts_utc"])
            age = (pd.Timestamp(now_utc) - taken).total_seconds() / 60
            entry.update(photo=taken.tz_convert(get_zone(s.tz)).strftime("%H:%M"), age_min=int(max(age, 0)),
                         people=None if r["people_yolo"] is None else int(r["people_yolo"]),
                         vehicles=None if r["vehicles_yolo"] is None or s.space_type != "parking" else int(r["vehicles_yolo"]),
                         state="live" if age <= 20 else "late" if age <= 60 else "stopped")
        out.append(entry)
    return out


def fixtures(table: pd.DataFrame) -> list[tuple[str, int]]:
    """(object type, latest steady count) for the things that do not move, most numerous first; types the room
    does not have (count 0) and types without a steady count yet are left out."""
    out = []
    for col in table.columns:
        if col.startswith("steady."):
            s = pd.to_numeric(table[col], errors="coerce").dropna()
            if len(s) and s.iloc[-1] > 0:
                out.append((col.split(".")[-1], int(s.iloc[-1])))
    return sorted(out, key=lambda x: (-x[1], x[0]))

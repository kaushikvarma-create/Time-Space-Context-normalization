"""Space Behaviour Intelligence: v1.0 dashboard (read-only).   Run:  scripts/dashboard.sh  (http://localhost:8502)

Shows whether every stage is running, the measured series with the latest stored forecast, and how good the
forecasts have been. All times are IST. The project root is the current directory, or SBI_ROOT if set.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from sbi.dashboard import data
from sbi.errors import SbiError
from sbi.forecast.accuracy import by_lead, by_target, live_points
from sbi.forecast.build import table_for
from sbi.settings import load_settings
from sbi.spaces import discover_space_files, load_space

st.set_page_config(layout="wide", page_title="SBI v1.0")


@st.cache_resource
def _setup():
    settings = load_settings(os.environ.get("SBI_ROOT"))
    spaces = {s.space_id: s for s in (load_space(f) for f in discover_space_files(settings.root))}
    return settings, spaces


@st.cache_data(ttl=120, show_spinner="Building the table ...")
def _table(space_id: str) -> pd.DataFrame:
    conn = data.connect_readonly(settings.db_path)
    try:
        return table_for(conn, settings, spaces[space_id]).table
    finally:
        conn.close()


try:
    settings, spaces = _setup()
    conn = data.connect_readonly(settings.db_path)
except (SbiError, OSError) as e:
    st.error(f"Setup problem: {e}")
    st.stop()

st.title("Space Behaviour Intelligence: v1.0")
st.caption("Photos every 15 minutes -> counts, outlines, scene descriptions, image vectors and probes -> "
           "Chronos-2 forecasts. All times IST. Read-only.")

st.subheader("System health (last 24 hours)")
st.dataframe(data.health(conn, list(spaces.values()), datetime.now(timezone.utc)), hide_index=True, width="stretch")
problems = data.log_problems(settings.resolve(settings.paths.logs) / "tick.log")
with st.expander(f"Recent warnings and errors in the tick log ({len(problems)})"):
    st.code("\n".join(problems) or "none")

space_id = st.sidebar.selectbox("Space", list(spaces))
days = st.sidebar.slider("Days of history", 1, 14, 3)
table = _table(space_id)
run, forecast = data.latest_forecast(conn, space_id)
choices = (run["targets"] if run else []) or [c for c in table.columns if c.startswith(("scene.", "count."))]
picked = st.sidebar.multiselect("Series", choices, default=choices[:4])

st.subheader(f"{space_id}: measured and forecast")
if run:
    st.caption(f"Latest forecast: run {run['run_id']} from {run['origin_local'].replace('T', ' ')} IST, "
               f"{run['model']}, covariate set '{run['variant']}', {run['horizon']} slots ahead; "
               f"band = {', '.join(c for c in forecast.columns if c.startswith('q'))} quantiles.")
else:
    st.info("No forecast stored yet for this space (the tick stores one every forecast.every_minutes, or run "
            "`sbi forecast`).")
for target in picked:
    cd = data.chart_data(table, forecast, target, days=days)
    if cd.empty:
        continue
    base = alt.Chart(cd).encode(x=alt.X("timestamp:T", title=None))
    band = base.transform_filter(alt.datum.kind == "forecast").mark_area(opacity=0.25).encode(
        y=alt.Y("low:Q", title=target), y2="high:Q")
    lines = base.mark_line().encode(y="value:Q", color=alt.Color("kind:N", legend=alt.Legend(orient="top")),
                                    strokeDash="kind:N")
    st.altair_chart((band + lines).properties(height=180), width="stretch")

st.subheader("Forecast accuracy")
points = live_points(conn, space_id, table)
if points.empty:
    st.info("No stored forecast has a measured outcome yet; they are scored once their time has passed.")
else:
    st.caption("Live: the forecasts actually stored, scored against what was later measured. Skill = 1 - MAE / MAE "
               "of holding the last value (positive is better). Coverage should be near the band's nominal level.")
    st.dataframe(by_lead(points).round(3), hide_index=True, width="stretch")
    with st.expander("Per series"):
        st.dataframe(by_target(points).round(3), hide_index=True, width="stretch")
bt = data.backtest_overall(settings.resolve("data/exports"), space_id)
if not bt.empty:
    st.caption("Backtest (`sbi backtest`): simulated forecasts from past moments; skill vs the weekday-by-time average.")
    st.dataframe(bt.round(3), hide_index=True, width="stretch")

if st.sidebar.checkbox("Show the latest photo", value=False):
    row = conn.execute("SELECT image_path, ts_local FROM frames WHERE space_id=? AND capture_status='ok' "
                       "ORDER BY ts_utc DESC LIMIT 1", (space_id,)).fetchone()
    if row:
        st.image(str(settings.root / row["image_path"]), caption=f"{space_id} at {row['ts_local'][:16]} IST")

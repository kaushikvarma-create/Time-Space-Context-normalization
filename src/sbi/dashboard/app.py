"""Space Behaviour Intelligence dashboard (read-only).   Run:  scripts/dashboard.sh  (http://localhost:8502)

A header and five tabs: what each space looks like now and in the next hours; how past forecasts compared with what happened;
whether every photo gets through every stage and how much of it reaches the forecasting model; every extracted
dimension with its latest value. All times are IST. The project root is the current directory, or SBI_ROOT if set.

The table behind a space is rebuilt only when a new photo or forecast arrives (data.table_stamp), so reloading and
the once-a-minute refresh are cheap.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from sbi.dashboard import data
from sbi.errors import SbiError
from sbi.forecast.accuracy import by_lead, by_target, by_weekday, live_points, outdated_runs
from sbi.forecast.build import table_for
from sbi.settings import load_settings
from sbi.spaces import discover_space_files, load_space

INK, MUTED, BLUE, GRID, SLATE = "#2f2f2c", "#a3a29a", "#2a78d6", "#e6e5e0", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]       # one colour per headline series: its tile and its chart
GOOD, WARN, BAD = "#0ca30c", "#eda100", "#d03b3b"           # reserved for state (captured / degraded / missing)
STATUS = {"ok": "#b5dfc0", "not read yet": "#8c8b84", "VLM failed": WARN, "no photo": BAD}
USE_TINT = {"forecast": "#dbe9fb", "input to model": "#d9f0e6", "table only": "#f1f0ec"}
HOURS = {"24 h": 24, "48 h": 48, "7 days": 168}
LEADS = {"1 h ahead": 4, "2 h ahead": 8, "4 h ahead": 16}

st.set_page_config(layout="wide", page_title="SBI", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1.4rem;padding-bottom:2rem} [data-testid='stMetricValue']"
            "{font-variant-numeric:tabular-nums}</style>", unsafe_allow_html=True)


@st.cache_resource
def _setup():
    settings = load_settings(os.environ.get("SBI_ROOT"))
    spaces = {s.space_id: s for s in (load_space(f) for f in discover_space_files(settings.root))}
    return settings, spaces


@st.cache_data(max_entries=4, show_spinner="Reading the database ...")
def _table(space_id: str, stamp: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(table, catalog). `stamp` is only the cache key: a new photo or forecast changes it."""
    conn = data.connect_readonly(settings.db_path)
    try:
        result = table_for(conn, settings, spaces[space_id])
        return result.table, result.catalog
    finally:
        conn.close()


@st.cache_data(max_entries=4, show_spinner=False)
def _points(space_id: str, stamp: tuple) -> pd.DataFrame:
    conn = data.connect_readonly(settings.db_path)
    try:
        return live_points(conn, space_id, _table(space_id, stamp)[0])
    finally:
        conn.close()


WHOLE_THINGS = ("scene.", "count.", "rollup.", "cell.")      # counts of objects: one decimal is plenty


def num(v, column: str = "") -> str:
    """A number at the precision worth reading: counts to one decimal, everything else to three figures."""
    if v is None or pd.isna(v):
        return "-"
    if column.startswith(WHOLE_THINGS) or abs(v) >= 100:
        return f"{v:.1f}".removesuffix(".0")
    return f"{v:.3g}"


def band(v, lo, hi, column: str = "") -> str:
    return "-" if pd.isna(v) else f"{num(v, column)}  ({num(lo, column)} to {num(hi, column)})"


KEY = ("<div style='color:#52514e;font-size:0.85rem;margin:0.2rem 0 0.4rem'>"
       "<b>&#9472;&#9472;</b> measured &nbsp;&nbsp; <b>&#9476;&#9476;</b> forecast, shaded = 10% to 90% &nbsp;&nbsp; "
       f"<span style='color:{MUTED}'><b>&middot;&middot;&middot;&middot;</b></span> usual at this time of day &nbsp;&nbsp; "
       f"<span style='color:{MUTED}'>|</span> latest photo</div>")


def tile(where, label_: str, value: str, note: str | None = None, color: str = SLATE) -> None:
    """A number with its name above and a note below, marked with a colour bar on the left."""
    where.markdown(
        f"<div style='border-left:4px solid {color};padding:0.1rem 0 0.2rem 0.75rem;margin-bottom:0.9rem'>"
        f"<div style='font-size:0.85rem;color:#52514e'>{label_}</div>"
        f"<div style='font-size:2rem;line-height:1.25;font-variant-numeric:tabular-nums'>{value}</div>"
        f"<div style='font-size:0.8rem;color:#52514e;min-height:1.1rem'>{note or ''}</div></div>", unsafe_allow_html=True)


STATE_DOT = {"live": "#35c46a", "late": "#f0b429", "stopped": "#ef6461", "no photos": "#8d99a6"}


def header(statuses: list[dict], clock: datetime) -> str:
    """The band at the top: what this is, the time, and one line per space saying whether it is live and what
    the latest photo showed."""
    def plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {'people' if word == 'person' else word + 's'}"
    cells = []
    for s in statuses:
        if s["photo"] is None:
            detail = "no photos yet"
        else:
            seen = [plural(s["vehicles"], "vehicle")] if s["vehicles"] is not None else []
            seen += [plural(s["people"], "person")] if s["people"] is not None else []
            detail = (f"photo {s['photo']} ({s['age_min']} min ago)" + (f" &middot; {', '.join(seen)}" if seen else "")
                      + (f" &middot; forecast {s['forecast']}" if s["forecast"] else ""))
        cells.append(
            f"<div style='margin-right:2.5rem'><span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
            f"background:{STATE_DOT[s['state']]};margin-right:0.45rem'></span><b>{s['name']}</b>"
            f"<span style='color:#aab4c0'> &nbsp;{s['state']}</span>"
            f"<div style='color:#d5dbe2;font-size:0.85rem;margin:0.1rem 0 0 1.05rem'>{detail}</div></div>")
    return ("<div style='background:#1d2733;color:#fff;border-radius:6px;padding:0.9rem 1.25rem 0.95rem;margin-bottom:0.6rem'>"
            "<div style='display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid #34414f;"
            "padding-bottom:0.55rem;margin-bottom:0.65rem'>"
            "<div><span style='font-size:1.3rem;font-weight:600;letter-spacing:0.01em'>Space Behaviour Intelligence</span>"
            "<span style='color:#aab4c0;margin-left:0.9rem;font-size:0.9rem'>a photo every 15 minutes, read into numbers, "
            "forecast 4 hours ahead</span></div>"
            f"<div style='color:#d5dbe2;font-variant-numeric:tabular-nums'>{clock:%a %d %b %Y, %H:%M} IST</div></div>"
            f"<div style='display:flex;flex-wrap:wrap'>{''.join(cells)}</div></div>")


def state_colour(share: float) -> str:
    return GOOD if share >= 0.98 else WARN if share >= 0.90 else BAD


def _plain(chart: alt.LayerChart, height: int) -> alt.LayerChart:
    """Shared look: quiet grid and axes, no frame, room on the right so the last time label is not cut."""
    return chart.properties(height=height, padding={"left": 4, "right": 28, "top": 4, "bottom": 4}) \
        .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor="#52514e", labelFontSize=11) \
        .configure_view(stroke=None)


def series_figure(cd: pd.DataFrame, title: str, now: pd.Timestamp, height: int = 240, color: str = BLUE) -> alt.Chart:
    """Measured (solid) and forecast (dashed, with its 10-90% band) in the series' colour; usual for the time of
    day as grey dots."""
    scale = alt.Scale(domain=["measured", "usual", "forecast"], range=[color, MUTED, color])
    dash = alt.Scale(domain=["measured", "usual", "forecast"], range=[[1, 0], [1, 3], [5, 3]])
    base = alt.Chart(cd).encode(x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%a %H:%M", labelAngle=0, tickCount=6)))
    area = base.transform_filter(alt.datum.kind == "forecast").mark_area(opacity=0.16, color=color).encode(
        y=alt.Y("low:Q", title=None), y2="high:Q")
    lines = base.mark_line(strokeWidth=1.8).encode(
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=True)),
        color=alt.Color("kind:N", scale=scale, legend=None),
        strokeDash=alt.StrokeDash("kind:N", scale=dash, legend=None))
    hover = base.mark_circle(size=70, opacity=0).encode(
        y="value:Q", tooltip=[alt.Tooltip("timestamp:T", format="%a %d %b %H:%M", title="time"), alt.Tooltip("kind:N", title=" "),
                              alt.Tooltip("value:Q", format=".3g"), alt.Tooltip("low:Q", format=".3g"),
                              alt.Tooltip("high:Q", format=".3g")])
    rule = alt.Chart(pd.DataFrame({"timestamp": [now]})).mark_rule(color=MUTED, strokeWidth=1).encode(x="timestamp:T")
    return _plain((area + rule + lines + hover).properties(title=alt.TitleParams(title, anchor="start", fontSize=13)), height)


def lead_figure(h: pd.DataFrame, title: str) -> alt.Chart:
    long = pd.concat([h.assign(kind="measured", value=h["measured"])[["timestamp", "kind", "value"]],
                      h.assign(kind="forecast", value=h["forecast"])[["timestamp", "kind", "value"]]], ignore_index=True).dropna()
    scale = alt.Scale(domain=["measured", "forecast"], range=[INK, BLUE])
    x = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%a %H:%M", labelAngle=0, tickCount=8))
    area = alt.Chart(h).mark_area(opacity=0.18, color=BLUE).encode(x=x, y=alt.Y("low:Q", title=None), y2="high:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=1.8).encode(
        x=x, y=alt.Y("value:Q", title=None), color=alt.Color("kind:N", scale=scale, legend=alt.Legend(orient="top", title=None)))
    hover = alt.Chart(h).mark_circle(size=70, opacity=0).encode(
        x=x, y="forecast:Q", tooltip=[alt.Tooltip("timestamp:T", format="%a %d %b %H:%M", title="time"),
                                      alt.Tooltip("measured:Q", format=".3g"), alt.Tooltip("forecast:Q", format=".3g"),
                                      alt.Tooltip("low:Q", format=".3g"), alt.Tooltip("high:Q", format=".3g")])
    return _plain((area + lines + hover).properties(title=alt.TitleParams(title, anchor="start", fontSize=13)), 340)


def grid_figure(g: pd.DataFrame) -> alt.Chart:
    days = list(dict.fromkeys(g.sort_values("date")["day"]))
    return alt.Chart(g).mark_rect(stroke="#fcfcfb", strokeWidth=1).encode(
        x=alt.X("minute:O", title="time of day", axis=alt.Axis(
            values=list(range(0, 1440, 180)), labelExpr="floor(datum.value / 60) + ':00'", labelAngle=0)),
        y=alt.Y("day:N", sort=days, title=None),
        color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS), range=list(STATUS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["day:N", "time:N", "status:N"]).properties(height=120 + 34 * len(days)).configure_view(stroke=None) \
        .configure_axis(domainColor=GRID, tickColor=GRID, labelColor="#52514e", labelFontSize=11)


def headline(space, run: dict | None, table: pd.DataFrame) -> list[str]:
    """Series shown first: the targets named outright in the space file, then the busiest counted types."""
    targets = run["targets"] if run else [c for c in table.columns if c.startswith("scene.")]
    named = [t for t in space.forecast_targets if t in targets]
    counts = [t for t in targets if t.startswith("count.") and t not in named]
    counts.sort(key=lambda c: -float(pd.to_numeric(table[c], errors="coerce").mean()) if c in table.columns else 0.0)
    return (named + counts)[:4] or targets[:4]


try:
    settings, spaces = _setup()
    data.connect_readonly(settings.db_path).close()            # fail here, with a clear message, if it cannot open
except (SbiError, OSError) as e:
    st.error(f"Setup problem: {e}")
    st.stop()


# A link can open a given space and tab:  ?space=parking&tab=pipeline   (tab: now | accuracy | pipeline | all | guide)
TABS = ["Now and next 4 hours", "Forecast vs actual", "Pipeline", "All dimensions", "Guide"]
GUIDE = Path(__file__).resolve().parents[3] / "docs" / "dashboard-guide.md"
_q = st.query_params
first_space = list(spaces).index(_q["space"]) if _q.get("space") in spaces else 0
first_tab = {"now": 0, "accuracy": 1, "pipeline": 2, "all": 3, "guide": 4}.get(_q.get("tab", "now"), 0)


@st.fragment(run_every=60)
def page() -> None:
    """One refresh. The connection is opened here, not once for the session: Streamlit runs the timed refresh in a
    different thread from the first page load, and an SQLite connection may only be used in the thread that made it."""
    conn = data.connect_readonly(settings.db_path)
    try:
        _page(conn)
    finally:
        conn.close()


def _page(conn) -> None:
    clock = datetime.now(timezone.utc)
    local = clock.astimezone(table_tz(next(iter(spaces.values()))))
    st.markdown(header(data.header_status(conn, list(spaces.values()), clock), local), unsafe_allow_html=True)
    top = st.columns([3, 2])
    space_id = top[0].radio("Space", list(spaces), index=first_space, horizontal=True, label_visibility="collapsed",
                            format_func=lambda s: spaces[s].name)
    space = spaces[space_id]
    stamp = data.table_stamp(conn, space_id)
    if not stamp[0]:
        st.info(f"No photos stored yet for {space.name}.")
        return
    try:
        table, catalog = _table(space_id, stamp)
    except SbiError as e:
        st.error(f"Could not read {space.name}: {e}")
        return
    run, forecast = data.latest_forecast(conn, space_id)
    now = table.index.max()
    top[1].markdown(f"<div style='text-align:right;color:#52514e;font-size:0.85rem'>showing <b>{space.name}</b> up to "
                    f"{now:%a %d %b %H:%M} &nbsp;|&nbsp; refreshes every minute, last at {local:%H:%M:%S}</div>",
                    unsafe_allow_html=True)
    t_now, t_acc, t_pipe, t_all, t_guide = st.tabs(TABS, default=TABS[first_tab])
    with t_guide:
        st.markdown(GUIDE.read_text() if GUIDE.exists() else "The guide file `docs/dashboard-guide.md` is missing.")

    targets = run["targets"] if run else []
    with t_now:
        heads = headline(space, run, table)
        left, right = st.columns([2, 3])
        row = conn.execute("SELECT image_path FROM frames WHERE space_id=? AND capture_status='ok' "
                           "ORDER BY ts_utc DESC LIMIT 1", (space_id,)).fetchone()
        if row and (settings.root / row["image_path"]).exists():
            left.image(str(settings.root / row["image_path"]), width="stretch")
        sm = data.summary(table, forecast, heads)
        tiles = right.columns(max(len(sm), 1))
        colour = {t: SERIES[i] for i, t in enumerate(heads[:len(SERIES)])}      # fixed per space, not per selection
        for where, r in zip(tiles, sm.itertuples(index=False)):
            note = None if pd.isna(r.usual) else f"{'+' if r.now >= r.usual else ''}{num(r.now - r.usual, r.column)} vs usual {num(r.usual, r.column)}"
            tile(where, r.series, num(r.now, r.column), note, colour.get(r.column, SLATE))
        allsm = data.summary(table, forecast, targets or heads)
        if not allsm.empty:
            cols_ = allsm["column"]
            show = pd.DataFrame({"series": allsm["series"], "now": [num(v, c_) for v, c_ in zip(allsm["now"], cols_)],
                                 "usual at this time": [num(v, c_) for v, c_ in zip(allsm["usual"], cols_)]})
            for lead in ("+1 h", "+2 h", "+4 h"):
                if lead in allsm:
                    show[f"{lead}  (10% to 90%)"] = [band(v, lo, hi, c_) for v, lo, hi, c_ in zip(allsm[lead], allsm[f"{lead} low"], allsm[f"{lead} high"], cols_)]
            right.dataframe(show, hide_index=True, width="stretch", height=250)
        if not run:
            st.info("No forecast stored yet for this space.")
        ctl = st.columns([3, 1])
        picked = ctl[0].multiselect("Series", targets or heads, default=heads, format_func=data.label, key=f"pick_{space_id}")
        hours = HOURS[ctl[1].radio("History", list(HOURS), horizontal=True, key="hours")]
        st.markdown(KEY, unsafe_allow_html=True)
        cols = st.columns(2)
        for i, target in enumerate(picked):
            cd = data.series_chart(table, forecast, target, hours=hours)
            if not cd.empty:
                cols[i % 2].altair_chart(series_figure(cd, data.label(target), now, color=colour.get(target, SLATE)), width="stretch")

    with t_acc:
        if not targets:
            st.info("No forecast stored yet for this space.")
        else:
            ctl = st.columns([3, 1])
            target = ctl[0].selectbox("Series", targets, format_func=data.label, key=f"acc_{space_id}")
            lead_name = ctl[1].radio("Forecast made", list(LEADS), horizontal=True, key="lead")
            old_runs = outdated_runs(conn, space_id)
            if old_runs:
                st.caption(f"{len(old_runs)} earlier forecast runs are not scored here: the photos behind them have since "
                           "been read again (a detector change), so those forecasts predicted readings that no longer "
                           "exist. Scoring starts again from the first forecast after the re-read.")
            h = data.lead_history(conn, space_id, table, target, LEADS[lead_name], skip_runs=old_runs)
            done = h.dropna(subset=["measured"])
            if done.empty:
                st.info("No forecast at this lead has a measured outcome yet.")
            else:
                err = (done["forecast"] - done["measured"]).abs()
                inside = ((done["measured"] >= done["low"]) & (done["measured"] <= done["high"])).mean()
                m = st.columns(4)
                tile(m[0], "Forecasts checked", str(len(done)))
                tile(m[1], "Average miss", f"{err.mean():.2g}", color=BLUE)
                tile(m[2], "Average measured value", f"{done['measured'].mean():.2g}", color=INK)
                tile(m[3], "Measured inside the shaded range", f"{inside:.0%}", "expected about 80%", BLUE)
                st.altair_chart(lead_figure(h, f"{data.label(target)}: forecast made {lead_name} vs what was measured"), width="stretch")
            points = _points(space_id, stamp)
            if not points.empty:
                wd = by_weekday(points, target)
                if not wd.empty:
                    st.markdown(f"**{data.label(target)} by day of week** (the day that was predicted, every lead time "
                                "together). A day where the forecast average sits far from the measured average is a "
                                "day the model does not understand yet.")
                    st.dataframe(wd.round(2), hide_index=True, width="stretch", column_config={
                        "inside band": st.column_config.NumberColumn(format="percent")})
                st.markdown("**All series together.** Skill = how much smaller the miss is than assuming nothing "
                            "changes (0 = no better, 1 = perfect, negative = worse). Coverage = share of measured "
                            "values inside the 10% to 90% band.")
                lead_t = by_lead(points).drop(columns=["model"]).rename(columns={
                    "variant": "inputs given", "lead": "how far ahead", "runs": "forecast runs", "targets": "series",
                    "points": "forecasts checked", "coverage": "inside band"})
                st.dataframe(lead_t.round(3), hide_index=True, width="stretch")
                st.markdown("**All series together, by day of week.**")
                st.dataframe(by_weekday(points).round(3), hide_index=True, width="stretch")
                per = by_target(points)
                per.insert(0, "series", per["target"].map(data.label))
                per = per.drop(columns=["model", "target"]).rename(columns={
                    "variant": "inputs given", "n": "forecasts checked", "mae": "average miss",
                    "persist_mae": "miss if nothing changed", "coverage": "inside band"})
                st.dataframe(per.round(3), hide_index=True, width="stretch", height=320)
            bt = data.backtest_overall(settings.resolve("data/exports"), space_id)
            if not bt.empty:
                st.markdown("**Backtest** (`sbi backtest`): forecasts replayed from past moments, skill vs the weekday-by-time average.")
                st.dataframe(bt.round(3), hide_index=True, width="stretch")

    with t_pipe:
        fn = data.funnel(conn, space, table)
        flow = data.dimension_flow(catalog, run)
        total = flow.iloc[-1]
        timing = data.tick_timing(settings.resolve(settings.paths.logs) / "tick.log")
        m = st.columns(5)
        tile(m[0], "Photos captured", f"{fn['photos'][1]} / {fn['photos'][0]}", f"{fn['share'][1]:.1%} of slots",
             state_colour(fn["share"][1]))
        tile(m[1], "Numbers per photo", str(int(total["extracted"])))
        tile(m[2], "Usable series", str(int(total["usable"])), "seen often enough, not constant")
        tile(m[3], "Given to Chronos-2", str(int(total["given_as_input"]) + int(total["forecast"])),
             f"{int(total['forecast'])} forecast, {int(total['given_as_input'])} inputs", BLUE)
        tile(m[4], "Time per 15-min run", f"{timing['median_s']:.0f} s" if timing.get("ticks") else "-",
             f"slowest {timing['max_s']:.0f} s of last {timing['ticks']}" if timing.get("ticks") else None)
        a, b = st.columns([2, 3])
        a.markdown("**Photos through each stage**")
        a.dataframe(fn, hide_index=True, width="stretch", column_config={
            "share": st.column_config.ProgressColumn("share of slots", min_value=0, max_value=1, format="percent")})
        gp = data.gaps(table)
        a.markdown(f"**Slots without a photo: {int(gp['slots'].sum()) if len(gp) else 0}**")
        if len(gp):
            a.dataframe(gp.assign(**{"from": gp["from"].dt.strftime("%a %d %b %H:%M"), "to": gp["to"].dt.strftime("%a %d %b %H:%M")}),
                        hide_index=True, width="stretch")
        b.markdown("**What is extracted and where it goes** (latest forecast run"
                   + (f", {run['origin_local'].replace('T', ' ')[:16]}, inputs '{run['variant']}'" if run else "") + ")")
        shown = flow.rename(columns={"given_as_input": "input to model"})
        b.dataframe(shown.style.map(lambda v: f"background-color: {USE_TINT['input to model']}" if v else "", subset=["input to model"])
                    .map(lambda v: f"background-color: {USE_TINT['forecast']}" if v else "", subset=["forecast"]),
                    hide_index=True, width="stretch", height=495)
        st.markdown("**Every 15-minute slot**")
        st.altair_chart(grid_figure(data.capture_grid(table)), width="stretch")
        problems = data.log_problems(settings.resolve(settings.paths.logs) / "tick.log")
        st.markdown(f"**Warnings and errors in the run log: {len(problems)}**")
        if problems:
            st.code("\n".join(problems))

    with t_all:
        dims = data.dimensions(table, catalog, run, forecast)
        for col in ("now", "usual now", "min", "max", "+1 h forecast"):
            dims[col] = pd.to_numeric(dims[col], errors="coerce").map(lambda v: v if pd.isna(v) else float(f"{v:.4g}"))
        dims["now (text)"] = dims["now (text)"].fillna("")
        c = st.columns([1, 1, 2])
        fam = c[0].multiselect("Family", sorted(dims["family"].unique()), key=f"fam_{space_id}",
                               format_func=lambda k: data.KIND_NAMES.get(k, k))
        use = c[1].multiselect("Use", ["forecast", "input to model", "table only"], key="use")
        text = c[2].text_input("Search", key="search", placeholder="chair, motorcycle, meeting ...")
        view = dims
        if fam:
            view = view[view["family"].isin(fam)]
        if use:
            view = view[view["use"].str.split(":").str[0].isin(use)]
        if text:
            view = view[view["dimension"].str.contains(text, case=False, regex=False) | view["column"].str.contains(text, case=False, regex=False)]
        st.caption(f"{len(view)} of {len(dims)} dimensions")
        tinted = view.style.map(lambda v: f"background-color: {USE_TINT[str(v).split(':')[0]]}", subset=["use"])
        st.dataframe(tinted, hide_index=True, width="stretch", height=420, column_config={
            "now": st.column_config.Column(width="small"),
            "coverage": st.column_config.ProgressColumn("coverage", min_value=0, max_value=1, format="percent")})
        plottable = [c_ for c_ in view["column"] if view.set_index("column").loc[c_, "family"] != "cat"]
        if plottable:
            one = st.selectbox("Plot", plottable, format_func=lambda c_: f"{data.label(c_)}   [{c_}]", key=f"plot_{space_id}")
            cd = data.series_chart(table, forecast, one, hours=168)
            if not cd.empty:
                st.markdown(KEY, unsafe_allow_html=True)
                st.altair_chart(series_figure(cd, data.label(one), now, height=300), width="stretch")


def table_tz(space):
    from sbi.timeutil import get_zone
    return get_zone(space.tz)


page()

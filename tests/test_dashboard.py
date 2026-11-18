"""Dashboard: the data it shows (health, log problems, charts, latest forecast) and a smoke run of the real app."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from sbi.dashboard import data
from sbi.pipeline import process_frame
from sbi.store.db import connect, migrate
from sbi.store.embeddings import put_embedding
from sbi.store.forecasts import insert_forecasts, insert_run
from sbi.store.registry import register_space

from .conftest import REPO, FakeDetector, FakeReader, FakeSource, synthetic_image, thursday_ist
from .test_flow import DETS, cap
from .test_forecast import make_table


def test_health_counts_the_last_24_hours_per_stage(conn, space, parking, root):
    now = thursday_ist(12, 0)
    for m in (0, 15, 30):
        fid = cap(conn, space, root, FakeSource([synthetic_image(m)]), thursday_ist(11, m)).frame_id
        process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root)
        if m == 0:
            put_embedding(conn, fid, "m", np.ones(4, np.float32))
    from sbi.errors import CameraError
    cap(conn, space, root, FakeSource([CameraError("down")]), thursday_ist(11, 45))
    cap(conn, space, root, FakeSource([synthetic_image()]), now - timedelta(days=2))                   # too old to count
    insert_run(conn, space_id="admin_cabin", origin_local="2026-11-12T11:30:00", model="chronos2:x", variant="calendar",
               horizon=4, step_minutes=15, context_slots=100, targets=["a"], covariates=[], quantiles=[0.1, 0.5, 0.9])
    h = data.health(conn, [space, parking], now).set_index("space")
    cab = h.loc["admin_cabin"]
    assert cab["photos 24 h"] == "3 / 96" and cab["failed captures"] == 1 and cab["processed"] == "100%"
    assert cab["image vector"] == "33%" and cab["last photo (IST)"] == "2026-11-12 11:45"
    assert cab["latest forecast (IST)"] == "2026-11-12 11:30" and cab["forecaster"] == "chronos2:x"
    assert h.loc["parking", "photos 24 h"] == "0 / 96" and h.loc["parking", "processed"] == "-"


def test_log_problems_keeps_only_warnings_and_errors_newest_last(tmp_path):
    log = tmp_path / "tick.log"
    log.write_text("a INFO ok\nb WARNING one\nc ERROR two\nd INFO fine\ne WARNING three\n")
    assert data.log_problems(log) == ["b WARNING one", "c ERROR two", "e WARNING three"]
    assert data.log_problems(log, limit=1) == ["e WARNING three"] and data.log_problems(tmp_path / "none") == []


def test_chart_data_joins_measured_history_and_the_forecast_band():
    table = make_table(200)
    ts = pd.date_range(table.index[-1] + timedelta(minutes=15), periods=3, freq="15min")
    fc = pd.DataFrame({"target": "scene.people_yolo", "timestamp": ts, "step": [1, 2, 3],
                       "q10": [1.0, 1.0, 1.0], "q50": [2.0, 2.0, 2.0], "q90": [3.0, 3.0, 3.0]})
    cd = data.chart_data(table, fc, "scene.people_yolo", days=1)
    measured, forecast = cd[cd["kind"] == "measured"], cd[cd["kind"] == "forecast"]
    assert len(measured) == 97 and measured["value"].iloc[-1] == 199.0                 # one day of 15-minute slots
    assert forecast[["value", "low", "high"]].iloc[0].tolist() == [2.0, 1.0, 3.0]
    assert data.chart_data(table, fc, "nope", days=1).empty
    assert (data.chart_data(table, pd.DataFrame(), "scene.people_yolo", days=1)["kind"] == "measured").all()


def test_latest_forecast_and_the_read_only_connection(root, conn):
    rid = insert_run(conn, space_id="admin_cabin", origin_local="2026-11-12T10:00:00", model="m", variant="calendar",
                     horizon=1, step_minutes=15, context_slots=10, targets=["a"], covariates=["cal.*"], quantiles=[0.1, 0.5, 0.9])
    insert_forecasts(conn, rid, pd.DataFrame({"timestamp": [pd.Timestamp("2026-11-12 10:15")], "target": "a", "step": 1,
                                              "q10": 1.0, "q50": 2.0, "q90": 3.0}))
    ro = data.connect_readonly(root / "t.db")
    run, f = data.latest_forecast(ro, "admin_cabin")
    assert run["run_id"] == rid and run["targets"] == ["a"] and f["q50"].tolist() == [2.0]
    none_run, none_frame = data.latest_forecast(ro, "parking")
    assert none_run is None and none_frame.empty
    import sqlite3
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro.execute("DELETE FROM forecast_runs")                                         # the dashboard cannot write


def test_the_dashboard_app_runs_end_to_end(root, space_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    (root / "config" / "pipeline.yaml").write_text((REPO / "config" / "pipeline.yaml").read_text())
    c = connect(root / "data" / "sbi.db")
    migrate(c)
    from sbi.spaces import load_space
    spaces = {s: load_space(root / "config" / "spaces" / f"{s}.yaml") for s in ("admin_cabin", "parking")}
    for s, sp in spaces.items():
        register_space(c, sp, f"config/spaces/{s}.yaml")
    sp = spaces["admin_cabin"]
    for i in range(12):
        fid = cap(c, sp, root, FakeSource([synthetic_image(i)]), thursday_ist(9, 0) + timedelta(minutes=15 * i)).frame_id
        process_frame(c, sp, fid, FakeDetector(DETS), FakeReader(), root=root)
    rid = insert_run(c, space_id="admin_cabin", origin_local="2026-11-12T10:00:00", model="chronos2:x", variant="calendar",
                     horizon=2, step_minutes=15, context_slots=5, targets=["scene.people_yolo"], covariates=["cal.*"],
                     quantiles=[0.1, 0.5, 0.9])
    insert_forecasts(c, rid, pd.DataFrame({"timestamp": pd.to_datetime(["2026-11-12 10:15", "2026-11-12 10:30"]),
                                           "target": "scene.people_yolo", "step": [1, 2], "q10": 1.0, "q50": 2.0, "q90": 3.0}))
    c.close()
    (root / "data" / "logs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "logs" / "tick.log").write_text("x WARNING tick skipped\n")
    monkeypatch.setenv("SBI_ROOT", str(root))
    at = AppTest.from_file(str(REPO / "src" / "sbi" / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Now and next 4 hours", "Forecast vs actual", "Pipeline", "All dimensions", "Guide"]
    page_text = " ".join(m.value for m in at.markdown)
    assert "Space Behaviour Intelligence" in page_text and "Admin cabin</b>" in page_text       # the header band
    assert "no photos yet" in page_text                                                         # parking has none
    assert "# Dashboard guide" in page_text and "Families of dimensions" in page_text           # the guide tab
    tiles = " ".join(m.value for m in at.markdown)
    assert "people (detector)" in tiles and "Given to Chronos-2" in tiles and "12 / 12" in tiles
    assert tiles.index("Photos captured") < tiles.index("12 / 12") < tiles.index("Numbers per photo")
    assert len(at.dataframe) >= 4                              # forecast table, stages, dimension flow, all dimensions
    kinds = []
    def walk(node):
        for child in (node.children.values() if isinstance(getattr(node, "children", None), dict) else []):
            kinds.append(getattr(child, "type", ""))
            walk(child)
    walk(at.main)
    assert kinds.count("vega_lite_chart") >= 3                 # series chart, capture grid, the dimension plot
    at.radio[0].set_value("parking").run()                     # a space with no photos stops cleanly, no traceback
    assert not at.exception and any("No photos stored yet for Parking" in i.value for i in at.info)


def test_the_app_opens_its_database_connection_inside_each_refresh():
    """The timed refresh runs in another thread than the first load; a connection made once for the session then
    fails with 'SQLite objects created in a thread can only be used in that same thread'."""
    import ast
    tree = ast.parse((REPO / "src" / "sbi" / "dashboard" / "app.py").read_text())
    def assigns_connection(node):
        return isinstance(node, ast.Assign) and "connect_readonly" in ast.unparse(node.value)
    assert not [n for n in tree.body if assigns_connection(n)                         # nothing kept at module level
                or (isinstance(n, ast.Try) and any(assigns_connection(c) for c in n.body))]
    page = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "page")
    assert any(assigns_connection(n) for n in page.body) and "conn.close()" in ast.unparse(page)


# ---------------------------------------------------------------- the numbers behind each tab

def _forecast(table, target, values, step0=1):
    ts = pd.date_range(table.index[-1] + timedelta(minutes=15), periods=len(values), freq="15min")
    return pd.DataFrame({"target": target, "timestamp": ts, "step": range(step0, step0 + len(values)),
                         "q10": [v - 1 for v in values], "q50": values, "q90": [v + 1 for v in values]})


def test_header_status_says_whether_each_space_is_live(conn, space, parking, root):
    fid = cap(conn, space, root, FakeSource([synthetic_image()]), thursday_ist(11, 0)).frame_id
    process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root)
    cap(conn, space, root, FakeSource([synthetic_image(1)]), thursday_ist(11, 15))                # captured, not read
    insert_run(conn, space_id="admin_cabin", origin_local="2026-11-12T11:00:00", model="m", variant="all", horizon=1,
               step_minutes=15, context_slots=1, targets=["a"], covariates=[], quantiles=[0.5])
    people = conn.execute("SELECT people_yolo FROM scene_state WHERE frame_id=?", (fid,)).fetchone()[0]
    at = lambda minutes: {s["space_id"]: s for s in data.header_status(conn, [space, parking], thursday_ist(11, 0) + timedelta(minutes=minutes))}
    cab = at(10)["admin_cabin"]
    assert (cab["photo"], cab["age_min"], cab["state"], cab["forecast"]) == ("11:00", 10, "live", "11:00")   # the read photo
    assert cab["people"] == people and cab["vehicles"] is None                                    # vehicles: parking only
    assert at(20)["admin_cabin"]["state"] == "live" and at(21)["admin_cabin"]["state"] == "late"
    assert at(60)["admin_cabin"]["state"] == "late" and at(61)["admin_cabin"]["state"] == "stopped"
    park = at(10)["parking"]
    assert (park["state"], park["photo"], park["forecast"], park["name"]) == ("no photos", None, None, "Parking area")


def test_labels_read_like_words():
    assert data.label("count.yolo_c05.chair") == "chair count (detector, confidence 0.05+)"
    assert data.label("area.sam2_c10.person") == "person area share (outline, confidence 0.10+)"
    assert data.label("scene.people_yolo") == "people (detector)" and data.label("scene.count_disagree") == "count disagree"
    assert data.label("count.vlm.motorcycle") == "motorcycle count (VLM)"
    assert data.label("area.sam2.chair") == "chair area share (outline)"
    assert data.label("cellarea.sam2.c3r2.chair") == "chair area in cell c3r2 (outline)"
    assert data.label("rel.vlm.person.sitting_on.chair") == "person sitting on chair (VLM)"
    assert data.label("probe.siglip2.screens_on") == "'screens on' score" and data.label("odd") == "odd"


def test_usual_is_the_mean_at_the_same_clock_time_on_earlier_days_only():
    idx = pd.date_range("2026-11-10 00:00", periods=96 * 3, freq="15min")
    s = pd.Series(np.repeat([10.0, 20.0, 99.0], 96), index=idx)                       # day 1 = 10, day 2 = 20, today = 99
    when = pd.DatetimeIndex(["2026-11-12 09:00", "2026-11-13 09:00"])
    assert data.usual(s, when, before=idx[-1]).tolist() == [15.0, 15.0]               # today is left out
    assert data.usual(s, when).tolist() == [43.0, 43.0]                               # without `before`, every day counts
    assert np.isnan(data.usual(s.iloc[:4], pd.DatetimeIndex(["2026-11-12 09:00"])).iloc[0])


def test_summary_gives_now_usual_and_the_forecast_at_each_lead():
    table = make_table(200)
    fc = _forecast(table, "scene.people_yolo", [float(v) for v in range(300, 316)])
    sm = data.summary(table, fc, ["scene.people_yolo", "count.yolo.chair", "nope"]).set_index("column")
    row = sm.loc["scene.people_yolo"]
    assert list(sm.index) == ["scene.people_yolo", "count.yolo.chair"]                 # unknown columns are skipped
    assert row["now"] == 199.0 and row["usual"] == pytest.approx((7 + 103) / 2)        # 01:45 on the two earlier days
    assert (row["+1 h"], row["+1 h low"], row["+1 h high"]) == (303.0, 302.0, 304.0)   # step 4
    assert row["+2 h"] == 307.0 and row["+4 h"] == 315.0 and row["series"] == "people (detector)"
    assert np.isnan(sm.loc["count.yolo.chair", "+1 h"])                                # no forecast for it


def test_series_chart_adds_the_usual_line_over_history_and_forecast():
    table = make_table(200)
    cd = data.series_chart(table, _forecast(table, "scene.people_yolo", [1.0, 2.0]), "scene.people_yolo", hours=24)
    assert cd.groupby("kind").size().to_dict() == {"forecast": 2, "measured": 97, "usual": 99}
    assert set(cd.loc[cd["kind"] == "usual", "timestamp"]) == set(cd.loc[cd["kind"] != "usual", "timestamp"])
    assert data.series_chart(table, pd.DataFrame(), "nope", hours=24).empty


def test_lead_history_pairs_each_forecast_with_what_was_measured(conn):
    table = make_table(200)
    def run(origin, values):
        rid = insert_run(conn, space_id="admin_cabin", origin_local=origin, model="m", variant="all", horizon=len(values),
                         step_minutes=15, context_slots=10, targets=["scene.people_yolo"], covariates=[], quantiles=[0.1, 0.5, 0.9])
        ts = pd.date_range(pd.Timestamp(origin) + timedelta(minutes=15), periods=len(values), freq="15min")
        insert_forecasts(conn, rid, pd.DataFrame({"timestamp": ts, "target": "scene.people_yolo", "step": range(1, len(values) + 1),
                                                  "q10": [v - 1 for v in values], "q50": values, "q90": [v + 1 for v in values]}))
    run("2026-11-12T10:00:00", [50.0, 51.0])
    run("2026-11-12T10:00:00", [70.0, 71.0])                                          # the same moment again: newest wins
    run("2026-11-14T01:30:00", [60.0, 61.0])                                          # step 2 lands after the table ends
    h = data.lead_history(conn, "admin_cabin", table, "scene.people_yolo", 2)
    assert h["timestamp"].astype(str).tolist() == ["2026-11-12 10:30:00", "2026-11-14 02:00:00"]
    assert h.loc[0, ["forecast", "low", "high", "measured"]].tolist() == [71.0, 70.0, 72.0, 42.0]
    assert np.isnan(h.loc[1, "measured"])                                             # not happened yet
    assert data.lead_history(conn, "parking", table, "scene.people_yolo", 2).empty
    assert data.lead_history(conn, "admin_cabin", table, "scene.people_yolo", 9).empty


def test_capture_grid_gaps_and_stamp(conn, space, root):
    table = make_table(10)
    table.loc[table.index[[2, 3, 7]], "slot_status"] = "no_frame"
    table.loc[table.index[5], "vlm_ok"] = False
    table.loc[table.index[9], ["slot_status", "vlm_ok"]] = ["unprocessed", False]     # captured, waiting to be read
    g = data.capture_grid(table)
    assert g["status"].value_counts().to_dict() == {"ok": 5, "no photo": 3, "VLM failed": 1, "not read yet": 1}
    assert g.loc[3, ["time", "minute"]].tolist() == ["00:45", 45]
    gp = data.gaps(table)
    assert gp["slots"].tolist() == [2, 1] and str(gp.loc[0, "from"]) == "2026-11-12 00:30:00" and str(gp.loc[0, "to"]) == "2026-11-12 00:45:00"
    assert data.gaps(make_table(5)).empty
    assert data.table_stamp(conn, "admin_cabin") == (0, 0, 0, "")
    fid = cap(conn, space, root, FakeSource([synthetic_image()]), thursday_ist(9, 0)).frame_id
    assert data.table_stamp(conn, "admin_cabin") == (fid, 0, 0, "")                    # captured, not yet read
    process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root)
    rid = insert_run(conn, space_id="admin_cabin", origin_local="2026-11-12T09:00:00", model="m", variant="all", horizon=1,
                     step_minutes=15, context_slots=1, targets=["a"], covariates=[], quantiles=[0.5])
    read_at = conn.execute("SELECT processed_utc FROM frames WHERE frame_id=?", (fid,)).fetchone()[0]
    assert data.table_stamp(conn, "admin_cabin") == (fid, 1, rid, read_at)
    conn.execute("UPDATE frames SET processed_utc='2030-01-01T00:00:00+00:00' WHERE frame_id=?", (fid,))
    assert data.table_stamp(conn, "admin_cabin")[3] == "2030-01-01T00:00:00+00:00"     # a re-read photo changes it too


def test_funnel_counts_one_photo_per_slot_through_each_stage(conn, space, root):
    from sbi.forecast.table import build_table
    for m in (0, 15, 45):                                                              # 09:30 has no photo
        fid = cap(conn, space, root, FakeSource([synthetic_image(m)]), thursday_ist(9, m)).frame_id
        if m != 45:
            process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root)
    put_embedding(conn, 1, "m", np.ones(4, np.float32))
    table = build_table(conn, "admin_cabin", tz=space.tz).table
    f = data.funnel(conn, space, table).set_index("stage")["photos"].to_dict()
    assert f == {"slots since first photo": 4, "captured": 3, "read by detector": 2, "read by VLM": 2, "outlined": 0,
                 "image vector": 1, "description scores": 0}


def test_tick_timing_reads_run_durations_from_the_log(tmp_path):
    log = tmp_path / "tick.log"
    log.write_text("2026-11-16T16:00:05+05:30 INFO x: detector yolo loaded on 0\n"
                   "2026-11-16T16:00:37+05:30 INFO sbi: tick done: 2 ok, 0 failed\n"
                   "2026-11-16T16:15:01+05:30 INFO x: detector yolo loaded on 0\n"
                   "2026-11-16T16:15:11+05:30 INFO sbi: tick done: 2 ok, 0 failed\n"
                   "2026-11-16T16:30:11+05:30 INFO sbi: tick done: 2 ok, 0 failed\n")       # no start: ignored
    assert data.tick_timing(log) == {"ticks": 2, "median_s": 21.0, "p95_s": pytest.approx(30.9), "max_s": 32.0}
    assert data.tick_timing(log, last=1)["median_s"] == 10.0
    assert data.tick_timing(tmp_path / "none") == {"ticks": 0}


def test_dimension_flow_and_the_dimension_list_say_where_each_column_goes():
    from .test_forecast import result_for
    res = result_for(make_table(200))
    run = {"targets": ["scene.people_yolo"], "covariates": ["count.yolo.chair", "emb.m.pc1", "cat.lighting", "cal.*"]}
    flow = data.dimension_flow(res.catalog, run).set_index("family")
    assert flow.loc["count", ["extracted", "usable", "given_as_input", "forecast"]].tolist() == [4, 3, 1, 0]
    assert flow.loc["scene", ["extracted", "given_as_input", "forecast"]].tolist() == [2, 0, 1]
    assert flow.loc["cal", "given_as_input"] == flow.loc["cal", "extracted"] == 7      # 'cal.*' means every calendar column
    assert flow.loc["total", "forecast"] == 1 and flow.loc["total", "given_as_input"] == 10
    assert data.dimension_flow(res.catalog, None).loc[lambda d: d["family"] == "total", "given_as_input"].item() == 0
    fc = _forecast(res.table, "scene.people_yolo", [5.0, 6.0, 7.0, 8.0])
    d = data.dimensions(res.table, res.catalog, run, fc).set_index("column")
    assert "cal.hour" not in d.index and len(d) == len(res.catalog) - 7
    assert d.loc["scene.people_yolo", ["now", "min", "max", "+1 h forecast", "use"]].tolist() == [199.0, 0.0, 199.0, 8.0, "forecast"]
    assert d.loc["count.yolo.chair", "use"] == "input to model" and d.loc["count.yolo.flat", "use"] == "table only: constant"
    assert d.loc["cat.lighting", "now (text)"] == "dark" and pd.isna(d.loc["cat.lighting", "now"])
    assert d.loc["scene.people_yolo", "dimension"] == "people (detector)"

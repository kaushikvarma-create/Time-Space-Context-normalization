"""Dashboard: the data it shows (health, log problems, charts, latest forecast) and a smoke run of the real app."""
import json
import shutil
from datetime import datetime, timedelta, timezone

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
    assert "v1.0" in at.title[0].value and len(at.dataframe) >= 2                 # health + live accuracy
    assert any("run 1" in c.value for c in at.caption)
    kinds = []
    def walk(node):
        for child in (node.children.values() if isinstance(getattr(node, "children", None), dict) else []):
            kinds.append(getattr(child, "type", ""))
            walk(child)
    walk(at.main)
    assert kinds.count("vega_lite_chart") >= 1                                    # the forecast chart rendered

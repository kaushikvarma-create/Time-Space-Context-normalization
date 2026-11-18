"""v1.0 operations: site calendar (known-future day types), scheduled forecasts inside the tick, live accuracy."""
import sys
import types
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sbi.errors import ConfigError, ForecastError
from sbi.forecast.accuracy import by_lead, by_target, lead_bucket, live_points
from sbi.forecast.backtest import BacktestPlan, run_backtest
from sbi.forecast.schedule import due, run_scheduled
from sbi.forecast.table import TableResult, build_catalog, calendar, to_chronos
from sbi.settings import CalendarSettings, ForecastSettings, TableSettings
from sbi.sitecal import CalendarSpec, country_days, load_calendar, read_site_file
from sbi.store.forecasts import insert_forecasts, insert_run
from sbi.tick import run_tick

from .conftest import FakeDetector, FakeReader, FakeSource, synthetic_image, thursday_ist
from .test_flow import DETS
from .test_forecast import CATALOG_ARGS, Q, SpyForecaster, make_table, result_for

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------- site calendar
def write(root, text, name="config/site_calendar.yaml"):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_site_file_entries_are_read_and_bad_entries_are_named(root):
    p = write(root, "days:\n  - {date: 2026-11-20, name: Office offsite}\n  - {date: 2026-12-25}\n")
    assert read_site_file(p) == {date(2026, 11, 20): "Office offsite", date(2026, 12, 25): "closed"}
    assert read_site_file(root / "nope.yaml") == {}
    for bad in ("days:\n  - {date: 20-11-2026}\n", "days:\n  - just text\n", "days: [\n"):
        with pytest.raises(ConfigError):
            read_site_file(write(root, bad))


def test_the_shipped_site_file_holds_only_fixed_date_national_days():
    from .conftest import REPO
    got = read_site_file(REPO / "config" / "site_calendar.yaml")
    assert {d.strftime("%m-%d") for d in got} == {"01-26", "08-15", "10-02"}


def test_country_holidays_are_optional_and_the_site_file_wins(root, monkeypatch):
    monkeypatch.setitem(sys.modules, "holidays", None)                       # package not installed
    assert country_days("IN", None, range(2026, 2027)) == {}
    fake = types.SimpleNamespace(country_holidays=lambda c, subdiv, years: {date(2026, 11, 8): "Diwali", date(2026, 1, 26): "R-Day"})
    monkeypatch.setitem(sys.modules, "holidays", fake)
    write(root, "days:\n  - {date: 2026-01-26, name: Republic Day}\n")
    spec = load_calendar(CalendarSettings(), root, range(2026, 2027))
    assert spec.holidays == {date(2026, 11, 8), date(2026, 1, 26)} and spec.names[date(2026, 1, 26)] == "Republic Day"
    assert load_calendar(CalendarSettings(country=None), root, range(2026, 2027)).holidays == {date(2026, 1, 26)}
    def unsupported(c, subdiv, years):
        raise NotImplementedError(c)
    monkeypatch.setitem(sys.modules, "holidays", types.SimpleNamespace(country_holidays=unsupported))
    with pytest.raises(ConfigError, match="not supported"):
        country_days("XX", None, range(2026, 2027))


def test_working_days_are_validated():
    assert CalendarSettings(working_days=[0, 1, 2, 3, 4, 5]).working_days[-1] == 5
    for bad in ([], [7], [1, 1]):
        with pytest.raises(ValidationError):
            CalendarSettings(working_days=bad)


def test_calendar_columns_mark_holidays_and_working_days_and_reach_the_future_frame():
    spec = CalendarSpec(frozenset({date(2026, 11, 13)}), frozenset({0, 1, 2, 3, 4, 5}))      # Fri holiday, Sat works
    idx = pd.date_range("2026-11-12 23:45", periods=4 * 96, freq="15min")                       # Thu 23:45 .. Mon
    cal = calendar(idx, spec)
    by_day = cal.groupby(idx.date)[["cal.is_holiday", "cal.is_working_day"]].max()
    assert by_day.loc[date(2026, 11, 13)].tolist() == [1, 0]                 # Friday: holiday, so not working
    assert by_day.loc[date(2026, 11, 14)].tolist() == [0, 1]                 # Saturday works at this site
    assert by_day.loc[date(2026, 11, 15)].tolist() == [0, 0]                 # Sunday off
    table = make_table(120)
    table = table.drop(columns=[c for c in table.columns if c.startswith("cal.")]).join(calendar(table.index, spec))
    res = TableResult(table, build_catalog(table, **CATALOG_ARGS), spec)
    ctx, fut = to_chronos(res, "s", ["scene.people_yolo"], horizon=96, covariates=[])
    assert "cal.is_holiday" in fut.columns and fut["cal.is_holiday"].max() == 1                  # the future knows Friday
    assert [c for c in fut.columns if c.startswith("cal.")] == [c for c in ctx.columns if c.startswith("cal.")]


def test_backtest_horizons_use_the_table_calendar():
    spec = CalendarSpec(frozenset({date(2026, 11, 13)}))
    seen = []
    class Spy(SpyForecaster):
        def forecast(self, context, future, *, targets, quantiles):
            seen.append(int(future["cal.is_holiday"].max()))
            return super().forecast(context, future, targets=targets, quantiles=quantiles)
    t = make_table(200)
    t = t.drop(columns=[c for c in t.columns if c.startswith("cal.")]).join(calendar(t.index, spec))
    plan = BacktestPlan(horizon=4, step=8, min_context=96, quantiles=Q, variants={"calendar": ["cal"]},
                        patterns=["scene.people_yolo"], catalog_args=CATALOG_ARGS, cal=spec)
    run_backtest(t, "s", [Spy()], plan, settings=ForecastSettings(), table_settings=TableSettings())
    assert 1 in seen                                          # horizons that fall on the holiday are told so


# ---------------------------------------------------------------- scheduled forecasts
def test_due_on_the_slot_grid_in_local_time():
    at = lambda h, m, s=0: datetime(2026, 11, 16, h, m, s, tzinfo=IST).astimezone(timezone.utc)
    assert due(at(15, 0, 1), "Asia/Kolkata", 60) and due(at(15, 14, 59), "Asia/Kolkata", 60)    # 15:00 slot, late tick
    assert not due(at(15, 15), "Asia/Kolkata", 60) and due(at(15, 15), "Asia/Kolkata", 15)
    assert due(at(0, 0), "Asia/Kolkata", 1440) and not due(at(12, 0), "Asia/Kolkata", 1440)
    assert not due(at(15, 0), "Asia/Kolkata", 0)
    with pytest.raises(ValidationError, match="multiple of the 15-minute tick"):
        ForecastSettings(every_minutes=20)
    assert ForecastSettings().every_minutes == 0                                  # off unless config switches it on


def scheduled(settings, every=60):
    return settings.model_copy(update={"forecast": settings.forecast.model_copy(update={"every_minutes": every, "min_context": 2})})


def tick(settings, space_dir, now, **kw):
    return run_tick(settings, sorted(space_dir.glob("*.yaml")), source=FakeSource([synthetic_image()]),
                    detector=FakeDetector(DETS), reader=FakeReader(), now=now, **kw)


def runs(settings):
    import sqlite3
    c = sqlite3.connect(settings.db_path)
    return c.execute("SELECT space_id, model FROM forecast_runs ORDER BY run_id").fetchall()


def test_the_tick_stores_a_forecast_per_space_only_when_due(settings, space_dir, monkeypatch):
    s = scheduled(settings)
    import sbi.forecast.schedule as sched
    monkeypatch.setattr(sched, "forecast_space", lambda conn, space, result, fc, **kw: types.SimpleNamespace(run_id=hash(space.space_id) % 1000))
    built = []
    monkeypatch.setattr(sched, "build_forecaster", lambda *a, **k: built.append(a) or SpyForecaster())
    out = tick(s, space_dir, thursday_ist(10, 15))                                 # not due: no model is even loaded
    assert all(o.ok for o in out) and built == []
    out = tick(s, space_dir, thursday_ist(11, 0))
    assert all(o.ok for o in out) and len(built) == 1                              # due: one model for all spaces


def test_scheduled_forecasts_really_store_runs_end_to_end(settings, space_dir, monkeypatch):
    # the table is synthetic (four ticks cannot make a series usable); tick, scheduler, runner and storage are real
    import sbi.forecast.schedule as sched
    from sbi.forecast.models import Persistence
    monkeypatch.setattr(sched, "table_for", lambda conn, st, sp: result_for(make_table(120)))
    s = scheduled(settings, every=30)
    for m in (0, 15, 30, 45):
        tick(s, space_dir, thursday_ist(10, m), forecaster=Persistence())
    got = runs(s)
    assert len(got) == 4 and {sp for sp, _ in got} == {"admin_cabin", "parking"} and {m for _, m in got} == {"persistence"}


def test_forecasting_problems_never_touch_capture(settings, space_dir, monkeypatch, caplog):
    s = scheduled(settings)
    import sbi.forecast.schedule as sched
    def no_model(*a, **k):
        raise ForecastError("weights missing")
    monkeypatch.setattr(sched, "build_forecaster", no_model)
    out = tick(s, space_dir, thursday_ist(10, 0))
    assert all(o.ok and o.processed == 1 for o in out) and "forecaster unavailable" in caplog.text
    out = tick(s, space_dir, thursday_ist(11, 0), forecaster=SpyForecaster(fail_when=lambda c: True))
    assert all(o.ok for o in out) and runs(s) == []                                # a model error stores nothing
    monkeypatch.setattr(sched, "table_for", lambda *a, **k: 1 / 0)                # a bug inside forecasting
    out = tick(s, space_dir, thursday_ist(12, 0), forecaster=SpyForecaster())
    assert all(o.ok for o in out) and "this is a bug" in caplog.text


def test_run_scheduled_reports_each_space(settings, space, parking, monkeypatch):
    s = scheduled(settings)
    import sbi.forecast.schedule as sched
    monkeypatch.setattr(sched, "table_for", lambda conn, st, sp: result_for(make_table(120)))
    from sbi.store.db import connect, migrate
    from sbi.store.registry import register_space
    c = connect(s.db_path); migrate(c)
    register_space(c, space, "a"); register_space(c, parking, "b")
    now = thursday_ist(11, 0)
    got = run_scheduled(c, s, [space, parking.model_copy(update={"forecast_targets": ["nothing.*"]})], now,
                        forecaster=SpyForecaster())
    assert got[0].run_id is not None and got[1].run_id is None and "matches a model-ready" in got[1].error
    assert run_scheduled(c, s, [space.model_copy(update={"forecast_targets": []})], now, forecaster=SpyForecaster()) == []


def test_an_unexpected_error_in_one_space_does_not_stop_the_next(settings, space, parking, monkeypatch, caplog):
    s = scheduled(settings)
    import sbi.forecast.schedule as sched
    def table(conn, st, sp):
        if sp.space_id == "admin_cabin":
            raise RuntimeError("unexpected")
        return result_for(make_table(120))
    monkeypatch.setattr(sched, "table_for", table)
    from sbi.store.db import connect, migrate
    from sbi.store.registry import register_space
    c = connect(s.db_path); migrate(c)
    register_space(c, space, "a"); register_space(c, parking, "b")
    got = run_scheduled(c, s, [space, parking], thursday_ist(11, 0), forecaster=SpyForecaster())
    assert got[0].run_id is None and got[0].error.startswith("unexpected") and got[1].run_id is not None
    assert "this is a bug" in caplog.text


# ---------------------------------------------------------------- live accuracy
def store_run(conn, origin, values, *, model="chronos2:m", step_minutes=15, lo=1.0, hi=1.0):
    """A stored forecast for target 'scene.people_yolo' from `origin`, one value per step."""
    rid = insert_run(conn, space_id="admin_cabin", origin_local=pd.Timestamp(origin).isoformat(), model=model,
                     variant="calendar", horizon=len(values), step_minutes=step_minutes, context_slots=100,
                     targets=["scene.people_yolo"], covariates=["cal.*"], quantiles=Q)
    ts = pd.date_range(pd.Timestamp(origin) + timedelta(minutes=15), periods=len(values), freq="15min")
    insert_forecasts(conn, rid, pd.DataFrame({"timestamp": ts, "target": "scene.people_yolo", "step": range(1, len(values) + 1),
                                              "q10": np.array(values) - lo, "q50": values, "q90": np.array(values) + hi}))
    return rid


def test_live_accuracy_scores_stored_forecasts_against_what_happened_and_against_persistence(conn):
    table = make_table(200)                                    # scene.people_yolo == row position
    origin = table.index[100]                                  # value 100 there
    store_run(conn, origin, [101.0, 102.0, 105.0, 104.0])      # truth: 101, 102, 103, 104
    store_run(conn, table.index[198], [199.0, 200.0, 201.0])   # only the first step has happened yet
    pts = live_points(conn, "admin_cabin", table)
    first = pts[pts["run_id"] == 1].sort_values("step")
    assert first["abs_err"].tolist() == [0.0, 0.0, 2.0, 0.0] and first["persist_err"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert first["covered"].tolist() == [1.0, 1.0, 0.0, 1.0] and first["lead_min"].tolist() == [15, 30, 45, 60]
    assert len(pts[pts["run_id"] == 2]) == 1                    # the future is not scored
    t = by_target(pts).iloc[0]
    assert t["n"] == 5 and t["skill"] == pytest.approx(1 - (2 / 5) / ((1 + 2 + 3 + 4 + 1) / 5))
    assert live_points(conn, "admin_cabin", table, since=str(table.index[150])).run_id.unique().tolist() == [2]
    assert live_points(conn, "admin_cabin", table, since="2026-11-14").run_id.unique().tolist() == [2]   # date-only form (run 2 starts 14 Nov 01:30)
    assert live_points(conn, "parking", table).empty
    steeper = table.assign(**{"scene.people_yolo": table["scene.people_yolo"] * 3})        # changes by 3 per slot
    assert live_points(conn, "admin_cabin", steeper)["scale"].unique().tolist() == [3.0]


def test_accuracy_by_lead_time_and_point_baselines_have_no_interval(conn):
    table = make_table(200)
    store_run(conn, table.index[100], list(np.arange(101.0, 117.0)))                            # perfect, 4 h
    store_run(conn, table.index[100], [100.0] * 16, model="persistence")
    leads = by_lead(live_points(conn, "admin_cabin", table))
    chronos = leads[leads["model"] == "chronos2:m"]
    assert chronos["lead"].tolist() == ["0-1 h", "1-2 h", "2-4 h"] and (chronos["skill"] == 1.0).all()
    assert leads[leads["model"] == "persistence"]["coverage"].isna().all()
    assert [lead_bucket(m) for m in (15, 60, 75, 240, 255)] == ["0-1 h", "0-1 h", "1-2 h", "2-4 h", "4 h +"]
    assert by_lead(pd.DataFrame()).empty and by_target(pd.DataFrame()).empty


def test_a_series_that_did_not_move_cannot_swamp_the_pooled_skill():
    pts = pd.DataFrame({
        "model": "m", "variant": "v", "run_id": 1, "lead_min": 15, "covered": 1.0,
        "target": ["still", "still", "busy", "busy"],
        "abs_err": [0.002, 0.002, 1.0, 1.0], "persist_err": [0.0, 0.0, 4.0, 4.0], "scale": [0.01, 0.01, 2.0, 2.0]})
    assert by_target(pts).set_index("target").loc["still", "skill"] != by_target(pts).set_index("target").loc["still", "skill"]  # NaN: no base
    pooled = by_lead(pts)["skill"].iloc[0]
    assert pooled == pytest.approx(1 - (0.4 + 1.0) / 4.0)          # (0.002/0.01*2 + 1/2*2) / (4/2*2)
    mean_of_ratios = np.mean([1 - 1.0 / 4.0])                       # the per-series view, 'still' has no ratio at all
    assert 0 < pooled < mean_of_ratios
    pts.loc[pts["target"] == "still", "persist_err"] = 0.0001       # tiny but non-zero: the old average would explode
    assert by_target(pts).set_index("target").loc["still", "skill"] < -10 and by_lead(pts)["skill"].iloc[0] > 0

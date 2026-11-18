"""Day-type profiles (a known-in-advance input for the forecaster) and the by-weekday accuracy table."""
import numpy as np
import pandas as pd
import pytest

from sbi.forecast.accuracy import by_weekday
from sbi.forecast.inputs import prepare
from sbi.forecast.profile_features import add_profiles, day_profile
from sbi.settings import PROFILE_KIND, ForecastSettings

from .test_forecast import FS, TS, make_table, result_for

DAY = 96


def week_series(days, value_of):
    """15-minute series starting Monday 2026-11-09; value_of(day_number, slot_of_day) -> value."""
    idx = pd.date_range("2026-11-09 00:00", periods=days * DAY, freq="15min")
    return pd.Series([float(value_of(i // DAY, i % DAY)) for i in range(len(idx))], index=idx)


def working(idx):
    return (idx.dayofweek < 5).astype(int)


def test_profile_uses_the_same_day_type_and_leaves_the_rows_own_day_out():
    s = week_series(7, lambda d, slot: (10 + d) if d < 5 else 0)          # Mon..Fri 10..14, weekend 0
    future = pd.date_range(s.index[-1] + pd.Timedelta(minutes=15), periods=4, freq="15min")       # the next Monday
    past, fut = day_profile(s, working(s.index), future, working(future))
    assert past[0] == pytest.approx((11 + 12 + 13 + 14) / 4)              # Monday: the other four working days
    assert past[4 * DAY] == pytest.approx((10 + 11 + 12 + 13) / 4)        # Friday likewise, never its own 14
    assert past[5 * DAY] == 0.0 and past[6 * DAY] == 0.0                  # Saturday from Sunday, Sunday from Saturday
    assert fut.tolist() == [pytest.approx(12.0)] * 4                      # one earlier Monday only: all working days


def test_profile_switches_to_the_same_weekday_once_it_has_been_seen_often_enough():
    s = week_series(21, lambda d, slot: 100 if d % 7 == 0 else (10 if d % 7 < 5 else 0))          # Mondays are special
    future = pd.date_range(s.index[-1] + pd.Timedelta(minutes=15), periods=2, freq="15min")       # a fourth Monday
    past, fut = day_profile(s, working(s.index), future, working(future))
    assert fut.tolist() == [100.0, 100.0]                                 # three earlier Mondays: weekday level
    assert past[0] == 100.0                                               # two other Mondays: weekday level too
    _, strict = day_profile(s, working(s.index), future, working(future), min_same_weekday=4)
    assert strict[0] == pytest.approx((3 * 100 + 12 * 10) / 15)           # not enough Mondays: all working days
    assert past[DAY] == 10.0 and past[5 * DAY] == 0.0                     # Tuesday and Saturday by their own weekday


def test_profile_follows_the_clock_time_and_falls_back_when_a_kind_of_day_is_new():
    s = week_series(3, lambda d, slot: slot)                              # Mon..Wed, the value is the time of day
    future = pd.date_range("2026-11-14 00:00", periods=3, freq="15min")   # a Saturday: no non-working day seen yet
    past, fut = day_profile(s, working(s.index), future, working(future))
    assert fut.tolist() == [0.0, 1.0, 2.0] and past[5] == 5.0             # any-day level, still by clock time
    one = week_series(1, lambda d, slot: 4.0).iloc[:8]                    # a single day: nothing to compare with
    p1, f1 = day_profile(one, working(one.index), future, working(future))
    assert (p1 == 4.0).all() and (f1 == 4.0).all()                        # the overall average
    gaps = week_series(3, lambda d, slot: slot); gaps.iloc[DAY + 5] = np.nan
    pg, _ = day_profile(gaps, working(gaps.index), future, working(future))
    assert pg[DAY + 5] == 5.0 and pg[5] == 5.0                            # a missing value neither breaks nor leaks
    empty = pd.Series(np.nan, index=one.index)
    assert np.isnan(day_profile(empty, working(one.index), future, working(future))[1]).all()


def test_profiles_reach_the_model_as_known_in_advance_columns_only_when_asked():
    result = result_for(make_table())
    plain = prepare(result, "s", ["scene.people_yolo", "count.yolo.chair"], ["cal"], horizon=4, settings=FS, table_settings=TS)
    assert not plain.profiles and not [c for c in plain.context.columns if c.startswith("prof.")]
    p = prepare(result, "s", ["scene.people_yolo", "count.yolo.chair"], ["cal", PROFILE_KIND], horizon=4, settings=FS, table_settings=TS)
    names = ["prof.scene.people_yolo", "prof.count.yolo.chair"]
    assert p.profiles and p.covariates == [] and all(n in p.context.columns and n in p.future.columns for n in names)
    assert p.future[names].notna().all().all() and p.context[names].notna().all().all() and len(p.future) == 4
    assert "scene.people_yolo" not in p.future.columns                    # the target itself is never given ahead
    both = prepare(result, "s", ["scene.people_yolo"], ["cal", "*", PROFILE_KIND], horizon=4, settings=FS, table_settings=TS)
    assert both.covariates and "prof.scene.people_yolo" in both.future.columns and "prof" not in both.covariates
    assert ForecastSettings().variants["all_profile"] == ["cal", "*", PROFILE_KIND]


def test_a_profile_never_sees_beyond_the_history_it_is_given():
    table = make_table(300)
    short, full = result_for(table.iloc[:200]), result_for(table)
    a = prepare(short, "s", ["scene.people_yolo"], ["cal", PROFILE_KIND], horizon=4, settings=FS, table_settings=TS)
    assert a.future["prof.scene.people_yolo"].max() < 200                 # the series equals its row number
    b = prepare(full, "s", ["scene.people_yolo"], ["cal", PROFILE_KIND], horizon=4, settings=FS, table_settings=TS)
    assert b.future["prof.scene.people_yolo"].max() >= 200                # with the later rows it is higher


def test_add_profiles_returns_the_new_column_names():
    idx = pd.date_range("2026-11-09", periods=8, freq="15min"); fidx = pd.date_range(idx[-1] + pd.Timedelta(minutes=15), periods=2, freq="15min")
    ctx = pd.DataFrame({"id": "s", "timestamp": idx, "a": np.arange(8.0)}); fut = pd.DataFrame({"id": "s", "timestamp": fidx})
    c2, f2, names = add_profiles(ctx, fut, ["a"], np.ones(8, int), np.ones(2, int))
    assert names == ["prof.a"] and list(c2.columns) == ["id", "timestamp", "a", "prof.a"] and list(f2.columns) == ["id", "timestamp", "prof.a"]


def points(rows):
    return pd.DataFrame([{"timestamp": pd.Timestamp(ts), "target": t, "y": y, "pred": pred, "abs_err": abs(y - pred),
                          "persist_err": pe, "scale": 1.0, "covered": cov} for ts, t, y, pred, pe, cov in rows])


def test_by_weekday_shows_which_days_are_forecast_badly():
    p = points([("2026-11-16 10:00", "a", 3.0, 1.0, 3.0, 0.0), ("2026-11-16 11:00", "a", 3.0, 0.0, 3.0, 0.0),   # Monday: badly
                ("2026-11-17 10:00", "a", 2.0, 2.0, 1.0, 1.0),                                                    # Tuesday: exact
                ("2026-11-15 10:00", "a", 0.0, 0.0, 0.0, 1.0), ("2026-11-16 10:00", "b", 5.0, 5.0, 2.0, 1.0)])
    one = by_weekday(p, "a").set_index("day")
    assert list(one.index) == ["Mon", "Tue", "Sun"]                       # Monday first, missing days left out
    assert one.loc["Mon"].tolist() == [2, 3.0, 0.5, 2.5, 3.0, 0.0]
    assert one.loc["Tue", "average miss"] == 0.0 and one.loc["Sun", "forecasts checked"] == 1
    every = by_weekday(p).set_index("day")
    assert every.loc["Mon", "forecasts checked"] == 3 and every.loc["Mon", "skill"] == pytest.approx(1 - 5 / 8)
    assert every.loc["Tue", "skill"] == 1.0 and list(every.columns) == ["forecasts checked", "skill", "inside band"]
    assert by_weekday(pd.DataFrame()).empty and by_weekday(pd.DataFrame(), "a").columns[1] == "forecasts checked"


def test_runs_made_before_their_history_was_read_again_are_not_scored(conn, space, root):
    from datetime import timedelta
    from sbi.dashboard import data
    from sbi.forecast.accuracy import live_points, outdated_runs
    from sbi.forecast.table import build_table
    from sbi.pipeline import process_frame
    from sbi.store.forecasts import insert_forecasts, insert_run
    from .conftest import FakeDetector, FakeReader, FakeSource, synthetic_image, thursday_ist
    from .test_flow import DETS, cap
    ids = []
    for i in range(8):
        fid = cap(conn, space, root, FakeSource([synthetic_image(i)]), thursday_ist(9, 0) + timedelta(minutes=15 * i)).frame_id
        process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root); ids.append(fid)
    def read_at(fid, when):
        conn.execute("UPDATE frames SET processed_utc=? WHERE frame_id=?", (when.isoformat(timespec="seconds"), fid))
    for i, fid in enumerate(ids):                                          # each photo read 10 s after it was taken
        read_at(fid, (thursday_ist(9, 0) + timedelta(minutes=15 * i, seconds=10)).astimezone(__import__("datetime").timezone.utc))
    def run(origin_min, created):
        local = lambda dt: dt.astimezone(__import__("datetime").timezone(timedelta(hours=5, minutes=30)))     # IST wall clock
        rid = insert_run(conn, space_id="admin_cabin", origin_local=local(thursday_ist(9, 0) + timedelta(minutes=origin_min)).strftime("%Y-%m-%dT%H:%M:%S"),
                         model="m", variant="all", horizon=1, step_minutes=15, context_slots=4, targets=["scene.people_yolo"],
                         covariates=[], quantiles=[0.1, 0.5, 0.9], created=created)
        ts = local(thursday_ist(9, 0) + timedelta(minutes=origin_min + 15))
        insert_forecasts(conn, rid, pd.DataFrame({"timestamp": [pd.Timestamp(ts.strftime("%Y-%m-%d %H:%M:%S"))], "target": "scene.people_yolo",
                                                  "step": 1, "q10": 0.0, "q50": 1.0, "q90": 2.0}))
        return rid
    first = run(60, thursday_ist(10, 0) + timedelta(seconds=20))           # made at 10:00 from photos read by then
    table = build_table(conn, "admin_cabin", tz=space.tz).table
    assert outdated_runs(conn, "admin_cabin") == set() and set(live_points(conn, "admin_cabin", table)["run_id"]) == {first}
    read_at(ids[0], thursday_ist(10, 40).astimezone(__import__("datetime").timezone.utc))     # one late retry of 5 photos (20%)...
    assert outdated_runs(conn, "admin_cabin", tolerance=0.25) == set()     # ...is tolerated when the tolerance allows it
    for fid in ids:                                                        # every photo read again at 10:40
        read_at(fid, thursday_ist(10, 40).astimezone(__import__("datetime").timezone.utc))
    second = run(90, thursday_ist(10, 41))                                 # made after the re-read
    assert outdated_runs(conn, "admin_cabin") == {first}
    assert set(live_points(conn, "admin_cabin", table)["run_id"]) == {second}
    assert len(data.lead_history(conn, "admin_cabin", table, "scene.people_yolo", 1)) == 2
    assert len(data.lead_history(conn, "admin_cabin", table, "scene.people_yolo", 1, skip_runs={first})) == 1
    assert outdated_runs(conn, "parking") == set()

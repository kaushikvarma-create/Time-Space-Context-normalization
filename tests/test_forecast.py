"""Forecasting stage: baselines, metrics, target/covariate selection, the leak-free backtest, storage, live runs, CLI."""
import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sbi.cli import main
from sbi.errors import ForecastError, StorageError
from sbi.forecast.backtest import BacktestPlan, origins_for, run_backtest
from sbi.forecast.inputs import predict, prepare, resolve_targets
from sbi.forecast.metrics import evaluate, overall, pinball, summarise
from sbi.forecast.models import Chronos2Forecaster, Persistence, SeasonalMean, SeasonalNaive, build_forecaster, qname
from sbi.forecast.runner import forecast_space
from sbi.forecast.table import TableResult, build_catalog, calendar
from sbi.pipeline import process_frame
from sbi.settings import ALL_KINDS, ForecastSettings, TableSettings
from sbi.spaces import SpaceConfig
from sbi.store.db import connect, migrate
from sbi.store.forecasts import insert_forecasts, insert_run, latest_run, load_forecast
from sbi.store.registry import register_space

from .conftest import FakeDetector, FakeReader, FakeSource, REPO, det, synthetic_image, thursday_ist
from .test_flow import DETS, cap

Q = [0.1, 0.5, 0.9]
FS, TS = ForecastSettings(), TableSettings()
CATALOG_ARGS = dict(min_coverage=0.5, min_presence=0.05, min_valid_slots=8, min_present_slots=3)


def make_table(n=200) -> pd.DataFrame:
    idx = pd.date_range("2026-11-12 00:00", periods=n, freq="15min", name="timestamp")
    t, rng = np.arange(n), np.random.default_rng(0)
    df = pd.DataFrame({"slot_status": "ok", "frame_id": t, "vlm_ok": True}, index=idx).join(calendar(idx))
    df["scene.people_yolo"] = t.astype(float)                       # = row position: a leak shows up as a larger value
    df["scene.people_vlm"] = 3 + 2 * np.sin(t / 12) + rng.normal(0, 0.2, n)
    df["count.yolo.chair"] = 5 + (t // 20) % 3 + rng.integers(0, 2, n)
    df["count.yolo_c05.chair"] = df["count.yolo.chair"] + 1                      # a threshold variant
    df["count.yolo.late"] = np.where(t < 150, 0, 1 + t % 3)                      # constant, then it starts to vary
    df["count.yolo.flat"] = 2
    df["area.sam2.chair"] = 0.05 + 0.01 * np.cos(t / 15)
    df["probe.m.people"] = np.sin(t / 10)
    df["emb.m.pc1"] = np.cos(t / 10)
    df["img.brightness"] = 100 + 20 * np.sin(t / 30)
    df["cat.lighting"] = np.where(idx.hour < 7, "dark", "lit")
    return df


def result_for(df: pd.DataFrame) -> TableResult:
    return TableResult(df, build_catalog(df, **CATALOG_ARGS))


def context_of(values: dict[str, list[float]], start="2026-11-09 00:00") -> pd.DataFrame:
    n = len(next(iter(values.values())))
    return pd.DataFrame({"id": "s", "timestamp": pd.date_range(start, periods=n, freq="15min"), **values})


def future_of(context: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return pd.DataFrame({"id": "s", "timestamp": pd.date_range(context["timestamp"].iloc[-1] + timedelta(minutes=15),
                                                                periods=horizon, freq="15min")})


class SpyForecaster:
    """Persistence with +-1 intervals that records exactly what it was shown."""
    name, probabilistic, uses_covariates = "spy", True, True

    def __init__(self, fail_when=lambda calls: False):
        self.calls, self.fail_when = [], fail_when

    def forecast(self, context, future, *, targets, quantiles):
        self.calls.append({"origin": context["timestamp"].max(), "first_future": future["timestamp"].min(),
                           "targets": list(targets), "cols": set(context.columns), "future_cols": list(future.columns),
                           "people_max": context["scene.people_yolo"].max() if "scene.people_yolo" in context else None})
        if self.fail_when(len(self.calls)):
            raise ForecastError("model failed")
        out = []
        for t in targets:
            last = context[t].dropna().iloc[-1]
            for step, ts in enumerate(future["timestamp"], 1):
                out.append({"timestamp": ts, "target": t, "step": step, qname(0.1): last - 1, qname(0.5): last,
                            qname(0.9): last + 1})
        return pd.DataFrame(out)


# ---------------------------------------------------------------- settings and config
def test_forecast_settings_rules():
    assert FS.variant == "all" and FS.variants["univariate"] == [] and "emb" in FS.variants["full"]
    assert FS.variants["all"] == ["cal", ALL_KINDS]
    for bad in ([0.1, 0.9], [0.5, 0.1], [0.0, 0.5], [0.5, 1.0], [0.125, 0.5, 0.9], []):
        with pytest.raises(ValidationError):
            ForecastSettings(quantiles=bad)
    with pytest.raises(ValidationError, match="not one of"):
        ForecastSettings(variant="nope")


def test_forecast_targets_in_space_files_and_their_validation(space, parking):
    assert "steady.*" in space.forecast_targets and "scene.vehicles_yolo" in parking.forecast_targets
    for bad in (["a", "a"], ["  "]):
        with pytest.raises(ValidationError, match="forecast_targets"):
            SpaceConfig.model_validate({**space.model_dump(), "forecast_targets": bad})


# ---------------------------------------------------------------- baselines
def test_persistence_holds_the_last_value_and_every_quantile_equals_it():
    ctx = context_of({"a": [1.0, 2.0, np.nan, 7.0, np.nan]})
    out = Persistence().forecast(ctx, future_of(ctx, 3), targets=["a"], quantiles=Q)
    assert out["q50"].tolist() == [7.0] * 3 and (out["q10"] == out["q90"]).all()
    assert out["step"].tolist() == [1, 2, 3] and out["target"].eq("a").all()


def test_seasonal_naive_takes_the_same_clock_time_from_the_latest_earlier_day_that_has_one():
    n = 3 * 96
    day = np.repeat(np.arange(3), 96)
    vals = (day * 1000 + np.tile(np.arange(96), 3)).astype(float)
    ctx = context_of({"a": list(vals)})
    fut = future_of(ctx, 4)
    out = SeasonalNaive().forecast(ctx, fut, targets=["a"], quantiles=Q)
    assert out["q50"].tolist() == [2000.0, 2001.0, 2002.0, 2003.0]               # slots 0-3 of day 3 <- same slots of day 2
    holed = vals.copy(); holed[2 * 96:2 * 96 + 2] = np.nan                       # yesterday's first two slots missing
    out = SeasonalNaive().forecast(context_of({"a": list(holed)}), fut, targets=["a"], quantiles=Q)
    assert out["q50"].tolist() == [1000.0, 1001.0, 2002.0, 2003.0]               # reach back one more day for those
    fresh = context_of({"a": [5.0] * 10})
    assert SeasonalNaive().forecast(fresh, future_of(fresh, 2), targets=["a"], quantiles=Q)["q50"].tolist() == [5.0, 5.0]


def test_seasonal_mean_uses_weekday_buckets_only_with_two_points_else_time_of_day_else_last_value():
    two_weeks = 14 * 96
    idx = pd.date_range("2026-11-02 00:00", periods=two_weeks, freq="15min")           # a Monday
    vals = (idx.dayofweek * 100 + idx.hour).astype(float)
    ctx = pd.DataFrame({"id": "s", "timestamp": idx, "a": vals})
    out = SeasonalMean().forecast(ctx, future_of(ctx, 8), targets=["a"], quantiles=Q)     # Mon 00:00..01:45
    assert out["q50"].tolist() == [0.0] * 4 + [1.0] * 4                                    # two Mondays agree
    one_week = ctx.iloc[: 7 * 96].copy()                                                   # one Monday only
    one_week["a"] = np.where(one_week["timestamp"].dt.dayofweek == 0, 999.0, one_week["a"])
    out = SeasonalMean().forecast(one_week, future_of(one_week, 4), targets=["a"], quantiles=Q)
    # next Monday 00:00 has only one Monday behind it, so the mean is over that clock time on all 7 days
    assert out["q50"].tolist() == [pytest.approx((999.0 + sum(range(100, 700, 100))) / 7)] * 4
    sparse = context_of({"a": [1.0, np.nan, 3.0]})
    assert SeasonalMean().forecast(sparse, future_of(sparse, 1), targets=["a"], quantiles=Q)["q50"].tolist() == [3.0]


def test_a_target_with_no_history_is_an_error_and_unknown_models_are_named():
    ctx = context_of({"a": [np.nan] * 5})
    with pytest.raises(ForecastError, match="no observed values"):
        Persistence().forecast(ctx, future_of(ctx, 2), targets=["a"], quantiles=Q)
    with pytest.raises(ForecastError, match="unknown forecaster"):
        build_forecaster("nope")
    assert build_forecaster("seasonal_mean").name == "seasonal_mean" and not build_forecaster("persistence").probabilistic


# ---------------------------------------------------------------- metrics
def forecast_frame(rows):
    return pd.DataFrame([{"timestamp": pd.Timestamp(t), "target": g, "step": s, "q10": a, "q50": b, "q90": c}
                         for t, g, s, a, b, c in rows])


def test_pinball_loss_by_hand():
    one = lambda y, p, tau: float(pinball(np.array([y]), np.array([p]), tau)[0])
    assert one(1.0, 0.0, 0.9) == pytest.approx(0.9)        # under-forecast costs tau per unit
    assert one(1.0, 2.0, 0.9) == pytest.approx(0.1)        # over-forecast costs (1 - tau) per unit
    assert one(1.0, 2.0, 0.1) == pytest.approx(0.9)
    assert one(1.0, 1.0, 0.5) == 0.0 and one(0.0, 3.0, 0.5) == pytest.approx(1.5)


def test_evaluate_scores_each_point_and_skips_unmeasured_slots():
    f = forecast_frame([("2026-11-12 00:15", "a", 1, 0.0, 2.0, 4.0), ("2026-11-12 00:30", "a", 2, 0.0, 2.0, 4.0),
                        ("2026-11-12 00:45", "a", 3, 0.0, 2.0, 4.0)])
    truth = pd.DataFrame({"a": [1.0, 7.0, np.nan]}, index=pd.to_datetime(["2026-11-12 00:15", "2026-11-12 00:30", "2026-11-12 00:45"]))
    e = evaluate(f, truth, Q, probabilistic=True)
    assert len(e) == 2 and e["abs_err"].tolist() == [1.0, 5.0] and e["covered"].tolist() == [1.0, 0.0]
    expect = np.mean([pinball(np.array([1.0]), np.array([p]), tau)[0] for p, tau in ((0.0, .1), (2.0, .5), (4.0, .9))])
    assert e["pinball"].iloc[0] == pytest.approx(expect)
    assert evaluate(f, truth, Q, probabilistic=False)["covered"].isna().all()              # point baselines have no interval
    assert evaluate(f.iloc[:0], truth, Q, probabilistic=True).empty


def test_summary_skill_compares_with_the_reference_on_the_same_points_only():
    pts = lambda fc, errs, extra=(): [{"origin": 0, "forecaster": fc, "variant": "-", "target": "a",
                                       "timestamp": pd.Timestamp("2026-11-12") + timedelta(minutes=15 * i), "y": 10.0,
                                       "abs_err": e, "pinball": e / 2, "covered": 1.0} for i, e in enumerate(errs)]
    raw = pd.DataFrame(pts("seasonal_mean", [4.0, 4.0]) + pts("good", [1.0, 2.0, 100.0]))   # "good" has an extra point
    s = summarise(raw).set_index("forecaster")
    assert s.loc["good", "n"] == 3 and s.loc["good", "mae"] == pytest.approx(103 / 3)
    assert s.loc["good", "skill"] == pytest.approx(1 - 1.5 / 4.0)                           # only the 2 shared points count
    assert s.loc["seasonal_mean", "skill"] == pytest.approx(0.0)
    assert s.loc["good", "wql"] == pytest.approx(2 * (103 / 2) / 30)
    o = overall(summarise(raw))
    assert o["forecaster"].tolist() == ["good", "seasonal_mean"] and o.iloc[0]["skill"] == pytest.approx(0.625)
    assert summarise(raw.iloc[:0]).empty


# ---------------------------------------------------------------- targets and covariate sets
def test_targets_come_from_model_ready_columns_in_pattern_order_without_variants():
    cat = result_for(make_table()).catalog
    got = resolve_targets(["scene.people_yolo", "count.yolo.*", "probe.*.people"], cat, 40)
    assert got[0] == "scene.people_yolo" and "count.yolo.chair" in got and "probe.m.people" in got
    assert "count.yolo_c05.chair" not in got and "count.yolo.flat" not in got            # variant, and constant
    assert "count.yolo_c05.chair" not in resolve_targets(["count.*"], cat, 40)           # a broad pattern skips variants too
    assert "count.yolo_c05.chair" in resolve_targets(["count.yolo_c05.chair"], cat, 40)  # exact name is allowed
    assert resolve_targets(["count.yolo.*", "scene.*"], cat, 2) == resolve_targets(["count.yolo.*", "scene.*"], cat, 40)[:2]
    for bad in (["nope.*"], ["cat.lighting"], ["cal.hour"]):
        with pytest.raises(ForecastError, match="matches a model-ready"):
            resolve_targets(bad, cat, 40)


def test_unmatched_pattern_is_reported_but_does_not_stop_the_others(caplog):
    with caplog.at_level(logging.WARNING):
        got = resolve_targets(["probe.*.nothing", "scene.people_yolo"], result_for(make_table()).catalog, 40)
    assert got == ["scene.people_yolo"] and "matches no model-ready column" in caplog.text


def prep(kinds, table=None, patterns=("scene.people_yolo", "count.yolo.*")):
    return prepare(result_for(table if table is not None else make_table()), "s", list(patterns), kinds, horizon=4,
                   settings=FS, table_settings=TS)


def test_variants_decide_which_inputs_exist():
    uni = prep([])
    assert uni.covariates == [] and not uni.calendar
    assert set(uni.context.columns) == {"id", "timestamp", *uni.targets}
    assert list(uni.future.columns) == ["id", "timestamp"]                              # nothing known in advance
    cal = prep(["cal"])
    assert cal.covariates == [] and cal.calendar and {c for c in cal.context.columns if c.startswith("cal.")}
    assert [c for c in cal.future.columns if c.startswith("cal.")] == [c for c in cal.context.columns if c.startswith("cal.")]
    det_ = prep(["cal", "scene", "count", "area", "img", "cat"])
    assert det_.covariates and not [c for c in det_.covariates if c.startswith(("emb.", "probe."))]
    assert not set(det_.covariates) & set(det_.targets) and "count.yolo_c05.chair" not in det_.covariates
    full = prep(["cal", "scene", "count", "area", "img", "cat", "emb", "probe"], patterns=("scene.people_yolo",))
    assert "emb.m.pc1" in full.covariates and "probe.m.people" in full.covariates
    assert full.context["cat.lighting"].dtype == object                                  # categorical stays text


def test_all_variant_gives_every_model_ready_column_uncapped():
    result = result_for(make_table())
    cat = result.catalog
    ts = TS.model_copy(update={"max_covariates": 1, "min_autocorr": 0.99})            # caps that `all` must ignore
    p = prepare(result, "s", ["scene.people_yolo"], ["cal", ALL_KINDS], horizon=4, settings=FS, table_settings=ts)
    ready = set(cat.loc[cat["model_ready"] & (cat["kind"] != "cal"), "column"]) - {"scene.people_yolo"}
    assert set(p.covariates) == ready and len(p.covariates) == len(ready) and p.calendar
    assert "count.yolo_c05.chair" in p.covariates and "emb.m.pc1" in p.covariates and "cat.lighting" in p.covariates
    assert "count.yolo.flat" not in p.covariates                                      # not model-ready (constant)
    assert not [c for c in p.covariates if c.startswith("cal.")]                       # calendar goes in separately
    assert set(p.covariates) <= set(p.context.columns)


class Negative:
    """A model that predicts below zero for everything."""
    name, probabilistic, uses_covariates = "neg", True, True

    def forecast(self, context, future, *, targets, quantiles):
        return pd.concat([pd.DataFrame({"timestamp": future["timestamp"], "target": t, "step": range(1, len(future) + 1),
                                        "q10": -2.0, "q50": -1.0, "q90": 0.5}) for t in targets], ignore_index=True)


def test_counts_are_never_forecast_below_zero_but_other_kinds_may_be():
    result = result_for(make_table())
    p = prepare(result, "s", ["count.yolo.chair", "area.sam2.chair"], ["cal"], horizon=4, settings=FS, table_settings=TS)
    out = predict(Negative(), p, Q, result.catalog)
    assert (out[["q10", "q50"]] == 0.0).all().all() and (out["q90"] == 0.5).all()          # clipped at zero
    p2 = prepare(result, "s", ["probe.m.people", "emb.m.pc1"], ["cal"], horizon=4, settings=FS, table_settings=TS)
    free = predict(Negative(), p2, Q, result.catalog)
    assert (free["q10"] == -2.0).all() and (free["q50"] == -1.0).all()                      # scores may be negative


# ---------------------------------------------------------------- backtest
def plan(variants=None, **kw):
    base = dict(horizon=4, step=8, min_context=96, quantiles=Q, variants=variants or {"calendar": ["cal"], "detector": ["cal", "scene", "count"]},
                patterns=["scene.people_yolo", "count.yolo.*"], catalog_args=CATALOG_ARGS)
    return BacktestPlan(**(base | kw))


def test_origins_leave_enough_history_behind_and_a_full_horizon_ahead():
    p = plan()
    got = origins_for(200, p)
    assert got[0] == 95 and got[-1] + p.horizon < 200 and all(b - a == 8 for a, b in zip(got, got[1:]))
    assert origins_for(99, p) == [] and origins_for(100, p) == [95]                # 4 truth rows exactly fit


def test_the_model_never_sees_rows_after_the_origin_and_the_future_frame_is_only_the_clock():
    spy = SpyForecaster()
    run = run_backtest(make_table(), "s", [spy], plan(), settings=FS, table_settings=TS)
    assert run.origins == len(origins_for(200, plan())) and run.failures == {}
    for c in spy.calls:
        pos = int((c["origin"] - pd.Timestamp("2026-11-12 00:00")) / timedelta(minutes=15))
        assert c["people_max"] == pos                                      # target history ends exactly at the origin
        assert c["first_future"] == c["origin"] + timedelta(minutes=15)
        assert not [x for x in c["future_cols"] if not x.startswith("cal.") and x not in ("id", "timestamp")]
    assert {r for r in run.raw["origin"]} == {c["origin"] for c in spy.calls}


def test_scores_use_the_right_truth_slot_for_every_step():
    # people_yolo equals the row position, so persistence at origin p predicts p and the truth at step k is p + k
    run = run_backtest(make_table(), "s", [Persistence()], plan(patterns=["scene.people_yolo"]), settings=FS, table_settings=TS)
    raw = run.raw
    assert len(raw) == run.origins * 4 and (raw["abs_err"] == raw["step"]).all()
    assert (raw["y"] - raw["pred"] == raw["step"]).all()


def test_columns_are_judged_from_the_history_before_each_origin_not_the_whole_table():
    spy = SpyForecaster()
    run_backtest(make_table(), "s", [spy], plan(variants={"calendar": ["cal"]}), settings=FS, table_settings=TS)
    first, last = spy.calls[0], spy.calls[-1]
    assert "count.yolo.late" not in first["targets"]                 # still constant at the first origin
    assert "count.yolo.late" in last["targets"]                      # varies by the last one


def test_baselines_run_once_per_origin_and_models_once_per_variant():
    spy = SpyForecaster()
    run = run_backtest(make_table(), "s", [spy, Persistence()], plan(), settings=FS, table_settings=TS)
    by = run.raw.groupby(["forecaster", "variant"]).size()
    assert set(by.index) == {("spy", "calendar"), ("spy", "detector"), ("persistence", "-")}
    assert by[("spy", "calendar")] == by[("spy", "detector")] > 0                    # same targets, same points
    assert len(spy.calls) == 2 * run.origins


def test_a_failing_forecaster_is_skipped_at_those_origins_and_counted():
    spy = SpyForecaster(fail_when=lambda calls: calls in (1, 4))
    run = run_backtest(make_table(), "s", [spy], plan(variants={"calendar": ["cal"]}), settings=FS, table_settings=TS)
    assert run.failures == {"spy/calendar": 2} and run.raw["origin"].nunique() == run.origins - 2


def test_too_little_history_is_an_error_with_the_numbers():
    with pytest.raises(ForecastError, match="need at least 100"):
        run_backtest(make_table(60), "s", [Persistence()], plan(), settings=FS, table_settings=TS)


# ---------------------------------------------------------------- the Chronos-2 adapter (stubbed model)
class StubPipeline:
    def __init__(self, frame=None, error=None):
        self.frame, self.error, self.kwargs = frame, error, None

    def predict_df(self, df, **kw):
        self.kwargs = {"df": df} | kw
        if self.error:
            raise self.error
        return self.frame


def adapter(pipe, cross=False):
    f = Chronos2Forecaster.__new__(Chronos2Forecaster)
    f._pipe, f.cross_learning, f.name = pipe, cross, "chronos2:stub"
    return f


def raw_output(targets, ts, quantile_cols=("0.1", "0.5", "0.9")):
    rows = [{"id": "s", "timestamp": t, "target_name": g, "predictions": 5.0, **{q: float(i) + float(q) for q in quantile_cols}}
            for i, t in enumerate(ts) for g in targets]                       # interleaved targets, like a batch
    return pd.DataFrame(rows)


def test_chronos_adapter_maps_the_real_output_format_and_passes_the_request_through():
    ctx = context_of({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    fut = future_of(ctx, 3)
    pipe = StubPipeline(raw_output(["b", "a"], list(fut["timestamp"])))
    out = adapter(pipe, cross=True).forecast(ctx, fut, targets=["a", "b"], quantiles=Q)
    assert list(out.columns) == ["timestamp", "target", "step", "q10", "q50", "q90"]
    assert out["target"].tolist() == ["a"] * 3 + ["b"] * 3 and out["step"].tolist() == [1, 2, 3] * 2
    assert out["timestamp"].tolist() == list(fut["timestamp"]) * 2 and out["q50"].iloc[:3].tolist() == [0.5, 1.5, 2.5]
    k = pipe.kwargs
    assert k["prediction_length"] == 3 and k["target"] == ["a", "b"] and k["cross_learning"] is True
    assert k["quantile_levels"] == Q and k["id_column"] == "id" and k["timestamp_column"] == "timestamp" and k["future_df"] is fut


def test_chronos_adapter_turns_model_failures_and_missing_quantiles_into_forecast_errors():
    ctx = context_of({"a": [1.0, 2.0, 3.0]})
    fut = future_of(ctx, 2)
    with pytest.raises(ForecastError, match="forecasting failed: boom"):
        adapter(StubPipeline(error=RuntimeError("boom"))).forecast(ctx, fut, targets=["a"], quantiles=Q)
    short = StubPipeline(raw_output(["a"], list(fut["timestamp"]), quantile_cols=("0.1", "0.5")))
    with pytest.raises(ForecastError, match="no column for quantile 0.9"):
        adapter(short).forecast(ctx, fut, targets=["a"], quantiles=Q)


def test_the_backtest_and_live_runs_both_apply_the_non_negative_rule(conn, space):
    run = run_backtest(make_table(), "s", [Negative()], plan(patterns=["count.yolo.chair", "probe.m.people"],
                       variants={"calendar": ["cal"]}), settings=FS, table_settings=TS)
    by = run.raw.groupby("target")["pred"].agg(["min", "max"])
    assert by.loc["count.yolo.chair"].tolist() == [0.0, 0.0] and by.loc["probe.m.people"].tolist() == [-1.0, -1.0]
    counting = space.model_copy(update={"forecast_targets": ["count.yolo.chair", "probe.*"]})
    stored = load_forecast(conn, live(conn, counting, forecaster=Negative()).run_id).set_index("target")
    assert (stored.loc["count.yolo.chair", ["q10", "q50"]] == 0.0).all().all()
    assert (stored.loc["probe.m.people", "q10"] == -2.0).all()


# ---------------------------------------------------------------- storage and live runs
def test_forecast_rows_round_trip_with_their_run(conn):
    run_id = insert_run(conn, space_id="admin_cabin", origin_local="2026-11-12T10:00:00", model="m", variant="calendar",
                        horizon=2, step_minutes=15, context_slots=100, targets=["a"], covariates=["cal.*"], quantiles=Q)
    f = forecast_frame([("2026-11-12 10:15", "a", 1, 1.0, 2.0, 3.0), ("2026-11-12 10:30", "a", 2, 1.5, 2.5, 3.5)])
    assert insert_forecasts(conn, run_id, f) == 6
    back = load_forecast(conn, run_id)
    assert back[["q10", "q50", "q90"]].to_numpy().tolist() == [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]] and back["step"].tolist() == [1, 2]
    assert latest_run(conn, "admin_cabin")["run_id"] == run_id and latest_run(conn, "parking") is None
    with pytest.raises(StorageError, match="no quantile columns"):
        insert_forecasts(conn, run_id, f[["timestamp", "target", "step"]])
    with pytest.raises(Exception):                                         # the same point twice violates the key
        insert_forecasts(conn, run_id, f)
    assert load_forecast(conn, 999).empty


def live(conn, space, df=None, forecaster=None, **kw):
    return forecast_space(conn, space, result_for(df if df is not None else make_table()), forecaster or SpyForecaster(),
                          settings=kw.pop("settings", FS), table_settings=TS, **kw)


def test_live_forecast_stores_the_run_with_the_inputs_it_used(conn, space):
    run = live(conn, space, variant="calendar")
    row = conn.execute("SELECT * FROM forecast_runs WHERE run_id=?", (run.run_id,)).fetchone()
    assert (row["model"], row["variant"], row["horizon"], row["context_slots"]) == ("spy", "calendar", 16, 200)
    assert row["origin_local"] == "2026-11-14T01:45:00"                                # slot 199 of the table
    assert json.loads(row["covariates_json"]) == ["cal.*"] and json.loads(row["targets_json"]) == run.targets
    assert conn.execute("SELECT COUNT(*) FROM forecasts WHERE run_id=?", (run.run_id,)).fetchone()[0] == len(run.targets) * 16 * 3
    assert {"q10", "q50", "q90"} <= set(load_forecast(conn, run.run_id).columns)
    detector_run = live(conn, space, variant="detector", horizon=4)
    assert json.loads(conn.execute("SELECT covariates_json FROM forecast_runs WHERE run_id=?", (detector_run.run_id,)).fetchone()[0])[-1] == "cal.*"
    assert detector_run.covariates and len(detector_run.forecast) == len(detector_run.targets) * 4
    all_run = live(conn, space, horizon=4)                                             # the default: everything
    stored = json.loads(conn.execute("SELECT covariates_json, variant FROM forecast_runs WHERE run_id=?",
                                     (all_run.run_id,)).fetchone()[0])
    assert stored == all_run.covariates + ["cal.*"] and len(all_run.covariates) > len(detector_run.covariates)


def test_live_forecast_refuses_bad_requests_and_a_failed_model_stores_nothing(conn, space):
    with pytest.raises(ForecastError, match="need at least 96"):
        live(conn, space, make_table(50))
    with pytest.raises(ForecastError, match="unknown variant"):
        live(conn, space, variant="nope")
    with pytest.raises(ForecastError, match="no forecast_targets"):
        live(conn, space.model_copy(update={"forecast_targets": []}))
    with pytest.raises(ForecastError, match="model failed"):
        live(conn, space, forecaster=SpyForecaster(fail_when=lambda c: True))
    assert conn.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0] == 0


# ---------------------------------------------------------------- command line
def test_cli_forecast_and_backtest_end_to_end_with_baselines(root, space_dir, capsys):
    cfg = root / "config"
    (cfg / "pipeline.yaml").write_text((REPO / "config" / "pipeline.yaml").read_text())
    c = connect(root / "data" / "sbi.db")
    migrate(c)
    spaces = {s: __import__("sbi.spaces", fromlist=["load_space"]).load_space(cfg / "spaces" / f"{s}.yaml") for s in ("admin_cabin", "parking")}
    for s, cfg_s in spaces.items():
        register_space(c, cfg_s, f"config/spaces/{s}.yaml")
    sp = spaces["admin_cabin"]
    for i in range(130):
        fid = cap(c, sp, root, FakeSource([synthetic_image(i)]), thursday_ist(0, 0) + timedelta(minutes=15 * i)).frame_id
        people = [det("person", (0.1 + 0.1 * k, 0.2, 0.18 + 0.1 * k, 0.6)) for k in range(i % 3)]
        process_frame(c, sp, fid, FakeDetector(DETS + people), FakeReader(), root=root)
    c.close()
    assert main(["--root", str(root), "forecast", "--space", "admin_cabin", "--model", "seasonal_mean", "--horizon", "4"]) == 0
    out = capsys.readouterr().out
    assert "run 1 | seasonal_mean" in out and "scene.people_yolo" in out and "stored in forecast_runs" in out
    assert main(["--root", str(root), "backtest", "--space", "admin_cabin", "--models", "seasonal_mean,persistence",
                 "--horizon", "4", "--step", "12"]) == 0
    out = capsys.readouterr().out
    assert "3 origins every 12 slots" in out and "seasonal_mean" in out and "persistence" in out
    assert (root / "data" / "exports" / "admin_cabin_backtest.csv").exists()
    assert main(["--root", str(root), "backtest", "--space", "admin_cabin", "--models", "persistence", "--min-context", "500"]) == 2
    assert main(["--root", str(root), "forecast", "--space", "nope", "--model", "persistence"]) == 2

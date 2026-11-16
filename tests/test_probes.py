"""Text-prompt probes: config rules, score storage, pipeline/tick failure handling, backfill, concept columns."""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from sbi.errors import CameraError, ScorerError, StorageError
from sbi.eval.stability import stability
from sbi.forecast.probe_features import concept_values
from sbi.forecast.table import build_table
from sbi.imageio import read_frame_image
from sbi.perception.factory import build_scorer
from sbi.pipeline import process_frame
from sbi.scoring import score_missing
from sbi.settings import ScorerSettings
from sbi.spaces import SpaceConfig
from sbi.store.frames import get_frame
from sbi.store.scores import frames_missing_scores, load_scores, put_scores
from sbi.tick import run_tick

from .conftest import FakeDetector, FakeReader, FakeScorer, FakeSource, REPO, synthetic_image, thursday_ist
from .test_flow import DETS, cap, count


def space_with(space: SpaceConfig, probes) -> SpaceConfig:
    return space.model_copy(update={"probes": probes})


def frame(conn, space, root, now=None, seed=0):
    return cap(conn, space, root, FakeSource([synthetic_image(seed)]), now or thursday_ist()).frame_id


# ---------------------------------------------------------------- config rules
def test_shipped_space_files_define_probes_and_a_stable_prompt_order(space, parking):
    assert set(space.probes) >= {"people", "empty", "lit", "dark"} and set(parking.probes) >= {"bikes", "empty", "night"}
    assert len(space.probe_prompts) == sum(len(v) for v in space.probes.values()) == len(set(space.probe_prompts))
    assert space.probe_prompts[:3] == space.probes["people"]                           # file order, no sorting


@pytest.mark.parametrize("probes, message", [
    ({"Bad Name": ["x"]}, "snake_case"),
    ({"a": []}, "no prompts"),
    ({"a": ["  "]}, "1-200 characters"),
    ({"a": ["x" * 201]}, "1-200 characters"),
    ({"a": ["same"], "b": ["same"]}, "used by both"),
    ({"a": ["same", "same"]}, "repeats a prompt"),
])
def test_bad_probe_definitions_are_rejected_with_the_reason(space, probes, message):
    with pytest.raises(ValidationError, match=message):
        SpaceConfig.model_validate({**space.model_dump(), "probes": probes})


def test_scorer_settings_and_factory():
    assert ScorerSettings().kind == "none" and build_scorer(ScorerSettings(), None) is None
    assert ScorerSettings(model="/models/x/siglip2-large").version == "siglip2-large"


# ---------------------------------------------------------------- storage
def test_scores_replace_per_prompt_and_scorers_stay_apart(conn, space, parking, root):
    a, b = frame(conn, space, root), frame(conn, parking, root)
    put_scores(conn, a, "s1", {"p": 1.0, "q": 2.0})
    put_scores(conn, a, "s1", {"p": 5.0})                                # replaces p only
    put_scores(conn, a, "s2", {"p": -1.0})
    put_scores(conn, b, "s1", {"p": 9.0})
    assert load_scores(conn, "admin_cabin", "s1") == {a: {"p": 5.0, "q": 2.0}}
    assert load_scores(conn, "admin_cabin", "s2") == {a: {"p": -1.0}}
    assert load_scores(conn, "admin_cabin", "s1", frame_ids={b}) == {}
    assert load_scores(conn, "parking", "s1") == {b: {"p": 9.0}}
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="non-finite"):
            put_scores(conn, a, "s1", {"p": bad})
    assert load_scores(conn, "admin_cabin", "s1")[a]["p"] == 5.0           # the bad write changed nothing


def test_missing_scores_name_exactly_the_prompts_each_frame_lacks(conn, space, root):
    ids = [frame(conn, space, root, thursday_ist(10, m)) for m in (0, 15, 30)]
    cap(conn, space, root, FakeSource([CameraError("camera down")]), thursday_ist(10, 45))   # no image: never listed
    put_scores(conn, ids[0], "s", {"a": 1, "b": 1})
    put_scores(conn, ids[1], "s", {"a": 1})
    put_scores(conn, ids[2], "other", {"a": 1, "b": 1})
    assert frames_missing_scores(conn, "admin_cabin", "s", ["a", "b"]) == [(ids[1], ["b"]), (ids[2], ["a", "b"])]
    assert frames_missing_scores(conn, "admin_cabin", "s", []) == []


def test_read_frame_image_handles_relative_absolute_and_missing(conn, space, root):
    fid = frame(conn, space, root)
    row = get_frame(conn, fid)
    assert read_frame_image(row, root).shape == (180, 320, 3)
    abs_row = dict(row) | {"image_path": str(root / row["image_path"])}
    assert read_frame_image(abs_row, Path("/nowhere")).shape == (180, 320, 3)
    assert read_frame_image(dict(row) | {"image_path": None}, root) is None
    (root / row["image_path"]).unlink()
    assert read_frame_image(row, root) is None


# ---------------------------------------------------------------- concepts
def test_concept_is_the_mean_of_its_paraphrases_and_unknown_if_any_is_missing():
    groups = {"empty": ["a", "b", "c"], "busy": ["d"], "nothing": []}
    got = concept_values({"a": 1.0, "b": 2.0, "c": 6.0, "d": -2.0, "extra": 99.0}, groups)
    assert got["empty"] == pytest.approx(3.0) and got["busy"] == -2.0 and np.isnan(got["nothing"])
    assert np.isnan(concept_values({"a": 1.0, "b": 2.0}, groups)["empty"])          # c missing: not a 2-prompt mean


# ---------------------------------------------------------------- pipeline
def run(conn, space, root, scorer, now=None, seed=0):
    fid = frame(conn, space, root, now, seed)
    return fid, process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root, scorer=scorer)


def test_pipeline_asks_every_prompt_once_and_stores_each_score(conn, space, root):
    scorer = FakeScorer({space.probe_prompts[0]: 4.5})
    fid, res = run(conn, space, root, scorer)
    assert res.scored and scorer.asked == [space.probe_prompts]
    got = load_scores(conn, "admin_cabin", "fake-scorer")[fid]
    assert set(got) == set(space.probe_prompts) and got[space.probe_prompts[0]] == 4.5
    process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root, scorer=FakeScorer({space.probe_prompts[0]: -1.0}))
    assert count(conn, "frame_scores") == len(space.probe_prompts)                   # replaced, not duplicated
    assert load_scores(conn, "admin_cabin", "fake-scorer")[fid][space.probe_prompts[0]] == -1.0


class ShortScorer(FakeScorer):
    def score(self, image, prompts):
        return super().score(image, prompts)[:-1]


@pytest.mark.parametrize("scorer", [FakeScorer(error=ScorerError("gpu gone")), ShortScorer()], ids=["raises", "wrong count"])
def test_scorer_failure_keeps_the_frame_without_scores(conn, space, root, scorer):
    fid, res = run(conn, space, root, scorer)
    assert res.processed and not res.scored and count(conn, "observations", source="yolo") == 4
    assert count(conn, "frame_scores") == 0 and get_frame(conn, fid)["processed_utc"]


def test_a_failed_rerun_never_wipes_stored_scores(conn, space, root):
    fid, _ = run(conn, space, root, FakeScorer())
    res = process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root,
                        scorer=FakeScorer(error=ScorerError("down")))
    assert res.processed and not res.scored and count(conn, "frame_scores") == len(space.probe_prompts)


def test_no_scorer_or_no_probes_means_no_model_call_and_no_rows(conn, space, root):
    assert not run(conn, space, root, None)[1].scored and count(conn, "frame_scores") == 0
    scorer = FakeScorer()
    _, res = run(conn, space_with(space, {}), root, scorer, thursday_ist(10, 15))
    assert res.processed and not res.scored and scorer.asked == []


def test_tick_scores_every_space_and_survives_a_scorer_that_cannot_load(settings, space_dir, monkeypatch):
    out = run_tick(settings, sorted(space_dir.glob("*.yaml")), source=FakeSource([synthetic_image()]),
                   detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist(), scorer=FakeScorer())
    assert all(o.ok and o.processed == 1 for o in out)
    def broken(*a, **k):
        raise ScorerError("weights missing")
    monkeypatch.setattr("sbi.tick.build_scorer", broken)
    out = run_tick(settings, sorted(space_dir.glob("*.yaml")), source=FakeSource([synthetic_image(1)]),
                   detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist(10, 15))
    assert all(o.ok and o.processed == 1 for o in out)
    import sqlite3
    c = sqlite3.connect(settings.db_path)
    assert c.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 4
    assert c.execute("SELECT COUNT(DISTINCT frame_id) FROM frame_scores").fetchone()[0] == 2


# ---------------------------------------------------------------- backfill
def test_backfill_scores_history_and_a_new_prompt_asks_only_the_new_question(conn, space, root):
    ids = [frame(conn, space, root, thursday_ist(10, m), seed=m) for m in (0, 15)]
    scorer = FakeScorer()
    rep = score_missing(conn, space, scorer, root=root)
    assert (rep.done, rep.failed) == (2, 0) and all(a == space.probe_prompts for a in scorer.asked)
    assert score_missing(conn, space, FakeScorer(), root=root).done == 0                  # nothing missing now
    grown = space_with(space, {**space.probes, "cold": ["a freezing cold room"]})
    again = FakeScorer()
    assert score_missing(conn, grown, again, root=root).done == 2
    assert again.asked == [["a freezing cold room"]] * 2                                    # only the new prompt
    assert set(load_scores(conn, "admin_cabin", "fake-scorer")[ids[0]]) == set(grown.probe_prompts)


def test_backfill_counts_unreadable_images_and_scorer_errors_and_keeps_going(conn, space, root):
    ids = [frame(conn, space, root, thursday_ist(10, m), seed=m) for m in (0, 15, 30)]
    (root / get_frame(conn, ids[0])["image_path"]).unlink()
    class Flaky(FakeScorer):
        def score(self, image, prompts):
            if not self.asked:
                self.asked.append(prompts)
                raise ScorerError("one-off")
            return super().score(image, prompts)
    rep = score_missing(conn, space, Flaky(), root=root)
    assert (rep.done, rep.failed) == (1, 2) and list(load_scores(conn, "admin_cabin", "fake-scorer")) == [ids[2]]
    # the unreadable oldest frame keeps failing; a limited run still gets its one success from the next frame
    assert score_missing(conn, space, FakeScorer(), root=root, limit=1) == type(rep)(done=1, failed=1)


# ---------------------------------------------------------------- table columns
def build(conn, space, **kw):
    return build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, probe_scorer="fake-scorer",
                       probe_groups=space.probes, **kw)


def test_table_has_one_column_per_concept_averaging_paraphrases_and_nan_where_unscored(conn, space, root):
    first = space.probes["people"]
    for m, scorer in enumerate([FakeScorer({p: v for p, v in zip(first, (1.0, 2.0, 6.0))}),
                                FakeScorer(error=ScorerError("x")),
                                FakeScorer({p: v for p, v in zip(first, (4.0, 4.0, 4.0))})]):
        run(conn, space, root, scorer, thursday_ist(10, 0) + timedelta(minutes=15 * m), seed=m)
    res = build(conn, space)
    t, cat = res.table, res.catalog.set_index("column")
    cols = [c for c in t.columns if c.startswith("probe.")]
    assert cols == [f"probe.fake-scorer.{c}" for c in space.probes]
    people = t["probe.fake-scorer.people"]
    assert people.iloc[0] == pytest.approx(3.0) and np.isnan(people.iloc[1]) and people.iloc[2] == pytest.approx(4.0)
    assert t["slot_status"].eq("ok").all()                                                 # unscored never voids a slot
    assert cat.loc["probe.fake-scorer.people", "kind"] == "probe" and not cat.loc["probe.fake-scorer.people", "variant"]


def test_a_concept_with_a_prompt_missing_for_a_frame_is_unknown_for_that_frame(conn, space, root):
    cold = ["a freezing cold room", "an icy office"]
    grown = {**space.probes, "cold": cold}
    f1, _ = run(conn, space, root, FakeScorer(), thursday_ist(10, 0))
    f2, _ = run(conn, space, root, FakeScorer(), thursday_ist(10, 15), seed=1)
    put_scores(conn, f1, "fake-scorer", {cold[0]: 2.0, cold[1]: 4.0})                      # asked both paraphrases
    put_scores(conn, f2, "fake-scorer", {cold[0]: 2.0})                                    # only one of the two
    t = build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, probe_scorer="fake-scorer",
                    probe_groups=grown).table
    assert t["slot_status"].eq("ok").all()                                                 # both slots really are in play
    assert t["probe.fake-scorer.cold"].iloc[0] == pytest.approx(3.0) and np.isnan(t["probe.fake-scorer.cold"].iloc[1])
    assert t["probe.fake-scorer.people"].notna().all()                                     # the complete concepts are fine


def test_table_without_a_scorer_name_or_scores_has_no_probe_columns(conn, space, root):
    run(conn, space, root, FakeScorer(), thursday_ist(10, 0))
    run(conn, space, root, FakeScorer(), thursday_ist(10, 15), seed=1)
    no_name = build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, probe_groups=space.probes).table
    other = build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, probe_scorer="other",
                        probe_groups=space.probes).table
    no_groups = build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, probe_scorer="fake-scorer").table
    for t in (no_name, other, no_groups):
        assert not [c for c in t.columns if c.startswith("probe.")]


def test_probe_columns_appear_in_the_stability_report(conn, space, root):
    for i in range(12):
        run(conn, space, root, FakeScorer({space.probes["people"][0]: float(np.sin(i / 3))}),
            thursday_ist(10, 0) + timedelta(minutes=15 * i), seed=i)
    rep = stability(build(conn, space).table, start="2026-11-12 10:00", end="2026-11-12 11:00")
    assert "probe.fake-scorer.people" in set(rep.loc[rep["kind"] == "probe", "column"])

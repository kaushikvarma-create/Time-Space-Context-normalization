"""Whole-scene image vectors: codec and storage, the DINOv2 preprocessing, pipeline/tick failure handling,
backfill, the derived per-slot features and their table columns."""
from datetime import timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from sbi.embedding import embed_missing
from sbi.errors import CameraError, EmbedderError, StorageError
from sbi.eval.stability import stability
from sbi.forecast.embed_features import cosine_distance, embedding_features, principal_axes
from sbi.forecast.table import build_table, select_covariates
from sbi.perception.embedder_dinov2 import PATCH, combine, prepare
from sbi.perception.factory import build_embedder
from sbi.pipeline import process_frame
from sbi.settings import EmbedderSettings
from sbi.store.embeddings import decode_vector, encode_vector, frames_missing_embedding, load_embeddings, put_embedding
from sbi.tick import run_tick

from .conftest import FakeDetector, FakeEmbedder, FakeReader, FakeSource, synthetic_image, thursday_ist
from .test_flow import DETS, cap, count

E0, E1, E2 = np.eye(8, dtype=np.float32)[:3]


# ---------------------------------------------------------------- codec and storage
def test_vector_codec_round_trips_within_half_precision_and_rejects_bad_input():
    v = np.random.default_rng(0).normal(size=1536).astype(np.float32)
    v /= np.linalg.norm(v)
    back = decode_vector(encode_vector(v), 1536)
    assert back.dtype == np.float32 and np.abs(back - v).max() < 1e-3
    assert len(encode_vector(v)) == 2 * 1536
    for bad in (np.array([1.0, np.nan]), np.zeros((2, 2)), np.array([])):
        with pytest.raises(ValueError):
            encode_vector(bad)
    with pytest.raises(ValueError, match="corrupt"):
        decode_vector(encode_vector(v)[:-1], 1536)


def frame(conn, space, root, now=None, seed=0):
    return cap(conn, space, root, FakeSource([synthetic_image(seed)]), now or thursday_ist()).frame_id


def test_models_coexist_replace_and_stay_inside_their_space(conn, space, parking, root):
    a, b = frame(conn, space, root), frame(conn, parking, root)
    put_embedding(conn, a, "m1", E0)
    put_embedding(conn, a, "m2", E1)
    put_embedding(conn, a, "m1", E2)                                   # same model again: replaced, not duplicated
    put_embedding(conn, b, "m1", E1)
    assert count(conn, "frame_embeddings") == 3
    vecs, bad = load_embeddings(conn, "admin_cabin", "m1")
    assert bad == [] and list(vecs) == [a] and vecs[a][2] == pytest.approx(1.0, abs=1e-3)
    assert load_embeddings(conn, "admin_cabin", "m1", frame_ids={b})[0] == {}
    assert list(load_embeddings(conn, "parking", "m1")[0]) == [b]


def test_corrupt_and_mixed_dimension_rows(conn, space, root):
    a, b = frame(conn, space, root), frame(conn, space, root, thursday_ist(10, 15))
    put_embedding(conn, a, "m", E0)
    conn.execute("INSERT INTO frame_embeddings(frame_id, model, dim, vec) VALUES(?,?,?,?)", (b, "m", 8, b"\x00\x01"))
    vecs, corrupt = load_embeddings(conn, "admin_cabin", "m")
    assert list(vecs) == [a] and corrupt == [b]                                  # the bad row reads as unknown
    conn.execute("UPDATE frame_embeddings SET dim=4, vec=? WHERE frame_id=?", (encode_vector(E0[:4]), b))
    with pytest.raises(StorageError, match="mixed dimensions"):
        load_embeddings(conn, "admin_cabin", "m")


def test_frames_missing_embedding_skips_done_failed_captures_and_other_models(conn, space, root):
    ids = [frame(conn, space, root, thursday_ist(10, m)) for m in (0, 15, 30)]
    cap(conn, space, root, FakeSource([CameraError("camera down")]), thursday_ist(10, 45))    # failed capture: no image
    put_embedding(conn, ids[0], "m", E0)
    put_embedding(conn, ids[1], "other", E0)
    assert frames_missing_embedding(conn, "admin_cabin", "m") == ids[1:]
    assert frames_missing_embedding(conn, "parking", "m") == []


# ---------------------------------------------------------------- settings, factory, preprocessing
def test_embedder_settings_version_width_rule_and_factory():
    s = EmbedderSettings(kind="dinov2", model="facebook/dinov2-base", width=448)
    assert s.version == "dinov2-base@448" and EmbedderSettings(model="/models/x/dinov2-large").version == "dinov2-large@448"
    with pytest.raises(ValidationError, match="multiple of 14"):
        EmbedderSettings(width=450)
    assert EmbedderSettings().kind == "none" and build_embedder(EmbedderSettings(), None) is None


def test_prepare_keeps_the_whole_frame_at_a_patch_aligned_size():
    for shape, expect in (((1080, 1920, 3), (3, 252, 448)), ((1620, 2880, 3), (3, 252, 448)), ((600, 800, 3), (3, 336, 448)),
                          ((700, 1000, 3), (3, 308, 448))):                    # 313.6 px snaps to 22 patches
        x = prepare(np.zeros(shape, np.uint8), 448)
        assert x.shape == expect and x.dtype == np.float32 and expect[1] % PATCH == 0
    white = prepare(np.full((90, 160, 3), 255, np.uint8), 224)
    assert white[0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)          # channel 0 normalised
    with pytest.raises(ValueError):
        prepare(np.zeros((0, 5, 3), np.uint8), 224)


def test_prepare_converts_bgr_to_rgb():
    bgr = np.zeros((14, 14, 3), np.uint8)
    bgr[..., 0] = 255                                                              # pure blue in BGR
    x = prepare(bgr, 14)
    assert x[2].min() > x[0].max()                                                 # RGB channel 2 (blue) is the bright one


def test_combine_is_cls_then_patch_mean_each_unit_length():
    cls_t, patches = np.array([3.0, 4.0]), np.array([[2.0, 0.0], [0.0, 2.0]])
    v = combine(cls_t, patches)
    assert v.shape == (4,) and v[:2] == pytest.approx([0.6, 0.8]) and v[2:] == pytest.approx([2 ** -0.5] * 2)
    assert np.isfinite(combine(np.zeros(2), np.zeros((3, 2)))).all()               # zero vectors do not divide by 0


# ---------------------------------------------------------------- derived features
def test_cosine_distance_rows_and_missing_values():
    a = np.array([[1, 0], [1, 0], [np.nan, 1], [0, 0]], dtype=float)
    b = np.array([[1, 0], [0, 1], [1, 0], [1, 0]], dtype=float)
    d = cosine_distance(a, b)
    assert d[0] == pytest.approx(0) and d[1] == pytest.approx(1) and np.isnan(d[2]) and np.isnan(d[3])


def test_drift_prev_is_nan_next_to_a_missing_slot_and_exact_elsewhere():
    x = np.array([E0, E0, E1, [np.nan] * 8, E1, E1], dtype=float)
    f = embedding_features(x, components=1)
    assert np.isnan(f["drift_prev"][0]) and f["drift_prev"][1] == pytest.approx(0) and f["drift_prev"][2] == pytest.approx(1)
    assert np.isnan(f["drift_prev"][3]) and np.isnan(f["drift_prev"][4]) and f["drift_prev"][5] == pytest.approx(0)
    assert np.isnan(f["pc1"][3]) and np.isnan(f["drift_typical"][3])


def test_pc1_follows_the_main_direction_with_a_deterministic_sign():
    t = np.linspace(-1, 1, 25)
    axis = np.zeros(8); axis[1], axis[0] = 0.8, 0.6                                # largest loading (index 1) positive
    noise = np.random.default_rng(0).normal(scale=0.01, size=(25, 8))
    x = 5.0 + t[:, None] * axis + noise
    f = embedding_features(x, components=2)
    assert np.corrcoef(f["pc1"], t)[0, 1] > 0.99                                   # same sign every run
    assert abs(np.corrcoef(f["pc2"], t)[0, 1]) < 0.5                               # second direction is not the first
    assert np.nanmax(f["drift_typical"]) < 0.05 and f["drift_typical"][0] > f["drift_typical"][12]   # ends of the sweep are less typical
    mean, axes = principal_axes(x, 2)
    assert np.linalg.norm(axes[0]) == pytest.approx(1) and axes[0] @ axes[1] == pytest.approx(0, abs=1e-6)


def test_every_axis_has_its_largest_loading_positive_whatever_the_data():
    for seed in range(12):
        x = np.random.default_rng(seed).normal(size=(30, 10)) * np.arange(1, 11)
        _, axes = principal_axes(x, 4)
        assert all(a[np.argmax(np.abs(a))] > 0 for a in axes), seed


def test_fit_window_keeps_later_data_out_of_the_components():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(20, 8))
    later = rng.normal(size=(10, 8)) * 10
    fit = np.r_[np.ones(20, bool), np.zeros(10, bool)]
    a = embedding_features(np.vstack([base, later]), components=3, fit=fit)
    b = embedding_features(np.vstack([base, later * -3]), components=3, fit=fit)      # change only the held-out rows
    for k in ("pc1", "pc2", "pc3", "drift_typical"):
        assert np.allclose(a[k][:20], b[k][:20]), k                                  # the fit rows read the same
    assert not np.allclose(a["pc1"][20:], b["pc1"][20:])                             # held-out rows are still projected
    whole = embedding_features(np.vstack([base, later]), components=3)
    assert not np.allclose(whole["pc1"][:20], a["pc1"][:20])                         # and fitting on all really differs


def test_too_little_data_gives_unknown_not_zero_and_extra_components_are_nan():
    one = embedding_features(np.array([E0], dtype=float), components=2)
    assert all(np.isnan(v).all() for v in one.values())
    three = embedding_features(np.array([E0, E1, E2], dtype=float), components=6)    # 3 rows -> at most 2 directions
    assert np.isfinite(three["pc1"]).all() and np.isfinite(three["pc2"]).all() and np.isnan(three["pc3"]).all()
    same = embedding_features(np.tile(E0.astype(float), (5, 1)), components=2)
    assert np.allclose(same["pc1"], 0) and np.allclose(same["drift_prev"][1:], 0)    # nothing changed: constant, not NaN
    assert embedding_features(np.empty((0, 8)), components=2)["pc1"].shape == (0,)


# ---------------------------------------------------------------- pipeline
def run(conn, space, root, embedder, now=None, seed=0):
    fid = frame(conn, space, root, now, seed)
    return fid, process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root, embedder=embedder)


def test_pipeline_stores_one_vector_per_frame_and_model(conn, space, root):
    fid, res = run(conn, space, root, FakeEmbedder([E1]))
    assert res.embedded and res.processed
    vecs, _ = load_embeddings(conn, "admin_cabin", "fake-emb@8")
    assert vecs[fid][1] == pytest.approx(1.0, abs=1e-3)
    process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root, embedder=FakeEmbedder([E2]))
    assert count(conn, "frame_embeddings") == 1                                     # a re-run replaces it
    assert load_embeddings(conn, "admin_cabin", "fake-emb@8")[0][fid][2] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("embedder", [FakeEmbedder(error=EmbedderError("gpu gone")), FakeEmbedder([[1.0, 2.0]])],
                         ids=["raises", "wrong size"])
def test_embedder_failure_keeps_the_frame_without_a_vector(conn, space, root, embedder):
    fid, res = run(conn, space, root, embedder)
    assert res.processed and not res.embedded and count(conn, "observations", source="yolo") == 4
    assert count(conn, "frame_embeddings") == 0
    assert conn.execute("SELECT processed_utc FROM frames WHERE frame_id=?", (fid,)).fetchone()[0]


def test_a_failed_rerun_never_wipes_a_stored_vector(conn, space, root):
    fid, _ = run(conn, space, root, FakeEmbedder([E1]))
    res = process_frame(conn, space, fid, FakeDetector(DETS), FakeReader(), root=root,
                        embedder=FakeEmbedder(error=EmbedderError("down")))
    assert res.processed and not res.embedded and count(conn, "frame_embeddings") == 1


def test_without_an_embedder_nothing_is_written(conn, space, root):
    _, res = run(conn, space, root, None)
    assert res.processed and not res.embedded and count(conn, "frame_embeddings") == 0


def test_tick_embeds_every_space_and_survives_an_embedder_that_cannot_load(settings, space_dir, monkeypatch):
    out = run_tick(settings, sorted(space_dir.glob("*.yaml")), source=FakeSource([synthetic_image()]),
                   detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist(), embedder=FakeEmbedder())
    assert all(o.ok and o.processed == 1 for o in out)
    def broken(*a, **k):
        raise EmbedderError("weights missing")
    monkeypatch.setattr("sbi.tick.build_embedder", broken)
    out = run_tick(settings, sorted(space_dir.glob("*.yaml")), source=FakeSource([synthetic_image(1)]),
                   detector=FakeDetector(DETS), reader=FakeReader(), now=thursday_ist(10, 15))
    assert all(o.ok and o.processed == 1 for o in out)                             # capture and counting carry on
    import sqlite3
    c = sqlite3.connect(settings.db_path)
    assert c.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 4
    assert c.execute("SELECT COUNT(*) FROM frame_embeddings").fetchone()[0] == 2


# ---------------------------------------------------------------- backfill
def test_backfill_fills_only_what_is_missing_and_reports_failures(conn, space, root):
    ids = [frame(conn, space, root, thursday_ist(10, m), seed=m) for m in (0, 15, 30, 45)]
    put_embedding(conn, ids[0], "fake-emb@8", E0)                                 # already done: not touched
    (root / conn.execute("SELECT image_path FROM frames WHERE frame_id=?", (ids[1],)).fetchone()[0]).unlink()
    emb = FakeEmbedder()
    rep = embed_missing(conn, "admin_cabin", emb, root=root)
    assert (rep.done, rep.failed) == (2, 1) and emb.calls == 2                    # unreadable image counted, run goes on
    assert sorted(load_embeddings(conn, "admin_cabin", "fake-emb@8")[0]) == [ids[0], ids[2], ids[3]]
    again = embed_missing(conn, "admin_cabin", FakeEmbedder(error=EmbedderError("x")), root=root)
    assert (again.done, again.failed) == (0, 1)                                    # only the unreadable one is left
    assert embed_missing(conn, "admin_cabin", FakeEmbedder(), root=root, limit=0).done == 0
    stuck = [frame(conn, space, root, thursday_ist(11, m), seed=m) for m in (0, 15)]      # newer frames, still missing
    (root / conn.execute("SELECT image_path FROM frames WHERE frame_id=?", (ids[1],)).fetchone()[0]).unlink(missing_ok=True)
    one = embed_missing(conn, "admin_cabin", FakeEmbedder(), root=root, limit=1)
    assert (one.done, one.failed) == (1, 1) and sorted(load_embeddings(conn, "admin_cabin", "fake-emb@8")[0])[-1] == stuck[0]


def test_backfill_survives_an_embedder_error_on_one_frame(conn, space, root):
    ids = [frame(conn, space, root, thursday_ist(10, m), seed=m) for m in (0, 15)]
    class Flaky(FakeEmbedder):
        def embed(self, image):
            if self.calls == 0:
                self.calls += 1
                raise EmbedderError("one-off")
            return super().embed(image)
    rep = embed_missing(conn, "admin_cabin", Flaky(), root=root)
    assert (rep.done, rep.failed) == (1, 1) and list(load_embeddings(conn, "admin_cabin", "fake-emb@8")[0]) == [ids[1]]


# ---------------------------------------------------------------- table columns
def build(conn, **kw):
    return build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, embed_model="fake-emb@8",
                       embed_components=2, **kw)


def test_table_has_image_vector_columns_and_nan_where_no_vector(conn, space, root):
    vecs = [E0, E0 + 0.1 * E1, None, 3 * E1 + E2, E1]
    for m, v in zip((0, 15, 30, 45, 60), vecs):
        run(conn, space, root, FakeEmbedder([v]) if v is not None else FakeEmbedder(error=EmbedderError("x")),
            thursday_ist(10, 0) + timedelta(minutes=m), seed=m)
    res = build(conn)
    t, cat = res.table, res.catalog.set_index("column")
    cols = [c for c in t.columns if c.startswith("emb.")]
    assert cols == ["emb.fake-emb.pc1", "emb.fake-emb.pc2", "emb.fake-emb.drift_prev", "emb.fake-emb.drift_typical"]
    assert t["emb.fake-emb.pc1"].isna().tolist() == [False, False, True, False, False]
    assert t["emb.fake-emb.drift_prev"].isna().tolist() == [True, False, True, True, False]   # needs both slots
    assert t["emb.fake-emb.drift_prev"].iloc[1] == pytest.approx(1 - 1 / 1.01 ** 0.5, abs=1e-4) and t["emb.fake-emb.drift_prev"].iloc[4] == pytest.approx(1 - 3 / 10 ** 0.5, abs=1e-4)
    assert cat.loc["emb.fake-emb.pc1", "kind"] == "emb" and not cat.loc["emb.fake-emb.pc1", "variant"]
    assert t["slot_status"].eq("ok").all()                                          # a missing vector never voids the slot


def test_table_without_a_model_or_vectors_has_no_image_columns(conn, space, root):
    run(conn, space, root, FakeEmbedder([E0]), thursday_ist(10, 0))
    run(conn, space, root, FakeEmbedder([E1]), thursday_ist(10, 15), seed=1)
    assert not [c for c in build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1).table.columns
                if c.startswith("emb.")]                                            # no model named
    other = build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, embed_model="some-other@1")
    assert not [c for c in other.table.columns if c.startswith("emb.")]            # nothing stored for that model


def test_table_fit_end_changes_what_the_components_are_fit_on(conn, space, root):
    rng = np.random.default_rng(3)
    for i in range(8):
        run(conn, space, root, FakeEmbedder([rng.normal(size=8) * (1 if i < 5 else 20)]),
            thursday_ist(10, 0) + timedelta(minutes=15 * i), seed=i)
    whole = build(conn).table["emb.fake-emb.pc1"]
    early = build(conn, embed_fit_end="2026-11-12 11:15").table["emb.fake-emb.pc1"]
    assert early.notna().all() and not np.allclose(whole.iloc[:5], early.iloc[:5])


def test_corrupt_vector_is_a_gap_with_a_warning(conn, space, root, caplog):
    ids = [run(conn, space, root, FakeEmbedder([v]), thursday_ist(10, 15 * i), seed=i)[0]
           for i, v in enumerate((E0, E1, E2))]
    conn.execute("UPDATE frame_embeddings SET vec=? WHERE frame_id=?", (b"\x00", ids[1]))
    with caplog.at_level("WARNING"):
        t = build(conn).table
    assert t["emb.fake-emb.pc1"].isna().tolist() == [False, True, False] and "corrupt" in caplog.text


def test_image_vector_columns_are_eligible_inputs_and_in_the_stability_report(conn, space, root):
    for i in range(12):
        run(conn, space, root, FakeEmbedder([np.cos(i / 3) * E0 + np.sin(i / 3) * E1 + 0.01 * (i % 2) * E2]),
            thursday_ist(10, 0) + timedelta(minutes=15 * i), seed=i)
    res = build(conn)
    cat = res.catalog.set_index("column")
    assert cat.loc["emb.fake-emb.pc1", "model_ready"]
    sel = select_covariates(res.catalog, targets=[], max_covariates=24, always=[], min_autocorr=0.3, autocorr_min_slots=1000)
    assert "emb.fake-emb.pc1" in sel
    rep = stability(res.table, start="2026-11-12 10:00", end="2026-11-12 11:00")
    got = set(rep.loc[rep["kind"] == "emb", "column"])
    assert {"emb.fake-emb.pc1", "emb.fake-emb.drift_prev"} <= got

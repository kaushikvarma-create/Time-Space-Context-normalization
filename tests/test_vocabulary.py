"""Open vocabulary: taxonomy, curation, geometry links, prompt v3 relations, and the Chronos-ready table."""
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from sbi.curation.curator import curate, pick_canonical, set_link_status
from sbi.errors import SbiError, VLMError
from sbi.forecast.table import build_table, to_chronos
from sbi.perception.geometry import overlap_links
from sbi.perception.scene_lmstudio import PREDICATES, build_prompt, build_schema, parse_reading
from sbi.perception.types import Relation
from sbi.pipeline import process_frame
from sbi.store.db import transaction
from sbi.taxonomy import Taxonomy, load_taxonomy, reresolve_observations, singular_candidates

from .conftest import FakeDetector, FakeReader, FakeSource, det, make_reading, synthetic_image, thursday_ist
from .test_flow import DETS, cap


def run_frame(conn, space, root, reading=None, dets=DETS, now=None, reader_error=None):
    c = cap(conn, space, root, FakeSource([synthetic_image()]), now or thursday_ist())
    process_frame(conn, space, c.frame_id, FakeDetector(dets),
                  FakeReader(reading, error=reader_error) if (reading or reader_error) else FakeReader(), root=root)
    return c.frame_id


def reading_with(objects=(), relations=(), people=2):
    r = make_reading(people=people, objects=objects)
    return type(r)(**{**r.__dict__, "relations": tuple(Relation(*x) for x in relations)})


# ---------------------------------------------------------------- taxonomy
def test_singular_candidates_do_not_mangle_double_s():
    assert "glass" in singular_candidates("glasses") and singular_candidates("boss") == ["boss"]


def test_taxonomy_resolution_follows_aliases_merges_and_is_a_closure():
    t = Taxonomy(known={"phone", "mobile_phone", "bag", "backpack", "item"}, alias={"cell phone": "mobile_phone"},
                 merged_into={"mobile_phone": "phone"}, parents={"backpack": {"bag"}, "bag": {"item"}, "item": {"backpack"}})
    assert t.resolve("Cell Phones") == "phone"            # alias -> merged chain, plural, case
    assert t.resolve("mobile_phone") == "phone"
    assert t.lookup("unicorn") is None and t.resolve("Unicorn Horn") == "unicorn_horn"
    assert t.ancestors("backpack") == {"bag", "item"}     # transitive, and the bag->item->backpack cycle terminates


# ---------------------------------------------------------------- curation (fake models)
class FakeEmbedder:
    name = "fake-embed"

    def __init__(self, groups):
        self.groups = groups                               # names in the same group get identical vectors;
        self.solo: dict[str, int] = {}                     # every other name gets its own orthogonal vector

    def embed(self, texts):
        vs = []
        for t in texts:
            g = next((i for i, grp in enumerate(self.groups) if t in grp), None)
            if g is None:
                g = len(self.groups) + self.solo.setdefault(t, len(self.solo))
            v = np.zeros(512); v[g] = 1.0; vs.append(v)
        return np.array(vs)


class FakeJudge:
    name = "fake-judge"

    def __init__(self, yes):
        self.yes, self.calls = set(yes), []

    def is_every(self, a, b):
        self.calls.append((a, b)); return (a, b) in self.yes


def seed_names(conn, *names):
    with transaction(conn):
        for n in names:
            conn.execute("INSERT OR IGNORE INTO object_registry(object_type,category,review_status) VALUES(?,'other','auto_new')", (n,))


def test_curate_merges_synonyms_links_hierarchy_and_remaps_history(conn, space, root):
    fid = run_frame(conn, space, root, reading_with(objects=(("smartphone", 2, "idle"), ("handset", 1, "idle"),
                                                             ("rucksack", 1, "present"))))
    emb = FakeEmbedder([{"smartphone", "handset", "phone"}, {"rucksack", "bag"}])
    judge = FakeJudge({("smartphone", "handset"), ("handset", "smartphone"), ("handset", "phone"),
                       ("smartphone", "phone"), ("rucksack", "bag")})
    rep = curate(conn, emb, judge, min_similarity=0.9, auto_approve=True)
    rels = {(k["child"], k["relation"], k["parent"]) for k in rep.links}
    assert ("rucksack", "is_a", "bag") in rels
    assert {r[1] for r in rels if {r[0], r[2]} == {"smartphone", "handset"}} == {"same"} and rep.reresolved_rows >= 1
    tax = load_taxonomy(conn)
    merged = tax.resolve("handset")
    assert tax.resolve("smartphone") == merged and tax.ancestors(merged) == {"phone"}    # one dimension, under phone
    stored = {r[0] for r in conn.execute("SELECT object_type FROM observations WHERE frame_id=? AND source='vlm'", (fid,))}
    assert not ({"smartphone", "handset"} <= stored)                         # history re-mapped to one name
    raws = {r[0] for r in conn.execute("SELECT raw_label FROM observations WHERE frame_id=? AND source='vlm'", (fid,))}
    assert {"smartphone", "handset"} <= raws                                 # raw labels preserved forever
    asked = len(judge.calls)
    curate(conn, emb, judge, min_similarity=0.9, auto_approve=True)         # second run: nothing re-asked
    assert len(judge.calls) == asked


def test_unrelated_pairs_never_link_and_proposals_need_approval(conn):
    seed_names(conn, "desk", "keyboard")
    judge = FakeJudge(set())
    rep = curate(conn, FakeEmbedder([{"desk", "keyboard"}]), judge, min_similarity=0.9, auto_approve=False)
    assert rep.links == [] and ("desk", "keyboard") in judge.calls and ("keyboard", "desk") in judge.calls
    seed_names(conn, "mobile_device", "tablet")
    rep = curate(conn, FakeEmbedder([{"mobile device", "tablet"}]), FakeJudge({("tablet", "mobile device")}),
                 min_similarity=0.9, auto_approve=False)
    assert rep.links[0]["status"] == "proposed" and load_taxonomy(conn).ancestors("tablet") == set()
    set_link_status(conn, rep.links[0]["link_id"], "approved")
    assert load_taxonomy(conn).ancestors("tablet") == {"mobile_device"}


def test_rejecting_a_merge_restores_both_dimensions(conn, space, root):
    run_frame(conn, space, root, reading_with(objects=(("projector", 1, "on"), ("beamer", 1, "on"))))
    rep = curate(conn, FakeEmbedder([{"projector", "beamer"}]), FakeJudge({("projector", "beamer"), ("beamer", "projector")}),
                 min_similarity=0.9, auto_approve=True)
    link = next(k for k in rep.links if k["relation"] == "same")
    assert load_taxonomy(conn).resolve("beamer") == load_taxonomy(conn).resolve("projector")
    out = set_link_status(conn, link["link_id"], "rejected")
    tax = load_taxonomy(conn)
    assert tax.resolve("beamer") != tax.resolve("projector") and out["reresolved_rows"] >= 1
    with pytest.raises(ValueError):
        set_link_status(conn, 99999, "approved")


def test_canonical_prefers_configured_names(conn):
    seed_names(conn, "cellphone")
    for _ in range(3):
        with transaction(conn):
            conn.execute("INSERT INTO frames(space_id,ts_utc,ts_local,tz,date,dow,dow_name,hour,minute,slot_15,week_of_year,"
                         "is_weekend,part_of_day,capture_status) VALUES('admin_cabin',hex(randomblob(4)),'t','IST','d',0,'Mon',"
                         "0,0,0,1,0,'night','ok')")
            fid = conn.execute("SELECT max(frame_id) FROM frames").fetchone()[0]
            conn.execute("INSERT INTO observations(frame_id,space_id,ts_utc,object_type,raw_label,source) "
                         "VALUES(?,'admin_cabin','t','cellphone','cellphone','vlm')", (fid,))
    assert pick_canonical(conn, "cellphone", "phone") == ("phone", "cellphone")   # seed beats more-observed auto_new


# ---------------------------------------------------------------- geometry + prompt v3
def test_overlap_links_are_symmetric_counted_and_ignore_same_label():
    d = [det("person", (0.1, 0.1, 0.3, 0.6)), det("chair", (0.15, 0.3, 0.3, 0.7)),
         det("person", (0.6, 0.1, 0.8, 0.6)), det("chair", (0.62, 0.3, 0.78, 0.7)), det("bag", (0.9, 0.9, 0.95, 0.95))]
    assert overlap_links(d) == [("chair", "overlaps", "person", 2)]


def test_prompt_v3_has_no_name_list_and_schema_has_relations(space):
    p = build_prompt(space)
    for name in ("whiteboard", "plant", "desktop_pc", "sofa"):        # vocabulary names must not prime the VLM
        assert name not in p
    assert build_schema(space)["properties"]["relations"]["items"]["properties"]["predicate"]["enum"] == PREDICATES


def test_parse_keeps_valid_relations_and_drops_malformed(space):
    data = {"people_count": 1, "people": [], "objects": [], "lighting": "dim", "door_state": "open", "tidiness": "ok",
            "summary": "", "relations": [{"subject": "person", "predicate": "using", "object": "laptop", "count": 1},
                                         {"subject": "person", "predicate": "teleporting", "object": "x", "count": 1},
                                         {"subject": "", "predicate": "on", "object": "desk", "count": 1},
                                         {"subject": "bag", "predicate": "on", "object": "floor", "count": 0}]}
    r = parse_reading(json.dumps(data), space, model="m", latency_s=0)
    assert r.relations == (Relation("person", "using", "laptop", 1),)


# ---------------------------------------------------------------- pipeline storage
def test_pipeline_stores_raw_labels_vlm_relations_and_geometry_links(conn, space, root):
    dets = [det("person", (0.1, 0.1, 0.3, 0.6)), det("chair", (0.15, 0.3, 0.3, 0.7))]
    fid = run_frame(conn, space, root, reading_with(objects=(("Office Chairs", 2, "idle"),),
                                                    relations=(("person", "sitting_on", "office chair", 1),)), dets=dets)
    raw = dict(conn.execute("SELECT raw_label, object_type FROM observations WHERE frame_id=? AND source='vlm' "
                            "AND raw_label!='person'", (fid,)).fetchall())
    assert raw == {"Office Chairs": "chair"}                               # alias 'office chair' -> chair, plural handled
    rels = {(r[0], r[1], r[2], r[3]) for r in conn.execute("SELECT source, subject, predicate, object FROM relations "
                                                           "WHERE frame_id=?", (fid,))}
    assert rels == {("vlm", "person", "sitting_on", "chair"), ("geometry", "chair", "overlaps", "person")}
    process_frame(conn, space, fid, FakeDetector([]), FakeReader(make_reading(0, ())), root=root)
    assert conn.execute("SELECT COUNT(*) FROM relations WHERE frame_id=?", (fid,)).fetchone()[0] == 0   # replaced


# ---------------------------------------------------------------- the Chronos-ready table
def build(conn, **kw):
    return build_table(conn, "admin_cabin", tz="Asia/Kolkata", min_valid_slots=1, **kw)


def test_table_grid_gaps_missing_semantics_and_vlm_failure(conn, space, root):
    from sbi.errors import CameraError
    run_frame(conn, space, root, reading_with(objects=(("chair", 3, "idle"),)), now=thursday_ist(10, 0))
    run_frame(conn, space, root, reading_with(objects=(("chair", 3, "idle"), ("helmet", 1, "present"))), now=thursday_ist(10, 2))  # same slot, later
    cap(conn, space, root, FakeSource([CameraError("down")]), thursday_ist(10, 15))
    # 10:30 has no frame at all
    run_frame(conn, space, root, reader_error=VLMError("lm studio down"), now=thursday_ist(10, 45))
    run_frame(conn, space, root, reading_with(objects=(("chair", 3, "idle"), ("pen", 2, "present"))), now=thursday_ist(11, 0))
    t = build(conn).table
    assert list(t.index.strftime("%H:%M")) == ["10:00", "10:15", "10:30", "10:45", "11:00"]   # regular grid, IST, naive
    assert list(t["slot_status"]) == ["ok", "camera_down", "no_frame", "ok", "ok"]
    assert t.loc["2026-11-12 10:00", "count.vlm.chair"] == 3 and "count.vlm.helmet" not in t   # earliest frame per slot wins
    assert t.loc["2026-11-12 10:00", "count.vlm.pen"] == 0 and t.loc["2026-11-12 11:00", "count.vlm.pen"] == 2  # absent = 0
    assert t[["count.yolo.person", "count.vlm.chair"]].iloc[1:3].isna().all().all()         # missing is NaN, never 0
    assert t.loc["2026-11-12 10:45", "count.yolo.person"] == 2                              # detector still counts
    assert np.isnan(t.loc["2026-11-12 10:45", "count.vlm.chair"])                           # VLM failed -> unknown
    assert pd.isna(t.loc["2026-11-12 10:45", "cat.lighting"]) and t.loc["2026-11-12 10:00", "cat.lighting"] == "artificial"
    assert t["cal.hour"].tolist() == [10, 10, 10, 10, 11] and t["cal.dow"].iloc[0] == 3       # calendar on every row


def test_table_rollups_cells_relations_and_catalog(conn, space, root):
    with transaction(conn):
        conn.execute("INSERT INTO object_links(child,parent,relation,status,source,created_utc) "
                     "VALUES('backpack','bag','is_a','approved','human','t')")
        conn.execute("INSERT OR IGNORE INTO object_registry(object_type,category,review_status) VALUES('backpack','item','auto_new')")
    overlapping = DETS + [det("person", (0.42, 0.25, 0.52, 0.85))]            # this person overlaps the chair box
    for i in range(10):
        objs = (("bag", 1, "present"), ("backpack", 2, "present")) + ((("pen", 1, "present"),) if i == 0 else ())
        run_frame(conn, space, root, reading_with(objects=objs, relations=(("person", "using", "laptop", 1),)),
                  dets=overlapping, now=thursday_ist(10, 0) + timedelta(minutes=15 * i))
    res = build(conn)
    t, cat = res.table, res.catalog.set_index("column")
    assert (t["rollup.vlm.bag"] == 3).all() and (t["count.vlm.bag"] == 1).all()           # bag + backpack roll-up
    assert "rollup.vlm.backpack" not in t and "rollup.vlm.chair" not in t                   # no roll-up without children
    for col in ("cell.yolo.R3C2.bag", "rel.vlm.person.using.laptop", "rel.geometry.chair.overlaps.person"):
        assert col in t, col
    assert cat.loc["cal.hour", "model_ready"] and cat.loc["cal.hour", "known_future"]
    assert not cat.loc["count.vlm.bag", "model_ready"] and cat.loc["count.vlm.bag", "reason"] == "constant"
    assert cat.loc["count.vlm.pen", "present_slots"] == 1 and not cat.loc["count.vlm.pen", "model_ready"]
    assert "flicker" in cat.loc["count.vlm.pen", "reason"]                       # seen once = flicker, kept in table


def test_structural_objects_are_kept_but_not_eligible_and_selection_is_capped(conn, space, root):
    from sbi.forecast.table import select_covariates
    for i in range(12):
        objs = (("wall", 1 + i % 3, "present"), ("chair", 3 + i % 2, "idle"), ("cup", i % 4, "present"))
        run_frame(conn, space, root, reading_with(objects=objs, relations=(("bag", "on", "floor", 1 + i % 2),)),
                  now=thursday_ist(9, 0) + timedelta(minutes=15 * i))
    res = build(conn, exclude_labels=("wall", "floor"))
    cat = res.catalog.set_index("column")
    assert "count.vlm.wall" in res.table and cat.loc["count.vlm.wall", "reason"] == "structural object (building part)"
    assert cat.loc["rel.vlm.bag.on.floor", "model_ready"]                         # relations to building parts stay
    sel = select_covariates(res.catalog, targets=["scene.people_vlm"], max_covariates=3,
                            always=["count.vlm.chair", "not_a_column"], min_autocorr=0.3, autocorr_min_slots=1000)
    assert sel[0] == "count.vlm.chair" and len(sel) == 3 and "scene.people_vlm" not in sel
    assert all(c in set(res.catalog.loc[res.catalog.model_ready, "column"]) for c in sel)


def test_to_chronos_shapes_match_predict_df_contract(conn, space, root):
    for i in range(10):
        run_frame(conn, space, root, reading_with(objects=(("chair", 3 + i % 2, "idle"),), people=i % 3),
                  now=thursday_ist(10, 0) + timedelta(minutes=15 * i))
    ctx, fut = to_chronos(build(conn), "admin_cabin", targets=["scene.people_vlm"], horizon=4)
    assert list(ctx.columns[:3]) == ["id", "timestamp", "scene.people_vlm"] and (ctx["id"] == "admin_cabin").all()
    assert len(ctx) == 10 and pd.infer_freq(ctx["timestamp"]) in ("15min", "15T")
    assert list(fut.columns) == ["id", "timestamp"] + [c for c in ctx.columns if c.startswith("cal.")]
    assert len(fut) == 4 and fut["timestamp"].iloc[0] == pd.date_range(ctx["timestamp"].iloc[-1], periods=2, freq="15min")[1]
    assert "scene.people_vlm" not in fut                                                       # no targets in future_df
    with pytest.raises(ValueError):
        to_chronos(build(conn), "admin_cabin", targets=["nope"], horizon=4)


def test_table_requires_data(conn):
    with pytest.raises(SbiError, match="no frames"):
        build(conn)

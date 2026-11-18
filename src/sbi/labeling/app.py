"""Blind ground-truth labelling UI.   Run:  scripts/label.sh
You count objects in the photo WITHOUT seeing what the models said (avoids anchoring). After saving, the
model counts are revealed next to yours. Times shown are in the space's local timezone (IST)."""
from __future__ import annotations

import os

import streamlit as st

from sbi.errors import SbiError
from sbi.labeling import core
from sbi.settings import load_settings
from sbi.spaces import discover_space_files, load_space
from sbi.store.db import connect, migrate

st.set_page_config(layout="wide", page_title="Label frames")


@st.cache_resource
def _init():
    """Settings and spaces are shared across runs; the database connection is NOT: Streamlit runs each page run in
    its own thread and an SQLite connection may only be used by the thread that opened it."""
    settings = load_settings(os.environ.get("SBI_ROOT"))
    first = connect(settings.db_path)
    try:
        migrate(first)
    finally:
        first.close()
    spaces = {s.space_id: s for s in (load_space(f) for f in discover_space_files(settings.root))}
    return settings, spaces


try:
    settings, spaces = _init()
    conn = connect(settings.db_path)              # one per page run, in this run's thread; closed when the run's
                                                  # script namespace is discarded (SQLite closes on garbage collection)
except SbiError as e:
    st.error(f"Setup problem: {e}")
    st.stop()

space_id = st.sidebar.selectbox("Space", list(spaces))
labeler = st.sidebar.text_input("Your name / initials", value="me")
space = spaces[space_id]
types = space.label_types
if not types:
    st.warning(f"{space_id} has no label_types in its config; nothing to label.")
    st.stop()
done, total = core.progress(conn, space_id)
st.sidebar.progress(min(done / max(total, 1), 1.0), text=f"{done} labelled of {total} available")
st.sidebar.caption("Target: about 50 per space, spread over different hours and days.")

ss = st.session_state
if ss.get("space") != space_id:
    ss.space, ss.fid, ss.revealed = space_id, core.next_unlabeled(conn, space_id), None
fid = ss.fid
if fid is None:
    st.success("No unlabelled frames left for this space. Come back after more data has been collected.")
    st.stop()

row = conn.execute("SELECT ts_local,dow_name,part_of_day FROM frames WHERE frame_id=?", (fid,)).fetchone()
st.subheader(f"{space_id} | frame {fid} | {row['dow_name']} {row['ts_local'][:16].replace('T', ' ')} IST "
             f"({row['part_of_day']})")
try:
    img = core.load_frame_image(conn, fid, settings.root)
except SbiError as e:
    st.error(str(e))
    if st.button("Skip this frame"):
        core.save_label(conn, fid, {}, usable=False, labeler=labeler, note=f"unreadable: {e}")
        ss.fid = core.next_unlabeled(conn, space_id)
        st.rerun()
    st.stop()

left, right = st.columns([3, 1])
if ss.revealed == fid:
    left.image(core.to_rgb_small(core.draw_boxes(conn, fid, img)), caption="Yellow boxes = YOLO detections")
    mc, yours = core.model_counts(conn, fid, types), ss.last_counts
    right.markdown("**You vs models**")
    right.table({"type": types, "you": [yours[t] for t in types], "yolo": [mc[t]["yolo"] for t in types],
                 "vlm": ["n/a" if mc[t]["vlm"] is None else mc[t]["vlm"] for t in types]})
    if right.button("Next frame", type="primary"):
        ss.fid, ss.revealed = core.next_unlabeled(conn, space_id), None
        st.rerun()
else:
    tabs = left.tabs(["Whole photo"] + [name for name, _ in core.quadrants(img)])
    tabs[0].image(core.to_rgb_small(img))
    for tab, (_, part) in zip(tabs[1:], core.quadrants(img)):
        tab.image(core.to_rgb_small(part, width=part.shape[1]))
    left.caption("Dense area? Count it in the quarter tabs (full resolution, no overlap), then add the quarters up.")
    with right.form("label_form"):
        st.markdown("**Count what you can see**")
        counts = {t: st.number_input(t, min_value=0, max_value=200, value=0, step=1, key=f"{fid}_{t}") for t in types}
        st.caption("Count partly hidden objects too if you can tell they are there.")
        unusable = st.checkbox("Too unclear to label (skip)")
        note = st.text_input("Note (optional)")
        if st.form_submit_button("Save", type="primary"):
            core.save_label(conn, fid, counts, usable=not unusable, labeler=labeler, note=note)
            ss.last_counts, ss.revealed = counts, fid
            if unusable:
                ss.fid, ss.revealed = core.next_unlabeled(conn, space_id), None
            st.rerun()

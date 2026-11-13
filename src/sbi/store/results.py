"""Engine-1 result rows: observations, scene state, events, processing bookkeeping.
All functions assume the caller holds a transaction."""
from __future__ import annotations

import json
import sqlite3

from ..grid import BBox


def clear_frame_results(conn: sqlite3.Connection, frame_id: int) -> None:
    """Remove a frame's previous results so re-extraction replaces them. new_object_type events are kept
    (the registry rows they describe are kept too)."""
    conn.execute("DELETE FROM observations WHERE frame_id=?", (frame_id,))
    conn.execute("DELETE FROM detections WHERE frame_id=?", (frame_id,))
    conn.execute("DELETE FROM relations WHERE frame_id=?", (frame_id,))
    conn.execute("DELETE FROM scene_state WHERE frame_id=?", (frame_id,))
    conn.execute("DELETE FROM events WHERE frame_id=? AND event_type!='new_object_type'", (frame_id,))


def insert_observation(conn: sqlite3.Connection, *, frame_id: int, space_id: str, ts_utc: str, object_type: str,
                       source: str, raw_label: str | None = None, count: int = 1, cell: tuple[int, int, str] | None = None,
                       bbox: BBox | None = None, confidence: float | None = None, status: str | None = None,
                       attrs: dict | None = None) -> None:
    col, row, cid = cell if cell else (None, None, None)
    conn.execute(
        """INSERT INTO observations(frame_id,space_id,ts_utc,object_type,raw_label,cell_col,cell_row,cell_id,
             bbox_norm,status,count,confidence,source,attrs_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (frame_id, space_id, ts_utc, object_type, raw_label or object_type, col, row, cid,
         json.dumps(list(bbox)) if bbox else None, status, count, confidence, source,
         json.dumps(attrs) if attrs else None))


def insert_detection(conn: sqlite3.Connection, *, frame_id: int, space_id: str, ts_utc: str, raw_label: str,
                     confidence: float, bbox: BBox, cell_id: str | None, counted: bool,
                     mask: bytes | None = None, mask_area: float | None = None) -> None:
    """One raw detector box (counted or not), with its outline if a segmenter ran. Counted boxes are also
    stored as observations."""
    conn.execute("INSERT INTO detections(frame_id,space_id,ts_utc,raw_label,confidence,bbox_norm,cell_id,counted,"
                 "mask,mask_area) VALUES(?,?,?,?,?,?,?,?,?,?)",
                 (frame_id, space_id, ts_utc, raw_label, confidence, json.dumps(list(bbox)), cell_id, int(counted),
                  mask, mask_area))


def insert_relation(conn: sqlite3.Connection, *, frame_id: int, space_id: str, ts_utc: str, subject: str,
                    predicate: str, obj: str, count: int, source: str, raw_subject: str | None = None,
                    raw_object: str | None = None) -> None:
    conn.execute(
        """INSERT INTO relations(frame_id,space_id,ts_utc,subject,predicate,object,raw_subject,raw_object,count,source)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (frame_id, space_id, ts_utc, subject, predicate, obj, raw_subject or subject, raw_object or obj, count, source))


def insert_scene_state(conn: sqlite3.Connection, *, frame_id: int, space_id: str, ts_utc: str,
                       people_yolo: int, people_vlm: int | None, vehicles_yolo: int, columns: dict,
                       summary: str | None, extra: dict | None) -> None:
    disagree = None if people_vlm is None else int(abs(people_yolo - people_vlm) > 1)
    conn.execute(
        """INSERT INTO scene_state(frame_id,space_id,ts_utc,people_yolo,people_vlm,count_disagree,vehicles_yolo,
             lighting,door_state,tidiness,weather_visible,summary,attrs_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (frame_id, space_id, ts_utc, people_yolo, people_vlm, disagree, vehicles_yolo, columns.get("lighting"),
         columns.get("door_state"), columns.get("tidiness"), columns.get("weather_visible"), summary,
         json.dumps(extra) if extra else None))


def previous_people(conn: sqlite3.Connection, space_id: str, ts_utc: str) -> int | None:
    """People count of the previous processed frame (VLM if present, else detector), or None."""
    row = conn.execute("SELECT people_yolo,people_vlm FROM scene_state WHERE space_id=? AND ts_utc<? "
                       "ORDER BY ts_utc DESC LIMIT 1", (space_id, ts_utc)).fetchone()
    if row is None:
        return None
    return row["people_vlm"] if row["people_vlm"] is not None else row["people_yolo"]


def insert_event(conn: sqlite3.Connection, *, space_id: str, ts_utc: str, frame_id: int, event_type: str,
                 details: dict | None = None) -> None:
    conn.execute("INSERT INTO events(space_id,ts_utc,frame_id,event_type,details_json) VALUES(?,?,?,?,?)",
                 (space_id, ts_utc, frame_id, event_type, json.dumps(details) if details else None))


def mark_processed(conn: sqlite3.Connection, frame_id: int, *, detector_version: str, detector_min_conf: float | None,
                   vlm_model: str | None, prompt_version: str | None, detector_raw: str, vlm_raw: str | None,
                   processed_utc: str, segmenter_version: str | None = None) -> None:
    conn.execute("""UPDATE frames SET detector_version=?, detector_min_conf=?, segmenter_version=?, vlm_model=?,
                      prompt_version=?, detector_raw=?, vlm_raw=?, processed_utc=? WHERE frame_id=?""",
                 (detector_version, detector_min_conf, segmenter_version, vlm_model, prompt_version, detector_raw,
                  vlm_raw, processed_utc, frame_id))

"""Engine 1: turn a stored frame into observations, scene state and events.

Design:
  * Slow work (detector, VLM) runs BEFORE any write transaction, so the DB lock is held only briefly.
  * All rows for a frame are written in ONE transaction: a failure leaves the frame exactly as it was.
  * Re-running a frame replaces its previous results (re-extract old images with a new prompt/model).
  * VLMError is handled (frame kept with detector results, VLM fields NULL: missing is not zero).
    DetectorError and everything else propagate: the frame stays unprocessed and is retried from the backlog.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

from .errors import StorageError, VLMError
from .grid import bbox_to_cell
from .perception.geometry import overlap_links
from .perception.normalise import canonical
from .perception.types import Detection, Detector, SceneReader, SceneReading
from .spaces import SpaceConfig
from .store import results
from .store.db import transaction
from .store.frames import get_frame
from .store.registry import category_of
from .timeutil import now_utc

log = logging.getLogger(__name__)

SCENE_COLUMNS = {"lighting", "door_state", "tidiness", "weather_visible"}


@dataclass(frozen=True)
class FrameResult:
    frame_id: int
    processed: bool
    reason: str | None = None          # why it was skipped
    people_yolo: int | None = None
    people_vlm: int | None = None
    boxes: int = 0
    vlm_ok: bool = False


def process_frame(conn, space: SpaceConfig, frame_id: int, detector: Detector, reader: SceneReader | None, *,
                  root: Path) -> FrameResult:
    frame = get_frame(conn, frame_id)
    if frame is None:
        raise StorageError(f"frame {frame_id} does not exist")
    if frame["capture_status"] != "ok" or not frame["image_path"]:
        return FrameResult(frame_id, False, reason=f"capture_status={frame['capture_status']}")
    path = Path(frame["image_path"])
    img = cv2.imread(str(path if path.is_absolute() else root / path))
    if img is None:
        return FrameResult(frame_id, False, reason="image unreadable")

    detections = detector.detect(img, space)          # DetectorError propagates; nothing written yet

    reading: SceneReading | None = None
    vlm_error: str | None = None
    if reader is not None:
        try:
            reading = reader.read(img, space)
        except VLMError as e:
            vlm_error = str(e)
            log.warning("%s frame %d: VLM unavailable, keeping detector results only: %s",
                        space.space_id, frame_id, vlm_error)

    with transaction(conn):
        return _write(conn, space, frame, detections, reading, vlm_error, detector.version)


def _write(conn, space: SpaceConfig, frame, detections: list[Detection], reading: SceneReading | None,
           vlm_error: str | None, detector_version: str) -> FrameResult:
    fid, sid, ts = frame["frame_id"], frame["space_id"], frame["ts_utc"]
    results.clear_frame_results(conn, fid)

    # detector rows: counts, boxes, grid cell (position is geometric; the VLM never sees the grid)
    for d in detections:
        label, _ = canonical(conn, d.label, space_id=sid, frame_id=fid, ts_utc=ts)
        cell = bbox_to_cell(d.bbox, space.grid.cols, space.grid.rows, space.grid.anchor)
        results.insert_observation(conn, frame_id=fid, space_id=sid, ts_utc=ts, object_type=label, source="yolo",
                                   raw_label=d.label, cell=cell, bbox=d.bbox, confidence=d.conf)
    for subj, pred, obj, n in overlap_links(detections):        # geometric links: deterministic, from boxes
        cs, _ = canonical(conn, subj, space_id=sid, frame_id=fid, ts_utc=ts)
        co, _ = canonical(conn, obj, space_id=sid, frame_id=fid, ts_utc=ts)
        results.insert_relation(conn, frame_id=fid, space_id=sid, ts_utc=ts, subject=cs, predicate=pred, obj=co,
                                count=n, source="geometry", raw_subject=subj, raw_object=obj)
    people_yolo = sum(1 for d in detections if d.label == "person")
    vehicles_yolo = sum(1 for d in detections if category_of(conn, d.label) == "vehicle")

    people_vlm: int | None = None
    columns: dict = {}
    extra: dict = {}
    if reading is not None:
        people_vlm = reading.people_count
        postures = Counter(p.get("posture") for p in reading.people if p.get("posture"))
        person, _ = canonical(conn, "person", space_id=sid, frame_id=fid, ts_utc=ts)
        results.insert_observation(conn, frame_id=fid, space_id=sid, ts_utc=ts, object_type=person, source="vlm",
                                   raw_label="person", count=people_vlm, status=postures.most_common(1)[0][0] if postures else None,
                                   attrs={"people": list(reading.people)})
        for o in reading.objects:
            if o.label.strip().lower() in ("person", "people"):
                continue                              # already stored from the structured people list
            label, _ = canonical(conn, o.label, space_id=sid, frame_id=fid, ts_utc=ts)
            results.insert_observation(conn, frame_id=fid, space_id=sid, ts_utc=ts, object_type=label,
                                       source="vlm", raw_label=o.label, count=o.count, status=o.status)
        for r in reading.relations:                             # relationships read from the photo
            cs, _ = canonical(conn, r.subject, space_id=sid, frame_id=fid, ts_utc=ts)
            co, _ = canonical(conn, r.object, space_id=sid, frame_id=fid, ts_utc=ts)
            results.insert_relation(conn, frame_id=fid, space_id=sid, ts_utc=ts, subject=cs, predicate=r.predicate,
                                    obj=co, count=r.count, source="vlm", raw_subject=r.subject, raw_object=r.object)
        columns = {k: v for k, v in reading.scene.items() if k in SCENE_COLUMNS}
        extra = {k: v for k, v in reading.scene.items() if k not in SCENE_COLUMNS}
    results.insert_scene_state(conn, frame_id=fid, space_id=sid, ts_utc=ts, people_yolo=people_yolo,
                               people_vlm=people_vlm, vehicles_yolo=vehicles_yolo, columns=columns,
                               summary=reading.summary if reading else None, extra=extra or None)

    # occupancy transitions vs the previous processed frame (stills: inferred between frames, not observed)
    prev = results.previous_people(conn, sid, ts)
    now_n = people_vlm if people_vlm is not None else people_yolo
    if prev is not None:
        if prev > 0 and now_n == 0:
            results.insert_event(conn, space_id=sid, ts_utc=ts, frame_id=fid, event_type="became_empty",
                                 details={"prev": prev, "now": now_n})
        elif prev == 0 and now_n > 0:
            results.insert_event(conn, space_id=sid, ts_utc=ts, frame_id=fid, event_type="became_occupied",
                                 details={"prev": prev, "now": now_n})

    results.mark_processed(
        conn, fid, detector_version=detector_version, vlm_model=reading.model if reading else None,
        prompt_version=reading.prompt_version if reading else None,
        detector_raw=json.dumps([{"label": d.label, "conf": d.conf, "bbox": list(d.bbox)} for d in detections]),
        vlm_raw=reading.raw if reading else (json.dumps({"error": vlm_error}) if vlm_error else None),
        processed_utc=now_utc().isoformat(timespec="seconds"))
    return FrameResult(fid, True, people_yolo=people_yolo, people_vlm=people_vlm, boxes=len(detections),
                       vlm_ok=reading is not None)

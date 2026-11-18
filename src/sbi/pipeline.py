"""Engine 1: turn a stored frame into observations, scene state and events.

Design:
  * Slow work (detector, VLM) runs BEFORE any write transaction, so the DB lock is held only briefly.
  * All rows for a frame are written in ONE transaction: a failure leaves the frame exactly as it was.
  * Re-running a frame replaces its previous results (re-extract old images with a new prompt/model).
  * Every detector box is stored as a raw detection; only boxes marked `counted` become observations, links
    and people/vehicle totals. Thresholds can therefore change later without re-running the model.
  * An optional segmenter outlines every stored box. SegmenterError is handled like VLMError: the frame is
    kept and its masks are NULL (unknown, not zero).
  * An optional embedder stores a whole-scene vector per frame. EmbedderError is handled like SegmenterError:
    the frame is kept without a vector (unknown, not zero), and a vector stored earlier is never wiped by a
    failed re-run. `sbi embed` fills in the missing ones later.
  * An optional scorer rates the photo against the space's plain-English probe prompts (`probes` in the space
    file). ScorerError is handled the same way: the frame is kept without scores, and scores stored earlier are
    never wiped by a failed re-run. `sbi score` fills in missing (frame, prompt) pairs later.
  * replay_vlm=True re-runs the detector only: the VLM part is rebuilt from the stored raw answer, unchanged.
  * VLMError is handled (frame kept with detector results, VLM fields NULL: missing is not zero).
    DetectorError and everything else propagate: the frame stays unprocessed and is retried from the backlog.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from .errors import EmbedderError, ScorerError, SegmenterError, StorageError, VLMError
from .imageio import read_frame_image
from .grid import bbox_to_cell
from .perception.dedupe import uncount_outline_duplicates
from .perception.geometry import overlap_links
from .perception.normalise import canonical
from .perception.masks import encode_mask
from .perception.types import Detection, Detector, Embedder, SceneReader, SceneReading, Scorer, SegMask, Segmenter
from .spaces import SpaceConfig
from .store import results
from .store.embeddings import put_embedding
from .store.scores import put_scores
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
    boxes: int = 0                     # counted boxes
    boxes_stored: int = 0              # all raw detections, counted or not
    vlm_ok: bool = False
    masks_ok: bool = False             # a segmenter ran for this frame
    embedded: bool = False             # a whole-scene vector was stored for this frame
    scored: bool = False               # probe scores were stored for this frame


def _stored_reading(frame, space: SpaceConfig, reader: SceneReader | None) -> tuple[SceneReading | None, str | None]:
    """The frame's VLM result as it was stored: (reading, None), (None, error text) or (None, None)."""
    raw = frame["vlm_raw"]
    if raw is None:
        return None, None
    if raw.startswith('{"error"'):
        return None, json.loads(raw).get("error")
    if reader is None:
        raise StorageError("replaying a stored VLM answer needs a scene reader to parse it")
    try:
        reading = reader.parse(raw, space)
    except (ValueError, KeyError, TypeError) as e:
        raise StorageError(f"frame {frame['frame_id']}: stored VLM answer cannot be replayed: {e}") from e
    return replace(reading, model=frame["vlm_model"], prompt_version=frame["prompt_version"]), None


def process_frame(conn, space: SpaceConfig, frame_id: int, detector: Detector, reader: SceneReader | None, *,
                  root: Path, replay_vlm: bool = False, segmenter: Segmenter | None = None,
                  embedder: Embedder | None = None, scorer: Scorer | None = None,
                  dedupe_outline_iou: float = 0.0) -> FrameResult:
    """`dedupe_outline_iou` > 0 (with a segmenter): counted boxes of one type whose outlines overlap that much are
    counted once (detector.dedupe_outline_iou)."""
    frame = get_frame(conn, frame_id)
    if frame is None:
        raise StorageError(f"frame {frame_id} does not exist")
    if frame["capture_status"] != "ok" or not frame["image_path"]:
        return FrameResult(frame_id, False, reason=f"capture_status={frame['capture_status']}")
    img = read_frame_image(frame, root)
    if img is None:
        return FrameResult(frame_id, False, reason="image unreadable")

    detections = detector.detect(img, space)          # DetectorError propagates; nothing written yet

    masks: list[SegMask] | None = None
    if segmenter is not None:
        try:
            masks = segmenter.segment(img, [d.bbox for d in detections])
            if len(masks) != len(detections):
                raise SegmenterError(f"{len(masks)} masks for {len(detections)} boxes")
        except SegmenterError as e:
            masks = None
            log.warning("%s frame %d: segmenter unavailable, storing boxes without outlines: %s",
                        space.space_id, frame_id, e)
    seg_version = segmenter.version if segmenter is not None and masks is not None else None
    if masks is not None and dedupe_outline_iou > 0:
        detections = uncount_outline_duplicates(detections, [m.grid for m in masks], dedupe_outline_iou)

    vector = None
    if embedder is not None:
        try:
            vector = embedder.embed(img)
            if vector.shape != (embedder.dim,):
                raise EmbedderError(f"vector of shape {vector.shape}, expected ({embedder.dim},)")
        except EmbedderError as e:
            vector = None
            log.warning("%s frame %d: embedder unavailable, storing the frame without a vector: %s",
                        space.space_id, frame_id, e)

    scores: dict[str, float] | None = None
    prompts = space.probe_prompts
    if scorer is not None and prompts:
        try:
            values = scorer.score(img, prompts)
            if len(values) != len(prompts):
                raise ScorerError(f"{len(values)} scores for {len(prompts)} prompts")
            scores = dict(zip(prompts, values))
        except ScorerError as e:
            log.warning("%s frame %d: scorer unavailable, storing the frame without probe scores: %s",
                        space.space_id, frame_id, e)

    reading: SceneReading | None = None
    vlm_error: str | None = None
    if replay_vlm:
        reading, vlm_error = _stored_reading(frame, space, reader)     # StorageError propagates; nothing written
    elif reader is not None:
        try:
            reading = reader.read(img, space)
        except VLMError as e:
            vlm_error = str(e)
            log.warning("%s frame %d: VLM unavailable, keeping detector results only: %s",
                        space.space_id, frame_id, vlm_error)

    with transaction(conn):
        res = _write(conn, space, frame, detections, reading, vlm_error, detector, masks, seg_version)
        if vector is not None and embedder is not None:
            put_embedding(conn, frame_id, embedder.version, vector)
        if scores is not None and scorer is not None:
            put_scores(conn, frame_id, scorer.version, scores)
    return replace(res, embedded=vector is not None, scored=scores is not None)


def _write(conn, space: SpaceConfig, frame, all_detections: list[Detection], reading: SceneReading | None,
           vlm_error: str | None, detector: Detector, masks: list[SegMask] | None = None,
           seg_version: str | None = None) -> FrameResult:
    fid, sid, ts = frame["frame_id"], frame["space_id"], frame["ts_utc"]
    results.clear_frame_results(conn, fid)

    # raw layer: every box the detector returned, with its confidence (position is geometric; the VLM never
    # sees the grid). Counted boxes also become observations below.
    for i, d in enumerate(all_detections):
        cell = bbox_to_cell(d.bbox, space.grid.cols, space.grid.rows, space.grid.anchor)
        m = masks[i] if masks is not None else None
        results.insert_detection(conn, frame_id=fid, space_id=sid, ts_utc=ts, raw_label=d.label, confidence=d.conf,
                                 bbox=d.bbox, cell_id=cell[2], counted=d.counted,
                                 mask=encode_mask(m.grid) if m else None, mask_area=m.area if m else None)
    detections = [d for d in all_detections if d.counted]
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
        conn, fid, detector_version=detector.version, detector_min_conf=detector.min_conf,
        vlm_model=reading.model if reading else None,
        prompt_version=reading.prompt_version if reading else None,
        detector_raw=json.dumps([{"label": d.label, "conf": d.conf, "bbox": list(d.bbox), "counted": d.counted}
                                 for d in all_detections]),
        vlm_raw=reading.raw if reading else (json.dumps({"error": vlm_error}) if vlm_error else None),
        processed_utc=now_utc().isoformat(timespec="seconds"), segmenter_version=seg_version)
    return FrameResult(fid, True, people_yolo=people_yolo, people_vlm=people_vlm, boxes=len(detections),
                       boxes_stored=len(all_detections),
                       vlm_ok=reading is not None, masks_ok=seg_version is not None)

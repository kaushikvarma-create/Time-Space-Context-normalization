"""One scheduled tick: for every space capture a frame, run Engine 1, refresh that hour's baseline.

Failure isolation: a problem in one space never stops the others, and a perception outage never stops
capture. Expected failures (SbiError) are logged as errors; anything else is a bug: logged with a traceback
at this top-level boundary (and still isolated) so one bad space cannot silently kill the run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import baseline
from .errors import ConfigError, DetectorError, SbiError
from .ingest.capture import capture_frame
from .perception.factory import build_detector, build_frame_source, build_scene_reader
from .perception.types import Detector, FrameSource, SceneReader
from .pipeline import process_frame
from .settings import Settings, load_nvr_credentials
from .spaces import SpaceConfig, load_space
from .store.db import connect, migrate
from .store.frames import pending_frame_ids
from .store.registry import register_space
from .timeutil import now_utc, time_dims

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpaceOutcome:
    space_id: str
    ok: bool
    frame_id: int | None = None
    capture_status: str | None = None
    processed: int = 0
    error: str | None = None


def run_tick(settings: Settings, space_files: list[Path], *, source: FrameSource | None = None,
             detector: Detector | None = None, reader: SceneReader | None = None, use_vlm: bool = True,
             now: datetime | None = None) -> list[SpaceOutcome]:
    """Components can be injected (tests, alternative implementations); otherwise built from settings."""
    now = now or now_utc()
    conn = connect(settings.db_path)
    try:
        migrate(conn)
        outcomes: list[SpaceOutcome] = []
        spaces: list[SpaceConfig] = []
        for f in space_files:
            try:
                s = load_space(f)
                register_space(conn, s, f.relative_to(settings.root) if f.is_relative_to(settings.root) else f)
                spaces.append(s)
            except SbiError as e:
                log.error("skipping space file %s: %s", f.name, e)
                outcomes.append(SpaceOutcome(f.stem, False, error=str(e)))

        if source is None:
            source = build_frame_source(settings.frame_source, load_nvr_credentials())   # ConfigError: fatal
        if detector is None:
            try:
                detector = build_detector(settings.detector, settings.root)
            except DetectorError as e:       # keep capturing; frames wait in the backlog until it recovers
                log.error("detector unavailable, frames will be queued unprocessed: %s", e)
        if reader is None and use_vlm:
            reader = build_scene_reader(settings.scene_reader)

        for space in spaces:
            outcomes.append(_run_space(conn, settings, space, source, detector, reader, now))
        return outcomes
    finally:
        conn.close()


def _run_space(conn, settings: Settings, space: SpaceConfig, source: FrameSource, detector: Detector | None,
               reader: SceneReader | None, now: datetime) -> SpaceOutcome:
    try:
        cap = capture_frame(conn, space, source, root=settings.root, image_dir=settings.image_dir, now=now)
        processed = 0
        if detector is not None:
            todo = [cap.frame_id] if cap.status == "ok" else []
            todo += [i for i in pending_frame_ids(conn, space.space_id, settings.processing.backlog_limit)
                     if i != cap.frame_id]
            for fid in todo:
                try:
                    res = process_frame(conn, space, fid, detector, reader, root=settings.root)
                except DetectorError as e:
                    log.error("%s frame %d: detector failed, leaving it queued: %s", space.space_id, fid, e)
                    break                    # do not hammer a broken detector with the rest of the backlog
                except SbiError as e:
                    log.error("%s frame %d: processing failed, leaving it queued: %s", space.space_id, fid, e)
                    continue
                if res.processed:
                    processed += 1
                    log.info("%s frame %d: people yolo=%s vlm=%s boxes=%d%s", space.space_id, fid, res.people_yolo,
                             res.people_vlm, res.boxes, "" if res.vlm_ok else " (no VLM)")
        if processed:
            d = time_dims(now, space.tz)
            baseline.rebuild(conn, space.space_id, alpha=settings.baseline.ewm_alpha, dow=d.dow, hour=d.hour)
        return SpaceOutcome(space.space_id, True, cap.frame_id, cap.status, processed)
    except SbiError as e:
        log.error("%s: tick failed: %s", space.space_id, e)
        return SpaceOutcome(space.space_id, False, error=str(e))
    except Exception as e:  # noqa: BLE001  top-level isolation boundary: log the traceback, keep other spaces going
        log.exception("%s: unexpected error (this is a bug)", space.space_id)
        return SpaceOutcome(space.space_id, False, error=f"unexpected: {e!r}")


__all__ = ["SpaceOutcome", "run_tick", "ConfigError"]

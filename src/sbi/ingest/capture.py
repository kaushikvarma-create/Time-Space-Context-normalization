"""Capture one frame for a space: grab, quality-check, store the image, record the frame row.

A camera failure is DATA, not an exception: it is recorded as capture_status='camera_down' (NULL metrics),
so averages exclude it and nothing is ever read as 'empty room'. A disk failure is an error (StorageError).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..errors import CameraError, StorageError
from ..perception.types import FrameSource
from ..spaces import SpaceConfig
from ..store.db import transaction
from ..store.frames import ImageMetrics, insert_frame
from ..timeutil import now_utc, time_dims

log = logging.getLogger(__name__)


def compute_metrics(img: np.ndarray) -> ImageMetrics:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ImageMetrics(width=int(img.shape[1]), height=int(img.shape[0]), brightness=float(gray.mean()),
                        contrast=float(gray.std()), sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()))


def is_bad_frame(m: ImageMetrics) -> bool:
    """Flat near-black or near-white frames usually mean a bad decode, not an empty room."""
    return m.contrast < 1.0 and (m.brightness < 5 or m.brightness > 250)


@dataclass(frozen=True)
class CaptureResult:
    frame_id: int
    status: str            # ok | camera_down | bad_frame
    created: bool          # False if this (space, timestamp) already existed
    error: str | None = None


def _image_name(ts_local: str) -> str:
    return ts_local.replace("-", "").replace(":", "") + ".jpg"      # e.g. 20261113T091143+0530.jpg (IST)


def capture_frame(conn, space: SpaceConfig, source: FrameSource, *, root: Path, image_dir: Path,
                  now: datetime | None = None) -> CaptureResult:
    dims = time_dims(now or now_utc(), space.tz)
    img, error = None, None
    try:
        img = source.grab(space)
    except CameraError as e:
        error = str(e)
        log.warning("%s: camera unavailable: %s", space.space_id, error)

    status, rel_path, metrics = "camera_down", None, None
    if img is not None:
        metrics = compute_metrics(img)
        if is_bad_frame(metrics):
            status, error = "bad_frame", "flat near-black/near-white frame"
            log.warning("%s: bad frame (brightness %.1f, contrast %.2f)", space.space_id, metrics.brightness,
                        metrics.contrast)
        else:
            status = "ok"
            path = image_dir / space.space_id / dims.date / _image_name(dims.ts_local)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError("cv2.imwrite returned False")
            except OSError as e:
                raise StorageError(f"cannot write image {path}: {e}") from e
            rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

    with transaction(conn):
        frame_id, created = insert_frame(conn, space.space_id, dims, status=status, image_path=rel_path,
                                         metrics=metrics)
    log.info("%s: %s frame_id=%d%s", space.space_id, status, frame_id, "" if created else " (already existed)")
    return CaptureResult(frame_id, status, created, error)

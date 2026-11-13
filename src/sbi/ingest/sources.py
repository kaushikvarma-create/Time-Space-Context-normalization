"""Camera sources. Credentials never appear in logs or exception messages."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from ..errors import CameraError
from ..settings import FrameSourceSettings, NvrCredentials
from ..spaces import CameraRef, SpaceConfig

log = logging.getLogger(__name__)


def rtsp_url(creds: NvrCredentials, camera: CameraRef, *, redact: bool = False) -> str:
    """Dahua-style NVR URL. With redact=True the user/password are masked (safe for logs)."""
    auth = "***:***" if redact else f"{quote(creds.user, safe='')}:{quote(creds.password.get_secret_value(), safe='')}"
    return f"rtsp://{auth}@{creds.host}:{creds.port}/cam/realmonitor?channel={camera.channel}&subtype={camera.subtype}"


class NvrRtspSource:
    def __init__(self, settings: FrameSourceSettings, creds: NvrCredentials):
        self.cfg, self._creds = settings, creds

    def grab(self, space: SpaceConfig) -> np.ndarray:
        # Must be set before the capture is opened (read by OpenCV's FFmpeg backend).
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.cfg.rtsp_transport}"
        where = rtsp_url(self._creds, space.camera, redact=True)
        cap = cv2.VideoCapture(rtsp_url(self._creds, space.camera), cv2.CAP_FFMPEG)
        timeout_ms = int(self.cfg.timeout_s * 1000)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        try:
            if not cap.isOpened():
                raise CameraError(f"cannot open stream {where}")
            frame = None
            for _ in range(self.cfg.warmup_frames):   # discard buffered frames; keep the freshest
                ok, f = cap.read()
                if ok:
                    frame = f
            if frame is None:
                raise CameraError(f"stream opened but no frame decoded: {where}")
            return frame
        finally:
            cap.release()


class FileSource:
    """Reads a still from disk. For tests and offline re-runs."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def grab(self, space: SpaceConfig) -> np.ndarray:
        img = cv2.imread(str(self.path))
        if img is None:
            raise CameraError(f"cannot read image file {self.path}")
        return img

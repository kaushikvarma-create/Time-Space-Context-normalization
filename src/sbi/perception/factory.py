"""Build components from settings. The `kind` registries are the single place to add a new implementation."""
from __future__ import annotations

from pathlib import Path

from ..errors import ConfigError
from ..settings import DetectorSettings, FrameSourceSettings, NvrCredentials, SceneReaderSettings
from .types import Detector, FrameSource, SceneReader


def build_detector(settings: DetectorSettings, root: Path) -> Detector:
    if settings.kind == "yolo":
        from .detector_yolo import YoloDetector  # lazy: keep torch out of imports that do not need it
        return YoloDetector(settings, root)
    raise ConfigError(f"unknown detector kind {settings.kind!r}")


def build_scene_reader(settings: SceneReaderSettings) -> SceneReader:
    if settings.kind == "lmstudio_openai":
        from .scene_lmstudio import LmStudioSceneReader
        return LmStudioSceneReader(settings)
    raise ConfigError(f"unknown scene reader kind {settings.kind!r}")


def build_frame_source(settings: FrameSourceSettings, creds: NvrCredentials) -> FrameSource:
    if settings.kind == "nvr_rtsp":
        from ..ingest.sources import NvrRtspSource
        return NvrRtspSource(settings, creds)
    raise ConfigError(f"unknown frame source kind {settings.kind!r}")

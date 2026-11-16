"""Build components from settings. The `kind` registries are the single place to add a new implementation."""
from __future__ import annotations

from pathlib import Path

from ..errors import ConfigError
from ..settings import (DetectorSettings, EmbedderSettings, FrameSourceSettings, NvrCredentials,
                        SceneReaderSettings, ScorerSettings, SegmenterSettings)
from .types import Detector, Embedder, FrameSource, SceneReader, Scorer, Segmenter


def build_detector(settings: DetectorSettings, root: Path) -> Detector:
    if settings.kind == "yolo":
        from .detector_yolo import YoloDetector  # lazy: keep torch out of imports that do not need it
        return YoloDetector(settings, root)
    raise ConfigError(f"unknown detector kind {settings.kind!r}")


def build_segmenter(settings: SegmenterSettings, root: Path) -> Segmenter | None:
    if settings.kind == "none":
        return None
    if settings.kind == "sam2":
        from .segmenter_sam2 import Sam2Segmenter
        return Sam2Segmenter(settings, root)
    raise ConfigError(f"unknown segmenter kind {settings.kind!r}")


def build_embedder(settings: EmbedderSettings, root: Path) -> Embedder | None:
    if settings.kind == "none":
        return None
    if settings.kind == "dinov2":
        from .embedder_dinov2 import DinoV2Embedder
        return DinoV2Embedder(settings, root)
    raise ConfigError(f"unknown embedder kind {settings.kind!r}")


def build_scorer(settings: ScorerSettings, root: Path) -> Scorer | None:
    if settings.kind == "none":
        return None
    if settings.kind == "siglip2":
        from .scorer_siglip2 import Siglip2Scorer
        return Siglip2Scorer(settings, root)
    raise ConfigError(f"unknown scorer kind {settings.kind!r}")


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

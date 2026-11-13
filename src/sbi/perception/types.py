"""Data types and the interfaces (Protocols) every perception component implements.

To add a new detector / scene reader / camera source: implement the Protocol, register it in
`factory.py`, and add its `kind` to the settings model. No pipeline code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..grid import BBox
from ..spaces import SpaceConfig


@dataclass(frozen=True)
class Detection:
    label: str            # already mapped to the space's canonical vocabulary
    conf: float
    bbox: BBox            # normalised [x1, y1, x2, y2]
    counted: bool = True  # False = below the counting threshold: stored as a raw detection, never counted


@dataclass(frozen=True)
class SceneObject:
    label: str            # raw label as read by the model (normalised later against the registry)
    count: int
    status: str


@dataclass(frozen=True)
class Relation:
    subject: str          # raw label as read by the model
    predicate: str        # from the fixed predicate list
    object: str
    count: int


@dataclass(frozen=True)
class SceneReading:
    people_count: int
    people: tuple[dict, ...]
    objects: tuple[SceneObject, ...]
    scene: dict[str, str | None]       # space-specific scene fields (lighting, tidiness, ...)
    summary: str
    raw: str                           # raw model text, always stored
    model: str
    prompt_version: str
    relations: tuple[Relation, ...] = ()
    latency_s: float = field(default=0.0, compare=False)


class Detector(Protocol):
    version: str
    min_conf: float       # lowest confidence it returns (the storage floor), recorded per frame

    def detect(self, image: np.ndarray, space: SpaceConfig) -> list[Detection]:
        """Every box down to min_conf; `counted` marks the ones above the counting threshold.
        Raise DetectorError on failure."""


class SceneReader(Protocol):
    def read(self, image: np.ndarray, space: SpaceConfig) -> SceneReading:
        """Raise VLMError on failure (after any retries)."""

    def parse(self, raw: str, space: SpaceConfig) -> SceneReading:
        """Rebuild a reading from a stored raw answer (no model call). Raise ValueError if unusable."""


class FrameSource(Protocol):
    def grab(self, space: SpaceConfig) -> np.ndarray:
        """Return a BGR image. Raise CameraError on failure."""

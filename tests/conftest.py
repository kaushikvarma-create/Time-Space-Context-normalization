"""Shared fixtures and fakes. No test touches the camera, the GPU, or LM Studio."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sbi.errors import CameraError, DetectorError, VLMError
from sbi.perception.types import Detection, SceneObject, SceneReading
from sbi.settings import SceneReaderSettings, Settings
from sbi.spaces import SpaceConfig, load_space
from sbi.store.db import connect, migrate
from sbi.store.registry import register_space

REPO = Path(__file__).resolve().parents[1]


def synthetic_image(seed: int = 0, w: int = 320, h: int = 180) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(40, 220, (h, w, 3), dtype=np.uint8)


def make_reading(people: int = 2, objects=(("chair", 3, "idle"),), scene=None, raw: str = "{}") -> SceneReading:
    return SceneReading(
        people_count=people, people=tuple({"posture": "seated", "activity": "typing"} for _ in range(people)),
        objects=tuple(SceneObject(*o) for o in objects),
        scene=scene if scene is not None else {"lighting": "artificial", "door_state": "unknown", "tidiness": "ok"},
        summary="test scene", raw=raw, model="fake-vlm", prompt_version="vtest")


class FakeSource:
    """Returns images in order; an Exception instance in the list is raised instead."""

    def __init__(self, items):
        self.items, self.calls = list(items), 0

    def grab(self, space):
        item = self.items[min(self.calls, len(self.items) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class FakeDetector:
    version = "fake-det@1"
    min_conf = 0.05

    def __init__(self, dets=(), error: Exception | None = None):
        self.dets, self.error, self.calls = list(dets), error, 0

    def detect(self, image, space):
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.dets)


class FakeReader:
    def __init__(self, reading: SceneReading | None = None, error: Exception | None = None):
        self.reading, self.error, self.calls = reading or make_reading(), error, 0

    def read(self, image, space):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reading

    def parse(self, raw, space):
        """Stored answers replay as the reading this fake was built with (raw text kept)."""
        from dataclasses import replace
        return replace(self.reading, raw=raw)


class FakeSegmenter:
    """Outlines each box as its own filled rectangle on a 64x36 grid (or returns the masks it was given)."""
    version = "fake-seg@64"

    def __init__(self, masks=None, error: Exception | None = None):
        self.masks, self.error, self.calls = masks, error, 0

    def segment(self, image, boxes):
        from sbi.perception.types import SegMask
        self.calls += 1
        if self.error:
            raise self.error
        if self.masks is not None:
            return list(self.masks)
        out = []
        for x1, y1, x2, y2 in boxes:
            g = np.zeros((36, 64), dtype=bool)
            g[round(y1 * 36):round(y2 * 36), round(x1 * 64):round(x2 * 64)] = True
            out.append(SegMask(round((x2 - x1) * (y2 - y1), 6), g))
        return out


class FakeEmbedder:
    """Returns the vectors it was given in order (or one derived from the image), or raises `error`."""
    version = "fake-emb@8"
    dim = 8

    def __init__(self, vectors=None, error: Exception | None = None):
        self.vectors, self.error, self.calls = list(vectors or []), error, 0

    def embed(self, image):
        self.calls += 1
        if self.error:
            raise self.error
        if self.vectors:
            return np.asarray(self.vectors.pop(0), dtype=np.float32)
        return (1.0 + float(image.mean()) / 255.0 * np.arange(self.dim)).astype(np.float32)


class FakeScorer:
    """Scores each prompt from its text (so tests can predict values), or raises `error`. Records what was asked."""
    version = "fake-scorer"

    def __init__(self, values: dict[str, float] | None = None, error: Exception | None = None):
        self.values, self.error, self.asked = values or {}, error, []

    def score(self, image, prompts):
        self.asked.append(list(prompts))
        if self.error:
            raise self.error
        return [self.values.get(p, float(len(p) % 7) - 3.0) for p in prompts]


def det(label: str, bbox, conf: float = 0.9, counted: bool = True) -> Detection:
    return Detection(label, conf, tuple(bbox), counted)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def settings(root: Path) -> Settings:
    return Settings(root=root, scene_reader=SceneReaderSettings(base_url="http://fake", model="fake-vlm"))


@pytest.fixture
def space() -> SpaceConfig:
    return load_space(REPO / "config" / "spaces" / "admin_cabin.yaml")


@pytest.fixture
def parking() -> SpaceConfig:
    return load_space(REPO / "config" / "spaces" / "parking.yaml")


@pytest.fixture
def conn(root: Path, space: SpaceConfig, parking: SpaceConfig):
    c = connect(root / "t.db")
    migrate(c)
    register_space(c, space, "config/spaces/admin_cabin.yaml")
    register_space(c, parking, "config/spaces/parking.yaml")
    yield c
    c.close()


@pytest.fixture
def space_dir(root: Path) -> Path:
    """tmp project root with real space configs copied in (for tick tests)."""
    d = root / "config" / "spaces"
    d.mkdir(parents=True)
    for n in ("admin_cabin", "parking"):
        shutil.copy(REPO / "config" / "spaces" / f"{n}.yaml", d)
    return d


def thursday_ist(hour: int = 10, minute: int = 0, week: int = 0) -> datetime:
    """A Thursday in IST (2026-11-12 + week*7 days) as an aware UTC datetime."""
    from datetime import timedelta
    ist = datetime(2026, 11, 12, hour, minute, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    return (ist + timedelta(days=7 * week)).astimezone(timezone.utc)


__all__ = ["CameraError", "DetectorError", "VLMError"]

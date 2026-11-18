"""Per-space configuration (one YAML per space), validated at load time.

Adding a new space = adding one YAML file under config/spaces/. Nothing in the code is room-specific.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ConfigError
from .timeutil import get_zone

_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Grid(_Strict):
    cols: int = Field(gt=0, le=64)
    rows: int = Field(gt=0, le=64)
    anchor: Literal["bottom_center", "center"] = "bottom_center"


class CameraRef(_Strict):
    channel: int = Field(ge=1)
    subtype: int = Field(0, ge=0)  # 0 = main stream, 1 = sub stream


class VocabEntry(_Strict):
    category: str
    mobility: Literal["fixed", "semi_static", "dynamic", "unknown"] = "unknown"


class SpaceConfig(_Strict):
    space_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    space_type: str
    camera: CameraRef
    tz: str = "Asia/Kolkata"
    cadence_minutes: int = Field(15, ge=1)
    label_types: list[str] = []
    mask_regions: list[tuple[float, float, float, float]] = []
    grid: Grid
    vocabulary: dict[str, VocabEntry]
    aliases: dict[str, str] = {}
    detector_map: dict[str, str] = {}
    scene_fields: dict[str, list[str]] = {}
    # Plain-English descriptions to score on every photo, grouped by concept. Several paraphrases per concept:
    # the table averages them, so one unlucky wording does not move the number.
    probes: dict[str, list[str]] = {}
    # What to forecast: table column names or shell-style patterns ("count.yolo.*", "probe.*.people"). Patterns
    # are matched against the table's model-ready columns when a forecast runs, so new object types are picked up.
    forecast_targets: list[str] = []

    @model_validator(mode="after")
    def _cross_checks(self) -> "SpaceConfig":
        get_zone(self.tz)
        problems: list[str] = []
        for label in self.vocabulary:
            if not _LABEL.match(label):
                problems.append(f"vocabulary label {label!r} must be lowercase snake_case")
        for alias, target in self.aliases.items():
            if target not in self.vocabulary:
                problems.append(f"alias {alias!r} -> {target!r}: target not in vocabulary")
        for coco, target in self.detector_map.items():
            if target not in self.vocabulary:
                problems.append(f"detector_map {coco!r} -> {target!r}: target not in vocabulary")
        for t in self.label_types:
            if t not in self.vocabulary:
                problems.append(f"label_types entry {t!r} not in vocabulary")
        for r in self.mask_regions:
            x1, y1, x2, y2 = r
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                problems.append(f"mask region {r} must be normalised [x1,y1,x2,y2] with x1<x2, y1<y2")
        if any(not t.strip() for t in self.forecast_targets) or len(set(self.forecast_targets)) != len(self.forecast_targets):
            problems.append("forecast_targets must be distinct, non-empty column names or patterns")
        seen: dict[str, str] = {}
        for concept, prompts in self.probes.items():
            if not _LABEL.match(concept):
                problems.append(f"probe concept {concept!r} must be lowercase snake_case")
            if not prompts:
                problems.append(f"probe concept {concept!r} has no prompts")
            for p in prompts:
                if not p.strip() or len(p) > 200:
                    problems.append(f"probe prompt {p!r} in {concept!r} must be 1-200 characters")
                if p in seen and seen[p] != concept:
                    problems.append(f"probe prompt {p!r} is used by both {seen[p]!r} and {concept!r}")
                seen[p] = concept
            if len(set(prompts)) != len(prompts):
                problems.append(f"probe concept {concept!r} repeats a prompt")
        for name, values in self.scene_fields.items():
            if not values:
                problems.append(f"scene field {name!r} has no allowed values")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def probe_prompts(self) -> list[str]:
        """Every distinct probe prompt of this space, in a stable order."""
        return list(dict.fromkeys(p for prompts in self.probes.values() for p in prompts))


def load_space(path: Path | str) -> SpaceConfig:
    path = Path(path)
    try:
        return SpaceConfig(**(yaml.safe_load(path.read_text()) or {}))
    except FileNotFoundError as e:
        raise ConfigError(f"space file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    except (ValidationError, TypeError) as e:
        raise ConfigError(f"invalid space config {path}:\n{e}") from e


def discover_space_files(root: Path) -> list[Path]:
    return sorted((root / "config" / "spaces").glob("*.yaml"))

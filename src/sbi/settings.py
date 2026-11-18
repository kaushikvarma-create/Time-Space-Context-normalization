"""Typed, validated global settings: config/pipeline.yaml + environment (.env). No magic defaults in code
paths: everything tunable is in the YAML, and unknown keys are rejected so typos fail loudly."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator

from .errors import ConfigError
from .timeutil import get_zone


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    db: str = "data/sbi.db"
    images: str = "data/images"
    logs: str = "data/logs"


class FrameSourceSettings(_Strict):
    kind: Literal["nvr_rtsp"] = "nvr_rtsp"
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    warmup_frames: int = Field(5, ge=1)
    timeout_s: float = Field(10, gt=0)


class DetectorSettings(_Strict):
    kind: Literal["yolo"] = "yolo"
    weights: str = "models/yolo26x.pt"
    imgsz: int = Field(1280, ge=320)
    conf: float = Field(0.25, gt=0, lt=1)            # counting threshold: boxes at or above it are counted
    store_conf: float = Field(0.05, gt=0, lt=1)      # storage floor: every box down to here is kept, uncounted
    class_conf: dict[str, float] = {}                # per-label counting threshold overrides (set from labels)
    iou: float = Field(0.5, gt=0, lt=1)
    dedupe_labels: list[str] = ["person"]
    dedupe_containment: float = Field(0.8, gt=0, le=1)
    dedupe_same_label_iou: float = Field(0.6, gt=0, le=1)

    @model_validator(mode="after")
    def _thresholds_above_floor(self) -> "DetectorSettings":
        for name, v in {"conf": self.conf, **{f"class_conf[{k}]": v for k, v in self.class_conf.items()}}.items():
            if not self.store_conf <= v < 1:
                raise ValueError(f"detector.{name}={v} must be >= store_conf ({self.store_conf}) and < 1")
        return self

    def count_threshold(self, label: str) -> float:
        return self.class_conf.get(label, self.conf)


class SegmenterSettings(_Strict):
    kind: Literal["none", "sam2"] = "none"           # none = no outlines; every mask-derived column stays empty
    weights: str = "models/sam2.1_l.pt"
    mask_width: int = Field(384, ge=32, le=2048)     # width of the stored low-resolution outline
    batch: int = Field(64, ge=1)                     # boxes per model call


class EmbedderSettings(_Strict):
    """Whole-scene image vectors. `model` and `width` are kept even when kind is none, so the table can still
    read the vectors stored earlier."""
    kind: Literal["none", "dinov2"] = "none"         # none = no new vectors are stored
    model: str = "facebook/dinov2-base"              # Hugging Face id or a local path
    width: int = Field(448, ge=56, le=1792)          # input width in pixels; height follows the photo's aspect

    @field_validator("width")
    @classmethod
    def _patch_multiple(cls, v: int) -> int:
        if v % 14:
            raise ValueError("width must be a multiple of 14 (the DINOv2 patch size)")
        return v

    @property
    def version(self) -> str:
        """Identifies the vector space: stored with every vector and used to pick vectors back out."""
        return f"{Path(self.model).name}@{self.width}"


class ScorerSettings(_Strict):
    """Scores every photo against the space's probe prompts (see SpaceConfig.probes). `model` is kept when kind
    is none so the table can still read scores stored earlier."""
    kind: Literal["none", "siglip2"] = "none"        # none = no new scores are stored
    model: str = "google/siglip2-base-patch16-224"   # Hugging Face id or a local path

    @property
    def version(self) -> str:
        return Path(self.model).name


class CalendarSettings(_Strict):
    """Known-in-advance day types (see sbi/sitecal.py)."""
    country: str | None = "IN"                       # public holidays via the optional `holidays` package
    subdivision: str | None = None                   # state code for regional holidays, if the package is used
    site_file: str = "config/site_calendar.yaml"     # the office's own holiday list and closures (authoritative)
    working_days: list[int] = [0, 1, 2, 3, 4]        # Monday = 0

    @field_validator("working_days")
    @classmethod
    def _days(cls, v: list[int]) -> list[int]:
        if not v or any(d not in range(7) for d in v) or len(set(v)) != len(v):
            raise ValueError("working_days must be distinct weekday numbers 0 (Monday) .. 6 (Sunday)")
        return v


ALL_KINDS = "*"            # in a variant: every model-ready column the pipeline extracted, no cap

DEFAULT_VARIANTS: dict[str, list[str]] = {
    "all": ["cal", ALL_KINDS],                         # calendar + every extracted dimension (live default)
    "univariate": [],                                  # the targets alone, no calendar
    "calendar": ["cal"],                               # + calendar (known in advance)
    "detector": ["cal", "scene", "count", "rollup", "area", "cell", "cellarea", "rel", "img", "cat"],
    "full": ["cal", "scene", "count", "rollup", "area", "cell", "cellarea", "rel", "img", "cat", "emb", "probe"],
}


class ForecastSettings(_Strict):
    """Forecasting stage. A variant is a named covariate set: the column kinds a forecaster may be given
    (`cal` = calendar, known in advance; the rest are past-only, chosen by the catalog's selection rule)."""
    model: str = "amazon/chronos-2"
    quantiles: list[float] = [0.1, 0.5, 0.9]
    horizon: int = Field(16, ge=1, le=1024)            # slots ahead (16 x 15 min = 4 h)
    min_context: int = Field(96, ge=8)                 # slots of history needed before forecasting (96 = 1 day)
    max_targets: int = Field(40, ge=1)                 # cap on forecast targets per space
    cross_learning: bool = False                       # let the model share information across targets (see docs)
    variant: str = "all"                               # covariate set for live forecasts
    every_minutes: int = Field(0, ge=0)                # the tick forecasts every space this often (0 = never;
                                                       # off unless config/pipeline.yaml switches it on)
    scheduled_model: str = "chronos2"                  # forecaster the tick uses
    variants: dict[str, list[str]] = DEFAULT_VARIANTS

    @field_validator("quantiles")
    @classmethod
    def _quantiles(cls, v: list[float]) -> list[float]:
        if not v or any(not 0 < q < 1 for q in v) or v != sorted(set(v)):
            raise ValueError("quantiles must be strictly increasing values between 0 and 1")
        if any(abs(q * 100 - round(q * 100)) > 1e-9 for q in v):
            raise ValueError("quantiles must be whole percents (0.1, 0.25, ...): they name the columns q10, q25, ...")
        if 0.5 not in v:
            raise ValueError("quantiles must include 0.5 (the point forecast)")
        return v

    @model_validator(mode="after")
    def _variant_known(self) -> "ForecastSettings":
        if self.every_minutes and self.every_minutes % 15:
            raise ValueError("forecast.every_minutes must be a multiple of the 15-minute tick (or 0 to switch off)")
        if self.variant not in self.variants:
            raise ValueError(f"forecast.variant {self.variant!r} is not one of {sorted(self.variants)}")
        return self


class SceneReaderSettings(_Strict):
    kind: Literal["lmstudio_openai"] = "lmstudio_openai"
    base_url: str
    model: str
    timeout_s: float = Field(120, gt=0)
    retries: int = Field(3, ge=1)
    max_tokens: int = Field(1200, ge=64)
    temperature: float = Field(0.1, ge=0, le=2)
    reasoning_effort: str | None = "none"
    image_width: int = Field(1344, ge=256)


class CurationSettings(_Strict):
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    judge_model: str = "qwen/qwen3.5-9b"
    min_similarity: float = Field(0.80, gt=0, lt=1)
    auto_approve: bool = True
    context: str = ("Object names produced by a vision model describing photos of an office cabin and a parking area. "
                    "In an office, 'notebook' means a paper notebook.")
    timeout_s: float = Field(60, gt=0)
    retries: int = Field(3, ge=1)


class TableSettings(_Strict):
    step_minutes: int = Field(15, ge=1)
    min_coverage: float = Field(0.5, ge=0, le=1)     # share of slots with a value
    min_presence: float = Field(0.05, ge=0, le=1)    # share of valid slots where a count is > 0
    min_valid_slots: int = Field(8, ge=1)
    min_present_slots: int = Field(3, ge=1)          # a count seen in fewer slots is a flicker, not a dimension
    exclude_labels: list[str] = ["wall", "floor", "ceiling", "pillar", "roof"]
    max_covariates: int = Field(24, ge=0)
    always_include: list[str] = []
    min_autocorr: float = Field(0.3, ge=-1, le=1)
    autocorr_min_slots: int = Field(96, ge=3)        # smoothness filter only once there is ~1 day of slots
    embed_components: int = Field(4, ge=0, le=64)    # principal components of the image vectors, as emb.*.pcN
    conf_levels: list[float] = [0.05, 0.10]          # extra detector thresholds to count at (count.yolo_cNN.*)
    mask_dup_iou: float = Field(0.5, gt=0, le=1)     # same-type outlines overlapping this much = one object
    cell_area_min: float = Field(0.02, ge=0, lt=1)   # smaller cell coverage is edge spill, recorded as 0

    @field_validator("conf_levels")
    @classmethod
    def _levels(cls, v: list[float]) -> list[float]:
        if any(not 0 < x < 1 for x in v) or len({round(x * 100) for x in v}) != len(v):
            raise ValueError("conf_levels must be distinct values in (0, 1), two decimals at most")
        return sorted(v)


class BaselineSettings(_Strict):
    ewm_alpha: float = Field(0.3, gt=0, le=1)


class ProcessingSettings(_Strict):
    backlog_limit: int = Field(5, ge=0)


class NvrCredentials(_Strict):
    host: str
    port: int = 554
    user: str
    password: SecretStr  # never printed: repr/str show '**********'


class Settings(_Strict):
    root: Path
    timezone: str = "Asia/Kolkata"
    paths: Paths = Paths()
    frame_source: FrameSourceSettings = FrameSourceSettings()
    detector: DetectorSettings = DetectorSettings()
    segmenter: SegmenterSettings = SegmenterSettings()
    embedder: EmbedderSettings = EmbedderSettings()
    scorer: ScorerSettings = ScorerSettings()
    forecast: ForecastSettings = ForecastSettings()
    calendar: CalendarSettings = CalendarSettings()
    scene_reader: SceneReaderSettings
    baseline: BaselineSettings = BaselineSettings()
    processing: ProcessingSettings = ProcessingSettings()
    curation: CurationSettings = CurationSettings()
    table: TableSettings = TableSettings()

    def resolve(self, relative: str | Path) -> Path:
        """Stored paths are relative to the project root; resolve them for IO."""
        p = Path(relative)
        return p if p.is_absolute() else self.root / p

    @property
    def db_path(self) -> Path:
        return self.resolve(self.paths.db)

    @property
    def image_dir(self) -> Path:
        return self.resolve(self.paths.images)

    @property
    def log_dir(self) -> Path:
        return self.resolve(self.paths.logs)


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader. Real environment variables win over the file."""
    if not path.exists():
        return
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"{path}:{n}: expected KEY=VALUE")
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def load_settings(root: Path | str | None = None, config_file: str = "config/pipeline.yaml") -> Settings:
    root = Path(root or Path.cwd()).resolve()
    load_dotenv(root / ".env")
    cfg_path = root / config_file
    try:
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    except FileNotFoundError as e:
        raise ConfigError(f"settings file not found: {cfg_path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {cfg_path}: {e}") from e
    reader = dict(raw.get("scene_reader") or {})
    if os.environ.get("SBI_LMSTUDIO_URL"):
        reader["base_url"] = os.environ["SBI_LMSTUDIO_URL"]
    if os.environ.get("SBI_VLM_MODEL"):
        reader["model"] = os.environ["SBI_VLM_MODEL"]
    raw["scene_reader"] = reader
    try:
        settings = Settings(root=root, **raw)
    except ValidationError as e:
        raise ConfigError(f"invalid settings in {cfg_path}:\n{e}") from e
    get_zone(settings.timezone)  # fail fast on a bad timezone name
    return settings


def load_nvr_credentials() -> NvrCredentials:
    """Read camera credentials from the environment; raise ConfigError naming what is missing."""
    missing = [k for k in ("SBI_NVR_HOST", "SBI_NVR_USER", "SBI_NVR_PASS") if not os.environ.get(k)]
    if missing:
        raise ConfigError(f"missing environment variables: {', '.join(missing)} (see .env.example)")
    try:
        return NvrCredentials(host=os.environ["SBI_NVR_HOST"], port=int(os.environ.get("SBI_NVR_PORT", "554")),
                              user=os.environ["SBI_NVR_USER"], password=os.environ["SBI_NVR_PASS"])
    except (ValueError, ValidationError) as e:
        raise ConfigError(f"invalid NVR credentials/port in environment: {e}") from e

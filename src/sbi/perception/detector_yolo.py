"""YOLO (Ultralytics) detector with a per-space class whitelist.

The model runs at the storage floor (`store_conf`) and every box is returned with its confidence. Boxes at or
above the counting threshold (`conf`, or a per-label `class_conf`) are marked `counted`; the rest are kept as
raw detections only. Both dedupe steps keep the stronger box, so the counted set is exactly what running the
model at the counting threshold would give."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..errors import DetectorError
from ..settings import DetectorSettings
from ..spaces import SpaceConfig
from .dedupe import dedupe_contained, dedupe_same_label_iou
from .types import Detection

log = logging.getLogger(__name__)

RawBox = tuple[list[float], float, str]      # (normalised xyxy, confidence, model class name)


def build_detections(raw: list[RawBox], space: SpaceConfig, cfg: DetectorSettings) -> list[Detection]:
    """Pure post-processing (no model): class whitelist, counting threshold, dedupe."""
    dets = []
    for xyxyn, conf, cls_name in raw:
        label = space.detector_map.get(cls_name)
        if label is None:   # class not whitelisted for this space: noise (e.g. refrigerator in a car park)
            continue
        conf = round(float(conf), 3)        # decide on the stored value, so `counted` always agrees with it
        dets.append(Detection(label, conf, tuple(round(v, 4) for v in xyxyn), counted=conf >= cfg.count_threshold(label)))
    dets = dedupe_same_label_iou(dets, cfg.dedupe_same_label_iou)
    return dedupe_contained(dets, set(cfg.dedupe_labels), cfg.dedupe_containment)


class YoloDetector:
    def __init__(self, settings: DetectorSettings, root: Path):
        self.cfg = settings
        weights = Path(settings.weights)
        weights = weights if weights.is_absolute() else root / weights
        if not weights.exists():
            raise DetectorError(f"detector weights not found: {weights}")
        try:
            import warnings
            # Environment quirk (two matplotlib installs): harmless, but it would spam every tick's log.
            warnings.filterwarnings("ignore", message="Unable to import Axes3D")
            import torch
            from ultralytics import YOLO
            self._model = YOLO(str(weights))
            self._device = 0 if torch.cuda.is_available() else "cpu"
        except Exception as e:  # third-party loaders raise many types; wrap into our boundary error
            raise DetectorError(f"failed to load detector {weights.name}: {e}") from e
        self.version = f"{weights.name}@{settings.imgsz}"
        self.min_conf = settings.store_conf
        log.info("detector %s loaded on %s", self.version, self._device)

    def detect(self, image: np.ndarray, space: SpaceConfig) -> list[Detection]:
        try:
            r = self._model.predict(image, imgsz=self.cfg.imgsz, conf=self.cfg.store_conf, iou=self.cfg.iou,
                                    verbose=False, device=self._device)[0]
            names = self._model.names
            raw = [(xyxyn, conf, names[int(cls)])
                   for xyxyn, conf, cls in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        except Exception as e:
            raise DetectorError(f"detector inference failed: {e}") from e
        return build_detections(raw, space, self.cfg)

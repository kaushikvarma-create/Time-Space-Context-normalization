"""YOLO (Ultralytics) detector with a per-space class whitelist."""
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
        log.info("detector %s loaded on %s", self.version, self._device)

    def detect(self, image: np.ndarray, space: SpaceConfig) -> list[Detection]:
        try:
            r = self._model.predict(image, imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou,
                                    verbose=False, device=self._device)[0]
            boxes = list(zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()))
            names = self._model.names
        except Exception as e:
            raise DetectorError(f"detector inference failed: {e}") from e
        dets = []
        for xyxyn, conf, cls in boxes:
            label = space.detector_map.get(names[int(cls)])
            if label is None:   # class not whitelisted for this space: noise (e.g. refrigerator in a car park)
                continue
            dets.append(Detection(label, round(float(conf), 3), tuple(round(v, 4) for v in xyxyn)))
        dets = dedupe_same_label_iou(dets, self.cfg.dedupe_same_label_iou)
        return dedupe_contained(dets, set(self.cfg.dedupe_labels), self.cfg.dedupe_containment)

"""SAM 2 (via Ultralytics) as a box-prompted segmenter: one outline per detector box.

SAM 2 does not name or find objects; it only traces the outline of what is inside each box it is given.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..errors import SegmenterError
from ..grid import BBox
from ..settings import SegmenterSettings
from .types import SegMask

log = logging.getLogger(__name__)


class Sam2Segmenter:
    def __init__(self, settings: SegmenterSettings, root: Path):
        self.cfg = settings
        weights = Path(settings.weights)
        weights = weights if weights.is_absolute() else root / weights
        if not weights.exists():
            raise SegmenterError(f"segmenter weights not found: {weights}")
        try:
            import warnings
            warnings.filterwarnings("ignore", message="Unable to import Axes3D")
            import torch
            from ultralytics import SAM
            self._torch = torch
            self._model = SAM(str(weights))
            self._device = 0 if torch.cuda.is_available() else "cpu"
        except Exception as e:  # third-party loaders raise many types; wrap into our boundary error
            raise SegmenterError(f"failed to load segmenter {weights.name}: {e}") from e
        self.version = f"{weights.name}@{settings.mask_width}"
        log.info("segmenter %s loaded on %s", self.version, self._device)

    def segment(self, image: np.ndarray, boxes: list[BBox]) -> list[SegMask]:
        if not boxes:
            return []
        h, w = image.shape[:2]
        gw = self.cfg.mask_width
        gh = max(1, round(gw * h / w))
        out: list[SegMask] = []
        try:
            torch = self._torch
            for start in range(0, len(boxes), self.cfg.batch):
                chunk = boxes[start:start + self.cfg.batch]
                pixel = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in chunk]
                r = self._model(image, bboxes=pixel, verbose=False, device=self._device)[0]
                if r.masks is None or len(r.masks.data) != len(chunk):
                    got = 0 if r.masks is None else len(r.masks.data)
                    raise SegmenterError(f"segmenter returned {got} masks for {len(chunk)} boxes")
                m = r.masks.data.float()                                     # (n, H, W), one per box, in order
                areas = m.mean(dim=(1, 2)).tolist()
                grids = (torch.nn.functional.interpolate(m[:, None], size=(gh, gw), mode="area")[:, 0] >= 0.5)
                for area, grid in zip(areas, grids.cpu().numpy()):
                    out.append(SegMask(round(float(area), 6), grid))
        except SegmenterError:
            raise
        except Exception as e:
            raise SegmenterError(f"segmenter inference failed: {e}") from e
        return out

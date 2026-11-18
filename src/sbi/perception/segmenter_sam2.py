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

    def _outlines(self, image: np.ndarray, pixel: list[list[float]]):
        """Masks for these boxes as an (n, H, W) float tensor in box order, or None when the model returned a
        different number of masks than boxes (it drops a box it finds nothing in)."""
        r = self._model(image, bboxes=pixel, verbose=False, device=self._device)[0]
        if r.masks is None or len(r.masks.data) != len(pixel):
            return None
        return r.masks.data.float()

    def segment(self, image: np.ndarray, boxes: list[BBox]) -> list[SegMask]:
        """One outline per box, in order. A box the model finds nothing in gets an empty outline (area 0) instead of
        costing the whole photo its outlines: when a batch comes back short, its boxes are asked one at a time."""
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
                m = self._outlines(image, pixel)
                if m is None:                                                # short batch: find the empty box(es)
                    singles = [self._outlines(image, [box]) for box in pixel]
                    found = [one for one in singles if one is not None]
                    if not found:
                        raise SegmenterError(f"segmenter returned no mask for any of {len(chunk)} boxes")
                    blank = torch.zeros_like(found[0])
                    m = torch.cat([one if one is not None else blank for one in singles])
                    log.info("segmenter: %d of %d boxes had nothing to outline (stored as empty outlines)",
                             len(singles) - len(found), len(chunk))
                areas = m.mean(dim=(1, 2)).tolist()
                grids = (torch.nn.functional.interpolate(m[:, None], size=(gh, gw), mode="area")[:, 0] >= 0.5)
                for area, grid in zip(areas, grids.cpu().numpy()):
                    out.append(SegMask(round(float(area), 6), grid))
        except SegmenterError:
            raise
        except Exception as e:
            raise SegmenterError(f"segmenter inference failed: {e}") from e
        return out

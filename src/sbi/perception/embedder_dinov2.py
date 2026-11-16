"""DINOv2 as a whole-scene embedder: one vector per photo, describing what the scene looks like.

The vector is [CLS | mean of patch tokens], each half L2-normalised, so cosine similarity on the whole vector is
the average of the two. The photo is resized whole (no centre crop: a crop would drop the edges of a wide camera
view) to `width` x (width * h / w, snapped to the 14-pixel patch grid). The camera's burned-in clock was tested
and does not move the vector measurably, so nothing is masked.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..errors import EmbedderError
from ..settings import EmbedderSettings

log = logging.getLogger(__name__)
PATCH = 14
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare(image_bgr: np.ndarray, width: int) -> np.ndarray:
    """BGR uint8 image -> normalised float32 array of shape (3, H, W) with W == width and H a multiple of 14."""
    h, w = image_bgr.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("empty image")
    out_h = max(PATCH, round(width * h / w / PATCH) * PATCH)
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (width, out_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)


def combine(cls_token: np.ndarray, patch_tokens: np.ndarray) -> np.ndarray:
    """[CLS | mean patch], each half unit length. cls_token: (d,), patch_tokens: (n, d)."""
    def unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
    return np.concatenate([unit(cls_token), unit(patch_tokens.mean(axis=0))]).astype(np.float32)


class DinoV2Embedder:
    def __init__(self, settings: EmbedderSettings, root: Path):
        self.cfg = settings
        model_ref = settings.model
        local = root / model_ref
        if not Path(model_ref).is_absolute() and local.exists():
            model_ref = str(local)
        try:
            import torch
            from transformers import AutoModel
            self._torch = torch
            try:    # prefer the local cache: no network round trip, and a tick works while offline
                model = AutoModel.from_pretrained(model_ref, local_files_only=True)
            except OSError:
                model = AutoModel.from_pretrained(model_ref)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.to(self._device).eval()
            self.dim = 2 * int(model.config.hidden_size)
        except Exception as e:  # third-party loaders raise many types; wrap into our boundary error
            raise EmbedderError(f"failed to load embedder {settings.model}: {e}") from e
        self.version = settings.version
        log.info("embedder %s loaded on %s (dim %d)", self.version, self._device, self.dim)

    def embed(self, image: np.ndarray) -> np.ndarray:
        try:
            torch = self._torch
            x = torch.from_numpy(prepare(image, self.cfg.width))[None].to(self._device)
            with torch.inference_mode():
                tokens = self._model(pixel_values=x).last_hidden_state[0].float().cpu().numpy()
            vec = combine(tokens[0], tokens[1:])
        except Exception as e:
            raise EmbedderError(f"embedding failed: {e}") from e
        if vec.size != self.dim or not np.isfinite(vec).all():
            raise EmbedderError(f"embedder returned an unusable vector (size {vec.size}, expected {self.dim})")
        return vec

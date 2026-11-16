"""SigLIP 2 as a text-prompt scorer: each plain-English description gets its own score for a photo.

Unlike CLIP's pick-one-of-N, SigLIP scores every prompt independently (a sigmoid per prompt, so several can be
true at once, or none). The raw logit is stored: 0 is an even-odds match, and logits keep their resolution where
probabilities would saturate near 0 and 1. The photo is squashed whole to the model's square input size.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..errors import ScorerError
from ..settings import ScorerSettings

log = logging.getLogger(__name__)
MAX_TOKENS = 64


class Siglip2Scorer:
    def __init__(self, settings: ScorerSettings, root: Path):
        self.cfg = settings
        ref = settings.model
        local = root / ref
        if not Path(ref).is_absolute() and local.exists():
            ref = str(local)
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._dtype = torch.float16 if self._device == "cuda" else torch.float32
            try:    # prefer the local cache: no network round trip, and a tick works while offline
                self._proc = AutoProcessor.from_pretrained(ref, local_files_only=True)
                model = AutoModel.from_pretrained(ref, dtype=self._dtype, local_files_only=True)
            except OSError:
                self._proc = AutoProcessor.from_pretrained(ref)
                model = AutoModel.from_pretrained(ref, dtype=self._dtype)
            self._model = model.to(self._device).eval()
        except Exception as e:  # third-party loaders raise many types; wrap into our boundary error
            raise ScorerError(f"failed to load scorer {settings.model}: {e}") from e
        self.version = settings.version
        log.info("scorer %s loaded on %s", self.version, self._device)

    def score(self, image: np.ndarray, prompts: list[str]) -> list[float]:
        if not prompts:
            return []
        try:
            torch = self._torch
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self._proc(text=[p.lower() for p in prompts], images=rgb, padding="max_length",
                                max_length=MAX_TOKENS, return_tensors="pt")
            inputs = {k: (v.to(self._device, self._dtype) if v.dtype.is_floating_point else v.to(self._device))
                      for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self._model(**inputs).logits_per_image[0].float().cpu().numpy()
        except Exception as e:
            raise ScorerError(f"scoring failed: {e}") from e
        if logits.shape != (len(prompts),) or not np.isfinite(logits).all():
            raise ScorerError(f"scorer returned an unusable result (shape {logits.shape}, expected ({len(prompts)},))")
        return [float(v) for v in logits]

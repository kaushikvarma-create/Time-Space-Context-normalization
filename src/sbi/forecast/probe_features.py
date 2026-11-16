"""Concept columns from stored probe scores.

A concept is a few paraphrases of one idea ("an empty parking area", "a parking lot with no vehicles", ...). Its
value for a frame is the mean raw score of its prompts, so one unlucky wording does not move the number. If any
prompt of a concept has no score for a frame (never asked, or the scorer failed) the concept is unknown (NaN) for
that frame: a mean over fewer paraphrases would be a different measurement, not a missing one.
"""
from __future__ import annotations

import numpy as np


def concept_values(scores: dict[str, float], groups: dict[str, list[str]]) -> dict[str, float]:
    """{concept: mean score} for one frame; NaN where any of the concept's prompts is missing."""
    out: dict[str, float] = {}
    for concept, prompts in groups.items():
        vals = [scores[p] for p in prompts if p in scores]
        out[concept] = float(np.mean(vals)) if prompts and len(vals) == len(prompts) else float("nan")
    return out

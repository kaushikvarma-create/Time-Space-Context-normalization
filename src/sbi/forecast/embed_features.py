"""Numbers derived from stored whole-scene image vectors, one value per 15-minute slot.

Computed when the table is built (never stored), so the rules can change without re-running a model:
  pcN           position along the N-th principal direction of how the scene varies (N = 1..components)
  drift_prev    cosine distance to the previous slot's vector       (how much the scene changed in 15 minutes)
  drift_typical cosine distance to the typical scene (mean vector)  (how unusual the scene looks right now)

Pure numpy: rows are slots, a NaN row is a slot without a vector (unknown), and every derived value that needs
that row is NaN too. `fit` says which slots may be used to learn the principal directions and the typical
scene; leave it as None to fit on everything, or pass the training window to keep later data out of the fit
(otherwise a back-test would see the future through the components).
"""
from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise 1 - cos(a, b); NaN where either row is NaN or has zero length."""
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.einsum("ij,ij->i", a, b)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        d = 1.0 - num / den
    d[~np.isfinite(d)] = np.nan
    return np.clip(d, 0.0, 2.0)


def principal_axes(x: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean, axes[count, dim]) of the rows of x. Axes are unit length, ordered by variance, and signed so the
    largest-magnitude loading is positive (so the same data always gives the same columns). Fewer than `count`
    axes come back when the data has fewer directions of variation (e.g. 3 rows give at most 2)."""
    mean = x.mean(axis=0)
    xc = x - mean
    n = x.shape[0]
    cov = xc.T @ xc / max(n - 1, 1)
    vals, vecs = np.linalg.eigh(cov)                         # ascending
    order = np.argsort(vals)[::-1][: min(count, max(n - 1, 0))]
    axes = vecs[:, order].T
    for i in range(axes.shape[0]):
        if axes[i][np.argmax(np.abs(axes[i]))] < 0:
            axes[i] = -axes[i]
    return mean, axes


def embedding_features(vectors: np.ndarray, *, components: int, fit: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """vectors: (slots, dim) float, NaN rows for slots without a vector. Returns name -> (slots,) arrays:
    pc1..pcN (N = components), drift_prev and drift_typical."""
    n_slots = vectors.shape[0]
    have = ~np.isnan(vectors).any(axis=1)
    use = have if fit is None else have & fit
    out: dict[str, np.ndarray] = {}

    prev = np.full_like(vectors, np.nan)
    prev[1:] = vectors[:-1]
    drift_prev = np.full(n_slots, np.nan)
    if n_slots > 1:
        drift_prev = cosine_distance(vectors, prev)

    typical = np.full(n_slots, np.nan)
    pcs = {f"pc{i + 1}": np.full(n_slots, np.nan) for i in range(components)}
    if use.sum() >= 2:
        mean, axes = principal_axes(vectors[use].astype(np.float64), components)
        if have.any():
            typical[have] = cosine_distance(vectors[have], np.broadcast_to(mean, vectors[have].shape))
            for i in range(axes.shape[0]):
                pcs[f"pc{i + 1}"][have] = (vectors[have] - mean) @ axes[i]
    out.update(pcs)
    out["drift_prev"] = drift_prev
    out["drift_typical"] = typical
    return out

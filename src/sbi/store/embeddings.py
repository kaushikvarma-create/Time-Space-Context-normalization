"""Whole-scene image vectors: encode/decode and the frame_embeddings table.

Vectors are stored as float16 (3 KB for 1536 numbers). The rounding error (~1e-3 relative) is far below the
scene-to-scene differences these vectors are used for (see the overnight probe in the research notes).
Write functions assume the caller holds a transaction, like the rest of the result writers.
"""
from __future__ import annotations

import sqlite3

import numpy as np

from ..errors import StorageError


def encode_vector(vec: np.ndarray) -> bytes:
    v = np.asarray(vec)
    if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all():
        raise ValueError("an embedding must be a non-empty 1-D vector of finite numbers")
    return v.astype("<f2").tobytes()


def decode_vector(blob: bytes, dim: int) -> np.ndarray:
    if len(blob) != 2 * dim:
        raise ValueError(f"corrupt embedding blob: {len(blob)} bytes for dim {dim}")
    return np.frombuffer(blob, dtype="<f2").astype(np.float32)


def put_embedding(conn: sqlite3.Connection, frame_id: int, model: str, vec: np.ndarray) -> None:
    """Store (or replace) the frame's vector for this model. Other models' rows are left alone."""
    blob = encode_vector(vec)
    conn.execute("INSERT OR REPLACE INTO frame_embeddings(frame_id, model, dim, vec) VALUES(?,?,?,?)",
                 (frame_id, model, int(np.asarray(vec).size), blob))


def load_embeddings(conn: sqlite3.Connection, space_id: str, model: str,
                    frame_ids: set[int] | None = None) -> tuple[dict[int, np.ndarray], list[int]]:
    """(frame_id -> vector, ids of corrupt rows) for one space and model. A corrupt row is left out, so that slot
    reads as unknown; vectors of mixed dimensions in one model are a StorageError."""
    out: dict[int, np.ndarray] = {}
    corrupt: list[int] = []
    for fid, dim, blob in conn.execute(
            "SELECT e.frame_id, e.dim, e.vec FROM frame_embeddings e JOIN frames f USING(frame_id) "
            "WHERE f.space_id=? AND e.model=?", (space_id, model)):
        if frame_ids is not None and fid not in frame_ids:
            continue
        try:
            out[fid] = decode_vector(blob, dim)
        except ValueError:
            corrupt.append(fid)
    dims = {v.size for v in out.values()}
    if len(dims) > 1:
        raise StorageError(f"embeddings for {space_id}/{model} have mixed dimensions {sorted(dims)}")
    return out, corrupt


def frames_missing_embedding(conn: sqlite3.Connection, space_id: str, model: str,
                             limit: int | None = None) -> list[int]:
    """Captured frames with a readable image path and no vector for this model, oldest first."""
    sql = ("SELECT f.frame_id FROM frames f LEFT JOIN frame_embeddings e "
           "ON e.frame_id=f.frame_id AND e.model=? "
           "WHERE f.space_id=? AND f.capture_status='ok' AND f.image_path IS NOT NULL AND e.frame_id IS NULL "
           "ORDER BY f.ts_utc")
    params: list = [model, space_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [r[0] for r in conn.execute(sql, params)]

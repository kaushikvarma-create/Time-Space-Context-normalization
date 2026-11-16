"""Fill in whole-scene vectors for frames that have none (history from before the embedder was switched on,
or frames where it failed). The live pipeline embeds new frames itself (pipeline.process_frame); this is the
catch-up path and shares the same storage function, so both write identical rows.

A frame that cannot be embedded (unreadable image, embedder failure) is counted and skipped; it never stops
the run, and nothing is written for it (unknown, not zero).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from .errors import EmbedderError
from .perception.types import Embedder
from .store.db import transaction
from .store.embeddings import frames_missing_embedding, put_embedding
from .store.frames import get_frame

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedReport:
    done: int
    failed: int


def embed_missing(conn, space_id: str, embedder: Embedder, *, root: Path, limit: int | None = None) -> EmbedReport:
    done = failed = 0
    for fid in frames_missing_embedding(conn, space_id, embedder.version, limit):
        frame = get_frame(conn, fid)
        path = Path(frame["image_path"])
        img = cv2.imread(str(path if path.is_absolute() else root / path))
        if img is None:
            log.warning("%s frame %d: image unreadable, skipping", space_id, fid)
            failed += 1
            continue
        try:
            vec = embedder.embed(img)
            if vec.shape != (embedder.dim,):
                raise EmbedderError(f"vector of shape {vec.shape}, expected ({embedder.dim},)")
        except EmbedderError as e:
            log.warning("%s frame %d: %s", space_id, fid, e)
            failed += 1
            continue
        with transaction(conn):
            put_embedding(conn, fid, embedder.version, vec)
        done += 1
    return EmbedReport(done, failed)

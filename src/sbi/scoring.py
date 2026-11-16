"""Ask every stored photo the space's current probe prompts that it has no score for yet.

Two uses, one path: scoring history after the scorer is switched on, and asking a NEW question of all past photos
(add a prompt to the space file, run `sbi score`). The live pipeline scores new frames itself
(pipeline.process_frame) and stores through the same function, so both write identical rows.

A frame that cannot be scored (unreadable image, scorer failure) is counted and skipped; it never stops the run,
and `limit` counts frames actually scored, so a frame that keeps failing cannot use up every limited run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .errors import ScorerError
from .imageio import read_frame_image
from .perception.types import Scorer
from .spaces import SpaceConfig
from .store.db import transaction
from .store.frames import get_frame
from .store.scores import frames_missing_scores, put_scores

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreReport:
    done: int
    failed: int


def score_missing(conn, space: SpaceConfig, scorer: Scorer, *, root: Path, limit: int | None = None) -> ScoreReport:
    done = failed = 0
    for fid, missing in frames_missing_scores(conn, space.space_id, scorer.version, space.probe_prompts):
        if limit is not None and done >= limit:
            break
        img = read_frame_image(get_frame(conn, fid), root)
        if img is None:
            log.warning("%s frame %d: image unreadable, skipping", space.space_id, fid)
            failed += 1
            continue
        try:
            values = scorer.score(img, missing)
            if len(values) != len(missing):
                raise ScorerError(f"{len(values)} scores for {len(missing)} prompts")
        except ScorerError as e:
            log.warning("%s frame %d: %s", space.space_id, fid, e)
            failed += 1
            continue
        with transaction(conn):
            put_scores(conn, fid, scorer.version, dict(zip(missing, values)))
        done += 1
    return ScoreReport(done, failed)

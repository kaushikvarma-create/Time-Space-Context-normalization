"""Text-prompt scores per frame (table frame_scores). Write functions assume the caller holds a transaction."""
from __future__ import annotations

import math
import sqlite3
from collections import defaultdict


def put_scores(conn: sqlite3.Connection, frame_id: int, scorer: str, scores: dict[str, float]) -> None:
    """Store (or replace) the frame's scores for these prompts. Other prompts and other scorers are left alone."""
    bad = [p for p, v in scores.items() if not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite scores for prompts {bad[:3]}")
    conn.executemany("INSERT OR REPLACE INTO frame_scores(frame_id, scorer, prompt, value) VALUES(?,?,?,?)",
                     [(frame_id, scorer, p, float(v)) for p, v in scores.items()])


def load_scores(conn: sqlite3.Connection, space_id: str, scorer: str,
                frame_ids: set[int] | None = None) -> dict[int, dict[str, float]]:
    """frame_id -> {prompt: score} for one space and scorer."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for fid, prompt, value in conn.execute(
            "SELECT s.frame_id, s.prompt, s.value FROM frame_scores s JOIN frames f USING(frame_id) "
            "WHERE f.space_id=? AND s.scorer=?", (space_id, scorer)):
        if frame_ids is None or fid in frame_ids:
            out[fid][prompt] = value
    return dict(out)


def frames_missing_scores(conn: sqlite3.Connection, space_id: str, scorer: str,
                          prompts: list[str]) -> list[tuple[int, list[str]]]:
    """Captured frames lacking a score for some of `prompts`, oldest first, with the prompts each one lacks."""
    have = load_scores(conn, space_id, scorer)
    out: list[tuple[int, list[str]]] = []
    for (fid,) in conn.execute("SELECT frame_id FROM frames WHERE space_id=? AND capture_status='ok' "
                               "AND image_path IS NOT NULL ORDER BY ts_utc", (space_id,)):
        missing = [p for p in prompts if p not in have.get(fid, {})]
        if missing:
            out.append((fid, missing))
    return out

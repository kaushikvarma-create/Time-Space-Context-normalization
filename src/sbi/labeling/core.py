"""Ground-truth labelling logic (UI-independent so it is testable)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from ..errors import StorageError
from ..store.db import transaction
from ..timeutil import now_utc


def progress(conn: sqlite3.Connection, space_id: str) -> tuple[int, int]:
    """(labelled, labellable) frames for a space."""
    done = conn.execute("SELECT COUNT(*) FROM label_frames l JOIN frames f USING(frame_id) WHERE f.space_id=?",
                        (space_id,)).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM frames WHERE space_id=? AND capture_status='ok' "
                         "AND processed_utc IS NOT NULL", (space_id,)).fetchone()[0]
    return done, total


STRATA = [(we, part) for we in (0, 1) for part in ("morning", "afternoon", "evening", "night")]


def stratum(dow: int, part_of_day: str) -> tuple[int, str]:
    return int(dow >= 5), part_of_day


def next_unlabeled(conn: sqlite3.Connection, space_id: str) -> int | None:
    """The next frame to label, spread evenly over (weekday/weekend x part of day): it comes from the stratum with
    the fewest labels so far (ties in a fixed order), and within a stratum from a stable pseudo-random order. So the
    first 50 labels cover busy and quiet times alike instead of mostly empty nights, and never only the newest."""
    labelled = {s: 0 for s in STRATA}
    for dow, part in conn.execute(
            "SELECT f.dow, f.part_of_day FROM label_frames l JOIN frames f USING(frame_id) WHERE f.space_id=?",
            (space_id,)):
        labelled[stratum(dow, part)] = labelled.get(stratum(dow, part), 0) + 1
    candidates: dict[tuple[int, str], int] = {}
    for fid, dow, part in conn.execute(
            """SELECT frame_id, dow, part_of_day FROM frames f WHERE space_id=? AND capture_status='ok'
                 AND processed_utc IS NOT NULL AND NOT EXISTS (SELECT 1 FROM label_frames l WHERE l.frame_id=f.frame_id)
               ORDER BY (frame_id * 7919) % 10007""", (space_id,)):
        candidates.setdefault(stratum(dow, part), fid)                  # first in the stable order per stratum
    if not candidates:
        return None
    order = {s: i for i, s in enumerate(STRATA)}
    best = min(candidates, key=lambda s: (labelled.get(s, 0), order.get(s, len(order))))
    return candidates[best]


def quadrants(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The photo cut into four full-resolution quarters (no overlap, so nothing is counted twice), for counting
    dense areas that are too small to see in the whole-photo view."""
    h, w = img.shape[:2]
    my, mx = h // 2, w // 2
    return [("Top left", img[:my, :mx]), ("Top right", img[:my, mx:]),
            ("Bottom left", img[my:, :mx]), ("Bottom right", img[my:, mx:])]


def save_label(conn: sqlite3.Connection, frame_id: int, counts: dict[str, int], *, usable: bool, labeler: str,
               note: str = "") -> None:
    if any(n < 0 for n in counts.values()):
        raise ValueError("counts must be non-negative")
    with transaction(conn):
        conn.execute("INSERT OR REPLACE INTO label_frames(frame_id,usable,labeler,labeled_utc,note) VALUES(?,?,?,?,?)",
                     (frame_id, int(usable), labeler, now_utc().isoformat(timespec="seconds"), note))
        conn.execute("DELETE FROM labels WHERE frame_id=?", (frame_id,))
        if usable:
            conn.executemany("INSERT INTO labels(frame_id,object_type,true_count) VALUES(?,?,?)",
                             [(frame_id, t, int(n)) for t, n in counts.items()])


def model_counts(conn: sqlite3.Connection, frame_id: int, types: list[str]) -> dict[str, dict]:
    """What the models said (revealed only after the human saves). vlm is None if the VLM failed."""
    raw = conn.execute("SELECT vlm_raw FROM frames WHERE frame_id=?", (frame_id,)).fetchone()
    vlm_failed = raw is None or raw[0] is None or raw[0].startswith('{"error"')
    out = {}
    for t in types:
        def total(source: str) -> int:
            return int(conn.execute("SELECT COALESCE(SUM(count),0) FROM observations WHERE frame_id=? "
                                    "AND object_type=? AND source=?", (frame_id, t, source)).fetchone()[0])
        out[t] = {"yolo": total("yolo"), "vlm": None if vlm_failed else total("vlm")}
    return out


def load_frame_image(conn: sqlite3.Connection, frame_id: int, root: Path) -> np.ndarray:
    row = conn.execute("SELECT image_path FROM frames WHERE frame_id=?", (frame_id,)).fetchone()
    if row is None or not row[0]:
        raise StorageError(f"frame {frame_id} has no stored image")
    p = Path(row[0])
    img = cv2.imread(str(p if p.is_absolute() else root / p))
    if img is None:
        raise StorageError(f"cannot read image {p}")
    return img


def draw_boxes(conn: sqlite3.Connection, frame_id: int, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for label, bb, conf in conn.execute(
            "SELECT object_type,bbox_norm,confidence FROM observations WHERE frame_id=? AND source='yolo' "
            "AND bbox_norm IS NOT NULL", (frame_id,)):
        x1, y1, x2, y2 = [int(v * s) for v, s in zip(json.loads(bb), (w, h, w, h))]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    return out


def to_rgb_small(img: np.ndarray, width: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.cvtColor(cv2.resize(img, (width, int(width * h / w))), cv2.COLOR_BGR2RGB)

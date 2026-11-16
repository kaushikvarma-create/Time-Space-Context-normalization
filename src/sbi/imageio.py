"""Reading a stored frame's image: one place for the path rule (relative to the project root unless absolute)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frame_image(frame, root: Path) -> np.ndarray | None:
    """BGR image of a frame row, or None when the file is missing or unreadable."""
    if not frame["image_path"]:
        return None
    return cv2.imread(str(root / frame["image_path"]))      # an absolute stored path wins over the root (pathlib)

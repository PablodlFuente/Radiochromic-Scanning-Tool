"""Lossless image-loading helpers used by the application and calibration UI."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_unchanged(file_path: str | Path) -> np.ndarray:
    """Read an image without changing its path or reducing its stored precision.

    ``cv2.imread`` has historically been unreliable with some Unicode paths on
    Windows.  Reading the bytes through NumPy and decoding them with OpenCV
    avoids renaming scientific input files.
    """
    path = Path(file_path)
    encoded = np.fromfile(path, dtype=np.uint8)
    if encoded.size == 0:
        raise OSError(f"Image file is empty: {path}")

    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not decode image: {path}")
    return image

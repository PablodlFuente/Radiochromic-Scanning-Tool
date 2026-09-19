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


def storage_bit_depth(image: np.ndarray) -> tuple[int, float]:
    """Return the array storage depth and its representable maximum.

    The observed maximum is deliberately not used: a dark 16-bit scan remains
    a 16-bit scan.  Acquisition depths packed into a wider integer container
    must be supplied by metadata or configuration rather than guessed from one
    image's content.
    """
    dtype = image.dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.bits, float(info.max)
    if np.issubdtype(dtype, np.floating):
        finite = image[np.isfinite(image)]
        observed_max = float(np.max(finite)) if finite.size else 1.0
        return np.dtype(dtype).itemsize * 8, max(observed_max, 1.0)
    raise TypeError(f"Unsupported image dtype: {dtype}")


def write_tiff_unchanged(file_path: str | Path, rgb_image: np.ndarray) -> None:
    """Write an RGB/grayscale TIFF while preserving the NumPy integer depth."""
    path = Path(file_path)
    image = rgb_image
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".tif", image)
    if not ok:
        raise OSError(f"Could not encode TIFF: {path}")
    encoded.tofile(path)

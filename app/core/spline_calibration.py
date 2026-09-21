"""Shape-preserving cubic calibration models and durable storage."""

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from app.utils.atomic_file import atomic_open


SPLINE_NAME = "spline_calibration.npz"


def prepare_spline_knots(doses, intensities):
    """Sort observations, average replicated doses and validate monotonicity."""
    doses = np.asarray(doses, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    finite = np.isfinite(doses) & np.isfinite(intensities)
    doses, intensities = doses[finite], intensities[finite]
    if doses.size < 3:
        raise ValueError("At least three finite calibration points are required for a spline.")

    order = np.argsort(doses, kind="stable")
    doses, intensities = doses[order], intensities[order]
    unique_doses, inverse = np.unique(doses, return_inverse=True)
    sums = np.bincount(inverse, weights=intensities)
    counts = np.bincount(inverse)
    mean_intensities = sums / counts
    if unique_doses.size < 3:
        raise ValueError("At least three distinct doses are required for a spline.")

    differences = np.diff(mean_intensities)
    scale = max(float(np.max(np.abs(mean_intensities))), 1.0)
    tolerance = np.finfo(np.float64).eps * scale * 64
    increasing = np.all(differences > tolerance)
    decreasing = np.all(differences < -tolerance)
    if not (increasing or decreasing):
        raise ValueError(
            "Spline intensities must be strictly monotonic after replicate doses are averaged. "
            "Review or exclude non-monotonic calibration points."
        )
    return unique_doses, mean_intensities


def evaluate_spline(doses, knot_doses, knot_intensities):
    """Evaluate the shape-preserving piecewise cubic dose-response curve."""
    return PchipInterpolator(knot_doses, knot_intensities, extrapolate=False)(doses)


def invert_spline_response(intensities, knot_doses, knot_intensities):
    """Convert intensities inside the calibrated spline interval to dose."""
    pixels = np.asarray(intensities, dtype=np.float64)
    knot_doses, knot_intensities = prepare_spline_knots(knot_doses, knot_intensities)
    if knot_intensities[0] > knot_intensities[-1]:
        inverse_intensity = knot_intensities[::-1]
        inverse_dose = knot_doses[::-1]
    else:
        inverse_intensity = knot_intensities
        inverse_dose = knot_doses

    low, high = float(inverse_intensity[0]), float(inverse_intensity[-1])
    tolerance = np.finfo(np.float64).eps * max(abs(low), abs(high), 1.0) * 64
    valid = np.isfinite(pixels) & (pixels >= low - tolerance) & (pixels <= high + tolerance)
    dose = np.full(pixels.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        inverse = PchipInterpolator(inverse_intensity, inverse_dose, extrapolate=False)
        dose[valid] = inverse(np.clip(pixels[valid], low, high))
        valid &= np.isfinite(dose)
        dose[~valid] = np.nan
    return dose, valid


def save_spline_calibration(path, channel_knots, bit_depth):
    """Atomically save the exact knots needed to reconstruct each spline."""
    destination = Path(path)
    payload = {"schema_version": np.array(1), "bit_depth": np.array(int(bit_depth))}
    for channel in ("R", "G", "B"):
        doses, intensities = channel_knots[channel]
        payload[f"dose_{channel}"] = np.asarray(doses, dtype=np.float64)
        payload[f"intensity_{channel}"] = np.asarray(intensities, dtype=np.float64)
    with atomic_open(destination, "wb") as handle:
        np.savez(handle, **payload)


def load_spline_calibration(path):
    """Load and validate a spline calibration without pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        bit_depth = int(archive["bit_depth"])
        channels = {}
        for channel in ("R", "G", "B"):
            channels[channel] = prepare_spline_knots(
                archive[f"dose_{channel}"], archive[f"intensity_{channel}"]
            )
    return channels, bit_depth

"""Numerical dosimetry primitives independent of the graphical interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CHANNELS = ("R", "G", "B")


@dataclass(frozen=True)
class CombinedEstimate:
    value: float
    uncertainty: float
    weights: np.ndarray
    method: str


def rational_response(dose, a: float, b: float, c: float):
    """Calibration response ``I = a + b / (D - c)``."""
    dose = np.asarray(dose, dtype=np.float64)
    return a + b / (dose - c)


def invert_rational_response(
    intensity,
    params: Iterable[float],
    dose_range: tuple[float, float] | None = None,
    *,
    extrapolation_margin_fraction: float = 0.0,
    allow_extrapolation: bool = False,
):
    """Convert intensity to dose and return validity/extrapolation masks."""
    a, b, c = (float(value) for value in params)
    intensity = np.asarray(intensity, dtype=np.float64)
    denominator = intensity - a
    tolerance = np.finfo(np.float64).eps * max(abs(a), abs(b), 1.0) * 16.0
    mathematically_valid = np.isfinite(intensity) & (np.abs(denominator) > tolerance)

    dose = np.full(intensity.shape, np.nan, dtype=np.float64)
    dose[mathematically_valid] = c + b / denominator[mathematically_valid]
    extrapolated = np.zeros(intensity.shape, dtype=bool)

    if dose_range is not None:
        low, high = sorted(float(value) for value in dose_range)
        range_tolerance = np.finfo(np.float64).eps * max(abs(low), abs(high), 1.0) * 32.0
        extrapolated = mathematically_valid & (
            (dose < low - range_tolerance) | (dose > high + range_tolerance)
        )
        if not allow_extrapolation:
            margin = max(0.0, float(extrapolation_margin_fraction)) * (high - low)
            accepted = mathematically_valid & (
                (dose >= low - margin - range_tolerance)
                & (dose <= high + margin + range_tolerance)
            )
            dose[~accepted] = np.nan
            mathematically_valid = accepted

    return dose, mathematically_valid, extrapolated


def parameter_uncertainty_for_mean_dose(
    intensities,
    params: Iterable[float],
    covariance,
) -> float:
    """Propagate shared fit-parameter uncertainty to the mean pixel dose.

    Fit parameters are shared by every pixel, so their contribution is based
    on the mean Jacobian and is not divided by the number of pixels.
    """
    a, b, _c = (float(value) for value in params)
    values = np.asarray(intensities, dtype=np.float64)
    denominator = values - a
    valid = np.isfinite(values) & (np.abs(denominator) > np.finfo(float).eps * max(abs(a), 1.0) * 16)
    if not np.any(valid):
        return float("nan")

    denominator = denominator[valid]
    jacobian = np.column_stack((
        b / denominator**2,
        1.0 / denominator,
        np.ones_like(denominator),
    ))
    mean_jacobian = np.mean(jacobian, axis=0)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return float("nan")
    if not np.allclose(covariance, covariance.T) or np.any(np.diag(covariance) < 0):
        return float("nan")
    scales = np.sqrt(np.maximum(np.diag(covariance), np.finfo(float).tiny))
    correlation = covariance / scales[:, None] / scales[None, :]
    if not np.all(np.isfinite(correlation)) or np.min(np.linalg.eigvalsh(correlation)) < -1e-8:
        return float("nan")
    variance = float(mean_jacobian @ covariance @ mean_jacobian.T)
    return float(np.sqrt(max(variance, 0.0)))


def combine_channel_estimates(means, uncertainties, method: str) -> CombinedEstimate:
    """Combine channel estimates with statistically interpretable weights."""
    means = np.asarray(means, dtype=np.float64)
    uncertainties = np.asarray(uncertainties, dtype=np.float64)
    valid = np.isfinite(means) & np.isfinite(uncertainties) & (uncertainties >= 0)

    if not np.any(valid):
        finite_means = means[np.isfinite(means)]
        value = float(np.mean(finite_means)) if finite_means.size else float("nan")
        return CombinedEstimate(value, float("nan"), np.zeros(means.shape), method)

    values = means[valid]
    errors = uncertainties[valid]
    if np.any(errors == 0):
        # A constant ROI legitimately has zero repeatability uncertainty.  Exact
        # inverse-variance weights are singular, so use equal channel weights and
        # retain any observed inter-channel disagreement as a finite standard
        # uncertainty instead of emitting NaN.
        value = float(np.mean(values))
        propagated = float(np.sqrt(np.sum(errors**2)) / values.size)
        disagreement = (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size > 1 else 0.0
        )
        full_weights = np.zeros(means.shape, dtype=np.float64)
        full_weights[valid] = 1.0 / values.size
        return CombinedEstimate(
            value, max(propagated, disagreement), full_weights, method
        )
    base_weights = 1.0 / errors**2
    weights = base_weights.copy()
    effective_method = method

    if method == "dersimonian_laird" and values.size > 1:
        fixed_mean = float(np.average(values, weights=base_weights))
        q_stat = float(np.sum(base_weights * (values - fixed_mean) ** 2))
        denominator = float(np.sum(base_weights) - np.sum(base_weights**2) / np.sum(base_weights))
        tau_squared = max(0.0, (q_stat - (values.size - 1)) / denominator) if denominator > 0 else 0.0
        weights = 1.0 / (errors**2 + tau_squared)
    elif method not in {"weighted_average", "birge_factor", "sensitivity_weighted"}:
        effective_method = "weighted_average"

    value = float(np.average(values, weights=weights))
    uncertainty = float(1.0 / np.sqrt(np.sum(weights)))

    if method == "birge_factor" and values.size > 1:
        chi_squared = float(np.sum(base_weights * (values - value) ** 2))
        uncertainty *= np.sqrt(max(1.0, chi_squared / (values.size - 1)))

    full_weights = np.zeros(means.shape, dtype=np.float64)
    full_weights[valid] = weights / np.sum(weights)
    return CombinedEstimate(value, uncertainty, full_weights, effective_method)

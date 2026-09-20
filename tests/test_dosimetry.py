import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.core.dosimetry import (
    combine_channel_estimates,
    invert_rational_response,
    parameter_uncertainty_for_mean_dose,
    rational_response,
)
from app.core.image_processor import ImageProcessor


class CalibrationMathTests(unittest.TestCase):
    def test_rational_model_round_trip(self):
        params = (1000.0, 5000.0, -0.5)
        expected = np.array([0.0, 1.0, 5.0, 10.0])
        intensity = rational_response(expected, *params)
        actual, valid, extrapolated = invert_rational_response(
            intensity, params, (0.0, 10.0)
        )
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.all(valid))
        self.assertFalse(np.any(extrapolated))

    def test_out_of_range_dose_is_flagged_and_masked(self):
        params = (1000.0, 5000.0, -0.5)
        intensity = rational_response(np.array([20.0]), *params)
        dose, valid, extrapolated = invert_rational_response(
            intensity, params, (0.0, 10.0)
        )
        self.assertTrue(extrapolated[0])
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(dose[0]))

    def test_parameter_uncertainty_is_not_reduced_by_pixel_count(self):
        params = (1000.0, 5000.0, -0.5)
        covariance = np.diag([4.0, 9.0, 0.01])
        one = parameter_uncertainty_for_mean_dose([2500.0], params, covariance)
        many = parameter_uncertainty_for_mean_dose([2500.0] * 1000, params, covariance)
        self.assertAlmostEqual(one, many, places=12)

    def test_sensitivity_mode_is_invariant_to_intensity_units(self):
        means = np.array([1.0, 1.2, 0.9])
        errors = np.array([0.1, 0.2, 0.3])
        result = combine_channel_estimates(means, errors, "sensitivity_weighted")
        scaled_representation = combine_channel_estimates(means, errors, "sensitivity_weighted")
        self.assertAlmostEqual(result.value, scaled_representation.value)
        self.assertAlmostEqual(result.uncertainty, scaled_representation.uncertainty)

    def test_birge_factor_inflates_only_when_channels_disagree(self):
        consistent = combine_channel_estimates([1.0, 1.01, 0.99], [0.1] * 3, "birge_factor")
        inconsistent = combine_channel_estimates([1.0, 2.0, 3.0], [0.1] * 3, "birge_factor")
        self.assertGreater(inconsistent.uncertainty, consistent.uncertainty)

    def test_unknown_uncertainty_is_not_reported_as_zero(self):
        result = combine_channel_estimates([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], "weighted_average")
        self.assertTrue(np.isnan(result.uncertainty))

    def test_roi_uncertainty_combines_sampling_and_parameter_covariance(self):
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.calibration_applied = True
        processor.calibration_fit_params = {"G": (1000.0, 5000.0, -0.5)}
        processor.calibration_param_covariances = {"G": np.diag([4.0, 9.0, 0.01])}
        dose_pixels = np.array([[1.0], [1.2], [0.8]])
        source_pixels = np.array([[4300.0], [3900.0], [4800.0]])

        means, deviations, uncertainties = processor._summarize_roi_pixels(
            dose_pixels, source_pixels
        )

        sampling = deviations[0] / np.sqrt(3)
        parameter = parameter_uncertainty_for_mean_dose(
            source_pixels[:, 0],
            processor.calibration_fit_params["G"],
            processor.calibration_param_covariances["G"],
        )
        self.assertAlmostEqual(means[0], 1.0)
        self.assertAlmostEqual(uncertainties[0], np.hypot(sampling, parameter))

    def test_roi_statistics_ignore_invalid_calibrated_pixels(self):
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.calibration_applied = False
        means, deviations, uncertainties = processor._summarize_roi_pixels(
            np.array([[1.0], [np.nan], [3.0]])
        )
        self.assertEqual(means[0], 2.0)
        self.assertEqual(deviations[0], 1.0)
        self.assertAlmostEqual(uncertainties[0], 1.0 / np.sqrt(2))

    def test_missing_fit_uncertainty_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            fit_path = Path(directory) / "fit_parameters.csv"
            with fit_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["Channel", "a", "b", "c", "dose_min", "dose_max", "bit_depth"],
                )
                writer.writeheader()
                for channel in ("R", "G", "B"):
                    writer.writerow({
                        "Channel": channel, "a": 0, "b": 100, "c": -1,
                        "dose_min": 0, "dose_max": 10, "bit_depth": 8,
                    })

            processor = ImageProcessor.__new__(ImageProcessor)
            processor.current_image = np.full((2, 2, 3), 50, dtype=np.uint8)
            processor.original_image = processor.current_image.copy()
            processor.flattened_image = None
            processor.flat_applied = False
            processor.image_max_value = 255
            processor.config = {
                "allow_calibration_extrapolation": False,
                "calibration_extrapolation_margin_fraction": 0.0,
            }
            processor._find_fit_parameters_file = lambda: str(fit_path)
            processor._compute_integral_images = lambda: None

            with patch("app.core.image_processor.verify_calibration_manifest", return_value=None):
                self.assertTrue(processor.apply_calibration())

        self.assertTrue(np.isnan(processor.calibration_param_covariances["R"]).all())
        _, _, uncertainties = processor._summarize_roi_pixels(
            processor.current_image.reshape(-1, 3),
            processor.calibration_source_image.reshape(-1, 3),
        )
        self.assertTrue(np.isnan(uncertainties).all())


if __name__ == "__main__":
    unittest.main()

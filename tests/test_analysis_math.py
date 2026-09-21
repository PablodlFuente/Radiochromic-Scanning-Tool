import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt

from custom_plugins.analysis_tools import AnalysisTab, fit_weighted_line
from app.ui.measurement_panel import MeasurementPanel


class AnalysisMathTests(unittest.TestCase):
    def test_weighted_line_recovers_known_coefficients(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = 2.0 * xs + 0.5
        slope, intercept, slope_se, intercept_se, r_squared = fit_weighted_line(
            xs, ys, np.full(xs.shape, 0.1), False
        )
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 0.5)
        self.assertGreater(slope_se, 0.0)
        self.assertGreater(intercept_se, 0.0)
        self.assertAlmostEqual(r_squared, 1.0)

    def test_insufficient_regression_data_returns_nan_not_zero_uncertainty(self):
        result = fit_weighted_line([1.0], [2.0], [0.1], False)
        self.assertTrue(all(np.isnan(value) for value in result))

    def test_phase_correlation_uses_clipped_roi_origin(self):
        analysis = AnalysisTab.__new__(AnalysisTab)
        height = width = 30
        center_x, center_y, radius = 2.0, 3.0, 7.0
        yy, xx = np.mgrid[:height, :width]
        dose = np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2)
            / (2 * (radius * 0.5) ** 2)
        )
        roi, roi_xx, roi_yy, mask, *_ = analysis._extract_circle_roi(
            dose, center_x, center_y, radius
        )
        measured = analysis._centroid_phase_correlation(
            roi, roi_xx, roi_yy, mask, center_x, center_y, radius
        )
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured[0], center_x, delta=0.15)
        self.assertAlmostEqual(measured[1], center_y, delta=0.15)

    def test_isobars_keep_only_dominant_smooth_contours(self):
        analysis = AnalysisTab.__new__(AnalysisTab)
        size, radius = 181, 70
        yy, xx = np.mgrid[:size, :size]
        rng = np.random.default_rng(7)
        dose = np.exp(-(((xx - 92) / 42) ** 2 + ((yy - 87) / 34) ** 2))
        dose += rng.normal(0.0, 0.08, dose.shape)

        contours = analysis._extract_isobars(dose, 90, 90, radius)

        self.assertGreaterEqual(len(contours), 3)
        self.assertLessEqual(len(contours), 5)
        self.assertTrue(all(len(contour) >= 8 for contour in contours))

    def test_dose_plot_is_opened_in_application_owned_window(self):
        analysis = AnalysisTab.__new__(AnalysisTab)
        analysis._collect_film_data = lambda: {
            "F": [("C1", 1.0, 2.0, 0.1), ("C2", 2.0, 4.0, 0.1)]
        }
        analysis._force_origin_var = SimpleNamespace(get=lambda: True)
        analysis._separate_rc_var = SimpleNamespace(get=lambda: False)
        analysis._show_analysis_data_var = SimpleNamespace(get=lambda: False)
        analysis._recompute_analysis_results = lambda: None
        analysis._fit_line = fit_weighted_line
        analysis.frame = object()
        with patch("app.ui.plot_window.show_figure") as show:
            analysis._plot_dose_vs_value()
        show.assert_called_once()
        plt.close("all")

    def test_line_profile_is_opened_in_application_owned_window(self):
        panel = MeasurementPanel.__new__(MeasurementPanel)
        panel.has_valid_measurement = True
        panel.current_measurement_data = object()
        panel.frame = object()
        panel.image_processor = SimpleNamespace(
            calibration_applied=False,
            get_last_measurement_raw_data=lambda: np.array([[1., 2., 3.], [2., 3., 4.]]),
            get_last_measurement_coordinates=lambda: np.array([[0., 0.], [0., 1.]]),
        )
        with patch("app.ui.plot_window.show_figure") as show:
            panel._show_interactive_graph()
        show.assert_called_once()
        plt.close("all")

    def test_measurement_limit_tracks_largest_image_side(self):
        panel = MeasurementPanel.__new__(MeasurementPanel)
        panel.image_processor = SimpleNamespace(
            original_image=np.zeros((877, 2111, 3), dtype=np.uint8),
            current_image=None,
        )
        self.assertEqual(panel._maximum_measurement_size(), 2111)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from custom_plugins.analysis_tools import AnalysisTab, fit_weighted_line


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


if __name__ == "__main__":
    unittest.main()

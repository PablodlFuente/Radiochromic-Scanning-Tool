import unittest

import numpy as np

from custom_plugins.auto_measurements.core.ctr_manager import (
    subtract_control,
    summarize_controls,
)


class ControlSubtractionTests(unittest.TestCase):
    def test_single_control_subtracted_from_itself_is_exact_zero(self):
        value, uncertainty = subtract_control(
            0.2, 0.03, 0.2, 0.03,
            member_index=0, member_count=1, member_uncertainty=0.03,
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(uncertainty, 0.0)

    def test_independent_sample_and_control_add_variances(self):
        value, uncertainty = subtract_control(1.2, 0.1, 0.2, 0.05)
        self.assertAlmostEqual(value, 1.0)
        self.assertAlmostEqual(uncertainty, np.hypot(0.1, 0.05))

    def test_member_control_uses_covariance_with_control_mean(self):
        control_mean, control_uncertainty = summarize_controls([0.2, 0.4], [0.1, 0.1])
        _, member_uncertainty = subtract_control(
            0.2, 0.1, control_mean, control_uncertainty,
            member_index=0, member_count=2, member_uncertainty=0.1,
        )
        naive = np.hypot(0.1, control_uncertainty)
        self.assertLess(member_uncertainty, naive)

    def test_control_summary_does_not_double_count_observed_scatter(self):
        mean, uncertainty = summarize_controls([0.1, 0.3], [0.2, 0.2])
        self.assertAlmostEqual(mean, 0.2)
        propagated = np.sqrt(0.2**2 + 0.2**2) / 2
        observed = np.std([0.1, 0.3], ddof=1) / np.sqrt(2)
        self.assertAlmostEqual(uncertainty, max(propagated, observed))


if __name__ == "__main__":
    unittest.main()

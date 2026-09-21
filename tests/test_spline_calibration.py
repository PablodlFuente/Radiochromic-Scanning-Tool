import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core.spline_calibration import (
    invert_spline_response,
    load_spline_calibration,
    prepare_spline_knots,
    save_spline_calibration,
)


class SplineCalibrationTests(unittest.TestCase):
    def test_unsorted_doses_and_replicates_produce_strict_knots(self):
        doses, values = prepare_spline_knots([2, 0, 1, 1], [20, 40, 30, 32])
        np.testing.assert_array_equal(doses, [0, 1, 2])
        np.testing.assert_array_equal(values, [40, 31, 20])

    def test_nonmonotonic_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly monotonic"):
            prepare_spline_knots([0, 1, 2], [40, 20, 30])

    def test_inverse_is_valid_only_inside_calibrated_intensity_range(self):
        dose, valid = invert_spline_response([45, 40, 30, 20, 15], [0, 1, 2], [40, 30, 20])
        np.testing.assert_array_equal(valid, [False, True, True, True, False])
        np.testing.assert_allclose(dose[valid], [0, 1, 2])
        self.assertTrue(np.isnan(dose[[0, 4]]).all())

    def test_round_trip_storage_preserves_exact_knots(self):
        knots = {channel: (np.array([0., 1., 2.]), np.array([40., 30., 20.]) - index)
                 for index, channel in enumerate("RGB")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spline_calibration.npz"
            save_spline_calibration(path, knots, 16)
            loaded, bit_depth = load_spline_calibration(path)
        self.assertEqual(bit_depth, 16)
        for channel in "RGB":
            np.testing.assert_array_equal(loaded[channel][0], knots[channel][0])
            np.testing.assert_array_equal(loaded[channel][1], knots[channel][1])


if __name__ == "__main__":
    unittest.main()

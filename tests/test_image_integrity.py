import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core.image_processor import ImageProcessor
from app.utils.image_io import read_image_unchanged, storage_bit_depth, write_tiff_unchanged


class ImageIOIntegrityTests(unittest.TestCase):
    def test_unicode_uint16_tiff_round_trip_preserves_rgb_values(self):
        image = np.array(
            [
                [[1, 257, 65535], [4096, 8192, 16384]],
                [[32768, 12345, 54321], [0, 42, 999]],
            ],
            dtype=np.uint16,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "película_ñ.tif"
            write_tiff_unchanged(path, image)
            decoded_bgr = read_image_unchanged(path)

        decoded_rgb = decoded_bgr[:, :, ::-1]
        self.assertEqual(decoded_rgb.dtype, np.uint16)
        np.testing.assert_array_equal(decoded_rgb, image)

    def test_dark_uint16_image_is_identified_from_storage_not_observed_values(self):
        depth, maximum = storage_bit_depth(np.full((4, 4), 12, dtype=np.uint16))
        self.assertEqual(depth, 16)
        self.assertEqual(maximum, 65535.0)


class FlatFieldGeometryTests(unittest.TestCase):
    @staticmethod
    def _processor(allow_resize=False):
        processor = ImageProcessor.__new__(ImageProcessor)
        processor.flat_field = np.ones((2, 2, 3), dtype=np.float64)
        processor.config = {"allow_flat_field_resize": allow_resize}
        return processor

    def test_mismatched_flat_field_is_rejected_by_default(self):
        image = np.ones((3, 3, 3), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._processor()._apply_flattening_3ch(image)

    def test_explicit_resize_preserves_shape_and_dtype(self):
        image = np.full((3, 3, 3), 1234, dtype=np.uint16)
        corrected = self._processor(allow_resize=True)._apply_flattening_3ch(image)
        self.assertEqual(corrected.shape, image.shape)
        self.assertEqual(corrected.dtype, image.dtype)
        np.testing.assert_array_equal(corrected, image)


if __name__ == "__main__":
    unittest.main()

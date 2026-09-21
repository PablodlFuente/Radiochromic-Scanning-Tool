import unittest

import cv2
import numpy as np

from custom_plugins import auto_measurements
from custom_plugins.auto_measurements.core.detection import DetectionEngine
from custom_plugins.auto_measurements.models import DetectionParams
from custom_plugins.auto_measurements.ui.main_tab import AutoMeasurementsTab


class ShapeDetectionTests(unittest.TestCase):
    def test_detect_squares_rejects_round_objects(self):
        image = np.full((300, 500), 255, dtype=np.uint8)
        cv2.rectangle(image, (70, 70), (190, 190), 0, 5)
        cv2.circle(image, (350, 130), 60, 0, 5)
        params = DetectionParams(
            min_circle_radius=40, max_circle_radius=90, param1=40
        )

        squares = DetectionEngine().detect_squares(image, params)

        self.assertEqual(len(squares), 1)
        x, y, width, height = squares[0]
        self.assertAlmostEqual(x + width / 2, 130, delta=5)
        self.assertAlmostEqual(y + height / 2, 130, delta=5)

    def test_measurement_geometry_must_be_fully_inside_rc(self):
        tab = AutoMeasurementsTab.__new__(AutoMeasurementsTab)
        tab.image_processor = type(
            "Processor", (), {"current_image": np.zeros((100, 100, 3))}
        )()

        self.assertTrue(tab._geometry_contains(
            "rectangle", (10, 10, 60, 60), "circle", (40, 40, 20)
        ))
        self.assertFalse(tab._geometry_contains(
            "rectangle", (10, 10, 60, 60), "circle", (65, 40, 20)
        ))

    def test_rc_move_is_invalid_when_it_leaves_a_child_behind(self):
        previous_overlay = auto_measurements._OVERLAY
        self.addCleanup(setattr, auto_measurements, "_OVERLAY", previous_overlay)
        tab = AutoMeasurementsTab.__new__(AutoMeasurementsTab)
        tab.image_processor = type(
            "Processor", (), {"current_image": np.zeros((120, 120, 3))}
        )()
        tab.tree = type(
            "Tree", (), {
                "get_children": lambda _self, item="": ("area",) if item == "film" else (),
                "parent": lambda _self, item: "film" if item == "area" else "",
            }
        )()
        auto_measurements._OVERLAY = {
            "item_to_shape": {
                "film": ("film", (60, 60, 50, 50)),
                "area": ("circle", (30, 30, 10)),
            },
            "geometry_by_item": {},
        }

        self.assertFalse(tab._shape_position_is_valid("film"))


if __name__ == "__main__":
    unittest.main()
